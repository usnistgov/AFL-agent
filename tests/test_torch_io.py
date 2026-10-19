import json

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")
pytest.importorskip("gpytorch")

from AFL.double_agent.Pipeline import Pipeline
from AFL.double_agent.PyTorchExtrapolator import DirichletGP, DirichletGPExtrapolator
from AFL.double_agent import torch_io
from AFL.double_agent.torch_io import save_model, load_model, is_model_ref


def make_dataset(n_samples=12, n_features=3, n_grid=7, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, n_features))
    y = (X[:, 0] > 0.5).astype(int) + (X[:, 1] > 0.5).astype(int)  # 3 classes
    grid = rng.random((n_grid, n_features))
    return xr.Dataset(
        {
            "composition": (("sample", "component"), X),
            "labels": (("sample",), y),
            "composition_grid": (("grid", "component"), grid),
        }
    )


def make_extrapolator(model_uri, method="mll"):
    return DirichletGPExtrapolator(
        feature_input_variable="composition",
        predictor_input_variable="labels",
        output_prefix="DirichletGP",
        grid_variable="composition_grid",
        grid_dim="grid",
        sample_dim="sample",
        component_dim="component",
        params={"method": method, "n_iterations": 3, "num_samples": 2, "num_warmup": 2},
        model_uri=model_uri,
        name="DirichletGP",
    )


def profile_score(q):
    """Toy score for the SRSF op tests: one value per restart."""
    return (q**2).mean(dim=(-2, -1))


@pytest.fixture(autouse=True)
def clear_cache():
    torch_io.clear_cache()
    yield
    torch_io.clear_cache()


def test_is_model_ref():
    assert is_model_ref("file:///tmp/x.pt")
    assert is_model_ref("tiled+http://host:8000/models/a")
    assert not is_model_ref("DirichletGP_model")
    assert not is_model_ref(3)


@pytest.mark.parametrize("method", ["mll", "mcmc"])
def test_dirichlet_gp_save_load_file_roundtrip(tmp_path, method):
    ds = make_dataset()
    op = make_extrapolator(str(tmp_path / "models"), method=method)
    op.calculate(ds)

    ref = str(op.output["DirichletGP_model"].values[()])
    assert ref.startswith("file://") and ref.endswith(".pt")
    assert load_model(ref) is op.model  # cached in-process

    torch_io.clear_cache()
    loaded = load_model(ref)
    assert loaded is not op.model
    assert isinstance(loaded, DirichletGP)
    assert loaded.is_mcmc == (method == "mcmc")
    assert loaded.num_classes == 3

    for (k1, v1), (k2, v2) in zip(op.model.state_dict().items(), loaded.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(v1, v2)

    grid = ds["composition_grid"].values
    torch.manual_seed(0)
    _, p_orig, _, _ = op.model.predict(grid)
    torch.manual_seed(0)
    labels, p_loaded, entropy, grad = loaded.predict(grid, return_gradient=True)
    assert torch.allclose(p_orig, p_loaded, atol=1e-6)
    assert labels.shape == (grid.shape[0],)
    assert entropy.shape == (grid.shape[0],)
    assert grad.shape == grid.shape


def test_extrapolator_output_matches_model_predict(tmp_path):
    ds = make_dataset()
    op = make_extrapolator(str(tmp_path))
    torch.manual_seed(1)
    op.calculate(ds)
    assert op.output["DirichletGP_y_prob"].shape == (7, 3)
    assert np.allclose(op.output["DirichletGP_y_prob"].values.sum(axis=1), 1.0)
    assert op.output["DirichletGP_entropy_gradient"].shape == (7, 3)


def test_no_model_uri_means_no_model_variable(tmp_path):
    ds = make_dataset()
    op = make_extrapolator(None)
    op.calculate(ds)
    assert "DirichletGP_model" not in op.output
    assert isinstance(op.model, DirichletGP)
    json.dumps(op.to_json())


def test_each_save_gives_fresh_ref(tmp_path):
    model = DirichletGP(np.random.rand(5, 2), np.array([0, 1, 0, 1, 1]))
    args = {"train_x": model.train_x, "train_y": model.train_y}
    r1 = save_model(model, str(tmp_path), args=args)
    r2 = save_model(model, str(tmp_path), args=args)
    assert r1 != r2


def test_save_model_rejects_unknown_scheme(tmp_path):
    model = DirichletGP(np.random.rand(5, 2), np.array([0, 1, 0, 1, 1]))
    with pytest.raises(ValueError):
        save_model(model, "s3://bucket/models", args={})


def test_pipeline_json_roundtrip_and_model_handoff(tmp_path):
    """The downstream op gets the model the upstream op trained in the same run."""
    from AFL.double_agent.PipelineOp import PipelineOp

    class Consumer(PipelineOp):
        def __init__(self, model_variable, output_variable, name="Consumer"):
            super().__init__(name=name, input_variable=[model_variable], output_variable=output_variable)
            self.model_variable = model_variable

        def calculate(self, dataset):
            model = load_model(str(dataset[self.model_variable].values[()]))
            _, _, entropy, _ = model.predict(dataset["composition_grid"].values)
            self.output[self.output_variable] = xr.DataArray(entropy.detach().numpy(), dims=["grid"])
            return self

    with Pipeline(name="p") as p:
        make_extrapolator(str(tmp_path / "m"))
        Consumer("DirichletGP_model", "consumer_entropy")

    f = tmp_path / "p.json"
    p.write_json(str(f))
    ops = json.load(open(f))["ops"]
    assert ops[0]["args"]["model_uri"] == str(tmp_path / "m")

    out = p.calculate(make_dataset(), disable_progress_bar=True)
    assert "DirichletGP_model" in out
    assert out["consumer_entropy"].shape == (7,)
    ref = str(out["DirichletGP_model"].values[()])
    assert load_model(ref) is p[0].model

    # Missing upstream variable fails loudly rather than loading a stale model.
    with pytest.raises(KeyError):
        p[1].calculate(make_dataset())


def test_srsf_acquisition_op_json_roundtrip_and_model_load(tmp_path):
    pytest.importorskip("geoopt")
    from AFL.double_agent.representations.functions import (
        SRSFHilbertSphere,
        SRSFHilbertSphereAcquisitionFunction,
    )
    from AFL.double_agent.spec import resolve_spec

    with Pipeline(name="p") as p:
        make_extrapolator(str(tmp_path / "m"))
        SRSFHilbertSphereAcquisitionFunction(
            model_variable="DirichletGP_model",
            manifold={"class": SRSFHilbertSphere, "args": {"n_sampling_points": 16}},
            acquisition=profile_score,
            composition_variable=None,
            bounds_t=[0.0, 1.0],
        )
    f = tmp_path / "p.json"
    p.write_json(str(f))
    p2 = Pipeline.read_json(str(f))
    acq = p2[1]
    assert acq.manifold["class"] == "AFL.double_agent.representations.functions.SRSFHilbertSphere"
    assert acq.acquisition["function"].endswith("profile_score")
    manifold = resolve_spec(acq.manifold)
    assert isinstance(manifold, SRSFHilbertSphere) and manifold.n_sampling_points == 16

    ds = make_dataset()
    p2[0].calculate(ds)
    ds = p2[0].add_to_dataset(ds)
    model = acq.load_model(ds)
    assert isinstance(model, DirichletGP)

    # a raw ref works in place of a variable name
    ref = str(ds["DirichletGP_model"].values[()])
    frozen = SRSFHilbertSphereAcquisitionFunction(
        model_variable=ref,
        manifold=SRSFHilbertSphere,
        acquisition=profile_score,
        bounds_t=[0, 1],
    )
    assert frozen.load_model(xr.Dataset()) is model
    with pytest.raises(KeyError):
        acq.load_model(xr.Dataset())


def test_botorch_regressor_save_load_roundtrip(tmp_path):
    pytest.importorskip("botorch")
    from AFL.double_agent.PyTorchExtrapolator import BoTorchRegressor

    rng = np.random.default_rng(0)
    X = rng.random((10, 2))
    ds = xr.Dataset(
        {
            "composition": (("sample", "component"), X),
            "target": (("sample",), X.sum(axis=1)),
            "composition_grid": (("grid", "component"), rng.random((6, 2))),
        }
    )
    op = BoTorchRegressor(
        feature_input_variable="composition",
        predictor_input_variable="target",
        output_prefix="GP",
        grid_variable="composition_grid",
        grid_dim="grid",
        sample_dim="sample",
        model_uri=str(tmp_path),
        name="GP",
    )
    op.calculate(ds)
    ref = str(op.output["GP_model"].values[()])
    torch_io.clear_cache()
    loaded = load_model(ref)
    assert type(loaded) is type(op.model)
    candidate = torch.as_tensor(ds["composition_grid"].values)
    with torch.no_grad():
        m1 = op.model.posterior(candidate).mean
        m2 = loaded.posterior(candidate).mean
    assert torch.allclose(m1, m2, atol=1e-6)


def test_botorch_regressor_then_posterior_acquisition_reuses_saved_model(tmp_path):
    pytest.importorskip("botorch")
    from AFL.double_agent.PyTorchExtrapolator import BoTorchRegressor
    from AFL.double_agent.AcquisitionFunction import BoTorchPosteriorAcquisition

    rng = np.random.default_rng(0)
    X = rng.random((10, 2))
    ds = xr.Dataset(
        {
            "composition": (("sample", "component"), X),
            "target": (("sample",), ((X - 0.5) ** 2).sum(axis=1)),
            "composition_grid": (("grid", "component"), rng.random((6, 2))),
        },
        coords={"component": ["a", "b"]},
    )
    bounds = {"a": {"min": 0.0, "max": 1.0}, "b": {"min": 0.0, "max": 1.0}}
    with Pipeline(name="p") as p:
        BoTorchRegressor(
            feature_input_variable="composition",
            predictor_input_variable="target",
            output_prefix="GP",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
            bounds=bounds,
            model_uri=str(tmp_path),
            name="GP",
        )
        BoTorchPosteriorAcquisition(
            feature_input_variable="composition",
            predictor_input_variable="target",
            grid_variable="composition_grid",
            bounds=bounds,
            model_variable="GP_model",
            output_prefix="GP",
            num_restarts=2,
            raw_samples=8,
        )
    p.write_json(str(tmp_path / "p.json"))
    p2 = Pipeline.read_json(str(tmp_path / "p.json"))
    out = p2.calculate(ds, disable_progress_bar=True)
    assert p2[1].model is p2[0].model
    assert out["GP_best_x"].dims == ("component",)
    assert np.all((out["GP_best_x"].values >= 0) & (out["GP_best_x"].values <= 1))
    assert np.isfinite(out["GP_best_f"].item())
