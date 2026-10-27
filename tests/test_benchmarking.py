import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")
pytest.importorskip("gpytorch")

from AFL.double_agent.benchmarking import PhaseMapF1Score, latin_hypercube, stratified_test_set
from AFL.double_agent.Pipeline import Pipeline
from AFL.double_agent.PyTorchExtrapolator import DirichletGPExtrapolator

BOUNDS = {"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.0, 1.0)}


def ground_truth(X):
    """Three classes from the first two features."""
    return (X[:, 0] > 0.5).astype(int) + (X[:, 1] > 0.5).astype(int)


def make_dataset(n_samples=30, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, 3))
    grid = rng.random((7, 3))
    return xr.Dataset(
        {
            "composition": (("sample", "component"), X),
            "labels": (("sample",), ground_truth(X)),
            "composition_grid": (("grid", "component"), grid),
        },
        coords={"component": list(BOUNDS)},
    )


def make_pipeline(model_uri, **kw):
    with Pipeline(name="bench") as pipeline:
        DirichletGPExtrapolator(
            feature_input_variable="composition",
            predictor_input_variable="labels",
            output_prefix="phase",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
            component_dim="component",
            params={"method": "mll", "n_iterations": 3},
            model_uri=model_uri,
            name="GP",
        )
        PhaseMapF1Score(ground_truth=ground_truth, bounds=BOUNDS, model_variable="phase_model",
                        n_per_phase=20, n_pool=500, seed=1, **kw)
    return pipeline


def test_latin_hypercube_respects_bounds():
    X = latin_hypercube({"a": (0, 1), "b": (10, 20)}, 50, seed=0)
    assert X.shape == (50, 2)
    assert (X[:, 0] >= 0).all() and (X[:, 0] <= 1).all()
    assert (X[:, 1] >= 10).all() and (X[:, 1] <= 20).all()


def test_stratified_test_set_balances_phases():
    X, y = stratified_test_set(ground_truth, BOUNDS, n_per_phase=15, n_pool=600, seed=0)
    assert X.shape == (45, 3)
    assert dict(zip(*np.unique(y, return_counts=True))) == {0: 15, 1: 15, 2: 15}
    assert (ground_truth(X) == y).all()


def test_phasemap_f1_score_in_pipeline(tmp_path):
    pipeline = make_pipeline(str(tmp_path))
    ds = pipeline.calculate(make_dataset(), disable_progress_bar=True)
    op = pipeline.search("PhaseMapF1Score")

    assert "phase_model" in op.input_variable
    assert ds.benchmark_f1.ndim == 0 and 0.0 <= ds.benchmark_f1.item() <= 1.0
    assert ds.benchmark_f1.attrs["average"] == "weighted"
    assert 0.0 <= ds.benchmark_accuracy.item() <= 1.0
    assert ds.benchmark_test_points.dims == ("benchmark_test", "benchmark_dim")
    assert list(ds.benchmark_dim.values) == list(BOUNDS)
    assert ds.sizes["benchmark_test"] == 60
    assert list(ds.benchmark_phase.values) == [0, 1, 2]
    assert ds.benchmark_recall.shape == (3,)
    assert (ds.benchmark_test_labels.values == ground_truth(ds.benchmark_test_points.values)).all()
    assert op.score["f1"] == pytest.approx(ds.benchmark_f1.item())
    assert op.score["n_points"] == 60 and op.score["n_phases_true"] == 3
    assert set(op.score["recall_per_phase"]) == {0, 1, 2}


def test_phasemap_f1_score_predict_override(tmp_path):
    # predict(model, X) replaces model.predict; an oracle scores 1 and a constant predictor 1/3
    oracle = make_pipeline(str(tmp_path), predict=lambda model, X: ground_truth(X))
    ds = oracle.calculate(make_dataset(), disable_progress_bar=True)
    assert ds.benchmark_f1.item() == pytest.approx(1.0)
    assert ds.benchmark_accuracy.item() == pytest.approx(1.0)
    assert ds.benchmark_n_phases_predicted.item() == 3

    constant = make_pipeline(str(tmp_path), predict=lambda model, X: np.ones(len(X), dtype=int))
    ds = constant.calculate(make_dataset(), disable_progress_bar=True)
    assert ds.benchmark_accuracy.item() == pytest.approx(1 / 3)
    assert ds.benchmark_n_phases_predicted.item() == 1
    np.testing.assert_allclose(ds.benchmark_recall.values, [0.0, 1.0, 0.0])


def test_phasemap_f1_score_reuses_test_set(tmp_path):
    pipeline = make_pipeline(str(tmp_path))
    op = pipeline.search("PhaseMapF1Score")
    ds1 = pipeline.calculate(make_dataset(seed=0), disable_progress_bar=True)
    points = op.test_points
    ds2 = pipeline.calculate(make_dataset(seed=1), disable_progress_bar=True)
    assert op.test_points is points
    np.testing.assert_array_equal(ds1.benchmark_test_points.values, ds2.benchmark_test_points.values)


def test_phasemap_f1_score_per_phase(tmp_path):
    pipeline = make_pipeline(str(tmp_path), average=None, output_prefix="eval")
    ds = pipeline.calculate(make_dataset(), disable_progress_bar=True)
    assert ds.eval_f1.dims == ("eval_phase",)
    assert ds.eval_f1.shape == (3,)
    assert pipeline.search("PhaseMapF1Score").score["f1_average"] == "none"


def test_phasemap_f1_score_requires_model_ref(tmp_path):
    op = PhaseMapF1Score(ground_truth=ground_truth, bounds=BOUNDS, n_pool=100, n_per_phase=5)
    with pytest.raises(KeyError, match="phase_model"):
        op.calculate(make_dataset())


def test_phasemap_f1_score_json_round_trip(tmp_path):
    pipeline = make_pipeline(str(tmp_path))
    data = pipeline.search("PhaseMapF1Score").to_json()
    # a module-level ground truth is stored by import path, so no live objects are needed
    assert data["args"]["ground_truth"] == {"function": f"{__name__}.ground_truth"}
    op = PhaseMapF1Score.from_json(data)
    assert {k: list(v) for k, v in op.bounds.items()} == {k: list(v) for k, v in BOUNDS.items()}
    assert op.resolve_ground_truth() is ground_truth


def test_phasemap_f1_score_live_ground_truth_is_a_reference(tmp_path):
    truth = lambda X: ground_truth(X)  # noqa: E731 - lambdas have no JSON form
    with Pipeline(name="bench") as pipeline:
        PhaseMapF1Score(ground_truth=truth, bounds=BOUNDS, name="F1")
    assert pipeline.objects == {"F1.ground_truth": truth}
    data = pipeline.search("F1").to_json()
    assert data["args"]["ground_truth"]["ref"] == "F1.ground_truth"
    op = PhaseMapF1Score.from_json(data, objects=pipeline.objects)
    assert op.ground_truth is truth


@pytest.fixture(scope="module")
def sas_instrument(tmp_path_factory):
    pytest.importorskip("sasmodels")
    pytest.importorskip("alphashape")
    pytest.importorskip("trimesh")
    from AFL.double_agent.VirtualInstruments import ProteinGlycerolTemperaturePhaseMap

    return ProteinGlycerolTemperaturePhaseMap(afl_home=tmp_path_factory.mktemp("afl"))


def test_virtual_instrument_ground_truth(sas_instrument):
    from AFL.double_agent.benchmarking import virtual_instrument_labels

    op = PhaseMapF1Score(ground_truth=sas_instrument, n_per_phase=5, n_pool=300, seed=0)
    bounds = op.resolve_bounds()
    assert list(bounds) == sas_instrument.components
    assert bounds["protein"] == (0.5, 75.0)
    X, y = op.test_set()
    assert X.shape[1] == 3 and set(np.unique(y)) <= {0, 1, 2}
    # the vectorised labelling matches label_for row by row
    expected = np.array([int(sas_instrument.label_for(row)[0]) for row in X])
    np.testing.assert_array_equal(virtual_instrument_labels(sas_instrument, X).astype(int), expected)
