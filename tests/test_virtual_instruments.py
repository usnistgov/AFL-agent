import subprocess
import sys
import textwrap

import numpy as np
import xarray as xr
import pytest

from AFL.double_agent.VirtualInstruments import GaussianVirtualInstrument


def test_gaussian_virtual_instrument_measures_batches(tmp_path):
    instrument = GaussianVirtualInstrument(afl_home=tmp_path)
    result = instrument.measure([[0.25, 0.05], [0.75, 0.10]])

    assert result["parameters"].dims == ("sample", "component")
    assert result["spectrum"].dims == ("sample", "wavelength")
    assert result["spectrum"].shape == (2, 201)
    np.testing.assert_array_equal(result["component"], ["mu", "sigma"])
    first_peak = result["spectrum"].isel(sample=0).argmax(dim="wavelength")
    second_peak = result["spectrum"].isel(sample=1).argmax(dim="wavelength")
    assert result["wavelength"].isel(wavelength=first_peak).item() == pytest.approx(0.25)
    assert result["wavelength"].isel(wavelength=second_peak).item() == pytest.approx(0.75)


@pytest.mark.parametrize(
    "parameters, message",
    [
        ([0.5], "shape"),
        ([[np.nan, 0.1]], "finite"),
        ([[0.5, 0.0]], "greater than zero"),
    ],
)
def test_gaussian_virtual_instrument_rejects_invalid_parameters(tmp_path, parameters, message):
    instrument = GaussianVirtualInstrument(afl_home=tmp_path)
    with pytest.raises(ValueError, match=message):
        instrument.measure(parameters)


def test_gaussian_virtual_instrument_works_without_afl_automation(tmp_path):
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import pathlib
        import sys

        class BlockAutomation(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "AFL.automation" or fullname.startswith("AFL.automation."):
                    raise ModuleNotFoundError(
                        f"No module named {{fullname!r}}", name=fullname
                    )
                return None

        for module_name in list(sys.modules):
            if module_name == "AFL.automation" or module_name.startswith("AFL.automation."):
                del sys.modules[module_name]
        sys.meta_path.insert(0, BlockAutomation())

        from AFL.double_agent.VirtualInstruments import GaussianVirtualInstrument

        instrument = GaussianVirtualInstrument(afl_home=pathlib.Path({str(tmp_path)!r}))
        result = instrument.measure([[0.5, 0.1]])
        assert result["spectrum"].shape == (1, 201)
        assert instrument.status() == ["Idle"]
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.fixture(scope="module")
def sas_instrument(tmp_path_factory):
    pytest.importorskip("alphashape")
    pytest.importorskip("trimesh")
    pytest.importorskip("sasmodels")
    from AFL.double_agent.VirtualInstruments import ProteinGlycerolTemperaturePhaseMap

    # AFL-automation's Driver persists config between runs, so pin the values we rely on.
    overrides = {k: ProteinGlycerolTemperaturePhaseMap.defaults[k] for k in
                 ["noise", "random_seed", "fast_locate", "locate_tolerance", "hull_tracing_ratio"]}
    return ProteinGlycerolTemperaturePhaseMap(overrides=overrides, afl_home=tmp_path_factory.mktemp("afl"))


def test_virtual_sas_instrument_setup(sas_instrument):
    from AFL.double_agent.VirtualInstruments import VirtualSASInstrument

    assert isinstance(sas_instrument, VirtualSASInstrument)
    assert sas_instrument.components == ["protein", "temperature", "glycerol"]
    assert set(sas_instrument.hulls) == {"0", "1", "2"}
    assert len(sas_instrument.reference_data) == 3
    assert set(sas_instrument.sasmodels) == {"0", "1", "2"}


def test_virtual_sas_instrument_measures_batches(sas_instrument):
    result = sas_instrument.measure(protein=20.0, glycerol=5.0, temperatures=[20.0, 5.0], batch_id=3)
    assert result["I"].dims == ("sample", "q")
    assert result["I"].shape[0] == 2
    assert result["I"].shape[1] == len(result["q"])
    assert np.all(np.diff(result["q"].values) >= 0)
    assert np.isfinite(result["I"].values).all()
    assert set(result["phases"].values.tolist()) <= {0, 1, 2}
    np.testing.assert_array_equal(result["ds_dim"], ["protein", "temperature", "glycerol"])
    np.testing.assert_array_equal(result["component"], ["protein", "glycerol"])
    np.testing.assert_array_equal(result["design_space"].sel(ds_dim="temperature"), [20.0, 5.0])
    np.testing.assert_array_equal(result["composition"].values, [[20.0, 5.0], [20.0, 5.0]])
    assert result["batch_sample_id"].values.tolist() == [3, 3]
    assert "sample" not in result.coords  # concatenable along "sample"
    # noiseless by default
    np.testing.assert_allclose(result["I"].values, result["I_noiseless"].values)


def test_virtual_sas_instrument_reproduces_boundary_labels(sas_instrument):
    """Points of the boundary dataset are mostly located in their own phase.

    Agreement is not perfect by design: ``fast_locate`` accepts the first hull within
    ``locate_tolerance`` (0.1 in min-max scaled space), so points near a boundary are
    attributed to whichever phase is checked first.
    """
    ds = sas_instrument.boundary_dataset
    comp = ds["composition"].transpose("sample", "component").values
    labels = ds["labels"].values
    rng = np.random.default_rng(0)
    agree = {label: [] for label in ["0", "1", "2"]}
    for label in agree:
        idx = np.flatnonzero(labels == label)
        for i in rng.choice(idx, size=15, replace=False):
            row = comp[i]
            # skip points outside the traced limits
            if not (0.5 <= row[0] <= 75 and 0.5 <= row[1] <= 45 and 0.5 <= row[2] <= 10):
                continue
            found, inside = sas_instrument.label_for(row)
            assert inside
            agree[label].append(found == label)
    assert all(agree["0"])
    assert np.mean(agree["1"]) > 0.5
    assert np.mean(sum(agree.values(), [])) > 0.5


def test_virtual_sas_instrument_simple_expose_interface(sas_instrument):
    ds = sas_instrument.simple_expose({"protein": 30.0, "temperature": 25.0, "glycerol": 4.0})
    assert ds["I"].dims == ("q",)
    assert ds.attrs["labels"] in {"0", "1", "2"}
    int(ds.attrs["labels"])  # must be castable to int
    np.testing.assert_array_equal(ds["component"], ["protein", "temperature", "glycerol"])


def test_virtual_sas_instrument_models_differ_by_phase(sas_instrument):
    curves = {label: sas_instrument.generate(label)[1] for label in ["0", "1", "2"]}
    assert not np.allclose(curves["0"], curves["1"])
    assert not np.allclose(curves["1"], curves["2"])


def test_virtual_sas_instrument_rejects_bad_shapes(sas_instrument):
    with pytest.raises(ValueError, match="finite"):
        sas_instrument.measure(protein=np.nan, glycerol=3.0, temperatures=[2.0])


def test_virtual_sas_instrument_noise_is_reproducible(tmp_path):
    pytest.importorskip("alphashape")
    pytest.importorskip("trimesh")
    from AFL.double_agent.VirtualInstruments import ProteinGlycerolTemperaturePhaseMap

    a = ProteinGlycerolTemperaturePhaseMap(overrides={"noise": 0.5, "random_seed": 3}, afl_home=tmp_path)
    b = ProteinGlycerolTemperaturePhaseMap(overrides={"noise": 0.5, "random_seed": 3}, afl_home=tmp_path)
    ra = a.measure(protein=30.0, glycerol=4.0, temperatures=25.0)
    rb = b.measure(protein=30.0, glycerol=4.0, temperatures=25.0)
    np.testing.assert_allclose(ra["I"].values, rb["I"].values)
    assert not np.allclose(ra["I"].values, ra["I_noiseless"].values)
    assert (ra["dI"].values > 0).all()


def test_virtual_sas_base_class_is_dataset_agnostic(tmp_path):
    """The base class starts empty and can be configured manually with any phase map."""
    pytest.importorskip("alphashape")
    pytest.importorskip("trimesh")
    pytest.importorskip("sasmodels")
    from AFL.double_agent.VirtualInstruments import ProteinGlycerolTemperaturePhaseMap, VirtualSASInstrument

    for key in ["boundary_dataset", "component_limits", "reference_configurations", "sasview_models"]:
        assert key not in VirtualSASInstrument.defaults
        assert key in ProteinGlycerolTemperaturePhaseMap.defaults

    # pin the dataset keys too: a persisted ~/.afl config from an older run may carry them
    overrides = {
        "noise": 0.0,
        "random_seed": 7,
        "boundary_dataset": None,
        "component_limits": {},
        "reference_configurations": [],
        "sasview_models": {},
    }
    base = VirtualSASInstrument(overrides=overrides, afl_home=tmp_path)
    assert base.boundary_dataset is None
    assert base.hulls == {}
    assert base.reference_data == []
    assert base.sasmodels == {}

    # a three-component toy phase map (hulls are 3D meshes): "a" for x < 0.5, "b" otherwise
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0.0, 1.0, size=(600, 3))
    labels = np.where(xyz[:, 0] < 0.5, "a", "b")
    ds = xr.Dataset(
        {"composition": (("sample", "component"), xyz), "labels": (("sample",), labels)},
        coords={"component": ["x", "y", "z"]},
        attrs={"components": "composition", "components_dim": "component", "labels": "labels"},
    )
    base.load_boundary_dataset(ds)
    base.trace_boundaries(hull_tracing_ratio=0.25)
    base.add_configuration_from_file("SANS/med_q.ABS")
    base.add_sasview_model("a", "power_law", {"scale": 1e-7, "background": 1.0, "power": 4.0})
    base.add_sasview_model("b", "sphere", {"scale": 0.5, "background": 1.0, "sld": 1.0, "sld_solvent": 6.0, "radius": 10})

    assert base.components == ["x", "y", "z"]
    assert set(base.hulls) == {"a", "b"}
    result = base.measure([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]])
    assert result["labels"].values.tolist() == ["a", "b"]
    assert result["I"].shape == (2, len(result["q"]))


def test_protein_glycerol_expose_returns_one_sample_without_sample_dim(sas_instrument):
    row = sas_instrument.expose(protein=30.0, glycerol=4.0, temperature=25.0)
    assert "sample" not in row.dims
    assert row["I"].dims == ("q",)
    assert row["design_space"].dims == ("ds_dim",)
    assert row["composition"].dims == ("component",)
    assert row["phases"].ndim == 0
    np.testing.assert_array_equal(
        row["design_space"].sel(ds_dim=["protein", "temperature", "glycerol"]).values, [30.0, 25.0, 4.0]
    )
    full = sas_instrument.measure(30.0, 4.0, [25.0])
    assert int(row["phases"]) == int(full["phases"].isel(sample=0))
