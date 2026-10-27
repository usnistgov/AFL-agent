import json

import numpy as np
import pytest

from AFL.double_agent.VirtualInstrumentDriver import (
    VirtualInstrumentDriver,
    _launch_instrument_class,
    available_instruments,
    instrument_spec,
    resolve_instrument_class,
)
from AFL.double_agent.VirtualInstruments import GaussianVirtualInstrument


def test_available_instruments_lists_virtual_instrument_classes():
    names = available_instruments()
    assert "GaussianVirtualInstrument" in names
    assert "ProteinGlycerolTemperaturePhaseMap" in names


def test_resolve_instrument_class_by_name_and_dotted_path():
    assert resolve_instrument_class("GaussianVirtualInstrument") is GaussianVirtualInstrument
    assert resolve_instrument_class("AFL.double_agent.VirtualInstruments.GaussianVirtualInstrument") is GaussianVirtualInstrument
    assert resolve_instrument_class(GaussianVirtualInstrument) is GaussianVirtualInstrument
    assert instrument_spec(GaussianVirtualInstrument) == "GaussianVirtualInstrument"


@pytest.mark.parametrize("spec", ["Nope", "AFL.double_agent.Pipeline.Pipeline", "no_such_module.Thing"])
def test_resolve_instrument_class_rejects_unknown_or_non_instruments(spec):
    with pytest.raises(ValueError):
        resolve_instrument_class(spec)


def test_driver_is_named_after_the_instrument_and_forwards_its_methods(tmp_path):
    driver = VirtualInstrumentDriver(afl_home=tmp_path, overrides={"instrument_overrides": {"noise": 0.0}})

    assert driver.name == "GaussianVirtualInstrument"
    assert isinstance(driver.instrument, GaussianVirtualInstrument)
    assert driver.config["instrument"] == "GaussianVirtualInstrument"
    assert driver.status() == ["Idle"]

    ds = driver.measure([[0.5, 0.1], [0.3, 0.2]])
    assert dict(ds.sizes) == {"sample": 2, "component": 2, "wavelength": 201}
    np.testing.assert_allclose(ds["parameters"].values, [[0.5, 0.1], [0.3, 0.2]])

    with pytest.raises(AttributeError):
        driver.not_a_method


def test_driver_and_instrument_keep_separate_config_files(tmp_path):
    VirtualInstrumentDriver(afl_home=tmp_path)
    assert (tmp_path / "GaussianVirtualInstrument.config.json").exists()
    assert (tmp_path / "virtual_instruments" / "GaussianVirtualInstrument.config.json").exists()


def test_instrument_overrides_reach_the_instrument(tmp_path):
    driver = VirtualInstrumentDriver(
        afl_home=tmp_path,
        overrides={"instrument_overrides": {"noise": 0.0, "n_wavelengths": 11}},
    )
    assert driver.get_instrument_config()["n_wavelengths"] == 11
    assert driver.measure([[0.5, 0.1]]).sizes["wavelength"] == 11


def test_set_instrument_config_updates_instrument_and_remembers_overrides(tmp_path):
    driver = VirtualInstrumentDriver(afl_home=tmp_path, overrides={"instrument_overrides": {"noise": 0.0}})
    config = driver.set_instrument_config(noise=0.25)
    assert config["noise"] == 0.25
    assert driver.instrument.config["noise"] == 0.25
    assert driver.config["instrument_overrides"] == {"noise": 0.25}


def test_server_attributes_are_shared_with_the_instrument(tmp_path):
    driver = VirtualInstrumentDriver(afl_home=tmp_path)
    assert driver.instrument.data is None
    sentinel = object()
    driver.data = sentinel
    driver.app = "app"
    assert driver.instrument.data is sentinel
    assert driver.instrument.app == "app"


def test_instrument_tasks_lists_instrument_methods_only(tmp_path):
    driver = VirtualInstrumentDriver(afl_home=tmp_path)
    tasks = driver.instrument_tasks()
    assert {"measure", "status"} <= set(tasks)
    assert "(parameters" in tasks["measure"]["signature"]
    # Driver-level tasks are served by the driver itself, not listed as instrument tasks.
    for name in ("set_config", "set_sample", "gather_defaults", "instrument_tasks"):
        assert name not in tasks


def test_instrument_info_describes_the_instrument(tmp_path):
    info = VirtualInstrumentDriver(afl_home=tmp_path).instrument_info()
    assert info["instrument"] == "GaussianVirtualInstrument"
    assert info["module"] == "AFL.double_agent.VirtualInstruments"
    assert "measure" in info["tasks"]
    assert info["config"]["n_wavelengths"] == 201


def test_driver_accepts_an_instrument_instance(tmp_path):
    instrument = GaussianVirtualInstrument(afl_home=tmp_path / "own", overrides={"amplitude": 2.0})
    driver = VirtualInstrumentDriver(afl_home=tmp_path, instrument=instrument)
    assert driver.instrument is instrument
    assert driver.name == "GaussianVirtualInstrument"
    assert float(driver.measure([[0.5, 0.1]])["spectrum"].max()) == pytest.approx(2.0)


def test_for_instrument_makes_a_subclass_defaulting_to_that_instrument(tmp_path):
    pytest.importorskip("alphashape")
    pytest.importorskip("trimesh")
    pytest.importorskip("sasmodels")
    cls = VirtualInstrumentDriver.for_instrument("ProteinGlycerolTemperaturePhaseMap")
    assert cls.__name__ == "ProteinGlycerolTemperaturePhaseMap"
    assert issubclass(cls, VirtualInstrumentDriver)
    assert cls.gather_defaults()["instrument"] == "ProteinGlycerolTemperaturePhaseMap"

    driver = cls(afl_home=tmp_path, overrides={"instrument_overrides": {"noise": 0.0, "random_seed": 3}})
    assert driver.name == "ProteinGlycerolTemperaturePhaseMap"
    assert driver.status()[0] == "Idle"
    exposure = driver.expose(protein=40.0, glycerol=5.0, temperature=20.0)
    assert "sample" not in exposure.dims
    assert {"phases", "composition", "design_space"} <= set(exposure.data_vars)
    assert driver.locate([40.0, 20.0, 5.0]) == ["1"]


def test_launch_instrument_class_reads_the_config_file(tmp_path):
    assert _launch_instrument_class([]) is GaussianVirtualInstrument

    config = tmp_path / "instrument.json"
    config.write_text(json.dumps({"instrument": "AFL.double_agent.VirtualInstruments.GaussianVirtualInstrument"}))
    assert _launch_instrument_class(["--config", str(config)]) is GaussianVirtualInstrument
    assert _launch_instrument_class(["--no-waitress", "--config", str(config), "-i"]) is GaussianVirtualInstrument

    config.write_text(json.dumps({"instrument": "Nope"}))
    with pytest.raises(ValueError):
        _launch_instrument_class(["--config", str(config)])

    config.write_text("[]")
    with pytest.raises(SystemExit):
        _launch_instrument_class(["--config", str(config)])
