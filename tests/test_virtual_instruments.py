import subprocess
import sys
import textwrap

import numpy as np
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
