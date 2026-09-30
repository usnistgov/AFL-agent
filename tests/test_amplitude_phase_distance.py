"""Tests for amplitude-phase target scoring."""

from importlib.util import find_spec

import numpy as np
import pytest
import xarray as xr

from AFL.double_agent.AmplitudePhaseDistance import AmplitudePhaseTargetScore
from AFL.double_agent.PipelineOp import PipelineOp

pytestmark = pytest.mark.skipif(
    find_spec("apdist") is None, reason="apdist is an optional dependency"
)


def test_amplitude_phase_target_score_defaults():
    operation = AmplitudePhaseTargetScore(target=[0.0, 1.0, 0.0, 0.0])

    assert operation.input_variable == "spline_spectrum"
    assert operation.output_variable == "score"
    assert operation.method == "discrete"
    assert operation.params == {"alpha": 0.5, "lam": 0.0, "grid_dim": 7}
    restored = PipelineOp.from_json(operation.to_json())
    assert isinstance(restored, AmplitudePhaseTargetScore)
    assert restored.params == operation.params


def test_amplitude_phase_target_score_calculate():
    domain = np.linspace(350.0, 850.0, 101)
    target = np.exp(-0.5 * ((domain - 600.0) / 40.0) ** 2)
    shifted = np.exp(-0.5 * ((domain - 650.0) / 40.0) ** 2)
    dataset = xr.Dataset(
        {
            "spline_spectrum": (
                ("sample", "spline_wavelength"),
                np.stack((target, shifted)),
            )
        },
        coords={"sample": [10, 20], "spline_wavelength": domain},
    )
    operation = AmplitudePhaseTargetScore(target=target.tolist())

    operation.calculate(dataset)
    result = operation.output["score"]

    assert result.dims == ("sample",)
    np.testing.assert_array_equal(result.coords["sample"], [10, 20])
    assert result.sel(sample=10).item() == pytest.approx(0.0)
    assert result.sel(sample=20).item() > 0.0


def test_amplitude_phase_target_score_validates_configuration():
    with pytest.raises(ValueError, match="between 0 and 1"):
        operation = AmplitudePhaseTargetScore(
            target=[0.0, 1.0, 0.0, 0.0], params={"alpha": 1.1}
        )
        operation.calculate(
            xr.Dataset(
                {
                    "spline_spectrum": (
                        ("sample", "spline_wavelength"),
                        [[0.0, 1.0, 0.0, 0.0]],
                    )
                },
                coords={"sample": [0], "spline_wavelength": np.linspace(0, 1, 4)},
            )
        )
