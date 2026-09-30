import numpy as np
import pytest
import xarray as xr
from botorch.utils.sampling import draw_sobol_samples

from AFL.double_agent.Generator import RandomBoundedDesign
from AFL.double_agent.util import optimization_bounds_to_tensor

BOUNDS = {
    "mu": {"min": 0.15, "max": 0.85},
    "sigma": {"min": 0.03, "max": 0.20},
}


def test_random_bounded_design_latin_hypercube_is_reproducible_and_bounded():
    first = RandomBoundedDesign(
        output_variable="parameters",
        bounds=BOUNDS,
        count=6,
        seed=7,
        method="latin_hypercube",
    ).calculate(xr.Dataset())
    second = RandomBoundedDesign(
        output_variable="parameters", bounds=BOUNDS, count=6, seed=7, method="lhs"
    ).calculate(xr.Dataset())

    first_parameters = first.output["parameters"]
    second_parameters = second.output["parameters"]
    assert first_parameters.dims == ("initial_sample", "component")
    np.testing.assert_array_equal(first_parameters["component"], ["mu", "sigma"])
    np.testing.assert_allclose(first_parameters, second_parameters)
    assert np.all((first_parameters.sel(component="mu") >= 0.15).values)
    assert np.all((first_parameters.sel(component="mu") <= 0.85).values)
    assert np.all((first_parameters.sel(component="sigma") >= 0.03).values)
    assert np.all((first_parameters.sel(component="sigma") <= 0.20).values)


def test_random_bounded_design_sobol_uses_botorch_sampling():
    operation = RandomBoundedDesign(
        output_variable="parameters",
        bounds=BOUNDS,
        count=6,
        seed=7,
        method="sobol",
    ).calculate(xr.Dataset())

    bounds = optimization_bounds_to_tensor(BOUNDS, component_names=["mu", "sigma"])
    expected = draw_sobol_samples(bounds=bounds, n=6, q=1, seed=7).squeeze(1)
    np.testing.assert_allclose(operation.output["parameters"], expected.cpu().numpy())
    assert operation.output["parameters"].dims == ("initial_sample", "component")


@pytest.mark.parametrize("alias", ["lhs", "lhc", "latin_hypercube"])
def test_random_bounded_design_accepts_latin_hypercube_aliases(alias):
    operation = RandomBoundedDesign(
        output_variable="parameters", bounds=BOUNDS, count=2, seed=7, method=alias
    )
    assert operation.method == "latin_hypercube"


@pytest.mark.parametrize(
    "bounds, count, method, message",
    [
        (BOUNDS, 0, "latin_hypercube", "positive"),
        ({}, 1, "latin_hypercube", "at least one"),
        ({"mu": {"min": 1.0}}, 1, "latin_hypercube", "min.*max"),
        ({"mu": {"min": 1.0, "max": 0.0}}, 1, "latin_hypercube", "exceeds"),
        (BOUNDS, 1, "random", "method must be"),
    ],
)
def test_random_bounded_design_rejects_invalid_configuration(bounds, count, method, message):
    with pytest.raises(ValueError, match=message):
        RandomBoundedDesign(
            output_variable="parameters",
            bounds=bounds,
            count=count,
            seed=7,
            method=method,
        )
