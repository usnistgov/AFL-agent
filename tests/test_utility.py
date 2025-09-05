# test_marginal_entropy.py
import numpy as np
import xarray as xr
import pytest

from AFL.double_agent.Utility import (
    MarginalEntropyOverDimension,
    MarginalEntropyAlongDimension,
)

@pytest.fixture
def sample_dataset():
    # Define coordinate system
    temps = np.array([300, 310])       # temperature dimension
    comps = np.array([[0.2, 0.8],      # fake "composition"
                      [0.5, 0.5],
                      [0.8, 0.2]])
    
    ds_dim = ["temperature", "comp1", "comp2"]
    n_points = len(temps) * len(comps)

    # Build a design grid with stacked (temperature, comp1, comp2)
    design_space_grid = np.zeros((n_points, len(ds_dim)))
    idx = 0
    for t in temps:
        for c in comps:
            design_space_grid[idx, :] = [t, c[0], c[1]]
            idx += 1

    # Fake probability distribution (normalize over classes)
    n_classes = 3
    probs = np.random.rand(n_points, n_classes)
    probs /= probs.sum(axis=1, keepdims=True)

    dataset = xr.Dataset(
        {
            "design_space_grid": (("point", "ds_dim"), design_space_grid),
            "probability": (("point", "class"), probs),
            # Must match ds_dim length = 3
            "next_composition": (("comp1_comp2_d",), np.array([0.5, 0.5])),
        },
        coords={
            "ds_dim": ds_dim,
            "point": np.arange(n_points),
            "class": np.arange(n_classes),
        },
    )
    return dataset



def test_marginal_entropy_over_dimension(sample_dataset):
    op = MarginalEntropyOverDimension(
        input_variable="probability",
        coordinate_dims=["temperature"],
        component_dim="ds_dim",
        grid_variable="design_space_grid",
        output_variable="composition_utility",
    )
    result = op.calculate(sample_dataset)

    assert "composition_utility" in result.output
    utility = result.output["composition_utility"].values
    assert utility.ndim == 1
    assert np.all(utility >= 0)  # entropy must be non-negative


def test_marginal_entropy_along_dimension(sample_dataset):
    op = MarginalEntropyAlongDimension(
        input_variable="probability",
        conditioning_point="next_composition",
        coordinate_dim="temperature",
        grid_variable="design_space_grid",
        component_dim="ds_dim",
        output_variable="temperature",
    )
    result = op.calculate(sample_dataset)

    assert "temperature" in result.output
    entropy_values = result.output["temperature"].values
    assert entropy_values.ndim == 1
    assert np.all(entropy_values >= 0)  # entropy must be non-negative
