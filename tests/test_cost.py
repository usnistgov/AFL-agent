"""
Unit tests for the AFL.double_agent.Cost module.
"""

import pytest
import numpy as np
import xarray as xr

from AFL.double_agent.Cost import (
    DesignSpaceHierarchyCost,
    BinaryProbabilityCost,
    MarginalCost,
    SlicedCost,
    UtilityWithCost,
)


@pytest.fixture
def sample_dataset():
    """Fixture for creating a simple dataset with grid + cost/probability values."""
    grid_points = 5
    comps = ["temperature", "pressure"]

    # design grid
    grid = xr.DataArray(
        np.random.rand(grid_points, len(comps)),
        dims=("grid", "ds_dim"),
        coords={"ds_dim": comps},
    )

    # query variable (for DesignSpaceHierarchyCost)
    query = xr.DataArray(
        np.random.rand(grid_points, len(comps)),
        dims=("grid", "ds_dim"),
        coords={"ds_dim": comps},
    )

    # probability values for BinaryProbabilityCost
    prob = xr.DataArray(
        np.random.dirichlet(np.ones(2), size=grid_points),
        dims=("grid", "class"),
    )

    # cost values
    cost = xr.DataArray(
        np.random.rand(grid_points),
        dims="grid"
    )

    # conditioning point for SlicedCost
    next_composition = xr.DataArray(
        grid.values[0],
        dims=("ds_dim",),
        coords={"ds_dim": comps},
    )

    ds = xr.Dataset(
        {
            "grid": grid,
            "query": query,
            "probability": prob,
            "cost": cost,
            "next_composition": next_composition,
        }
    )
    return ds


def test_design_space_hierarchy_cost(sample_dataset):
    coords_order = ["temperature", "pressure"]
    op = DesignSpaceHierarchyCost(
        input_variable="query",
        grid_variable="grid",
        grid_dim="grid",
        component_dim="ds_dim",
        coordinates_order=coords_order,
        coordinates_offsets=[0.5, 0.5],
        output_variable="cost_hierarchy",
    )
    result = op.calculate(sample_dataset)

    assert "cost_hierarchy" in result.output
    arr = result.output["cost_hierarchy"]
    assert arr.dims == ("grid",)
    assert "description" in arr.attrs
    assert arr.shape[0] == sample_dataset.dims["grid"]


def test_binary_probability_cost(sample_dataset):
    op = BinaryProbabilityCost(
        input_variable="probability",
        grid_variable="grid",
        cost_label=1,
        output_variable="binary_cost",
    )
    result = op.calculate(sample_dataset)

    assert "binary_cost" in result.output
    arr = result.output["binary_cost"]
    assert arr.dims == ("grid",)
    assert np.all((arr.values >= 0) & (arr.values <= 1))


def test_marginal_cost(sample_dataset):
    op = MarginalCost(
        input_variable="cost",
        coordinate_dims=["temperature"],
        component_dim="ds_dim",
        grid_variable="grid",
        output_variable="marginal_cost",
    )
    result = op.calculate(sample_dataset)

    assert "marginal_cost" in result.output
    arr = result.output["marginal_cost"]
    assert "description" in arr.attrs
    assert arr.ndim == 1
    assert any("domain" in k for k in result.output.keys())


def test_sliced_cost(sample_dataset):
    op = SlicedCost(
        input_variable="cost",
        conditioning_point="next_composition",
        coordinate_dim="temperature",
        grid_variable="grid",
        component_dim="ds_dim",
        output_variable="slice_cost",
    )
    result = op.calculate(sample_dataset)

    assert "slice_cost" in result.output
    arr = result.output["slice_cost"]
    assert arr.ndim == 1
    assert "description" in arr.attrs
    assert any("domain" in k for k in result.output.keys())


def test_utility_with_cost(sample_dataset):
    acq = xr.DataArray(
        np.random.rand(sample_dataset.dims["grid"]),
        dims=("grid",),
    )
    sample_dataset["acq"] = acq
    sample_dataset["c1"] = xr.DataArray(
        np.random.rand(sample_dataset.dims["grid"]),
        dims=("grid",),
    )
    sample_dataset["c2"] = xr.DataArray(
        np.random.rand(sample_dataset.dims["grid"]),
        dims=("grid",),
    )

    op = UtilityWithCost(
        input_variable="acq",
        cost_variables=["c1", "c2"],
        output_variable="acq_with_cost",
    )
    result = op.calculate(sample_dataset)

    assert "acq_with_cost" in result.output
    arr = result.output["acq_with_cost"]
    assert arr.dims == ("grid",)
    assert "description" in arr.attrs
    assert np.all(arr.values <= sample_dataset["acq"].values)
