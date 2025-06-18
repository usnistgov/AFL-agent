"""
Global pytest configuration and fixtures for AFL.double_agent tests.
"""

import os
import sys
import pytest
import numpy as np
import xarray as xr
from pathlib import Path

# Add the parent directory to sys.path to allow importing AFL
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AFL.double_agent import Pipeline, PipelineOp
from AFL.double_agent.data import load_dataset

@pytest.fixture(scope="session")
def example_dataset1():
    """
    Load the example dataset from AFL.double_agent.data.
    """
    try:
        return load_dataset("example_dataset")
    except FileNotFoundError:
        pytest.skip("Example dataset not found, skipping test")


@pytest.fixture(scope="session")
def example_dataset2():
    """
    Load the example dataset from AFL.double_agent.data.
    """
    try:
        return load_dataset("synthetic_sans")
    except FileNotFoundError:
        pytest.skip("Example dataset not found, skipping test")


@pytest.fixture(scope="session")
def example_dataset1_cluster_result():
    """
    Load the example cluster result dataset from the tests/data directory.

    Returns:
        xr.Dataset: The loaded cluster result dataset
    """
    try:
        file_path = Path(__file__).parent / "data" / "example1_clustering_result.nc"
        return xr.open_dataset(file_path)
    except FileNotFoundError:
        pytest.skip("Example cluster result dataset not found, skipping test")


@pytest.fixture
def simple_pipeline():
    """
    Create a simple pipeline with no operations using the context manager.
    """
    with Pipeline(name="TestPipeline") as pipeline:
        pass
    return pipeline


@pytest.fixture
def tmp_netcdf_file(tmp_path):
    """
    Create a temporary file path for netCDF files.
    """
    return tmp_path / "test_data.nc"


# Custom pytest plugin for checking xarray equality with better error messages
def pytest_assertrepr_compare(op, left, right):
    if isinstance(left, xr.Dataset) and isinstance(right, xr.Dataset) and op == "==":
        result = []
        result.append("DataSet comparison failed")

        # Check variables
        left_vars = set(left.data_vars)
        right_vars = set(right.data_vars)

        if left_vars != right_vars:
            result.append(f"Different variables: {left_vars.symmetric_difference(right_vars)}")

        # Compare dimensions
        left_dims = {k: v.dims for k, v in left.data_vars.items()}
        right_dims = {k: v.dims for k, v in right.data_vars.items()}

        for var in left_vars.intersection(right_vars):
            if left_dims[var] != right_dims[var]:
                result.append(f"Different dimensions for variable {var}:")
                result.append(f"  Left:  {left_dims[var]}")
                result.append(f"  Right: {right_dims[var]}")

        return result

    return None


def assert_pipeline_result(
    result: xr.Dataset, expected_vars: list[str], shape_checks: dict[str, tuple] = None
) -> None:
    """
    Assert that a pipeline result contains expected variables with correct shapes.

    Parameters
    ----------
    result : xr.Dataset
        The pipeline result dataset
    expected_vars : list[str]
        List of variable names that should be in the result
    shape_checks : dict[str, tuple]
        Optional dictionary mapping variable names to expected shapes

    Raises
    ------
    AssertionError
        If any expectations are not met
    """
    # Check that all expected variables exist
    for var in expected_vars:
        assert var in result, f"Expected variable '{var}' not found in result"

    # Check shapes if provided
    if shape_checks:
        for var, expected_shape in shape_checks.items():
            if var in result:
                actual_shape = result[var].shape
                assert (
                    actual_shape == expected_shape
                ), f"Variable '{var}' has shape {actual_shape}, expected {expected_shape}"
            else:
                assert False, f"Cannot check shape of '{var}', variable not found"
