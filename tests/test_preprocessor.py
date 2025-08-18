"""
Unit tests for AFL.double_agent preprocessor operations.
"""

import pytest
import numpy as np
import xarray as xr
from AFL.double_agent.Preprocessor import SavgolFilter


def test_savgol_filter_basic(example_dataset2):
    """Test basic functionality of SavgolFilter preprocessor."""
    # Create filter with default parameters
    filter_op = SavgolFilter(
        input_variable="I",
        output_variable="filtered_data",
        dim="q",
        window_length=31,
        polyorder=2,
        derivative=0,
    )

    # Apply filter to dataset
    filter_op.calculate(example_dataset2)
    result = filter_op.output["filtered_data"]

    # Basic checks
    assert "filtered_data" in filter_op.output
    assert isinstance(result, xr.DataArray)
    assert not np.any(np.isnan(result))  # No NaN values should be present
    assert result.dims[-1] == "log_q"  # Should have renamed q dimension


def test_savgol_filter_derivative(example_dataset2):
    """Test SavgolFilter with derivative calculation."""
    # Create filter to compute first derivative
    filter_op = SavgolFilter(
        input_variable="I",
        output_variable="derivative",
        dim="q",
        window_length=31,
        polyorder=2,
        derivative=1,
    )

    # Apply filter
    filter_op.calculate(example_dataset2)
    result = filter_op.output["derivative"]

    # Check derivative properties
    assert "derivative" in filter_op.output
    assert isinstance(result, xr.DataArray)
    assert not np.any(np.isnan(result))

    # First derivative should have some positive and negative values
    assert np.any(result > 0)
    assert np.any(result < 0)


def test_savgol_filter_log_scale(example_dataset2):
    """Test SavgolFilter with log scaling enabled."""
    # Create filter with log scaling
    filter_op = SavgolFilter(
        input_variable="I",
        output_variable="log_filtered",
        dim="q",
        window_length=31,
        polyorder=2,
        derivative=0,
        apply_log_scale=True,
    )

    # Apply filter
    filter_op.calculate(example_dataset2)
    result = filter_op.output["log_filtered"]

    # Check log scaling properties
    assert "log_filtered" in filter_op.output
    assert isinstance(result, xr.DataArray)
    assert not np.any(np.isnan(result))
    assert result.dims[-1] == "log_q"  # Should have log_q dimension


def test_savgol_filter_data_trimming(example_dataset2):
    """Test SavgolFilter with data range trimming."""
    # Get original q range
    q_min = float(example_dataset2.q.min())
    q_max = float(example_dataset2.q.max())
    q_mid = (q_min + q_max) / 2

    # Create filter with trimmed range
    filter_op = SavgolFilter(
        input_variable="I",
        output_variable="trimmed_data",
        dim="q",
        window_length=31,
        polyorder=2,
        derivative=0,
        xlo=q_mid,  # Only use upper half of q range
        xhi=q_max,
    )

    # Apply filter
    filter_op.calculate(example_dataset2)
    result = filter_op.output["trimmed_data"]

    # Check trimming
    assert "trimmed_data" in filter_op.output
    assert isinstance(result, xr.DataArray)
    assert not np.any(np.isnan(result))
    assert float(result.log_q.min()) >= np.log10(q_mid)  # Data should start at or after q_mid


def test_savgol_filter_index_trimming(example_dataset2):
    """Test SavgolFilter trimming using integer indices."""
    # Trim using indices rather than coordinate values
    xlo_isel = 10
    xhi_isel = 50
    q_min_expected = float(example_dataset2.q.isel(q=xlo_isel))
    q_max_expected = float(example_dataset2.q.isel(q=xhi_isel - 1))

    filter_op = SavgolFilter(
        input_variable="I",
        output_variable="index_trimmed",
        dim="q",
        window_length=31,
        polyorder=2,
        derivative=0,
        xlo_isel=xlo_isel,
        xhi_isel=xhi_isel,
    )

    filter_op.calculate(example_dataset2)
    result = filter_op.output["index_trimmed"]

    assert "index_trimmed" in filter_op.output
    assert isinstance(result, xr.DataArray)
    assert not np.any(np.isnan(result))
    assert len(result.log_q) == filter_op.npts
    assert float(result.log_q.min()) == pytest.approx(np.log10(q_min_expected))
    assert float(result.log_q.max()) == pytest.approx(np.log10(q_max_expected))
