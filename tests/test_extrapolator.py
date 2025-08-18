"""
Unit tests for the AFL.double_agent.Extrapolator module.
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch

from AFL.double_agent.Extrapolator import (
    GaussianProcessRegressor,
    GaussianProcessClassifier,
    Extrapolator,
    DummyExtrapolator,
)


@pytest.mark.unit
class TestExtrapolator:
    """Tests for the Extrapolator base class."""

    def test_extrapolator_creation(self):
        """Test creating an Extrapolator."""
        extrap = Extrapolator(
            feature_input_variable="features",
            predictor_input_variable="labels",
            output_variables=["mean", "variance"],
            output_prefix="extrap",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
        )

        assert extrap.feature_input_variable == "features"
        assert extrap.predictor_input_variable == "labels"
        assert extrap.output_prefix == "extrap"


@pytest.mark.unit
class TestDummyExtrapolator:
    """Tests for the DummyExtrapolator class."""

    def test_dummy_extrapolator_creation(self):
        """Test creating a DummyExtrapolator."""
        extrap = DummyExtrapolator(
            feature_input_variable="features",
            predictor_input_variable="labels",
            output_prefix="extrap",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
        )

        assert extrap.feature_input_variable == "features"
        assert extrap.predictor_input_variable == "labels"
        assert extrap.output_prefix == "extrap"

    def test_dummy_extrapolator_calculation(self):
        """Test DummyExtrapolator calculations."""
        # Create test dataset
        ds = xr.Dataset(
            {
                "features": xr.DataArray(np.random.rand(5, 3), dims=["sample", "feature"]),
                "labels": xr.DataArray(np.random.rand(5), dims=["sample"]),
            }
        )

        # Create extrapolator
        extrap = DummyExtrapolator(
            feature_input_variable="features",
            predictor_input_variable="labels",
            output_prefix="extrap",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
        )

        # Mock the actual calculation method to avoid any external dependencies
        with patch.object(DummyExtrapolator, "calculate", return_value=extrap) as mock_calculate:
            # Set up the output variables that would normally be created by calculate
            extrap.output = {
                "extrap_mean": xr.DataArray(np.random.rand(5), dims=["sample"]),
                "extrap_variance": xr.DataArray(np.random.rand(5), dims=["sample"]),
            }

            # Call add_to_dataset which we're really testing here
            result = extrap.add_to_dataset(ds)

            # Check that the variables were added
            assert "extrap_mean" in result
            assert "extrap_variance" in result
