"""
Unit tests for the AFL.double_agent.data module.
"""

import os
import tempfile
from pathlib import Path

import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock

from AFL.double_agent.data import (
    get_data_dir,
    list_datasets,
    load_dataset,
    example_dataset1,
)


@pytest.mark.unit
class TestData:
    """Tests for the data module."""

    def test_get_data_dir(self):
        """Test that get_data_dir returns a valid path."""
        data_dir = get_data_dir()

        # Check that the path exists and is a directory
        assert data_dir.exists()
        assert data_dir.is_dir()

        # Check that it's the expected path
        assert "double_agent" in str(data_dir)
        assert "data" in str(data_dir)

    def test_list_datasets(self):
        """Test listing available datasets."""
        datasets = list_datasets()

        # Check that it returns a list
        assert isinstance(datasets, list)

        # The example dataset should be included
        assert "example_dataset" in datasets

    def test_load_dataset(self):
        """Test loading a dataset."""
        # Load the example dataset
        ds = load_dataset("example_dataset")

        # Check that it's an xarray Dataset
        assert isinstance(ds, xr.Dataset)

        # Check that it has expected content
        assert len(ds.dims) > 0
        assert len(ds.data_vars) > 0

    def test_load_nonexistent_dataset(self):
        """Test that loading a nonexistent dataset raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent_dataset")

    def test_example_dataset1(self):
        """Test the example_dataset1 helper function."""
        # Load the dataset using the helper
        ds = example_dataset1()

        # Check that it's an xarray Dataset
        assert isinstance(ds, xr.Dataset)

        # Check that it has expected content
        assert len(ds.dims) > 0
        assert len(ds.data_vars) > 0

    def test_dataset_list_warning(self):
        """Test warning when data directory doesn't exist."""
        with patch("AFL.double_agent.data.get_data_dir") as mock_get_data_dir:
            # Return a nonexistent path
            nonexistent_path = Path("/nonexistent/path")
            mock_get_data_dir.return_value = nonexistent_path

            # Check that list_datasets warns and returns empty list
            with pytest.warns(UserWarning):
                datasets = list_datasets()
                assert datasets == []
