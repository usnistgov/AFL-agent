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


import subprocess
import sys

import xarray as xr

from AFL.double_agent import _netcdf


def test_load_netcdf_retries_with_h5netcdf(monkeypatch, tmp_path):
    """The default engine failing with an HDF error triggers one retry with h5netcdf."""
    engines = []
    sentinel = xr.Dataset({"a": ("x", [1.0, 2.0])})

    def flaky(p, **kw):
        engines.append(kw.get("engine"))
        if "engine" not in kw:
            raise OSError("NetCDF: HDF error")
        return sentinel

    monkeypatch.setattr(xr, "load_dataset", flaky)
    monkeypatch.setattr(_netcdf, "_fallback_available", lambda: True)
    assert _netcdf.load_netcdf(tmp_path / "t.nc") is sentinel
    assert engines == [None, "h5netcdf"]


def test_load_netcdf_does_not_mask_errors_with_explicit_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(xr, "load_dataset", lambda p, **kw: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError, match="boom"):
        _netcdf.load_netcdf(tmp_path / "missing.nc", engine="netcdf4")


def test_save_netcdf_retries_with_h5netcdf(monkeypatch, tmp_path):
    calls = []

    def flaky(self, p, **kw):
        calls.append(kw.get("engine"))
        if "engine" not in kw:
            raise RuntimeError("NetCDF: Problem with HDF5 dimscales.")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", flaky)
    monkeypatch.setattr(_netcdf, "_fallback_available", lambda: True)
    _netcdf.save_netcdf(xr.Dataset({"a": ("x", [3.0])}), tmp_path / "t.nc")
    assert calls == [None, "h5netcdf"]


def test_netcdf_roundtrip_after_h5py_import(tmp_path):
    """Writing and reading through the helpers works even when h5py was imported first."""
    pytest.importorskip("h5py")
    pytest.importorskip("h5netcdf")
    code = (
        "import h5py\n"
        "import xarray as xr\n"
        "from AFL.double_agent._netcdf import load_netcdf, save_netcdf\n"
        f"path = {str(tmp_path / 't.nc')!r}\n"
        "save_netcdf(xr.Dataset({'a': ('x', [1.0, 2.0])}), path)\n"
        "assert load_netcdf(path)['a'].values.tolist() == [1.0, 2.0]\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_packaged_datasets_load_after_h5py_import():
    """Importing h5py first must not break the packaged NetCDF files (mismatched HDF5 builds)."""
    pytest.importorskip("h5py")
    code = (
        "import h5py\n"
        "from AFL.double_agent.data import load_dataset\n"
        "print(load_dataset('example_dataset').sizes)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
