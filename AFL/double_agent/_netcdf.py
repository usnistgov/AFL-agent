"""NetCDF reads and writes that survive mismatched HDF5 builds of ``netCDF4`` and ``h5py``.

Some environments install ``netCDF4`` and ``h5py`` wheels that bundle different HDF5 builds.
Whichever is imported first loads its HDF5 library and the other then fails: xarray's default
``netcdf4`` engine raises ``NetCDF: HDF error`` on reads and ``Problem with HDF5 dimscales`` on
writes once ``h5py`` (pulled in by ``AFL.automation``) has been imported. ``AFL.double_agent``
therefore imports ``netCDF4`` before anything that imports ``h5py``, and the helpers here retry
with the ``h5netcdf`` engine, which uses the already loaded ``h5py``, if the default engine
still fails.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

FALLBACK_ENGINE = "h5netcdf"


def _fallback_available() -> bool:
    try:
        import h5netcdf  # noqa: F401
    except ImportError:
        return False
    return True


def load_netcdf(path: str | Path, lazy: bool = False, **kwargs) -> xr.Dataset:
    """``xr.open_dataset`` (``lazy=True``) or ``xr.load_dataset`` with the HDF5 fallback."""
    reader = xr.open_dataset if lazy else xr.load_dataset
    try:
        return reader(path, **kwargs)
    except (OSError, RuntimeError):
        if "engine" in kwargs or not _fallback_available():
            raise
        return reader(path, engine=FALLBACK_ENGINE, **kwargs)


def save_netcdf(dataset: xr.Dataset, path: str | Path, **kwargs) -> None:
    """``Dataset.to_netcdf`` with the HDF5 fallback."""
    try:
        dataset.to_netcdf(path, **kwargs)
    except (OSError, RuntimeError):
        if "engine" in kwargs or not _fallback_available():
            raise
        # the failed write leaves a partially created, still locked file behind
        Path(path).unlink(missing_ok=True)
        dataset.to_netcdf(path, engine=FALLBACK_ENGINE, **kwargs)
