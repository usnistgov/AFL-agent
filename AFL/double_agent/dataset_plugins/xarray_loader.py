import xarray as xr

extensions = ['.nc', '.netcdf']

def load(path: str) -> xr.Dataset:
    """Load NetCDF files using xarray."""
    return xr.load_dataset(path)
