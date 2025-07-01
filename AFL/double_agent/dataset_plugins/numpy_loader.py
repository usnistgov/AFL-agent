import numpy as np
import xarray as xr

extensions = ['.npy', '.npz']

def load(path: str) -> xr.Dataset:
    """Load NumPy array files into an xarray.Dataset."""
    if path.endswith('.npy'):
        arr = np.load(path)
        if arr.ndim == 1:
            data = xr.DataArray(arr, dims=['dim_0'])
        else:
            dims = [f'dim_{i}' for i in range(arr.ndim)]
            data = xr.DataArray(arr, dims=dims)
        return data.to_dataset(name='array')
    elif path.endswith('.npz'):
        npz = np.load(path)
        data_vars = {k: xr.DataArray(v, dims=[f'dim_{i}' for i in range(v.ndim)]) for k, v in npz.items()}
        return xr.Dataset(data_vars)
    else:
        raise ValueError('Unsupported file extension for numpy loader')
