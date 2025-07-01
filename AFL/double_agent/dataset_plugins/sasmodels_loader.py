import xarray as xr
import pandas as pd

extensions = ['.ses', '.xml', '.dat']

def load(path: str) -> xr.Dataset:
    """Load SAS data files using xarray as a placeholder."""
    # Placeholder: attempt to open as text data
    try:
        data = xr.Dataset.from_dataframe(pd.read_csv(path, comment='#', delim_whitespace=True))
    except Exception:
        data = xr.Dataset()
    return data
