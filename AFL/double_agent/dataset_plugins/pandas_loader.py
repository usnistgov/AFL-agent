import pandas as pd
import xarray as xr

extensions = ['.csv', '.tsv']

def load(path: str) -> xr.Dataset:
    """Load CSV/TSV files into an xarray.Dataset."""
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep='\t')
    return df.to_xarray()
