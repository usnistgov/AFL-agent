import numpy as np, xarray as xr
from AFL.double_agent.AcquisitionFunction import MaxValueAF

def _ds():
    grid = np.linspace(0, 1, 11).reshape(-1, 1)
    return xr.Dataset({"grid": (("g", "d"), grid), "u": (("g",), grid[:, 0] ** 2)})

def test_default_sample_dim_is_af_sample():
    out = MaxValueAF(["u"], "grid", grid_dim="g").calculate(_ds()).output["next_samples"]
    assert out.dims == ("AF_sample", "d")

def test_custom_sample_dim_and_count():
    out = MaxValueAF(["u"], "grid", grid_dim="g", count=3, sample_dim="batch").calculate(_ds()).output["next_samples"]
    assert out.dims == ("batch", "d") and out.sizes["batch"] == 3
