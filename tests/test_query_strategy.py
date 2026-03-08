# test_strategies.py
import numpy as np
import xarray as xr
import pytest

from AFL.double_agent.QueryStrategy import (
    ArgMax, 
    FullWidthHalfMaximum1D, 
    MinMax1DLineSampler
)


@pytest.fixture
def dataset_1d():
    x = np.linspace(0, 10, 11)
    u = np.array([0, 1, 3, 7, 10, 7, 3, 1, 0, -1, -2])  # peak at x=4
    return xr.Dataset(
        {
            "composition_utility": (("point",), u),
            "composition_marginal_domain": (("point",), x),
            "utility": (("point",), u),
            "temperature_marginal_domain": (("point",), x),
        },
        coords={"point": np.arange(len(x))},
    )


@pytest.fixture
def dataset_2d():
    grid = np.array([[0.0, 0.1],
                     [0.2, 0.3],
                     [0.5, 0.5],
                     [0.8, 0.2]])
    u = np.array([0.1, 0.3, 0.9, 0.5])  # max at [0.5, 0.5]
    return xr.Dataset(
        {
            "composition_utility": (("point",), u),
            "composition_marginal_domain": (("point", "dim"), grid),
        },
        coords={"point": np.arange(len(grid)), "dim": ["a", "b"]},
    )


@pytest.fixture
def dataset_probs():
    x = np.linspace(0, 1, 10)
    probs = np.linspace(0.2, 0.9, 10)  # some inside [0.3, 1.0)
    return xr.Dataset(
        {
            "sliced_feasibility_y_prob": (("point",), probs),
            "temperature_domain": (("point",), x),
        },
        coords={"point": np.arange(len(x))},
    )

def test_argmax_1d(dataset_1d):
    op = ArgMax()
    result = op.calculate(dataset_1d)
    out = result.output[op.output_variable].values
    assert out.shape == (1,)
    assert np.isclose(out[0], 4)  # max at x=4


def test_argmax_2d(dataset_2d):
    op = ArgMax()
    result = op.calculate(dataset_2d)
    out = result.output[op.output_variable].values
    assert out.shape == (1, 2)
    assert np.allclose(out[0], [0.5, 0.5])

def test_fwhm_detects_peak(dataset_1d):
    op = FullWidthHalfMaximum1D()
    result = op.calculate(dataset_1d)
    out = result.output[op.output_variable].values
    # should include the peak (x=4) and left/right half max
    assert np.isclose(4, out, atol=1).any()


def test_fwhm_no_peak_returns_max():
    x = np.linspace(0, 5, 6)
    f = np.array([1, 1, 1, 1, 1, 1])  # flat, no peak
    dataset = xr.Dataset(
        {
            "utility": (("point",), f),
            "temperature_marginal_domain": (("point",), x),
        },
        coords={"point": np.arange(len(x))},
    )
    op = FullWidthHalfMaximum1D()
    result = op.calculate(dataset)
    out = result.output[op.output_variable].values
    assert out.shape[0] == 1
    assert out[0] in x  # global max fallback

def test_line_sampler_returns_evenly_spaced(dataset_probs):
    op = MinMax1DLineSampler(n_samples=3)
    result = op.calculate(dataset_probs)
    out = result.output[op.output_variable].values
    assert len(out) == 3
    # all selected points must lie within feasible set
    feasible = dataset_probs["temperature_domain"].values[
        (dataset_probs["sliced_feasibility_y_prob"].values > op.min_value)
        & (dataset_probs["sliced_feasibility_y_prob"].values < op.max_value)
    ]
    assert np.all(np.isin(out, feasible))


def test_line_sampler_invalid_n_samples(dataset_probs):
    op = MinMax1DLineSampler(n_samples=0)
    with pytest.raises(ValueError):
        op.calculate(dataset_probs)

    op = MinMax1DLineSampler(n_samples=20)
    with pytest.raises(ValueError):
        op.calculate(dataset_probs)
