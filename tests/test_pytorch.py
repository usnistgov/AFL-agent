import pytest
import numpy as np
import torch
import xarray as xr

from AFL.double_agent.PyTorchExtrapolator import DirichletGPExtrapolator, GPModel


def make_toy_dataset(n_samples=5, n_features=2, n_grid=3):
    # Features: shape (samples, features)
    X = np.random.rand(n_samples, n_features)
    # Labels: all same for degenerate case
    y = np.zeros(n_samples, dtype=int)
    # Grid: points for prediction
    grid = np.random.rand(n_grid, n_features)

    return xr.Dataset(
        {
            "features": (("sample", "feature"), X),
            "labels": ("sample", y),
            "grid": (("grid", "feature"), grid),
        }
    )


def test_init_sets_attributes():
    extrap = DirichletGPExtrapolator(
        feature_input_variable="features",
        predictor_input_variable="labels",
        output_prefix="test_",
        grid_variable="grid",
        grid_dim="grid",
        sample_dim="sample",
        component_dim="feature",
        params={"learning_rate": 0.1, "n_iterations": 1},
    )
    assert extrap.output_prefix == "test_"
    assert extrap.component_dim == "feature"
    assert extrap.params["n_iterations"] == 1


def test_calculate_single_class_branch():
    dataset = make_toy_dataset()
    extrap = DirichletGPExtrapolator(
        "features", "labels", "out_", "grid", "grid", "sample", "feature"
    )
    result = extrap.calculate(dataset)

    # Check outputs exist
    assert f"out_mean" in result.output
    assert f"out_entropy" in result.output
    assert f"out_y_prob" in result.output

    # Shapes match grid
    n_grid = dataset["grid"].shape[0]
    assert result.output["out_mean"].shape[0] == n_grid
    assert result.output["out_y_prob"].shape[0] == n_grid


def test_predict_mll_with_dummy_model():
    # Dummy model: returns a torch distribution with 2 classes
    class DummyModel:
        def __call__(self, x):
            mean = torch.zeros(2, x.shape[0])
            cov = torch.eye(x.shape[0]).repeat(2, 1, 1)
            return torch.distributions.MultivariateNormal(mean, cov)

    extrap = DirichletGPExtrapolator(
        "features", "labels", "out_", "grid", "grid", "sample", "feature"
    )
    x = np.random.rand(4, 2)
    pred_labels, probs, entropy, grad = extrap._predict_mll(x, DummyModel())

    assert pred_labels.shape[0] == x.shape[0]
    assert probs.shape[0] == x.shape[0]
    assert entropy.shape[0] == x.shape[0]
    assert grad.shape == x.shape


def test_mll_runs_on_tiny_dataset():
    n_samples, n_features = 3, 2
    X = torch.rand(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples,))

    extrap = DirichletGPExtrapolator(
        "features", "labels", "out_", "grid", "grid", "sample", "feature",
        params={"learning_rate": 0.1, "n_iterations": 1}
    )
    model, likelihood = extrap.mll(X, y, n_iterations=1)

    assert isinstance(model, GPModel)
    assert model.training is True
