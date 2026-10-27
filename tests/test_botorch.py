"""
Unit tests for BoTorch-based extrapolation and acquisition helpers.
"""

import warnings
from unittest.mock import patch

import json
import numpy as np
import pytest
import torch
import xarray as xr

from AFL.double_agent.variables import Variable
from AFL.double_agent.AcquisitionFunction import BoTorchAcquisition, BoTorchPosteriorAcquisition
from AFL.double_agent.PyTorchExtrapolator import BoTorchRegressor
from AFL.double_agent.BoTorchMixin import BoTorchMixin


def _make_explicit_bounds():
    return {
        "red": {"min": 0.0, "max": 1.0},
        "green": {"min": 0.0, "max": 1.0},
        "blue": {"min": 0.0, "max": 1.0},
    }


def _expected_bounds_array():
    return np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def _make_botorch_dataset() -> xr.Dataset:
    component = ["red", "green", "blue"]
    composition = xr.DataArray(
        np.array(
            [
                [0.70, 0.20, 0.10],
                [0.20, 0.60, 0.20],
                [0.10, 0.30, 0.60],
            ]
        ),
        dims=["sample", "component"],
        coords={"sample": [0, 1, 2], "component": component},
    )
    score = xr.DataArray(np.array([0.15, 0.35, 0.55]), dims=["sample"], coords={"sample": [0, 1, 2]})
    composition_grid = xr.DataArray(
        np.array(
            [
                [0.60, 0.30, 0.10],
                [0.25, 0.50, 0.25],
                [0.15, 0.25, 0.60],
                [0.34, 0.33, 0.33],
            ]
        ),
        dims=["grid", "component"],
        coords={"grid": [0, 1, 2, 3], "component": component},
    )
    return xr.Dataset(
        {
            "composition": composition,
            "score": score,
            "composition_grid": composition_grid,
        }
    )


def _make_single_sample_botorch_dataset() -> xr.Dataset:
    component = ["red", "green", "blue"]
    return xr.Dataset(
        {
            "composition": xr.DataArray(
                np.array([[0.70, 0.20, 0.10]]),
                dims=["sample", "component"],
                coords={"sample": [0], "component": component},
            ),
            "score": xr.DataArray(
                np.array([0.15]),
                dims=["sample"],
                coords={"sample": [0]},
            ),
            "composition_grid": xr.DataArray(
                np.array(
                    [
                        [0.70, 0.20, 0.10],
                        [0.25, 0.50, 0.25],
                        [0.10, 0.30, 0.60],
                    ]
                ),
                dims=["grid", "component"],
                coords={"grid": [0, 1, 2], "component": component},
            ),
        }
    )


class _FakePosterior:
    def __init__(self, mean, variance):
        self.mean = torch.as_tensor(mean, dtype=torch.double)
        self.variance = torch.as_tensor(variance, dtype=torch.double)


_FAKE_ACQ = object()


def closeness_to_target(X, target=0.3):
    """Module-level torch score: scalar for one (q, d) candidate set, higher near ``target``."""
    return -((X - target) ** 2).sum()


def numpy_closeness_to_target(X, target=0.3):
    """Same score written against NumPy; cannot be differentiated by autograd."""
    X = np.asarray(X)
    return -np.sum((X - target) ** 2)


def posterior_mean_score(X, model, best_f):
    """Score that consumes the fitted surrogate."""
    return model.posterior(X).mean.sum() - best_f


def distance_to_observed(X, observed, weight=1.0):
    """Score that consumes a dataset variable (``observed`` resolves to a tensor)."""
    return -weight * torch.cdist(X, observed).min()


class _FakeModel:
    def __init__(self, posterior_mean, posterior_variance):
        self._posterior_mean = torch.as_tensor(posterior_mean, dtype=torch.double)
        self._posterior_variance = torch.as_tensor(posterior_variance, dtype=torch.double)

    def posterior(self, x):
        return _FakePosterior(self._posterior_mean, self._posterior_variance)


FAKE_MODEL_REF = "file:///fake/model.pt"


def _with_model_ref(dataset, variable="bayesopt_model"):
    """Mimic an upstream regressor that saved its model and wrote the ref to the dataset."""
    dataset[variable] = xr.DataArray(FAKE_MODEL_REF)
    return dataset


def _patch_loaded_model(model):
    """Acquisition ops never fit: the model comes from ``torch_io.load_model``."""
    return patch("AFL.double_agent.torch_io.load_model", return_value=model)


def test_fit_single_task_gp_logs_inputs_when_botorch_rejects_nans(caplog):
    class _InputDataError(Exception):
        pass

    def reject_nan_inputs(**kwargs):
        raise _InputDataError("Input data contains NaN values.")

    fake_botorch = {
        "Standardize": lambda m: object(),
        "SingleTaskGP": reject_nan_inputs,
    }
    train_x = torch.tensor([[0.1], [float("nan")]], dtype=torch.double)
    train_y = torch.tensor([[1.0], [2.0]], dtype=torch.double)

    with patch.object(BoTorchMixin, "import_botorch", return_value=fake_botorch), pytest.raises(_InputDataError):
        BoTorchMixin.fit_single_task_gp(train_x, train_y, standardize=False)

    assert "train_x=tensor([[0.1000],\n        [   nan]]" in caplog.text
    assert "train_y=tensor([[1." in caplog.text
    assert "standardize=False" in caplog.text


@pytest.mark.unit
class TestBoTorchRegressor:
    def test_botorch_regressor_outputs_grid_statistics(self):
        dataset = _make_botorch_dataset()
        fake_model = _FakeModel(
            posterior_mean=[[0.9], [0.6], [0.2], [0.4]],
            posterior_variance=[[0.05], [0.04], [0.03], [0.02]],
        )

        regressor = BoTorchRegressor(
            feature_input_variable="composition",
            predictor_input_variable="score",
            output_prefix="bayesopt",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
            objective_direction="minimize",
            standardize=False,
            bounds=_make_explicit_bounds(),
        )

        with patch.object(BoTorchRegressor, "fit_single_task_gp", return_value=fake_model) as mock_fit:
            result = regressor.calculate(dataset)

        mock_fit.assert_called_once()
        np.testing.assert_allclose(result.output["bayesopt_mean"].values, [-0.9, -0.6, -0.2, -0.4])
        np.testing.assert_allclose(result.output["bayesopt_variance"].values, [0.05, 0.04, 0.03, 0.02])
        assert result.output["bayesopt_best_f"].item() == pytest.approx(0.15)
        assert result.output["bayesopt_is_simplex"].item() is False
        np.testing.assert_allclose(result.output["bayesopt_bounds"].values, _expected_bounds_array())
        assert result.output["bayesopt_bounds"].dims == ("bound", "component")
        assert "bayesopt_best_x" not in result.output

    def test_botorch_regressor_single_sample(self):
        dataset = _make_single_sample_botorch_dataset()
        fake_model = _FakeModel(
            posterior_mean=[[0.9], [0.4], [0.2]],
            posterior_variance=[[0.05], [0.03], [0.01]],
        )

        regressor = BoTorchRegressor(
            feature_input_variable="composition",
            predictor_input_variable="score",
            output_prefix="bayesopt",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
            objective_direction="minimize",
            standardize=False,
        )

        with patch.object(BoTorchRegressor, "fit_single_task_gp", return_value=fake_model) as mock_fit:
            result = regressor.calculate(dataset)

        mock_fit.assert_called_once()
        np.testing.assert_allclose(result.output["bayesopt_mean"].values, [-0.9, -0.4, -0.2])
        np.testing.assert_allclose(result.output["bayesopt_variance"].values, [0.05, 0.03, 0.01])
        assert result.output["bayesopt_best_f"].item() == pytest.approx(0.15)
        assert result.output["bayesopt_is_simplex"].item() is False
        assert "bayesopt_best_x" not in result.output

    def test_botorch_regressor_rejects_posterior_grid_shape_mismatch(self):
        dataset = _make_single_sample_botorch_dataset()
        fake_model = _FakeModel(
            posterior_mean=[[0.9]],
            posterior_variance=[[0.05]],
        )

        regressor = BoTorchRegressor(
            feature_input_variable="composition",
            predictor_input_variable="score",
            output_prefix="bayesopt",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
            objective_direction="minimize",
            standardize=False,
        )

        with patch.object(BoTorchRegressor, "fit_single_task_gp", return_value=fake_model):
            with pytest.raises(ValueError, match="Posterior outputs must match the candidate grid length"):
                regressor.calculate(dataset)


def _make_posterior_acquisition(**overrides):
    kwargs = dict(
        feature_input_variable="composition",
        grid_variable="composition_grid",
        grid_dim="grid",
        objective_direction="minimize",
        bounds=_make_explicit_bounds(),
        model_variable="bayesopt_model",
        output_prefix="bayesopt",
    )
    kwargs.update(overrides)
    return BoTorchPosteriorAcquisition(**kwargs)


@pytest.mark.unit
class TestBoTorchPosteriorAcquisition:
    def test_single_sample_preserves_output_shapes(self):
        dataset = _with_model_ref(_make_single_sample_botorch_dataset())
        fake_model = _FakeModel(
            posterior_mean=[[0.7], [0.4], [0.2]],
            posterior_variance=[[0.03], [0.02], [0.01]],
        )
        optimized_x = torch.tensor([[0.70, 0.20, 0.10]], dtype=torch.double)

        op = _make_posterior_acquisition()
        with _patch_loaded_model(fake_model), patch.object(
            BoTorchPosteriorAcquisition, "optimize_posterior_mean", return_value=(optimized_x, 0.15)
        ):
            result = op.calculate(dataset)

        assert result.output["bayesopt_best_x"].dims == ("component",)
        np.testing.assert_allclose(result.output["bayesopt_best_x"].values, [0.70, 0.20, 0.10])
        assert result.output["bayesopt_best_f"].item() == pytest.approx(0.15)
        assert op.model is fake_model

    def test_simplex_optimization_uses_constraints(self):
        dataset = _with_model_ref(_make_botorch_dataset())
        fake_model = _FakeModel(
            posterior_mean=[[0.8], [0.5], [0.3], [0.1]],
            posterior_variance=[[0.02], [0.02], [0.02], [0.02]],
        )
        optimized_x = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.double)

        op = _make_posterior_acquisition(num_restarts=4, raw_samples=16, is_simplex=True)
        with _patch_loaded_model(fake_model), patch.object(
            BoTorchMixin, "make_simplex_constraints", return_value=[("simplex",)]
        ) as mock_constraints, patch.object(
            BoTorchPosteriorAcquisition, "optimize_posterior_mean", return_value=(optimized_x, 0.12)
        ) as mock_optimize:
            result = op.calculate(dataset)

        mock_constraints.assert_called_once_with(3)
        optimize_kwargs = mock_optimize.call_args.kwargs
        expected_bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double)
        assert torch.equal(optimize_kwargs["bounds"], expected_bounds)
        assert optimize_kwargs["inequality_constraints"] == [("simplex",)]
        assert op.num_restarts == 4
        assert op.raw_samples == 16
        np.testing.assert_allclose(result.output["bayesopt_best_x"].values, [0.2, 0.3, 0.5])
        assert list(result.output["bayesopt_best_x"].coords["component"].values) == ["red", "green", "blue"]
        assert result.output["bayesopt_best_f"].item() == pytest.approx(0.12)

    def test_falls_back_to_best_grid_point_when_optimization_fails(self):
        dataset = _with_model_ref(_make_single_sample_botorch_dataset())
        fake_model = _FakeModel(
            posterior_mean=[[0.9], [0.4], [0.2]],
            posterior_variance=[[0.05], [0.03], [0.01]],
        )

        op = _make_posterior_acquisition()
        with _patch_loaded_model(fake_model), patch.object(
            BoTorchPosteriorAcquisition,
            "optimize_posterior_mean",
            side_effect=RuntimeError("probability tensor contains either `inf`, `nan` or element < 0"),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = op.calculate(dataset)

        assert any("falling back to the best evaluated grid point" in str(item.message) for item in caught)
        # The model is trained on negated targets when minimizing, so the internal posterior
        # mean [0.9, 0.4, 0.2] corresponds to objective values [-0.9, -0.4, -0.2]; the best
        # (lowest) objective is at the first grid point.
        np.testing.assert_allclose(result.output["bayesopt_best_x"].values, [0.70, 0.20, 0.10])
        assert result.output["bayesopt_best_f"].item() == pytest.approx(-0.9)

    def test_requires_explicit_bounds(self):
        with pytest.raises(ValueError, match="requires explicit `bounds`"):
            _make_posterior_acquisition(bounds=None)

    def test_requires_model_variable(self):
        with pytest.raises(ValueError, match="model_variable"):
            _make_posterior_acquisition(model_variable=None)

    def test_loads_model_from_dataset_variable(self):
        dataset = _with_model_ref(_make_botorch_dataset())
        fake_model = _FakeModel(
            posterior_mean=[[0.8], [0.5], [0.3], [0.1]],
            posterior_variance=[[0.02], [0.02], [0.02], [0.02]],
        )
        optimized_x = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.double)

        op = _make_posterior_acquisition()
        with patch.object(BoTorchMixin, "fit_single_task_gp") as mock_fit, _patch_loaded_model(
            fake_model
        ) as mock_load, patch.object(
            BoTorchPosteriorAcquisition, "optimize_posterior_mean", return_value=(optimized_x, 0.12)
        ):
            result = op.calculate(dataset)

        mock_fit.assert_not_called()
        mock_load.assert_called_once_with(FAKE_MODEL_REF)
        assert op.model is fake_model
        assert result.output["bayesopt_best_f"].item() == pytest.approx(0.12)

    def test_missing_model_variable_fails_loudly(self):
        op = _make_posterior_acquisition()
        with pytest.raises(KeyError, match="bayesopt_model"):
            op.calculate(_make_botorch_dataset())

    def test_posterior_mean_is_maximized_through_mixin_optimize_acqf(self):
        dataset = _with_model_ref(_make_botorch_dataset())
        fake_model = _FakeModel(
            posterior_mean=[[0.8], [0.5], [0.3], [0.1]],
            posterior_variance=[[0.02], [0.02], [0.02], [0.02]],
        )
        optimized_x = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.double)
        posterior_mean = object()

        op = _make_posterior_acquisition()
        with _patch_loaded_model(fake_model), patch.object(
            BoTorchPosteriorAcquisition, "botorch", {"PosteriorMean": lambda model: posterior_mean}
        ), patch.object(
            BoTorchPosteriorAcquisition, "optimize_acqf", return_value=(optimized_x, torch.tensor([0.9]))
        ) as mock_optimize:
            result = op.calculate(dataset)

        assert mock_optimize.call_args.args[0] is posterior_mean
        assert mock_optimize.call_args.kwargs["q"] == 1
        # minimize: the GP lives in negated space, so best_f comes back negated
        assert result.output["bayesopt_best_f"].item() == pytest.approx(-0.9)


@pytest.mark.unit
class TestBoTorchAcquisition:
    def test_botorch_acquisition_uses_dataset_best_f_and_outputs_next_sample(self):
        dataset = _with_model_ref(_make_botorch_dataset())
        dataset["bayesopt_best_f"] = xr.DataArray(0.22)
        fake_model = _FakeModel(
            posterior_mean=[[0.0], [0.0], [0.0], [0.0]],
            posterior_variance=[[0.01], [0.01], [0.01], [0.01]],
        )
        optimized_x = torch.tensor([[0.25, 0.50, 0.25]], dtype=torch.double)
        acq_values = torch.tensor([0.1, 0.4, 0.2, 0.3], dtype=torch.double)

        acquisition = BoTorchAcquisition(
            feature_input_variable="composition",
            grid_variable="composition_grid",
            grid_dim="grid",
            objective_direction="minimize",
            output_prefix="bayesopt",
            output_variable="next_sample",
            model_variable="bayesopt_model",
            best_f_variable="bayesopt_best_f",
            count=1,
            bounds=_make_explicit_bounds(),
        )
        assert set(acquisition.input_variable) >= {"composition", "composition_grid", "bayesopt_model", "bayesopt_best_f"}

        with _patch_loaded_model(fake_model), patch.object(
            BoTorchAcquisition, "build_builtin_acquisition", return_value=_FAKE_ACQ
        ) as mock_build, patch.object(
            BoTorchAcquisition, "evaluate_acquisition_on_grid", return_value=acq_values
        ) as mock_eval, patch.object(
            BoTorchAcquisition,
            "optimize_acquisition_function",
            return_value=(optimized_x, torch.tensor(0.4, dtype=torch.double)),
        ) as mock_optimize:
            result = acquisition.calculate(dataset)

        assert mock_build.call_args.kwargs["best_f"] == pytest.approx(-0.22)
        assert mock_build.call_args.args[0] == "logei"
        assert mock_optimize.call_args.args[0] is _FAKE_ACQ
        expected_bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double)
        assert torch.equal(mock_optimize.call_args.kwargs["bounds"], expected_bounds)
        np.testing.assert_allclose(result.output["bayesopt_decision_surface"].values, [0.1, 0.4, 0.2, 0.3])
        np.testing.assert_allclose(result.output["bayesopt_bounds"].values, _expected_bounds_array())
        assert result.output["bayesopt_bounds"].dims == ("bound", "component")
        np.testing.assert_allclose(result.output["next_sample"].values, [[0.25, 0.50, 0.25]])
        assert list(result.output["next_sample"].coords["component"].values) == ["red", "green", "blue"]

    def test_botorch_acquisition_single_sample_keeps_2d_next_sample_output(self):
        dataset = _with_model_ref(_make_single_sample_botorch_dataset())
        dataset["bayesopt_best_f"] = xr.DataArray(0.15)
        fake_model = _FakeModel(
            posterior_mean=[[0.0], [0.0], [0.0]],
            posterior_variance=[[0.01], [0.01], [0.01]],
        )
        optimized_x = torch.tensor([0.25, 0.50, 0.25], dtype=torch.double)
        acq_values = torch.tensor([0.3, 0.2, 0.1], dtype=torch.double)

        acquisition = BoTorchAcquisition(
            feature_input_variable="composition",
            grid_variable="composition_grid",
            grid_dim="grid",
            objective_direction="minimize",
            output_prefix="bayesopt",
            output_variable="next_sample",
            model_variable="bayesopt_model",
            best_f_variable="bayesopt_best_f",
            count=1,
            bounds=_make_explicit_bounds(),
        )

        with _patch_loaded_model(fake_model), patch.object(
            BoTorchAcquisition, "build_builtin_acquisition", return_value=_FAKE_ACQ
        ) as mock_build, patch.object(
            BoTorchAcquisition, "evaluate_acquisition_on_grid", return_value=acq_values
        ), patch.object(
            BoTorchAcquisition,
            "optimize_acquisition_function",
            return_value=(optimized_x, torch.tensor(0.3, dtype=torch.double)),
        ):
            result = acquisition.calculate(dataset)

        assert result.output["bayesopt_decision_surface"].dims == ("grid",)
        assert result.output["next_sample"].dims == ("next_sample", "component")
        assert result.output["next_sample"].shape == (1, 3)
        np.testing.assert_allclose(result.output["next_sample"].values, [[0.25, 0.50, 0.25]])

    def test_botorch_acquisition_uses_dataset_simplex_flag_without_warning(self):
        dataset = _with_model_ref(_make_botorch_dataset())
        dataset["bayesopt_is_simplex"] = xr.DataArray(True)
        dataset["bayesopt_best_f"] = xr.DataArray(0.15)
        fake_model = _FakeModel(
            posterior_mean=[[0.0], [0.0], [0.0], [0.0]],
            posterior_variance=[[0.01], [0.01], [0.01], [0.01]],
        )
        optimized_x = torch.tensor(
            [
                [0.20, 0.30, 0.50],
                [0.45, 0.35, 0.20],
            ],
            dtype=torch.double,
        )
        acq_values = torch.tensor([0.5, 0.1, 0.3, 0.2], dtype=torch.double)

        acquisition = BoTorchAcquisition(
            feature_input_variable="composition",
            grid_variable="composition_grid",
            grid_dim="grid",
            objective_direction="minimize",
            output_prefix="bayesopt",
            output_variable="next_sample",
            model_variable="bayesopt_model",
            best_f_variable="bayesopt_best_f",
            count=2,
            acquisition=None,
            bounds=_make_explicit_bounds(),
            is_simplex=False,
        )

        with _patch_loaded_model(fake_model), patch.object(
            BoTorchAcquisition, "build_builtin_acquisition", return_value=_FAKE_ACQ
        ) as mock_build, patch.object(
            BoTorchAcquisition, "evaluate_acquisition_on_grid", return_value=acq_values
        ) as mock_eval, patch.object(
            BoTorchAcquisition,
            "optimize_acquisition_function",
            return_value=(optimized_x, torch.tensor(0.5, dtype=torch.double)),
        ) as mock_optimize:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = acquisition.calculate(dataset)

        assert not caught
        assert mock_eval.call_args.kwargs["q"] == 2
        assert mock_build.call_args.args[0] == "qlogei"
        assert mock_optimize.call_args.kwargs["inequality_constraints"] is None
        np.testing.assert_allclose(result.output["next_sample"].values, optimized_x.numpy())

    def test_botorch_acquisition_simplex_passes_constraints(self):
        dataset = _with_model_ref(_make_botorch_dataset())
        dataset["bayesopt_best_f"] = xr.DataArray(0.55)
        fake_model = _FakeModel(
            posterior_mean=[[0.0], [0.0], [0.0], [0.0]],
            posterior_variance=[[0.01], [0.01], [0.01], [0.01]],
        )
        optimized_x = torch.tensor([[0.34, 0.33, 0.33]], dtype=torch.double)
        acq_values = torch.tensor([0.2, 0.1, 0.4, 0.3], dtype=torch.double)

        acquisition = BoTorchAcquisition(
            feature_input_variable="composition",
            grid_variable="composition_grid",
            grid_dim="grid",
            objective_direction="maximize",
            output_prefix="bayesopt",
            output_variable="next_sample",
            model_variable="bayesopt_model",
            best_f_variable="bayesopt_best_f",
            count=1,
            acquisition="logei",
            bounds=_make_explicit_bounds(),
            is_simplex=True,
        )

        with _patch_loaded_model(fake_model), patch.object(
            BoTorchAcquisition, "build_builtin_acquisition", return_value=_FAKE_ACQ
        ) as mock_build, patch.object(
            BoTorchMixin, "make_simplex_constraints", return_value=[("simplex",)]
        ) as mock_constraints, patch.object(
            BoTorchAcquisition, "evaluate_acquisition_on_grid", return_value=acq_values
        ), patch.object(
            BoTorchAcquisition,
            "optimize_acquisition_function",
            return_value=(optimized_x, torch.tensor(0.4, dtype=torch.double)),
        ) as mock_optimize:
            result = acquisition.calculate(dataset)

        mock_constraints.assert_called_once_with(3)
        assert mock_optimize.call_args.kwargs["inequality_constraints"] == [("simplex",)]
        np.testing.assert_allclose(result.output["next_sample"].values, [[0.34, 0.33, 0.33]])

    def test_botorch_acquisition_falls_back_to_best_grid_point_when_optimization_fails(self):
        dataset = _with_model_ref(_make_single_sample_botorch_dataset())
        dataset["bayesopt_best_f"] = xr.DataArray(0.15)
        fake_model = _FakeModel(
            posterior_mean=[[0.0], [0.0], [0.0]],
            posterior_variance=[[0.01], [0.01], [0.01]],
        )
        acq_values = torch.tensor([0.3, 0.2, 0.1], dtype=torch.double)

        acquisition = BoTorchAcquisition(
            feature_input_variable="composition",
            grid_variable="composition_grid",
            grid_dim="grid",
            objective_direction="minimize",
            output_prefix="bayesopt",
            output_variable="next_sample",
            model_variable="bayesopt_model",
            best_f_variable="bayesopt_best_f",
            count=1,
            bounds=_make_explicit_bounds(),
        )

        with _patch_loaded_model(fake_model), patch.object(
            BoTorchAcquisition, "build_builtin_acquisition", return_value=_FAKE_ACQ
        ) as mock_build, patch.object(
            BoTorchAcquisition, "evaluate_acquisition_on_grid", return_value=acq_values
        ), patch.object(
            BoTorchAcquisition,
            "optimize_acquisition_function",
            side_effect=RuntimeError("probability tensor contains either `inf`, `nan` or element < 0"),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = acquisition.calculate(dataset)

        assert any("falling back to the best evaluated grid point" in str(item.message) for item in caught)
        np.testing.assert_allclose(result.output["bayesopt_decision_surface"].values, [0.3, 0.2, 0.1])
        np.testing.assert_allclose(result.output["next_sample"].values, [[0.70, 0.20, 0.10]])

    def test_botorch_acquisition_requires_explicit_bounds(self):
        common = dict(
            feature_input_variable="composition",
            grid_variable="composition_grid",
        )
        with pytest.raises(TypeError, match="bounds"):
            BoTorchAcquisition(**common)
        with pytest.raises(ValueError, match="requires explicit `bounds`"):
            BoTorchAcquisition(bounds=None, **common)

    def test_builtin_acquisition_without_model_variable_fails_loudly(self):
        from AFL.double_agent.AcquisitionFunction import AcquisitionError

        dataset = _make_botorch_dataset()
        dataset["bayesopt_best_f"] = xr.DataArray(0.15)
        op = _make_score_acquisition(acquisition="logei", best_f_variable="bayesopt_best_f")
        with pytest.raises(AcquisitionError, match="model_variable"):
            op.calculate(dataset)

    def test_builtin_acquisition_without_best_f_variable_fails_loudly(self):
        from AFL.double_agent.AcquisitionFunction import AcquisitionError

        dataset = _with_model_ref(_make_botorch_dataset())
        op = _make_score_acquisition(acquisition="logei", model_variable="bayesopt_model")
        with _patch_loaded_model(_FakeModel([[0.0]] * 4, [[0.01]] * 4)), pytest.raises(
            AcquisitionError, match="best_f_variable"
        ):
            op.calculate(dataset)


def _make_score_acquisition(**overrides):
    kwargs = dict(
        feature_input_variable="composition",
        grid_variable="composition_grid",
        grid_dim="grid",
        output_prefix="bayesopt",
        output_variable="next_sample",
        bounds=_make_explicit_bounds(),
        num_restarts=4,
        raw_samples=32,
    )
    kwargs.update(overrides)
    return BoTorchAcquisition(**kwargs)


@pytest.mark.unit
class TestBoTorchAcquisitionCustomAcquisition:
    def test_torch_score_function_is_optimized_over_bounds_without_fitting_gp(self):
        torch.manual_seed(0)
        op = _make_score_acquisition(acquisition={"function": closeness_to_target, "args": {"target": 0.25}})
        with patch.object(BoTorchMixin, "fit_single_task_gp") as mock_fit:
            result = op.calculate(_make_botorch_dataset())

        mock_fit.assert_not_called()
        assert op.acquisition_function.differentiable is True
        np.testing.assert_allclose(result.output["next_sample"].values, [[0.25, 0.25, 0.25]], atol=1e-3)
        grid = _make_botorch_dataset()["composition_grid"].values
        np.testing.assert_allclose(
            result.output["bayesopt_decision_surface"].values, -((grid - 0.25) ** 2).sum(axis=1)
        )

    def test_bare_function_is_accepted(self):
        torch.manual_seed(0)
        op = _make_score_acquisition(acquisition=closeness_to_target)
        result = op.calculate(_make_botorch_dataset())
        np.testing.assert_allclose(result.output["next_sample"].values, [[0.3, 0.3, 0.3]], atol=1e-3)

    def test_numpy_score_function_falls_back_to_finite_differences(self):
        torch.manual_seed(0)
        op = _make_score_acquisition(acquisition=numpy_closeness_to_target)
        with patch.object(BoTorchMixin, "fit_single_task_gp") as mock_fit, patch.object(
            BoTorchAcquisition, "optimize_acqf", wraps=op.optimize_acqf
        ) as mock_optimize:
            result = op.calculate(_make_botorch_dataset())

        mock_fit.assert_not_called()
        assert op.acquisition_function.differentiable is False
        assert mock_optimize.call_args.kwargs["with_grad"] is False
        assert mock_optimize.call_args.kwargs["q"] == 1
        np.testing.assert_allclose(result.output["next_sample"].values, [[0.3, 0.3, 0.3]], atol=1e-2)

    def test_function_receives_joint_candidate_set_without_batch_dimension(self):
        seen = []

        def record_shape(X):
            seen.append(tuple(X.shape))
            return -((X - 0.3) ** 2).sum()

        from AFL.double_agent.BoTorchMixin import ScoreFunctionAcquisition

        acq = ScoreFunctionAcquisition(record_shape)
        out = acq(torch.zeros(5, 2, 3, dtype=torch.double))
        assert out.shape == (5,)
        assert set(seen) == {(2, 3)}

    def test_batch_optimization_returns_count_points(self):
        torch.manual_seed(0)
        op = _make_score_acquisition(acquisition=closeness_to_target, count=2)
        result = op.calculate(_make_botorch_dataset())
        assert result.output["next_sample"].shape == (2, 3)
        np.testing.assert_allclose(result.output["next_sample"].values, 0.3, atol=1e-3)

    def test_score_function_requesting_model_loads_upstream_model(self):
        dataset = _with_model_ref(_make_botorch_dataset())
        dataset["bayesopt_best_f"] = xr.DataArray(0.15)
        fake_model = _FakeModel(
            posterior_mean=[[0.0], [0.0], [0.0], [0.0]],
            posterior_variance=[[0.01], [0.01], [0.01], [0.01]],
        )
        optimized_x = torch.tensor([[0.25, 0.50, 0.25]], dtype=torch.double)
        gp, best_f = Variable("model"), Variable("best_f")
        op = _make_score_acquisition(
            acquisition=lambda X: posterior_mean_score(X, gp, best_f),
            objective_direction="minimize",
            model_variable="bayesopt_model",
            best_f_variable="bayesopt_best_f",
        )
        assert "ref" in op.to_json()["args"]["acquisition"]
        with patch.object(BoTorchMixin, "fit_single_task_gp") as mock_fit, _patch_loaded_model(
            fake_model
        ) as mock_load, patch.object(
            BoTorchAcquisition,
            "optimize_acquisition_function",
            return_value=(optimized_x, torch.tensor(0.0, dtype=torch.double)),
        ):
            result = op.calculate(dataset)

        mock_fit.assert_not_called()
        mock_load.assert_called_once_with(FAKE_MODEL_REF)
        assert op.model is fake_model
        # best_f is the upstream regressor's incumbent, negated into maximization space
        assert op._provided["best_f"] == pytest.approx(-0.15)
        np.testing.assert_allclose(result.output["bayesopt_decision_surface"].values, 0.15)
        np.testing.assert_allclose(result.output["next_sample"].values, optimized_x.numpy())

    def test_score_function_requesting_model_without_model_variable_fails_loudly(self):
        from AFL.double_agent.AcquisitionFunction import AcquisitionError

        gp = Variable("model")
        op = _make_score_acquisition(acquisition=lambda X: gp.posterior(X).mean.sum())
        with pytest.raises(AcquisitionError, match="model_variable"):
            op.calculate(_make_botorch_dataset())

    def test_score_function_args_resolve_dataset_variables_without_gp_fit(self):
        dataset = _make_botorch_dataset()
        optimized_x = torch.tensor([[0.25, 0.50, 0.25]], dtype=torch.double)
        observed = Variable("composition")
        op = _make_score_acquisition(acquisition=lambda X: distance_to_observed(X, observed, weight=2.0))
        assert "composition" in op.input_variable  # discovered from the lambda's closure
        with patch.object(BoTorchMixin, "fit_single_task_gp") as mock_fit, patch.object(
            BoTorchAcquisition,
            "optimize_acquisition_function",
            return_value=(optimized_x, torch.tensor(0.0, dtype=torch.double)),
        ):
            result = op.calculate(dataset)

        mock_fit.assert_not_called()
        grid = torch.as_tensor(dataset["composition_grid"].values, dtype=torch.double)
        observed = torch.as_tensor(dataset["composition"].values, dtype=torch.double)
        expected = [distance_to_observed(x[None], observed, weight=2.0).item() for x in grid]
        np.testing.assert_allclose(result.output["bayesopt_decision_surface"].values, expected)

    def test_function_spec_args_resolve_variables(self):
        dataset = _make_botorch_dataset()
        optimized_x = torch.tensor([[0.25, 0.50, 0.25]], dtype=torch.double)
        op = _make_score_acquisition(
            acquisition={
                "function": distance_to_observed,
                "args": {"observed": Variable("composition"), "weight": 2.0},
            }
        )
        assert op.acquisition["args"]["observed"] == {"variable": "composition"}  # JSON form
        payload = json.loads(json.dumps(op.to_json()))
        restored = BoTorchAcquisition.from_json(payload)
        with patch.object(
            BoTorchAcquisition,
            "optimize_acquisition_function",
            return_value=(optimized_x, torch.tensor(0.0, dtype=torch.double)),
        ):
            result = restored.calculate(dataset)
        assert isinstance(restored.acquisition_function.kwargs["observed"], torch.Tensor)
        assert np.isfinite(result.output["bayesopt_decision_surface"].values).all()

    def test_botorch_class_spec_is_constructed_with_model(self):
        pytest.importorskip("botorch")
        dataset = _make_botorch_dataset()
        train_x = torch.as_tensor(dataset["composition"].values, dtype=torch.double)
        train_y = torch.as_tensor(dataset["score"].values, dtype=torch.double).unsqueeze(-1)
        model = BoTorchMixin.fit_single_task_gp(train_x=train_x, train_y=train_y, standardize=False)
        from botorch.acquisition import UpperConfidenceBound

        torch.manual_seed(0)
        op = _make_score_acquisition(
            acquisition={"class": UpperConfidenceBound, "args": {"beta": 2.0}}, model_variable="bayesopt_model"
        )
        assert op.acquisition == {"class": "botorch.acquisition.analytic.UpperConfidenceBound", "args": {"beta": 2.0}}
        with _patch_loaded_model(model):
            result = op.calculate(_with_model_ref(dataset))

        assert isinstance(op.acquisition_function, UpperConfidenceBound)
        assert op.acquisition_function.beta == pytest.approx(2.0)
        assert result.output["next_sample"].shape == (1, 3)
        assert np.isfinite(result.output["bayesopt_decision_surface"].values).all()

    def test_function_spec_round_trips_through_json(self):
        op = _make_score_acquisition(acquisition={"function": closeness_to_target, "args": {"target": 0.1}})
        spec = op.acquisition
        assert spec == {"function": f"{closeness_to_target.__module__}.closeness_to_target", "args": {"target": 0.1}}

        import json

        payload = json.loads(json.dumps(op.to_json()))
        restored = BoTorchAcquisition.from_json(payload)
        assert restored.acquisition == spec
        torch.manual_seed(0)
        result = restored.calculate(_make_botorch_dataset())
        np.testing.assert_allclose(result.output["next_sample"].values, [[0.1, 0.1, 0.1]], atol=1e-3)

    def test_lambda_is_kept_live_and_exported_as_ref(self):
        from AFL.double_agent.variables import UnresolvedReferenceError

        op = _make_score_acquisition(acquisition=lambda X: -((X - 0.3) ** 2).sum())
        payload = json.loads(json.dumps(op.to_json()))
        assert payload["args"]["acquisition"]["ref"] == f"{op.name}.acquisition"
        with pytest.raises(UnresolvedReferenceError):
            BoTorchAcquisition.from_json(payload)
        restored = BoTorchAcquisition.from_json(payload, objects=op.live_objects())
        torch.manual_seed(0)
        result = restored.calculate(_make_botorch_dataset())
        np.testing.assert_allclose(result.output["next_sample"].values, 0.3, atol=1e-3)

    def test_non_scalar_score_is_reported(self):
        def per_point_scores(X):
            return X.sum(-1)

        from AFL.double_agent.BoTorchMixin import ScoreFunctionAcquisition

        acq = ScoreFunctionAcquisition(per_point_scores)
        with pytest.raises(ValueError, match="single scalar"):
            acq(torch.zeros(4, 2, 3, dtype=torch.double))
