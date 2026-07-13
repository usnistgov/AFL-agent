"""
Unit tests for BoTorch-based extrapolation and acquisition helpers.
"""

import warnings
from unittest.mock import patch

import numpy as np
import pytest
import torch
import xarray as xr

from AFL.double_agent.AcquisitionFunction import BoTorchAcquisition
from AFL.double_agent.PyTorchExtrapolator import BoTorchRegressor


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


class _FakeModel:
    def __init__(self, posterior_mean, posterior_variance, simplex_transform=None):
        self._posterior_mean = torch.as_tensor(posterior_mean, dtype=torch.double)
        self._posterior_variance = torch.as_tensor(posterior_variance, dtype=torch.double)
        self._simplex_input_transform = simplex_transform

    def posterior(self, x):
        return _FakePosterior(self._posterior_mean, self._posterior_variance)


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

        with patch("AFL.double_agent.PyTorchExtrapolator.fit_single_task_gp", return_value=fake_model) as mock_fit:
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

        with patch("AFL.double_agent.PyTorchExtrapolator.fit_single_task_gp", return_value=fake_model) as mock_fit:
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

        with patch("AFL.double_agent.PyTorchExtrapolator.fit_single_task_gp", return_value=fake_model):
            with pytest.raises(ValueError, match="Posterior outputs must match the candidate grid length"):
                regressor.calculate(dataset)

    def test_botorch_regressor_single_sample_preserves_output_shapes(self):
        dataset = _make_single_sample_botorch_dataset()
        fake_model = _FakeModel(
            posterior_mean=[[0.7], [0.4], [0.2]],
            posterior_variance=[[0.03], [0.02], [0.01]],
        )
        optimized_x = torch.tensor([[0.70, 0.20, 0.10]], dtype=torch.double)

        regressor = BoTorchRegressor(
            feature_input_variable="composition",
            predictor_input_variable="score",
            output_prefix="bayesopt",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
            objective_direction="minimize",
            standardize=False,
            posterior_optimize=True,
            bounds=_make_explicit_bounds(),
        )

        with patch("AFL.double_agent.PyTorchExtrapolator.fit_single_task_gp", return_value=fake_model), patch(
            "AFL.double_agent.PyTorchExtrapolator.optimize_posterior_mean",
            return_value=(optimized_x, 0.15),
        ):
            result = regressor.calculate(dataset)

        assert result.output["bayesopt_mean"].dims == ("grid",)
        assert result.output["bayesopt_variance"].dims == ("grid",)
        assert result.output["bayesopt_best_x"].dims == ("component",)
        np.testing.assert_allclose(result.output["bayesopt_best_x"].values, [0.70, 0.20, 0.10])
        assert result.output["bayesopt_best_f"].item() == pytest.approx(0.15)

    def test_botorch_regressor_simplex_optimization_uses_constraints(self):
        dataset = _make_botorch_dataset()
        fake_model = _FakeModel(
            posterior_mean=[[0.8], [0.5], [0.3], [0.1]],
            posterior_variance=[[0.02], [0.02], [0.02], [0.02]],
            simplex_transform=lambda x: x + 0.01,
        )
        optimized_x = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.double)

        regressor = BoTorchRegressor(
            feature_input_variable="composition",
            predictor_input_variable="score",
            output_prefix="bayesopt",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
            objective_direction="minimize",
            standardize=False,
            posterior_optimize=True,
            posterior_optimize_restarts=4,
            posterior_optimize_raw_samples=16,
            bounds=_make_explicit_bounds(),
            is_simplex=True,
        )

        with patch("AFL.double_agent.PyTorchExtrapolator.fit_single_task_gp", return_value=fake_model), patch(
            "AFL.double_agent.PyTorchExtrapolator.make_simplex_constraints",
            return_value=[("simplex",)],
        ) as mock_constraints, patch(
            "AFL.double_agent.PyTorchExtrapolator.optimize_posterior_mean",
            return_value=(optimized_x, 0.12),
        ) as mock_optimize:
            result = regressor.calculate(dataset)

        mock_constraints.assert_called_once_with(3)
        optimize_kwargs = mock_optimize.call_args.kwargs
        expected_bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double)
        assert torch.equal(optimize_kwargs["bounds"], expected_bounds)
        assert optimize_kwargs["inequality_constraints"] == [("simplex",)]
        assert optimize_kwargs["num_restarts"] == 4
        assert optimize_kwargs["raw_samples"] == 16
        assert result.output["bayesopt_is_simplex"].item() is True
        np.testing.assert_allclose(result.output["bayesopt_best_x"].values, [0.2, 0.3, 0.5])
        assert list(result.output["bayesopt_best_x"].coords["component"].values) == ["red", "green", "blue"]
        assert result.output["bayesopt_best_f"].item() == pytest.approx(0.12)

    def test_botorch_regressor_falls_back_to_best_grid_point_when_posterior_optimization_fails(self):
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
            posterior_optimize=True,
            bounds=_make_explicit_bounds(),
        )

        with patch("AFL.double_agent.PyTorchExtrapolator.fit_single_task_gp", return_value=fake_model), patch(
            "AFL.double_agent.PyTorchExtrapolator.optimize_posterior_mean",
            side_effect=RuntimeError("probability tensor contains either `inf`, `nan` or element < 0"),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = regressor.calculate(dataset)

        assert any("falling back to the best evaluated grid point" in str(item.message) for item in caught)
        np.testing.assert_allclose(result.output["bayesopt_best_x"].values, [0.10, 0.30, 0.60])
        assert result.output["bayesopt_best_f"].item() == pytest.approx(-0.2)

    def test_botorch_regressor_requires_explicit_bounds_for_posterior_optimization(self):
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
            posterior_optimize=True,
        )

        with patch("AFL.double_agent.PyTorchExtrapolator.fit_single_task_gp", return_value=fake_model):
            with pytest.raises(ValueError, match="requires explicit `bounds`"):
                regressor.calculate(dataset)


@pytest.mark.unit
class TestBoTorchAcquisition:
    def test_botorch_acquisition_uses_dataset_best_f_and_outputs_next_sample(self):
        dataset = _make_botorch_dataset()
        dataset["bayesopt_best_f"] = xr.DataArray(0.22)
        fake_model = _FakeModel(
            posterior_mean=[[0.0], [0.0], [0.0], [0.0]],
            posterior_variance=[[0.01], [0.01], [0.01], [0.01]],
        )
        optimized_x = torch.tensor([[0.25, 0.50, 0.25]], dtype=torch.double)
        acq_values = torch.tensor([0.1, 0.4, 0.2, 0.3], dtype=torch.double)

        acquisition = BoTorchAcquisition(
            feature_input_variable="composition",
            predictor_input_variable="score",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
            objective_direction="minimize",
            standardize=False,
            output_prefix="bayesopt",
            output_variable="next_sample",
            best_f_variable="bayesopt_best_f",
            count=1,
            bounds=_make_explicit_bounds(),
        )

        with patch("AFL.double_agent.AcquisitionFunction.fit_single_task_gp", return_value=fake_model), patch(
            "AFL.double_agent.AcquisitionFunction.evaluate_log_expected_improvement",
            return_value=acq_values,
        ) as mock_eval, patch(
            "AFL.double_agent.AcquisitionFunction.optimize_acquisition_function",
            return_value=(optimized_x, torch.tensor(0.4, dtype=torch.double)),
        ) as mock_optimize:
            result = acquisition.calculate(dataset)

        assert mock_eval.call_args.kwargs["best_f"] == pytest.approx(-0.22)
        assert mock_optimize.call_args.kwargs["acquisition_kind"] == "logei"
        expected_bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double)
        assert torch.equal(mock_optimize.call_args.kwargs["bounds"], expected_bounds)
        np.testing.assert_allclose(result.output["bayesopt_decision_surface"].values, [0.1, 0.4, 0.2, 0.3])
        np.testing.assert_allclose(result.output["bayesopt_bounds"].values, _expected_bounds_array())
        assert result.output["bayesopt_bounds"].dims == ("bound", "component")
        np.testing.assert_allclose(result.output["next_sample"].values, [[0.25, 0.50, 0.25]])
        assert list(result.output["next_sample"].coords["component"].values) == ["red", "green", "blue"]

    def test_botorch_acquisition_single_sample_keeps_2d_next_sample_output(self):
        dataset = _make_single_sample_botorch_dataset()
        fake_model = _FakeModel(
            posterior_mean=[[0.0], [0.0], [0.0]],
            posterior_variance=[[0.01], [0.01], [0.01]],
        )
        optimized_x = torch.tensor([0.25, 0.50, 0.25], dtype=torch.double)
        acq_values = torch.tensor([0.3, 0.2, 0.1], dtype=torch.double)

        acquisition = BoTorchAcquisition(
            feature_input_variable="composition",
            predictor_input_variable="score",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
            objective_direction="minimize",
            standardize=False,
            output_prefix="bayesopt",
            output_variable="next_sample",
            count=1,
            bounds=_make_explicit_bounds(),
        )

        with patch("AFL.double_agent.AcquisitionFunction.fit_single_task_gp", return_value=fake_model), patch(
            "AFL.double_agent.AcquisitionFunction.evaluate_log_expected_improvement",
            return_value=acq_values,
        ), patch(
            "AFL.double_agent.AcquisitionFunction.optimize_acquisition_function",
            return_value=(optimized_x, torch.tensor(0.3, dtype=torch.double)),
        ):
            result = acquisition.calculate(dataset)

        assert result.output["bayesopt_decision_surface"].dims == ("grid",)
        assert result.output["next_sample"].dims == ("next_sample", "component")
        assert result.output["next_sample"].shape == (1, 3)
        np.testing.assert_allclose(result.output["next_sample"].values, [[0.25, 0.50, 0.25]])

    def test_botorch_acquisition_simplex_qlogei_warns_on_geometry_mismatch(self):
        dataset = _make_botorch_dataset()
        dataset["bayesopt_is_simplex"] = xr.DataArray(True)
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
            predictor_input_variable="score",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
            objective_direction="minimize",
            standardize=False,
            output_prefix="bayesopt",
            output_variable="next_sample",
            count=2,
            acquisition_kind="auto",
            bounds=_make_explicit_bounds(),
            is_simplex=False,
        )

        with patch("AFL.double_agent.AcquisitionFunction.fit_single_task_gp", return_value=fake_model), patch(
            "AFL.double_agent.AcquisitionFunction.evaluate_qlog_expected_improvement",
            return_value=acq_values,
        ) as mock_eval, patch(
            "AFL.double_agent.AcquisitionFunction.optimize_acquisition_function",
            return_value=(optimized_x, torch.tensor(0.5, dtype=torch.double)),
        ) as mock_optimize:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = acquisition.calculate(dataset)

        assert any("simplex-aware geometry" in str(item.message) for item in caught)
        assert mock_eval.call_args.kwargs["q"] == 2
        assert mock_optimize.call_args.kwargs["acquisition_kind"] == "qlogei"
        assert mock_optimize.call_args.kwargs["inequality_constraints"] is None
        np.testing.assert_allclose(result.output["next_sample"].values, optimized_x.numpy())

    def test_botorch_acquisition_simplex_passes_constraints(self):
        dataset = _make_botorch_dataset()
        fake_model = _FakeModel(
            posterior_mean=[[0.0], [0.0], [0.0], [0.0]],
            posterior_variance=[[0.01], [0.01], [0.01], [0.01]],
        )
        optimized_x = torch.tensor([[0.34, 0.33, 0.33]], dtype=torch.double)
        acq_values = torch.tensor([0.2, 0.1, 0.4, 0.3], dtype=torch.double)

        acquisition = BoTorchAcquisition(
            feature_input_variable="composition",
            predictor_input_variable="score",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
            objective_direction="maximize",
            standardize=False,
            output_prefix="bayesopt",
            output_variable="next_sample",
            count=1,
            acquisition_kind="logei",
            bounds=_make_explicit_bounds(),
            is_simplex=True,
        )

        with patch("AFL.double_agent.AcquisitionFunction.fit_single_task_gp", return_value=fake_model), patch(
            "AFL.double_agent.AcquisitionFunction.make_simplex_constraints",
            return_value=[("simplex",)],
        ) as mock_constraints, patch(
            "AFL.double_agent.AcquisitionFunction.evaluate_log_expected_improvement",
            return_value=acq_values,
        ), patch(
            "AFL.double_agent.AcquisitionFunction.optimize_acquisition_function",
            return_value=(optimized_x, torch.tensor(0.4, dtype=torch.double)),
        ) as mock_optimize:
            result = acquisition.calculate(dataset)

        mock_constraints.assert_called_once_with(3)
        assert mock_optimize.call_args.kwargs["inequality_constraints"] == [("simplex",)]
        np.testing.assert_allclose(result.output["next_sample"].values, [[0.34, 0.33, 0.33]])

    def test_botorch_acquisition_falls_back_to_best_grid_point_when_optimization_fails(self):
        dataset = _make_single_sample_botorch_dataset()
        fake_model = _FakeModel(
            posterior_mean=[[0.0], [0.0], [0.0]],
            posterior_variance=[[0.01], [0.01], [0.01]],
        )
        acq_values = torch.tensor([0.3, 0.2, 0.1], dtype=torch.double)

        acquisition = BoTorchAcquisition(
            feature_input_variable="composition",
            predictor_input_variable="score",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
            objective_direction="minimize",
            standardize=False,
            output_prefix="bayesopt",
            output_variable="next_sample",
            count=1,
            bounds=_make_explicit_bounds(),
        )

        with patch("AFL.double_agent.AcquisitionFunction.fit_single_task_gp", return_value=fake_model), patch(
            "AFL.double_agent.AcquisitionFunction.evaluate_log_expected_improvement",
            return_value=acq_values,
        ), patch(
            "AFL.double_agent.AcquisitionFunction.optimize_acquisition_function",
            side_effect=RuntimeError("probability tensor contains either `inf`, `nan` or element < 0"),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = acquisition.calculate(dataset)

        assert any("falling back to the best evaluated grid point" in str(item.message) for item in caught)
        np.testing.assert_allclose(result.output["bayesopt_decision_surface"].values, [0.3, 0.2, 0.1])
        np.testing.assert_allclose(result.output["next_sample"].values, [[0.70, 0.20, 0.10]])

    def test_botorch_acquisition_requires_explicit_bounds(self):
        dataset = _make_botorch_dataset()
        fake_model = _FakeModel(
            posterior_mean=[[0.0], [0.0], [0.0], [0.0]],
            posterior_variance=[[0.01], [0.01], [0.01], [0.01]],
        )
        acq_values = torch.tensor([0.1, 0.4, 0.2, 0.3], dtype=torch.double)

        acquisition = BoTorchAcquisition(
            feature_input_variable="composition",
            predictor_input_variable="score",
            grid_variable="composition_grid",
            grid_dim="grid",
            sample_dim="sample",
            objective_direction="minimize",
            standardize=False,
            output_prefix="bayesopt",
            output_variable="next_sample",
            count=1,
        )

        with patch("AFL.double_agent.AcquisitionFunction.fit_single_task_gp", return_value=fake_model), patch(
            "AFL.double_agent.AcquisitionFunction.evaluate_log_expected_improvement",
            return_value=acq_values,
        ):
            with pytest.raises(ValueError, match="requires explicit `bounds`"):
                acquisition.calculate(dataset)
