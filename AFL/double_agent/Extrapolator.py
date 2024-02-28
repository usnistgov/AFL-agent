"""
Extrapolators take discrete sample data and extrapolate the data onto a provided grid.
"""

from typing import List, Optional

import numpy as np
import sklearn.gaussian_process  # type: ignore
import sklearn.gaussian_process.kernels  # type: ignore
import xarray as xr
from typing_extensions import Self

from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify


class Extrapolator(PipelineOp):
    """Base class for all extrapolators """
    def __init__(
        self,
        feature_input_variable: str,
        predictor_input_variable: str,
        output_variables: List[str],
        output_prefix: str,
        grid_variable: str,
        grid_dim: str,
        sample_dim: str,
        name: str = "Extrapolator",
    ) -> None:
        """
        Parameters
        ----------
        feature_input_variable : str
            The name of the `xarray.Dataset` data variable to use as the input to the model that will be extrapolating
            the discrete data. This is typically a sample composition variable.

        predictor_input_variable : str
            The name of the `xarray.Dataset` data variable to use as the output of the model that will be extrapolating
            the discrete data. This is typically a class label or property variable.

        output_variables: List[str]
            The list of variables that will be output by this class.

        output_prefix: str
            The string prefix to apply to each output variable before inserting into the output `xarray.Dataset`

        grid_variable: str
            The name of the `xarray.Dataset` data variable to use as an evaluation grid.

        grid_dim: str
            The xarray dimension over each grid_point. Grid equivalent to sample.

        output_prefix: Optional[str]
            If provided, all outputs of this `PipelineOp` will be prefixed with this string

        sample_dim: str
            The `xarray` dimension over the discrete 'samples' in the `feature_input_variable`. This is typically
            a variant of `sample` e.g., `saxs_sample`.

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
        """

        super().__init__(
            name=name,
            input_variable=[
                feature_input_variable,
                predictor_input_variable,
                grid_variable,
            ],
            output_variable=[
                output_prefix + "_" + o for o in listify(output_variables)
            ],
            output_prefix=output_prefix,
        )
        self.feature_input_variable = feature_input_variable
        self.predictor_input_variable = predictor_input_variable
        self.grid_variable = grid_variable
        self.sample_dim = sample_dim
        self.grid_dim = grid_dim

        self._banned_from_attrs.extend(["kernel"])

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.Dataset`"""
        return NotImplementedError(".calculate must be implemented in subclasses")  # type: ignore


class DummyExtrapolator(Extrapolator):
    def __init__(
        self,
        feature_input_variable: str,
        predictor_input_variable: str,
        output_prefix: str,
        grid_variable: str,
        grid_dim: str,
        sample_dim: str,
        name="DummyExtrapolator",
    ) -> None:
        """
        Parameters
        ----------
        feature_input_variable : str
            The name of the `xarray.Dataset` data variable to use as the input to the model that will be extrapolating
            the discrete data. This is typically a sample composition variable.

        predictor_input_variable : str
            The name of the `xarray.Dataset` data variable to use as the output of the model that will be extrapolating
            the discrete data. This is typically a class label or property variable.

        output_prefix: str
            The string prefix to apply to each output variable before inserting into the output `xarray.Dataset`

        grid_variable: str
            The name of the `xarray.Dataset` data variable to use as an evaluation grid.

        grid_dim: str
            The xarray dimension over each grid_point. Grid equivalent to sample.

        sample_dim: str
            The `xarray` dimension over the discrete 'samples' in the `feature_input_variable`. This is typically
            a variant of `sample` e.g., `saxs_sample`.

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
        """
        super().__init__(
            name=name,
            feature_input_variable=feature_input_variable,
            predictor_input_variable=predictor_input_variable,
            output_variables=["mean", "var"],
            output_prefix=output_prefix,
            grid_variable=grid_variable,
            grid_dim=grid_dim,
            sample_dim=sample_dim,
        )

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.Dataset`"""
        grid = dataset[self.grid_variable].transpose(self.sample_dim, ...)
        dummy = xr.DataArray(np.zeros_like(grid), dims=grid.dims)
        self.output[self._prefix_output("mean")] = dummy.copy()
        self.output[self._prefix_output("var")] = dummy.copy()
        return self


class GaussianProcessClassifier(Extrapolator):
    """Use a Gaussian process classifier to extrapolate class labels at discrete compositions onto a composition grid"""
    def __init__(
        self,
        feature_input_variable: str,
        predictor_input_variable: str,
        output_prefix: str,
        grid_variable: str,
        grid_dim: str,
        sample_dim: str,
        kernel: Optional[object] = None,
        optimizer: str = "fmin_l_bfgs_b",
        name: str = "GaussianProcessClassifier",
    ) -> None:
        """
        Parameters
        ----------
        feature_input_variable : str
            The name of the `xarray.Dataset` data variable to use as the input to the model that will be extrapolating
            the discrete data. This is typically a sample composition variable.

        predictor_input_variable : str
            The name of the `xarray.Dataset` data variable to use as the output of the model that will be extrapolating
            the discrete data. For this `PipelineOp` this should be a class label vector.

        output_prefix: str
            The string prefix to apply to each output variable before inserting into the output `xarray.Dataset`

        grid_variable: str
            The name of the `xarray.Dataset` data variable to use as an evaluation grid.

        grid_dim: str
            The xarray dimension over each grid_point. Grid equivalent to sample.

        sample_dim: str
            The `xarray` dimension over the discrete 'samples' in the `feature_input_variable`. This is typically
            a variant of `sample` e.g., `saxs_sample`.

        kernel: Optional[object]
            A optional sklearn.gaussian_process.kernel to use the classifier. If not provided, will default to
            `Matern`.

        optimizer: str
            The name of the optimizer to use in optimizer the gaussian process parameters

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
        """

        super().__init__(
            name=name,
            feature_input_variable=feature_input_variable,
            predictor_input_variable=predictor_input_variable,
            output_variables=["mean", "entropy"],
            output_prefix=output_prefix,
            grid_variable=grid_variable,
            grid_dim=grid_dim,
            sample_dim=sample_dim,
        )

        if kernel is None:
            self.kernel = sklearn.gaussian_process.kernels.Matern(
                length_scale=1.0, nu=1.5
            )
        else:
            self.kernel = kernel

        self.output_prefix = output_prefix
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = None

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.Dataset`"""
        X = dataset[self.feature_input_variable].transpose(self.sample_dim, ...)
        y = dataset[self.predictor_input_variable].transpose(self.sample_dim, ...)
        grid = dataset[self.grid_variable]

        if len(np.unique(y)) == 1:

            self.output[self._prefix_output("mean")] = xr.DataArray(
                np.ones(dataset.grid.shape), dims=[self.grid_dim]
            )
            self.output[self._prefix_output("entropy")] = xr.DataArray(
                np.ones(dataset.grid.shape), dims=[self.grid_dim]
            )
            self.output[self._prefix_output("y_prob")] = xr.DataArray(
                np.ones(dataset.grid.shape), dims=[self.grid_dim]
            )

        else:
            clf = sklearn.gaussian_process.GaussianProcessClassifier(
                kernel=self.kernel, optimizer=self.optimizer
            ).fit(X.values, y.values)

            # entropy approach to classification probabilities
            mean = clf.predict_proba(grid.values)
            entropy = -np.sum(np.log(mean) * mean, axis=-1)

            # A messier way to achieve a similar thing
            y_prob = 1 - (mean - (1 / len(np.unique(y))))

            self.output[self._prefix_output("mean")] = xr.DataArray(
                mean.argmax(-1), dims=self.grid_dim
            )
            self.output[self._prefix_output("entropy")] = xr.DataArray(
                entropy, dims=self.grid_dim
            )
            self.output[self._prefix_output("y_prob")] = xr.DataArray(
                entropy, dims=self.grid_dim
            )

        return self


class GaussianProcessRegressor(Extrapolator):
    """Use a Gaussian process regressor to extrapolate a property at discrete points onto a provided composition grid"""
    def __init__(
        self,
        feature_input_variable,
        predictor_input_variable,
        output_prefix,
        grid_variable,
        grid_dim,
        sample_dim,
        predictor_uncertainty_variable=None,
        optimizer="fmin_l_bfgs_b",
        kernel=None,
        name="GaussianProcessRegressor",
        fix_nans=True,
    ) -> None:
        """
        Parameters
        ----------
        feature_input_variable : str
            The name of the `xarray.Dataset` data variable to use as the input to the model that will be extrapolating
            the discrete data. This is typically a sample composition variable.

        predictor_input_variable : str
            The name of the `xarray.Dataset` data variable to use as the output of the model that will be extrapolating
            the discrete data. For this `PipelineOp` this should be a class label vector.

        output_prefix: str
            The string prefix to apply to each output variable before inserting into the output `xarray.Dataset`

        grid_variable: str
            The name of the `xarray.Dataset` data variable to use as an evaluation grid.

        grid_dim: str
            The xarray dimension over each grid_point. Grid equivalent to sample.

        sample_dim: str
            The `xarray` dimension over the discrete 'samples' in the `feature_input_variable`. This is typically
            a variant of `sample` e.g., `saxs_sample`.

        kernel: Optional[object]
            A optional sklearn.gaussian_process.kernel to use the classifier. If not provided, will default to
            `Matern`.

        optimizer: str
            The name of the optimizer to use in optimizer the gaussian process parameters

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
        """

        super().__init__(
            name=name,
            feature_input_variable=feature_input_variable,
            predictor_input_variable=predictor_input_variable,
            output_variables=["mean", "var"],
            output_prefix=output_prefix,
            grid_variable=grid_variable,
            grid_dim=grid_dim,
            sample_dim=sample_dim,
        )
        self.predictor_uncertainty_variable = predictor_uncertainty_variable
        if predictor_uncertainty_variable is not None:
            self.input_variable.append(predictor_uncertainty_variable)

        if kernel is None:
            self.kernel = sklearn.gaussian_process.kernels.Matern(
                length_scale=[0.1], length_scale_bounds=(1e-3, 1e0), nu=1.5
            )
        else:
            self.kernel = kernel

        if optimizer is not None:
            self.optimizer = "fmin_l_bfgs_b"
        else:
            self.optimizer = None

        self.predictor_uncertainty_variable = predictor_uncertainty_variable
        self._banned_from_attrs.append("predictor_uncertainty_variable")

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.Dataset`"""
        X = (
            dataset[self.feature_input_variable]
            .transpose(self.sample_dim, ...)
            .dropna(dim=self.sample_dim)
        )
        y = (
            dataset[self.predictor_input_variable]
            .transpose(self.sample_dim, ...)
            .dropna(dim=self.sample_dim)
        )

        grid = dataset[self.grid_variable]

        if self.predictor_uncertainty_variable is not None:
            dy = dataset[self.predictor_uncertainty_variable].transpose(
                self.sample_dim, ...
            )
            reg = sklearn.gaussian_process.GaussianProcessRegressor(
                kernel=self.kernel, alpha=dy.values, optimizer=self.optimizer
            ).fit(X.values, y.values)

            reg_type = "heteroscedastic"

        else:
            reg = sklearn.gaussian_process.GaussianProcessRegressor(
                kernel=self.kernel, optimizer=self.optimizer
            ).fit(X.values, y.values)
            reg_type = "homoscedastic"

        mean, std = reg.predict(grid.values, return_std=True)

        self.output[self._prefix_output("mean")] = xr.DataArray(
            mean, dims=self.grid_dim
        )
        self.output[self._prefix_output("var")] = xr.DataArray(std, dims=self.grid_dim)
        self.output[self._prefix_output("var")].attrs["heteroscedastic"] = reg_type

        return self
