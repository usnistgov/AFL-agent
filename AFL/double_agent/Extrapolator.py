"""
Extrapolation tools for extending discrete sample data to continuous spaces.

This module provides classes for extrapolating data from discrete sample points to continuous
spaces, particularly useful in materials science and machine learning applications. The extrapolators
can work with both classification and regression tasks.

Key features:
- Support for Gaussian Process Classification and Regression
- Handling of uncertainty in measurements
- Visualization tools for extrapolation results
- Flexible kernel selection for GP models
- Support for different sample and grid dimensions
"""

from typing import List

import numpy as np
import sklearn.gaussian_process  # type: ignore
import sklearn.gaussian_process.kernels  # type: ignore
import xarray as xr
from typing_extensions import Self
import matplotlib.pyplot as plt

from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify


class Extrapolator(PipelineOp):
    """Base class for extrapolating discrete sample data onto continuous spaces.

    This abstract base class provides common functionality for extrapolating data from
    discrete sample points to a continuous grid. It handles data management and provides
    visualization capabilities.

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

    sample_dim: str
        The `xarray` dimension over the discrete 'samples' in the `feature_input_variable`. This is typically
        a variant of `sample` e.g., `saxs_sample`.

    name: str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
    """
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
        self.grid = None# store grid for plotting

        self._banned_from_attrs.extend(["kernel"])

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this extrapolator to the supplied dataset.
        
        This method must be implemented by subclasses to define how the extrapolation
        is performed.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing the sample points and grid

        Returns
        -------
        Self
            The extrapolator instance with updated outputs
        """
        return NotImplementedError(".calculate must be implemented in subclasses")  # type: ignore

    def plot(self,**mpl_kwargs) -> plt.Figure:
        """Plot the extrapolation results.
        
        Creates visualization of the extrapolated data, with different plotting styles
        depending on the data dimensions and type.

        Parameters
        ----------
        **mpl_kwargs : dict
            Additional keyword arguments to pass to matplotlib plotting functions

        Returns
        -------
        plt.Figure
            The matplotlib figure containing the plots
        """
        n = len(self.output)
        if n>0:
            fig, axes = plt.subplots(n,1,figsize=(6,n*4))
            if n>1:
                axes = list(axes.flatten())
            else:
                axes = [axes]

            for i,(name,data) in enumerate(self.output.items()):
                if 'grid' in data.dims:
                    non_grid_dim = [d for d in self.grid.dims if d != self.grid_dim][0]
                    x = self.grid.isel({non_grid_dim:0})
                    y = self.grid.isel({non_grid_dim:1})
                    c = data.values
                    axes[i].scatter(x=x,y=y,c=list(c),**mpl_kwargs)

                elif 'sample' in data.dims:
                    data.plot(hue='sample',ax=axes[i],**mpl_kwargs)
                else:
                    data.plot(ax=axes[i],**mpl_kwargs)
                axes[i].set(title=name)
            return fig
        else:
            return plt.figure()


class DummyExtrapolator(Extrapolator):
    """Simple extrapolator that returns zero values.
    
    This extrapolator serves as a baseline implementation, returning arrays of zeros
    for both mean and variance predictions. Useful for testing and as a template.

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
        """Apply this dummy extrapolator to the supplied dataset.
        
        Creates arrays of zeros for both mean and variance predictions.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing the sample points and grid

        Returns
        -------
        Self
            The dummy extrapolator instance with zero-valued outputs
        """
        grid = dataset[self.grid_variable].transpose(self.sample_dim, ...)
        dummy = xr.DataArray(np.zeros_like(grid), dims=grid.dims)
        self.output[self._prefix_output("mean")] = dummy.copy()
        self.output[self._prefix_output("var")] = dummy.copy()
        return self


class GaussianProcessClassifier(Extrapolator):
    """Gaussian Process classifier for extrapolating class labels.
    
    This extrapolator uses scikit-learn's GaussianProcessClassifier to predict class
    probabilities across the grid based on discrete labeled samples. It provides both
    class predictions and uncertainty estimates through entropy.

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

    kernel: str
        The name of the sklearn.gaussian_process.kernel to use the classifier. If not provided, will default to
        `Matern`.
    
    kernel_kwargs: dict
        Additional keyword arguments to pass to the sklearn.gaussian_process.kernel

    optimizer: str
        The name of the optimizer to use in optimizer the gaussian process parameters

    name: str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
    """

    def __init__(
        self,
        feature_input_variable: str,
        predictor_input_variable: str,
        output_prefix: str,
        grid_variable: str,
        grid_dim: str,
        sample_dim: str,
        kernel: str = 'Matern',
        kernel_kwargs: dict = {'length_scale':1.0, 'nu':1.5},
        optimizer: str = "fmin_l_bfgs_b",
        name: str = "GaussianProcessClassifier",
    ) -> None:

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
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs
        self.output_prefix = output_prefix
        self.optimizer = optimizer 


    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this GP classifier to the supplied dataset.
        
        Fits a Gaussian Process classifier to the input data and makes predictions
        across the grid, including class probabilities and entropy-based uncertainty.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing labeled samples and prediction grid

        Returns
        -------
        Self
            The GP classifier instance with predictions and uncertainty estimates
        """
        X = dataset[self.feature_input_variable].transpose(self.sample_dim, ...)
        y = dataset[self.predictor_input_variable].transpose(self.sample_dim, ...)
        self.grid = dataset[self.grid_variable]# store grid for plotting

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
            kernel  = getattr(sklearn.gaussian_process.kernels, self.kernel)(**self.kernel_kwargs)

            clf = sklearn.gaussian_process.GaussianProcessClassifier(
                kernel=kernel, optimizer=self.optimizer
            ).fit(X.values, y.values)

            # entropy approach to classification probabilities
            mean = clf.predict_proba(self.grid.values)
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
    """Gaussian Process regressor for extrapolating continuous values.
    
    This extrapolator uses scikit-learn's GaussianProcessRegressor to predict continuous
    values across the grid based on discrete samples. It handles measurement uncertainty
    and provides both mean predictions and variance estimates.

    Parameters
    ----------
    feature_input_variable : str
        The name of the `xarray.Dataset` data variable to use as the input to the model that will be extrapolating
        the discrete data. This is typically a sample composition variable.

    predictor_input_variable : str
        The name of the `xarray.Dataset` data variable to use as the output of the model that will be extrapolating
        the discrete data. For this `PipelineOp` this should be a continuous value vector.

    output_prefix: str
        The string prefix to apply to each output variable before inserting into the output `xarray.Dataset`

    grid_variable: str
        The name of the `xarray.Dataset` data variable to use as an evaluation grid.

    grid_dim: str
        The xarray dimension over each grid_point. Grid equivalent to sample.

    sample_dim: str
        The `xarray` dimension over the discrete 'samples' in the `feature_input_variable`. This is typically
        a variant of `sample` e.g., `saxs_sample`.

    predictor_uncertainty_variable: str | None
        Variable containing uncertainty estimates for the predictor values

    optimizer: str
        The name of the optimizer to use in optimizer the gaussian process parameters

    kernel: str | None
        The name of the sklearn.gaussian_process.kernel to use the regressor. If not provided, will default to
        `Matern`.

    name: str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()

    fix_nans: bool
        Whether to handle NaN values in the input data
    """

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
        kernel: str = 'Matern',
        kernel_kwargs: dict = {'length_scale':1.0, 'nu':1.5},
        name="GaussianProcessRegressor",
        fix_nans=True,
    ) -> None:

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
        
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs
        self.optimizer = optimizer 

        self.predictor_uncertainty_variable = predictor_uncertainty_variable
        self._banned_from_attrs.append("predictor_uncertainty_variable")

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this GP regressor to the supplied dataset.
        
        Fits a Gaussian Process regressor to the input data and makes predictions
        across the grid, including mean values and variance estimates. Can handle
        heteroscedastic noise if uncertainty values are provided.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing samples and prediction grid

        Returns
        -------
        Self
            The GP regressor instance with predictions and uncertainty estimates
        """
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

        self.grid = dataset[self.grid_variable]

        if self.predictor_uncertainty_variable is not None:
            dy = dataset[self.predictor_uncertainty_variable].transpose(
                self.sample_dim, ...
            )
            kernel  = getattr(sklearn.gaussian_process.kernels, self.kernel)(**self.kernel_kwargs)
            reg = sklearn.gaussian_process.GaussianProcessRegressor(
                kernel=kernel, alpha=dy.values, optimizer=self.optimizer
            ).fit(X.values, y.values)

            reg_type = "heteroscedastic"

        else:
            kernel  = getattr(sklearn.gaussian_process.kernels, self.kernel)(**self.kernel_kwargs)
            reg = sklearn.gaussian_process.GaussianProcessRegressor(
                kernel=kernel, optimizer=self.optimizer
            ).fit(X.values, y.values)
            reg_type = "homoscedastic"

        mean, std = reg.predict(self.grid.values, return_std=True)
        var = std*std
        self.reg = reg

        self.output[self._prefix_output("mean")] = xr.DataArray( mean, dims=self.grid_dim )
        self.output[self._prefix_output("mean")].attrs["reg_type"] = reg_type
        self.output[self._prefix_output("mean")].attrs.update(reg.kernel_.get_params())

        self.output[self._prefix_output("var")] = xr.DataArray(var, dims=self.grid_dim)
        self.output[self._prefix_output("var")].attrs["reg_type"] = reg_type
        self.output[self._prefix_output("var")].attrs.update(reg.kernel_.get_params())

        return self
