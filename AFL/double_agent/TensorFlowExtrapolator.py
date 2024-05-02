"""
Extrapolators take discrete sample data and extrapolate the data onto a provided grid.

This file segments all extapolators that require tensorflow.
"""

from typing import List, Optional
from typing_extensions import Self

import numpy as np
import xarray as xr
import tqdm  # type: ignore

from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify

import tensorflow as tf  # type: ignore
import gpflow


class TFExtrapolator(PipelineOp):
    """Base class for all tensorflow based extrapolators"""

    def __init__(
        self,
        feature_input_variable: str,
        predictor_input_variable: str,
        output_variables: List[str],
        output_prefix: str,
        grid_variable: str,
        grid_dim: str,
        sample_dim: str,
        optimize: bool,
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
        self.optimize = True

        self._banned_from_attrs.extend(["kernel", "opt_logs"])

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.Dataset`"""
        return NotImplementedError(".calculate must be implemented in subclasses")  # type: ignore


class TFGaussianProcessClassifier(TFExtrapolator):
    """Use a Gaussian process classifier to extrapolate class labels at discrete compositions onto a composition grid"""

    def __init__(
        self,
        feature_input_variable: str,
        predictor_input_variable: str,
        output_prefix: str,
        grid_variable: str,
        grid_dim: str,
        sample_dim: str,
        optimize: bool = True,
        kernel: Optional[gpflow.kernels.Kernel] = None,
        name: str = "TFGaussianProcessClassifier",
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

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
        """

        super().__init__(
            name=name,
            feature_input_variable=feature_input_variable,
            predictor_input_variable=predictor_input_variable,
            output_variables=["mean", "variance"],
            output_prefix=output_prefix,
            grid_variable=grid_variable,
            grid_dim=grid_dim,
            sample_dim=sample_dim,
            optimize = optimize,
        )

        if kernel is None:
            self.kernel: gpflow.kernels.Kernel = gpflow.kernels.Matern32(
                variance=0.1, lengthscales=0.1
            )
        else:
            self.kernel = kernel

        self.output_prefix = output_prefix

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

        else:
            n_classes: int = len(np.unique(y.values))
            data = (X, y)

            invlink = gpflow.likelihoods.RobustMax(n_classes)
            likelihood = gpflow.likelihoods.MultiClass(n_classes, invlink=invlink)
            model = gpflow.models.VGP(
                data=data,
                kernel=self.kernel,
                likelihood=likelihood,
                num_latent_gps=n_classes,
            )
            display(model)

            if self.optimize:
                print('Training!')
                opt = gpflow.optimizers.Scipy()
                self.opt_logs = opt.minimize(
                    model.training_loss_closure(),
                    model.trainable_variables,
                    options=dict(maxiter=1000),
                )
                
            display(model)

            mean, variance = model.predict_y(grid.values)

            param_dict = {k:v.numpy() for k,v in gpflow.utilities.parameter_dict(model).items()}
            self.output[self._prefix_output("mean")] = xr.DataArray(
                mean.numpy().argmax(-1), dims=self.grid_dim
            )
            self.output[self._prefix_output("mean")].attrs.update(param_dict)
            self.output[self._prefix_output("variance")] = xr.DataArray(
                variance.numpy().sum(-1), dims=self.grid_dim
            )
            self.output[self._prefix_output("variance")].attrs.update(param_dict)

        return self
