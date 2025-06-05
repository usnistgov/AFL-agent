import numpy as np
import xarray as xr
from typing_extensions import Self
from typing import Optional, Dict, Any
import gpytorch
import torch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from AFL.double_agent.Extrapolator import Extrapolator

class DirichletGPExtrapolator(Extrapolator):
    """Gaussian Process classifier for extrapolating class labels.
    
    This extrapolator uses  Dirichlet-based Gaussian Processes to transform classification
    labels into continuous probabilities.

    We use the implementation from GPyTorch from here : 
    https://docs.gpytorch.ai/en/v1.13/examples/01_Exact_GPs/GP_Regression_on_Classification_Labels.html

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

    params: Dict
        Parameters to git a GP model using GPyTorch.
        
        learning_rate : Learning rate for torch.optim.Adam (float, 0.1)
        n_iterations : Total number of epochs to train the GP (int, 200) 
        verbose : Whether to print training loss stastistics (bool, False)   

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
        params: Optional[Dict[str, Any]] = None,
        name: str = "DirichletGPExtrapolator",
    ) -> None:

        super().__init__(
            name=name,
            feature_input_variable=feature_input_variable,
            predictor_input_variable=predictor_input_variable,
            output_variables=["y_prob"],
            output_prefix=output_prefix,
            grid_variable=grid_variable,
            grid_dim=grid_dim,
            sample_dim=sample_dim,
        )
        self.output_prefix = output_prefix
        self.params = params

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this GP classifier to the supplied dataset.
        
        Fits a Gaussian Process classifier to the input data and makes predictions
        across the grid, including class probabilities.

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
        self.grid = dataset[self.grid_variable]

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
            train_x = torch.from_numpy(X)
            train_y = torch.from_numpy(y).long().squeeze()
            likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True)
            model = GPModel(y, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)
            model, likelihood = self.fit(train_x, model, likelihood, self.params)
            
            model.eval()
            likelihood.eval()

            with gpytorch.settings.fast_pred_var(), torch.no_grad():
                test_dist = model(self.grid)
                pred_means = test_dist.loc

                pred_samples = test_dist.sample(torch.Size((256,))).exp()
                probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)

                self.output[self._prefix_output("mean")] = xr.DataArray(
                    pred_means.to_numpy().argmax(-1), dims=self.grid_dim
                )
                self.output[self._prefix_output("y_prob")] = xr.DataArray(
                    probabilities.to_numpy(), dims=self.grid_dim
                )

        return self 

    def fit(self, train_x, model, likelihood, **kwargs):
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.get("learning_rate" , 0.1)) 
        mll = ExactMarginalLogLikelihood(likelihood, model)
        n_iterations = kwargs.get("n_iterations" , 200)
        for i in range(n_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, likelihood.transformed_targets).sum()
            loss.backward()
            optimizer.step()
            if kwargs.get("verbose", False):
                if (i % 50 == 0) or (i==n_iterations-1):
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                        i + 1, n_iterations, loss.item(),
                        model.covar_module.base_kernel.lengthscale.mean().item(),
                        model.likelihood.second_noise_covar.noise.mean().item()
                    )
                )

        return model, likelihood

class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


