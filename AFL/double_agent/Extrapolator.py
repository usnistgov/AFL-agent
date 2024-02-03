"""
PairMetrics are PipelineOps that produce pair matrices as results

"""
import xarray as xr
import numpy as np

from AFL.double_agent.PipelineOp import PipelineOp

import sklearn.gaussian_process
import sklearn.gaussian_process.kernels
import scipy.stats


class Extrapolator(PipelineOp):
    def __init__(self, feature_input_variable, predictor_input_variable, output_variable, grid_variable, grid_dim,
                 sample_dim, name='Extrapolator'):
        super().__init__(name=name, input_variable=[feature_input_variable, predictor_input_variable, grid_variable],
                         output_variable=output_variable)
        self.feature_input_variable = feature_input_variable
        self.predictor_input_variable = predictor_input_variable
        self.grid_variable = grid_variable
        self.sample_dim = sample_dim
        self.grid_dim = grid_dim


class DummyExtrapolator(Extrapolator):
    def __init__(self, feature_input_variable, predictor_input_variable, output_variable, grid_variable, grid_dim,
                 sample_dim, name='DummyExtrapolator'):
        super().__init__(name=name, feature_input_variable=feature_input_variable,
                         predictor_input_variable=predictor_input_variable, output_variable=output_variable,
                         grid_variable=grid_variable, grid_dim=grid_dim, sample_dim=sample_dim)

    def calculate(self, dataset):
        grid = dataset[self.grid_variable].transpose(self.sample_dim,...)
        dummy = xr.DataArray(np.zeros_like(grid), dims=grid.dims)
        self.output[self.output_variable + "_mean"] = dummy.copy()
        self.output[self.output_variable + "_var"] = dummy.copy()
        return self


class GaussianProcessClassifier(Extrapolator):
    def __init__(self, feature_input_variable, predictor_input_variable, output_variable, grid_variable, grid_dim,
                 sample_dim, kernel=None, name='GaussianProcessClassifier'):

        super().__init__(name=name, feature_input_variable=feature_input_variable,
                         predictor_input_variable=predictor_input_variable, output_variable=output_variable,
                         grid_variable=grid_variable, grid_dim=grid_dim, sample_dim=sample_dim)

        if kernel is None:
            self.kernel = sklearn.gaussian_process.kernels.Matern(length_scale=1.0, nu=1.5)
        else:
            self.kernel = kernel

    def calculate(self, dataset):
        X = dataset[self.feature_input_variable].transpose(self.sample_dim, ...)
        y = dataset[self.predictor_input_variable].transpose(self.sample_dim, ...)
        grid = dataset[self.grid_variable]

        clf = sklearn.gaussian_process.GaussianProcessClassifier(kernel=self.kernel).fit(X.values, y.values)

        mean = clf.predict_proba(grid.values)
        entropy = -np.sum(np.log(mean)*mean,axis=-1)

        self.output[self.output_variable + "_mean"] = xr.DataArray(mean.argmax(-1),dims=self.grid_dim)
        self.output[self.output_variable + "_entropy"] = xr.DataArray(entropy,dims=self.grid_dim)

        return self

