"""
PairMetrics are PipelineOps that produce pair matrices as results
"""
import copy

import numpy as np
import xarray as xr
import scipy.spatial
from sklearn.metrics import pairwise

from AFL.double_agent.PipelineOp import PipelineOp


class PairMetric(PipelineOp):
    def __init__(self, input_variable, output_variable, sample_dim='sample', params=None, name='PairMetric',
                 constrain_same=None, constrain_different=None):

        super().__init__(name=name,input_variable=input_variable, output_variable=output_variable)

        self.W = None
        self.sample_dim = sample_dim

        if params is None:
            self.params = {}
        else:
            self.params = params

        if constrain_same is None:
            self.constrain_same = []
        else:
            self.constrain_same = constrain_same

        if constrain_different is None:
            self.constrain_different = []
        else:
            self.constrain_different = constrain_different

        self._banned_from_attrs.append('W')

    def __getitem__(self, index):
        return self.W[index]

    def __array__(self, dtype=None):
        return self.W.astype(dtype)

    def normalize1(self):
        W = self.W.copy()
        diag = np.diag(W)
        W = W / np.sqrt(np.outer(diag, diag))
        return W

    def normalize2(self):
        W = self.W.copy()
        diag = np.diag(W)
        W = (W - W.min()) / (W.max() - W.min())
        return W

    def apply_constraints(self):
        W = self.W.copy()
        for i, j in self.constrain_same:
            W[i, j] = 1.0
            W[j, i] = 1.0

        for i, j in self.constrain_different:
            W[i, j] = 0.0
            W[j, i] = 0.0
        return W

class Dummy(PairMetric):
    def __init__(self, input_variable, output_variable, sample_dim, params=None, name='DummyMetric', constrain_same=None,
                 constrain_different=None):

        super().__init__(name=name,input_variable=input_variable, output_variable=output_variable, sample_dim=sample_dim,
                         params=params, constrain_same=constrain_same, constrain_different=constrain_different)

    def calculate(self, dataset):
        n_samples = dataset.sizes[self.sample_dim]
        self.W = np.identity(n_samples)
        return self


class Similarity(PairMetric):
    def __init__(self, input_variable, output_variable, sample_dim, params=None, name='SimilarityMetric', constrain_same=None,
                 constrain_different=None):

        super().__init__(name=name,input_variable=input_variable, output_variable=output_variable, sample_dim=sample_dim,
                         params=params, constrain_same=constrain_same, constrain_different=constrain_different)

    def calculate(self, dataset):

        data1 = self._get_variable(dataset)

        if len(data1.shape) == 1:
            data1 = data1.expand_dims("feature", axis=1)
        self.W = pairwise.pairwise_kernels(
            data1,
            filter_params=True,
            **self.params
        )

        dims = dims=[self.sample_dim+'_i',self.sample_dim+'_j']
        self.output[self.output_variable] = xr.DataArray(self.W,dims=dims)

        return self


class Distance(PairMetric):
    def __init__(self, input_variable, output_variable, sample_dim, params=None, name='DistanceMetric', constrain_same=None,
                 constrain_different=None):

        super().__init__(name=name,input_variable=input_variable, output_variable=output_variable, sample_dim=sample_dim,
                         params=params, constrain_same=constrain_same, constrain_different=constrain_different)

    def calculate(self, dataset):
        data1 = self._get_variable(dataset)

        self.W = pairwise.pairwise_distances(
            data1,
            **self.params
        )

        dims = dims=[self.sample_dim+'_i',self.sample_dim+'_j']
        self.output[self.output_variable] = xr.DataArray(self.W,dims=dims)
        return self

class Delaunay(PairMetric):
    def __init__(self, input_variable, output_variable, sample_dim, params=None, name='DelaunayMetric',
                 constrain_same=None, constrain_different=None):

        super().__init__(name=name,input_variable=input_variable, output_variable=output_variable, sample_dim=sample_dim,
                         params=params, constrain_same=constrain_same, constrain_different=constrain_different)

    def calculate(self, dataset):
        """
        Computes the Delaunay triangulation of the given points
        :param x: array of shape (num_nodes, 2)
        :return: the computed adjacency matrix
        """
        data1 = self._get_variable(dataset)
        tri = scipy.spatial.Delaunay(data1)
        edges_explicit = np.concatenate((tri.vertices[:, :2],
                                         tri.vertices[:, 1:],
                                         tri.vertices[:, ::2]), axis=0)
        adj = np.zeros((x.shape[0], x.shape[0]))
        adj[edges_explicit[:, 0], edges_explicit[:, 1]] = 1.
        self.W = np.clip(adj + adj.T, 0, 1)

        return self


class CombineMetric(PairMetric):
    def __init__(self, input_variable_list, output_variable, sample_dim, combine_by, combine_by_powers=None,
                 combine_by_coeffs=None, params=None, name='CombineMetric', constrain_same=None, constrain_different=None):

        super().__init__(name=name, input_variable=input_variable_list, output_variable=output_variable,
                         sample_dim=sample_dim, params=params, constrain_same=constrain_same,
                         constrain_different=constrain_different)

        self.combine_by = combine_by
        self.combine_by_powers = combine_by_powers
        self.combine_by_coeffs = combine_by_coeffs
        if self.combine_by == 'prod':
            self.combine = self.prod
        elif self.combine_by == 'sum':
            self.combine = self.sum
        else:
            raise ValueError('Combine by function not recognized. Must be "sum" or "prod"')

    def prod(self, data_list):
        if self.combine_by_powers is not None:
            assert len(self.combine_by_powers) == len(data_list)
            powers = copy.deepcopy(self.combine_by_powers)
        else:
            powers = [1] * len(data_list)

        data_list = copy.deepcopy(data_list)

        # np methods use the __array__ accessor and return W
        value = np.power(data_list.pop(0), powers.pop(0))
        for data, power in zip(data_list, powers):
            value *= np.power(data, power)
        return value

    def sum(self, data_list):
        if self.combine_by_powers is not None:
            assert len(self.combine_by_coeffs) == len(data_list)
            coeffs = copy.deepcopy(self.combine_by_coeffs)
        else:
            coeffs = [1] * len(data_list)

        data_list = copy.deepcopy(data_list)

        # np methods use the __array__ accessor and return W
        value = np.multiply(data_list.pop(0), coeffs.pop(0))
        for data, coeff in zip(data_list, coeffs):
            value += np.multiply(data, coeff)

        return value

    def calculate(self, dataset):
        W_list = []
        for name in self.input_variable:
            W_list.append(dataset[name])

        self.W = self.combine(W_list)
        self.W = self.normalize1()
        self.W = self.apply_constraints()

        dims = dims=[self.sample_dim+'_i',self.sample_dim+'_j']
        self.output[self.output_variable] = xr.DataArray(self.W,dims=dims)

        return self


