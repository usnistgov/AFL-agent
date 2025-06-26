"""PipelineOps for Pairwise Metrics

This module contains operations that compute pairwise relationships between samples.
PairMetrics generate matrices that capture similarity, distance, or other relationships between pairs of data points.

These metrics are useful for:
- Measuring similarity or distance between samples
- Constructing adjacency matrices for graph-based algorithms
- Identifying clusters or patterns in data
- Quantifying relationships between different observations

Each PairMetric is implemented as a PipelineOp that can be composed with others in a processing pipeline.
"""

import copy
from numbers import Number
from typing import Optional, List, Dict, Any

import numpy as np
import scipy.spatial  # type: ignore
import xarray as xr
from sklearn.metrics import pairwise  # type: ignore
from typing_extensions import Self

from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify


class PairMetric(PipelineOp):
    """Base class for all PairMetrics that produce similarity or distance matrices

    This abstract base class provides common functionality for computing and manipulating
    pairwise metrics between samples. It handles similarity constraints, normalization,
    and provides a framework for different metric implementations.

    Parameters
    ----------
    input_variable : str | List[str]
        The name of the data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the dataset
    sample_dim : str, default="sample"
        The dimension containing different samples
    params : Optional[Dict[str, Any]], default=None
        Additional parameters for metric calculation
    constrain_same : Optional[List], default=None
        List of pairs that should have perfect similarity
    constrain_different : Optional[List], default=None
        List of pairs that should have zero similarity
    name : str, default="PairMetric"
        The name to use when added to a Pipeline
    """
    def __init__(
        self,
        input_variable: str | List[str],
        output_variable: str,
        sample_dim: str = "sample",
        params: Optional[Dict[str, Any]] = None,
        constrain_same: Optional[List] = None,
        constrain_different: Optional[List] = None,
        name: str = "PairMetric",
    ) -> None:

        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )

        self.W = np.eye(1)
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

        self._banned_from_attrs.append("W")
        self._banned_from_attrs.append("params")
        self._banned_from_attrs.append("combine")

    def __getitem__(self, index):
        return self.W[index]

    def __array__(self, dtype=None):
        return self.W.astype(dtype)

    def normalize1(self) -> np.ndarray:
        """Normalize similarity matrix such that the diagonal values are all equal to 1"""
        W = self.W.copy()
        diag = np.diag(W)
        W = W / np.sqrt(np.outer(diag, diag))
        return W

    def normalize2(self):
        """Normalize similarity matrix such that all values are between 0 and 1"""
        W = self.W.copy()
        W = (W - W.min()) / (W.max() - W.min())
        return W

    def apply_constraints(self):
        """
        Constrain pairs in the similarity matrix to be perfectly similar (S[i,j]=1.0) or
        perfectly dissimilar (S[i,j]=0.0).
        """
        W = self.W.copy()
        for i, j in self.constrain_same:
            W[i, j] = 1.0
            W[j, i] = 1.0

        for i, j in self.constrain_different:
            W[i, j] = 0.0
            W[j, i] = 0.0
        return W

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.Dataset`"""
        return NotImplementedError(".calculate must be implemented in subclasses")  # type: ignore


class Dummy(PairMetric):
    """PairMetric that returns only self-similarity (identity matrix)
    
    This simple metric creates an identity matrix where diagonal elements (self-similarity)
    are 1.0 and all off-diagonal elements are 0.0. This can be useful as a baseline
    or for testing purposes.

    Parameters
    ----------
    input_variable : str
        The name of the data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the dataset
    sample_dim : str
        The dimension containing different samples
    params : Optional[Dict[str, Any]], default=None
        Additional parameters for metric calculation (not used in this class)
    constrain_same : Optional[List], default=None
        List of pairs that should have perfect similarity
    constrain_different : Optional[List], default=None
        List of pairs that should have zero similarity
    name : str, default="DummyMetric"
        The name to use when added to a Pipeline
    """
    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        sample_dim: str,
        params: Optional[Dict[str, Any]] = None,
        constrain_same: Optional[List] = None,
        constrain_different: Optional[List] = None,
        name: str = "DummyMetric",
    ) -> None:

        super().__init__(
            name=name,
            input_variable=input_variable,
            output_variable=output_variable,
            sample_dim=sample_dim,
            params=params,
            constrain_same=constrain_same,
            constrain_different=constrain_different,
        )

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.Dataset`"""
        n_samples = dataset.sizes[self.sample_dim]
        self.W = np.identity(n_samples)
        return self


class Similarity(PairMetric):
    """Computes pairwise similarity between samples using kernel functions
    
    This class uses scikit-learn's pairwise_kernels to compute similarity matrices
    between samples. Various kernel functions can be specified through the params
    dictionary (e.g., 'linear', 'rbf', 'polynomial').

    For details on available kernel functions and their parameters, see:
    https://scikit-learn.org/stable/modules/metrics.html#metrics

    Parameters
    ----------
    input_variable : str
        The name of the data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the dataset
    sample_dim : str
        The dimension containing different samples
    params : Optional[Dict[str, Any]], default=None
        Parameters for the kernel function, including 'metric' to specify the kernel type
    constrain_same : Optional[List], default=None
        List of pairs that should have perfect similarity
    constrain_different : Optional[List], default=None
        List of pairs that should have zero similarity
    name : str, default="SimilarityMetric"
        The name to use when added to a Pipeline
    """
    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        sample_dim: str,
        params: Optional[Dict[str, Any]] = None,
        constrain_same: Optional[List] = None,
        constrain_different: Optional[List] = None,
        name="SimilarityMetric",
    ) -> None:

        super().__init__(
            name=name,
            input_variable=input_variable,
            output_variable=output_variable,
            sample_dim=sample_dim,
            params=params,
            constrain_same=constrain_same,
            constrain_different=constrain_different,
        )

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.Dataset`"""

        data1 = self._get_variable(dataset)

        if len(data1.shape) == 1:
            data1 = data1.expand_dims("feature", axis=1)
        self.W = pairwise.pairwise_kernels(data1, filter_params=True, **self.params)

        dims = [self.sample_dim + "_i", self.sample_dim + "_j"]
        self.output[self.output_variable] = xr.DataArray(self.W, dims=dims)  # type: ignore
        self.output[self.output_variable].attrs.update(self.params)  # type: ignore

        return self


class Distance(PairMetric):
    """Computes pairwise distances between samples
    
    This class uses scikit-learn's pairwise_distances to compute distance matrices
    between samples. Various distance metrics can be specified through the params
    dictionary (e.g., 'euclidean', 'manhattan', 'cosine').

    For details on available distance metrics and their parameters, see:
    https://scikit-learn.org/stable/modules/metrics.html#metrics

    Parameters
    ----------
    input_variable : str
        The name of the data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the dataset
    sample_dim : str
        The dimension containing different samples
    params : Optional[Dict[str, Any]], default=None
        Parameters for the distance function, including 'metric' to specify the distance type
    constrain_same : Optional[List], default=None
        List of pairs that should have perfect similarity
    constrain_different : Optional[List], default=None
        List of pairs that should have zero similarity
    name : str, default="DistanceMetric"
        The name to use when added to a Pipeline
    """
    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        sample_dim: str,
        params: Optional[Dict[str, Any]] = None,
        constrain_same: Optional[List] = None,
        constrain_different: Optional[List] = None,
        name="DistanceMetric",
    ) -> None:

        super().__init__(
            name=name,
            input_variable=input_variable,
            output_variable=output_variable,
            sample_dim=sample_dim,
            params=params,
            constrain_same=constrain_same,
            constrain_different=constrain_different,
        )

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.Dataset`"""
        data1 = self._get_variable(dataset)

        self.W = pairwise.pairwise_distances(data1, **self.params)

        dims = [self.sample_dim + "_i", self.sample_dim + "_j"]
        self.output[self.output_variable] = xr.DataArray(self.W, dims=dims)  # type: ignore
        return self


class CombineMetric(PairMetric):
    """Combines multiple similarity/distance matrices into a single matrix
    
    This class allows for the combination of multiple similarity or distance matrices
    using either product or sum operations. Each matrix can be weighted differently
    using powers (for product) or coefficients (for sum).

    Parameters
    ----------
    input_variables : List[str]
        List of variable names containing similarity/distance matrices to combine
    output_variable : str
        The name of the variable to be inserted into the dataset
    sample_dim : str
        The dimension containing different samples
    combine_by : str
        Method to combine matrices, either "prod" (product) or "sum"
    combine_by_powers : Optional[List[Number]], default=None
        List of powers to apply to each matrix when using "prod" combination
    combine_by_coeffs : Optional[List[Number]], default=None
        List of coefficients to multiply each matrix by when using "sum" combination
    params : Optional[str], default=None
        Additional parameters
    constrain_same : Optional[List], default=None
        List of pairs that should have perfect similarity
    constrain_different : Optional[List], default=None
        List of pairs that should have zero similarity
    name : str, default="CombineMetric"
        The name to use when added to a Pipeline
    """
    def __init__(
        self,
        input_variables: List[str],
        output_variable: str,
        sample_dim: str,
        combine_by: str,
        combine_by_powers: Optional[List[Number]] = None,
        combine_by_coeffs: Optional[List[Number]] = None,
        params: Optional[str] = None,
        constrain_same: Optional[List] = None,
        constrain_different: Optional[List] = None,
        name="CombineMetric",
    ) -> None:

        super().__init__(
            name=name,
            input_variable=input_variables,
            output_variable=output_variable,
            sample_dim=sample_dim,
            params=params,
            constrain_same=constrain_same,
            constrain_different=constrain_different,
        )

        self.combine_by = combine_by
        self.combine_by_powers = combine_by_powers
        self.combine_by_coeffs = combine_by_coeffs
        if self.combine_by == "prod":
            self.combine = self.prod
        elif self.combine_by == "sum":
            self.combine = self.sum
        else:
            raise ValueError(
                'Combine by function not recognized. Must be "sum" or "prod"'
            )

    def prod(self, data_list: List) -> np.ndarray:
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

    def sum(self, data_list: List) -> np.ndarray:
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

    def calculate(self, dataset: xr.Dataset) -> Self:
        W_list = []
        for name in listify(self.input_variable):
            W_list.append(dataset[name])

        self.W = self.combine(W_list)
        self.W = self.normalize1()
        self.W = self.apply_constraints()

        dims = [self.sample_dim + "_i", self.sample_dim + "_j"]
        self.output[self.output_variable] = xr.DataArray(self.W, dims=dims)  # type: ignore

        return self
