"""
Graph operations and utilities for network analysis.

This module provides graph-based operations for analyzing relationships between
data points, including Delaunay triangulation for constructing neighborhood graphs
and local membership probability calculations for community analysis.

Classes
-------
DelaunayGraph
    Constructs a graph based on Delaunay triangulation of input coordinates.
LocalMembershipProbability
    Calculates local membership probabilities for nodes based on their
    neighborhood structure and label assignments.

Examples
--------
>>> import xarray as xr
>>> from AFL.double_agent.Graph import DelaunayGraph
>>> 
>>> # Create a dataset with 2D coordinates
>>> coords = xr.DataArray([[0, 0], [1, 0], [0.5, 1]], dims=["sample", "feature"])
>>> ds = xr.Dataset({"composition": coords})
>>> 
>>> # Build Delaunay graph
>>> graph_op = DelaunayGraph(
...     input_variable="composition",
...     output_variable="adjacency",
...     sample_dim="sample"
... )
>>> result = graph_op.calculate(ds)
>>> adjacency_matrix = result.output["adjacency"]

"""
import numpy as np
import scipy.spatial  # type: ignore
import networkx as nx
import xarray as xr
from typing import Optional
from typing_extensions import Self

from AFL.double_agent.PipelineOp import PipelineOp


class DelaunayGraph(PipelineOp):
    """Construct a graph based on Delaunay triangulation.

    Parameters
    ----------
    input_variable : str
        Name of variable containing positions used to compute the triangulation.
    output_variable : str
        Name of the adjacency matrix variable to create.
    sample_dim : str, default "sample"
        Dimension of the samples in the dataset.
    name : str, default "DelaunayGraph"
        Name of the PipelineOp.
    """

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        sample_dim: str = "sample",
        name: str = "DelaunayGraph",
    ) -> None:
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )
        self.sample_dim = sample_dim
        self.graph: Optional[nx.Graph] = None
        self._banned_from_attrs.append("graph")

    def calculate(self, dataset: xr.Dataset) -> Self:
        data = self._get_variable(dataset)
        tri = scipy.spatial.Delaunay(data.values)
        simplices = tri.simplices if hasattr(tri, "simplices") else tri.vertices
        edges = np.concatenate(
            (simplices[:, :2], simplices[:, 1:], simplices[:, ::2]), axis=0
        )
        n = data.shape[0]
        adj = np.zeros((n, n))
        adj[edges[:, 0], edges[:, 1]] = 1.0
        adj = np.clip(adj + adj.T, 0, 1)
        self.graph = nx.from_numpy_array(adj)
        dims = [f"{self.sample_dim}_i", f"{self.sample_dim}_j"]
        self.output[self.output_variable] = xr.DataArray(adj, dims=dims)
        return self


class LocalMembershipProbability(PipelineOp):
    """Estimate local membership probabilities using a neighborhood graph.

    Parameters
    ----------
    labels_variable : str
        Name of variable containing integer cluster labels.
    adjacency_variable : str
        Name of variable containing adjacency/similarity matrix.
    output_variable : str
        Name of the variable to store membership probabilities.
    sample_dim : str, default "sample"
        Dimension for samples in the dataset.
    prob_variable : Optional[str], default None
        Optional variable containing prior probabilities ``(sample, labels_unique)``.
    v : float, default 1.0
        Exponent controlling sharpness of probabilities.
    name : str, default "LocalMembershipProbability"
        Name of the PipelineOp.
    """

    def __init__(
        self,
        labels_variable: str,
        adjacency_variable: str,
        output_variable: str,
        prob_variable: Optional[str] = None,
        sample_dim: str = "sample",
        v: float = 1.0,
        name: str = "LocalMembershipProbability",
    ) -> None:
        super().__init__(
            name=name,
            input_variable=[labels_variable, adjacency_variable, prob_variable],
            output_variable=output_variable,
        )
        self.labels_variable = labels_variable
        self.adjacency_variable = adjacency_variable
        self.prob_variable = prob_variable
        self.sample_dim = sample_dim
        self.v = v
        self.probabilities: Optional[np.ndarray] = None
        self._banned_from_attrs.append("probabilities")

    def calculate(self, dataset: xr.Dataset) -> Self:
        labels = dataset[self.labels_variable].values
        unique_labels = np.unique(labels)
        W = dataset[self.adjacency_variable].values
        n_samples = W.shape[0]

        if self.prob_variable is not None:
            prior = dataset[self.prob_variable].values
            n_clusters = prior.shape[1]
        else:
            n_clusters = len(unique_labels)
            prior = None

        cumulative = np.zeros((n_clusters, n_samples))
        counts = np.zeros((n_clusters, n_samples))

        for i in range(n_samples):
            if prior is None:
                c = int(labels[i])
                counts[c, i] += 1
            else:
                counts[:, i] += prior[i]
                cumulative[:, i] += prior[i]

            neighbors = np.nonzero(W[i])[0]
            weights = W[i, neighbors]
            for j, w in zip(neighbors, weights):
                if prior is None:
                    cj = int(labels[j])
                    counts[cj, i] += 1
                    cumulative[cj, i] += w
                else:
                    counts[:, i] += prior[j]
                    cumulative[:, i] += prior[j] * w

        avg = np.divide(cumulative, counts, out=np.zeros_like(cumulative), where=counts != 0)
        node_sum = np.nansum(avg ** self.v, axis=0)
        prob_mat = np.nan_to_num(
            np.divide(avg ** self.v, node_sum, where=node_sum != 0), nan=0.0
        )
        probabilities = prob_mat.T
        self.probabilities = probabilities
        dims = [self.sample_dim, "labels_unique"]
        self.output[self.output_variable] = xr.DataArray(probabilities, dims=dims,coords={"labels_unique":unique_labels})
        return self
