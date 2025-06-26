import numpy as np
import xarray as xr
import networkx as nx
import pytest

from AFL.double_agent.Graph import DelaunayGraph, LocalMembershipProbability

@pytest.mark.unit
def test_delaunay_graph(example_dataset1):
    op = DelaunayGraph(
        input_variable="composition",
        output_variable="delaunay_graph",
        sample_dim="sample",
    )
    result = op.calculate(example_dataset1).add_to_dataset(example_dataset1)

    n = example_dataset1.sizes["sample"]
    assert "delaunay_graph" in result
    assert result["delaunay_graph"].shape == (n, n)
    assert op.graph is not None
    assert isinstance(op.graph, nx.Graph)
    assert op.graph.number_of_nodes() == n


@pytest.mark.unit
def test_local_membership_probability():
    labels = xr.DataArray([0, 1, 0], dims=["sample"])
    adj = xr.DataArray(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        dims=["sample_i", "sample_j"],
    )
    ds = xr.Dataset({"labels": labels, "adj": adj})

    op = LocalMembershipProbability(
        labels_variable="labels",
        adjacency_variable="adj",
        output_variable="prob",
        sample_dim="sample",
    )
    result = op.calculate(ds).add_to_dataset(ds)

    expected = np.array(
        [
            [1 / 3, 2 / 3],
            [1.0, 0.0],
            [1 / 3, 2 / 3],
        ]
    )
    np.testing.assert_allclose(result["prob"].values, expected)
    assert result["prob"].dims == ("sample", "labels_unique")
