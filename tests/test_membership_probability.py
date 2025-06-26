import numpy as np
import pytest
import xarray as xr

from AFL.double_agent.Labeler import ClusterMembershipProbability


@pytest.mark.unit
def test_membership_probability(example_dataset1_cluster_result):
    op = ClusterMembershipProbability(
        similarity_variable="similarity",
        labels_variable="labels",
        output_variable="probs",
        sample_dim="sample",
    )

    op.calculate(example_dataset1_cluster_result)
    probs = op.output["probs"]

    assert probs.dims[0] == "sample"
    assert np.allclose(probs.sum(dim="labels_unique"), 1.0)
    assert probs.shape[0] == example_dataset1_cluster_result.dims["sample"]
