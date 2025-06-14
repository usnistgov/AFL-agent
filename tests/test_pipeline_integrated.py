"""
Integration tests for AFL.double_agent pipeline workflows.
"""

import os
import tempfile
from pathlib import Path

import pytest
import numpy as np
import xarray as xr

from AFL.double_agent import Pipeline
from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.Preprocessor import SavgolFilter, Standardize
from AFL.double_agent.Extrapolator import GaussianProcessRegressor
from AFL.double_agent.PairMetric import Similarity
from AFL.double_agent.Labeler import SpectralClustering


@pytest.mark.integration
class TestPipelineIntegrated:
    """Integration tests for pipeline workflows."""

    @pytest.fixture
    def clustering_pipeline(self):
        """Create a reusable clustering pipeline for tests."""
        with Pipeline("ClusteringPipeline") as pipeline:
            SavgolFilter(
                input_variable="measurement", output_variable="derivative", dim="x", derivative=1
            )

            Similarity(
                input_variable="derivative",
                output_variable="similarity",
                sample_dim="sample",
                params={"metric": "laplacian", "gamma": 1e-4},
            )

            SpectralClustering(
                input_variable="similarity",
                output_variable="labels",
                dim="sample",
                params={"n_phases": 2},
            )
        return pipeline

    def test_basic_workflow(
        self, example_dataset1, example_dataset1_cluster_result, clustering_pipeline
    ):
        """Test a basic pipeline workflow with real components."""
        # Run the pipeline
        result = clustering_pipeline.calculate(example_dataset1)

        # Check results
        xr.testing.assert_equal(
            result["composition"], example_dataset1_cluster_result["composition"]
        )
        xr.testing.assert_equal(
            result["ground_truth_labels"], example_dataset1_cluster_result["ground_truth_labels"]
        )
        xr.testing.assert_equal(
            result["measurement"], example_dataset1_cluster_result["measurement"]
        )
        xr.testing.assert_equal(
            result["composition_grid"], example_dataset1_cluster_result["composition_grid"]
        )
        xr.testing.assert_allclose(result["derivative"], example_dataset1_cluster_result["derivative"])
        xr.testing.assert_allclose(result["similarity"], example_dataset1_cluster_result["similarity"])
        xr.testing.assert_equal(result["labels"], example_dataset1_cluster_result["labels"])

    def test_save_load_pipeline_with_compute(self, example_dataset1, tmp_path, clustering_pipeline):
        """Test saving, loading, and computing with a pipeline."""
        # Save pipeline
        save_path = tmp_path / "test_pipeline.json"
        clustering_pipeline.write_json(str(save_path))

        # Run original pipeline
        original_result = clustering_pipeline.calculate(example_dataset1.copy())

        # Load pipeline
        loaded_pipeline = Pipeline.read_json(str(save_path))

        # Run loaded pipeline
        loaded_result = loaded_pipeline.calculate(example_dataset1.copy())

        # Check results
        xr.testing.assert_equal(original_result["composition"], loaded_result["composition"])
        xr.testing.assert_equal(
            original_result["ground_truth_labels"], loaded_result["ground_truth_labels"]
        )
        xr.testing.assert_equal(original_result["measurement"], loaded_result["measurement"])
        xr.testing.assert_equal(original_result["derivative"], loaded_result["derivative"])
        xr.testing.assert_equal(original_result["similarity"], loaded_result["similarity"])
        xr.testing.assert_equal(original_result["labels"], loaded_result["labels"])
