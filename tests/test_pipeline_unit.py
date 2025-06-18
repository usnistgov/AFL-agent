"""
Unit tests for the AFL.double_agent.Pipeline module.
"""

import os

import networkx as nx
import numpy as np
import pytest

from AFL.double_agent import Pipeline
from unittest.mock import patch

from tests.utils import (
    MockPipelineOp,
    MockPreprocessor,
)


@pytest.mark.unit
class TestPipelineUnit:
    """Tests for the Pipeline class."""

    def test_pipeline_creation(self):
        """Test creating a Pipeline."""
        pipeline = Pipeline(name="TestPipeline")

        assert pipeline.name == "TestPipeline"
        assert len(pipeline.ops) == 0
        assert pipeline.result is None

    def test_pipeline_creation_with_ops(self):
        """Test creating a Pipeline with operations."""
        op1 = MockPipelineOp("Op1", "x", "y")
        op2 = MockPipelineOp("Op2", "y", "z")

        pipeline = Pipeline(name="TestPipeline", ops=[op1, op2])

        assert pipeline.name == "TestPipeline"
        assert len(pipeline.ops) == 2
        assert pipeline.ops[0] == op1
        assert pipeline.ops[1] == op2

    def test_pipeline_context_manager(self):
        """Test creating a Pipeline using the context manager."""
        with Pipeline("TestPipeline") as pipeline:
            op1 = MockPipelineOp(name="Op1", input_variable="x", output_variable="y")
            op2 = MockPipelineOp(name="Op2", input_variable="y", output_variable="z")

        assert pipeline.name == "TestPipeline"
        assert len(pipeline.ops) == 2
        assert pipeline.ops[0].name == "Op1"
        assert pipeline.ops[1].name == "Op2"

    def test_pipeline_append(self):
        """Test adding an operation to a Pipeline."""
        pipeline = Pipeline("TestPipeline")
        op = MockPipelineOp("TestOp", "x", "y")

        result = pipeline.append(op)

        assert len(pipeline.ops) == 1
        assert pipeline.ops[0] == op
        assert result == pipeline  # Method should return self for chaining

    def test_pipeline_extend(self):
        """Test extending a Pipeline with multiple operations."""
        pipeline = Pipeline("TestPipeline")
        op1 = MockPipelineOp("Op1", "x", "y")
        op2 = MockPipelineOp("Op2", "y", "z")

        result = pipeline.extend([op1, op2])

        assert len(pipeline.ops) == 2
        assert pipeline.ops[0] == op1
        assert pipeline.ops[1] == op2
        assert result == pipeline  # Method should return self for chaining

    def test_pipeline_iter(self):
        """Test iterating over Pipeline operations."""
        with Pipeline("TestPipeline") as pipeline:
            op1 = MockPipelineOp(name="Op1", input_variable="x", output_variable="y")
            op2 = MockPipelineOp(name="Op2", input_variable="y", output_variable="z")

        ops = list(pipeline)

        assert len(ops) == 2
        assert ops[0].name == "Op1"
        assert ops[1].name == "Op2"

    def test_pipeline_getitem(self):
        """Test accessing Pipeline operations by index."""
        with Pipeline("TestPipeline") as pipeline:
            op1 = MockPipelineOp(name="Op1", input_variable="x", output_variable="y")
            op2 = MockPipelineOp(name="Op2", input_variable="y", output_variable="z")

        assert pipeline[0].name == "Op1"
        assert pipeline[1].name == "Op2"

    def test_pipeline_search(self):
        """Test searching for operations by name."""
        with Pipeline("TestPipeline") as pipeline:
            op1 = MockPipelineOp(name="Op1", input_variable="x", output_variable="y")
            op2 = MockPipelineOp(name="Op2", input_variable="y", output_variable="z")
            op3 = MockPipelineOp(name="SpecialOp", input_variable="z", output_variable="w")

        assert pipeline.search("Op1").name == "Op1"
        assert pipeline.search("Op2").name == "Op2"
        assert pipeline.search("SpecialOp").name == "SpecialOp"
        assert pipeline.search("NonExistentOp") is None

    def test_pipeline_search_contains(self):
        """Test searching for operations with contains mode."""
        with Pipeline("TestPipeline") as pipeline:
            op1 = MockPipelineOp(name="ProcessingOp", input_variable="x", output_variable="y")
            op2 = MockPipelineOp(name="FilteringOp", input_variable="y", output_variable="z")

        assert pipeline.search("Processing", contains=True).name == "ProcessingOp"
        assert pipeline.search("Filter", contains=True).name == "FilteringOp"
        assert pipeline.search("Op", contains=True).name == "ProcessingOp"  # Returns first match
        assert pipeline.search("NonExistent", contains=True) is None

    def test_pipeline_repr(self):
        """Test the string representation of a Pipeline."""
        pipeline = Pipeline("TestPipeline")

        assert repr(pipeline) == "<Pipeline TestPipeline N=0>"

        with Pipeline("TestPipeline") as pipeline:
            op1 = MockPipelineOp(name="Op1", input_variable="x", output_variable="y")
            op2 = MockPipelineOp(name="Op2", input_variable="y", output_variable="z")

        assert repr(pipeline) == "<Pipeline TestPipeline N=2>"

    def test_pipeline_copy(self):
        """Test copying a Pipeline."""
        with Pipeline("TestPipeline") as pipeline:
            op1 = MockPipelineOp(name="Op1", input_variable="x", output_variable="y")
            op2 = MockPipelineOp(name="Op2", input_variable="y", output_variable="z")

        pipeline_copy = pipeline.copy()

        assert pipeline_copy is not pipeline
        assert pipeline_copy.name == pipeline.name
        assert len(pipeline_copy.ops) == len(pipeline.ops)

        # Check the operations are the same but not identical objects
        assert pipeline_copy.ops[0].name == pipeline.ops[0].name
        assert pipeline_copy.ops[1].name == pipeline.ops[1].name
        assert pipeline_copy.ops[0] is not pipeline.ops[0]
        assert pipeline_copy.ops[1] is not pipeline.ops[1]

    def test_calculate_single_op(self, example_dataset1):
        """Test calculating a pipeline with a single operation."""
        # Create pipeline with one operation
        with Pipeline("TestPipeline") as pipeline:
            MockPreprocessor(name="ProcessOp", input_variable="measurement", output_variable="processed")

        # Calculate
        result = pipeline.calculate(example_dataset1)

        # Check results
        assert "processed" in result
        assert result["processed"].dims == result["measurement"].dims
        assert result["processed"].shape == result["measurement"].shape
        # Verify transformation was applied (multiplication by 2.0)
        np.testing.assert_allclose(result["processed"].values, result["measurement"].values * 2.0)

    def test_calculate_multiple_ops(self, example_dataset1):
        """Test calculating a pipeline with multiple operations."""
        # Create pipeline with multiple operations
        with Pipeline("TestPipeline") as pipeline:
            # First operation: process measurement data
            MockPreprocessor("ProcessOp", input_variable="measurement", output_variable="processed")

            # Second operation: copy processed to output
            MockPipelineOp(name="CopyOp", input_variable="processed", output_variable="output")

        # Calculate
        result = pipeline.calculate(example_dataset1)

        # Check results
        assert "processed" in result
        assert "output" in result
        assert result["processed"].dims == result["measurement"].dims
        assert result["output"].dims == result["processed"].dims
        # Verify both operations were applied correctly
        np.testing.assert_allclose(result["processed"].values, result["measurement"].values * 2.0)
        np.testing.assert_allclose(result["output"].values, result["processed"].values)

    def test_calculate_with_failure(self, example_dataset1):
        """Test pipeline calculation with an operation that fails."""
        # Create pipeline with an operation that will fail
        with Pipeline("TestPipeline") as pipeline:
            MockPipelineOp(
                name="FailingOp",
                input_variable="measurement",
                output_variable="output",
                fail_on_calculate=True,
            )

        # Test that calculation raises the appropriate exception
        with pytest.raises(RuntimeError, match="Calculation deliberately failed"):
            pipeline.calculate(example_dataset1)

    def test_input_output_variables(self):
        """Test getting input and output variables from a pipeline."""
        with Pipeline("TestPipeline") as pipeline:
            MockPipelineOp(name="Op1", input_variable="x", output_variable="y")
            MockPipelineOp(name="Op2", input_variable="y", output_variable="z")
            MockPipelineOp(name="Op3", input_variable="a", output_variable="b")

        # Make the graph first
        pipeline.make_graph()

        # Check input variables (should be unique)
        inputs = pipeline.input_variables()
        assert sorted(inputs) == ["a", "x"]

        # Check output variables
        outputs = pipeline.output_variables()
        assert sorted(outputs) == ["b", "z"]

    def test_serialization_json(self, tmp_path):
        """Test saving and loading a pipeline."""
        # Create a pipeline to save
        with Pipeline("TestPipeline") as pipeline:
            MockPipelineOp(name="Op1", input_variable="x", output_variable="y")
            MockPipelineOp(name="Op2", input_variable="y", output_variable="z")

        # Save the pipeline
        filename = tmp_path / "test_pipeline.pkl"
        pipeline.write_json(str(filename))

        # Check that the file exists
        assert os.path.exists(filename)

        # Load the pipeline
        loaded_pipeline = Pipeline.read_json(str(filename))

        # Check that it's the same
        assert loaded_pipeline.name == pipeline.name
        assert len(loaded_pipeline.ops) == len(pipeline.ops)
        assert loaded_pipeline.ops[0].name == pipeline.ops[0].name
        assert loaded_pipeline.ops[1].name == pipeline.ops[1].name

    def test_serialization_pkl(self, tmp_path):
        """Test saving and loading a pipeline."""
        # Create a pipeline to save
        with Pipeline("TestPipeline") as pipeline:
            MockPipelineOp(name="Op1", input_variable="x", output_variable="y")
            MockPipelineOp(name="Op2", input_variable="y", output_variable="z")

        # Save the pipeline
        filename = tmp_path / "test_pipeline.pkl"
        pipeline.write_pkl(str(filename))

        # Check that the file exists
        assert os.path.exists(filename)

        # Load the pipeline
        loaded_pipeline = Pipeline.read_pkl(str(filename))

        # Check that it's the same
        assert loaded_pipeline.name == pipeline.name
        assert len(loaded_pipeline.ops) == len(pipeline.ops)
        assert loaded_pipeline.ops[0].name == pipeline.ops[0].name
        assert loaded_pipeline.ops[1].name == pipeline.ops[1].name

    def test_make_graph(self):
        """Test creating a graph representation of the pipeline."""
        # Create a pipeline with a linear flow
        with Pipeline("TestPipeline") as pipeline:
            MockPipelineOp(name="Op1", input_variable="x", output_variable="y")
            MockPipelineOp(name="Op2", input_variable="y", output_variable="z")

        # Make the graph
        pipeline.make_graph()

        # Check that the graph was created
        assert pipeline.graph is not None
        assert isinstance(pipeline.graph, nx.DiGraph)
        assert len(pipeline.graph.nodes) > 0

        # Check edges - should be a connection from op1 to op2
        # through the variable y
        edges = list(pipeline.graph.edges)
        assert len(edges) > 0

    def test_draw(self):
        """Test drawing the pipeline graph."""
        # Create a simple pipeline
        with Pipeline("TestPipeline") as pipeline:
            MockPipelineOp(name="Op1", input_variable="x", output_variable="y")
            MockPipelineOp(name="Op2", input_variable="y", output_variable="z")

        # Mock the necessary components for drawing
        with (
            patch("matplotlib.pyplot.figure") as mock_figure,
            patch("networkx.nx_agraph.pygraphviz_layout") as mock_layout,
            patch("networkx.draw") as mock_draw,
            patch("networkx.draw_networkx_edge_labels") as mock_edge_labels,
        ):

            # Set up the mock layout to return a valid position dictionary
            mock_layout.return_value = {"x": (0, 0), "y": (1, 1), "z": (2, 2)}

            # Call the drawing function
            fig = pipeline.draw(figsize=(10, 10), edge_labels=True)

            # Check that the plotting functions were called
            assert mock_figure.called
            assert mock_layout.called
            assert mock_draw.called
            assert mock_edge_labels.called
