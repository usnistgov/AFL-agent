"""
Unit tests for the AFL.double_agent.PipelineOp module.
"""

import pytest
import numpy as np
import xarray as xr

from tests.utils import MockPipelineOp


@pytest.mark.unit
class TestPipelineOp:
    """Tests for the PipelineOp class."""

    def test_pipelineop_creation(self):
        """Test creating a PipelineOp."""
        op = MockPipelineOp(input_variable="x", output_variable="y", name="TestOp")

        assert op.input_variable == "x"
        assert op.output_variable == "y"
        assert op.name == "TestOp"

    def test_pipelineop_default_name(self):
        """Test that PipelineOp generates a default name if none is provided."""
        op = MockPipelineOp(input_variable="x", output_variable="y")

        assert op.name is not None
        assert "PipelineOp" in op.name

    def test_pipelineop_variable_lists(self):
        """Test that PipelineOp correctly handles input/output variable lists."""
        # Test with string inputs
        op_single = MockPipelineOp(name="TestOp", input_variable="x", output_variable="y")
        assert op_single.input_variable == "x"
        assert op_single.output_variable == "y"

        # Test with list inputs
        op_multi = MockPipelineOp(
            name="TestOp", input_variable=["x", "z"], output_variable=["y", "w"]
        )
        assert op_multi.input_variable == ["x", "z"]
        assert op_multi.output_variable == ["y", "w"]

    def test_pipelineop_repr(self):
        """Test the string representation of a PipelineOp."""
        op = MockPipelineOp(name="TestOp", input_variable="x", output_variable="y")

        # Check that repr includes essential info
        repr_str = repr(op)
        assert "TestOp" in repr_str

    def test_copy(self):
        """Test copying a PipelineOp."""
        op = MockPipelineOp(input_variable="x", output_variable="y", name="TestOp")

        # Make a copy
        op_copy = op.copy()

        # Check that it's a new object with the same properties
        assert op_copy is not op
        assert op_copy.name == op.name
        assert op_copy.input_variable == op.input_variable
        assert op_copy.output_variable == op.output_variable

    def test_mock_pipelineop_calculate(self):
        """Test calculation with MockPipelineOp."""
        # Create dataset
        data = [np.arange(128)] * 10
        ds = xr.Dataset({"measurement": (("sample", "q"), data)})

        # Create op
        op = MockPipelineOp(name="TestOp", input_variable="measurement", output_variable="labels")

        # Calculate
        result = op.calculate(ds)

        # Check that the op returned itself
        assert result is op

        # Check that the calculation was performed
        assert op.calculated
        assert "labels" in result.output
        np.testing.assert_array_equal(result.output["labels"].values, data)
