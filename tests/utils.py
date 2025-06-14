"""
Test utility functions and mock classes for AFL.double_agent tests.
"""

import numpy as np
import xarray as xr
from typing import Optional, List, Dict, Any, Union

from AFL.double_agent.PipelineOp import PipelineOp


class MockPipelineOp(PipelineOp):
    """
    A mock PipelineOp for testing Pipeline behavior.

    This mock operation simply copies the input variable to the output variable.
    """

    def __init__(
        self,
        input_variable: str | None = None,
        output_variable: str | None = None,
        fail_on_calculate: bool = False,
        name: str = "MockPipelineOp",
    ):
        """
        Initialize the MockPipelineOp.

        Parameters
        ----------
        name : Optional[str]
            The name of the operation
        input_variable : str
            The name of the input variable in the dataset
        output_variable : str
            The name of the output variable in the dataset
        fail_on_calculate : bool
            If True, raise an exception when calculate is called
        """
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)
        self.calculated = False
        self.fail_on_calculate = fail_on_calculate

    def calculate(self, dataset: xr.Dataset, **kwargs) -> PipelineOp:
        """
        Mock calculation that copies input to output.

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to operate on

        Returns
        -------
        PipelineOp
            Self reference for method chaining
        """
        if self.fail_on_calculate:
            raise RuntimeError("Calculation deliberately failed for testing")

        if self.input_variable in dataset:
            self.output = {self.output_variable: dataset[self.input_variable].copy()}
        else:
            # Create dummy data if input doesn't exist
            self.output = {
                self.output_variable: xr.DataArray(
                    np.ones((2, 2)), dims=("x", "y"), coords={"x": [0, 1], "y": [0, 1]}
                )
            }

        self.calculated = True
        return self


class MockPreprocessor(PipelineOp):
    """
    A mock preprocessor for testing.

    Applies a simple transformation to the input data:
    - For 1D data: adds a constant
    - For 2D data: multiplies by a constant
    """

    def __init__(
        self,
        name: Optional[str] = None,
        input_variable: str = None,
        output_variable: str = None,
        transform_value: float = 2.0,
    ):
        """
        Initialize the MockPreprocessor.

        Parameters
        ----------
        name : Optional[str]
            The name of the operation
        input_variable : str
            The name of the input variable in the dataset
        output_variable : str
            The name of the output variable in the dataset
        transform_value : float
            The value to use in transformations
        """
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)
        self.transform_value = transform_value

    def calculate(self, dataset: xr.Dataset, **kwargs) -> PipelineOp:
        """
        Apply a simple transformation to the input data.

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to operate on

        Returns
        -------
        PipelineOp
            Self reference for method chaining
        """
        if self.input_variable not in dataset:
            raise ValueError(f"Input variable {self.input_variable} not found in dataset")

        input_data = dataset[self.input_variable]

        # Apply different transformations based on dimensionality
        if len(input_data.dims) == 1:
            # For 1D data: add constant
            self.output = {self.output_variable: input_data + self.transform_value}
        else:
            # For 2D+ data: multiply by constant
            self.output = {self.output_variable: input_data * self.transform_value}

        return self
