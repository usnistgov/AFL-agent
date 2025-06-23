"""Example PipelineOp with proper type annotations for testing dictionary parameter detection."""

from typing import Dict, List, Optional, Union
import xarray as xr
from AFL.double_agent.PipelineOp import PipelineOp


class ExampleTypedOp(PipelineOp):
    """Example PipelineOp with various parameter types including dictionaries.
    
    This serves as a test case for the parameter type detection system.
    """
    
    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        config: Dict[str, Union[str, int, float]],
        options: Dict[str, bool] = None,
        settings: dict = None,
        name: str = "ExampleTypedOp",
        scale_factor: float = 1.0,
        threshold: int = 10,
        enabled: bool = True,
        tags: List[str] = None,
        processing_steps: List[str] = None,
        filters: list = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            name=name,
            input_variable=input_variable,
            output_variable=output_variable
        )
        
        self.config = config or {}
        self.options = options or {}
        self.settings = settings or {}
        self.scale_factor = scale_factor
        self.threshold = threshold
        self.enabled = enabled
        self.tags = tags or []
        self.processing_steps = processing_steps or []
        self.filters = filters or []
        self.metadata = metadata or {}
    
    def calculate(self, dataset: xr.Dataset):
        """Simple calculation that demonstrates the operation."""
        data = self._get_variable(dataset)
        
        # Apply scale factor
        result = data * self.scale_factor
        
        # Store result
        self.output[self.output_variable] = result
        
        return self 