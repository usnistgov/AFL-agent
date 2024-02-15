"""
PairMetrics are PipelineOps that produce pair matrices as results

"""
import xarray as xr
import numpy as np

import sklearn.gaussian_process
import sklearn.gaussian_process.kernels

from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify

class MaskGrid(PipelineOp):
    """
    applies a mask to a grid and renames the output dimension
    """
    def __init__(self, input_variable, output_variable, mask_variable, output_prefix, grid_dim_in, grid_dim_out, name='MaskGrid'):
        super().__init__(
            name = name ,
            input_variable= input_variable,
            output_variable = output_variable,
            output_prefix = output_prefix
        )

        self.output_prefix = output_prefix
        self.input_variable = input_variable
        self.output_variable = output_variable
        self.grid_dim_in = grid_dim_in
        self.grid_dim_out = grid_dim_out
        self.mask_variable = mask_variable

    def calculate(self,dataset):
        grid = dataset[self.input_variable]
        # print(grid)
        
        mask = dataset[self.mask_variable]
        # print(grid[mask])
        
        masked_grid =grid[mask].rename({self.grid_dim_in:self._prefix_output(self.grid_dim_out)})
        # print('sans_masked_comps_grid',masked_grid)
        self.output[self._prefix_output(self.output_variable)] = masked_grid
        return self


