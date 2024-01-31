"""
ToDo:
- DummyGP
- TensorFlowGP
- SklearnGP

- Overhaul transforms? Should these be


"""
import xarray as xr
import numpy as np
import tqdm

from AFL.double_agent.Pipeline import PipelineOpBase


class Extrapolator(PipelineOpBase):
    def __init__(self, feature_input_variable, predictor_input_variable, output_variable, grid,name='Extrapolator'):

        super().__init__(name=name,input_variable=[feature_input_variable,predictor_input_variable],
                         output_variable=output_variable)
        self.feature_input_variable = feature_input_variable
        self.predictor_input_variable = predictor_input_variable
        self.grid = grid

class DummyExtrapolator(Extrapolator):
    def __init__(self, feature_input_variable, predictor_input_variable, output_variable,
                 grid, name='DummyExtrapolator'):

            super().__init__(name=name, feature_input_variable=feature_input_variable,
                             predictor_input_variable=predictor_input_variable,
                             output_variable=output_variable,grid=grid)

    def calculate(self,components):
        dummy = xr.DataArray(np.zeros_like(self.grid),dims=self.grid.dims)
        self.output[self.output_variable+"_mean"] = dummy.copy()
        self.output[self.output_variable+"_var"] = dummy.copy()
        return self

    
