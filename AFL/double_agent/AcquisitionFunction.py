"""Acquisition Functions

Parent class with methods for parsing decision array

break out decision pieces into more functions
- get_top_n_pct
- check_already_measured

support for producing n-predictions

"""
import numpy as np
import xarray as xr

from AFL.double_agent.util import listify
from AFL.double_agent.Pipeline import PipelineOpBase


class AcquisitionFunction(PipelineOpBase):
    def __init__(self, input_variables, grid_variable, grid_dim='grid', output_prefix=None, count=1,
                 output_variable="next_samples", name='AcquisitionFunctionBase'):
        """
        Parameters
        ----------
        count: int
            Number of 'next_samples' to generate

        """
        super().__init__(input_variable=listify(input_variables), output_variable=output_variable, name=name,
                         output_prefix=output_prefix)

        self.count = count
        self.grid_variable = grid_variable
        self.grid_dim = grid_dim

        self._banned_from_attrs.append('acquisition')


class RandomAF(AcquisitionFunction):
    def __init__(self, grid_variable, grid_dim, output_prefix, count=1, name='RandomAF'):
        super().__init__(input_variables=grid_variable, grid_variable=grid_variable, grid_dim=grid_dim,
                         output_prefix=output_prefix, count=count, name=name)

        self.acquisition = None

    def calculate(self, dataset):
        self.acquisition = xr.Dataset()

        self.acquisition['comp_grid'] = dataset[self.grid_variable].transpose(self.grid_dim, ...)

        self.acquisition['decision_surface'] = xr.DataArray(
            np.random.random(size=self.acquisition['grid'].values.shape),
            dims=self.grid_dim
        )

        self.output[self._prefix_output('decision_surface')] = self.acquisition['decision_surface']
        self.output[self._prefix_output('decision_surface')].attrs['description'] = (
            "The final acquisition surface that is evaluated to determine the next_sample"
        )

        return self
