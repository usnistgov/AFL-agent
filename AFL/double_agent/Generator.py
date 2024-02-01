"""Generators take parametric input to generate data
"""

from itertools import product

import numpy as np
import xarray as xr

from AFL.double_agent.Pipeline import PipelineOpBase


class Generator(PipelineOpBase):
    """Base class stub for all generators"""

    def __init__(self, input_variable='Generator', output_variable=None, name='GeneratorBase'):
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)


class CartesianGrid(Generator):
    """Generator that produces a cartesian grid according to use provided min/max/step"""
    def __init__(self, output_variable, grid_spec, sample_dim, name='CartesianGridGenerator'):
        """
        Parameters
        ----------
        output_variable : str
            The dataset key to write the result of this PipelineOp into

        grid_spec: dict of dicts
            Dictionary where each top-level key corresponds to a component in the system. Each top-level key points to a
            subdictionary that defines the mininum, maximum, and step size for that component with keys: min, max, step.

        """
        #using intput_variable just as a placeholder for visualization purposes
        super().__init__(name=name, input_variable='CartesianGridGenerator', output_variable=output_variable)
        self.grid_spec = grid_spec
        self.components = list(grid_spec.keys())
        self.sample_dim = sample_dim

    def calculate(self, dataset):
        grid_list = []
        for component in self.components:
            print(component)
            spec = self.grid_spec[component]
            grid_list.append(np.linspace(spec['min'], spec['max'], spec['steps']))

        pts = np.array(list(product(*grid_list)))
        self.output[self.output_variable] = (
            xr.DataArray(pts, dims=[self.sample_dim, 'component'], coords={'component': self.components})
        )
        return self


class BarycentricGrid(Generator):
    def __init__(self, output_variable, components, sample_dim, pts_per_row=50, basis=1.0, dim=3, eps=1e-9,
                 name='BarycentricGridGenerator'):
        #using intput_variable just as a placeholder for visualization purposes
        super().__init__(name=name, input_variable=name, output_variable=output_variable)
        self.components = components
        self.sample_dim = sample_dim
        self.pts_per_row = pts_per_row
        self.basis = basis
        self.dim = dim
        self.eps = eps

    def calculate(self, dataset):
        grid_list = []
        for i in product(*[np.linspace(0, 1.0, self.pts_per_row)] * (self.dim - 1)):
            if sum(i) > (1.0 + self.eps):
                continue

            j = 1.0 - sum(i)

            if j < (0.0 - self.eps):
                continue
            pt = [k * self.basis for k in [*i, j]]
            grid_list.append(pt)

        pts = np.array(grid_list)
        self.output[self.output_variable] = (
            xr.DataArray(pts, dims=[self.sample_dim, 'component'], coords={'component': self.components})
        )
        return self
