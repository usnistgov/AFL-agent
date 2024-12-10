"""Generators take parametric input to generate data"""

from typing import Dict, List

import numpy as np
import xarray as xr
from itertools import product
from scipy.stats import multivariate_normal  # type: ignore
from typing_extensions import Self

from AFL.double_agent.PipelineOp import PipelineOp


class Generator(PipelineOp):
    """Base class stub for all generators"""

    def __init__(
        self,
        output_variable: str,
        input_variable: str = "Generator",
        name: str = "GeneratorBase",
    ) -> None:
        """
        Parameters
        ----------
        input_variable : str
            Generators generally do not use input variables but this can be used to name the input node for a generator

        output_variable : str
            The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search()

        """
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        return NotImplementedError("Calculate must be implemented in subclasses")  # type: ignore


class CartesianGrid(Generator):
    """Generator that produces a cartesian grid according to use provided min/max/step"""

    def __init__(
        self,
        output_variable: str,
        grid_spec: Dict[str, Dict[str, int | float]],
        sample_dim: str,
        component_dim: str = 'component',
        name: str = "CartesianGridGenerator",
    ):
        """
        Parameters
        ----------
        output_variable : str
            The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

        grid_spec: dict of dicts Dictionary where each top-level key corresponds to a component in the system. Each
        top-level key points to a subdictionary that defines the mininum, maximum, and step size for that component
        with keys: min, max, steps.

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search()

        """
        # using intput_variable just as a placeholder for visualization purposes
        super().__init__(
            name=name,
            input_variable="CartesianGridGenerator",
            output_variable=output_variable,
        )
        self.grid_spec = grid_spec
        self.components = list(grid_spec.keys())
        self.sample_dim = sample_dim
        self.component_dim = component_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        grid_list = []
        for component in self.components:
            spec = self.grid_spec[component]
            grid_list.append(np.linspace(spec["min"], spec["max"], spec["steps"]))  # type: ignore

        pts = np.array(list(product(*grid_list)))
        self.output[self.output_variable] = xr.DataArray(
            pts,
            dims=[self.sample_dim, self.component_dim],
            coords={self.component_dim: self.components},
        )
        return self


class BarycentricGrid(Generator):
    def __init__(
        self,
        output_variable: str,
        components: List[str],
        sample_dim: str,
        pts_per_row: int = 50,
        basis: float = 1.0,
        dim: int = 3,
        eps: float = 1e-9,
        name="BarycentricGridGenerator",
    ):
        # using input_variable just as a placeholder for visualization purposes
        super().__init__(
            name=name, input_variable=name, output_variable=output_variable
        )
        self.components = components
        self.sample_dim = sample_dim
        self.pts_per_row = pts_per_row
        self.basis = basis
        self.dim = dim
        self.eps = eps

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
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
        self.output[self.output_variable] = xr.DataArray(
            pts,
            dims=[self.sample_dim, "component"],
            coords={"component": self.components},
        )
        return self


class GaussianPoints(Generator):
    def __init__(
        self,
        input_variable: str,
        sample_dim: str,
        output_variable: str,
        grid_variable: str,
        grid_dim: str,
        comps_dim: str = "component",
        exclusion_depth: float = 1e-3,
        exclusion_radius: float = 1e-3,
        name: str = "GaussianPointsGenerator",
    ):
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )

        self.comps_variable = input_variable
        self.sample_dim = sample_dim
        self.comps_dim = comps_dim
        self.grid_dim = grid_dim
        self.grid_variable = grid_variable
        self.exclusion_radius = exclusion_radius
        self.exclusion_depth = exclusion_depth

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        excluded_comps = dataset[self.comps_variable]
        n_comps = excluded_comps.sizes[self.comps_dim]

        grid = dataset[self.grid_variable]
        gaussian_points = np.zeros(grid.sizes[self.grid_dim])
        normalization = np.sqrt(
            (2 * np.pi) ** n_comps
            * np.linalg.det(np.eye(n_comps) * self.exclusion_radius)
        )
        for i, coord in excluded_comps.groupby(self.sample_dim, squeeze=False):
            pdf = multivariate_normal.pdf(
                grid, mean=coord.values.squeeze(), cov=self.exclusion_radius
            )
            gaussian_points = (
                gaussian_points + self.exclusion_depth * normalization * pdf
            )

        self.output[self.output_variable] = xr.DataArray(
            gaussian_points, dims=[self.grid_dim]
        )
        self.output[self.output_variable].attrs[
            "description"
        ] = "A field of multidimensional gaussian points."
        return self
