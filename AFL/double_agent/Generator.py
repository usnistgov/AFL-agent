"""
Data generation tools for creating synthetic datasets and sampling spaces.

This module provides classes for generating various types of data structures commonly used
in materials science and machine learning applications. The generators can create regular
grids, compositional spaces, and specialized point distributions.

Key features:
- Cartesian grid generation with flexible specifications
- Barycentric grid generation for compositional spaces
- Gaussian point distributions for exclusion zones
- Support for multi-dimensional spaces
- Integration with xarray data structures
"""

from typing import Dict, List

import numpy as np
import xarray as xr
from itertools import product
from scipy.stats import multivariate_normal  # type: ignore
from typing_extensions import Self

from AFL.double_agent.PipelineOp import PipelineOp


class Generator(PipelineOp):
    """Base class for all data generation operations.
    
    This abstract base class provides common functionality for generating synthetic data
    or sampling spaces. Unlike most PipelineOps, Generators typically don't require
    input data but instead create new data based on parameters.

    Parameters
    ----------
    input_variable : str
        Generators generally do not use input variables but this can be used to name 
        the input node for a generator

    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this 
        `PipelineOp`

    name: str
        The name to use when added to a Pipeline. This name is used when calling 
        Pipeline.search()
    """

    def __init__(
        self,
        output_variable: str,
        input_variable: str = "Generator",
        name: str = "GeneratorBase",
    ) -> None:
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this generator to the supplied dataset.
        
        This method must be implemented by subclasses to define how the data
        generation is performed.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset (typically not used by generators)

        Returns
        -------
        Self
            The generator instance with generated outputs
        """
        return NotImplementedError("Calculate must be implemented in subclasses")  # type: ignore


class CartesianGrid(Generator):
    """Generator that produces a cartesian grid according to user-provided specifications.
    
    Creates a regular grid in N-dimensional space where each dimension can have its own
    min, max, and step size specifications. The resulting grid contains all possible
    combinations of points along each dimension.

    Parameters
    ----------
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset`

    grid_spec : Dict[str, Dict[str, int | float]]
        Dictionary where each top-level key corresponds to a component in the system.
        Each top-level key points to a subdictionary that defines the minimum, maximum,
        and step size for that component with keys: min, max, steps.

    sample_dim : str
        Name of the dimension for different samples/points in the grid

    component_dim : str, default='component'
        Name of the dimension for different components

    name : str, default="CartesianGridGenerator"
        The name to use when added to a Pipeline
    """

    def __init__(
        self,
        output_variable: str,
        grid_spec: Dict[str, Dict[str, int | float]],
        sample_dim: str,
        component_dim: str = 'component',
        name: str = "CartesianGridGenerator",
    ):
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
        """Generate the cartesian grid based on specifications.
        
        Creates a grid by taking the cartesian product of points along each dimension
        as specified in the grid_spec.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset (not used by this generator)

        Returns
        -------
        Self
            The generator instance with the created grid
        """
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
    """Generator that produces a grid in barycentric coordinates.
    
    Creates a grid suitable for compositional spaces where the sum of components
    must equal a fixed value (typically 1.0). The grid is generated by systematically
    sampling points that satisfy the barycentric constraint.

    Parameters
    ----------
    output_variable : str
        The name of the variable to be inserted into the dataset

    components : List[str]
        List of component names for the compositional space

    sample_dim : str
        Name of the dimension for different samples/points

    pts_per_row : int, default=50
        Number of points to sample along each row of the simplex

    basis : float, default=1.0
        The sum constraint for the compositions (typically 1.0)

    dim : int, default=3
        Number of dimensions in the compositional space

    eps : float, default=1e-9
        Small value for numerical stability in equality comparisons

    name : str, default="BarycentricGridGenerator"
        The name to use when added to a Pipeline
    """

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
        """Generate the barycentric grid.
        
        Creates a grid of points that satisfy the barycentric constraint by
        systematically sampling the simplex space.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset (not used by this generator)

        Returns
        -------
        Self
            The generator instance with the created barycentric grid
        """
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
    """Generator that creates Gaussian-distributed points for exclusion zones.
    
    This generator places Gaussian distributions centered at specified points,
    useful for creating exclusion zones or smooth transitions around specific
    locations in the sampling space.

    Parameters
    ----------
    input_variable : str
        The name of the variable containing points to center Gaussians around

    sample_dim : str
        Name of the dimension for different samples/points

    output_variable : str
        The name of the variable to be inserted into the dataset

    grid_variable : str
        The name of the grid variable to evaluate Gaussians on

    grid_dim : str
        Name of the grid dimension

    comps_dim : str, default="component"
        Name of the components dimension

    exclusion_depth : float, default=1e-3
        Maximum value of the Gaussian distributions

    exclusion_radius : float, default=1e-3
        Width parameter for the Gaussian distributions

    name : str, default="GaussianPointsGenerator"
        The name to use when added to a Pipeline
    """

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
        """Generate Gaussian-distributed points.
        
        Places multivariate normal distributions centered at each input point,
        creating a field of Gaussian peaks that can be used for exclusion zones
        or smooth transitions.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing points to center Gaussians around and
            the grid to evaluate them on

        Returns
        -------
        Self
            The generator instance with the created Gaussian field
        """
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
