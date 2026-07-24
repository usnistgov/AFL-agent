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

    grid_spec : Dict[str, Dict[str, int | float]] | None, default=None
        Optional per-component bounds with keys min and max. When omitted, all
        components default to [0.0, 1.0].

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
        grid_spec: Dict[str, Dict[str, int | float]] | None = None,
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
        self.grid_spec = grid_spec or {
            component: {"min": 0.0, "max": 1.0} for component in components
        }
        self.pts_per_row = pts_per_row
        self.basis = basis
        self.dim = dim
        self.eps = eps

        if self.dim != len(self.components):
            raise ValueError("dim must match the number of components")

        missing = [component for component in self.components if component not in self.grid_spec]
        if missing:
            raise ValueError(f"grid_spec missing components: {missing}")

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
        component_specs = [self.grid_spec[component] for component in self.components]
        candidate_axes = [
            np.linspace(spec.get("min", 0.0), spec.get("max", 1.0), self.pts_per_row)
            for spec in component_specs[:-1]
        ]
        last_spec = component_specs[-1]
        last_min = float(last_spec.get("min", 0.0))
        last_max = float(last_spec.get("max", 1.0))

        for i in product(*candidate_axes):
            if sum(i) > (1.0 + self.eps):
                continue

            j = 1.0 - sum(i)

            if j < (0.0 - self.eps):
                continue
            if j < (last_min - self.eps) or j > (last_max + self.eps):
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


def sample_bounded_simplex(
    lower: np.ndarray,
    upper: np.ndarray,
    n_samples: int,
    *,
    basis: float = 1.0,
    alpha: float = 1.0,
    rng: np.random.Generator | None = None,
    eps: float = 1e-9,
) -> np.ndarray:
    """Sample points inside a box-constrained simplex.

    Each sampled point ``x`` satisfies ``lower <= x <= upper`` and
    ``sum(x) == basis`` up to numerical tolerance.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if lower.ndim != 1 or upper.ndim != 1 or lower.shape != upper.shape:
        raise ValueError("lower and upper must be one-dimensional arrays of equal length")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if np.any(lower > upper + eps):
        raise ValueError("Box bounds are invalid: lower bounds exceed upper bounds")

    lower_sum = float(lower.sum())
    upper_sum = float(upper.sum())
    if lower_sum > (basis + eps) or upper_sum < (basis - eps):
        raise ValueError(
            "Box bounds are infeasible: no point can satisfy the bounds and simplex sum"
        )

    rng = np.random.default_rng() if rng is None else rng
    n_dim = lower.shape[0]
    points = np.empty((n_samples, n_dim), dtype=float)

    for sample_idx in range(n_samples):
        remaining = float(basis)
        point = np.empty(n_dim, dtype=float)
        permutation = rng.permutation(n_dim)

        for position, component_idx in enumerate(permutation):
            future_indices = permutation[position + 1 :]
            if future_indices.size == 0:
                value = remaining
            else:
                future_lower = float(lower[future_indices].sum())
                future_upper = float(upper[future_indices].sum())
                min_value = max(float(lower[component_idx]), remaining - future_upper)
                max_value = min(float(upper[component_idx]), remaining - future_lower)

                if max_value < (min_value - eps):
                    raise ValueError(
                        "Box bounds are infeasible: no admissible interval remains while sampling"
                    )

                if (max_value - min_value) <= eps:
                    value = min_value
                else:
                    value = min_value + rng.beta(alpha, alpha) * (max_value - min_value)

            point[component_idx] = value
            remaining -= value

        if np.any(point < (lower - eps)) or np.any(point > (upper + eps)):
            raise ValueError("Failed to sample a point inside the bounded simplex")
        if not np.isclose(point.sum(), basis, atol=max(eps, 1e-12), rtol=0.0):
            raise ValueError("Failed to sample a point that satisfies the simplex sum")

        points[sample_idx] = point

    return points


class RandomBoundedSimplex(Generator):
    """Generator that randomly samples points in a box-constrained simplex."""

    def __init__(
        self,
        output_variable: str,
        components: List[str],
        n_samples: int,
        sample_dim: str,
        grid_spec: Dict[str, Dict[str, int | float]] | None = None,
        basis: float = 1.0,
        alpha: float = 1.0,
        component_dim: str = "component",
        random_seed: int | None = None,
        eps: float = 1e-9,
        name: str = "RandomBoundedSimplexGenerator",
    ):
        super().__init__(
            name=name,
            input_variable=name,
            output_variable=output_variable,
        )
        self.components = components
        self.n_samples = n_samples
        self.sample_dim = sample_dim
        self.grid_spec = grid_spec or {
            component: {"min": 0.0, "max": basis} for component in components
        }
        self.basis = basis
        self.alpha = alpha
        self.component_dim = component_dim
        self.random_seed = random_seed
        self.eps = eps

        missing = [component for component in self.components if component not in self.grid_spec]
        if missing:
            raise ValueError(f"grid_spec missing components: {missing}")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")

    def calculate(self, dataset: xr.Dataset) -> Self:
        lower = np.asarray(
            [float(self.grid_spec[component].get("min", 0.0)) for component in self.components],
            dtype=float,
        )
        upper = np.asarray(
            [float(self.grid_spec[component].get("max", self.basis)) for component in self.components],
            dtype=float,
        )
        rng = np.random.default_rng(self.random_seed)
        pts = sample_bounded_simplex(
            lower=lower,
            upper=upper,
            n_samples=self.n_samples,
            basis=self.basis,
            alpha=self.alpha,
            rng=rng,
            eps=self.eps,
        )
        self.output[self.output_variable] = xr.DataArray(
            pts,
            dims=[self.sample_dim, self.component_dim],
            coords={self.component_dim: self.components},
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
