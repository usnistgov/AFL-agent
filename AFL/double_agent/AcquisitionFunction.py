"""
Acquisition functions for active learning and optimization.

This module provides various acquisition functions that help guide the selection of new samples
in active learning and optimization tasks. Each acquisition function evaluates a set of candidate
points and determines which points would be most valuable to sample next.

Key features:
- Support for single and multi-point acquisition
- Ability to exclude previously sampled points
- Random and maximum value acquisition strategies
- Upper confidence bound acquisition
- Support for multi-modal acquisition with masking

"""

import copy
import warnings
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import xarray as xr
from typing_extensions import Self

from AFL.double_agent.Generator import GaussianPoints
from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.spec import resolve_function
from AFL.double_agent.variables import bind_dataset, find_variables, resolve_variables
from AFL.double_agent.util import listify
from AFL.double_agent.BoTorchMixin import BoTorchMixin, Bounds, ScoreFunctionAcquisition


class AcquisitionError(Exception):
    """Exception raised when an error in the acquisition decision occurs"""

    pass


class AcquisitionFunction(PipelineOp):
    """Base acquisition function for selecting next points to sample.

    Acquisition functions gather one or more inputs and use that information to choose one or more points from a
    supplied composition grid. This base class provides common functionality for implementing specific
    acquisition strategies.

    Parameters
    ----------
    input_variables : List[str]
        The name of the `xarray.Dataset` data variables to extract from the input `xarray.Dataset`

    grid_variable: str
        The name of the `xarray.Dataset` data variable to use as a evaluation grid.

    grid_dim: str
        The xarray dimension over each grid_point. Grid equivalent to sample.

    output_prefix: Optional[str]
        If provided, all outputs of this `PipelineOp` will be prefixed with this string

    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

    decision_rtol: float
        The next sample will be randomly chosen from all grid points that are within `decision_rtol` percent of the
        maximum of the decision surface. This

    excluded_comps_variables: Optional[List[str]]
        A list of `xarray.Dataset` composition variables to use in building an exclusion surface that is added to
        the decision surface. This exclusion surface is built but placing multidimensional inverted Gaussians at
        every composition point specified in the `excluded_comps_variables`. This is done using the `GaussianPoints`
        generator.

    excluded_comps_dim: str
        The `xarray` dimension over the components of a composition.

    exclusion_radius: float
        The width of the Gaussian placed by the `GaussianPoints` generator. See that documentation for more details.

    count: int
        The number of samples to pull from the grid.

    sample_dim: str
        The dimension name of the chosen samples in `output_variable`. Give each acquisition
        function in a pipeline its own `sample_dim` when they select different numbers of points.

    name: str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
    """

    def __init__(
        self,
        input_variables: List[str],
        grid_variable: str,
        grid_dim: str = "grid",
        output_prefix: Optional[str] = None,
        output_variable: str = "next_compositions",
        decision_rtol: float = 0.05,
        random_fraction: float = 0.0,
        excluded_comps_variables: Optional[List[str]] = None,
        excluded_comps_dim: Optional[str] = None,
        exclusion_radius: float = 1e-3,
        count: int = 1,
        sample_dim: str = "AF_sample",
        name: str = "AcquisitionFunctionBase",
    ) -> None:
        super().__init__(
            input_variable=listify(input_variables)
            + [grid_variable],  # add in grid variable for visualization
            output_variable=output_variable,
            name=name,
            output_prefix=output_prefix,
        )

        self.input_variables = input_variables
        self.sample_dim = sample_dim
        self.excluded_comps_variables = excluded_comps_variables
        self.excluded_comps_dim = excluded_comps_dim
        self.count = count
        self.grid_variable = grid_variable
        self.grid_dim = grid_dim
        self.decision_rtol = decision_rtol
        if random_fraction < 0.0 or random_fraction > 1.0:
            raise ValueError(
                f"random_fraction must be within [0, 1], got {random_fraction}."
            )
        self.random_fraction = random_fraction
        self.exclusion_radius = exclusion_radius

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`
        
        This method must be implemented by subclasses to define how the acquisition function
        evaluates and selects points from the grid.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing variables to evaluate

        Returns
        -------
        Self
            The acquisition function instance with updated outputs
        """
        return NotImplementedError(".calculate must be implemented in subclasses")  # type: ignore

    def _get_excluded_comps(self, dataset: xr.Dataset) -> xr.DataArray | None:
        """Gather all compositions listed in excluded_comps_variables into a single array.
        
        This helper method combines all the compositions that should be excluded into a single
        DataArray for use in creating exclusion zones.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing composition variables to exclude

        Returns
        -------
        xr.DataArray | None
            Combined array of compositions to exclude, or None if no exclusions specified
        """
        if self.excluded_comps_variables is None:
            return None

        merge_list = []
        for var in listify(self.excluded_comps_variables):
            d = dataset[var]
            d = d.transpose(..., self.excluded_comps_dim)
            d = d.rename({d.dims[0]: "sample"})

            if d.sizes["sample"] == 0:  # nothing measured yet in this domain
                continue
            merge_list.append(d)

        self.merge_list = merge_list
        if not merge_list:
            return None
        excluded_comps = xr.concat(merge_list, dim="sample")
        excluded_comps = xr.DataArray(
            np.unique(excluded_comps, axis=0), dims=excluded_comps.dims
        )
        return excluded_comps

    def exclude_previous_samples(self, dataset: xr.Dataset) -> xr.Dataset:
        """Modify the decision surface by placing Gaussian exclusion zones.
        
        This method modifies the decision surface by adding Gaussian exclusion zones around
        previously measured compositions to prevent resampling of similar points.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing the decision surface and compositions to exclude

        Returns
        -------
        xr.Dataset
            Modified dataset with updated decision surface including exclusion zones

        Raises
        ------
        AcquisitionError
            If required variables are missing from the dataset
        """
        if (
            ("decision_surface" not in dataset)
            or ("comp_grid" not in dataset)
            or ("excluded_comps" not in dataset)
        ):
            raise AcquisitionError(
                (
                    """Acquisition function must pass .exclude_previous_samples an xarray.Dataset with variables, """
                    f"""'decision_surface' and 'comp_grid' and 'excluded_comps'. """
                    f"""The supplied dataset had: {dataset.keys()}."""
                )
            )

        exclusion_depth = dataset["decision_surface"].max().values[()]
        gp_op = GaussianPoints(
            input_variable="excluded_comps",
            sample_dim="sample",
            comps_dim=self.excluded_comps_dim,
            output_variable="excluded_surface",
            grid_variable="comp_grid",
            grid_dim=self.grid_dim,
            exclusion_radius=self.exclusion_radius,
            exclusion_depth=exclusion_depth,  # type: ignore
        )
        gp_op.calculate(dataset)

        self.output.update(gp_op.output)

        self.output[self._prefix_output("pre_exclude_decision_surface")] = dataset[
            "decision_surface"
        ].copy()
        self.output[self._prefix_output("pre_exclude_decision_surface")].attrs[
            "description"
        ] = "The acquisition surface before it is modified by an exclusion field"

        dataset["decision_surface"] -= gp_op.output["excluded_surface"]
        return dataset

    def get_next_samples(self, dataset: xr.Dataset) -> None:
        """Choose the next compositions by evaluating the decision surface.

        This method finds all compositions that are within decision_rtol of the maximum values
        of the decision surface, then performs epsilon-greedy sampling for each requested pick:
        with probability random_fraction, pick from all valid points; otherwise pick from the
        top decision_rtol set.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing the decision surface and composition grid

        Raises
        ------
        AcquisitionError
            If required variables are missing or if no valid points are found
        """
        if ("decision_surface" not in dataset) or ("comp_grid" not in dataset):
            raise AcquisitionError(
                (
                    """Acquisition function must pass .get_next_sample an xarray.Dataset with variables, """
                    f"""'decision_surface' and 'comp_grid'. The supplied dataset had: {list(dataset.keys())}."""
                )
            )

        # need to add integer 'coords' to the grid dimension. These allow us to index the parent
        # array after finding the optimal next samples
        dataset = dataset.assign_coords(
            {self.grid_dim: np.arange(dataset.sizes[self.grid_dim])}
        )

        decision_values = dataset["decision_surface"].values
        valid_mask = np.isfinite(decision_values)
        if not np.any(valid_mask):
            raise AcquisitionError(
                "Decision surface does not contain any finite values."
            )

        all_indices = dataset[self.grid_dim].values
        valid_indices = all_indices[valid_mask]
        if len(valid_indices) < self.count:
            raise AcquisitionError(
                (
                    "Unable to find enough valid gridpoints in decision surface to "
                    f"sample {self.count} points."
                )
            )

        # find indices of all samples within self.decision_rtol of the maximum
        close_mask = np.isclose(
            decision_values,
            np.max(decision_values[valid_mask]),
            rtol=self.decision_rtol,
            atol=0,
        )
        close_mask &= valid_mask
        close_indices = all_indices[close_mask]

        all_pool = set(valid_indices.tolist())
        top_pool = set(close_indices.tolist())

        next_indices = []
        for _ in range(self.count):
            if not all_pool:
                raise AcquisitionError(
                    "Unable to find enough valid gridpoints to satisfy requested sample count."
                )

            choose_random = np.random.random() < self.random_fraction
            pool = all_pool if choose_random else top_pool
            if not pool:
                pool = all_pool if all_pool else top_pool
            if not pool:
                raise AcquisitionError(
                    (
                        "Unable to find gridpoint in decision surface that satisfies all constraints. "
                        f"This can occur when acquisition_rtol (currently {self.decision_rtol}) is "
                        f"too low or exclusion_radius (currently {self.exclusion_radius}) is too high."
                    )
                )

            selected_index = int(np.random.choice(list(pool)))
            next_indices.append(selected_index)
            all_pool.discard(selected_index)
            top_pool.discard(selected_index)

        next_samples = dataset.sel({self.grid_dim: next_indices}).comp_grid
        next_samples = next_samples.rename({self.grid_dim: self.sample_dim})
        next_samples = next_samples.reset_index(self.sample_dim, drop=True)

        self.output[self.output_variable] = (
            next_samples  # should already be a xr.DataArray
        )


class RandomAF(AcquisitionFunction):
    """Randomly choose points from the grid with optional exclusion of previous measurements.
    
    This acquisition function implements a simple random sampling strategy where points are chosen
    uniformly at random from the available grid points. It can optionally exclude previously
    measured points using Gaussian exclusion zones.

    Parameters
    ----------
    grid_variable: str
        The name of the `xarray.Dataset` data variable to use as a evaluation grid.

    grid_dim: str
        The xarray dimension over each grid_point. Grid equivalent to sample.

    output_prefix: Optional[str]
        If provided, all outputs of this `PipelineOp` will be prefixed with this string

    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

    decision_rtol: float
        The next sample will be randomly chosen from all grid points that are within `decision_rtol` percent of the
        maximum of the decision surface. This

    excluded_comps_variables: Optional[List[str]]
        A list of `xarray.Dataset` composition variables to use in building an exclusion surface that is added to
        the decision surface. This exclusion surface is built but placing multidimensional inverted Gaussians at
        every composition point specified in the `excluded_comps_variables`. This is done using the `GaussianPoints`
        generator.

    excluded_comps_dim: str
        The `xarray` dimension over the components of a composition.

    exclusion_radius: float
        The width of the Gaussian placed by the `GaussianPoints` generator. See that documentation for more details.

    count: int
        The number of samples to pull from the grid.

    name: str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
    """

    def __init__(
        self,
        grid_variable: str,
        grid_dim: str = "grid",
        output_prefix: Optional[str] = None,
        output_variable: str = "next_samples",
        decision_rtol: float = 0.05,
        random_fraction: float = 0.0,
        excluded_comps_variables: Optional[str] = None,
        excluded_comps_dim: Optional[str] = None,
        exclusion_radius: float = 1e-3,
        count: int = 1,
        name: str = "RandomAF",
    ) -> None:
        super().__init__(
            input_variables=[],
            grid_variable=grid_variable,
            grid_dim=grid_dim,
            output_prefix=output_prefix,
            output_variable=output_variable,
            decision_rtol=decision_rtol,
            random_fraction=random_fraction,
            excluded_comps_variables=excluded_comps_variables,
            excluded_comps_dim=excluded_comps_dim,
            exclusion_radius=exclusion_radius,
            count=count,
            name=name,
        )

        # this doesn't need to be an attribute, just convenient for debugging
        self.acquisition = None

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this RandomAF to the supplied dataset.
        
        Creates a random decision surface and optionally applies exclusion zones
        around previously measured points.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing the grid to sample from

        Returns
        -------
        Self
            The RandomAF instance with updated outputs
        """
        self.acquisition = xr.Dataset()

        self.acquisition["comp_grid"] = dataset[self.grid_variable].transpose(
            self.grid_dim, ...
        )

        self.acquisition["decision_surface"] = xr.DataArray(
            np.random.random(size=self.acquisition["grid"].values.shape),
            dims=self.grid_dim,
        )

        excluded_comps = self._get_excluded_comps(dataset)
        if excluded_comps is not None:
            self.acquisition["excluded_comps"] = self._get_excluded_comps(dataset)
            self.exclude_previous_samples(self.acquisition)

        self.output[self._prefix_output("decision_surface")] = self.acquisition[
            "decision_surface"
        ]
        self.output[self._prefix_output("decision_surface")].attrs[
            "description"
        ] = "The final acquisition surface that is evaluated to determine the next_sample"

        self.get_next_samples(self.acquisition)

        return self


class MultimodalMask_MaxValueAF(AcquisitionFunction):
    """Acquisition function that selects points based on maximum values within specific phase regions.
    
    This acquisition function extends MaxValueAF by adding the ability to focus sampling in specific
    phase regions of the composition space. It uses a mask label variable to identify different
    phase regions and selects points based on their proximity to target phase coordinates.

    Parameters
    ----------
    decision_variable : str
        The name of the variable containing the decision surface values
    
    mask_label_variable : str
        The name of the variable containing phase region labels
    
    phase_select_coords : dict[str,float]
        Dictionary mapping phase component names to target coordinates
    
    grid_variable : str
        The name of the `xarray.Dataset` data variable to use as a evaluation grid
    
    grid_dim : str, default="grid"
        The xarray dimension over each grid_point. Grid equivalent to sample.
    
    combine_coeffs : Optional[List[float]], default=None
        If provided, the input variables will be scaled by these coefficients before being summed
    
    output_prefix : Optional[str], default=None
        If provided, all outputs of this `PipelineOp` will be prefixed with this string
    
    output_variable : str, default="next_samples"
        The name of the variable to be inserted into the dataset
    
    decision_rtol : float, default=0.05
        The next sample will be randomly chosen from all grid points that are within
        decision_rtol percent of the maximum of the decision surface
    
    excluded_comps_variables : Optional[List[str]], default=None
        List of composition variables to use in building an exclusion surface
    
    excluded_comps_dim : Optional[str], default=None
        The dimension over the components of a composition
    
    exclusion_radius : float, default=1e-3
        The width of the Gaussian placed by the GaussianPoints generator
    
    count : int, default=1
        The number of samples to pull from the grid
    
    name : str, default="MaxValueAF"
        The name to use when added to a Pipeline
    """

    def __init__(
            self,
            decision_variable: str,
            mask_label_variable: str,
            phase_select_coords: dict[str,float],
            grid_variable: str,
            grid_dim: str = "grid",
            combine_coeffs: Optional[List[float]] = None,
            output_prefix: Optional[str] = None,
            output_variable: str = "next_samples",
            decision_rtol: float = 0.05,
            random_fraction: float = 0.0,
            excluded_comps_variables: Optional[List[str]] = None,
            excluded_comps_dim: Optional[str] = None,
            exclusion_radius: float = 1e-3,
            count: int = 1,
            name: str = "MaxValueAF",
    ) -> None:
        super().__init__(
            input_variables=[mask_label_variable,decision_variable],
            grid_variable=grid_variable,
            grid_dim=grid_dim,
            output_prefix=output_prefix,
            output_variable=output_variable,
            decision_rtol=decision_rtol,
            random_fraction=random_fraction,
            excluded_comps_variables=excluded_comps_variables,
            excluded_comps_dim=excluded_comps_dim,
            exclusion_radius=exclusion_radius,
            count=count,
            name=name,
        )
        self.combine_coeffs = combine_coeffs
        self.decision_variable = decision_variable
        self.mask_label_variable = mask_label_variable
        self.phase_select_coords = phase_select_coords

        # this doesn't need to be an attribute, just convenient for debugging
        self.acquisition = None

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this MultimodalMask_MaxValueAF to the supplied dataset.
        
        Creates a decision surface by combining the decision variable with phase region
        masking based on proximity to target phase coordinates.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing variables to evaluate

        Returns
        -------
        Self
            The MultimodalMask_MaxValueAF instance with updated outputs
        """
        self.acquisition = xr.Dataset()

        self.acquisition["comp_grid"] = dataset[self.grid_variable].transpose(
            self.grid_dim, ...
        )

        decision_surface = dataset[self.decision_variable].copy()
        if (decision_surface.max() - decision_surface.min()) < 1e-16:
            pass
        else:  # normalize
            decision_surface = (decision_surface - decision_surface.min()) / (
                    decision_surface.max() - decision_surface.min()
            )

        # Apply phase region masking
        if len(np.unique(dataset[self.mask_label_variable]))>1:
            label_com = self.acquisition["comp_grid"].groupby(dataset[self.mask_label_variable]).mean()
            component_dim = self.acquisition["comp_grid"].dims[1] # should always be [1] due to the transpose above
            phase_select_coords = xr.Dataset(self.phase_select_coords).to_array(component_dim)
            diff = (label_com - phase_select_coords).pipe(np.square).sum(component_dim).pipe(np.sqrt)
            selected_label = label_com[diff.argmin()][self.mask_label_variable].values[()]
            mask = dataset[self.mask_label_variable] == selected_label
            decision_surface[~mask] = 0 #set all non-selected values on the decision surface to 0

        self.acquisition["decision_surface"] = decision_surface

        excluded_comps = self._get_excluded_comps(dataset)
        if excluded_comps is not None:
            self.acquisition["excluded_comps"] = self._get_excluded_comps(dataset)
            self.exclude_previous_samples(self.acquisition)

        self.output[self._prefix_output("decision_surface")] = self.acquisition[
            "decision_surface"
        ]
        self.output[self._prefix_output("decision_surface")].attrs[
            "description"
        ] = "The final acquisition surface that is evaluated to determine the next_sample"

        self.get_next_samples(self.acquisition)

        return self


class MaxValueAF(AcquisitionFunction):
    """Acquisition function that selects points based on maximum values.
    
    This acquisition function chooses points by finding the maximum values in the decision surface.
    It can combine multiple input variables with optional scaling coefficients and supports
    exclusion of previously measured points.

    Parameters
    ----------
    input_variables : List[str]
        The name of the `xarray.Dataset` data variables to extract from the input `xarray.Dataset`

    grid_variable: str
        The name of the `xarray.Dataset` data variable to use as a evaluation grid.

    grid_dim: str
        The xarray dimension over each grid_point. Grid equivalent to sample.

    combine_coeffs: Optional[List[float]]
        If provided, the `self.input_variables` will be scaled by these coefficients before being summed.

    output_prefix: Optional[str]
        If provided, all outputs of this `PipelineOp` will be prefixed with this string

    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

    decision_rtol: float
        The next sample will be randomly chosen from all grid points that are within `decision_rtol` percent of the
        maximum of the decision surface. This

    excluded_comps_variables: Optional[List[str]]
        A list of `xarray.Dataset` composition variables to use in building an exclusion surface that is added to
        the decision surface. This exclusion surface is built but placing multidimensional inverted Gaussians at
        every composition point specified in the `excluded_comps_variables`. This is done using the `GaussianPoints`
        generator.

    excluded_comps_dim: str
        The `xarray` dimension over the components of a composition.

    exclusion_radius: float
        The width of the Gaussian placed by the `GaussianPoints` generator. See that documentation for more details.

    count: int
        The number of samples to pull from the grid.

    sample_dim: str
        The dimension name of the chosen samples in `output_variable` (default "AF_sample").

    name: str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
    """

    def __init__(
        self,
        input_variables: List[str],
        grid_variable: str,
        grid_dim: str = "grid",
        combine_coeffs: Optional[List[float]] = None,
        output_prefix: Optional[str] = None,
        output_variable: str = "next_samples",
        decision_rtol: float = 0.05,
        random_fraction: float = 0.0,
        excluded_comps_variables: Optional[List[str]] = None,
        excluded_comps_dim: Optional[str] = None,
        exclusion_radius: float = 1e-3,
        count: int = 1,
        sample_dim: str = "AF_sample",
        name: str = "MaxValueAF",
    ) -> None:
        super().__init__(
            input_variables=input_variables,
            grid_variable=grid_variable,
            grid_dim=grid_dim,
            output_prefix=output_prefix,
            output_variable=output_variable,
            decision_rtol=decision_rtol,
            random_fraction=random_fraction,
            excluded_comps_variables=excluded_comps_variables,
            excluded_comps_dim=excluded_comps_dim,
            exclusion_radius=exclusion_radius,
            count=count,
            sample_dim=sample_dim,
            name=name,
        )
        self.combine_coeffs = combine_coeffs

        # this doesn't need to be an attribute, just convenient for debugging
        self.acquisition = None

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this MaxValueAF to the supplied dataset.
        
        Combines multiple input variables with optional scaling coefficients to create
        a decision surface based on maximum values.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing variables to evaluate

        Returns
        -------
        Self
            The MaxValueAF instance with updated outputs
        """
        self.acquisition = xr.Dataset()

        self.acquisition["comp_grid"] = dataset[self.grid_variable].transpose(
            self.grid_dim, ...
        )

        # make sure it's actually a list, listify will pass through tuples
        input_variables = list(listify(self.input_variables))

        if self.combine_coeffs is not None:
            assert len(self.combine_coeffs) == len(input_variables)
            coeffs = copy.deepcopy(self.combine_coeffs)
        else:
            coeffs = [1] * len(input_variables)

        decision_surface = coeffs.pop(0) * dataset[input_variables.pop(0)].copy()
        for coeff, input_variable in zip(coeffs, input_variables):
            decision_surface += coeff * dataset[input_variable]

        if (decision_surface.max() - decision_surface.min()) < 1e-16:
            pass
        else:  # normalize
            decision_surface = (decision_surface - decision_surface.min()) / (
                decision_surface.max() - decision_surface.min()
            )
        self.acquisition["decision_surface"] = decision_surface

        excluded_comps = self._get_excluded_comps(dataset)
        if excluded_comps is not None:
            self.acquisition["excluded_comps"] = self._get_excluded_comps(dataset)
            self.exclude_previous_samples(self.acquisition)

        self.output[self._prefix_output("decision_surface")] = self.acquisition[
            "decision_surface"
        ]
        self.output[self._prefix_output("decision_surface")].attrs[
            "description"
        ] = "The final acquisition surface that is evaluated to determine the next_sample"

        self.get_next_samples(self.acquisition)

        return self


class PseudoUCB(AcquisitionFunction):
    """Upper Confidence Bound (UCB) acquisition function.
    
    This acquisition function implements a pseudo Upper Confidence Bound strategy where
    points are selected based on a weighted combination of mean predictions and uncertainty
    estimates. The weights (lambdas) control the exploration-exploitation trade-off.

    Parameters
    ----------
    input_variables : List[str]
        The name of the `xarray.Dataset` data variables to extract from the input `xarray.Dataset`

    grid_variable: str
        The name of the `xarray.Dataset` data variable to use as a evaluation grid.

    grid_dim: str
        The xarray dimension over each grid_point. Grid equivalent to sample.

    lambdas: List[float]
        Scaling parameters for each input variable to control exploration-exploitation trade-off

    output_prefix: Optional[str]
        If provided, all outputs of this `PipelineOp` will be prefixed with this string

    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

    decision_rtol: float
        The next sample will be randomly chosen from all grid points that are within `decision_rtol` percent of the
        maximum of the decision surface. This

    excluded_comps_variables: Optional[List[str]]
        A list of `xarray.Dataset` composition variables to use in building an exclusion surface that is added to
        the decision surface. This exclusion surface is built but placing multidimensional inverted Gaussians at
        every composition point specified in the `excluded_comps_variables`. This is done using the `GaussianPoints`
        generator.

    excluded_comps_dim: str
        The `xarray` dimension over the components of a composition.

    exclusion_radius: float
        The width of the Gaussian placed by the `GaussianPoints` generator. See that documentation for more details.

    count: int
        The number of samples to pull from the grid.

    name: str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
    """

    def __init__(
        self,
        input_variables: List[str],
        grid_variable: str,
        grid_dim: str = "grid",
        lambdas=None,
        output_prefix: Optional[str] = None,
        output_variable: str = "next_samples",
        decision_rtol: float = 0.05,
        random_fraction: float = 0.0,
        excluded_comps_variables: Optional[List[str]] = None,
        excluded_comps_dim: Optional[str] = None,
        exclusion_radius: float = 1e-3,
        count: int = 1,
        name: str="PseudoUCB",
    ) -> None:

        super().__init__(
            input_variables=input_variables,
            grid_variable=grid_variable,
            grid_dim=grid_dim,
            output_prefix=output_prefix,
            output_variable=output_variable,
            decision_rtol=decision_rtol,
            random_fraction=random_fraction,
            excluded_comps_variables=excluded_comps_variables,
            excluded_comps_dim=excluded_comps_dim,
            exclusion_radius=exclusion_radius,
            count=count,
            name=name,
        )

        if len(lambdas) != len(input_variables):
            raise ValueError(
                (
                    f"""there are not the same number 'lambda' scaling params to 'input_variables', """
                    f"""check the inputs:   {len(lambdas)} to {len(input_variables)}"""
                )
            )

        # this doesn't need to be an attribute, just convenient for debugging
        self.acquisition = None

        # might make this a list corresponding to the input variables for more complicated
        self.lambdas = lambdas

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this PseudoUCB to the supplied dataset.
        
        Creates a decision surface by combining input variables with lambda weights
        to balance exploration and exploitation.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing variables to evaluate

        Returns
        -------
        Self
            The PseudoUCB instance with updated outputs
        """
        self.acquisition = xr.Dataset()

        self.acquisition["comp_grid"] = dataset[self.grid_variable].transpose(
            self.grid_dim, ...
        )

        # make sure it's actually a list, listify will pass through tuples
        input_variables = list(listify(self.input_variables))

        # this might work and do what I wanted with the pUCB lamdas
        decision_surface = self.lambdas.pop(0) * dataset[input_variables.pop(0)].copy()
        for lamb, input_variable in zip(self.lambdas, input_variables):
            decision_surface += lamb * dataset[input_variable]

        # #is this necessary?
        # decision_surface = (decision_surface - decision_surface.min()) / (
        #             decision_surface.max() - decision_surface.min())
        self.acquisition["decision_surface"] = decision_surface

        excluded_comps = self._get_excluded_comps(dataset)
        if excluded_comps is not None:
            self.acquisition["excluded_comps"] = self._get_excluded_comps(dataset)
            self.exclude_previous_samples(self.acquisition)

        self.output[self._prefix_output("decision_surface")] = self.acquisition[
            "decision_surface"
        ]
        self.output[self._prefix_output("decision_surface")].attrs[
            "description"
        ] = "The final acquisition surface that is evaluated to determine the next_sample"

        self.get_next_samples(self.acquisition)
        return self


class BoTorchAcquisition(BoTorchMixin, AcquisitionFunction):
    r"""Optimize an acquisition score over the design space with BoTorch.

    The score is evaluated on the candidate grid (written to ``decision_surface``) and
    then maximized continuously within ``bounds`` with ``optimize_acqf``, optionally
    constrained to the simplex. By default the score is BoTorch's LogEI (``count == 1``)
    or qLogEI (``count > 1``) on a Gaussian-process surrogate; ``acquisition`` replaces
    it with a user supplied function or any BoTorch acquisition class.

    This op never fits a model. Surrogates come from an upstream regressor or classifier
    that saved its model (``model_uri`` on
    :class:`~AFL.double_agent.PyTorchExtrapolator.BoTorchRegressor` or
    :class:`~AFL.double_agent.PyTorchExtrapolator.DirichletGPExtrapolator`); point
    ``model_variable`` at the ref it wrote and ``best_f_variable`` at its ``best_f``.

    Parameters
    ----------
    feature_input_variable : str
        Name of the feature variable; its ``component`` coordinate labels the output points.
    grid_variable : str
        Name of the candidate grid variable.
    bounds : Bounds
        Optimization bounds for ``optimize_acqf``, required. Either the per-component
        format used by `BarycentricGrid` and `CartesianGrid`,
        ``{component: {"min": lower, "max": upper}}``, a mapping of component to
        ``(lower, upper)``, or an ordered sequence of ``(lower, upper)`` pairs. Bounds
        must be expressed in the same feature space as `feature_input_variable` and
        `grid_variable`.
    grid_dim : str, default="grid"
        Dimension indexing candidate grid points.
    model_variable : str, optional
        Dataset variable holding a model ref written by an upstream regressor or
        classifier (e.g. ``"bayesopt_model"``). Required when the acquisition needs a
        model: the built-in LogEI/qLogEI, a BoTorch acquisition class, or a score that
        references ``Variable("model")``.
    best_f_variable : str, optional
        Dataset variable containing the incumbent objective value in original units
        (e.g. ``"bayesopt_best_f"`` from :class:`BoTorchRegressor`). Required when the
        acquisition needs ``best_f``: the built-ins, or a score referencing
        ``Variable("best_f")``.
    objective_direction : str, default="maximize"
        Direction the upstream regressor was configured with. The regressor fits the GP
        to negated targets when minimizing, so ``best_f`` is negated here to match the
        ``model``'s maximization space. Acquisition scores are always maximized.
    output_prefix : str, optional
        Prefix applied to generated outputs.
    output_variable : str, default="next_samples"
        Name of the selected next-sample output variable.
    decision_rtol : float, default=0.05
        Relative tolerance used by the base acquisition class.
    excluded_comps_variables : list[str], optional
        Previously sampled compositions to exclude.
    excluded_comps_dim : str, optional
        Component dimension for excluded compositions.
    exclusion_radius : float, default=1e-3
        Radius used for exclusion masking.
    count : int, default=1
        Number of points ``q`` to optimize jointly.
    acquisition : callable | str | dict | type, optional
        What to maximize. ``None`` (default) selects LogEI or qLogEI from ``count``;
        the strings ``"logei"`` and ``"qlogei"`` select one explicitly.

        **A Python callable.** Any callable, including a lambda, closure or
        ``functools.partial``, called as ``acquisition(X)`` where ``X`` is a
        ``torch.Tensor`` of shape ``(count, n_components)``: one candidate set of
        ``count`` design points in the feature space of ``grid_variable``, without a
        batch dimension. It must return a single scalar (tensor, NumPy scalar or
        float), higher being better. Torch functions are optimized with exact
        gradients; NumPy functions are detected automatically and optimized with
        finite differences. Everything else the score needs is ordinary Python: close
        over settings and objects directly, and use
        :class:`~AFL.double_agent.variables.Variable` placeholders for values that live
        in the dataset at calculate time. ``Variable("model")`` and
        ``Variable("best_f")`` resolve to the model loaded from ``model_variable`` and
        the incumbent from ``best_f_variable`` (in maximization space);
        ``Variable("dataset")`` to the input ``xarray.Dataset``. A ``Variable`` naming
        any other dataset variable (e.g. a classifier's ``"phase_model"`` ref) resolves
        from the dataset directly and is added to the op's ``input_variable``.

        Lambdas and closures are kept in memory and exported as ``{"ref": ...}`` in the
        pipeline JSON; see :class:`~AFL.double_agent.PipelineOp.PipelineOp`. A
        module-level function is stored by import path and can be passed as a string.
        ``{"function": f, "args": {...}}`` is accepted too: ``args`` are passed as
        keyword arguments (``Variable`` values inside are resolved first).

        **A BoTorch acquisition class.** ``{"class": UpperConfidenceBound, "args":
        {"beta": 2.0}}`` or an import path string. The class is constructed at
        calculate time with ``model=`` the loaded model plus ``args``; use
        ``"best_f": Variable("best_f")`` in ``args`` for classes that need the incumbent.
    num_restarts : int, default=10
        Number of L-BFGS-B restarts used by ``optimize_acqf``.
    raw_samples : int, default=128
        Number of raw samples used to seed the restarts.
    is_simplex : bool, default=False
        When True, acquisition optimization is constrained to the simplex.

        Notes
        -----
        The simplex is mathematically defined by the equality
        :math:`\sum_i x_i = 1` together with non-negativity. BoTorch's optimizer
        interface consumes linear inequality constraints, so the equality is
        represented as the equivalent pair
        :math:`\sum_i x_i \ge 1` and :math:`-\sum_i x_i \ge -1`.
    name : str, default="BoTorchAcquisition"
        Pipeline operation name.

    Examples
    --------
    Maximize a user-defined score over the unit cube, no model needed::

        def closeness_to_target(X, target=0.3):
            return -((X - target) ** 2).sum()

        op = BoTorchAcquisition(
            feature_input_variable="composition",
            grid_variable="composition_grid",
            acquisition=lambda X: closeness_to_target(X, target=0.25),
            bounds={"a": {"min": 0, "max": 1}, "b": {"min": 0, "max": 1}, "c": {"min": 0, "max": 1}},
        )

    LogEI on the surrogate fit by an upstream regressor::

        with Pipeline() as p:
            BoTorchRegressor(..., output_prefix="gp", model_uri="/tmp/models")
            BoTorchAcquisition(..., model_variable="gp_model", best_f_variable="gp_best_f")

    A score that uses the surrogate and a dataset variable::

        from AFL.double_agent.variables import Variable

        gp, observed = Variable("model"), Variable("composition")

        def ucb_away_from_observed(X, model, observed, beta=2.0):
            posterior = model.posterior(X)
            return (posterior.mean + beta * posterior.variance.sqrt()).sum() + torch.cdist(X, observed).min()

        op = BoTorchAcquisition(..., model_variable="gp_model",
                                acquisition=lambda X: ucb_away_from_observed(X, gp, observed, beta=1.0))

    A BoTorch acquisition class::

        op = BoTorchAcquisition(..., model_variable="gp_model",
                                acquisition={"class": "botorch.acquisition.UpperConfidenceBound",
                                             "args": {"beta": 2.0}})
    """

    BUILTIN_ACQUISITIONS = ("logei", "qlogei")
    PROVIDED_KEYS = ("model", "best_f", "dataset")  # Variable names the op itself provides

    def __init__(
        self,
        feature_input_variable: str,
        grid_variable: str,
        bounds: Bounds,
        grid_dim: str = "grid",
        model_variable: Optional[str] = None,
        best_f_variable: Optional[str] = None,
        objective_direction: str = "maximize",
        output_prefix: Optional[str] = None,
        output_variable: str = "next_samples",
        decision_rtol: float = 0.05,
        excluded_comps_variables: Optional[List[str]] = None,
        excluded_comps_dim: Optional[str] = None,
        exclusion_radius: float = 1e-3,
        count: int = 1,
        acquisition: Optional[Union[Callable, str, dict, type]] = None,
        num_restarts: int = 10,
        raw_samples: int = 128,
        is_simplex: bool = False,
        name: str = "BoTorchAcquisition",
    ) -> None:
        input_variables = [feature_input_variable]
        input_variables += [v for v in (model_variable, best_f_variable) if v is not None]
        super().__init__(
            input_variables=input_variables,
            grid_variable=grid_variable,
            grid_dim=grid_dim,
            output_prefix=output_prefix,
            output_variable=output_variable,
            decision_rtol=decision_rtol,
            excluded_comps_variables=excluded_comps_variables,
            excluded_comps_dim=excluded_comps_dim,
            exclusion_radius=exclusion_radius,
            count=count,
            name=name,
        )
        self.feature_input_variable = feature_input_variable
        self.objective_direction = objective_direction
        self.acquisition = acquisition
        self.best_f_variable = best_f_variable
        self._init_botorch(
            bounds=bounds,
            is_simplex=is_simplex,
            model_variable=model_variable,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            require_bounds=True,
        )
        # Populated by calculate(): the acquisition module built from ``self.acquisition``
        # (which only holds the JSON spec), kept for inspection.
        self.acquisition_function = None
        self._provided = {}

    def _parse_acquisition(self):
        """Return ``(kind, target, args)`` with kind in {"builtin", "class", "function"}."""
        spec = self.acquisition
        if spec is None:
            spec = "qlogei" if self.count > 1 else "logei"
        if isinstance(spec, str) and spec.lower() in self.BUILTIN_ACQUISITIONS:
            return "builtin", spec.lower(), {}
        if isinstance(spec, dict):
            target = spec.get("class", spec.get("function"))
            if target is None:
                raise ValueError(
                    f"{self.name}: acquisition spec needs a 'function' or 'class' key, got keys {sorted(spec)}"
                )
            target = target if callable(target) else resolve_function(target)
            args = dict(spec.get("args") or {})
        else:
            target, args = resolve_function(spec), {}
        return ("class" if isinstance(target, type) else "function"), target, args

    def _provide(self, dataset: xr.Dataset, needed: set) -> None:
        """Fill ``self._provided`` with the upstream ``model`` / ``best_f`` that the acquisition needs."""
        self._provided = {}
        if "model" in needed:
            if self.model_variable is None:
                raise AcquisitionError(
                    f"{self.name}: this acquisition needs a surrogate model but no `model_variable` was given. "
                    "Fit the GP in an upstream regressor/classifier with `model_uri` set and pass the name of "
                    "its model ref output here (e.g. model_variable='bayesopt_model')."
                )
            self._provided["model"] = self.load_model(dataset)
        if "best_f" in needed:
            if self.best_f_variable is None or self.best_f_variable not in dataset:
                raise AcquisitionError(
                    f"{self.name}: this acquisition needs `best_f` but `best_f_variable` "
                    f"{self.best_f_variable!r} is not set or not in the dataset. Pass the upstream regressor's "
                    "best_f output (e.g. best_f_variable='bayesopt_best_f')."
                )
            best_f = float(dataset[self.best_f_variable].values[()])
            # the regressor trains on negated targets when minimizing; match that space
            self._provided["best_f"] = -best_f if self.objective_direction == "minimize" else best_f

    def build_acquisition(self, dataset: xr.Dataset):
        """Build the torch module that is evaluated on the grid and maximized by BoTorch.

        Dispatches on ``self.acquisition``: ``None`` / built-in name -> BoTorch LogEI or
        qLogEI; callable (or function spec) -> :class:`ScoreFunctionAcquisition`; class
        spec -> that class constructed with ``model`` and ``args``. ``model`` and
        ``best_f`` are taken from ``model_variable`` / ``best_f_variable``, never fit here.
        Sets ``self._provided``, the objects handed to Variables during ``calculate``.
        """
        kind, target, args = self._parse_acquisition()
        if kind == "builtin":
            self._provide(dataset, {"model", "best_f"})
            return self.build_builtin_acquisition(target, model=self._provided["model"], best_f=self._provided["best_f"])

        referenced = set(find_variables([target, args])) & {"model", "best_f"}
        # a Variable("model") that is also a real dataset variable resolves from the dataset
        needed = {name for name in referenced if name not in dataset}
        if kind == "class":
            needed.add("model")
        self._provide(dataset, needed)

        with bind_dataset(dataset, self._provided):
            kwargs = resolve_variables(args)
        if kind == "class":  # a BoTorch acquisition class
            kwargs.setdefault("model", self._provided["model"])
            return target(**kwargs)
        return ScoreFunctionAcquisition(target, kwargs=kwargs)

    def optimize_acquisition_function(
        self,
        acq_function,
        bounds: torch.Tensor,
        inequality_constraints=None,
        with_grad: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Maximize ``acq_function`` over ``bounds`` for ``self.count`` joint points.

        Thin wrapper over :meth:`BoTorchMixin.optimize_acqf` that fixes ``q`` to ``count``;
        kept as a method so subclasses can swap the optimizer.
        """
        return self.optimize_acqf(
            acq_function,
            bounds=bounds,
            q=self.count,
            constraints=inequality_constraints,
            with_grad=with_grad,
        )

    def calculate(self, dataset: xr.Dataset) -> Self:
        self.acquisition_function = self.build_acquisition(dataset)
        candidate_x = self.candidates(dataset)

        # Variables inside the score resolve against this dataset for the whole optimization.
        with bind_dataset(dataset, self._provided):
            # Evaluate the score on the grid for the decision surface. For score functions this
            # also detects whether autograd gradients are available.
            acq_values = self.evaluate_acquisition_on_grid(
                self.acquisition_function,
                candidate_x.detach().requires_grad_(True),
                q=self.count,
            )
            with_grad = getattr(self.acquisition_function, "differentiable", None) is not False

            grid = dataset[self.grid_variable]
            acq_numpy = acq_values.detach().cpu().numpy().reshape(-1)
            decision_surface = xr.DataArray(
                acq_numpy.reshape(grid.shape[:-1]) if grid.ndim > 1 else acq_numpy,
                dims=grid.dims[:-1],
                coords={dim: grid.coords[dim] for dim in grid.dims[:-1] if dim in grid.coords},
            )
            self.output[self._prefix_output("decision_surface")] = decision_surface
            self.output[self._prefix_output("decision_surface")].attrs["description"] = (
                "Acquisition function values evaluated on the grid"
            )

            optimization_bounds = self.bounds_tensor(dataset)
            try:
                optimized_x, _ = self.optimize_acquisition_function(
                    self.acquisition_function,
                    bounds=optimization_bounds,
                    inequality_constraints=self.constraints(candidate_x.shape[-1]),
                    with_grad=with_grad,
                )
            except RuntimeError:
                warnings.warn(
                    "BoTorch acquisition optimization failed; falling back to the best evaluated grid point(s).",
                    stacklevel=2,
                )
                return self._fallback_to_grid(dataset, decision_surface)

        optimized_points = optimized_x.detach().cpu().numpy()
        if optimized_points.ndim == 1:
            optimized_points = optimized_points[np.newaxis, :]

        component_coords = dataset[self.feature_input_variable].coords.get("component", None)
        self.output[self.output_variable] = xr.DataArray(
            optimized_points,
            dims=("next_sample", "component"),
            coords={
                "component": component_coords.values
                if component_coords is not None
                else range(optimized_points.shape[1])
            },
        )
        self.output[self.output_variable].attrs["description"] = (
            "Next sample point(s) found by maximizing the acquisition score with BoTorch"
        )
        return self

    def _fallback_to_grid(self, dataset: xr.Dataset, decision_surface: xr.DataArray) -> Self:
        """Pick ``count`` grid points from ``decision_surface`` with the base-class machinery."""
        acquisition_dataset = xr.Dataset(
            {
                "decision_surface": decision_surface,
                "comp_grid": dataset[self.grid_variable].transpose(self.grid_dim, ...),
            }
        )
        excluded_comps = self._get_excluded_comps(dataset)
        if excluded_comps is not None:
            acquisition_dataset["excluded_comps"] = excluded_comps
            self.exclude_previous_samples(acquisition_dataset)
            self.output[self._prefix_output("decision_surface")] = acquisition_dataset["decision_surface"]
        self.get_next_samples(acquisition_dataset)
        return self


class BoTorchPosteriorAcquisition(BoTorchMixin, PipelineOp):
    r"""Optimize the posterior mean of a BoTorch GP to locate the best predicted point.

    The surrogate is loaded from the ref an upstream
    :class:`~AFL.double_agent.PyTorchExtrapolator.BoTorchRegressor` saved (``model_uri``
    there, ``model_variable`` here); this op never fits a model.

    Parameters
    ----------
    feature_input_variable : str
        Name of the feature variable (compositions); its ``component`` coordinate labels ``best_x``.
    grid_variable : str
        Name of the candidate grid variable. Used for the fallback when optimization fails.
    bounds : Bounds
        Explicit optimization bounds in the same per-component format used by
        `BarycentricGrid` and `CartesianGrid`, i.e. ``{component: {"min": lower, "max": upper}}``,
        or as ordered ``(lower, upper)`` pairs. Required.
    model_variable : str
        Dataset variable holding the model ref written by the upstream regressor
        (e.g. ``"bayesopt_model"``).
    grid_dim : str, default="grid"
        Dimension indexing candidate grid points.
    objective_direction : str, default="minimize"
        Direction the upstream regressor was configured with; the GP posterior lives in
        maximization space and ``best_f`` is mapped back to original units.
    is_simplex : bool, default=False
        When True, optimization is constrained to the simplex :math:`\sum_i x_i = 1`.
    num_restarts : int, default=10
        Number of restarts for the BoTorch optimizer.
    raw_samples : int, default=128
        Number of raw samples for the BoTorch optimizer.
    output_prefix : str, optional
        Prefix applied to the ``best_x`` and ``best_f`` outputs.
    name : str, default="BoTorchPosteriorAcquisition"
        Pipeline operation name.

    Outputs
    -------
    best_x : xr.DataArray
        Optimized point, dims ``("component",)``.
    best_f : xr.DataArray
        Posterior mean at ``best_x`` in the original objective units.
    """

    def __init__(
        self,
        feature_input_variable: str,
        grid_variable: str,
        bounds: Bounds,
        model_variable: str,
        grid_dim: str = "grid",
        objective_direction: str = "minimize",
        is_simplex: bool = False,
        num_restarts: int = 10,
        raw_samples: int = 128,
        output_prefix: Optional[str] = None,
        name: str = "BoTorchPosteriorAcquisition",
    ) -> None:
        super().__init__(
            name=name,
            input_variable=[feature_input_variable, grid_variable, model_variable],
            output_variable=[
                f"{output_prefix}_best_x" if output_prefix else "best_x",
                f"{output_prefix}_best_f" if output_prefix else "best_f",
            ],
            output_prefix=output_prefix,
        )
        if model_variable is None:
            raise ValueError(f"{name} requires `model_variable`; it never fits a model itself.")
        self.feature_input_variable = feature_input_variable
        self.grid_variable = grid_variable
        self.grid_dim = grid_dim
        self.objective_direction = objective_direction
        self._init_botorch(
            bounds=bounds,
            is_simplex=is_simplex,
            model_variable=model_variable,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            require_bounds=True,
        )

    def optimize_posterior_mean(self, model, bounds: torch.Tensor, inequality_constraints=None) -> tuple[torch.Tensor, float]:
        """Maximize the GP posterior mean; returns ``(best_x, best_f)`` with ``best_f`` in original units."""
        posterior_mean = self.botorch["PosteriorMean"](model)
        best_x, values = self.optimize_acqf(posterior_mean, bounds=bounds, q=1, constraints=inequality_constraints)
        best_f = float(values.max().detach().cpu().item())
        if self.objective_direction == "minimize":
            best_f = -best_f
        return best_x, best_f

    def calculate(self, dataset: xr.Dataset) -> Self:
        model = self.load_model(dataset)
        candidate_x = self.candidates(dataset)
        optimization_bounds = self.bounds_tensor(dataset, write_output=False)

        try:
            best_x, best_f = self.optimize_posterior_mean(
                model,
                bounds=optimization_bounds,
                inequality_constraints=self.constraints(candidate_x.shape[-1]),
            )
        except RuntimeError:
            warnings.warn(
                "BoTorch posterior optimization failed; falling back to the best evaluated grid point.",
                stacklevel=2,
            )
            # the model is trained on negated targets when minimizing, so argmax is always right
            best_x, best_f = self.best_grid_point(model.posterior(candidate_x).mean, candidate_x)
            if self.objective_direction == "minimize":
                best_f = -best_f

        feature_coords = dataset[self.feature_input_variable].coords.get("component")
        self.output[self._prefix_output("best_x")] = xr.DataArray(
            best_x.squeeze(0).detach().cpu().numpy(),
            dims=("component",),
            coords={
                "component": feature_coords.values if feature_coords is not None else np.arange(best_x.shape[-1])
            },
        )
        self.output[self._prefix_output("best_x")].attrs["description"] = (
            "Point maximizing (or minimizing) the GP posterior mean within bounds"
        )
        self.output[self._prefix_output("best_f")] = xr.DataArray(best_f)
        self.output[self._prefix_output("best_f")].attrs["description"] = (
            "GP posterior mean at best_x, in original objective units"
        )
        return self
