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
from typing import List, Optional

import numpy as np
import xarray as xr
from typing_extensions import Self

from AFL.double_agent.Generator import GaussianPoints
from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import (
    dataset_to_botorch_candidates,
    dataset_to_botorch_training_data,
    evaluate_log_expected_improvement,
    evaluate_qlog_expected_improvement,
    fit_single_task_gp,
    get_observed_best_f,
    listify,
    make_simplex_constraints,
    optimization_bounds_to_xarray,
    optimization_bounds_to_tensor,
    optimize_acquisition_function,
)

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

            merge_list.append(d)

        self.merge_list = merge_list
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
        next_samples = next_samples.rename({self.grid_dim: "AF_sample"})
        next_samples = next_samples.reset_index("AF_sample", drop=True)

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


class BoTorchAcquisition(AcquisitionFunction):
    r"""BoTorch-based acquisition optimization with optional simplex constraints.

    Parameters
    ----------
    feature_input_variable : str
        Name of the feature variable used to fit the surrogate model.
    predictor_input_variable : str
        Name of the scalar objective variable.
    grid_variable : str
        Name of the candidate grid variable.
    grid_dim : str, default="grid"
        Dimension indexing candidate grid points.
    sample_dim : str, default="sample"
        Dimension indexing observed samples.
    objective_direction : str, default="maximize"
        Whether the objective is maximized or minimized.
    standardize : bool, default=True
        Whether to standardize the GP outputs.
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
        Number of points to optimize jointly.
    acquisition_kind : str, default="auto"
        Acquisition function family. "auto" selects logEI or qLogEI from `count`.
    best_f_variable : str, optional
        Dataset variable containing the incumbent objective value.
    bounds : dict | list[tuple[float, float]] | None, default=None
        Explicit optimization bounds passed through to BoTorch's optimizer.
        This may be supplied in the same per-component format used by
        `BarycentricGrid` and `CartesianGrid`, i.e.
        ``{component: {"min": lower, "max": upper}}``, or as ordered
        ``(lower, upper)`` pairs. Bounds must be expressed in the same feature
        space as `feature_input_variable` and `grid_variable`.
    is_simplex : bool, default=False
        When True, acquisition optimization is constrained to the simplex.
        The GP surrogate is still fit directly in the original feature space.

        Notes
        -----
        The simplex is mathematically defined by the equality
        :math:`\sum_i x_i = 1` together with non-negativity. BoTorch's optimizer
        interface consumes linear inequality constraints, so the equality is
        represented as the equivalent pair
        :math:`\sum_i x_i \ge 1` and :math:`-\sum_i x_i \ge -1`.
    name : str, default="BoTorchAcquisition"
        Pipeline operation name.
    """

    def __init__(
        self,
        feature_input_variable: str,
        predictor_input_variable: str,
        grid_variable: str,
        grid_dim: str = "grid",
        sample_dim: str = "sample",
        objective_direction: str = "maximize",
        standardize: bool = True,
        output_prefix: Optional[str] = None,
        output_variable: str = "next_samples",
        decision_rtol: float = 0.05,
        excluded_comps_variables: Optional[List[str]] = None,
        excluded_comps_dim: Optional[str] = None,
        exclusion_radius: float = 1e-3,
        count: int = 1,
        acquisition_kind: str = "auto",
        best_f_variable: Optional[str] = None,
        bounds: Optional[dict] = None,
        is_simplex: bool = False,
        name: str = "BoTorchAcquisition",
    ) -> None:
        super().__init__(
            input_variables=[feature_input_variable, predictor_input_variable],
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
        self.predictor_input_variable = predictor_input_variable
        self.sample_dim = sample_dim
        self.objective_direction = objective_direction
        self.standardize = standardize
        self.acquisition_kind = acquisition_kind
        self.best_f_variable = best_f_variable
        self.bounds = bounds
        self.is_simplex = is_simplex
        self.acquisition = None

    def calculate(self, dataset: xr.Dataset) -> Self:
        train_x, train_y = dataset_to_botorch_training_data(
            dataset=dataset,
            feature_input_variable=self.feature_input_variable,
            predictor_input_variable=self.predictor_input_variable,
            sample_dim=self.sample_dim,
            objective_direction=self.objective_direction,
        )
        candidate_x = dataset_to_botorch_candidates(
            dataset=dataset,
            grid_variable=self.grid_variable,
            grid_dim=self.grid_dim,
        )
        model = fit_single_task_gp(
            train_x=train_x,
            train_y=train_y,
            standardize=self.standardize,
        )

        if self.best_f_variable is not None and self.best_f_variable in dataset:
            best_f = float(dataset[self.best_f_variable].values[()])
        else:
            best_f = get_observed_best_f(
                train_y,
                objective_direction=self.objective_direction,
            )

        internal_best_f = -best_f if self.objective_direction == "minimize" else best_f

        inequality_constraints = (
            make_simplex_constraints(candidate_x.shape[-1]) if self.is_simplex else None
        )

        acquisition_kind = self.acquisition_kind
        if acquisition_kind == "auto":
            acquisition_kind = "qlogei" if self.count > 1 else "logei"

        # Evaluate acquisition function on the grid for decision surface
        if acquisition_kind == "logei":
            acq_values = evaluate_log_expected_improvement(
                model=model,
                candidate_x=candidate_x,
                best_f=internal_best_f,
            )
        else:  # qlogei
            acq_values = evaluate_qlog_expected_improvement(
                model=model,
                candidate_x=candidate_x,
                best_f=internal_best_f,
                q=self.count,
            )
        
        # Create decision_surface by reshaping acquisition values to grid structure
        grid_variable_data = dataset[self.grid_variable]
        acq_numpy = acq_values.detach().cpu().numpy().reshape(-1)
        
        # Reshape to match the grid structure (all dims except the sample dimension)
        grid_shape = grid_variable_data.shape
        decision_surface = xr.DataArray(
            acq_numpy.reshape(grid_shape[:-1]) if len(grid_shape) > 1 else acq_numpy,
            dims=grid_variable_data.dims[:-1],
            coords={dim: grid_variable_data.coords[dim] for dim in grid_variable_data.dims[:-1]},
        )
        
        self.output[self._prefix_output("decision_surface")] = decision_surface
        self.output[self._prefix_output("decision_surface")].attrs["description"] = (
            "Acquisition function values evaluated on the grid"
        )

        if self.bounds is None:
            raise ValueError("BoTorchAcquisition requires explicit `bounds` for optimize_acqf.")
        component_dim = dataset[self.grid_variable].dims[-1]
        component_names = dataset[self.grid_variable].coords.get(component_dim)
        optimization_bounds = optimization_bounds_to_tensor(
            self.bounds,
            component_names=component_names.values if component_names is not None else None,
        )
        if component_names is not None:
            self.output[self._prefix_output("bounds")] = optimization_bounds_to_xarray(
                optimization_bounds,
                component_names=component_names.values,
                variable_name=self._prefix_output("bounds"),
            )

        # Use BoTorch's optimize_acqf to find optimal points
        try:
            optimized_x, _ = optimize_acquisition_function(
                model=model,
                bounds=optimization_bounds,
                best_f=internal_best_f,
                acquisition_kind=acquisition_kind,
                q=self.count,
                num_restarts=10,
                raw_samples=128,
                inequality_constraints=inequality_constraints,
            )
            optimized_points = optimized_x.detach().cpu().numpy()
        except RuntimeError:
            warnings.warn(
                "BoTorch acquisition optimization failed; falling back to the best evaluated grid point(s).",
                stacklevel=2,
            )
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
                self.output[self._prefix_output("decision_surface")] = acquisition_dataset[
                    "decision_surface"
                ]
            self.get_next_samples(acquisition_dataset)
            return self
        
        # Convert optimized points to numpy array and create output
        
        # Handle single vs. batch optimization
        if optimized_points.ndim == 1:
            optimized_points = optimized_points[np.newaxis, :]
        
        # Get component names from the feature input variable
        component_coords = dataset[self.feature_input_variable].coords.get(
            "component", None
        )
        
        # Create output for next samples
        next_samples = xr.DataArray(
            optimized_points,
            dims=("next_sample", "component"),
            coords={
                "component": component_coords.values if component_coords is not None else range(optimized_points.shape[1])
            },
        )
        
        self.output[self.output_variable] = next_samples
        self.output[self.output_variable].attrs["description"] = (
            "Next sample point(s) optimized using BoTorch acquisition function"
        )
        
        return self
