"""
Acquisition functions gather one or more inputs and use that information to choose one or more points from a
from a supplied composition grid.
"""

import copy
from typing import List, Optional

import numpy as np
import xarray as xr
from typing_extensions import Self

from AFL.double_agent.Generator import GaussianPoints
from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify


class AcquisitionError(Exception):
    """Exception raised when an error in the acquisition decision occurs"""

    pass


class AcquisitionFunction(PipelineOp):
    """Base acquisition function"""

    def __init__(
        self,
        input_variables: List[str],
        grid_variable: str,
        grid_dim: str = "grid",
        output_prefix: Optional[str] = None,
        output_variable: str = "next_compositions",
        decision_rtol: float = 0.05,
        excluded_comps_variables: Optional[List[str]] = None,
        excluded_comps_dim: str = "component",
        exclusion_radius: float = 1e-3,
        count: int = 1,
        name: str = "AcquisitionFunctionBase",
    ) -> None:
        """
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
        self.exclusion_radius = exclusion_radius

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        return NotImplementedError(".calculate must be implemented in subclasses")  # type: ignore

    def _get_excluded_comps(self, dataset: xr.Dataset) -> xr.DataArray | None:
        """
        Gather all compositions listed in `self.excluded_comps_variables` and combine into a single
        `xarray.DataArray`
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
        """Modify the decision surface by placing Gaussian exclusion zones over previously measured compositions."""

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
        ]
        self.output[self._prefix_output("pre_exclude_decision_surface")].attrs[
            "description"
        ] = "The acquisition surface before it is modified by an exclusion field"

        dataset["decision_surface"] -= gp_op.output["excluded_surface"]
        return dataset

    def get_next_samples(self, dataset: xr.Dataset) -> None:
        """Choose the next composition by evaluating the decision surface in a provided dataset

        This method first finds all compositions that are within `self.decision_rtol` of the maximum values of the
        decision surface. From this set of compositions, it randomly chooses `self.count` composition as the next
        sample composition.
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

        # find indices of all samples within self.decision_rtol of the maximum
        close_mask = np.isclose(
            dataset.decision_surface,
            dataset.decision_surface.max(),
            rtol=self.decision_rtol,
            atol=0,
        )
        indices = dataset.sel({self.grid_dim: close_mask})[self.grid_dim].values

        if len(indices) < self.count:
            raise AcquisitionError(
                (
                    """Unable to find gridpoint in decision surface that satisfies all constraints. """
                    f"""This often occurs when acquisition_rtol (currently {self.decision_rtol}) """
                    f"""is too low or when the exclusion_radius (currently {self.exclusion_radius}) """
                    """is too high for the current problem state."""
                )
            )

        # randomly shuffle and gather the requested number of indices and compositions
        np.random.shuffle(indices)
        next_indices = indices[: self.count]

        next_samples = dataset.sel({self.grid_dim: next_indices}).comp_grid
        next_samples = next_samples.rename({self.grid_dim: "AF_sample"})
        next_samples = next_samples.reset_index("AF_sample", drop=True)

        self.output[self.output_variable] = (
            next_samples  # should already be a xr.DataArray
        )


class RandomAF(AcquisitionFunction):
    """Randomly choose from the grid with or without excluding previous measurement compositions"""

    def __init__(
        self,
        grid_variable: str,
        grid_dim: str = "grid",
        output_prefix: Optional[str] = None,
        output_variable: str = "next_samples",
        decision_rtol: float = 0.05,
        excluded_comps_variables: Optional[str] = None,
        excluded_comps_dim: Optional[str] = None,
        exclusion_radius: float = 1e-3,
        count: int = 1,
        name: str = "RandomAF",
    ) -> None:
        """
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
        super().__init__(
            input_variables=[],
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

        # this doesn't need to be an attribute, just convenient for debugging
        self.acquisition = None

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
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


class MaxValueAF(AcquisitionFunction):

    def __init__(
        self,
        input_variables: List[str],
        grid_variable: str,
        grid_dim: str = "grid",
        combine_coeffs: Optional[List[float]] = None,
        output_prefix: Optional[str] = None,
        output_variable: str = "next_samples",
        decision_rtol: float = 0.05,
        excluded_comps_variables: Optional[List[str]] = None,
        excluded_comps_dim: Optional[str] = None,
        exclusion_radius: float = 1e-3,
        count: int = 1,
        name: str = "MaxValueAF",
    ):
        """
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
        self.combine_coeffs = combine_coeffs

        # this doesn't need to be an attribute, just convenient for debugging
        self.acquisition = None

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
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

    def __init__(
        self,
        input_variables: List[str],
        grid_variable: str,
        grid_dim: str = "grid",
        lambdas=None,
        output_prefix: Optional[str] = None,
        output_variable: str = "next_samples",
        decision_rtol: float = 0.05,
        excluded_comps_variables: Optional[List[str]] = None,
        excluded_comps_dim: Optional[str] = None,
        exclusion_radius: float = 1e-3,
        count: int = 1,
        name: str="PseudoUCB",
    ):
        """
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
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
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
