"""Acquisition Functions

Parent class with methods for parsing decision array

break out decision pieces into more functions
- get_top_n_pct
- check_already_measured

support for producing n-predictions

to_add
- exclude_point_functino
- multivariate gaussian generation on grid (no pipelineop)

"""
import numpy as np
import xarray as xr
import copy

from AFL.double_agent.util import listify
from AFL.double_agent.PipelineOp import PipelineOp

from AFL.double_agent.Generator import GaussianPoints


class AcquisitionError(Exception):
    """Exception raised when an error in the acquisition decision occurs"""
    pass


class AcquisitionFunction(PipelineOp):
    def __init__(self, input_variables, grid_variable, grid_dim='grid', output_prefix=None,
                 output_variable="next_compositions", decision_rtol=0.05,
                 excluded_comps_variables=None, excluded_comps_dim='component', exclusion_radius=1e-3,
                 count=1, name='AcquisitionFunctionBase'):
        """
        Parameters
        ----------
        count: int
            Number of 'next_samples' to generate

        """
        super().__init__(input_variable=listify(input_variables) + [grid_variable], output_variable=output_variable,
                         name=name,
                         output_prefix=output_prefix)

        self.input_variables = input_variables
        self.excluded_comps_variables = excluded_comps_variables
        self.excluded_comps_dim = excluded_comps_dim
        self.count = count
        self.grid_variable = grid_variable
        self.grid_dim = grid_dim
        self.decision_rtol = decision_rtol
        self.exclusion_radius = exclusion_radius
        self._banned_from_attrs.append('acquisition')

    def _get_excluded_comps(self, dataset):

        merge_list = []
        for var in listify(self.excluded_comps_variables):
            merge_list.append(
                dataset[var].transpose(..., self.excluded_comps_dim).rename({dataset[var].dims[0]: 'sample'})
            )

        excluded_comps = xr.concat(merge_list, dim='sample')
        excluded_comps = xr.DataArray(np.unique(excluded_comps, axis=0), dims=excluded_comps.dims)
        return excluded_comps

    def exclude_previous_samples(self, dataset: xr.Dataset) -> xr.Dataset:
        """Modify the decision surface by placing exclusion zones over previously measured compositions."""

        if ('decision_surface' not in dataset) or ('comp_grid' not in dataset) or ('excluded_comps' not in dataset):
            raise AcquisitionError((
                """Acquisition function must pass .exclude_previous_samples an xarray.Dataset with variables, """
                f"""'decision_surface' and 'comp_grid' and 'excluded_comps'. """
                f"""The supplied dataset had: {dataset.keys()}."""
            ))

        exclusion_depth = dataset['decision_surface'].max().values[()]
        gp_op = GaussianPoints(
            input_variable='excluded_comps',
            sample_dim='sample',
            comps_dim=self.excluded_comps_dim,
            output_variable='excluded_surface',
            grid_variable='comp_grid',
            grid_dim=self.grid_dim,
            exclusion_radius=self.exclusion_radius,
            exclusion_depth=exclusion_depth,
        )
        gp_op.calculate(dataset)

        self.output.update(gp_op.output)

        dataset['decision_surface'] += gp_op.output['excluded_surface']
        return dataset

    def get_next_samples(self, dataset):
        """Choose the next composition by evaluating the decision surface in a provided dataset

        This method first finds all compositions that are within `self.decision_rtol` of the maximum values of the
        decision surface. From this set of compositions, it randomly chooses `self.count` composition as the next
        sample composition.

        """
        if ('decision_surface' not in dataset) or ('comp_grid' not in dataset):
            raise AcquisitionError((
                """Acquisition function must pass .get_next_sample an xarray.Dataset with variables, """
                f"""'decision_surface' and 'comp_grid'. The supplied dataset had: {list(dataset.keys())}."""
            ))

        # need to add integer 'coords' to the grid dimension. These allow us to index the parent
        # array after finding the optimal next samples
        dataset = dataset.assign_coords({self.grid_dim: np.arange(dataset.sizes[self.grid_dim])})

        # find indices of all samples within self.decison_rtol of the maximum
        close_mask = np.isclose(dataset.decision_surface, dataset.decision_surface.max(), rtol=self.decision_rtol,
                                atol=0)
        indices = dataset.sel({self.grid_dim: close_mask})[self.grid_dim].values

        if len(indices) < self.count:
            raise AcquisitionError((
                """Unable to find gridpoint in decision surface that satisfies all constraints. """
                """This often occurs when acquisition_rtol (currently {self.acquisition_rtol}) """
                """is too low or when the exclusion_radius (currently {self.exclusion_radius}) """
                """is too high for the current problem state."""
            ))

        # randomly shuffle and gather the requested number of indices and compositions
        np.random.shuffle(indices)
        next_indices = indices[:self.count]

        next_samples = dataset.sel({self.grid_dim: next_indices}).comp_grid
        next_samples = next_samples.rename({self.grid_dim:'AF_sample'})

        self.output[self.output_variable] = next_samples  # should already be a xr.DataArray


class RandomAF(AcquisitionFunction):

    def __init__(self, grid_variable, grid_dim='grid', output_prefix=None,
                 output_variable='next_samples', decision_rtol=0.05,
                 excluded_comps_variables=None, excluded_comps_dim=None, exclusion_radius=1e-3,
                 count=1, name='RandomAF'):
        super().__init__(
            input_variables=[], grid_variable=grid_variable, grid_dim=grid_dim,
            output_prefix=output_prefix, output_variable=output_variable, decision_rtol=decision_rtol,
            excluded_comps_variables=excluded_comps_variables, excluded_comps_dim=excluded_comps_dim,
            exclusion_radius=exclusion_radius, count=count, name=name
        )

        # this doesn't need to be an attribute, just convenient for debugging
        self.acquisition = None

    def calculate(self, dataset):
        self.acquisition = xr.Dataset()

        self.acquisition['comp_grid'] = dataset[self.grid_variable].transpose(self.grid_dim, ...)

        self.acquisition['decision_surface'] = xr.DataArray(
            np.random.random(size=self.acquisition['grid'].values.shape),
            dims=self.grid_dim
        )

        self.acquisition['excluded_comps'] = self._get_excluded_comps(dataset)
        self.exclude_previous_samples(self.acquisition)

        self.output[self._prefix_output('decision_surface')] = self.acquisition['decision_surface']
        self.output[self._prefix_output('decision_surface')].attrs['description'] = (
            "The final acquisition surface that is evaluated to determine the next_sample"
        )

        self.get_next_samples(self.acquisition)

        return self


class MaxValueAF(AcquisitionFunction):

    def __init__(self, input_variables, grid_variable, grid_dim='grid', combine_coeffs=None, output_prefix=None,
                 output_variable='next_samples', decision_rtol=0.05,
                 excluded_comps_variables=None, excluded_comps_dim=None, exclusion_radius=1e-3,
                 count=1, name='MaxValueAF'):

        super().__init__(
            input_variables=input_variables, grid_variable=grid_variable, grid_dim=grid_dim,
            output_prefix=output_prefix, output_variable=output_variable, decision_rtol=decision_rtol,
            excluded_comps_variables=excluded_comps_variables, excluded_comps_dim=excluded_comps_dim,
            exclusion_radius=exclusion_radius, count=count, name=name
        )
        self.combine_coeffs = combine_coeffs

        # this doesn't need to be an attribute, just convenient for debugging
        self.acquisition = None

    def calculate(self, dataset):
        self.acquisition = xr.Dataset()

        self.acquisition['comp_grid'] = dataset[self.grid_variable].transpose(self.grid_dim, ...)

        # make sure it's actually a list, listify will pass-through tuples
        input_variables = list(listify(self.input_variables))

        if self.combine_coeffs is not None:
            assert len(self.combine_coeffs) == len(input_variables)
            coeffs = copy.deepcopy(self.combine_coeffs)
        else:
            coeffs = [1] * len(input_variables)

        decision_surface = coeffs.pop(0) * dataset[input_variables.pop(0)].copy()
        for coeff, input_variable in zip(coeffs, input_variables):
            decision_surface += coeff * dataset[input_variable]

        self.acquisition['decision_surface'] = decision_surface

        self.acquisition['excluded_comps'] = self._get_excluded_comps(dataset)
        self.exclude_previous_samples(self.acquisition)

        self.output[self._prefix_output('decision_surface')] = self.acquisition['decision_surface']
        self.output[self._prefix_output('decision_surface')].attrs['description'] = (
            "The final acquisition surface that is evaluated to determine the next_sample"
        )

        self.get_next_samples(self.acquisition)

        return self
