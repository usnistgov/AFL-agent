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


from AFL.double_agent.util import listify
from AFL.double_agent.PipelineOp import PipelineOp

from AFL.double_agent.Generator import GaussianPoints


class AcquisitionError(Exception):
    """Exception raised when an error in the acquisition decision occurs"""
    pass


class AcquisitionFunction(PipelineOp):
    def __init__(self, input_variables, grid_variable, grid_dim='grid', output_prefix=None,
                 output_variable="next_compositions", decision_rtol=0.05,
                 excluded_comps_variables=None, excluded_component_dim='component',
                 exclusion_depth=1e-3, exclusion_radius=1e-3,
                 count=1, name='AcquisitionFunctionBase'):
        """
        Parameters
        ----------
        count: int
            Number of 'next_samples' to generate

        """
        super().__init__(input_variable=listify(input_variables), output_variable=output_variable, name=name,
                         output_prefix=output_prefix)

        self.excluded_comps_variables = excluded_comps_variables
        self.count = count
        self.grid_variable = grid_variable
        self.grid_dim = grid_dim
        self.decision_rtol = decision_rtol
        self.exclusion_radius = exclusion_radius
        self.exclusion_depth = exclusion_depth
        ## self.excluded_component_dim = excluded_component_dim
        self._banned_from_attrs.append('acquisition')


    def _get_excluded_comps(self,dataset):
        return xr.merge([dataset[variable] for variable in listify(self.excluded_comps_variables)])

    def exclude_previous_samples(self, dataset: xr.Dataset, excluded_comps: xr.Dataset) -> xr.Dataset:
        """Modify the decision surface by placing exclusion zones over previously measured compositions."""

        if ('decision_surface' not in dataset) or ('comp_grid' not in dataset):
            raise AcquisitionError((
                """Acquisition function must pass .exclude_previous_samples an xarray.Dataset with variables, """
                f"""'decision_surface' and 'comp_grid'. The supplied dataset had: {dataset.keys()}."""
            ))

        decision_mod = np.zeros_like(dataset.grid.values)
        n_comps = dataset.comp_grid.transpose(self.grid_dim, ...).values.shape[1]
        normalization = np.sqrt((2 * np.pi) ** n_comps * np.linalg.det(np.eye(n_comps) * self.exclusion_radius))
        for i, coord in excluded_comps.groupby(self.grid_dim):
            decision_mod = decision_mod + (
                    self.exclusion_depth * normalization * multivariate_normal.pdf(dataset[self.grid_variable],
                                                                                   mean=coord.values.squeeze(),
                                                                                   cov=self.exclusion_radius)
            )

        self.output[self._prefix_output('pre_exclude_decision_surface')] = dataset['decision_surface']
        self.output[self._prefix_output('pre_exclude_decision_surface')].attrs['description'] = (
            "The acquisition surface before previously measured samples are excluded."
        )

        dataset['decision_surface'] += decision_mod
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
                f"""'decision_surface' and 'comp_grid'. The supplied dataset had: {dataset.keys()}."""
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

        self.output[self.output_variable] = next_samples  # should already be a xr.DataArray


class RandomAF(AcquisitionFunction):

    def __init__(self, grid_variable, grid_dim='grid', output_prefix=None,
                 output_variable='next_samples', decision_rtol=0.05,
                 excluded_comps_variables=None, exclusion_depth=1e-3, exclusion_radius=1e-3,
                 count=1, name='RandomAF'):

        super().__init__(
            input_variables=grid_variable, grid_variable=grid_variable, grid_dim=grid_dim,
            output_prefix=output_prefix, output_variable=output_variable, decision_rtol=decision_rtol,
            excluded_comps_variables=excluded_comps_variables, exclusion_depth=exclusion_depth,
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

        excluded_comps = self._get_excluded_comps(dataset)
        print(excluded_comps)
        self.exclude_previous_samples(dataset=self.acquisition, excluded_comps=excluded_comps)

        self.output[self._prefix_output('decision_surface')] = self.acquisition['decision_surface']
        self.output[self._prefix_output('decision_surface')].attrs['description'] = (
            "The final acquisition surface that is evaluated to determine the next_sample"
        )

        return self
