"""PipelineOps for Data Preprocessing

Preprocessing ops generally take in measurement data and scale, correct, and transform it

"""
import warnings
import numpy as np
import xarray as xr
from scipy.signal import savgol_filter
from AFL.double_agent.Pipeline import PipelineOpBase


class Preprocessor(PipelineOpBase):
    """Base class stub for all preprocessors"""

    def __init__(self, input_variable=None, output_variable=None, name='PreprocessorBase'):
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)


class SavgolFilter(Preprocessor):
    def __init__(self, input_variable, output_variable, dim='q', xlo=None, xhi=None, xlo_isel=None, xhi_isel=None,
                 pedestal=None, npts=250, derivative=0, window_length=31, polyorder=2,
                 apply_log_scale=True, name='SavgolFilter'):

        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        self.dim = dim
        self.npts = npts
        self.xlo = xlo
        self.xhi = xhi
        self.xlo_isel = xlo_isel
        self.xhi_isel = xhi_isel
        self.pedestal = pedestal
        self.derivative = derivative
        self.apply_log_scale = apply_log_scale
        self.polyorder = polyorder
        self.window_length = window_length

    def calculate(self, dataset):
        data1 = self._get_variable(dataset)

        if self.dim not in data1.coords:
            raise ValueError((
                f"""dim argument of PipelineOp '{self.name}' set to '{self.dim}' but not found in coords"""
            ))

        if (self.xlo is not None) or (self.xhi is not None):
            if (self.xlo_isel is not None) or (self.xhi_isel is not None):
                warnings.warn((
                    'Both value and index based indexing have been specified! '
                    'Value indexing will be applied first!'
                ), stacklevel=2)

        data1 = data1.sel({self.dim: slice(self.xlo, self.xhi)})
        data1 = data1.isel({self.dim: slice(self.xlo_isel, self.xhi_isel)})

        if self.apply_log_scale:
            dim = 'log_' + self.dim
            data1 = data1.where(data1 > 0.0, drop=True)
            data1 = data1.pipe(np.log10)
            data1[self.dim] = np.log10(data1[self.dim])
            data1 = data1.rename({self.dim: dim})
        else:
            dim = self.dim

        # need to remove duplicate values
        data1 = data1.groupby(dim, squeeze=False).mean()

        # set minimum value of scattering to pedestal value and fill nans with this value
        if self.pedestal is not None:
            data1 += self.pedestal
            data1 = data1.where(~np.isnan(data1)).fillna(self.pedestal)

        # interpolate to constant log(dq) grid
        if self.apply_log_scale:
            xnew = np.geomspace(data1[dim].min(), data1[dim].max(), self.npts)
        else:
            xnew = np.linspace(data1[dim].min(), data1[dim].max(), self.npts)
        dx = xnew[1] - xnew[0]
        data1 = data1.interp({dim: xnew})

        # filter out any q that have NaN
        data1 = data1.dropna(dim, 'any')

        # take derivative
        data1_filtered = savgol_filter(data1.values.T, window_length=self.window_length, polyorder=self.polyorder,
                                       delta=dx, axis=0,
                                       deriv=self.derivative)

        self.output[self.output_variable] = data1.copy(data=data1_filtered.T)

        return self


class SubtractMin(Preprocessor):
    """
    Baseline variable by subtracting minimum value

    """

    def __init__(self, input_variable, output_variable, dim, name='SubtractMin'):
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        self.dim = dim

    def calculate(self, dataset):
        data1 = self._get_variable(dataset)

        self.output[self.output_variable] = data1 - data1.min(self.dim)

        return self


class BarycentricToTernaryXY(Preprocessor):
    """
    Transform from ternary coordinates to xy coordinates

    Note ---- Adapted from BaryCentric transform mpltern:
    https://github.com/yuzie007/mpltern/blob/master/mpltern/ternary/transforms.py

    """

    def __init__(self, input_variable, output_variable, sample_dim, name=None):
        """Constructor

        Parameters
        ----------
        input_variable : str
            The data variable to extract from the dataset

        output_variable : str
            The dataset key to write the result of this PipelineOp into

        sample_dim: str
            The name of the sample_dimension for this calculation


        """
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        xmin = 0.5 - 1.0 / np.sqrt(3)
        xmax = 0.5 + 1.0 / np.sqrt(3)
        self.corners = [(0.5, 1.0), (xmin, 0.0), (xmax, 0.0)]

        self.sample_dim = sample_dim

    def calculate(self, dataset):
        bary = self._get_variable(dataset)
        bary = bary.transpose(self.sample_dim, ...).values
        bary = bary / np.sum(bary, axis=1)[:, np.newaxis]
        xy = np.dot(bary, self.corners)

        self.output[self.output_variable] = xr.DataArray(xy, dims=[self.sample_dim, 'xy'])
        self.output[self.output_variable].attrs["description"] = ("xy coordinates from barycentric variable '{"
                                                                  "self.input_variable}'")
        return self


class TernaryXYToBarycentric(Preprocessor):
    """
    Transform to ternary coordinates from xy coordinates

    Note ---- Adapted from BaryCentric transform mpltern:
    https://github.com/yuzie007/mpltern/blob/master/mpltern/ternary/transforms.py

    """

    def __init__(self, input_variable, output_variable, sample_dim, name=None):
        """Constructor

        Parameters
        ----------
        input_variable : str
            The data variable to extract from the dataset

        output_variable : str
            The dataset key to write the result of this PipelineOp into

        sample_dim: str
            The name of the sample_dimension for this calculation


        """
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        xmin = 0.5 - 1.0 / np.sqrt(3)
        xmax = 0.5 + 1.0 / np.sqrt(3)
        self.corners = [(0.5, 1.0), (xmin, 0.0), (xmax, 0.0)]

        self.sample_dim = sample_dim

    def calculate(self, dataset):
        xy = self._get_variable(dataset)
        xy = xy.transpose(self.sample_dim, ...).values
        xys = np.column_stack((xy, np.ones(xy.shape[0])))
        v = np.column_stack((self.corners, np.ones(3)))
        bary = np.dot(xys, np.linalg.inv(v))
        self.output[self.output_variable] = xr.DataArray(bary, dims=[self.sample_dim, 'component'])
        self.output[self.output_variable].attrs["description"] = ("Barycentric coordinates from variable '{"
                                                                  "self.input_variable}'")
        return self