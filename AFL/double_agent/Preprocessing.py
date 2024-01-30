"""
Preprocessing ops generally take in measurement data and scale, correct, and transform it
"""
import warnings
import numpy as np
from scipy.signal import savgol_filter
from AFL.double_agent.Pipeline import PipelineOpBase


class SavgolFilter(PipelineOpBase):
    def __init__(self, input_variable, output_variable, dim='q', xlo=None, xhi=None, xlo_isel=None, xhi_isel=None,
                 pedestal=None, npts=250, derivative=0, window_length=31, polyorder=2,
                 apply_log_scale=True, name=None):

        if name is None:
            name = 'SavgolFilter'

        super().__init__(name=name,input_variable=input_variable, output_variable=output_variable)

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


class SubtractMin(PipelineOpBase):
    """
    Baseline variable by subtracting minimum value

    """
    def __init__(self, input_variable, output_variable, dim, name=None):

        if name is None:
            name='SubtractMin'

        super().__init__(name=name,input_variable=input_variable, output_variable=output_variable)

        self.dim = dim

    def calculate(self, dataset):
        data1 = self._get_variable(dataset)

        self.output[self.output_variable] = data1 - data1.min(self.dim)

        return self
