"""PipelineOps for Data Preprocessing

This module contains preprocessing operations that transform, normalize, and prepare data for analysis.
Preprocessors handle tasks such as:
- Scaling and normalizing data
- Transforming between coordinate systems
- Filtering and smoothing signals
- Extracting features from raw measurements
- Converting between different data representations

Each preprocessor is implemented as a PipelineOp that can be composed with others in a processing pipeline.
"""

import warnings
from numbers import Number
from typing import Union, Optional, List, Dict

import numpy as np
import sympy
import xarray as xr
from scipy.signal import savgol_filter  # type: ignore
from typing_extensions import Self

from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify


class Preprocessor(PipelineOp):
    """Base class stub for all preprocessors

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    name : str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
    """

    def __init__(
        self,
        input_variable: str = None,
        output_variable: str = None,
        name: str = "PreprocessorBase",
    ) -> None:
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        return NotImplementedError(".calculate must be implemented in subclasses")  # type: ignore

class SAXSLogLogTransform(Preprocessor):
    """Pre-processing class to transform SAXS data into log-log scale.

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    dim: str
        The dimension in the `xarray.Dataset` to apply this transform over
    name : str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
    """    
    def __init__(
        self,
        input_variable: str = None,
        output_variable: str = None,
        dim: str = "q",
        name: str = "PreprocessorBase",
    ) -> None:
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )
        self.dim = dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)

        # add log_{dim}
        dim = "log_" + self.dim
        data1 = data1.where(data1>0.0, drop=True)
        data1 = data1.pipe(np.log10)
        data1[self.dim] = np.log10(data1[self.dim])
        data1 = data1.rename({self.dim: dim})

        self.output[self.output_variable] = data1
        self.output[self.output_variable].attrs[
            "description"
        ] = f"SAS data log-log transformed"

        return self
    
class SavgolFilter(Preprocessor):
    """Smooth and take derivatives of input data via a Savitsky-Golay filter

    This `PipelineOp` cleans measurement data and takes smoothed derivatives using `scipy.signal.savgol_filter`. Below
    is a summary of the steps taken.

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    dim: str
        The dimension in the `xarray.Dataset` to apply this filter over
    xlo, xhi: Optional[Number]
        The values of the input dimension (dim, above) to trim the data to
    xlo_isel, xhi_isel: Optional[int]
        The integer indices of the input dimension (dim, above) to trim the data to
    pedestal: Optional[Number]
        This value is added to the input_variable to establish a fixed data 'floor'
    npts: int
        The size of the grid to interpolate onto
    derivative: int
        The order of the derivative to return. If derivative=0, the data is smoothed with no derivative taken.
    window_length: int
        The width of the window used in the savgol smoothing. See `scipy.signal.savgol_filter` for more information.
    polyorder: int
        The order of polynomial used in the savgol smoothing. See `scipy.signal.savgol_filter` for more information.
    apply_log_scale: bool
        If True, the `input_variable` and associated `dim` coordinated are scaled with `numpy.log10`
    name: str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()

    Notes
    -----
    This `PipelineOp` performs the following steps:

    1. Data is trimmed to `(xlo, xhi)` and then `(xlo_isel, xhi_isel)` in that order. The former trims the data to a
    numerical while the latter trims to integer indices. It is generally not advisable to supply both varieties and a
    warning will be raised if this is attempted.

    2. If `apply_log_scale = True`, both the `input_variable` and `dim` data will be scaled with `numpy.log10`. A new
    `xarray` dimension and coordinate will be created with the name `log_{dim}`.

    3. All duplicate data (multiple data values at the same `dim` coordinates) are removed by taking the average of the
    duplicates.

    4. If `pedestal` is specified, the pedestal value is added to the data and all NaNs are filled with the pedestal

    5. The data is interpolated onto a constant grid with `npts` values from the trimmed minimum to the trimmed maximum.  If `apply_log_scale=True`, the grid is geometrically rather than linearly spaced.

    6. All remaining NaN values are dropped along `dim`

    7. Finally, `scipy.signal.savgol_filter` is applied with the `window_length`, `polyorder`, and `derivative`
    parameters specified in the constructor.

    """

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        dim: str = "q",
        xlo: Optional[Number] = None,
        xhi: Optional[Number] = None,
        xlo_isel: Optional[int] = None,
        xhi_isel: Optional[int] = None,
        pedestal: Optional[Number] = None,
        npts: int = 250,
        derivative: int = 0,
        window_length: int = 31,
        polyorder: int = 2,
        apply_log_scale: bool = True,
        name: str = "SavgolFilter",
    ) -> None:
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )

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

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)

        if self.dim not in data1.coords:
            raise ValueError(
                (
                    f"""dim argument of PipelineOp '{self.name}' set to '{self.dim}' but not found in coords"""
                )
            )

        if (self.xlo is not None) or (self.xhi is not None):
            if (self.xlo_isel is not None) or (self.xhi_isel is not None):
                warnings.warn(
                    (
                        "Both value and index based indexing have been specified! "
                        "Value indexing will be applied first!"
                    ),
                    stacklevel=2,
                )

        data1 = data1.transpose(self.dim,...)

        data1 = data1.sel({self.dim: slice(self.xlo, self.xhi)})
        data1 = data1.isel({self.dim: slice(self.xlo_isel, self.xhi_isel)})

        if self.apply_log_scale:
            dim = "log_" + self.dim
            data1 = data1.where(data1 > 0.0, drop=True)
            data1 = data1.pipe(np.log10)
            data1[self.dim] = np.log10(data1[self.dim])
            data1 = data1.rename({self.dim: dim})
        else:
            dim = "sg_"+self.dim
            data1[dim] = data1[self.dim]

        # need to remove duplicate values
        data1 = data1.groupby(dim, squeeze=False).mean()

        # set minimum value of scattering to pedestal value and fill nans with this value
        if self.pedestal is not None:
            data1 += self.pedestal
            data1 = data1.where(~np.isnan(data1)).fillna(self.pedestal)
        
        # interpolate to constant lin or log(dq) grid
        x_new = np.linspace(data1[dim].min().item(), data1[dim].max().item(), self.npts)
        dx = float(x_new[1] - x_new[0])
        data1 = data1.bfill(dim).ffill(dim).interpolate_na(dim)
        data1 = data1.interp({dim: x_new})

        # filter out any q that have NaN
        data1 = data1.dropna(dim, how="any")



        # take derivative
        data1_filtered = savgol_filter(
            data1.values,
            window_length=self.window_length,
            polyorder=self.polyorder,
            delta=dx,
            axis=0,
            deriv=self.derivative,
        )

        self.output[self.output_variable] = data1.copy(data=data1_filtered).transpose(...,dim)
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Savitsky-Golay filtered data"

        return self


class SubtractMin(Preprocessor):
    """Baseline input variable by subtracting minimum value"""

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        dim: str,
        name: str = "SubtractMin",
    ) -> None:
        """
        Parameters
        ----------
        input_variable : str
            The name of the `xarray.Dataset` data variable to extract from the input dataset

        output_variable : str
            The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

        dim: str
             The dimension to use when calculating the data minimum

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search(
        """
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )

        self.dim = dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)

        self.output[self.output_variable] = data1 - data1.min(self.dim)
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Subtracted minimum value along dimension '{self.dim}'"

        return self

class Subtract(Preprocessor):
    """Baseline input variable by subtracting a value"""

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        dim: str,
        value: float | str,
        coord_value: bool = True,
        name: str = "Subtract",
    ) -> None:
        """
        Parameters
        ----------
        input_variable : str
            The name of the `xarray.Dataset` data variable to extract from the input dataset

        output_variable : str
            The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

        dim: str
             The dimension to use when calculating the data minimum

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search(
        """
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )

        self.dim = dim
        self.value = value
        self.coord_value = coord_value

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)
        
        if isinstance(self.value,str) and self.coord_value:
            value = dataset[self.value].item()
            subtract_value = data1.sel({self.dim:value},method='nearest')
        elif self.coord_value:
            subtract_value = data1.sel({self.dim:self.value},method='nearest')
        else:
            subtract_value = self.value
            
        self.output[self.output_variable] = data1 - subtract_value
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Subtracted minimum value along dimension '{self.dim}'"

        return self


class Standardize(Preprocessor):
    """Standardize the data to have min 0 and max 1

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    dim : str
        The dimension used for calculating the data minimum
    component_dim : str | None, default="component"
        The dimension for component-wise operations
    scale_variable : str | None, default=None
        If specified, the min/max of this data variable in the supplied `xarray.Dataset` will be used to scale the
        data rather than min/max of the `input_variable` or the supplied `min_val` or `max_val`
    min_val : Number | None, default=None
        Value used to scale the data minimum
    max_val : Number | None, default=None
        Value used to scale the data maximum
    name : str, default="Standardize"
        The name to use when added to a Pipeline
    """

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        dim: str,
        component_dim: str | None = "component",
        scale_variable: str | None = None,
        min_val: Number | None = None,
        max_val: Number | None = None,
        name: str = "Standardize",
    ) -> None:
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )

        self.dim = dim
        self.component_dim = component_dim
        self.min_val = min_val
        self.max_val = max_val
        self.scale_variable = scale_variable

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)

        if self.max_val is None:
            if self.scale_variable is None:
                max_val = data1.max(self.dim)
            else:
                max_val = dataset[self.scale_variable].max(self.dim)
        elif isinstance(self.max_val,dict):
            max_val = xr.Dataset(self.max_val).to_array(self.component_dim)
        else:
            max_val = self.max_val

        if self.min_val is None:
            if self.scale_variable is None:
                min_val = data1.min(self.dim)
            else:
                min_val = dataset[self.scale_variable].min(self.dim)
        elif isinstance(self.min_val, dict):
            min_val = xr.Dataset(self.min_val).to_array(self.component_dim)
        else:
            min_val = self.min_val

        self.output[self.output_variable] = (data1 - min_val) / (max_val - min_val)
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Data normalized to have range 0 -> 1"

        return self


class Destandardize(Preprocessor):
    """Transform the data from 0->1 scaling

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    dim : str
        The dimension used for calculating the data minimum
    component_dim : str | None, default="component"
        The dimension for component-wise operations
    scale_variable : str | None, default=None
        If specified, the min/max of this data variable in the supplied `xarray.Dataset` will be used to scale the
        data rather than min/max of the `input_variable` or the supplied `min_val` or `max_val`
    min_val : Number | None, default=None
        Value used to scale the data minimum
    max_val : Number | None, default=None
        Value used to scale the data maximum
    name : str, default="Destandardize"
        The name to use when added to a Pipeline
    """

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        dim: str,
        component_dim: str | None = "component",
        scale_variable: str | None = None,
        min_val: Number | None = None,
        max_val: Number | None = None,
        name: str = "Destandardize",
    ) -> None:

        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )

        self.dim = dim
        self.component_dim = component_dim
        self.min_val = min_val
        self.max_val = max_val
        self.scale_variable = scale_variable

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)

        if self.max_val is None:
            if self.scale_variable is None:
                max_val = data1.max(self.dim)
            else:
                max_val = dataset[self.scale_variable].max(self.dim)
        elif isinstance(self.max_val,dict):
            max_val = xr.Dataset(self.max_val).to_array(self.component_dim)
        else:
            max_val = self.max_val

        if self.min_val is None:
            if self.scale_variable is None:
                min_val = data1.min(self.dim)
            else:
                min_val = dataset[self.scale_variable].min(self.dim)
        elif isinstance(self.min_val, dict):
            min_val = xr.Dataset(self.min_val).to_array(self.component_dim)
        else:
            min_val = self.min_val

        self.output[self.output_variable] = data1 * (max_val - min_val) + min_val
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Data normalized to have range 0 -> 1"

        return self


class Zscale(Preprocessor):
    """Z-scale the data to have mean 0 and standard deviation scaling

        Parameters
        ----------
        input_variable : str
            The name of the `xarray.Dataset` data variable to extract from the input dataset

        output_variable : str
            The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

        dim: str
             The dimension to use when calculating the data minimum

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search(
        """

    def __init__(
        self, input_variable: str, output_variable: str, dim: str, name: str = "Zscale"
    ) -> None:
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )

        self.mean = None
        self.std = None
        self.dim = dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)
        self.mean = data1.mean()
        self.std = data1.std()

        # this standardizes between the min and the max values, this isn't z-scaling...
        self.output[self.output_variable] = (data1 - self.mean) / self.std
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Data Zscaled to have range 0 ~ -1 -> 1"

        return self


class ZscaleError(Preprocessor):
    """Scale the y_err data, first input is y, second input is y_err

    Parameters
    ----------
    input_variables : Union[Optional[str], List[str]]
        The names of the input variables - first is y, second is y_err
    output_variable : str
        The name of the variable to be inserted into the dataset
    dim : str
        The dimension to use when calculating the data minimum
    name : str, default="Zscale_error"
        The name to use when added to a Pipeline
    """

    def __init__(
        self,
        input_variables: Union[Optional[str], List[str]],
        output_variable: str,
        dim: str,
        name: str = "Zscale_error",
    ) -> None:
        super().__init__(
            name=name, input_variable=input_variables, output_variable=output_variable
        )

        self.dim = dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)
        data_y_err = data1[self.input_variable[1]]
        data_y = data1[self.input_variable[0]]

        self.output[self.output_variable] = data_y_err / data_y.std()
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Uncertainty normalized to have range 0 -> inf"

        return self


class BarycentricToTernaryXY(Preprocessor):
    """
    Transform from ternary coordinates to xy coordinates

    Note ---- Adapted from BaryCentric transform mpltern:
    https://github.com/yuzie007/mpltern/blob/master/mpltern/ternary/transforms.py

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset

    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

    sample_dim: str
         The dimension to use when calculating the data minimum

    name: str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search(
    """

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        sample_dim: str,
        name: str = "BarycentricToTernaryXY",
    ) -> None:
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )

        x_min = 0.5 - 1.0 / np.sqrt(3)
        x_max = 0.5 + 1.0 / np.sqrt(3)
        self.corners = np.array([(0.5, 1.0), (x_min, 0.0), (x_max, 0.0)])

        self.sample_dim = sample_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        bary = self._get_variable(dataset)
        bary = bary.transpose(self.sample_dim, ...).values
        bary = bary / np.sum(bary, axis=1)[:, np.newaxis]
        xy = np.dot(bary, self.corners)

        self.output[self.output_variable] = xr.DataArray(
            xy, dims=[self.sample_dim, "xy"]
        )
        self.output[self.output_variable].attrs["description"] = (
            "barycentric coordinates from xy variable " f"'{self.input_variable}'"
        )
        return self


class TernaryXYToBarycentric(Preprocessor):
    """Transform to ternary coordinates from xy coordinates

    Note 
    ---- 
    Adapted from BaryCentric transform mpltern:
    https://github.com/yuzie007/mpltern/blob/master/mpltern/ternary/transforms.py

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    sample_dim : str
        The dimension to use when calculating the data minimum
    name : str, default="TernaryXYToBarycentric"
        The name to use when added to a Pipeline
    """

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        sample_dim: str,
        name: str = "TernaryXYToBarycentric",
    ) -> None:
        """
        Parameters
        ----------
        input_variable : str
            The name of the `xarray.Dataset` data variable to extract from the input dataset

        output_variable : str
            The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

        sample_dim: str
             The dimension to use when calculating the data minimum

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search(
        """
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )

        x_min = 0.5 - 1.0 / np.sqrt(3)
        x_max = 0.5 + 1.0 / np.sqrt(3)
        self.corners = [(0.5, 1.0), (x_min, 0.0), (x_max, 0.0)]

        self.sample_dim = sample_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        xy = self._get_variable(dataset)
        xy = xy.transpose(self.sample_dim, ...).values
        xys = np.column_stack((xy, np.ones(xy.shape[0])))
        v = np.column_stack((self.corners, np.ones(3)))
        bary = np.dot(xys, np.linalg.inv(v))
        self.output[self.output_variable] = xr.DataArray(
            bary, dims=[self.sample_dim, "component"]
        )
        self.output[self.output_variable].attrs["description"] = (
            "XY coordinates from barycentric variable " f"'{self.input_variable}'"
        )
        return self


class SympyTransform(Preprocessor):
    """Transform data using sympy expressions

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    sample_dim : str
        The sample dimension i.e., the dimension of compositions or grid points
    component_dim : str, default="component"
        The dimension of the component of each gridpoint
    transforms : Dict[str,object]
        A dictionary of transforms (sympy expressions) to evaluate to generate new variables. For this method to
        function, the transforms must be completely specified except for the names in component_dim of the
        input_variable
    transform_dim : str
        The name of the dimension that the 'component_dim' will be transformed to
    name : str, default="SympyTransform"
        The name to use when added to a Pipeline

    Example
    -------
    ```python
    from AFL.double_agent import *
    import sympy
    with Pipeline() as p:
        CartesianGrid(
            output_variable='comps',
            grid_spec={
                'A':{'min':1,'max':25,'steps':5},
                'B':{'min':1,'max':25,'steps':5},
                'C':{'min':1,'max':25,'steps':5},
            },
            sample_dim='grid'
        )

        A,B,C = sympy.symbols('A B C')
        vA = A/(A+B+C)
        vB = B/(A+B+C)
        vC = C/(A+B+C)
        SympyTransform(
            input_variable='comps',
            output_variable='trans_comps',
            sample_dim='grid',
            transforms={'vA':vA,'vB':vB,'vC':vC},
            transform_dim='trans_component'
        )

    p.calculate(xr.Dataset())# returns dataset with grid and transformed grid
    ```
    """

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        sample_dim: str,
        transforms: Dict[str, object],
        transform_dim: str,
        component_dim: str = "component",
        name: str = "SympyTransform",
    ) -> None:

        # must convert to strings for JSON serialization
        transforms = {k:str(v) for k,v in transforms.items()}

        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )
        self.sample_dim = sample_dim
        self.component_dim = component_dim
        self.transforms = transforms
        self.transform_dim = transform_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data = dataset[self.input_variable].transpose(
            self.sample_dim, self.component_dim
        )

        # need to construct a dict of arrays
        comps = {
            k: v.squeeze().values
            for k, v in data.groupby(self.component_dim, squeeze=False)
        }

        # apply transform
        new_comps = xr.Dataset()
        for name, transform in self.transforms.items():
            transform = sympy.sympify(transform)
            symbols = list(transform.free_symbols)
            lam = sympy.lambdify(symbols, transform)
            new_comps[name] = (
                (self.sample_dim,),
                listify(lam(**{k.name: comps[k.name] for k in symbols})),
            )

        new_comps = new_comps.to_array(self.transform_dim).transpose(
            ..., self.transform_dim
        )

        self.output[self.output_variable] = new_comps
        self.output[self.output_variable].attrs[
            "description"
        ] = "Variables transformed using sympy expressions"

        return self


class Extrema(Preprocessor):
    """Find the extrema of a data variable"""

    def __init__(
            self,
            input_variable: str,
            output_variable: str,
            dim: str,
            return_coords: bool = False,
            operator='max',
            slice: Optional[List] = None,
            slice_dim: Optional[str] = None,
            name: str = "Extrema",
    ) -> None:
        """
        Parameters
        ----------
        input_variable : str
            The name of the `xarray.Dataset` data variable to extract from the input dataset

        output_variable : str
            The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

        dim: str
             The dimension to use when calculating the data minimum

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search(
        """
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )

        self.dim = dim
        self.operator = operator
        self.return_coords = return_coords
        self.slice = slice
        self.slice_dim = slice_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data = self._get_variable(dataset)

        if self.slice is not None:
            data = data.sel({self.slice_dim: slice(*self.slice)})

        if self.return_coords:
            if 'arg' not in self.operator:
                operator = 'arg' + self.operator
            else:
                operator = self.operator
            idx = getattr(data, operator)(dim=self.dim)
            self.output[self.output_variable] = data[self.dim][idx]
        else:
            self.output[self.output_variable] = getattr(data, self.operator)(dim=self.dim)

        self.output[self.output_variable].attrs[
            "description"
        ] = f"Extrema of {self.input_variable} with operator {self.operator} and return_coords {self.return_coords}"

        return self

class VarsToArray(Preprocessor):
    """Convert multiple variables into a single array

    Parameters
    ----------
    input_variables : List
        List of input variables to combine into an array
    output_variable : str
        The name of the variable to be inserted into the dataset
    variable_dim : str
        The dimension name for the variables in the output array
    squeeze : bool, default=False
        Whether to squeeze out single-dimension axes
    variable_mapping : Dict, default=None
        Optional mapping to rename variables
    name : str, default='VarsToArray'
        The name to use when added to a Pipeline
    """

    def __init__(
        self, 
        input_variables: List, 
        output_variable: str, 
        variable_dim: str,
        squeeze: bool = False,
        variable_mapping: Dict = None,
        name: str = 'VarsToArray'):

        super().__init__(name=name, input_variable=input_variables, output_variable=output_variable)
        print(self.input_variable,self.output_variable)

        if variable_mapping is None:
            self.variable_mapping = {}
        else:
            self.variable_mapping = variable_mapping
        
        self.variable_dim = variable_dim
        self.squeeze = squeeze

    def calculate(self, dataset):
        output = dataset[self.input_variable].rename_vars(self.variable_mapping).to_array(self.variable_dim)      
        if self.squeeze:
            output = output.squeeze()
        print(self.output_variable)
        print(output)
        self.output[self.output_variable] = output
        return self

class ArrayToVars(Preprocessor):
    """Convert an array into multiple variables

    Parameters
    ----------
    input_variable : str
        The name of the array variable to split into separate variables
    output_variables : list
        The names of the variables to create from the array
    split_dim : str
        The dimension to split the array along
    postfix : str, default=''
        String to append to output variable names
    squeeze : bool, default=False
        Whether to squeeze out single-dimension axes
    name : str, default='DatasetToVars'
        The name to use when added to a Pipeline
    """

    def __init__(
        self, 
        input_variable: str, 
        output_variables: list, 
        split_dim: list,
        postfix: str = '',
        squeeze: bool = False,
        name: str = 'DatasetToVars'):

        super().__init__(name=name, input_variable=input_variable, output_variable=output_variables)
        print(self.input_variable,self.output_variable)
        
        self.split_dim = split_dim
        self.squeeze = squeeze
        self.output_variables = output_variables
        self.postfix= postfix
    def calculate(self, dataset):
        input_arr = dataset[self.input_variable]   
        if self.squeeze:
            output = output.squeeze()
        for var in self.output_variables:
            print(var)
    
            self.output[var+self.postfix] = input_arr.sel({self.split_dim:var})
            
        return self

