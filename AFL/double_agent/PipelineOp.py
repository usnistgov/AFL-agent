import copy
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Dict, List

import matplotlib.pyplot as plt
import xarray as xr
from typing_extensions import Self

from AFL.double_agent.PipelineContext import PipelineContext, NoContextException


class PipelineOp(ABC):
    """Abstract base class for data processors. All operations in AFL.double_agent should inherit PipelineOpBase.

    Parameters
    ----------
    name : Optional[str] | List[str]
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
    input_variable : Optional[str] | List[str]
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : Optional[str] | List[str]
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    input_prefix : Optional[str] | List[str]
        Prefix for input variables when using pattern matching
    output_prefix : Optional[str] | List[str]
        Prefix for output variables when using pattern matching
    """

    def __init__(self,
                 name: Optional[str] | List[str] = None,
                 input_variable: Optional[str] | List[str] = None,
                 output_variable: Optional[str] | List[str] = None,
                 input_prefix: Optional[str] | List[str] = None,
                 output_prefix: Optional[str] | List[str] = None):

        if all(x is None for x in [input_variable, output_variable, input_prefix, output_prefix]):
            warnings.warn(
                'No input/output information set for PipelineOp...this is likely an error',
                stacklevel=2
            )

        if name is None:
            self.name = 'PipelineOp'
        else:
            self.name = name

        self.input_variable = input_variable
        self.output_variable = output_variable
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix

        self.output: Dict[str, xr.DataArray] = {}

        try:
            # try to add this object to current pipeline on context stack
            PipelineContext.get_context().append(self)
        except NoContextException:
            # silently continue for those working outside a context manager
            pass

        # variables to exclude when constructing attrs dict for xarray
        self._banned_from_attrs = ['output', '_banned_from_attrs']

    @abstractmethod
    def calculate(self, dataset: xr.Dataset) -> Self:
        pass

    def __repr__(self) -> str:
        return f'<PipelineOp:{self.name}>'

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def _prefix_output(self, variable_name: str) -> str:
        prefixed_variable = copy.deepcopy(variable_name)
        if self.output_prefix is not None:
            prefixed_variable = f'{self.output_prefix}_{prefixed_variable}'
        return prefixed_variable

    def _get_attrs(self) -> Dict:
        output_dict = copy.deepcopy(self.__dict__)
        for key in self._banned_from_attrs:
            try:
                del output_dict[key]
            except KeyError:
                pass

        #sanitize
        for key in output_dict.keys():
            output_dict[key] = str(output_dict[key])
            # if output_dict[key] is None:
            #     output_dict[key] = 'None'
            # elif type(output_dict[key]) is bool:
            #     output_dict[key] = str(output_dict[key])

        return output_dict

    def _get_variable(self, dataset: xr.Dataset) -> xr.DataArray:
        if self.input_variable is None and self.input_prefix is None:
            raise ValueError((
                """Can't get variable for {self.name} without input_variable """
                """or input_prefix specified in constructor """
            ))

        if self.input_variable is not None and self.input_prefix is not None:
            raise ValueError((
                """Both input_variable and input_prefix were specified in constructor. """
                """Only one should be specified to avoid ambiguous operation"""
            ))

        if self.input_variable is not None:
            output = dataset[self.input_variable].copy()

        elif self.input_prefix is not None:
            raise NotImplementedError
        else:
            output = None

        return output

    def add_to_dataset(self, dataset, copy_dataset=True):
        """Adds (xarray) data in output dictionary to provided xarray dataset"""
        if copy_dataset:
            dataset1 = dataset.copy()
        else:
            dataset1 = dataset

        for name, value in self.output.items():
            if isinstance(value, xr.Dataset):
                # add PipelineOp meta variables to attributes
                for data_var in value:
                    value[data_var].attrs.update(self._get_attrs())
                dataset1 = xr.merge([dataset1, value])
            elif isinstance(value, xr.DataArray):
                # add PipelineOp meta variables to attributes
                value.attrs.update(self._get_attrs())
                dataset1[name] = value
            else:
                raise ValueError((
                    f"""Items in output dictionary of PipelineOp {self.name} must be xr.Dataset or xr.DataArray """
                    f"""Found variable named {name} of type {type(value)}."""
                ))
        return dataset1

    def add_to_tiled(self, tiled_data):
        """Adds data in output dictionary to provided tiled catalogue"""
        raise NotImplementedError
        # This needs to handle/deconstruct xarray types!!
        # for name, dataarray in self.output.items():
        #     tiled_data.add_array(name, value.values)

    def plot(self,**mpl_kwargs) -> plt.Figure:
        n = len(self.output)
        if n>0:
            fig, axes = plt.subplots(n,1,figsize=(8,n*4))
            if n>1:
                axes = list(axes.flatten())
            else:
                axes = [axes]

            for i,(name,data) in enumerate(self.output.items()):
                if 'sample' in data.dims:
                    data = data.plot(hue='sample',ax=axes[i],**mpl_kwargs)
                else:
                    data.plot(ax=axes[i],**mpl_kwargs)
                axes[i].set(title=name)
            return fig
        else:
            return plt.figure()
