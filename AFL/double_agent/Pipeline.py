"""
All operators/calculations in agent should be formatted as PipelinesOps double_agent


All PipelineOps should:
- take input_variable and output_variable in their constructor
- have a .calculate method
    - with exact signature as base class (PipelineOpBase)
    - that  writes an xr.Dataset or xr.DataArray (preferred)
    - that returns self

"""
import copy
import warnings
from abc import ABC, abstractmethod

import xarray as xr

from AFL.double_agent.util import listify


class Pipeline:
    """
    Container class for defining and building pipelines.
    """
    def __init__(self,ops=None):
        if ops is None:
            self.ops = []
        else:
            self.ops = ops


    def __iter__(self):
        for op in self.ops:
            yield op

    def __repr__(self):
        return f'<Pipeline N={len(self.ops)}>'

    def append(self, op):
        self.ops.append(op)
        return self

    def copy(self):
        return copy.deepcopy(self)


    def draw(self):
        import networkx as nx
        G = nx.DiGraph()
        for op in self:
            output = op.output_variable
            G.add_node(output)
            # need to handle case where input_variables is a list
            for input in listify(op.input_variable):
                G.add_node(input)
                G.add_edge(input, op.output_variable)

        pos = nx.nx_agraph.pygraphviz_layout(G, prog='dot')
        nx.draw(G, with_labels=True, pos=pos)

    def validate(self):
        raise NotImplementedError

    def calculate(self, dataset, copy=True,tiled_data=None):
        """Execute all operations in pipeline on provided dataset"""
        if copy:
            dataset1 = dataset.copy()
        else:
            dataset1 = dataset

        for op in self.ops:
            op.calculate(dataset1)
            dataset1 = op.add_to_dataset(dataset1,copy=False)

            if tiled_data is not None:
                op.add_to_tiled(tiled_data)
        return dataset1


class PipelineOpBase(ABC):
    """
    Abstract base class for data processors. All operations in AFL.agent should inherit PipelineOpBase.
    """

    def __init__(self, name=None, input_variable=None, output_variable=None, input_prefix=None, output_prefix=None):
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

        self.output = {}

        # variables to exclude when constructing attrs dict for xarray
        self._banned_from_attrs = ['output']


    @abstractmethod
    def calculate(self, data):
        pass

    def __repr__(self):
        return f'<PipelineOp:{self.name}>'

    def _get_attrs(self):
        output_dict = copy.deepcopy(self.__dict__)
        for key in self._banned_from_attrs:
            try:
                del output_dict[key]
            except KeyError:
                pass
        return output_dict

    def _get_variable(self, data):
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
            output = data[self.input_variable].copy()

        elif self.input_prefix is not None:
            raise NotImplementedError

        return output

    def copy(self):
        return copy.deepcopy(self)

    def add_to_dataset(self, dataset,copy=True):
        """Adds (xarray) data in output dictionary to provided xarray dataset"""
        if copy:
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
