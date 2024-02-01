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
import pickle
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import xarray as xr
import networkx as nx

from AFL.double_agent.util import listify
from AFL.double_agent.Context import PipelineContext,NoContextException


class Pipeline(PipelineContext):
    """
    Container class for defining and building pipelines.
    """

    def __init__(self, name=None, ops=None):
        if ops is None:
            self.ops = []
        else:
            self.ops = ops

        if name is None:
            self.name = "Pipeline"
        else:
            self.name = name

        # placeholder for networkx graph
        self.graph = None

    def __iter__(self):
        """ALlows pipelines to be iterated over"""
        for op in self.ops:
            yield op

    def __getitem__(self, i):
        """Allows PipelineOps to be viewed from the pipeline via an index"""
        return self.ops[i]

    def __repr__(self):
        return f'<Pipeline {self.name} N={len(self.ops)}>'

    def print(self):
        """Print a summary of the pipeline"""
        print(f"{'i':>3s}) {'PipelineOp':35s} {'input_variable'} ---> {'output_variable'}")
        print(f"{'--':>4s} {'-' * 10:35s} {'-' * 14}      {'-' * 15}")
        for i, op in enumerate(self):
            print(f"{i:3d}) {'<' + op.name + '>':35s} {op.input_variable} ---> {op.output_variable}")


    def append(self, op):
        """Mirrors the behavior of python lists"""
        self.ops.append(op)
        return self

    def extend(self, op):
        """Mirrors the behavior of python lists"""
        self.ops.extend(op)
        return self

    def clear_outputs(self):
        """Clears the ouptut dict of all PipelineOps in this pipeline"""
        for op in self:
            op.output = {}

    def copy(self):
        return copy.deepcopy(self)

    def write(self, filename):
        """Write pipeline to disk as a pkl

        Parameters
        ----------
        filename: str
            Filename or filepath to be written
        """
        pipeline = self.copy()
        pipeline.clear_outputs()

        with open(filename, 'wb') as f:
            pickle.dump(pipeline, f)

    @staticmethod
    def read(filename):
        """Read pipeline from pickle file on disk


        Usage
        -----
        ```python
        from AFL.double_agent.Pipeline import Pipeline
        pipeline1 = Pipeline.read('pickled_pipeline.pkl')
        ````

        """
        with open(filename, 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline

    def make_graph(self):
        """Build a networkx graph representation of this pipeline"""

        self.graph = nx.DiGraph()
        self.graph_edge_labels = {}
        for op in self:
            output_variable = op.output_variable
            if output_variable is None:
                continue
            self.graph.add_node(output_variable)
            # need to handle case where input_variables is a list
            for input_variable in listify(op.input_variable):
                if input_variable is None:
                    continue
                self.graph.add_node(input_variable)
                self.graph.add_edge(input_variable, output_variable)
                self.graph_edge_labels[input_variable, output_variable] = op.name

    def draw(self, figsize=(8, 8), edge_labels=True):
        """Draw the pipeline as a graph"""
        self.make_graph()

        plt.figure(figsize=figsize)
        pos = nx.nx_agraph.pygraphviz_layout(self.graph, prog='dot')
        nx.draw(self.graph, with_labels=True, pos=pos, node_size=1000)
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos, self.graph_edge_labels)

    def input_variables(self):
        """Get the input variables needed for the pipeline to fully execute

        Warning
        -------
        This list will currently include "Generators" that aren't required to be in the pipeline. These will hopefully
        be distinguishable by having Generator in the name, but this isn't enforced at the moment.
        """
        self.make_graph()
        return [n for n, d in self.graph.in_degree() if d == 0]

    def output_variables(self):
        """Get the outputs variables of the pipeline"""
        self.make_graph()
        return [n for n, d in self.graph.out_degree() if d == 0]

    def validate(self):
        raise NotImplementedError

    def calculate(self, dataset, tiled_data=None):
        """Execute all operations in pipeline on provided dataset"""
        dataset1 = dataset.copy()

        for op in self.ops:
            op.calculate(dataset1)
            dataset1 = op.add_to_dataset(dataset1, copy_dataset=False)

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

        #try to add to context
        try:
            Context.get_context().append(self)
        except NoContextException:
            pass  #silently continue


        # variables to exclude when constructing attrs dict for xarray
        self._banned_from_attrs = ['output', '_banned_from_attrs']

    @abstractmethod
    def calculate(self, dataset):
        pass

    def __repr__(self):
        return f'<PipelineOp:{self.name}>'

    def copy(self):
        return copy.deepcopy(self)

    def _prefix_output(self,variable_name):
        prefixed_variable = copy.deepcopy(variable_name)
        if self.output_prefix is not None:
            prefixed_variable = self.output_prefix + '_' + prefixed_variable
        return prefixed_variable

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
