"""
Pipeline module for the double_agent package.

This module provides the Pipeline class, which serves as a container for defining
and executing computational workflows. Pipelines are composed of PipelineOp objects
that perform specific operations on data.

PipelineOps in this system follow these conventions:
- Each operator takes input_variable and output_variable parameters in its constructor
- Each operator implements a .calculate method that:
  - Maintains the same signature as the base class (PipelineOp)
  - Writes results to an xarray Dataset or DataArray
  - Returns self for method chaining
"""

import copy
import pickle
import re
from typing import Generator, Optional, List
import warnings
import datetime
import json
import pathlib

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import xarray as xr
from tqdm.auto import tqdm
from typing_extensions import Self

from AFL.double_agent.PipelineContext import PipelineContext
from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify


class Pipeline(PipelineContext):
    """
    Container class for defining and executing computational workflows.

    The Pipeline class serves as a framework for organizing and running sequences of
    operations (PipelineOps) on data. Each operation in the pipeline takes input data,
    performs a specific transformation, and produces output data that can be used by
    subsequent operations.

    Parameters
    ----------
    name : Optional[str], default=None
        Name of the pipeline. If None, defaults to "Pipeline".
    ops : Optional[List], default=None
        List of PipelineOp objects to initialize the pipeline with.

    Attributes
    ----------
    result : Any
        Stores the final result after pipeline execution
    ops : List
        List of PipelineOp objects in the pipeline
    graph : nx.DiGraph
        NetworkX directed graph representation of the pipeline
    graph_edge_labels : Dict
        Edge labels for the pipeline graph visualization
    """

    def __init__(
        self,
        name: Optional[str] = None,
        ops: Optional[List] = None,
        description: Optional[str] = None,
    ) -> None:
        self.result = None
        self.ops = ops or []
        self.description = str(description)
        self.name = name or "Pipeline"

        # placeholder for networkx graph
        self.graph = None
        self.graph_edge_labels = None

    def __iter__(self) -> Generator[PipelineOp, Self, None]:
        """Allows pipelines to be iterated over"""
        for op in self.ops:
            yield op

    def __getitem__(self, i: int) -> PipelineOp:
        """Allows PipelineOps to be viewed from the pipeline via an index"""
        return self.ops[i]

    def __repr__(self) -> str:
        return f"<Pipeline {self.name} N={len(self.ops)}>"

    def search(self, name: str, contains: bool = False) -> Optional[PipelineOp]:
        for op in self:
            if contains and (name in op.name):
                return op
            elif name == op.name:
                return op
        return None

    def print(self) -> None:
        """Print a summary of the pipeline"""
        print(f"{'PipelineOp':40s} {'input_variable'} ---> {'output_variable'}")
        print(f"{'-' * 10:40s} {'-' * 35}")
        for i, op in enumerate(self):
            print(
                f"{i:<3d}) {'<' + op.name + '>':35s} {op.input_variable} ---> {op.output_variable}"
            )

        print()
        print("Input Variables")
        print("---------------")
        for i, data in enumerate(self.input_variables()):
            print(f"{i}) {data}")

        print()
        print("Output Variables")
        print("----------------")
        for i, data in enumerate(self.output_variables()):
            print(f"{i}) {data}")

    def print_code(self) -> None:
        """String representation of approximate code to generate this pipeline

        Run this method to produce a string of Python code that should
        recreate this Pipeline.

        """

        output_string = f'with Pipeline(name = "{self.name}") as p:\n'

        for op in self:
            args = op._stored_args
            output_string += f"    {type(op).__name__}(\n"
            for k, v in args.items():
                if isinstance(v, str):
                    output_string += f'        {k}="{v}",\n'
                else:
                    output_string += f"        {k}={v},\n"
            output_string += f"    )\n\n"

        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                # Create IPython magic for creating executable code
                ip = get_ipython()

                # First display the code with syntax highlighting for visibility
                from IPython.display import display, Code

                # display(Code(output_string, language="python"))

                # Define a temporary magic to create a new cell with the code
                ip.set_next_input(output_string)
                print("Pipeline code has been prepared in a new cell below.")
            else:
                print(output_string)
        except NameError:
            print(output_string)

    def append(self, op: PipelineOp) -> Self:
        """Mirrors the behavior of python lists"""
        self.ops.append(op)
        return self

    def extend(self, op: List[PipelineOp]) -> Self:
        """Mirrors the behavior of python lists"""
        self.ops.extend(op)
        return self

    def clear_outputs(self):
        """Clears the output dict of all PipelineOps in this pipeline"""
        for op in self:
            op.output = {}

    def copy(self) -> Self:
        return copy.deepcopy(self)

    def write_json(
        self, filename: str, overwrite=False, description: Optional[str] = None
    ):
        """Write pipeline to disk as a JSON

        Parameters
        ----------
        filename: str
            Filename or filepath to be written
        overwrite: bool, default=False
            Whether to overwrite an existing file
        description: str, optional
            A descriptive text about the pipeline's purpose and functionality
        """

        if not overwrite and pathlib.Path(filename).exists():
            raise FileExistsError()

        pipeline_dict = {
            "name": self.name,
            "date": datetime.datetime.now().strftime("%m/%d/%y %H:%M:%S-%f"),
            "description": (
                str(description) if description is not None else self.description
            ),
            "ops": [op.to_json() for op in self],
        }

        with open(filename, "w") as f:
            json.dump(pipeline_dict, f, indent=1)

        print(f"Pipeline successfully written to {filename}.")

    @staticmethod
    def read_json(filename: str):
        """Read pipeline from json file on disk

        Usage
        -----
        ```python
        from AFL.double_agent.Pipeline import Pipeline
        pipeline1 = Pipeline.read_json('pickled_pipeline.pkl')
        ````
        """
        with open(filename, "r") as f:
            pipeline_dict = json.load(f)

        pipeline = Pipeline(
            name=pipeline_dict["name"],
            ops=[PipelineOp.from_json(op) for op in pipeline_dict["ops"]],
            description=pipeline_dict["description"],
        )

        return pipeline

    def write_pkl(self, filename: str):
        """Write pipeline to disk as a pkl

        .. warning::
            Please use the read_json and write_json methods. The pickle methods
            are insecure and prone to errors.

        Parameters
        ----------
        filename: str
            Filename or filepath to be written
        """
        pipeline = self.copy()
        pipeline.clear_outputs()

        with open(filename, "wb") as f:
            pickle.dump(pipeline, f)

    @staticmethod
    def read_pkl(filename: str):
        """Read pipeline from pickle file on disk

        .. warning::
            Please use the  `read_json`  and `write_json` methods. The pickle methods
            are insecure and prone to errors.

        Usage
        -----
        ```python
        from AFL.double_agent.Pipeline import Pipeline
        pipeline1 = Pipeline.read('pickled_pipeline.pkl')
        ````

        """
        with open(filename, "rb") as f:
            pipeline = pickle.load(f)
        return pipeline

    def make_graph(self):
        """Build a networkx graph representation of this pipeline"""

        self.graph = nx.DiGraph()
        self.graph_edge_labels = {}
        for op in self:
            for output_variable in listify(op.output_variable):
                if output_variable is None:
                    continue
                self.graph.add_node(output_variable)
                # need to handle case where input_variables is a list
                for input_variable in listify(op.input_variable):
                    if input_variable is None:
                        continue
                    self.graph.add_node(input_variable)
                    self.graph.add_edge(input_variable, output_variable, name=op.name)
                    self.graph_edge_labels[input_variable, output_variable] = op.name

    def draw(self, figsize=(8, 8), edge_labels=True):
        """Draw the pipeline as a graph"""
        self.make_graph()

        fig = plt.figure(figsize=figsize)
        pos = nx.nx_agraph.pygraphviz_layout(self.graph, prog="dot")
        nx.draw(self.graph, with_labels=True, pos=pos, node_size=1000)
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos, self.graph_edge_labels)
        return fig

    def draw_plotly(self):
        import plotly.graph_objects as go

        self.make_graph()
        pos = nx.nx_agraph.pygraphviz_layout(self.graph, prog="dot")

        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            marker=dict(reversescale=True, color=[], size=20, line_width=2),
        )

        node_text = []
        regex = re.compile("[\[\]']")
        for node in self.graph.nodes():
            VarIn = np.unique(list(n[0] for n in self.graph.in_edges(node)))
            VarOut = np.unique(list(n[0] for n in self.graph.out_edges(node)))
            OpIn = np.unique([self.graph_edge_labels[i, node] for i in VarIn])

            OpIn = regex.sub("", str(OpIn)).replace(" ", ", ")
            VarIn = regex.sub("", str(VarIn)).replace(" ", ", ")
            VarOut = regex.sub("", str(VarOut)).replace(" ", ", ")
            node_text.append(
                f"{'Node: ':9s}{node}<br>{'OpIn: ':9s}{OpIn}<br>{'VarIn: ':9s}{VarIn}<br>{'VarOut: ':9s}{VarOut}"
            )

        node_trace.text = node_text

        fig = go.Figure([node_trace, edge_trace])
        fig.update_layout(
            height=750,
            width=750,
            showlegend=False,
            template="simple_white",
            xaxis={"visible": False},
            yaxis={"visible": False},
            margin=dict(l=20, r=20, t=20, b=20),
        )
        return fig

    def input_variables(self) -> List[str]:
        """Get the input variables needed for the pipeline to fully execute

        Warning
        -------
        This list will currently include "Generators" that aren't required to be in the pipeline. These will hopefully
        be distinguishable by having Generator in the name, but this isn't enforced at the moment.
        """
        self.make_graph()
        return [n for n, d in self.graph.in_degree() if d == 0]  # type: ignore

    def output_variables(self) -> List[str]:
        """Get the outputs variables of the pipeline"""
        self.make_graph()
        return [n for n, d in self.graph.out_degree() if d == 0]  # type: ignore

    def calculate(
        self, dataset: xr.Dataset, tiled_data=None, disable_progress_bar: bool = False
    ) -> xr.Dataset:
        """Execute all operations in pipeline on provided dataset"""
        dataset1 = dataset.copy()

        for op in (pbar := tqdm(self.ops, disable=disable_progress_bar)):
            pbar.set_postfix_str(f"{op.name:25s}")
            op.calculate(dataset1)
            dataset1 = op.add_to_dataset(dataset1, copy_dataset=False)

            if tiled_data is not None:
                op.add_to_tiled(tiled_data)

        self.result = dataset1
        return dataset1
