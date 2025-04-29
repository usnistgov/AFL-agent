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

    def draw_plotly(self, data_as_edges: bool = False):
        """Draw an interactive visualization of the pipeline graph using plotly.

        Parameters:
            data_as_edges (bool): If True, shows data variables as edges and transforms as nodes.
                                If False (default), shows data variables as nodes and transforms as edges.

        Returns:
            plotly.graph_objects.Figure: Interactive plotly figure
        """
        import plotly.graph_objects as go

        if not data_as_edges:
            # Original behavior - data as nodes, transforms as edge labels
            self.make_graph()
            G = self.graph.copy()
        else:
            # New behavior - transforms as nodes, data as edge labels
            G = nx.DiGraph()
            # Create nodes for each unique transform
            transform_nodes = set()
            for op in self:
                if op.name not in transform_nodes:
                    G.add_node(op.name)
                    transform_nodes.add(op.name)

                # Add edges for each input-output pair
                for input_var in listify(op.input_variable):
                    if input_var is not None:
                        # Find the transform that produces this input
                        producer = None
                        for prev_op in self:
                            if input_var in listify(prev_op.output_variable):
                                producer = prev_op.name
                                break

                        if producer:
                            G.add_edge(producer, op.name, name=input_var)
                        else:
                            # This is an input variable with no producer
                            input_node = f"INPUT: {input_var}"
                            G.add_node(input_node)
                            G.add_edge(input_node, op.name, name=input_var)

                # Add edges for outputs that aren't consumed
                for output_var in listify(op.output_variable):
                    if output_var is not None:
                        # Check if this output is consumed
                        is_consumed = False
                        for next_op in self:
                            if output_var in listify(next_op.input_variable):
                                is_consumed = True
                                break

                        if not is_consumed:
                            output_node = f"OUTPUT: {output_var}"
                            G.add_node(output_node)
                            G.add_edge(op.name, output_node, name=output_var)

        pos = nx.nx_agraph.pygraphviz_layout(G, prog="dot")

        # Draw edges
        edge_x = []
        edge_y = []
        edge_text = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            if data_as_edges:
                edge_text.extend(
                    [edge[2].get("name", ""), edge[2].get("name", ""), None]
                )

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#666666"),
            hoverinfo="text" if data_as_edges else "none",
            text=edge_text if data_as_edges else None,
            mode="lines",
        )

        # Draw nodes
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        # Prepare node colors and text based on node type
        if not data_as_edges:
            # Original behavior - color nodes based on their role in data flow
            root_nodes = set(n for n, d in G.in_degree() if d == 0)
            leaf_nodes = set(n for n, d in G.out_degree() if d == 0)

            node_colors = []
            node_text = []
            for node in G.nodes():
                if node in root_nodes:
                    node_colors.append("#c6dcff")
                elif node in leaf_nodes:
                    node_colors.append("#f0f5ff")
                else:
                    node_colors.append("#e1edff")

                # Create hover text with operation information
                VarIn = list(n[0] for n in G.in_edges(node))
                VarOut = list(n[0] for n in G.out_edges(node))
                OpIn = [G.edges[i, node].get("name", "") for i in VarIn]

                node_text.append(
                    f"Node: {node}<br>"
                    f"OpIn: {', '.join(str(op) for op in OpIn)}<br>"
                    f"VarIn: {', '.join(str(v) for v in VarIn)}<br>"
                    f"VarOut: {', '.join(str(v) for v in VarOut)}"
                )
        else:
            # New behavior - color nodes based on their type (input, transform, output)
            node_colors = []
            node_text = []
            for node in G.nodes():
                if str(node).startswith("INPUT:"):
                    node_colors.append("#c6dcff")
                elif str(node).startswith("OUTPUT:"):
                    node_colors.append("#f0f5ff")
                else:
                    node_colors.append("#e1edff")

                # Create hover text with transform information
                in_edges = list(G.in_edges(node, data=True))
                out_edges = list(G.out_edges(node, data=True))

                node_text.append(
                    f"Transform: {node}<br>"
                    f"Inputs: {', '.join(e[2].get('name', '') for e in in_edges)}<br>"
                    f"Outputs: {', '.join(e[2].get('name', '') for e in out_edges)}"
                )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=list(G.nodes()),
            textposition="bottom center",
            marker=dict(
                color=node_colors,
                size=30 if data_as_edges else 20,
                line=dict(width=2, color="#666666"),
            ),
            hovertext=node_text,
        )

        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace])

        # Update layout
        fig.update_layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            height=750,
            width=750,
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

    def draw(
        self,
        orientation: str = "TB",
        show_labels: bool = True,
        show_edge_labels: bool = False,
        data_as_edges: bool = False,
        figsize=(12, 8),
    ):
        """Draw a new visualization of the pipeline graph using dot layout.

        Parameters:
            orientation (str): Layout orientation. 'TB' for top-to-bottom (default) or 'LR' for left-to-right.
            show_labels (bool): Whether to display node labels. Defaults to True.
            show_edge_labels (bool): Whether to display edge labels (operation names). Defaults to False.
            data_as_edges (bool): If True, shows data variables as edges and transforms as nodes.
                                If False (default), shows data variables as nodes and transforms as edges.
            figsize (tuple): Figure size for the matplotlib figure.

        Returns:
            matplotlib.figure.Figure: The drawn figure.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np

        if not data_as_edges:
            # Original behavior - data as nodes, transforms as edge labels
            self.make_graph()
            G = self.graph.copy()
        else:
            # New behavior - transforms as nodes, data as edge labels
            G = nx.DiGraph()
            # Create nodes for each unique transform
            transform_nodes = set()
            for op in self:
                if op.name not in transform_nodes:
                    G.add_node(op.name)
                    transform_nodes.add(op.name)

                # Add edges for each input-output pair
                for input_var in listify(op.input_variable):
                    if input_var is not None:
                        # Find the transform that produces this input
                        producer = None
                        for prev_op in self:
                            if input_var in listify(prev_op.output_variable):
                                producer = prev_op.name
                                break

                        if producer:
                            G.add_edge(producer, op.name, name=input_var)
                        else:
                            # This is an input variable with no producer
                            input_node = f"INPUT: {input_var}"
                            G.add_node(input_node)
                            G.add_edge(input_node, op.name, name=input_var)

                # Add edges for outputs that aren't consumed
                for output_var in listify(op.output_variable):
                    if output_var is not None:
                        # Check if this output is consumed
                        is_consumed = False
                        for next_op in self:
                            if output_var in listify(next_op.input_variable):
                                is_consumed = True
                                break

                        if not is_consumed:
                            output_node = f"OUTPUT: {output_var}"
                            G.add_node(output_node)
                            G.add_edge(op.name, output_node, name=output_var)

        # Set the graph direction
        prog = "dot"
        if orientation.upper() == "TB":
            graph_orientation = "dot"  # Default is TB
        else:
            graph_orientation = "dot -Grankdir=LR"  # Force left-to-right

        # Get the dot layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog=graph_orientation)
        except ImportError:
            raise ImportError("This layout requires installing pygraphviz or pydot.")

        # Create the plot with adjusted figsize if edge labels are shown
        if show_edge_labels:
            figsize = (figsize[0] * 1.2, figsize[1] * 1.2)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # Draw edges
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color="#666666",
            width=1.0,
            alpha=0.8,
            arrows=True,
            arrowsize=15,
            ax=ax,
            node_size=2000,  # This affects arrow positioning
        )

        # Draw edge labels if requested or if showing data as edges
        if show_edge_labels or data_as_edges:
            edge_labels = nx.get_edge_attributes(G, "name")
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_size=8,
                font_color="#666666",
                alpha=0.7,
                label_pos=0.5,  # Centered on the edge
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
                ax=ax,
            )

        if not data_as_edges:
            # Original behavior - color nodes based on their role in the data flow
            root_nodes = [n for n, d in G.in_degree() if d == 0]
            leaf_nodes = [n for n, d in G.out_degree() if d == 0]
            middle_nodes = list(set(G.nodes()) - set(root_nodes) - set(leaf_nodes))

            if root_nodes:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=root_nodes,
                    node_color="#c6dcff",
                    node_size=2000,
                    edgecolors="#666666",
                    linewidths=2,
                    ax=ax,
                )
            if middle_nodes:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=middle_nodes,
                    node_color="#e1edff",
                    node_size=2000,
                    edgecolors="#666666",
                    linewidths=1.5,
                    ax=ax,
                )
            if leaf_nodes:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=leaf_nodes,
                    node_color="#f0f5ff",
                    node_size=2000,
                    edgecolors="#666666",
                    linewidths=1.5,
                    ax=ax,
                )
        else:
            # New behavior - color nodes based on their type (input, transform, output)
            input_nodes = [n for n in G.nodes() if str(n).startswith("INPUT:")]
            output_nodes = [n for n in G.nodes() if str(n).startswith("OUTPUT:")]
            transform_nodes = list(
                set(G.nodes()) - set(input_nodes) - set(output_nodes)
            )

            if input_nodes:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=input_nodes,
                    node_color="#c6dcff",
                    node_size=2000,
                    edgecolors="#666666",
                    linewidths=2,
                    ax=ax,
                )
            if transform_nodes:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=transform_nodes,
                    node_color="#e1edff",
                    node_size=2500,
                    edgecolors="#666666",
                    linewidths=1.5,
                    ax=ax,
                )
            if output_nodes:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=output_nodes,
                    node_color="#f0f5ff",
                    node_size=2000,
                    edgecolors="#666666",
                    linewidths=1.5,
                    ax=ax,
                )

        if show_labels:
            if data_as_edges:
                # Adjust label positions slightly for better readability
                label_pos = {k: (v[0], v[1] - 0.01) for k, v in pos.items()}
            else:
                label_pos = pos
            nx.draw_networkx_labels(
                G,
                label_pos,
                font_size=10,
                font_color="#333333",
                font_weight="bold",
                ax=ax,
            )

        # Remove axis and set tight layout
        ax.axis("off")
        plt.tight_layout()

        return fig
