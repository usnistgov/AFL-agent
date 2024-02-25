"""
All operators/calculations in agent should be formatted as PipelinesOps double_agent


All PipelineOps should:
- take input_variable and output_variable in their constructor
- have a .calculate method
    - with exact signature as base class (PipelineOpBase)
    - that writes a xr.Dataset or xr.DataArray (preferred)
    - that returns self

"""

import copy
import pickle
import re
from typing import Generator, Optional, List

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
    Container class for defining and building pipelines.
    """

    def __init__(self, name: Optional[str] = None, ops: Optional[List] = None) -> None:
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

    def write(self, filename: str):
        """Write pipeline to disk as a pkl

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
    def read(filename: str):
        """Read pipeline from pickle file on disk

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

        plt.figure(figsize=figsize)
        pos = nx.nx_agraph.pygraphviz_layout(self.graph, prog="dot")
        nx.draw(self.graph, with_labels=True, pos=pos, node_size=1000)
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos, self.graph_edge_labels)

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

    def calculate(
        self, dataset: xr.Dataset, tiled_data=None, disable_progress_bar=False
    ) -> xr.Dataset:
        """Execute all operations in pipeline on provided dataset"""
        dataset1 = dataset.copy()

        for op in (pbar := tqdm(self.ops, disable=disable_progress_bar)):
            pbar.set_postfix_str(f"{op.name:25s}")
            op.calculate(dataset1)
            dataset1 = op.add_to_dataset(dataset1, copy_dataset=False)

            if tiled_data is not None:
                op.add_to_tiled(tiled_data)
        return dataset1
