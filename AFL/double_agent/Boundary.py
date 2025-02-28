"""
Boundary analysis tools for phase diagram and compositional spaces.

This module provides tools for analyzing and comparing boundaries in phase diagrams
and compositional spaces. It includes functionality for creating concave hulls around
phase regions and computing similarity scores between different boundary sets.

Key features:
- Concave hull generation for phase regions
- Boundary comparison metrics
- Support for 2D compositional spaces
- Visualization tools for boundary analysis
- Handling of multi-phase systems
"""

from typing import Dict, List, Optional, Generator
from typing_extensions import Self

import numpy as np
import xarray as xr

from sklearn.metrics import pairwise_distances_argmin_min  # type: ignore

from shapely import MultiPoint, Point, concave_hull  # type: ignore
from shapely.geometry import mapping, shape  # type: ignore
from shapely.errors import GEOSException

from plotly import graph_objects as go

from ast import literal_eval

from AFL.double_agent.PipelineOp import PipelineOp


# Shapely types to skip over for boundary score
EXCLUDED_HULL_TYPES = ["Point", "LineString"]


class ConcaveHull(PipelineOp):
    """Creates concave hulls around points in 2D space.
    
    This class generates concave hulls (alpha shapes) around sets of points,
    useful for identifying phase boundaries in compositional spaces. It can
    handle multiple phases and provides options for hull generation parameters.

    Parameters
    ----------
    input_variable : str
        The name of the variable containing the points to create hulls around

    output_prefix : str
        Prefix to use for output variable names

    hull_tracing_ratio : float, default=0.2
        Controls the concavity of the hull. Lower values create tighter hulls

    drop_phases : Optional[List], default=None
        List of phase labels to exclude from hull generation

    component_dim : str, default="components"
        Name of the dimension containing component coordinates

    label_variable : Optional[str], default=None
        Name of variable containing phase labels. If None, all points are treated
        as one phase

    segmentize_length : float, default=0.025
        Length to use when segmenting hull boundaries for smoother visualization

    allow_holes : bool, default=False
        Whether to allow holes in the generated hulls

    name : str, default="ConcaveHull"
        Name to use when added to a Pipeline
    """

    def __init__(
        self,
        input_variable: str,
        output_prefix: str,
        hull_tracing_ratio: float = 0.2,
        drop_phases: Optional[List] = None,
        component_dim: str = "components",
        label_variable: Optional[str] = None,
        segmentize_length: float = 0.025,
        allow_holes: bool = False,
        name: str = "ConcaveHull",
    ) -> None:
        super().__init__(
            name=name,
            input_variable=input_variable,
            output_variable=output_prefix + "_" + "hulls",
            output_prefix=output_prefix,
        )

        self.hull_tracing_ratio = hull_tracing_ratio
        self.component_dim = component_dim
        self.label_variable = label_variable
        self.segmentize_length = segmentize_length
        self.label_variable = label_variable
        self.drop_phases = drop_phases
        self.allow_holes = allow_holes

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Generate concave hulls for each phase in the dataset.
        
        Creates concave hulls around points grouped by their phase labels.
        Handles 2D data only and includes options for hull parameters.

        Parameters
        ----------
        dataset : xr.Dataset
            The input dataset containing points and optional phase labels

        Returns
        -------
        Self
            The ConcaveHull instance with generated hulls

        Raises
        ------
        ValueError
            If the input data is not 2D
        """
        input_data = self._get_variable(dataset).transpose(..., self.component_dim)
        if input_data.sizes[self.component_dim] != 2:
            raise ValueError(
                """ConcaveHull only works for 2D data. If working with ternary, do a Barycentric transform first."""
            )

        if self.label_variable is None:
            dim_name = input_data.dims[0]
            labels = xr.DataArray(np.ones(input_data.sizes[dim_name]), dims=dim_name)
        else:
            labels = dataset[self.label_variable]

        if self.drop_phases is None:
            self.drop_phases = []

        hulls = {}
        for label, xy in input_data.groupby(labels):
            if label in self.drop_phases:
                continue
            self.xys = MultiPoint(xy.values)
            try:
                hull = concave_hull(
                    self.xys, ratio=self.hull_tracing_ratio, allow_holes=self.allow_holes
                )
            except GEOSException:
                hulls[label] = Point(
                    [-3.14, 3.14]
                )  # dummy point at pi-pi, "Point" type is in EXCLUDED_HULL_TYPES
            else:
                hull = hull.segmentize(self.segmentize_length)
                hulls[label] = hull

        # This is a little...gross, but I'm not sure of a better way to store the variable length arrays
        # associated with concave hulls. We could store each hull as its own DataArray but we would need
        # separate dimensions for each hull...which seems like a mess. The compromise is to store a string
        # representation of the hull
        hulls_serialized = {}
        for k, v in hulls.items():
            hulls_serialized[k] = str(mapping(v))
        hulls_xr = xr.Dataset(hulls_serialized).to_array(
            self._prefix_output("hull_labels")
        )

        self.output[self._prefix_output("hulls")] = hulls_xr
        self.output[self._prefix_output("hulls")].attrs[
            "description"
        ] = "String representation of shapely concave hulls"

        return self


def _get_xy(hull):
    """Extract xy coordinates from shapely object.
    
    Parameters
    ----------
    hull : shapely.geometry
        The shapely geometry object to extract coordinates from

    Returns
    -------
    tuple[list, list]
        The x and y coordinates of the hull boundary
    """
    try:
        x, y = hull.boundary.xy
    except NotImplementedError:
        if hasattr(hull.boundary, "geoms"):
            x = []
            y = []
            for geom in list(hull.boundary.geoms):
                xx, yy = geom.xy
                x.extend(xx)
                y.extend(yy)
        else:
            x, y = None, None
    return x, y


def _get_hulls(
    dataset: xr.Dataset, hull_variable: str
):  # -> Generator[object, (xr.Dataset,str), None]: # to-fix when I have wifi...
    """Iterate over a data variable of shapely hull strings and convert to shapely objects.

    Parameters
    ----------
    dataset: xr.Dataset
        Input dataset containing hull data

    hull_variable: str
        Name of variable containing a data variable of stringified shapely objects. Should have only
        one dimension with a coordinate of hull_labels.

    Yields
    ------
    tuple[str, shapely.geometry]
        Tuple of (hull_label, hull_object) for each hull in the dataset
    """
    dims = dataset[hull_variable].dims
    if len(dims) > 1:
        raise ValueError(
            f"The hulls variable {hull_variable} should only have one dimension. It has {dims}"
        )
    dim = dims[0]

    for hull_label in dataset[hull_variable][dim].values:
        hull_str = dataset[hull_variable].sel(**{dim: hull_label}).item()
        hull = shape(literal_eval(hull_str))
        yield hull_label, hull


def _calculate_perimeter_score(hull1, hull2):
    """Calculate a similarity score between two hull perimeters.
    
    Computes a score based on the average minimum distance between points
    on the perimeters of two hulls.

    Parameters
    ----------
    hull1 : shapely.geometry
        First hull to compare
    hull2 : shapely.geometry
        Second hull to compare

    Returns
    -------
    dict
        Dictionary containing score statistics and visualization data
    """
    hull1_xy = np.vstack(_get_xy(hull1)).T
    hull2_xy = np.vstack(_get_xy(hull2)).T

    idx1, dist1 = pairwise_distances_argmin_min(hull2_xy, hull1_xy, metric="euclidean")
    idx2, dist2 = pairwise_distances_argmin_min(hull1_xy, hull2_xy, metric="euclidean")

    # build index list of closest pairs
    idx1 = np.vstack([np.arange(idx1.shape[0]), idx1]).T
    idx2 = np.vstack([idx2, np.arange(idx2.shape[0])]).T

    dist = np.hstack([dist1, dist2])
    idx = np.vstack([idx1, idx2])

    # remove duplicate pairs from idx and dist
    unique = np.unique(idx, axis=0, return_index=True)[1]
    idx = idx[unique]
    dist = dist[unique]

    # build list of distance pair coordinates for plotting
    coord = []
    for i, j in idx:
        coord.extend([hull1_xy[j], hull2_xy[i], [np.nan, np.nan]])
    coord = np.array(coord)

    out = {
        "mean": dist.mean(),
        "std": dist.std(),
        "pair_dist": dist,
        "pair_idx": idx,
        "pair_coord": coord,
        "hull1_xy": hull1_xy,
        "hull2_xy": hull2_xy,
    }

    return out


def _make_boundary_plots(score_dict):
    """Create plotly visualization of hull boundaries and their comparison.
    
    Parameters
    ----------
    score_dict : dict
        Dictionary containing hull coordinates and comparison data

    Returns
    -------
    dict
        Dictionary of plotly traces for visualization
    """
    gt_xy = score_dict["hull1_xy"]
    xy = score_dict["hull2_xy"]
    pair_xy = score_dict["pair_coord"]

    out = {}
    out["GT"] = go.Scatter(
        x=gt_xy.T[0],
        y=gt_xy.T[1],
        mode="lines+markers",
        line={"color": "black"},
        marker={"symbol": "circle-open"},
        name="GT",
    )
    out["AL"] = go.Scatter(
        x=xy.T[0],
        y=xy.T[1],
        mode="lines+markers",
        line={"color": "green"},
        marker={"symbol": "triangle-up-open"},
        name="AL",
    )
    out["pairs"] = go.Scatter(
        x=pair_xy[:, 0], y=pair_xy[:, 1], line={"color": "red"}, opacity=0.5, name=None
    )
    return out


class BoundaryScore(PipelineOp):
    """Computes similarity scores between two sets of phase boundaries.
    
    This class calculates metrics comparing the boundaries of phase regions
    between two different datasets (e.g., ground truth vs. predicted phases).
    It provides both numerical scores and optional visualization.

    Parameters
    ----------
    gt_hull_variable : str
        Name of variable containing ground truth hull data

    al_hull_variable : str
        Name of variable containing comparison hull data

    output_prefix : str, default="boundary_score"
        Prefix to use for output variable names

    name : str, default="BoundaryScore"
        Name to use when added to a Pipeline

    plot : bool, default=False
        Whether to create visualizations of the boundary comparisons
    """

    def __init__(
        self,
        gt_hull_variable: str,
        al_hull_variable: str,
        output_prefix: str = "boundary_score",
        name: str = "BoundaryScore",
        plot: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            input_variable=[gt_hull_variable, al_hull_variable],
            output_variable=[output_prefix + "_" + i for i in ["mean", "std"]],
            output_prefix=output_prefix,
        )

        self.gt_hull_variable = gt_hull_variable
        self.al_hull_variable = al_hull_variable
        self.plot = plot

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Calculate boundary similarity scores between hull sets.
        
        Computes mean and standard deviation of distances between corresponding
        points on hull boundaries. Optionally creates visualizations of the
        comparison.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset containing both sets of hulls to compare

        Returns
        -------
        Self
            The BoundaryScore instance with computed scores
        """
        all_scores = []
        for gt_name, gt_hull in _get_hulls(dataset, self.gt_hull_variable):
            if gt_hull.geom_type in EXCLUDED_HULL_TYPES:
                continue
            for al_name, al_hull in _get_hulls(dataset, self.al_hull_variable):
                if al_hull.geom_type in EXCLUDED_HULL_TYPES:
                    continue

                try:
                    score = _calculate_perimeter_score(gt_hull, al_hull)
                except NotImplementedError:
                    continue

                score["GT"] = gt_name
                score["AL"] = al_name
                all_scores.append(score)

        # find best matches for each ground truth phase
        all_scores = sorted(all_scores, key=lambda x: x["mean"])
        best_scores: Dict = {}
        for score in all_scores:
            check1 = [
                (score["AL"] == value["AL"]) for key, value in best_scores.items()
            ]
            check2 = [
                (score["GT"] == value["GT"]) for key, value in best_scores.items()
            ]
            if not (any(check1) or any(check2)):
                best_scores[score["GT"]] = score

        if self.plot:
            for label, score_dict in best_scores.items():
                boundary_plots = _make_boundary_plots(score_dict)
                fig = go.FigureWidget(
                    layout={
                        "width": 500,
                        "height": 500,
                        "margin": dict(l=20, r=20, t=20, b=20),
                        "legend": dict(
                            yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor=None
                        ),
                    }
                )
                fig.add_trace(boundary_plots["GT"])
                fig.add_trace(boundary_plots["AL"])
                fig.add_trace(boundary_plots["pairs"])
                fig.show()

        self.output[self._prefix_output("mean")] = xr.Dataset(
            {k: v["mean"] for k, v in best_scores.items()}
        ).to_array("hull_label")
        self.output[self._prefix_output("mean")].attrs[
            "description"
        ] = "Mean of boundary score for all matched phases"

        self.output[self._prefix_output("std")] = xr.Dataset(
            {k: v["std"] for k, v in best_scores.items()}
        ).to_array("hull_label")
        self.output[self._prefix_output("std")].attrs[
            "description"
        ] = "Standard deviation of boundary score for all matched phases"

        return self
