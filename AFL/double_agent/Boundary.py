from typing import Dict, List, Optional
from typing_extensions import Self

import numpy as np
import xarray as xr

from sklearn.metrics import pairwise_distances_argmin_min  # type: ignore

from shapely import MultiPoint, concave_hull  # type: ignore
from shapely.geometry import mapping, shape  # type: ignore

from ast import literal_eval

from AFL.double_agent.PipelineOp import PipelineOp


class ConcaveHull(PipelineOp):
    def __init__(
        self,
        input_variable: str,
        output_prefix: str,
        hull_tracing_ratio: float = 0.2,
        drop_phases: Optional[List] = None,
        component_dim: str = "components",
        label_variable: Optional[str] = None,
        segmentize_length: float = 0.025,
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

    def calculate(self, dataset: xr.Dataset) -> Self:
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
            hull = concave_hull(
                self.xys, ratio=self.hull_tracing_ratio, allow_holes=True
            )
            hull = hull.segmentize(self.segmentize_length)
            hulls[label] = hull

        # This is a little...gross but I'm not sure of a better way to store the variable length arrays
        # associated with concave hulls. We could store each hull as its own dataarray but we would need
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
    try:
        x, y = hull.boundary.xy
    except NotImplementedError:
        if hasattr(hull, "geoms"):
            x = []
            y = []
            for geom in list(hull.boundary.geoms):
                xx, yy = geom.xy
                x.extend(xx)
                y.extend(yy)
        else:
            x, y = None, None
    return x, y


def _calculate_perimeter_score(hull1, hull2):
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


class BoundaryScore(PipelineOp):
    def __init__(
        self,
        gt_hull_variable: str,
        al_hull_variable: str,
        output_prefix: str = "boundary_score",
        name: str = "BoundaryScore",
    ) -> None:
        super().__init__(
            name=name,
            input_variable=[gt_hull_variable, al_hull_variable],
            output_variable=[output_prefix + "_" + i for i in ["mean", "std"]],
            output_prefix=output_prefix,
        )

        self.gt_hull_variable = gt_hull_variable
        self.al_hull_variable = al_hull_variable

    def calculate(self, dataset: xr.Dataset) -> Self:
        gt_hulls = dataset[self.gt_hull_variable]
        al_hulls = dataset[self.al_hull_variable]

        all_scores = []
        for da_gt_hull in gt_hulls:
            gt_name = list(da_gt_hull.coords.values())[0].item()  # this should be cleaned up...
            gt_hull_str = da_gt_hull.item()
            gt_hull = shape(literal_eval(gt_hull_str))

            for da_al_hull in al_hulls:
                al_name = list(da_al_hull.coords.values())[0].item()  # this should be cleaned up...
                al_hull_str = da_al_hull.item()
                al_hull = shape(literal_eval(al_hull_str))

                if al_hull.geom_type == "Point":
                    continue
                elif al_hull.geom_type == "LineString":
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