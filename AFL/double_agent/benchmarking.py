"""Benchmarking ops: score an active-learned model against a known ground truth.

In a simulated campaign the "instrument" knows the true phase map, so the model the
pipeline fits can be scored after every iteration. :class:`PhaseMapF1Score` does this
for a classifier: it draws a phase-stratified test set from the design space, labels it
with the ground truth, predicts it with the model an upstream extrapolator wrote to the
dataset and reports the F1 score, accuracy and per-phase recall as dataset variables.

The ground truth is either a *callable* ``X -> labels`` or a
:class:`~AFL.double_agent.VirtualInstruments.VirtualSASInstrument` (its traced hulls
label the points, see :func:`virtual_instrument_labels`). The model comes from a model
ref (see :mod:`~AFL.double_agent.torch_io`), so the extrapolator must be given a
``model_uri``.

Typical use, after a ``DirichletGPExtrapolator(..., output_prefix="phase", model_uri=...)``::

    PhaseMapF1Score(
        ground_truth=sim,                       # virtual instrument or callable X -> labels
        bounds={c: LIMITS[c] for c in COMPONENTS},  # feature order of the model
        model_variable="phase_model",
        average="weighted",
        n_per_phase=100,
        seed=0,
    )
    ...
    ds = pipeline.calculate(ds_running)
    print(ds.benchmark_f1.item(), ds.benchmark_accuracy.item())
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr
from scipy.stats import qmc
from sklearn.metrics import f1_score
from typing_extensions import Self

from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.spec import resolve_function

__all__ = ["PhaseMapF1Score", "latin_hypercube", "virtual_instrument_labels", "stratified_test_set"]

Bounds = Mapping[str, Tuple[float, float]]


def latin_hypercube(bounds: Bounds, n_samples: int, seed: int = 0) -> np.ndarray:
    """Latin hypercube sample of the box ``bounds``; columns follow the key order of ``bounds``."""
    names = list(bounds)
    unit = qmc.LatinHypercube(d=len(names), seed=seed).random(n_samples)
    lo = np.array([bounds[c][0] for c in names], dtype=float)
    hi = np.array([bounds[c][1] for c in names], dtype=float)
    return qmc.scale(unit, lo, hi)


def virtual_instrument_labels(instrument, X: np.ndarray) -> np.ndarray:
    """Ground-truth labels of a :class:`~AFL.double_agent.VirtualInstruments.VirtualSASInstrument` at the rows of ``X``.

    Vectorised equivalent of calling ``instrument.label_for`` row by row: a point gets the
    first hull (in ``instrument.hulls`` order) that contains it within ``locate_tolerance``,
    otherwise the hull with the largest signed distance (the nearest). Columns of ``X`` must
    follow ``instrument.components``. Falls back to ``label_for`` for instruments without hulls.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    hulls = getattr(instrument, "hulls", None)
    if not hulls:
        return np.asarray([instrument.label_for(row)[0] for row in X])
    from trimesh.proximity import signed_distance

    points = instrument.scaler.transform(X)
    tolerance = float(instrument.config["locate_tolerance"])
    phases = list(hulls)
    dist = np.stack([np.asarray(signed_distance(hulls[p], points), dtype=float) for p in phases])  # (phase, n)
    inside = (dist > 0) | (np.abs(dist) < tolerance)
    first_inside = np.argmax(inside, axis=0)
    nearest = np.argmax(dist, axis=0)
    index = np.where(inside.any(axis=0), first_inside, nearest)
    return np.asarray([phases[i] for i in index])


def _as_labels(values) -> np.ndarray:
    """Labels as a flat integer array when every label is integer-like, else as strings."""
    values = np.asarray(values).reshape(-1)
    if hasattr(values, "detach"):  # pragma: no cover - torch tensors are converted by callers
        values = values.detach().cpu().numpy()
    try:
        return values.astype(int)
    except (TypeError, ValueError):
        return values.astype(str)


def stratified_test_set(
    ground_truth: Callable[[np.ndarray], np.ndarray],
    bounds: Bounds,
    n_per_phase: int = 100,
    n_pool: int = 20000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Phase-stratified test points: ``n_per_phase`` points from inside each phase.

    A Latin hypercube pool of ``n_pool`` points over ``bounds`` is labelled with
    ``ground_truth`` and the same number of points is drawn without replacement from every
    phase, so phases that occupy a small fraction of the design space carry the same weight
    as large ones. A phase with fewer than ``n_per_phase`` pool points contributes all of
    them. Returns ``(X, y_true)``.
    """
    rng = np.random.default_rng(seed)
    pool = latin_hypercube(bounds, n_pool, seed)
    labels = _as_labels(ground_truth(pool))
    keep = []
    for phase in np.unique(labels):
        idx = np.flatnonzero(labels == phase)
        keep.append(rng.choice(idx, size=min(n_per_phase, idx.size), replace=False))
    keep = np.concatenate(keep)
    return pool[keep], labels[keep]


def _default_predict(model, X: np.ndarray):
    """Most probable class of ``model`` at ``X``: the first output of ``model.predict`` (or all of it)."""
    out = model.predict(X)
    return out[0] if isinstance(out, (tuple, list)) else out


class PhaseMapF1Score(PipelineOp):
    """F1 score of a phase classifier against a known ground truth on a stratified test set.

    The test set comes from :func:`stratified_test_set` (equal count per phase, drawn once
    and reused on every ``calculate``), the model loaded from ``model_variable`` predicts it
    and the labels are compared with ``sklearn.metrics.f1_score``.

    Parameters
    ----------
    ground_truth : callable | VirtualSASInstrument
        Either ``f(X) -> labels`` for an ``(n, n_features)`` array in the feature order of
        ``bounds`` (a module-level function may be given by import path), or a virtual
        instrument with traced hulls, labelled with :func:`virtual_instrument_labels`. A live
        object has no JSON form and is exported as a ``{"ref": ...}`` entry of the pipeline
        JSON (see :class:`~AFL.double_agent.PipelineOp.PipelineOp`).
    bounds : dict[str, (low, high)], optional
        Design-space box in the model's feature order. Defaults to the instrument's
        ``component_limits`` ordered by ``instrument.components`` when ``ground_truth`` is a
        virtual instrument.
    model_variable : str
        Dataset variable holding the model ref written by the extrapolator
        (``"{output_prefix}_model"``, requires its ``model_uri``).
    predict : callable, optional
        ``predict(model, X) -> labels``; defaults to the first output of ``model.predict(X)``
        (the labels of a :class:`~AFL.double_agent.PyTorchExtrapolator.DirichletGP`).
    average : str | None
        ``sklearn.metrics.f1_score`` averaging: ``"weighted"`` (default), ``"macro"``,
        ``"micro"`` or ``None`` for one score per phase.
    n_per_phase, n_pool, seed : int
        Stratified test set settings, see :func:`stratified_test_set`.
    output_prefix : str
        Outputs are ``{prefix}_f1`` (0-d, or over ``{prefix}_phase`` when ``average=None``),
        ``{prefix}_accuracy``, ``{prefix}_recall`` (over ``{prefix}_phase``),
        ``{prefix}_n_phases_predicted``, and the test set ``{prefix}_test_points``
        (``{prefix}_test`` x ``{prefix}_dim``), ``{prefix}_test_labels`` and
        ``{prefix}_test_predictions``. After ``calculate`` the same numbers are in the
        ``score`` dict of the op.
    """

    def __init__(
        self,
        ground_truth: Union[Callable[[np.ndarray], np.ndarray], str, object],
        bounds: Optional[Bounds] = None,
        model_variable: str = "phase_model",
        predict: Optional[Union[Callable, str]] = None,
        average: Optional[str] = "weighted",
        n_per_phase: int = 100,
        n_pool: int = 20000,
        seed: int = 0,
        output_prefix: str = "benchmark",
        name: str = "PhaseMapF1Score",
    ) -> None:
        super().__init__(
            name=name,
            input_variable=model_variable,
            output_variable=f"{output_prefix}_f1",
            output_prefix=output_prefix,
        )
        self.ground_truth = ground_truth
        self.bounds = None if bounds is None else {str(k): (float(v[0]), float(v[1])) for k, v in bounds.items()}
        self.model_variable = model_variable
        self.predict = predict
        self.average = average
        self.n_per_phase = n_per_phase
        self.n_pool = n_pool
        self.seed = seed
        # filled by calculate(); the test set is drawn once and reused across iterations
        self.test_points: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None
        self.score: Dict[str, Any] = {}
        self._banned_from_attrs.extend(["test_points", "test_labels", "score"])

    def resolve_bounds(self) -> Dict[str, Tuple[float, float]]:
        """``self.bounds`` or the instrument's ``component_limits`` in ``components`` order."""
        if self.bounds is not None:
            return dict(self.bounds)
        truth = self.ground_truth
        limits = getattr(truth, "config", {}).get("component_limits") if hasattr(truth, "config") else None
        components = getattr(truth, "components", None)
        if not limits or components is None:
            raise ValueError(f"{self.name}: pass `bounds` unless ground_truth is a virtual instrument with component_limits")
        return {str(c): (float(limits[c][0]), float(limits[c][1])) for c in components}

    def resolve_ground_truth(self) -> Callable[[np.ndarray], np.ndarray]:
        """``X -> labels`` built from ``self.ground_truth``."""
        truth = self.ground_truth
        if hasattr(truth, "label_for"):
            return lambda X: virtual_instrument_labels(truth, X)
        return resolve_function(truth) if not callable(truth) else truth

    def resolve_predict(self) -> Callable[[Any, np.ndarray], Any]:
        if self.predict is None:
            return _default_predict
        return resolve_function(self.predict) if not callable(self.predict) else self.predict

    def test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """The cached stratified test set ``(X, y_true)``; drawn on first use."""
        if self.test_points is None:
            self.test_points, self.test_labels = stratified_test_set(
                self.resolve_ground_truth(), self.resolve_bounds(),
                n_per_phase=self.n_per_phase, n_pool=self.n_pool, seed=self.seed,
            )
        return self.test_points, self.test_labels

    def load_model(self, dataset: xr.Dataset):
        from AFL.double_agent.torch_io import load_model

        if self.model_variable not in dataset:
            raise KeyError(
                f"{self.name}: model variable {self.model_variable!r} not found in dataset; give the "
                "upstream extrapolator a `model_uri` so it writes the model ref."
            )
        return load_model(str(dataset[self.model_variable].values[()]))

    def calculate(self, dataset: xr.Dataset) -> Self:
        X, y_true = self.test_set()
        model = self.load_model(dataset)
        y_pred = self.resolve_predict()(model, X)
        if hasattr(y_pred, "detach"):
            y_pred = y_pred.detach().cpu().numpy()
        y_pred = _as_labels(y_pred)
        if y_pred.dtype != y_true.dtype:  # one side integer-like, the other not: compare as strings
            y_true, y_pred = y_true.astype(str), y_pred.astype(str)

        phases = np.unique(y_true)
        f1 = f1_score(y_true, y_pred, labels=phases, average=self.average, zero_division=0)
        recall = np.array([np.mean(y_pred[y_true == p] == p) for p in phases])
        self.score = {
            "f1": [float(v) for v in f1] if self.average is None else float(f1),
            "f1_average": "none" if self.average is None else self.average,
            "accuracy": float(np.mean(y_pred == y_true)),
            "recall_per_phase": {p.item() if hasattr(p, "item") else p: float(r) for p, r in zip(phases, recall)},
            "n_phases_true": int(phases.size),
            "n_phases_predicted": int(np.unique(y_pred).size),
            "n_points": int(y_true.size),
        }

        p = self.output_prefix
        phase_dim, test_dim = f"{p}_phase", f"{p}_test"
        components = list(self.resolve_bounds())
        f1_attrs = {"average": self.score["f1_average"], "description": "sklearn f1_score against the ground truth"}
        if self.average is None:
            self.output[f"{p}_f1"] = xr.DataArray(np.asarray(f1, dtype=float), dims=[phase_dim], coords={phase_dim: phases}, attrs=f1_attrs)
        else:
            self.output[f"{p}_f1"] = xr.DataArray(float(f1), attrs=f1_attrs)
        self.output[f"{p}_accuracy"] = xr.DataArray(self.score["accuracy"])
        self.output[f"{p}_recall"] = xr.DataArray(recall, dims=[phase_dim], coords={phase_dim: phases})
        self.output[f"{p}_n_phases_predicted"] = xr.DataArray(self.score["n_phases_predicted"])
        self.output[f"{p}_test_points"] = xr.DataArray(X, dims=[test_dim, f"{p}_dim"], coords={f"{p}_dim": components})
        self.output[f"{p}_test_labels"] = xr.DataArray(y_true, dims=[test_dim])
        self.output[f"{p}_test_predictions"] = xr.DataArray(y_pred, dims=[test_dim])
        return self
