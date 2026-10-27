"""Plotting helpers for the bioformulations case-study notebooks.

The functions draw the figures of the baseline and the manifold active learning campaigns on the
``ProteinGlycerolTemperaturePhaseMap`` virtual instrument:

* ``plot_phase_map`` -- a 3D scatter of the design-space grid coloured by phase label, used for
  the ground truth of the simulator
* ``plot_baseline_iteration`` / ``plot_manifold_iteration`` -- the six-panel figure of one
  iteration: sampled points, most probable phase and predictive entropy of the fitted model,
  the two acquisition stages of the current proposal and the temperatures measured so far
* ``plot_score_history`` -- the F1 score and accuracy of the model against the simulator over
  the campaign

Colours follow the AFL-bioformulations convention: fixed phase colours, ``magma`` for magnitudes
(entropy, acquisition) and ``viridis`` for ordered sequences (optimiser steps, batches).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.cm import ScalarMappable

PHASE_COLORS = ["tab:red", "tab:olive", "tab:cyan"]  # S-L, Single, L-L
PHASE_NAMES = ["S-L", "Single", "L-L"]
MAGNITUDE_CMAP = "magma"
SEQUENCE_CMAP = "viridis"
SCATTER_KWARGS = {"s": 30, "edgecolors": "k", "linewidths": 0.5, "alpha": 0.85, "depthshade": False}
GRID_KWARGS = {"s": 12, "alpha": 0.7, "depthshade": False, "linewidths": 0}

Limits = Mapping[str, Tuple[float, float]]
TemperatureHistory = Dict[int, np.ndarray]  # batch_id -> measured temperatures


# --------------------------------------------------------------------------- building blocks
def _xy(components: Sequence[str]) -> list:
    return [c for c in components if c != "temperature"]


def _xyz(data: xr.DataArray, components: Sequence[str], component_dim: str = "ds_dim"):
    """(protein, glycerol, temperature) columns of a (..., component_dim) array."""
    xy = _xy(components)
    sel = lambda c: data.sel({component_dim: c}).values.reshape(-1)  # noqa: E731
    return sel(xy[0]), sel(xy[1]), sel("temperature")


def _style_3d(ax, limits: Limits, components: Sequence[str], title: str) -> None:
    xy = _xy(components)
    ax.set_xlim(*limits[xy[0]])
    ax.set_ylim(*limits[xy[1]])
    ax.set_zlim(*limits["temperature"])
    ax.set_xlabel(xy[0])
    ax.set_ylabel(xy[1])
    ax.set_zlabel("temperature", labelpad=1)
    ax.set_title(title)


def _phase_colorbar(fig, ax, n_phases: int = 3):
    """Discrete colour bar for the phase labels next to ``ax``."""
    cmap = mcolors.ListedColormap(PHASE_COLORS[:n_phases])
    norm = mcolors.BoundaryNorm(np.arange(n_phases + 1), cmap.N)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.1,
                        ticks=np.arange(n_phases) + 0.5)
    cbar.ax.set_yticklabels(PHASE_NAMES[:n_phases])
    return cbar


def draw_phase_grid(fig, ax, grid: xr.DataArray, labels, limits: Limits, components: Sequence[str],
                    title: str) -> None:
    """3D scatter of the design-space ``grid`` (dims ``(..., ds_dim)``) coloured by phase ``labels``."""
    x, y, z = _xyz(grid, components)
    labels = np.asarray(labels).astype(int).reshape(-1)
    for label in np.unique(labels):
        m = labels == label
        ax.scatter(x[m], y[m], z[m], color=PHASE_COLORS[label], **GRID_KWARGS)
    _phase_colorbar(fig, ax)
    _style_3d(ax, limits, components, title)


def draw_samples(ax, ds_running: xr.Dataset, batch_id: int, limits: Limits, components: Sequence[str]) -> None:
    """Measured points coloured by phase; the batch ``batch_id`` is ringed."""
    x, y, z = _xyz(ds_running["design_space"], components)
    labels = ds_running["phases"].values.astype(int).reshape(-1)
    for label in np.unique(labels):
        m = labels == label
        ax.scatter(x[m], y[m], z[m], color=PHASE_COLORS[label], label=PHASE_NAMES[label], **SCATTER_KWARGS)
    new = ds_running["batch_sample_id"].values == batch_id
    ax.scatter(x[new], y[new], z[new], s=90, facecolors="none", edgecolors="k", linewidths=1.2,
               depthshade=False, label="latest batch")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _style_3d(ax, limits, components, "Sampled points")


def draw_model_entropy(fig, ax, ds_result: xr.Dataset, limits: Limits, components: Sequence[str]) -> None:
    """Predictive entropy of the classifier on the design-space grid."""
    x, y, z = _xyz(ds_result["design_space_grid"], components)
    img = ax.scatter(x, y, z, c=ds_result["phase_entropy"].values.reshape(-1), cmap=MAGNITUDE_CMAP, **GRID_KWARGS)
    fig.colorbar(img, ax=ax, shrink=0.6, pad=0.1, label="entropy S(x)")
    _style_3d(ax, limits, components, "Model: predictive entropy")


def draw_composition_surface(fig, ax, xy_grid: np.ndarray, values: np.ndarray, limits: Limits,
                             components: Sequence[str], title: str, label: str,
                             measured: Optional[np.ndarray] = None, nxt: Optional[np.ndarray] = None) -> None:
    """2D acquisition landscape over (protein, glycerol) with the measured and the next composition."""
    xy = _xy(components)
    img = ax.tricontourf(xy_grid[:, 0], xy_grid[:, 1], values, levels=20, cmap=MAGNITUDE_CMAP)
    if measured is not None and len(measured):
        ax.scatter(measured[:, 0], measured[:, 1], s=18, color="k", label="measured")
    if nxt is not None:
        ax.scatter(nxt[:, 0], nxt[:, 1], marker="*", s=180, c="w", edgecolors="k", label="next")
    ax.set_xlim(*limits[xy[0]])
    ax.set_ylim(*limits[xy[1]])
    ax.set_xlabel(xy[0])
    ax.set_ylabel(xy[1])
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.colorbar(img, ax=ax, label=label)


def draw_temperature_history(fig, ax, temperature_history: TemperatureHistory, limits: Limits) -> None:
    """Measured temperatures of every batch, ordered within the batch and coloured by batch."""
    batches = sorted(temperature_history)
    cmap = plt.get_cmap(SEQUENCE_CMAP)
    norm = mcolors.Normalize(vmin=min(batches), vmax=max(max(batches), min(batches) + 1))
    for b in batches:
        t = np.sort(np.asarray(temperature_history[b], dtype=float))
        u = np.linspace(0.0, 1.0, len(t)) if len(t) > 1 else np.array([0.5])
        ax.plot(u, t, marker="o", ms=4, lw=1.5, color=cmap(norm(b)))
    lo, hi = limits["temperature"]
    ax.plot([0, 1], [lo, hi], color="0.6", ls="--", lw=1, label="uniform ramp")
    ax.set_xlim(0, 1)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("position in batch")
    ax.set_ylabel("temperature")
    ax.set_title("Measured temperatures per batch")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="batch")


def _iteration_figure(ds_result, ds_running, batch_id, limits, components, title):
    """Figure with the three shared top panels; returns ``(fig, bottom_axes)``."""
    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(wspace=0.45, hspace=0.35)
    ax3d = [fig.add_subplot(2, 3, col + 1, projection="3d") for col in range(3)]
    ax2d = [fig.add_subplot(2, 3, col + 4) for col in range(3)]
    draw_samples(ax3d[0], ds_running, batch_id, limits, components)
    draw_phase_grid(fig, ax3d[1], ds_result["design_space_grid"], ds_result["phase_mean"].values,
                    limits, components, "Model: most probable phase")
    draw_model_entropy(fig, ax3d[2], ds_result, limits, components)
    fig.suptitle(f"{title}, batch {batch_id}, {ds_running.sizes['sample']} measurements")
    return fig, ax2d


# --------------------------------------------------------------------------- figures
def plot_phase_map(grid: np.ndarray, labels, limits: Limits, components: Sequence[str],
                   title: str = "Simulator: ground-truth phase map"):
    """3D scatter of ``grid`` (rows ordered like ``components``) coloured by phase ``labels``."""
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    grid = xr.DataArray(np.asarray(grid), dims=["ds_grid", "ds_dim"], coords={"ds_dim": list(components)})
    draw_phase_grid(fig, ax, grid, labels, limits, components, title)
    return fig


def plot_baseline_iteration(ds_result: xr.Dataset, ds_running: xr.Dataset, batch_id: int,
                            next_temperatures: Sequence[float], temperature_history: TemperatureHistory,
                            limits: Limits, components: Sequence[str]):
    """Six panels of one baseline (grid acquisition) iteration.

    ``ds_result`` is the pipeline output holding the proposal (``composition_next``,
    ``temperature_next``), ``ds_running`` the measured dataset whose batch ``batch_id`` is ringed and
    ``next_temperatures`` the temperatures selected at the proposed composition.
    """
    xy = _xy(components)
    prefix = "_".join(xy)  # MarginalEntropyOverDimension names its outputs "<complement dims>_*"
    comp_dim = f"{prefix}_d"
    fig, ax2d = _iteration_figure(ds_result, ds_running, batch_id, limits, components, "Baseline AL")

    # composition acquisition: entropy over temperature on the (protein, glycerol) grid, measured points excluded
    grid = ds_result[f"{prefix}_domain"].transpose(..., comp_dim).sel({comp_dim: xy}).values
    surface = ds_result["composition_decision_surface"].values.reshape(-1)
    measured = ds_result[f"{prefix}_sampled"].sel({comp_dim: xy}).values
    nxt = ds_result["composition_next"].sel({comp_dim: xy}).values.reshape(-1, 2)
    draw_composition_surface(fig, ax2d[0], grid, surface, limits, components,
                             "Composition acquisition (entropy over T, excluded)", "decision surface",
                             measured=measured, nxt=nxt)

    # temperature acquisition at the chosen composition
    ax = ax2d[1]
    t = ds_result["temperature_domain"].values.reshape(-1)
    order = np.argsort(t)
    ax.plot(t[order], ds_result["temperature_utility"].values.reshape(-1)[order], color="0.3", lw=2,
            label="entropy along T")
    ax.plot(t[order], ds_result["temperature_decision_surface"].values.reshape(-1)[order],
            color=plt.get_cmap(MAGNITUDE_CMAP)(0.6), lw=2, ls="-.", label="decision surface")
    for i, ti in enumerate(np.sort(np.asarray(next_temperatures, dtype=float))):
        ax.axvline(ti, color="k", ls="--", lw=1, label="selected batch" if i == 0 else None)
    ax.set_xlim(*limits["temperature"])
    ax.set_xlabel("temperature")
    ax.set_ylabel("utility")
    ax.set_title(f"Temperature acquisition at ({nxt[0, 0]:.1f}, {nxt[0, 1]:.1f})")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    draw_temperature_history(fig, ax2d[2], temperature_history, limits)
    return fig


def plot_manifold_iteration(ds_result: xr.Dataset, ds_running: xr.Dataset, batch_id: int,
                            next_temperatures: Sequence[float], temperature_history: TemperatureHistory,
                            limits: Limits, components: Sequence[str], n_trajectory_curves: int = 12):
    """Six panels of one manifold (BoTorch composition + SRSF profile) iteration.

    ``ds_result`` is the pipeline output holding the proposal (``composition_next``,
    ``profile_function``, ``profile_trajectory``, ``profile_loss``), ``ds_running`` the measured
    dataset whose batch ``batch_id`` is ringed and ``next_temperatures`` the temperatures the
    proposed profile is discretised into.
    """
    xy = _xy(components)
    lo, hi = limits["temperature"]
    to_temperature = lambda f: lo + (hi - lo) * np.asarray(f)  # noqa: E731
    fig, ax2d = _iteration_figure(ds_result, ds_running, batch_id, limits, components, "Manifold AL")

    # composition acquisition landscape (expected profile entropy) and its optimum
    grid = ds_result["composition_grid"].transpose(..., "component").sel(component=xy).values
    surface = ds_result["composition_decision_surface"].values.reshape(-1)
    measured = np.unique(ds_running["composition"].sel(component=xy).values, axis=0)
    nxt = ds_result["composition_next"].sel(component=xy).values.reshape(-1, 2)
    draw_composition_surface(fig, ax2d[0], grid, surface, limits, components,
                             "Composition acquisition (expected profile entropy)", "score",
                             measured=measured, nxt=nxt)

    # function-space optimisation of the temperature profile
    ax = ax2d[1]
    t = ds_result["profile_function"]["profile_t"].values
    f_star = to_temperature(ds_result["profile_function"].values[:, 0])
    traj = ds_result["profile_trajectory"].values[:, :, 0]  # (steps, t)
    steps = np.unique(np.linspace(0, traj.shape[0] - 1, n_trajectory_curves).astype(int))
    cmap = plt.get_cmap(SEQUENCE_CMAP)
    norm = mcolors.Normalize(vmin=0, vmax=traj.shape[0] - 1)
    for s in steps:
        ax.plot(t, to_temperature(traj[s]), color=cmap(norm(s)), lw=1, alpha=0.8)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="optimiser step")
    ax.plot(t, f_star, color="k", ls="--", lw=2, label="optimal profile f*")
    measured_t = np.sort(np.asarray(next_temperatures, dtype=float))
    if np.all(np.diff(f_star) >= 0):
        u_measured = np.interp(measured_t, f_star, t)
    else:
        u_measured = np.linspace(0, 1, len(measured_t))
    ax.scatter(u_measured, measured_t, s=60, c="w", edgecolors="k", zorder=5, label="measured")
    best = int(ds_result["profile_restart_star"].values)
    final_score = -ds_result["profile_loss"].values[best, -1]
    ax.set_xlim(0, 1)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("profile coordinate t")
    ax.set_ylabel("temperature")
    ax.set_title(f"Profile optimisation at ({nxt[0, 0]:.1f}, {nxt[0, 1]:.1f}), score {final_score:.3f}")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    draw_temperature_history(fig, ax2d[2], temperature_history, limits)
    return fig


def plot_score_history(history, ax=None):
    """F1 score and accuracy of the model against the simulator over the acquired compositions.

    ``history`` is a ``pandas.DataFrame`` (or list of records) with ``iteration``, ``f1`` and ``accuracy``.
    """
    import pandas as pd

    history = pd.DataFrame(history)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(history["iteration"], history["f1"], marker="o", lw=2, label="F1 score")
    ax.plot(history["iteration"], history["accuracy"], marker="s", lw=2, ls="--", label="accuracy")
    ax.set_xlabel("acquired compositions")
    ax.set_ylabel("model vs simulator")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(frameon=False, loc="lower right")
    return ax
