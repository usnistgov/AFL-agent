import itertools
from typing import Union, List, Optional

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from AFL.double_agent.util import listify


def plot_surface_mpl(
        dataset: xr.Dataset,
        component_variable: str,
        component_dim: str = 'component',
        labels: Union[Optional[str], List] = None,
        set_axes_labels: bool = True,
        ternary: bool = True,
        **mpl_kw) -> List[plt.Artist]:
    components = dataset.coords[component_dim]

    if len(components) == 3 and ternary:
        try:
            import mpltern
        except ImportError as e:
            raise ImportError('Could not import mpltern. Please install via conda or pip') from e
        projection = 'ternary'
    elif len(components) == 2:
        projection = None
    else:
        raise ValueError(f'plot_surface only compatible with 2 or 3 components. You passed: {components}')

    coords = dataset[component_variable].transpose(..., component_dim)

    if labels is None:
        if 'labels' in dataset.coords:
            labels = dataset.coords['labels'].values
        elif 'labels' in dataset:
            labels = dataset['labels'].values
        else:
            labels = np.zeros(coords.shape[0])
    elif isinstance(labels, str) and (labels in dataset):
        labels = dataset[labels].values

    if ('cmap' not in mpl_kw) and ('color' not in mpl_kw):
        mpl_kw['cmap'] = 'viridis'
    if 'edgecolor' not in mpl_kw:
        mpl_kw['edgecolor'] = 'face'

    if 'ax' in mpl_kw:
        ax = mpl_kw.pop('ax')
    else:
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=projection))

    artists = ax.tripcolor(*coords.T, labels, **mpl_kw)

    if set_axes_labels:
        if projection == 'ternary':
            labels = {k: v.values[()] for k, v in zip(['tlabel', 'llabel', 'rlabel'], components)}
        else:
            labels = {k: v.values[()] for k, v in zip(['xlabel', 'ylabel'], components)}
        ax.set(**labels)
        ax.grid('on', color='black')
    return artists


def plot_scatter_mpl(
        dataset: xr.Dataset,
        component_variable: str,
        component_dim: str = 'component',
        labels: Union[Optional[str], List] = None,
        discrete_labels: bool = True,
        set_axes_labels: bool = True,
        ternary: bool = True,
        **mpl_kw) -> List[plt.Artist]:
    components = dataset.coords[component_dim]

    if len(components) == 3 and ternary:
        try:
            import mpltern
        except ImportError as e:
            raise ImportError('Could not import mpltern. Please install via conda or pip') from e
        projection = 'ternary'
    elif len(components) == 2:
        projection = None
    else:
        raise ValueError(f'plot_surface only compatible with 2 or 3 components. You passed: {components}')

    coords = dataset[component_variable].transpose(..., component_dim).values

    if 'ax' in mpl_kw:
        ax = mpl_kw.pop('ax')
    else:
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=projection))

    if discrete_labels:
        markers = itertools.cycle(listify(mpl_kw.get('marker', ['^', 'v', '<', '>', 'o', 'd', 'p', 'x'])))
        artists = []
        for label in np.unique(labels):
            mask = (labels == label)
            mpl_kw['marker'] = next(markers)
            artists.append(ax.scatter(*coords[mask].T, **mpl_kw))
    else:
        artists = ax.scatter(*coords.T, c=labels, **mpl_kw)

    if set_axes_labels:
        if projection == 'ternary':
            labels = {k: v.values[()] for k, v in zip(['tlabel', 'llabel', 'rlabel'], components)}
        else:
            labels = {k: v.values[()] for k, v in zip(['xlabel', 'ylabel'], components)}
        ax.set(**labels)
        ax.grid('on', color='black')
    return artists
