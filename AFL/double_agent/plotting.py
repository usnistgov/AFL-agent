"""
Visualization tools for materials composition and phase data.

This module provides plotting utilities for visualizing compositional data and phase
diagrams. It supports both 2D and ternary plots using matplotlib, with special
handling for labeled data and phase regions.

Key features:
- Support for both 2D and ternary plots
- Surface and scatter plot styles
- Automatic handling of phase labels
- Flexible customization through matplotlib parameters
- Integration with xarray data structures
"""

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
        ternary: bool = False,
        **mpl_kw) -> List[plt.Artist]:
    """Create a surface plot of compositional data using matplotlib.
    
    Generates a filled surface plot of compositional data, supporting both
    2D Cartesian and ternary plots. The surface is colored according to
    phase labels or other specified values.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing compositional data and optional labels

    component_variable : str
        Name of the variable containing component coordinates

    component_dim : str, default='component'
        Name of the dimension containing component names

    labels : Union[Optional[str], List], default=None
        Labels for coloring the surface. Can be:
        - None: Will try to use 'labels' from dataset
        - str: Name of variable in dataset containing labels
        - List: Direct list of label values

    set_axes_labels : bool, default=True
        Whether to set axis labels using component names

    ternary : bool, default=False
        Whether to create a ternary plot for 3-component data

    **mpl_kw : dict
        Additional keyword arguments passed to matplotlib's tripcolor

    Returns
    -------
    List[plt.Artist]
        List of created matplotlib artists

    Raises
    ------
    ValueError
        If number of components is not 2 or 3
    ImportError
        If mpltern is not installed for ternary plots
    """
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
        ternary: bool = False,
        **mpl_kw) -> List[plt.Artist]:
    """Create a scatter plot of compositional data using matplotlib.
    
    Generates a scatter plot of compositional data, supporting both 2D
    Cartesian and ternary plots. Points can be colored and marked according
    to phase labels or other values.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing compositional data and optional labels

    component_variable : str
        Name of the variable containing component coordinates

    component_dim : str, default='component'
        Name of the dimension containing component names

    labels : Union[Optional[str], List], default=None
        Labels for coloring points. Can be:
        - None: Will try to use 'labels' from dataset
        - str: Name of variable in dataset containing labels
        - List: Direct list of label values

    discrete_labels : bool, default=True
        Whether to treat labels as discrete categories (True) or continuous values (False)

    set_axes_labels : bool, default=True
        Whether to set axis labels using component names

    ternary : bool, default=False
        Whether to create a ternary plot for 3-component data

    **mpl_kw : dict
        Additional keyword arguments passed to matplotlib's scatter

    Returns
    -------
    List[plt.Artist]
        List of created matplotlib artists

    Raises
    ------
    ValueError
        If number of components is not 2 or 3
    ImportError
        If mpltern is not installed for ternary plots

    Notes
    -----
    When discrete_labels is True, different marker styles are used for each
    unique label value, cycling through a predefined set of markers.
    """
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

    if (labels is None):
        if ('labels' in dataset.coords):
            labels = dataset.coords['labels'].values
        elif ('labels' in dataset):
            labels = dataset['labels'].values
        else:
            labels = np.zeros(coords.shape[0])
    elif isinstance(labels, str) and (labels in dataset):
        labels = dataset[labels].values

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
