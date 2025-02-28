"""
Visualization tools for materials composition and phase data.

This module provides plotting utilities for visualizing compositional data and phase
diagrams. It supports both 2D and ternary plots using matplotlib, with special
handling for labeled data and phase regions.

Key features:
- Support for 2D, 3D and ternary plots
- Surface and scatter plot styles
- Automatic handling of phase labels
- Flexible customization through matplotlib parameters
- Integration with xarray data structures
"""

import itertools
from typing import Union, List, Optional, Dict, Any

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

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
        Whether to create a ternary plot for 3-component data.
        If False with 3 components, will create a 3D plot using matplotlib's 3D projection.

    **mpl_kw : dict
        Additional keyword arguments passed to matplotlib's plotting functions

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
    elif len(components) == 3 and not ternary:
        projection = '3d'
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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=projection)

    # For 3 components with ternary=False, use 3D projection
    if len(components) == 3 and not ternary:
        x, y, z = coords.T
        
        # Create a triangulation for the 3D surface
        from matplotlib.tri import Triangulation
        tri = Triangulation(x, y)
        
        # Plot the triangulated surface in 3D
        artists = ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=mpl_kw.get('cmap'), 
                                 edgecolor=mpl_kw.get('edgecolor'), alpha=mpl_kw.get('alpha', 0.8))
        
        # Add color based on labels if provided
        if labels is not None:
            artists.set_array(labels)
    else:
        artists = ax.tripcolor(*coords.T, labels, **mpl_kw)

    if set_axes_labels:
        if projection == 'ternary':
            labels_dict = {k: v.values[()] for k, v in zip(['tlabel', 'llabel', 'rlabel'], components)}
            ax.set(**labels_dict)
        elif projection == '3d':
            ax.set_xlabel(components[0].values[()])
            ax.set_ylabel(components[1].values[()])
            ax.set_zlabel(components[2].values[()])
        else:
            labels_dict = {k: v.values[()] for k, v in zip(['xlabel', 'ylabel'], components)}
            ax.set(**labels_dict)
        
        if projection != '3d':  # Only set grid for 2D plots
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
        Whether to create a ternary plot for 3-component data.
        If False with 3 components, will create a 3D plot using matplotlib's 3D projection.

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
    elif len(components) == 3 and not ternary:
        projection = '3d'
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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=projection)

    # For 3 components with ternary=False, use 3D projection
    if len(components) == 3 and not ternary:
        x, y, z = coords.T
        
        if discrete_labels:
            markers = itertools.cycle(listify(mpl_kw.get('marker', ['^', 'v', '<', '>', 'o', 'd', 'p', 'x'])))
            artists = []
            for label in np.unique(labels):
                mask = (labels == label)
                mpl_kw['marker'] = next(markers)
                # Use scatter3D for 3D scatter plots
                artists.append(ax.scatter3D(x[mask], y[mask], z[mask], **mpl_kw))
        else:
            # Use scatter3D for 3D scatter plots
            artists = ax.scatter3D(x, y, z, c=labels, **mpl_kw)
    else:
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
            labels_dict = {k: v.values[()] for k, v in zip(['tlabel', 'llabel', 'rlabel'], components)}
            ax.set(**labels_dict)
        elif projection == '3d':
            ax.set_xlabel(components[0].values[()])
            ax.set_ylabel(components[1].values[()])
            ax.set_zlabel(components[2].values[()])
        else:
            labels_dict = {k: v.values[()] for k, v in zip(['xlabel', 'ylabel'], components)}
            ax.set(**labels_dict)
        
        if projection != '3d':  # Only set grid for 2D plots
            ax.grid('on', color='black')
    
    return artists


def plot_surface_plotly(
        dataset: xr.Dataset,
        component_variable: str,
        component_dim: str = 'component',
        labels: Union[Optional[str], List] = None,
        set_axes_labels: bool = True,
        ternary: bool = False,
        **plotly_kw) -> Dict[str, Any]:
    """Create a surface plot of compositional data using Plotly.
    
    Generates an interactive filled surface plot of compositional data, supporting both
    2D Cartesian, 3D, and ternary plots. The surface is colored according to
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
        Whether to create a ternary plot for 3-component data.
        If False with 3 components, will create a 3D plot.

    **plotly_kw : dict
        Additional keyword arguments passed to plotly's plotting functions

    Returns
    -------
    Dict[str, Any]
        Plotly figure object

    Raises
    ------
    ValueError
        If number of components is not 2 or 3
    ImportError
        If plotly is not installed
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError('Could not import plotly. Please install via pip install plotly') from e

    components = dataset.coords[component_dim]

    if len(components) not in [2, 3]:
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

    # Set default colormap if not provided
    colorscale = plotly_kw.get('colorscale', 'Viridis')
    
    # Create figure with appropriate layout
    if len(components) == 3 and ternary:
        fig = go.Figure()
        
        # Extract coordinates
        a, b, c = coords.T
        
        # Create ternary scatter plot with color based on labels
        fig.add_trace(go.Scatterternary(
            a=a, b=b, c=c,
            mode='markers',
            marker=dict(
                color=labels,
                colorscale=colorscale,
                size=10,
                showscale=True
            ),
            hovertext=[f'Label: {l}' for l in labels],
        ))
        
        # Set layout for ternary plot
        fig.update_layout(
            ternary=dict(
                aaxis=dict(title=components[0].values[()], min=0, linewidth=2),
                baxis=dict(title=components[1].values[()], min=0, linewidth=2),
                caxis=dict(title=components[2].values[()], min=0, linewidth=2),
                bgcolor='white'
            ),
            **plotly_kw.get('layout', {})
        )
    
    elif len(components) == 3 and not ternary:
        fig = go.Figure()
        
        # Extract coordinates
        x, y, z = coords.T
        
        # Create a mesh3d surface
        from scipy.spatial import Delaunay
        tri = Delaunay(np.column_stack([x, y]))
        
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            intensity=labels,
            colorscale=colorscale,
            opacity=0.8,
            **{k: v for k, v in plotly_kw.items() if k not in ['layout', 'colorscale']}
        ))
        
        # Set layout for 3D plot
        if set_axes_labels:
            fig.update_layout(
                scene=dict(
                    xaxis_title=components[0].values[()],
                    yaxis_title=components[1].values[()],
                    zaxis_title=components[2].values[()]
                ),
                **plotly_kw.get('layout', {})
            )
    
    else:  # 2D plot
        fig = go.Figure()
        
        # Extract coordinates
        x, y = coords.T
        
        # Create a contour plot for 2D surface
        from scipy.interpolate import griddata
        
        # Create a grid for interpolation
        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate labels onto the grid
        zi = griddata((x, y), labels, (xi_grid, yi_grid), method='linear')
        
        # Create contour plot
        fig.add_trace(go.Contour(
            z=zi,
            x=xi,
            y=yi,
            colorscale=colorscale,
            **{k: v for k, v in plotly_kw.items() if k not in ['layout', 'colorscale']}
        ))
        
        # Set layout for 2D plot
        if set_axes_labels:
            fig.update_layout(
                xaxis_title=components[0].values[()],
                yaxis_title=components[1].values[()],
                **plotly_kw.get('layout', {})
            )
    
    return fig


def plot_scatter_plotly(
        dataset: xr.Dataset,
        component_variable: str,
        component_dim: str = 'component',
        labels: Union[Optional[str], List] = None,
        discrete_labels: bool = True,
        set_axes_labels: bool = True,
        ternary: bool = False,
        **plotly_kw) -> Dict[str, Any]:
    """Create a scatter plot of compositional data using Plotly.
    
    Generates an interactive scatter plot of compositional data, supporting both 2D
    Cartesian, 3D, and ternary plots. Points can be colored and marked according
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
        Whether to create a ternary plot for 3-component data.
        If False with 3 components, will create a 3D plot.

    **plotly_kw : dict
        Additional keyword arguments passed to plotly's plotting functions

    Returns
    -------
    Dict[str, Any]
        Plotly figure object

    Raises
    ------
    ValueError
        If number of components is not 2 or 3
    ImportError
        If plotly is not installed

    Notes
    -----
    When discrete_labels is True, different marker symbols are used for each
    unique label value, cycling through a predefined set of markers.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError('Could not import plotly. Please install via pip install plotly') from e

    components = dataset.coords[component_dim]

    if len(components) not in [2, 3]:
        raise ValueError(f'plot_scatter only compatible with 2 or 3 components. You passed: {components}')

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

    # Set default colormap if not provided
    colorscale = plotly_kw.get('colorscale', 'Viridis')
    
    # Create figure
    fig = go.Figure()
    
    # Define marker symbols for discrete labels
    plotly_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 
                      'triangle-down', 'triangle-left', 'triangle-right', 'pentagon', 
                      'hexagon', 'star', 'hexagram', 'star-triangle-up', 'star-triangle-down']
    
    if len(components) == 3 and ternary:
        # Extract coordinates
        a, b, c = coords.T
        
        if discrete_labels:
            # Create a trace for each unique label
            for i, label in enumerate(np.unique(labels)):
                mask = (labels == label)
                symbol = plotly_symbols[i % len(plotly_symbols)]
                
                fig.add_trace(go.Scatterternary(
                    a=a[mask], b=b[mask], c=c[mask],
                    mode='markers',
                    marker=dict(
                        symbol=symbol,
                        size=10,
                        color=i,  # Use index for color
                    ),
                    name=f'Label {label}',
                    **{k: v for k, v in plotly_kw.items() if k not in ['layout', 'colorscale']}
                ))
        else:
            # Create a single trace with continuous color scale
            fig.add_trace(go.Scatterternary(
                a=a, b=b, c=c,
                mode='markers',
                marker=dict(
                    color=labels,
                    colorscale=colorscale,
                    size=10,
                    showscale=True
                ),
                hovertext=[f'Label: {l}' for l in labels],
                **{k: v for k, v in plotly_kw.items() if k not in ['layout', 'colorscale']}
            ))
        
        # Set layout for ternary plot
        fig.update_layout(
            ternary=dict(
                aaxis=dict(title=components[0].values[()], min=0, linewidth=2),
                baxis=dict(title=components[1].values[()], min=0, linewidth=2),
                caxis=dict(title=components[2].values[()], min=0, linewidth=2),
                bgcolor='white'
            ),
            **plotly_kw.get('layout', {})
        )
    
    elif len(components) == 3 and not ternary:
        # Extract coordinates
        x, y, z = coords.T
        
        if discrete_labels:
            # Create a trace for each unique label
            for i, label in enumerate(np.unique(labels)):
                mask = (labels == label)
                symbol = plotly_symbols[i % len(plotly_symbols)]
                
                fig.add_trace(go.Scatter3d(
                    x=x[mask], y=y[mask], z=z[mask],
                    mode='markers',
                    marker=dict(
                        symbol=symbol,
                        size=6,
                        color=i,  # Use index for color
                    ),
                    name=f'Label {label}',
                    **{k: v for k, v in plotly_kw.items() if k not in ['layout', 'colorscale']}
                ))
        else:
            # Create a single trace with continuous color scale
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    color=labels,
                    colorscale=colorscale,
                    size=6,
                    showscale=True
                ),
                hovertext=[f'Label: {l}' for l in labels],
                **{k: v for k, v in plotly_kw.items() if k not in ['layout', 'colorscale']}
            ))
        
        # Set layout for 3D plot
        if set_axes_labels:
            fig.update_layout(
                scene=dict(
                    xaxis_title=components[0].values[()],
                    yaxis_title=components[1].values[()],
                    zaxis_title=components[2].values[()]
                ),
                **plotly_kw.get('layout', {})
            )
    
    else:  # 2D plot
        # Extract coordinates
        x, y = coords.T
        
        if discrete_labels:
            # Create a trace for each unique label
            for i, label in enumerate(np.unique(labels)):
                mask = (labels == label)
                symbol = plotly_symbols[i % len(plotly_symbols)]
                
                fig.add_trace(go.Scatter(
                    x=x[mask], y=y[mask],
                    mode='markers',
                    marker=dict(
                        symbol=symbol,
                        size=10,
                        color=i,  # Use index for color
                    ),
                    name=f'Label {label}',
                    **{k: v for k, v in plotly_kw.items() if k not in ['layout', 'colorscale']}
                ))
        else:
            # Create a single trace with continuous color scale
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(
                    color=labels,
                    colorscale=colorscale,
                    size=10,
                    showscale=True
                ),
                hovertext=[f'Label: {l}' for l in labels],
                **{k: v for k, v in plotly_kw.items() if k not in ['layout', 'colorscale']}
            ))
        
        # Set layout for 2D plot
        if set_axes_labels:
            fig.update_layout(
                xaxis_title=components[0].values[()],
                yaxis_title=components[1].values[()],
                **plotly_kw.get('layout', {})
            )
    
    return fig
