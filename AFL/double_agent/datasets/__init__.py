"""
Datasets module for AFL.double_agent.

This module provides access to example datasets that can be used with AFL.double_agent.
"""

import os
import pathlib
import importlib.resources
import warnings

import xarray as xr

# Get the path to the data directory
def get_data_dir():
    """
    Get the path to the data directory.
    
    Returns
    -------
    pathlib.Path
        Path to the data directory.
    """
    # First try to find the data directory in the package
    try:
        # For Python 3.9+
        with importlib.resources.files("AFL") as p:
            package_data_dir = p / "data"
            if package_data_dir.exists():
                return package_data_dir
    except (ImportError, AttributeError):
        pass
    
    # Try to find the data directory relative to this file
    module_dir = pathlib.Path(__file__).parent
    
    # Check several possible locations
    possible_locations = [
        # Inside the package
        module_dir.parent.parent / "data",
        # At the project root (development mode)
        module_dir.parent.parent.parent / "data",
        # One level up from project root (some development environments)
        module_dir.parent.parent.parent.parent / "data",
    ]
    
    for location in possible_locations:
        if location.exists():
            return location
    
    # If we can't find the data directory, warn the user and return the most likely location
    warnings.warn(
        "Could not find data directory. Looked in: " + 
        ", ".join(str(loc) for loc in possible_locations)
    )
    return possible_locations[0]  # Return the first location as a fallback

def list_datasets():
    """
    List all available datasets.
    
    Returns
    -------
    list
        List of available dataset names.
    """
    data_dir = get_data_dir()
    if not data_dir.exists():
        warnings.warn(f"Data directory not found at {data_dir}")
        return []
    
    return [f.stem for f in data_dir.glob("*.nc")]

def load_dataset(name):
    """
    Load a dataset by name.
    
    Parameters
    ----------
    name : str
        Name of the dataset to load.
        
    Returns
    -------
    xarray.Dataset
        The loaded dataset.
        
    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.
    """
    data_dir = get_data_dir()
    file_path = data_dir / f"{name}.nc"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {file_path}. "
            f"Data directory: {data_dir}. "
            f"Available datasets: {list_datasets()}"
        )
    
    return xr.open_dataset(file_path)

# Define specific dataset loaders
def example_dataset1():
    """
    Load the example dataset.
    
    Returns
    -------
    xarray.Dataset
        The example dataset.
    """
    return load_dataset("example_dataset")

# Add all datasets as module-level variables
__all__ = ["load_dataset", "list_datasets", "example_dataset1"] 