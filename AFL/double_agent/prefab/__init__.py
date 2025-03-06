"""
Prefabricated Pipelines module for AFL.double_agent.

This module provides access to prefabricated pipelines that can be used with AFL.double_agent.
It allows loading, listing, and combining multiple pipelines into a single pipeline.
"""

import os
import pathlib
import warnings
import json
from typing import List, Dict, Optional, Union

import xarray as xr

from AFL.double_agent.Pipeline import Pipeline

# Get the path to the prefab directory
def get_prefab_dir():
    """
    Get the path to the prefabricated pipelines directory.
    
    Returns
    -------
    pathlib.Path
        Path to the prefab directory.
    """
    # The prefab directory is located in AFL/double_agent/prefab
    module_dir = pathlib.Path(__file__).parent
    prefab_dir = module_dir
    
    if not prefab_dir.exists():
        warnings.warn(f"Prefab directory not found at {prefab_dir}")
    
    return prefab_dir

def list_prefabs(display_table: bool = True):
    """
    List all available prefabricated pipelines.
    
    Parameters
    ----------
    display_table : bool, default=True
        Whether to display the results in a formatted table with descriptions.
        If False, returns just the list of prefab names.
    
    Returns
    -------
    list or None
        If display_table is False, returns a list of available prefabricated pipeline names.
        If display_table is True, prints a formatted table and returns None.
    """
    prefab_dir = get_prefab_dir()
    if not prefab_dir.exists():
        warnings.warn(f"Prefab directory not found at {prefab_dir}")
        return []
    
    prefab_files = list(prefab_dir.glob("*.json"))
    
    if not display_table:
        return [f.stem for f in prefab_files]
    
    if not prefab_files:
        print("No prefabricated pipelines found.")
        return None
    
    # Get descriptions from each prefab file
    prefabs_info = []
    max_name_len = 4  # Minimum width for "Name" column
    max_desc_len = 11  # Minimum width for "Description" column
    
    for file_path in prefab_files:
        try:
            with open(file_path, 'r') as f:
                prefab_data = json.load(f)
            
            name = file_path.stem
            description = prefab_data.get('description', 'No description available')
            
            # Track maximum lengths for formatting
            max_name_len = max(max_name_len, len(name))
            max_desc_len = max(max_desc_len, len(description))
            
            prefabs_info.append((name, description))
        except Exception as e:
            warnings.warn(f"Error reading prefab {file_path.name}: {str(e)}")
    
    # Print formatted table
    header = f"| {'Name':<{max_name_len}} | {'Description':<{max_desc_len}} |"
    separator = f"|-{'-'*max_name_len}-|-{'-'*max_desc_len}-|"
    
    print("\nAvailable Prefabricated Pipelines:")
    print(separator)
    print(header)
    print(separator)
    
    for name, description in sorted(prefabs_info):
        print(f"| {name:<{max_name_len}} | {description:<{max_desc_len}} |")
    
    print(separator)
    print(f"Total: {len(prefabs_info)} prefabricated pipeline(s)")
    
    return None

def load_prefab(name: str) -> Pipeline:
    """
    Load a prefabricated pipeline by name.
    
    Parameters
    ----------
    name : str
        Name of the prefabricated pipeline to load.
        
    Returns
    -------
    Pipeline
        The loaded pipeline.
        
    Raises
    ------
    FileNotFoundError
        If the prefabricated pipeline does not exist.
    """
    prefab_dir = get_prefab_dir()
    file_path = prefab_dir / f"{name}.json"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Prefabricated pipeline '{name}' not found at {file_path}. "
            f"Prefab directory: {prefab_dir}. "
            f"Available prefabs: {list_prefabs()}"
        )
    
    return Pipeline.read_json(str(file_path))

def combine_prefabs(prefab_names: List[str], new_name: Optional[str] = None) -> Pipeline:
    """
    Combine multiple prefabricated pipelines into a single pipeline.
    
    Parameters
    ----------
    prefab_names : List[str]
        List of prefabricated pipeline names to combine.
    new_name : Optional[str], default=None
        Name for the combined pipeline. If None, a name will be generated from the component pipelines.
        
    Returns
    -------
    Pipeline
        The combined pipeline.
        
    Raises
    ------
    FileNotFoundError
        If any of the prefabricated pipelines do not exist.
    ValueError
        If no prefabricated pipelines are provided.
    """
    if not prefab_names:
        raise ValueError("No prefabricated pipelines provided for combination.")
    
    if len(prefab_names) == 1:
        return load_prefab(prefab_names[0])
    
    # Generate a combined name if not provided
    if new_name is None:
        new_name = f"Combined_{'_'.join(prefab_names)}"
    
    # Load all pipelines
    pipelines = [load_prefab(name) for name in prefab_names]
    
    # Create a new pipeline with the first pipeline's ops
    combined_pipeline = pipelines[0].copy()
    combined_pipeline.name = new_name
    
    # Add all operations from subsequent pipelines
    for pipeline in pipelines[1:]:
        combined_pipeline.extend(pipeline.ops)
    
    return combined_pipeline

def save_prefab(pipeline: Pipeline, name: Optional[str] = None, overwrite: bool = False, description: Optional[str] = None) -> str:
    """
    Save a pipeline as a prefabricated pipeline.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to save.
    name : Optional[str], default=None
        Name to save the pipeline as. If None, uses the pipeline's name.
    overwrite : bool, default=False
        Whether to overwrite an existing prefabricated pipeline with the same name.
    description : Optional[str], default=None
        A descriptive text about the pipeline's purpose and functionality.
        If None and pipeline has a description attribute, that will be used.
        
    Returns
    -------
    str
        The path to the saved prefabricated pipeline file.
        
    Raises
    ------
    FileExistsError
        If a prefabricated pipeline with the given name already exists and overwrite=False.
    """
    prefab_dir = get_prefab_dir()
    
    # Use the pipeline's name if no name is provided
    if name is None:
        name = pipeline.name
    
    # Use the pipeline's description if available and none provided
    if description is None and hasattr(pipeline, 'description'):
        description = pipeline.description
    
    # Ensure valid filename
    name = name.replace(" ", "_")
    file_path = prefab_dir / f"{name}.json"
    
    # Check if file exists and overwrite is not allowed
    if file_path.exists() and not overwrite:
        raise FileExistsError(
            f"Prefabricated pipeline '{name}' already exists at {file_path}. "
            f"Use overwrite=True to overwrite."
        )
    
    # Save the pipeline with description
    pipeline.write_json(str(file_path), overwrite=True, description=description)
    
    return str(file_path)

# Define any example prefabs that should be available by default
def example_prefab1():
    """
    Load an example prefabricated pipeline.
    
    Returns
    -------
    Pipeline
        An example prefabricated pipeline.
    """
    try:
        return load_prefab("example_prefab")
    except FileNotFoundError:
        warnings.warn("Example prefab not found. You may need to create example prefabs.")
        return Pipeline(name="empty_example")

# Export public functions
__all__ = [
    "get_prefab_dir", 
    "list_prefabs", 
    "load_prefab", 
    "combine_prefabs", 
    "save_prefab", 
    "example_prefab1"
] 