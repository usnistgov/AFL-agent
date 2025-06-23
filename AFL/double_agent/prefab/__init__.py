"""
Prefabricated Pipelines module for AFL.double_agent.

This module provides access to prefabricated pipelines that can be used with AFL.double_agent.
It allows loading, listing, and combining multiple pipelines into a single pipeline.
"""

import os
import pathlib
import warnings
import json
import shutil
from typing import List, Dict, Optional, Union

from AFL.double_agent.Pipeline import Pipeline

# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------

_USER_PREFAB_SUBDIR = pathlib.Path.home() / ".afl" / "prefab"


def _ensure_user_prefabs_exist():
    """Ensure the user prefab directory exists and contains the default prefabs.

    This helper is called lazily by :func:`get_prefab_dir` so that the copy is
    performed the first time AFL.double_agent.prefab is imported *after* the
    package has been installed.  This avoids the need for a custom install
    hook while still providing the expected behaviour that the prefabs are
    available under ``~/.afl/prefab`` immediately after installation.
    """

    # Create ~/.afl/prefab if it does not yet exist
    _USER_PREFAB_SUBDIR.mkdir(parents=True, exist_ok=True)

    # Path that contains the prefabs *inside* the installed package
    _package_prefab_dir = pathlib.Path(__file__).parent

    # Copy each distributed prefab JSON into the user folder *once* – if the
    # user has already created a file with the same name we will not clobber
    # it in order to preserve any local modifications.
    for json_file in _package_prefab_dir.glob("*.json"):
        dest_file = _USER_PREFAB_SUBDIR / json_file.name
        if not dest_file.exists():
            try:
                shutil.copy2(json_file, dest_file)
            except Exception as exc:
                warnings.warn(
                    f"Could not copy prefab '{json_file.name}' to user folder: {exc}"
                )

# Call the helper immediately so that the folder is ready by the time any
# public API accesses it.
_ensure_user_prefabs_exist()

def get_prefab_dir():
    """Return the directory from which prefab pipelines should be read.

    The search path is now the *user* directory ``~/.afl/prefab``.  The
    directory is created on demand and populated with the default prefab JSON
    files shipped with the package (see :func:`_ensure_user_prefabs_exist`).
    """

    if not _USER_PREFAB_SUBDIR.exists():
        # This should rarely happen because the helper is run at import time,
        # but we protect against accidental deletions that may happen during a
        # long-running session.
        _ensure_user_prefabs_exist()

    return _USER_PREFAB_SUBDIR

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

def combine_prefabs(prefab_names: List[str], new_name: str | None = None) -> Pipeline:
    """
    Combine multiple prefabricated pipelines into a single pipeline.
    
    Parameters
    ----------
    prefab_names : List[str]
        List of prefabricated pipeline names to combine.
    new_name : str | None, default=None
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

def save_prefab(pipeline: Pipeline, name: str | None = None, overwrite: bool = False, description: str | None = None) -> str:
    """
    Save a pipeline as a prefabricated pipeline.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to save.
    name : str | None, default=None
        Name to save the pipeline as. If None, uses the pipeline's name.
    overwrite : bool, default=False
        Whether to overwrite an existing prefabricated pipeline with the same name.
    description : str | None, default=None
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
    
    # Use Pipeline.write_json to serialize first
    pipeline.write_json(str(file_path), overwrite=True, description=description)

    # ------------------------------------------------------------------
    # Inject package version information
    # ------------------------------------------------------------------
    try:
        from importlib.metadata import version as _pkg_version, PackageNotFoundError

        try:
            pkg_version = _pkg_version("AFL-agent")
        except PackageNotFoundError:
            # Fallback to _version module if package metadata unavailable (editable/dev installs)
            from AFL.double_agent import _version as _ver  # type: ignore
            pkg_version = getattr(_ver, "__version__", "unknown")

        # Load JSON we just wrote, add field, and overwrite file
        with open(file_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        data["afl_agent_version"] = pkg_version

        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=1)
    except Exception as exc:
        warnings.warn(
            f"Failed to annotate prefab '{file_path.name}' with package version: {exc}"
        )

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