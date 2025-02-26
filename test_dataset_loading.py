#!/usr/bin/env python3
"""
Test script to verify that the dataset loading functionality works correctly.
"""

import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from AFL.double_agent.datasets import example_dataset1, list_datasets
    
    print("Available datasets:", list_datasets())
    
    # Load the example dataset
    ds = example_dataset1()
    
    print("\nSuccessfully loaded example_dataset1:")
    print(f"Dataset dimensions: {dict(ds.dims)}")
    print(f"Dataset variables: {list(ds.data_vars)}")
    print(f"Dataset coordinates: {list(ds.coords)}")
    
    # Print a small sample of the data
    print("\nSample data:")
    for var in list(ds.data_vars)[:2]:  # Show first two variables
        print(f"\n{var}:")
        print(ds[var].isel({dim: slice(0, 3) for dim in ds[var].dims}))
    
except ImportError as e:
    print(f"Error importing dataset module: {e}")
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")
except Exception as e:
    print(f"Unexpected error: {e}") 