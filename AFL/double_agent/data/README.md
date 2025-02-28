# Example Datasets for AFL-agent

This directory contains example datasets used by the AFL-agent library. These datasets are automatically loaded when using the `AFL.double_agent.datasets` module.

## Available Datasets

Currently, the following datasets are available:

- `example_dataset.nc`: A synthetic dataset with compositions, measurements, and ground truth labels.

## Usage

You can access these datasets using the datasets module:

```python
from AFL.double_agent.datasets import example_dataset1, list_datasets, load_dataset

# Load the example dataset
ds = example_dataset1()

# List all available datasets
available_datasets = list_datasets()
print(available_datasets)

# Load a dataset by name
ds = load_dataset("example_dataset")

# Print information about the dataset
print(f"Dataset dimensions: {dict(ds.sizes)}")
print(f"Dataset variables: {list(ds.data_vars)}")
print(f"Dataset coordinates: {list(ds.coords)}")
```

## Adding New Datasets

To add new datasets to the package:

1. Add the NetCDF (.nc) file to this directory
2. Update the `AFL/double_agent/datasets/__init__.py` file to include a function to load the new dataset
3. Update the documentation in `docs/source/tutorials/using_datasets.rst` 