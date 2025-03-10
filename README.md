<p align="center">
  <img src="https://raw.githubusercontent.com/usnistgov/AFL-agent/main/docs/source/_static/pipeline_horizontal_light.svg" alt="AFL-Agent Logo" width="600">
</p>

# AFL-Agent: Framework for Autonomous Material Science

[![Documentation](https://img.shields.io/badge/docs-pages.nist.gov-blue)](https://pages.nist.gov/AFL-agent/en/latest/index.html) 
[![Tests](https://github.com/usnistgov/AFL-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/usnistgov/AFL-agent/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/usnistgov/AFL-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/usnistgov/AFL-agent)

AFL-agent is a Python library that implements the autonomous active learning agent described in the manuscript *"Autonomous Small-Angle Scattering for Accelerated Soft Material Formulation Optimization"*. The library is designed to autonomously guide experimental measurement campaigns to efficiently map phase boundaries in soft material formulations using advanced machine learning techniques.

## Overview
The AFL-agent library offers robust tools for building autonomous active learning agents tailored for material science applications. It integrates machine learning techniques with experimental design strategies to optimize experimental campaigns and efficiently map phase boundaries and optimize material properties in soft material formulations.

**Key Features:**

- Library of machine learning operations that can be composed into pipelines
- Pipelines are modular, visualizable, serializable, and self-documenting
- Trivial support for multimodal data processing
- All intermediate pipeline operations are stored in a xarray-based data model
- Rich visualization tools for analyzing calculations

## Installation

```
pip install git+https://github.com/usnistgov/afl-agent
```

## Quick Start

Below is an example showcasing how to build a pipeline for choosing a sample composition


```python
from AFL.double_agent import *
import numpy as np
import xarray as xr

with Pipeline() as pipeline:

    SavgolFilter(
        input_variable='measurement', 
        output_variable='derivative', 
        dim='x', 
        derivative=1
        )

    Similarity(
        input_variable='derivative', 
        output_variable='similarity', 
        params={'metric': 'cosine'}
        )

    SpectralClustering(
        input_variable='similarity',
        output_variable='labels',
        )

    GaussianProcessClassifier(
        feature_input_variable='composition',
        predictor_input_variable='labels',
        output_prefix='extrap',
    )

    MaxValueAF(
        input_variable='extrap_variance',
        output_variable='next_sample'
    )



# Generate synthetic data
n_samples = 10
n_points = 100
x = np.linspace(0, 10, n_points)
measurements = ... # data from your measurement
compositions = ... # composition of your samples

# Create dataset
ds = xr.Dataset(
    data_vars={
        'measurement': (['sample', 'x'], measurements),
        'composition': (['sample', 'components'], compositions)
    },
    coords={
        'x': x,
        'components': ['A', 'B', 'C']
    }
)

# Run the pipeline
ds_out = pipeline.calculate(ds)

#ds_out contains the following variables:
# - measurement: the original measurement data
# - derivative: the first derivative of the measurement data
# - similarity: the similarity between the samples
# - labels: the labels assigned to the samples
# - extrap_variance: the variance of the Gaussian process prediction
# - next_sample: the next sample to measure
```

## Citation

If you use AFL-agent in your research, please cite the manuscript:

*"Autonomous Small-Angle Scattering for Accelerated Soft Material Formulation Optimization"* (under review)

