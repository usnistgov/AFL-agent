Quick Start Example
===================

This short example will help you get started with AFL-agent. See
:doc:'building_pipelines' for a more detailed tutorial. 

.. code-block:: python

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

Understanding the Example
-------------------------

Let's break down what's happening in this example:

1. First, we import the necessary modules from AFL-agent and other dependencies.
2. We create a pipeline using the context manager syntax (`with Pipeline() as pipeline:`).
3. We add several operations to the pipeline:

   - `SavgolFilter`: Calculates derivatives of the measurement data

   - `Similarity`: Computes similarity between measurement data

   - `SpectralClustering`: Groups (i.e., clusters) similar measurement data together

   - `GaussianProcessClassifier`: Extrapolates the clustering labels for all compositions

   - `MaxValueAF`: Selects the next sample to measure as the composition of highest entropy in phase label

4. We create a synthetic dataset with measurements and compositions
5. Finally, we run the pipeline on our dataset

Next Steps
----------

Now that you've seen a basic example, you might want to:

* Build more complicated pipelines: :doc:`building_pipelines`
* Understand the pipeline concept: :doc:`../explanations/architecture`