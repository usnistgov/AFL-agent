Building Pipelines
===================

.. warning::

    This tutorial is a work in progress.

Here we'll go into more detail on the Quick Start Example from
:doc:'getting_started'. In this example, we'll build a pipeline that
uses a Savitzky-Golay filter to compute the first derivative of the
measurement, then computes the similarity between the derivative and
itself, then clusters the data using spectral clustering, and finally
fits a Gaussian Process classifier to the data.


Input Data
----------

First let's define the input data for the pipeline. This codebase uses
:py:class:`xarray.Dataset` to store the data. This is a powerful and flexible
data structure for working with multi-dimensional data.

.. code-block:: python

   import numpy as np
   import xarray as xr

   # !!! these should be specific data so users understand the shape of the data
   measurements = ... # data from your measurement (e.g. SANS, SAXS, UV-vis, etc.)
   x = ... #x values of your data, (e.g. q-values, energy, wavenumber, wavelength, etc.)
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

.. warning::

   Show a screenshot of the dataset output.

   Plot the dataset?



Pipeline Step 1: Savitzky-Golay Filter
--------------------------------------

To begin, we'll instantiate a :py:class:`SavgolFilter` object using the a context
manager (i.e., the 'with' construct shown below). Using this approach, each
Pipeline operation that is defined in the context is automatically added to the
``my_first_pipeline`` variable.


.. code-block:: python

   from AFL.double_agent import *

   with Pipeline() as my_first_pipeline:

       SavgolFilter(
           input_variable='measurement', 
           output_variable='derivative', 
           dim='x', 
           derivative=1
           )

Going over the keyword arguments one by one:

- The ``input_variable`` keyword argument specifies the name of the variable in the dataset that will be used as
  the input to the Savitzky-Golay filter.
- The ``output_variable`` keyword argument specifies the name of the new variable that will be added to the dataset.
- The ``dim`` keyword argument specifies the dimension along which the filter will be applied.
- The ``derivative`` keyword argument specifies the order of the derivative to be computed.

We can inspect the pipeline by printing the ``my_first_pipeline`` variable.

.. code-block:: python

   my_first_pipeline.print()

.. warning::

   Add a screenshot of the pipeline printout.

Finally, we can run the pipeline on the dataset and plot the results.

.. code-block:: python

   ds = my_first_pipeline.calculate(ds)

   ds.measurement.isel(sample=0).plot()
   ds.derivative.isel(sample=0).plot()


Pipeline Step 2: Similarity
---------------------------


Pipeline Step 3: Spectral Clustering
-----------------------------------


Pipeline Step 4: Gaussian Process Classifier
--------------------------------------------


Pipeline Step 5: Acquisition Function
-------------------------------------


Full Pipeline
--------------

Let's loook at the full pipeline defined all at once.


.. code-block:: python

   from AFL.double_agent import *

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

