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


.. raw:: html

    <div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">\n<defs>\n<symbol id="icon-database" viewBox="0 0 32 32">\n<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>\n<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>\n<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>\n</symbol>\n<symbol id="icon-file-text2" viewBox="0 0 32 32">\n<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>\n<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>\n<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>\n<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>\n</symbol>\n</defs>\n</svg>\n<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.\n *\n */\n\n:root {\n  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));\n  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));\n  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));\n  --xr-border-color: var(--jp-border-color2, #e0e0e0);\n  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);\n  --xr-background-color: var(--jp-layout-color0, white);\n  --xr-background-color-row-even: var(--jp-layout-color1, white);\n  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);\n}\n\nhtml[theme="dark"],\nhtml[data-theme="dark"],\nbody[data-theme="dark"],\nbody.vscode-dark {\n  --xr-font-color0: rgba(255, 255, 255, 1);\n  --xr-font-color2: rgba(255, 255, 255, 0.54);\n  --xr-font-color3: rgba(255, 255, 255, 0.38);\n  --xr-border-color: #1f1f1f;\n  --xr-disabled-color: #515151;\n  --xr-background-color: #111111;\n  --xr-background-color-row-even: #111111;\n  --xr-background-color-row-odd: #313131;\n}\n\n.xr-wrap {\n  display: block !important;\n  min-width: 300px;\n  max-width: 700px;\n}\n\n.xr-text-repr-fallback {\n  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */\n  display: none;\n}\n\n.xr-header {\n  padding-top: 6px;\n  padding-bottom: 6px;\n  margin-bottom: 4px;\n  border-bottom: solid 1px var(--xr-border-color);\n}\n\n.xr-header > div,\n.xr-header > ul {\n  display: inline;\n  margin-top: 0;\n  margin-bottom: 0;\n}\n\n.xr-obj-type,\n.xr-array-name {\n  margin-left: 2px;\n  margin-right: 10px;\n}\n\n.xr-obj-type {\n  color: var(--xr-font-color2);\n}\n\n.xr-sections {\n  padding-left: 0 !important;\n  display: grid;\n  grid-template-columns: 150px auto auto 1fr 0 20px 0 20px;\n}\n\n.xr-section-item {\n  display: contents;\n}\n\n.xr-section-item input {\n  display: inline-block;\n  opacity: 0;\n  height: 0;\n}\n\n.xr-section-item input + label {\n  color: var(--xr-disabled-color);\n}\n\n.xr-section-item input:enabled + label {\n  cursor: pointer;\n  color: var(--xr-font-color2);\n}\n\n.xr-section-item input:focus + label {\n  border: 2px solid var(--xr-font-color0);\n}\n\n.xr-section-item input:enabled + label:hover {\n  color: var(--xr-font-color0);\n}\n\n.xr-section-summary {\n  grid-column: 1;\n  color: var(--xr-font-color2);\n  font-weight: 500;\n}\n\n.xr-section-summary > span {\n  display: inline-block;\n  padding-left: 0.5em;\n}\n\n.xr-section-summary-in:disabled + label {\n  color: var(--xr-font-color2);\n}\n\n.xr-section-summary-in + label:before {\n  display: inline-block;\n  content: "►";\n  font-size: 11px;\n  width: 15px;\n  text-align: center;\n}\n\n.xr-section-summary-in:disabled + label:before {\n  color: var(--xr-disabled-color);\n}\n\n.xr-section-summary-in:checked + label:before {\n  content: "▼";\n}\n\n.xr-section-summary-in:checked + label > span {\n  display: none;\n}\n\n.xr-section-summary,\n.xr-section-inline-details {\n  padding-top: 4px;\n  padding-bottom: 4px;\n}\n\n.xr-section-inline-details {\n  grid-column: 2 / -1;\n}\n\n.xr-section-details {\n  display: none;\n  grid-column: 1 / -1;\n  margin-bottom: 5px;\n}\n\n.xr-section-summary-in:checked ~ .xr-section-details {\n  display: contents;\n}\n\n.xr-array-wrap {\n  grid-column: 1 / -1;\n  display: grid;\n  grid-template-columns: 20px auto;\n}\n\n.xr-array-wrap > label {\n  grid-column: 1;\n  vertical-align: top;\n}\n\n.xr-preview {\n  color: var(--xr-font-color3);\n}\n\n.xr-array-preview,\n.xr-array-data {\n  padding: 0 5px !important;\n  grid-column: 2;\n}\n\n.xr-array-data,\n.xr-array-in:checked ~ .xr-array-preview {\n  display: none;\n}\n\n.xr-array-in:checked ~ .xr-array-data,\n.xr-array-preview {\n  display: inline-block;\n}\n\n.xr-dim-list {\n  display: inline-block !important;\n  list-style: none;\n  padding: 0 !important;\n  margin: 0;\n}\n\n.xr-dim-list li {\n  display: inline-block;\n  padding: 0;\n  margin: 0;\n}\n\n.xr-dim-list:before {\n  content: "(";\n}\n\n.xr-dim-list:after {\n  content: ")";\n}\n\n.xr-dim-list li:not(:last-child):after {\n  content: ",";\n  padding-right: 5px;\n}\n\n.xr-has-index {\n  font-weight: bold;\n}\n\n.xr-var-list,\n.xr-var-item {\n  display: contents;\n}\n\n.xr-var-item > div,\n.xr-var-item label,\n.xr-var-item > .xr-var-name span {\n  background-color: var(--xr-background-color-row-even);\n  margin-bottom: 0;\n}\n\n.xr-var-item > .xr-var-name:hover span {\n  padding-right: 5px;\n}\n\n.xr-var-list > li:nth-child(odd) > div,\n.xr-var-list > li:nth-child(odd) > label,\n.xr-var-list > li:nth-child(odd) > .xr-var-name span {\n  background-color: var(--xr-background-color-row-odd);\n}\n\n.xr-var-name {\n  grid-column: 1;\n}\n\n.xr-var-dims {\n  grid-column: 2;\n}\n\n.xr-var-dtype {\n  grid-column: 3;\n  text-align: right;\n  color: var(--xr-font-color2);\n}\n\n.xr-var-preview {\n  grid-column: 4;\n}\n\n.xr-index-preview {\n  grid-column: 2 / 5;\n  color: var(--xr-font-color2);\n}\n\n.xr-var-name,\n.xr-var-dims,\n.xr-var-dtype,\n.xr-preview,\n.xr-attrs dt {\n  white-space: nowrap;\n  overflow: hidden;\n  text-overflow: ellipsis;\n  padding-right: 10px;\n}\n\n.xr-var-name:hover,\n.xr-var-dims:hover,\n.xr-var-dtype:hover,\n.xr-attrs dt:hover {\n  overflow: visible;\n  width: auto;\n  z-index: 1;\n}\n\n.xr-var-attrs,\n.xr-var-data,\n.xr-index-data {\n  display: none;\n  background-color: var(--xr-background-color) !important;\n  padding-bottom: 5px !important;\n}\n\n.xr-var-attrs-in:checked ~ .xr-var-attrs,\n.xr-var-data-in:checked ~ .xr-var-data,\n.xr-index-data-in:checked ~ .xr-index-data {\n  display: block;\n}\n\n.xr-var-data > table {\n  float: right;\n}\n\n.xr-var-name span,\n.xr-var-data,\n.xr-index-name div,\n.xr-index-data,\n.xr-attrs {\n  padding-left: 25px !important;\n}\n\n.xr-attrs,\n.xr-var-attrs,\n.xr-var-data,\n.xr-index-data {\n  grid-column: 1 / -1;\n}\n\ndl.xr-attrs {\n  padding: 0;\n  margin: 0;\n  display: grid;\n  grid-template-columns: 125px auto;\n}\n\n.xr-attrs dt,\n.xr-attrs dd {\n  padding: 0;\n  margin: 0;\n  float: left;\n  padding-right: 10px;\n  width: auto;\n}\n\n.xr-attrs dt {\n  font-weight: normal;\n  grid-column: 1;\n}\n\n.xr-attrs dt:hover span {\n  display: inline-block;\n  background: var(--xr-background-color);\n  padding-right: 10px;\n}\n\n.xr-attrs dd {\n  grid-column: 2;\n  white-space: pre-wrap;\n  word-break: break-all;\n}\n\n.xr-icon-database,\n.xr-icon-file-text2,\n.xr-no-icon {\n  display: inline-block;\n  vertical-align: middle;\n  width: 1em;\n  height: 1.5em !important;\n  stroke-width: 0;\n  stroke: currentColor;\n  fill: currentColor;\n}\n</style><pre class=\'xr-text-repr-fallback\'>&lt;xarray.Dataset&gt; Size: 2kB\nDimensions:       (sample: 100, components: 2)\nCoordinates:\n  * components    (components) &lt;U1 8B &#x27;A&#x27; &#x27;B&#x27;\nDimensions without coordinates: sample\nData variables:\n    compositions  (sample, components) float64 2kB 7.403 16.21 ... 6.487 13.54</pre><div class=\'xr-wrap\' style=\'display:none\'><div class=\'xr-header\'><div class=\'xr-obj-type\'>xarray.Dataset</div></div><ul class=\'xr-sections\'><li class=\'xr-section-item\'><input id=\'section-669ff04d-8bd4-4f46-b289-2fb2ba1b6ea9\' class=\'xr-section-summary-in\' type=\'checkbox\' disabled ><label for=\'section-669ff04d-8bd4-4f46-b289-2fb2ba1b6ea9\' class=\'xr-section-summary\'  title=\'Expand/collapse section\'>Dimensions:</label><div class=\'xr-section-inline-details\'><ul class=\'xr-dim-list\'><li><span>sample</span>: 100</li><li><span class=\'xr-has-index\'>components</span>: 2</li></ul></div><div class=\'xr-section-details\'></div></li><li class=\'xr-section-item\'><input id=\'section-00e0b792-4c04-468b-a9c8-87fc6475f9e7\' class=\'xr-section-summary-in\' type=\'checkbox\'  checked><label for=\'section-00e0b792-4c04-468b-a9c8-87fc6475f9e7\' class=\'xr-section-summary\' >Coordinates: <span>(1)</span></label><div class=\'xr-section-inline-details\'></div><div class=\'xr-section-details\'><ul class=\'xr-var-list\'><li class=\'xr-var-item\'><div class=\'xr-var-name\'><span class=\'xr-has-index\'>components</span></div><div class=\'xr-var-dims\'>(components)</div><div class=\'xr-var-dtype\'>&lt;U1</div><div class=\'xr-var-preview xr-preview\'>&#x27;A&#x27; &#x27;B&#x27;</div><input id=\'attrs-a13d0fc4-f6e1-4d3c-970b-2ca8a33c315d\' class=\'xr-var-attrs-in\' type=\'checkbox\' disabled><label for=\'attrs-a13d0fc4-f6e1-4d3c-970b-2ca8a33c315d\' title=\'Show/Hide attributes\'><svg class=\'icon xr-icon-file-text2\'><use xlink:href=\'#icon-file-text2\'></use></svg></label><input id=\'data-19b81466-b102-440f-b3ad-302b5b900990\' class=\'xr-var-data-in\' type=\'checkbox\'><label for=\'data-19b81466-b102-440f-b3ad-302b5b900990\' title=\'Show/Hide data repr\'><svg class=\'icon xr-icon-database\'><use xlink:href=\'#icon-database\'></use></svg></label><div class=\'xr-var-attrs\'><dl class=\'xr-attrs\'></dl></div><div class=\'xr-var-data\'><pre>array([&#x27;A&#x27;, &#x27;B&#x27;], dtype=&#x27;&lt;U1&#x27;)</pre></div></li></ul></div></li><li class=\'xr-section-item\'><input id=\'section-8ab7d7ad-deb8-4b7d-933e-e9a7f77ad5f0\' class=\'xr-section-summary-in\' type=\'checkbox\'  checked><label for=\'section-8ab7d7ad-deb8-4b7d-933e-e9a7f77ad5f0\' class=\'xr-section-summary\' >Data variables: <span>(1)</span></label><div class=\'xr-section-inline-details\'></div><div class=\'xr-section-details\'><ul class=\'xr-var-list\'><li class=\'xr-var-item\'><div class=\'xr-var-name\'><span>compositions</span></div><div class=\'xr-var-dims\'>(sample, components)</div><div class=\'xr-var-dtype\'>float64</div><div class=\'xr-var-preview xr-preview\'>7.403 16.21 2.674 ... 6.487 13.54</div><input id=\'attrs-b583e436-16ea-4708-bf24-a459fd8facf0\' class=\'xr-var-attrs-in\' type=\'checkbox\' disabled><label for=\'attrs-b583e436-16ea-4708-bf24-a459fd8facf0\' title=\'Show/Hide attributes\'><svg class=\'icon xr-icon-file-text2\'><use xlink:href=\'#icon-file-text2\'></use></svg></label><input id=\'data-881cc496-93fa-4b67-a763-9cdd1e8200e2\' class=\'xr-var-data-in\' type=\'checkbox\'><label for=\'data-881cc496-93fa-4b67-a763-9cdd1e8200e2\' title=\'Show/Hide data repr\'><svg class=\'icon xr-icon-database\'><use xlink:href=\'#icon-database\'></use></svg></label><div class=\'xr-var-attrs\'><dl class=\'xr-attrs\'></dl></div><div class=\'xr-var-data\'><pre>array([[ 7.40333199, 16.20926476],\n       [ 2.67378057, 21.03046807],\n       [ 3.10150523, 20.4528175 ],\n       [ 7.39557556,  8.94702517],\n       [ 4.40942667,  6.94122655],\n       [ 6.72468826,  8.94123472],\n       [ 6.26171144,  2.49234715],\n       [ 4.52580996, 21.20419329],\n       [ 8.94100591, 12.92675046],\n       [ 9.87438903, 20.91388171],\n       [ 3.38449274,  9.21218184],\n       [ 6.36338262,  2.252374  ],\n       [ 4.67458917,  3.86194731],\n       [ 2.95716416, 17.31640739],\n       [ 3.62596052, 22.26073862],\n       [ 6.26036938,  2.32437859],\n       [ 9.17709512, 11.31530861],\n       [ 6.30283897, 11.88378513],\n       [ 9.11981729, 14.27343681],\n       [ 5.03833654, 23.31030186],\n...\n       [ 9.93830989, 20.53133213],\n       [ 8.83685992,  9.00257655],\n       [ 1.07828786, 10.64558206],\n       [ 1.22278866, 17.0164704 ],\n       [ 2.76412991, 22.70732898],\n       [ 2.30827862, 17.54677582],\n       [ 0.63533838,  6.61943845],\n       [ 4.87486834,  9.5850181 ],\n       [ 8.02435875,  0.18369401],\n       [ 1.27077277,  3.32348765],\n       [ 7.41038908, 17.55388693],\n       [ 9.02987718,  9.44301853],\n       [ 3.13077562,  1.6974947 ],\n       [ 7.51851777,  7.27266184],\n       [ 3.75578862, 15.50710087],\n       [ 0.36724792,  8.5270102 ],\n       [ 9.35490755, 10.16972519],\n       [ 7.82589475,  4.97632632],\n       [ 9.78112162, 22.14384886],\n       [ 6.48651103, 13.53629204]])</pre></div></li></ul></div></li><li class=\'xr-section-item\'><input id=\'section-4d50aa93-fa61-4efa-896c-d0524409d143\' class=\'xr-section-summary-in\' type=\'checkbox\'  ><label for=\'section-4d50aa93-fa61-4efa-896c-d0524409d143\' class=\'xr-section-summary\' >Indexes: <span>(1)</span></label><div class=\'xr-section-inline-details\'></div><div class=\'xr-section-details\'><ul class=\'xr-var-list\'><li class=\'xr-var-item\'><div class=\'xr-index-name\'><div>components</div></div><div class=\'xr-index-preview\'>PandasIndex</div><input type=\'checkbox\' disabled/><label></label><input id=\'index-767bbfd3-c7b6-4f57-8d24-db18a3876d0d\' class=\'xr-index-data-in\' type=\'checkbox\'/><label for=\'index-767bbfd3-c7b6-4f57-8d24-db18a3876d0d\' title=\'Show/Hide index repr\'><svg class=\'icon xr-icon-database\'><use xlink:href=\'#icon-database\'></use></svg></label><div class=\'xr-index-data\'><pre>PandasIndex(Index([&#x27;A&#x27;, &#x27;B&#x27;], dtype=&#x27;object&#x27;, name=&#x27;components&#x27;))</pre></div></li></ul></div></li><li class=\'xr-section-item\'><input id=\'section-57366338-daa1-41d5-835e-9bdbfcf16a41\' class=\'xr-section-summary-in\' type=\'checkbox\' disabled ><label for=\'section-57366338-daa1-41d5-835e-9bdbfcf16a41\' class=\'xr-section-summary\'  title=\'Expand/collapse section\'>Attributes: <span>(0)</span></label><div class=\'xr-section-inline-details\'></div><div class=\'xr-section-details\'><dl class=\'xr-attrs\'></dl></div></li></ul></div></div>'




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

