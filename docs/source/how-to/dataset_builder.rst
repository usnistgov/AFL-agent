Using the Dataset Builder
=========================

The Dataset Builder is a lightweight web application that allows you to upload
local data files and combine them into an ``xarray.Dataset``.  It is served by
the :class:`AFL.double_agent.DatasetBuilderDriver`.

Usage
-----

1. Start the driver::

     python -m AFL.double_agent.DatasetBuilderDriver

2. Open ``http://localhost:5051/dataset_builder`` in your browser.

3. Add a series of files using the **Add Files** button. Each row in the table
   allows you to choose the loader plugin, rename dimensions and specify
   coordinates.

4. Tick the files you wish to combine and press **Combine Selected**.  The
   resulting ``xarray.Dataset`` will be rendered on the page using its HTML
   representation.

Plugins
-------

Loader plugins provide support for different file formats.  The driver ships
with loaders for NumPy, pandas CSV/TSV, NetCDF, and basic SAS formats.  Custom
loaders can be added by placing new modules in ``AFL.double_agent.dataset_plugins``.
