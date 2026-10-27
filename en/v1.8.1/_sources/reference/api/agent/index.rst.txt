AFL.agent package
=================

.. warning::
   The ``agent`` module is deprecated and will be removed in a future version. Please use ``double_agent`` instead.

Core Components
---------------

.. toctree::
   :maxdepth: 2

   AcquisitionFunction
   EI_AcquisitionFunction
   UCB_AcquisitionFunction
   GaussianProcess
   HscedGaussianProcess
   PhaseLabeler
   PhaseMap
   PhaseMap_pandas
   Metric

Drivers and Automation
-----------------------

.. toctree::
   :maxdepth: 2

   AgentClient
   SAS_AgentDriver
   Multimodal_AgentDriver
   DowShampoo_SampleDriver
   SANS_AL_SampleDriver
   SAXS_AL_SampleDriver
   SAS_AL_SampleDriver
   SAS_Grid_AL_SampleDriver
   Multimodal_AL_SampleDriver
   vSAS_NoDeck_AL_SampleDriver
   WatchDog

Data Processing
----------------

.. toctree::
   :maxdepth: 2

   xarray_extensions
   reduce_usaxs
   util 