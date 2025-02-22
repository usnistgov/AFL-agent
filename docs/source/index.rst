AFL-agent
=========

AFL-agent is a Python library that implements autonomous active learning agents for material science applications, as described in the manuscript *"Autonomous Small-Angle Scattering for Accelerated Soft Material Formulation Optimization"*. The library is designed to autonomously guide experimental measurement campaigns to efficiently map phase boundaries and optimize material properties in soft material formulations using advanced machine learning techniques.


Key Features
-----------

* Library of machine learning operations that can be composed into pipelines
* Pipelines are modular, visualizable, serializable, and self-documenting
* Trivial support for multimodal data processing
* All intermediate pipeline operations are stored in a xarray-based data model
* Rich visualization tools for analyzing calculations


Installation
-----------

You can install AFL-agent using pip:

.. code-block:: bash

   pip install git+https://github.com/usnistgov/afl-agent

For development installation with all optional dependencies:


.. code-block:: bash

   git clone https://github.com/usnistgov/afl-agent.git
   cd afl-agent
   pip install -e .[all]

Documentation Contents
--------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials/index
   how-to/index
   explanations/index
   reference/index

Citation
--------

If you use AFL-agent in your research, please cite the manuscript:

*"Autonomous Small-Angle Scattering for Accelerated Soft Material Formulation Optimization"* (under review)


