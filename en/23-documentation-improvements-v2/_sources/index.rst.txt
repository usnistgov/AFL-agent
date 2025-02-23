AFL-agent
=========

.. only:: html 

    .. image:: _static/agent_loop.svg
        :alt: AFL-agent Logo
        :width: 800


AFL-agent is a Python library that implements autonomous active learning agents for material science applications, as described in the manuscript *"Autonomous Small-Angle Scattering for Accelerated Soft Material Formulation Optimization"*. The library is designed to autonomously guide experimental measurement campaigns to efficiently map phase boundaries and optimize material properties in soft material formulations using advanced machine learning techniques.


Key Features
------------

* Library of machine learning operations that can be composed into pipelines
* Pipelines are modular, visualizable, serializable, and self-documenting
* Trivial support for multimodal data processing
* All intermediate pipeline operations are stored in a xarray-based data model
* Rich visualization tools for analyzing calculations


Installation
------------

You can install AFL-agent using pip:

.. code-block:: bash

   pip install git+https://github.com/usnistgov/afl-agent


Please see the :doc:`tutorials/installation` page for more details.


Citation
--------

If you use AFL-agent in your research, please cite the manuscript:

*"Autonomous Small-Angle Scattering for Accelerated Soft Material Formulation Optimization"* (under review)

Contents
--------

This documentation is organized according to the philosphy described by `diataxis.fr <https://diataxis.fr>`_. It is organized into four sections:

* :doc:`Tutorials <tutorials/index>`: Step-by-step guides for beginners to use AFL-agent
* :doc:`How-to <how-to/index>`: Guides for specific tasks
* :doc:`Explanations <explanations/index>`: Discussions outlining the principles and underlying concepts of AFL-agent
* :doc:`Reference <reference/index>`: Detailed technical reference 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials/installation
   tutorials/index
   how-to/index
   explanations/index
   reference/index
