AFL-agent
=========

.. only:: html 

    .. image:: _static/pipeline_horizontal_light.svg
        :alt: AFL-agent Logo
        :width: 800
        :class: only-light

    .. image:: _static/pipeline_horizontal_dark.svg
        :alt: AFL-agent Logo
        :width: 800
        :class: only-dark



AFL-agent is a Python library that allows users to implement active learning
agents for material science applications. [1] Rather than providing canned
algorithms, the library provides a framework that allows users to build their own.
This is achieved through a modular, extensible API that allows users to compose
multiple machine learning operations into executable pipelines.

If you use AFL-agent in your research, please cite the manuscript:

[1] *"Autonomous Small-Angle Scattering for Accelerated Soft Material Formulation Optimization"* (under review)


Key Features
------------

* Library of machine learning operations that can be composed into executable pipelines
* Pipelines are modular, visualizable, serializable, and self-documenting
* All intermediate pipeline operations are stored in a xarray-based data model
* Rich visualization tools for analyzing calculations
* Trivial support for multimodal data processing
* Support for phase boundary mapping and material property optimization


Autonomous Formulation Lab
--------------------------

The Autonomous Formulation Lab (AFL) is a National Institute of Standards and
Technology (NIST) program that seeks to accelerate the discovery and
optimization of soft materials through the development and application of
autonomous techniques to high-value measurements. Specifically, we design
robotic platforms that autonomously mix, synthesize and evaluate soft materials
and we study them using small-angle scattering (SANS) and small-angle X-ray
scattering (SAXS) and other techniques.


Installation
------------

You can install AFL-agent using pip:

.. code-block:: bash

   pip install git+https://github.com/usnistgov/afl-agent


Please see the :doc:`tutorials/installation` page for more details.

Documentation Structure
------------------------

This documentation is organized according to the philosphy described by Daniele Procida at `diataxis.fr <https://diataxis.fr>`_. It is organized into four sections:

* :doc:`Tutorials <tutorials/index>`: Step-by-step guides for beginners
* :doc:`How-to <how-to/index>`: Guides for specific tasks
* :doc:`Explanations <explanations/index>`: Discussions of underlying principles and concepts
* :doc:`Reference <reference/index>`: Detailed technical reference 


Table of Contents
------------------
.. toctree::
   :maxdepth: 2

   tutorials/installation
   tutorials/index
   how-to/index
   explanations/index
   reference/index
