Installation Guide
=================

This guide will help you install AFL-agent and set up your development environment.

Basic Installation
----------------

You can install AFL-agent using pip:

.. code-block:: bash

   pip install AFL-agent

Development Installation
---------------------

For development work, we recommend installing from source:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/AutoFormulationLab/AFL-agent.git
      cd AFL-agent

2. Create and activate a virtual environment (recommended):

   .. code-block:: bash

      # Using venv
      python -m venv venv
      source venv/bin/activate  # On Unix/macOS
      # OR
      .\venv\Scripts\activate  # On Windows

      # OR using conda
      conda env create -f env.yml
      conda activate afl-agent

3. Install in development mode:

   .. code-block:: bash

      pip install -e .

Dependencies
----------

AFL-agent requires the following main dependencies:

- Python >= 3.8
- NumPy
- xarray
- pandas
- scikit-learn
- torch (optional, for deep learning features)
- tensorflow (optional, for deep learning features)

All required dependencies will be automatically installed with pip. Optional dependencies can be installed using:

.. code-block:: bash

   pip install AFL-agent[ml]  # For machine learning dependencies
   pip install AFL-agent[all]  # For all optional dependencies

Troubleshooting
-------------

Common Installation Issues
^^^^^^^^^^^^^^^^^^^^^^^^

1. **Dependency Conflicts**
   
   If you encounter dependency conflicts, try creating a fresh virtual environment:

   .. code-block:: bash

      python -m venv fresh-env
      source fresh-env/bin/activate
      pip install AFL-agent

2. **Build Errors**

   If you encounter build errors, make sure you have the required system dependencies:

   .. code-block:: bash

      # On Ubuntu/Debian
      sudo apt-get update
      sudo apt-get install python3-dev build-essential

   For other operating systems, please refer to your system's package manager.

Getting Help
^^^^^^^^^^

If you continue to experience issues, please:

1. Check our `GitHub Issues <https://github.com/AutoFormulationLab/AFL-agent/issues>`_ page
2. Search for existing issues that might address your problem
3. Create a new issue if your problem hasn't been reported 