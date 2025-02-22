Installation Guide
=================

This guide covers different ways to install AFL-agent based on your needs.

Prerequisites
------------

Before installing AFL-agent, ensure you have:

* Python 3.11 or later
* pip (Python package installer)
* git (for installation from source)

Basic Installation
----------------

The simplest way to install AFL-agent is directly from GitHub using pip:

.. code-block:: bash

   pip install git+https://github.com/usnistgov/afl-agent

Development Installation
----------------------

For development work, you'll want to clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/usnistgov/afl-agent.git
   cd afl-agent
   pip install -e .

Optional Dependencies
-------------------

AFL-agent has several optional dependency groups that can be installed based on your needs:

* ``jupyter``: Jupyter notebook support
* ``tensorflow``: TensorFlow and GPflow dependencies
* ``pytorch``: PyTorch dependencies
* ``automation``: Dependencies for automation features
* ``dev``: Development tools and testing dependencies

Install with optional dependencies using square brackets:

.. code-block:: bash

   # Install with jupyter support
   pip install -e .[jupyter]

   # Install with multiple optional dependencies
   pip install -e .[jupyter,tensorflow]

   # Install all optional dependencies
   pip install -e .[all]

Installation from a Specific Branch
--------------------------------

To install from a specific branch or commit:

.. code-block:: bash

   # Install from a branch
   pip install git+https://github.com/usnistgov/afl-agent.git@branch-name

   # Install from a specific commit
   pip install git+https://github.com/usnistgov/afl-agent.git@commit-hash

Offline Installation
------------------

For environments without internet access, you can create a wheel file:

1. On a machine with internet access:

   .. code-block:: bash

      git clone https://github.com/usnistgov/afl-agent.git
      cd afl-agent
      pip wheel .

2. Copy the generated ``.whl`` file to the offline machine and install:

   .. code-block:: bash

      pip install AFL_agent-version-py3-none-any.whl

Troubleshooting
-------------

Common installation issues and solutions:

1. **Version Conflicts**: If you encounter dependency conflicts, try creating a new virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install git+https://github.com/usnistgov/afl-agent

2. **Missing Dependencies**: If you see import errors after installation, ensure you have the necessary optional dependencies:

   .. code-block:: bash

      pip install -e .[all]

3. **Build Failures**: Make sure you have the latest pip and build tools:

   .. code-block:: bash

      pip install --upgrade pip setuptools wheel 