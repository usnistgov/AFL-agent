# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath('../..'))

# Import the package to get the version
import AFL.double_agent

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AFL-agent'
copyright = '2025, Tyler B. Martin, Peter A. Beaucage, Duncan R. Sutherland'
author = 'Tyler B. Martin, Peter A. Beaucage, Duncan R. Sutherland'

version = AFL.double_agent.__version__.split('+')[0]
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'imported-members': False,
    'show-inheritance': True,
}

# Suppress warnings about external dependencies
suppress_warnings = [
    'ref.ref',
    'ref.term',
    'toc.not_included',
    'toc.excluded',
]

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True
add_module_names = False

templates_path = []
exclude_patterns = []

language = 'en'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# -- Options for LaTeX output ---------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
}

# Grouping the document tree into LaTeX files
latex_documents = [
    ('index', 'AFL-agent.tex', 'AFL-agent Documentation',
     'Tyler B. Martin, Peter A. Beaucage, Duncan R. Sutherland', 'manual'),
]

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'plotly': ('https://plotly.com/python-api-reference/', None),
}
