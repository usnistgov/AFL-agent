# # Configuration file for the Sphinx documentation builder.
# #
# # For the full list of built-in configuration values, see the documentation:
# # https://www.sphinx-doc.org/en/master/usage/configuration.html
# 
# import os
# import sys
# from pathlib import Path
# 
# # Add the project root directory to the Python path
# sys.path.insert(0, os.path.abspath('../..'))
# 
# # Import the package to get the version
# import AFL.double_agent
# 
# # -- Project information -----------------------------------------------------
# # https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
# 
# project = 'AFL-agent'
# copyright = '2025, Tyler B. Martin, Peter A. Beaucage, Duncan R. Sutherland'
# author = 'Tyler B. Martin, Peter A. Beaucage, Duncan R. Sutherland'
# 
# version = AFL.double_agent.__version__.split('+')[0]
# release = version
# 
# # -- General configuration ---------------------------------------------------
# # https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# 
# extensions = [
#     'sphinx.ext.autodoc',
#     'sphinx.ext.viewcode',
#     'sphinx.ext.napoleon',
#     'sphinx.ext.intersphinx',
#     'sphinx.ext.autosummary',
#     'sphinx.ext.mathjax',
# ]
# 
# # Autodoc settings
# autodoc_default_options = {
#     'members': True,
#     'member-order': 'bysource',
#     'special-members': '__init__',
#     'undoc-members': True,
#     'exclude-members': '__weakref__',
#     'imported-members': False,
#     'show-inheritance': True,
# }
# 
# # Suppress warnings about external dependencies
# suppress_warnings = [
#     'ref.ref',
#     'ref.term',
#     'toc.not_included',
#     'toc.excluded',
# ]
# 
# # Autosummary settings
# autosummary_generate = True
# autosummary_imported_members = True
# add_module_names = False
# 
# templates_path = []
# exclude_patterns = []
# 
# language = 'en'
# 
# # Napoleon settings
# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = True
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True
# napoleon_type_aliases = None
# 
# # -- Options for HTML output -------------------------------------------------
# # https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# 
# html_theme = 'pydata_sphinx_theme'
# html_static_path = ['_static']
# 
# # -- Options for LaTeX output ---------------------------------------------
# latex_elements = {
#     'papersize': 'letterpaper',
#     'pointsize': '10pt',
# }
# 
# # Grouping the document tree into LaTeX files
# latex_documents = [
#     ('index', 'AFL-agent.tex', 'AFL-agent Documentation',
#      'Tyler B. Martin, Peter A. Beaucage, Duncan R. Sutherland', 'manual'),
# ]
# 
# # Intersphinx configuration
# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3', None),
#     'numpy': ('https://numpy.org/doc/stable/', None),
#     'scipy': ('https://docs.scipy.org/doc/scipy/', None),
#     'pandas': ('https://pandas.pydata.org/docs/', None),
#     'matplotlib': ('https://matplotlib.org/stable/', None),
#     'sklearn': ('https://scikit-learn.org/stable/', None),
#     'xarray': ('https://docs.xarray.dev/en/stable/', None),
#     'plotly': ('https://plotly.com/python-api-reference/', None),
# }
# 


# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0,os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = 'AFL-agent'
copyright = (
    ": Official Contribution of the US Government.  Not subject to copyright in the United States."
)
author = 'Tyler B. Martin, Peter A. Beaucage, Duncan R. Sutherland'


# The full version, including alpha/beta/rc tags
from AFL.double_agent import __version__
version = AFL.double_agent.__version__.split('+')[0]
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",  # numpydoc and google docstrings
]

# Ignore annoying type exception warnings which often come from newlines
nitpick_ignore = [("py:class", "type")]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

html_theme = "pydata_sphinx_theme"
# html_logo = "source/_static/Logo_PyHyperO9_Light.svg"
# html_theme_options = {
#     "logo": {
#         "image_light": "source/_images/Logo_PyHyperO9_Light.svg",
#         "image_dark": "source/_images/Logo_PyHyperO10_Dark.svg",
#     },
#     "github_url": "https://github.com/usnistgov/PyHyperScattering",
#     "collapse_navigation": True,
#     #   "external_links": [
#     #       {"name": "Learn", "url": "https://numpy.org/numpy-tutorials/"},
#     #       {"name": "NEPs", "url": "https://numpy.org/neps"}
#     #       ],
#     "header_links_before_dropdown": 6,
#     # Add light/dark mode and documentation version switcher:
#     "navbar_end": ["theme-switcher", "navbar-icon-links"],
# 
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["source/_static"]
