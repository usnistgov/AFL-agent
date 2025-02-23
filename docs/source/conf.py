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
version = __version__.split('+')[0]
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

# html_theme = "pydata_sphinx_theme"
# # html_logo = "source/_static/agent_loop.svg"
# html_theme_options = {
#     # "logo": {
#     #     "image_light": "source/_static/agent_loop.svg",
#     #     "image_dark": "source/_static/agent_loop.svg",
#     # },
#     "github_url": "https://github.com/usnistgov/afl-agent",
#     "collapse_navigation": True,
#     #       ],
#     "header_links_before_dropdown": 6,
#     # Add light/dark mode and documentation version switcher:
#     "navbar_end": ["theme-switcher", "navbar-icon-links"],
# 
# }
import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["source/_static"]
