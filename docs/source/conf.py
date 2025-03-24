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
import shutil

# Add the parent directory to the Python path
sys.path.insert(0,os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = 'AFL-agent'
copyright = (
    ": Official Contribution of the US Government.  Not subject to copyright in the United States"
)
author = 'Tyler B. Martin, Peter A. Beaucage, Duncan R. Sutherland'


# The full version, including alpha/beta/rc tags
#from AFL.double_agent import __version__
#version = None#__version__.split('+')[0]
#release = None#version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",  # numpydoc and google docstrings
    "sphinx_copybutton",
    "nbsphinx",
]

nbsphinx_requirejs_path = "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"

# Ignore annoying type exception warnings which often come from newlines
nitpick_ignore = [("py:class", "type")]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_extra_path = ["tutorials/iframe_figures"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import os
from importlib import import_module

html_theme = "pydata_sphinx_theme"
theme_module = import_module(html_theme.replace("-", "_"))
html_theme_path = [os.path.dirname(os.path.abspath(theme_module.__file__))]

# Add the favicon
html_favicon = "_static/logo_light.svg"

# Add the logo to replace the title text
html_logo = "_static/logo_text_large_light.svg"

html_theme_options = {
    "github_url": "https://github.com/usnistgov/afl-agent",
    "collapse_navigation": True,
    "header_links_before_dropdown": 6,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "logo": {
        "image_light": "_static/logo_text_large_light.svg",
        "image_dark": "_static/logo_text_large_dark.svg",
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# Copy the iframe_figures directories to the build output
# This is a workaround to allow the iframe_figures from plotly 
# to be displayed in the build output
def copy_iframe_figures_dirs(app, exception):
    """
    After the build, walk through the source directory, find all directories
    named "iframe_figures", and copy them (with their contents) to the corresponding
    location in the build output.
    """
    if exception is not None:
        return

    src_root = app.srcdir
    out_dir = app.builder.outdir

    # Walk through the source directory.
    for root, dirs, _ in os.walk(src_root):
        for dirname in dirs:
            if dirname == "iframe_figures":
                # Full path to the found iframe_figures directory in the source.
                src_iframe_dir = os.path.join(root, dirname)
                # Determine relative path from the source root.
                rel_path = os.path.relpath(src_iframe_dir, src_root)
                # Determine the target path in the build directory.
                target_dir = os.path.join(out_dir, rel_path)

                # If a directory already exists at the target, remove it.
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)

                # Now copy the entire directory tree.
                shutil.copytree(src_iframe_dir, target_dir)

def setup(app):
    app.connect("build-finished", copy_iframe_figures_dirs)
