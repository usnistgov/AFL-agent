# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Source code dir relative to this file
project = 'AFL-agent'
copyright = '2024, Tyler Martin, Peter Beaucage'
author = 'Tyler Martin, Peter Beaucage'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
	'sphinx.ext.duration',
	'sphinx.ext.doctest',
	'sphinx.ext.autodoc',
	'sphinx.ext.autosummary',
	'sphinx.ext.napoleon',  # Support for Google and NumPy style docstrings
	'sphinx.ext.viewcode',  # Add links to source code
	'sphinx.ext.intersphinx',  # Link to other project's documentation
	'sphinx_rtd_theme',  # Read the Docs theme
	'sphinx.ext.todo',  # For development notes
	'sphinx.ext.coverage',  # For documentation coverage reports
]

# Intersphinx configuration
intersphinx_mapping = {
	'python': ('https://docs.python.org/3', None),
}

# Autodoc settings
autodoc_default_options = {
	'members': True,
	'member-order': 'bysource',
	'special-members': '__init__',
	'undoc-members': True,
	'exclude-members': '__weakref__'
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# Warning settings
nitpicky = True  # Be extra picky about references
suppress_warnings = ['ref.python']  # Suppress specific warning types if needed

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
	'navigation_depth': 4,
	'titles_only': False,
	'style_external_links': True,
	'style_nav_header_background': '#2980B9',
	'collapse_navigation': True,
	'sticky_navigation': True,
	'navigation_with_keys': True,
}
html_static_path = ['_static']

# These paths are either relative to html_static_path or fully qualified paths
html_css_files = [
	'custom.css',
]
