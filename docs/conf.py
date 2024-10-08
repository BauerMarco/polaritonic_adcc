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
sys.path.insert(0, os.path.abspath('..'))

import source
import source.full_qed_matrix
import source.functions
import source.__init__
import source.qed_adc_in_std_basis_with_self_en
import source.qed_amplitude_vec
import source.qed_matrix_from_diag_adc
import source.qed_matrix_working_equations
import source.qed_mp
import source.qed_npadc_exstates
import source.qed_npadc_s2s_tdm_terms
import source.qed_ucc
import source.refstate
import source.workflow
import source.solver.davidson
import source.solver.diis
import source.solver.LanczosIterator
import source.solver.lanczos
import source.solver.orthogonaliser

# -- Project information -----------------------------------------------------

project = 'Polaritonic_adcc'
copyright = '2023, Marco Bauer'
author = 'Marco Bauer'

# The full version, including alpha/beta/rc tags
release = '0.1.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    # 'sphinx.ext.imgmath',
    # 'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinxcontrib.apidoc',
    'sphinx.ext.autosummary',
]
# extensions = ['sphinx_automodapi.automodapi']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'backends', 'testdata']

# pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

#html_css_files = [
#    'custom_theme.css',
#]

# html_theme_options = {
#    "canonical_url": "https://polaritonic_adcc.org/"
# }

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# __init__ docstring appears in documentation
autoclass_content = "init"


numpydoc_show_class_members = False

apidoc_module_dir = '../source'
apidoc_output_dir = 'apidoc_generated'
apidoc_excluded_paths = ['testdata', 'backends']
apidoc_separate_modules = True

