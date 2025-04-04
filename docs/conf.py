# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from jaxopt.version import __version__


# -- Project information -----------------------------------------------------

project = 'JAXopt'
copyright = '2021-2022, the JAXopt authors'
author = 'JAXopt authors'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon', # napoleon on top of autodoc: https://stackoverflow.com/a/66930447 might correct some warnings
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_autodoc_typehints',
    'myst_nb',
    "sphinx_remove_toctrees",
    'sphinx_rtd_theme',
    'sphinx_gallery.gen_gallery',
    'sphinx_copybutton',
]

sphinx_gallery_conf = {
     'examples_dirs': '../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     'ignore_pattern': r'_test\.py',  # no gallery for test of examples
     "doc_module": "jaxopt",
     "backreferences_dir": os.path.join("modules", "generated"),
}


source_suffix = ['.rst', '.ipynb', '.md']

autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": True}

master_doc = 'index'

autodoc_typehints = 'description'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    'build/html',
    'build/jupyter_execute',
    'README.md',
    '_build',
    '**.ipynb_checkpoints',
    # Ignore markdown source for notebooks; myst-nb builds from the ipynb
    'notebooks/deep_learning/*.md',
    'notebooks/distributed/*.md',
    'notebooks/implicit_diff/*.md']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'sphinx_rtd_theme'
html_logo = ''
html_favicon = ''

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]
html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "google", # Username
    "github_repo": "jaxopt", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/docs/", # Path in the checkout to the docs root
}


# -- Options for myst ----------------------------------------------
nb_execution_mode = "force"
nb_execution_allow_errors = False
nb_execution_fail_on_error = True  # Requires https://github.com/executablebooks/MyST-NB/pull/296
myst_enable_extensions = ['dollarmath'] # To display maths in notebook

# Notebook cell execution timeout; defaults to 30.
nb_execution_timeout = 100

# List of patterns, relative to source directory, that match notebook
# files that will not be executed.
nb_execution_excludepatterns = [
    # Slow notebook
    'notebooks/deep_learning/*.*',
    'notebooks/distributed/*.*',
    'notebooks/implicit_diff/dataset_distillation.*',
    'notebooks/implicit_diff/maml.*',
]
