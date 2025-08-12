# CCCL Documentation Configuration File
# Generated to replace repo-docs with direct Sphinx usage

import textwrap
from datetime import datetime
import os
import sys
import subprocess

# Add extension directory to path
sys.path.insert(0, os.path.abspath('_ext'))

# -- Project information -----------------------------------------------------

project = "CUDA Core Compute Libraries"
copyright = f"{datetime.now().year}, NVIDIA Corporation"
author = "NVIDIA Corporation"

# Version information
try:
    with open("VERSION.md", "r") as f:
        version = f.read().strip()
except:
    version = "latest"

release = version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "sphinx.ext.doctest",
    "breathe",  # For Doxygen integration - has built-in embed:rst support
    "exhale",  # Generate API reference pages from Doxygen XML
    "sphinx_design",  # For dropdown, card, and other directives
    "sphinx_copybutton",
    "nbsphinx",
    # "rst_processor",  # Disabled - breathe handles embed:rst natively
]

# Determine which project to build API docs for
api_project = os.environ.get('EXHALE_PROJECT', 'cub')  # default to cub

# Breathe configuration for Doxygen integration
breathe_projects = {
    "cub": "_build/doxygen/cub/xml",
    "thrust": "_build/doxygen/thrust/xml",
    "libcudacxx": "_build/doxygen/libcudacxx/xml",
    "cudax": "_build/doxygen/cudax/xml",
}

breathe_default_project = api_project
breathe_default_members = ('members', 'undoc-members')
breathe_show_enumvalue_initializer = True
breathe_domain_by_extension = {"cuh": "cpp", "h": "cpp", "hpp": "cpp"}

# Configure cpp domain to handle cub namespace
cpp_index_common_prefix = ['cub::']
cpp_id_attributes = []
cpp_paren_attributes = []

# Add support for .rst and .md files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Exclude patterns
exclude_patterns = [
    "_build",
    "_repo",
    "tools",
    "VERSION.md",
    "Thumbs.db",
    ".DS_Store",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "nvidia_sphinx_theme"

html_theme_options = {
    "repository_url": "https://github.com/NVIDIA/cccl",
    "use_repository_button": True,
    "navigation_depth": 4,
    "show_toc_level": 2,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "sidebar_includehidden": True,
    "collapse_navigation": False,
}

html_static_path = ["_static"] if os.path.exists("_static") else []

# Images directory
if os.path.exists("img"):
    html_static_path.append("img")

html_favicon = "img/logo.png" if os.path.exists("img/logo.png") else None

html_title = "CUDA Core Compute Libraries"

# Logo settings for nvidia-sphinx-theme
html_logo = "img/logo.png" if os.path.exists("img/logo.png") else None

# -- Options for extensions --------------------------------------------------

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# External links configuration
extlinks = {
    "github": ("https://github.com/NVIDIA/cccl/blob/main/%s", "%s"),
}


# Project-specific configurations for Exhale
project_configs = {
    'cub': {
        'containmentFolder': './cub/api_generated',
        'rootFileTitle': 'CUB API Reference',
        'doxygenStripFromPath': '../cub',
    },
    'thrust': {
        'containmentFolder': './thrust/api_generated', 
        'rootFileTitle': 'Thrust API Reference',
        'doxygenStripFromPath': '../thrust',
    },
}

# Exhale configuration for automated API generation
if api_project in project_configs:
    config = project_configs[api_project]
    exhale_args = {
        "containmentFolder": config['containmentFolder'],
        "rootFileName": "exhale_api.rst",
        "rootFileTitle": config['rootFileTitle'],
        "doxygenStripFromPath": config['doxygenStripFromPath'],
        "createTreeView": True,
        "exhaleExecutesDoxygen": False,
        "verboseBuild": True,
    }
else:
    # Default to CUB if invalid project specified
    exhale_args = {
        "containmentFolder": "./cub/api",
        "rootFileName": "exhale_api.rst",
        "rootFileTitle": "CUB API Reference",
        "doxygenStripFromPath": "../cub",
        "createTreeView": True,
        "exhaleExecutesDoxygen": False,
        "verboseBuild": True,
    }

# Config numpydoc
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = False

# Config copybutton
copybutton_prompt_text = ">>> |$ |# "
autosummary_imported_members = False
autosummary_generate = True
autoclass_content = "class"


def setup(app):
    if os.path.exists("_static/custom.css"):
        app.add_css_file("custom.css")
