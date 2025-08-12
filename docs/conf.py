# CCCL Documentation Configuration File
# Generated to replace repo-docs with direct Sphinx usage

import textwrap
from datetime import datetime
import os
import sys
import subprocess

# -- Project information -----------------------------------------------------

project = "CUDA C++ Core Libraries"
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
    "breathe",  # For Doxygen integration
    "exhale",  # For automatic C++ API generation
    "sphinx_design",  # For dropdown, card, and other directives
    "sphinx_copybutton",
    "nbsphinx"
]

# Breathe configuration for Doxygen integration
breathe_projects = {
    "cccl": "_build/doxygen/xml",  # For exhale
    "cub": "_build/doxygen/cub/xml",
    "thrust": "_build/doxygen/thrust/xml",
    "libcudacxx": "_build/doxygen/libcudacxx/xml",
    "cudax": "_build/doxygen/cudax/xml",
}

breathe_default_project = "cccl"
breathe_default_members = ('members', 'undoc-members')

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

html_title = "CUDA C++ Core Libraries"

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


# Exhale configuration for automatic C++ API generation
# Using targeted exclusions to avoid template parsing issues

exhale_args = {
    "containmentFolder": "_build/_api",
    "rootFileName": "index.rst",
    "doxygenStripFromPath": "../",
    "rootFileTitle": "C++ API",
    "createTreeView": True,
    "exhaleExecutesDoxygen": False,  # We run doxygen separately in Makefile
    "listingExclude": [
        "pointer_traits",  # Complex templates cause parsing issues
        "__memory",  # Skip internal memory utilities
        "__detail",  # Skip detail namespaces
        "detail",
    ],
    "exhaleDoxygenStdin": textwrap.dedent(r'''
        PROJECT_NAME = "CUDA C++ Core Libraries"
        BRIEF_MEMBER_DESC = YES
        BUILTIN_STL_SUPPORT = YES
        DOT_IMAGE_FORMAT = svg
        EXCLUDE_PATTERNS = */tests/* */test/* */examples/* */benchmarks/* */__pycache__/* */detail/* */__memory/* */__detail/*
        EXCLUDE_SYMBOLS = "@*" "*detail*" "*__*" "CUB_DETAIL*" "THRUST_DETAIL*" "_Has*"
        EXTENSION_MAPPING = cu=C++ cuh=C++
        EXTRACT_ALL = YES
        FILE_PATTERNS = *.c *.cc *.cpp *.h *.hpp *.cu *.cuh
        HAVE_DOT = NO
        HIDE_UNDOC_MEMBERS = NO
        INPUT = ../cub/cub ../thrust/thrust
        INTERACTIVE_SVG = NO
        SOURCE_BROWSER = NO
        ENABLE_PREPROCESSING = YES
        MACRO_EXPANSION = YES
        EXPAND_ONLY_PREDEF = NO
        SKIP_FUNCTION_MACROS = YES
        QUIET = YES
        WARNINGS = NO
        WARN_IF_UNDOCUMENTED = NO
        PREDEFINED += "__device__="
        PREDEFINED += "__host__="
        PREDEFINED += "__global__="
        PREDEFINED += "__forceinline__="
        PREDEFINED += "_CCCL_DOXYGEN_INVOKED"
        PREDEFINED += "_CCCL_HOST_DEVICE="
        PREDEFINED += "_CCCL_DEVICE="
        PREDEFINED += "_CCCL_HOST="
        PREDEFINED += "_CCCL_FORCEINLINE="
        GENERATE_XML = YES
        XML_OUTPUT = xml
    ''')
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
