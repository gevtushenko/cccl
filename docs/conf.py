# CCCL Documentation Configuration File
# Generated to replace repo-docs with direct Sphinx usage

import textwrap
from datetime import datetime
import os
import sys
import subprocess

# Add extension directory to path
sys.path.insert(0, os.path.abspath('_ext'))

# Add Python CCCL package to path for autodoc
python_package_path = os.path.abspath('../python/cuda_cccl')
if os.path.exists(python_package_path):
    sys.path.insert(0, python_package_path)

# Pre-configure numpy mock to support type annotations
# This must be done before autodoc tries to import the modules
class MockNumpyModule:
    """Mock numpy module that supports type annotations"""
    class ndarray:
        pass
    
    def __or__(self, other):
        """Support union type syntax (|)"""
        return type('UnionType', (), {})
    
    def __getattr__(self, name):
        """Return mock for any attribute access"""
        return type(name, (), {})

# Pre-inject numpy mock if needed
if 'numpy' not in sys.modules:
    mock_numpy = MockNumpyModule()
    sys.modules['numpy'] = mock_numpy
    sys.modules['np'] = mock_numpy

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
    "breathe",  # For Doxygen integration - has built-in embed:rst support
    # "exhale",  # Disabled - causing build timeouts, API docs handled by breathe
    "sphinx_design",  # For dropdown, card, and other directives
    "sphinx_copybutton",
    "nbsphinx",
    # "rst_processor",  # Disabled - breathe handles embed:rst natively
    "auto_api_generator",  # Automatically generate API reference pages from Doxygen XML
]

# Breathe configuration for Doxygen integration
breathe_projects = {
    "cub": "_build/doxygen/cub/xml",
    "thrust": "_build/doxygen/thrust/xml",
    "libcudacxx": "_build/doxygen/libcudacxx/xml",
    "cudax": "_build/doxygen/cudax/xml",
}

breathe_default_project = "cub"
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

# Enable type hints to be shown in the documentation
autodoc_typehints = "description"
autodoc_type_aliases = {}

# Set Python domain primary for intersphinx
primary_domain = 'py'

# Mock imports for Python documentation - these modules may not be installed
autodoc_mock_imports = [
    "numba",
    "numba.core",
    "numba.core.cgutils",
    "numba.core.extending",
    "numba.core.typing",
    "numba.core.typing.ctypes_utils",
    "numba.core.typing.templates",
    "numba.cuda",
    "numba.cuda.cudadecl",
    "numba.cuda.dispatcher",
    "numba.extending",
    "numba.types",
    "pynvjitlink",
    "cuda.bindings",
    "cuda.bindings.driver",
    "cuda.bindings.runtime",
    "cuda.bindings.path_finder",
    "cuda.core",
    "cuda.core.experimental",
    "cuda.core.experimental._utils",
    "cuda.core.experimental._utils.cuda_utils",
    "llvmlite",
    "llvmlite.ir",
    "numpy",
    "cupy",
    "cuda.cccl.parallel.experimental._bindings",
    "cuda.cccl.parallel.experimental._bindings_impl"
]

# External links configuration
extlinks = {
    "github": ("https://github.com/NVIDIA/cccl/blob/main/%s", "%s"),
}


# Exhale not used - API documentation is handled directly through breathe directives

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
    
    # Fix for type annotations with mocked numpy - ensure numpy.ndarray exists
    import sys
    if 'numpy' in sys.modules and hasattr(sys.modules['numpy'], '__class__'):
        # If numpy is mocked, add ndarray attribute
        import types
        if not hasattr(sys.modules['numpy'], 'ndarray'):
            # Create a simple mock class for ndarray
            class MockNdarray:
                pass
            sys.modules['numpy'].ndarray = MockNdarray
