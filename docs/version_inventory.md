# Documentation Build Environment Inventory

This document contains the complete inventory of versions and dependencies used for building CCCL documentation on this machine.

## System Information

- **OS**: Linux 6.8.0-65-generic
- **Architecture**: x86_64
- **Python Version**: 3.10.12
- **Python Executable**: `/usr/bin/python`
- **pip Version**: 25.2
- **Python Path**: 
  - `/usr/lib/python310.zip`
  - `/usr/lib/python3.10`
  - `/usr/lib/python3.10/lib-dynload`
  - `/home/gevtushenko/.local/lib/python3.10/site-packages`
  - `/usr/local/lib/python3.10/dist-packages`
  - `/usr/local/lib/python3.10/dist-packages/openconnect_pulse_gui-0.1.dev21+g2c4050a-py3.10.egg`
  - `/usr/lib/python3/dist-packages`

## Core Documentation Tools

### Sphinx
- **Version**: 8.1.3
- **Location**: `/home/gevtushenko/.local/bin/sphinx-build`
- **Required Version**: >=8.1.3 (from requirements.txt)

### Doxygen
- **Version**: 1.9.1
- **Location**: `/usr/bin/doxygen`
- **Required Version**: Not specified in requirements.txt (system dependency)

## Python Dependencies

All Python packages are installed in `/home/gevtushenko/.local/lib/python3.10/site-packages`

### Primary Documentation Dependencies (from requirements.txt)

| Package | Installed Version | Required Version | Status |
|---------|------------------|------------------|---------|
| sphinx | 8.1.3 | >=8.1.3 | ✅ |
| breathe | 4.36.0 | >=4.36.0 | ✅ |
| exhale | 0.3.7 | >=0.3.0 | ✅ |
| nvidia-sphinx-theme | 0.0.8 | >=0.0.8 | ✅ |
| myst-parser | 4.0.1 | >=4.0.1 | ✅ |
| sphinx-copybutton | 0.5.2 | >=0.5.2 | ✅ |
| sphinx-design | 0.6.1 | >=0.6.1 | ✅ |
| nbsphinx | 0.9.7 | >=0.9.0 | ✅ |
| numpydoc | 1.9.0 | >=1.5.0 | ✅ |

### Additional Sphinx Dependencies (auto-installed)

| Package | Version |
|---------|---------|
| pydata-sphinx-theme | 0.16.1 |
| sphinxcontrib-applehelp | 2.0.0 |
| sphinxcontrib-devhelp | 2.0.0 |
| sphinxcontrib-htmlhelp | 2.1.0 |
| sphinxcontrib-jsmath | 1.0.1 |
| sphinxcontrib-qthelp | 2.0.0 |
| sphinxcontrib-serializinghtml | 2.0.0 |

## Configuration Files

### Sphinx Configuration
- **File**: `docs/conf.py`
- **Theme**: nvidia_sphinx_theme
- **Extensions**: 
  - sphinx.ext.autodoc
  - sphinx.ext.autosummary
  - sphinx.ext.intersphinx
  - sphinx.ext.napoleon
  - sphinx.ext.extlinks
  - sphinx.ext.mathjax
  - sphinx.ext.graphviz
  - sphinx.ext.doctest
  - breathe
  - sphinx_design
  - sphinx_copybutton
  - nbsphinx
  - auto_api_generator

### Doxygen Configuration
- **File**: `docs/Doxyfile.in`
- **Output**: XML only (for Breathe integration)
- **Input**: `../../../cub/cub ../../../thrust/thrust`
- **Exclusions**: `*/detail/* */test/* */tests/* */examples/* */__detail/*`

### Breathe Configuration
- **Projects**:
  - cub: `_build/doxygen/cub/xml`
  - thrust: `_build/doxygen/thrust/xml`
  - libcudacxx: `_build/doxygen/libcudacxx/xml`
  - cudax: `_build/doxygen/cudax/xml`
- **Default Project**: cub
- **Domain Mapping**: `.cuh`, `.h`, `.hpp` → cpp

## Build Process

The documentation build process involves:
1. **Doxygen**: Generates XML documentation from C++ source files
2. **Breathe**: Converts Doxygen XML to Sphinx-compatible format
3. **Sphinx**: Generates final HTML documentation

## Installation Commands

To replicate this environment on another machine:

```bash
# Install system dependencies
sudo apt-get install doxygen python3 python3-pip

# Install Python dependencies
pip install -r docs/requirements.txt
```

## Troubleshooting Notes

- Sphinx is installed in user space (`~/.local/bin/`)
- Doxygen is installed system-wide (`/usr/bin/`)
- All Python packages are installed in user space
- The build process requires both Doxygen and Sphinx to be available in PATH

## Version Compatibility Notes

- Sphinx 8.1.3 is a recent version that may not be available on all systems
- Doxygen 1.9.1 is a stable version that should be widely available
- All Python dependencies meet or exceed the minimum requirements specified in requirements.txt
