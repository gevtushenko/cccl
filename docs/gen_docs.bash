#!/usr/bin/env bash

# This script builds CCCL documentation using Sphinx directly
#
# Usage: ./gen_docs.bash [clean]

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
cd $SCRIPT_PATH

# Configuration
SPHINXOPTS="${SPHINXOPTS:---keep-going}"
SPHINXBUILD="${SPHINXBUILD:-python3 -m sphinx.cmd.build}"
BUILDDIR="_build"
DOXYGEN="${DOXYGEN:-doxygen}"

# Handle clean command
if [ "$1" = "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf ${BUILDDIR}/*
    exit 0
fi

## Clean image directory, without this any artifacts will prevent fetching
rm -rf img
mkdir -p img

# Pull cub images
if [ ! -d cubimg ]; then
    git clone -b gh-pages https://github.com/NVlabs/cub.git cubimg
fi

if [ ! -n "$(find cubimg -name 'example_range.png')" ]; then
    wget -q https://raw.githubusercontent.com/NVIDIA/NVTX/release-v3/docs/images/example_range.png -O cubimg/example_range.png
fi

if [ ! -n "$(find img -name '*.png')" ]; then
    wget -q https://docs.nvidia.com/cuda/_static/Logo_and_CUDA.png -O img/logo.png

    # Parse files and collects unique names ending with .png
    imgs=( $(grep -R -o -h '[[:alpha:][:digit:]_]*.png' ../cub/cub | uniq) )
    imgs+=( "cub_overview.png" "nested_composition.png" "tile.png" "blocked.png" "striped.png" )

    for img in "${imgs[@]}"
    do
        echo ${img}
        cp cubimg/${img} img/${img}
    done
fi

# Check if documentation dependencies are installed
echo "Checking for documentation dependencies..."
if ! python3 -c "import sphinx" 2>/dev/null; then
    echo "Installing documentation dependencies..."
    pip3 install -r requirements.txt || {
        echo "Error: Failed to install documentation dependencies"
        echo "Please install manually: pip3 install -r requirements.txt"
        exit 1
    }
fi

# Generate Doxygen XML in parallel (if doxygen is available)
if which ${DOXYGEN} > /dev/null 2>&1; then
    echo "Generating Doxygen XML..."
    mkdir -p ${BUILDDIR}/doxygen/cub ${BUILDDIR}/doxygen/thrust ${BUILDDIR}/doxygen/cudax ${BUILDDIR}/doxygen/libcudacxx
    
    # Run all Doxygen builds in parallel
    (cd cub && ${DOXYGEN} Doxyfile) &
    (cd thrust && ${DOXYGEN} Doxyfile) &
    (cd cudax && ${DOXYGEN} Doxyfile) &
    (cd libcudacxx && ${DOXYGEN} Doxyfile) &
    wait
    
    echo "Doxygen complete"
else
    echo "Skipping Doxygen (not installed)"
fi

# Build Sphinx HTML documentation
echo "Building documentation with Sphinx..."
${SPHINXBUILD} -b html . ${BUILDDIR}/html ${SPHINXOPTS}

echo "Documentation build complete! HTML output is in ${BUILDDIR}/html/"