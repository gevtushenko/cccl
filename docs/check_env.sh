#!/bin/bash

# CCCL Documentation Environment Checker
# Quick shell script to check if all required tools are available

echo "CCCL Documentation Build Environment Check"
echo "=========================================="

# Check Python
echo "Python:"
if command -v python3 &> /dev/null; then
    python3 --version
else
    echo "❌ python3 not found"
    exit 1
fi

# Check pip
echo -e "\nPip:"
if command -v pip &> /dev/null; then
    pip --version
else
    echo "❌ pip not found"
    exit 1
fi

# Check Doxygen
echo -e "\nDoxygen:"
if command -v doxygen &> /dev/null; then
    doxygen --version
else
    echo "❌ doxygen not found"
    exit 1
fi

# Check sphinx-build
echo -e "\nSphinx-build:"
if command -v sphinx-build &> /dev/null; then
    sphinx-build --version
else
    echo "❌ sphinx-build not found"
    exit 1
fi

# Check if we're in the right directory
echo -e "\nDirectory check:"
if [ -f "requirements.txt" ]; then
    echo "✅ Found requirements.txt"
else
    echo "❌ requirements.txt not found - run this script from the docs/ directory"
    exit 1
fi

if [ -f "conf.py" ]; then
    echo "✅ Found conf.py"
else
    echo "❌ conf.py not found - run this script from the docs/ directory"
    exit 1
fi

echo -e "\n✅ Basic environment check passed!"
echo "Run 'python check_versions.py' for detailed version checking."
