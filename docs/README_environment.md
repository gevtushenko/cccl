# CCCL Documentation Build Environment

This directory contains tools and configuration for building CCCL documentation using Sphinx, Breathe, and Doxygen.

## Quick Start

### 1. Check Your Environment

Before building documentation, verify your environment:

```bash
# Quick check (shell script)
./check_env.sh

# Detailed version check (Python script)
python check_versions.py
```

### 2. Install Dependencies

If the environment check fails, install the required dependencies:

```bash
# Install system dependencies
sudo apt-get install doxygen python3 python3-pip

# Install Python dependencies (exact versions for reproducible builds)
pip install -r requirements.txt

# OR for development with latest compatible versions
pip install -r requirements-dev.txt
```

### 3. Build Documentation

```bash
# Build all documentation
make html

# Or use the build script
../gen_docs.bash
```

## Version Strategy

This project uses **exact versions** in `requirements.txt` to ensure reproducible documentation builds across different machines. This approach:

- **Prevents "works on my machine" issues** - Everyone gets the same versions
- **Eliminates build failures** from incompatible version updates
- **Makes troubleshooting easier** - Known working versions
- **Improves CI/CD reliability** - Consistent builds

### Version Files

- **`requirements.txt`** - Exact versions (recommended for most users)
- **`requirements-dev.txt`** - Minimum versions (for development/testing)
- **`requirements-exact.txt`** - Backup of exact versions

## Environment Inventory

The complete inventory of versions used on the working machine is documented in `version_inventory.md`. This includes:

- System information (OS, Python version, etc.)
- All Python package versions
- Doxygen and Sphinx versions
- Configuration details

## Troubleshooting

### Common Issues

1. **Sphinx version too old**
   - Error: "Sphinx version X.Y.Z is too old"
   - Solution: `pip install --upgrade sphinx>=8.1.3`

2. **Doxygen not found**
   - Error: "doxygen: command not found"
   - Solution: `sudo apt-get install doxygen`

3. **Breathe import error**
   - Error: "No module named 'breathe'"
   - Solution: `pip install breathe>=4.36.0`

4. **Theme not found**
   - Error: "nvidia_sphinx_theme not found"
   - Solution: `pip install nvidia-sphinx-theme>=0.0.8`

### Version Compatibility

- **Sphinx**: 8.1.3+ (recent version, may not be available on older systems)
- **Doxygen**: 1.9.1 (stable, widely available)
- **Python**: 3.8+ (3.10.12 on working machine)
- **All Python packages**: See `requirements.txt` for exact versions, `requirements-dev.txt` for minimum versions

### Environment Differences

If documentation builds on one machine but fails on another, check:

1. **Python version**: Use `python --version`
2. **Package versions**: Use `python check_versions.py`
3. **System tools**: Use `./check_env.sh`
4. **Installation locations**: Check if tools are in PATH

### Replicating the Working Environment

To replicate the exact working environment:

1. Use the same Python version (3.10.12)
2. Install the exact package versions listed in `version_inventory.md`
3. Use the same Doxygen version (1.9.1)
4. Ensure all tools are in PATH

## Files

- `requirements.txt` - Python package dependencies (exact versions for reproducible builds)
- `requirements-dev.txt` - Python package dependencies (minimum versions for development)
- `requirements-exact.txt` - Backup of exact versions
- `conf.py` - Sphinx configuration
- `Doxyfile.in` - Doxygen configuration template
- `version_inventory.md` - Complete version inventory
- `check_versions.py` - Detailed version checker
- `check_env.sh` - Quick environment checker
- `gen_docs.bash` - Documentation build script

## Support

If you encounter issues:

1. Run both check scripts and note any failures
2. Compare versions with `version_inventory.md`
3. Check the troubleshooting section above
4. Ensure all dependencies are installed in the correct locations
