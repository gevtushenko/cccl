#!/usr/bin/env python3
"""
Version checker for CCCL documentation build environment.

This script checks if all required versions for building CCCL documentation
are available on the current machine.
"""

import sys
import subprocess
import pkg_resources
from packaging import version

# Required versions from requirements.txt (exact versions)
REQUIRED_PACKAGES = {
    'sphinx': '8.1.3',
    'breathe': '4.36.0',
    'exhale': '0.3.7',
    'nvidia-sphinx-theme': '0.0.8',
    'myst-parser': '4.0.1',
    'sphinx-copybutton': '0.5.2',
    'sphinx-design': '0.6.1',
    'nbsphinx': '0.9.7',
    'numpydoc': '1.9.0',
}


def check_python_version():
    """Check Python version."""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print("✅ Python version is compatible")
    return True


def check_package_version(package_name, min_version):
    """Check if a package is installed and meets minimum version."""
    try:
        installed_version = pkg_resources.get_distribution(
            package_name).version
        if version.parse(installed_version) >= version.parse(min_version):
            print(f"✅ {package_name}: {installed_version} (>= {min_version})")
            return True
        else:
            print(
                f"❌ {package_name}: {installed_version} (required >= {min_version})")
            return False
    except pkg_resources.DistributionNotFound:
        print(f"❌ {package_name}: not installed (required >= {min_version})")
        return False


def check_doxygen():
    """Check if Doxygen is available."""
    try:
        result = subprocess.run(['doxygen', '--version'],
                                capture_output=True, text=True, check=True)
        doxygen_version = result.stdout.strip()
        print(f"✅ Doxygen: {doxygen_version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Doxygen: not found or not working")
        return False


def check_sphinx_build():
    """Check if sphinx-build is available."""
    try:
        result = subprocess.run(['sphinx-build', '--version'],
                                capture_output=True, text=True, check=True)
        sphinx_version = result.stdout.strip()
        print(f"✅ sphinx-build: {sphinx_version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ sphinx-build: not found or not working")
        return False


def main():
    """Main function to check all dependencies."""
    print("CCCL Documentation Build Environment Checker")
    print("=" * 50)

    all_good = True

    # Check Python version
    if not check_python_version():
        all_good = False

    print("\nChecking Python packages:")
    print("-" * 30)

    # Check Python packages
    for package, min_version in REQUIRED_PACKAGES.items():
        if not check_package_version(package, min_version):
            all_good = False

    print("\nChecking system tools:")
    print("-" * 30)

    # Check system tools
    if not check_doxygen():
        all_good = False

    if not check_sphinx_build():
        all_good = False

    print("\n" + "=" * 50)
    if all_good:
        print("✅ All dependencies are satisfied!")
        print("You should be able to build CCCL documentation.")
    else:
        print("❌ Some dependencies are missing or outdated.")
        print("Please install missing packages:")
        print("  pip install -r docs/requirements.txt")
        print("  sudo apt-get install doxygen")

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
