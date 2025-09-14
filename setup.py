from setuptools import setup, find_packages
from pathlib import Path

def get_version():
    """Read version from crysio/__init__.py (nested structure)"""
    # For nested structure: crysio/crysio/__init__.py doesn't exist
    # Version should be in root __init__.py
    init_file = Path(__file__).parent / "__init__.py"
    
    if init_file.exists():
        try:
            with open(init_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("__version__"):
                        if '"' in line:
                            return line.split('"')[1]
                        elif "'" in line:
                            return line.split("'")[1]
        except Exception as e:
            print(f"Error reading version from {init_file}: {e}")
    
    # Fallback
    return "0.2.1"

def get_requirements():
    """Read requirements.txt"""
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        try:
            with open(req_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip() and not line.startswith("#")]
        except:
            pass
    
    return [
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "requests>=2.28.0",
        "tqdm>=4.64.0",
    ]

# Debug info
print(f"Setup.py running from: {Path(__file__).parent}")
print(f"Looking for packages with find_packages()...")

packages = find_packages()
print(f"Found packages: {packages}")

# Check if crysio subdirectory exists
crysio_subdir = Path(__file__).parent / "crysio"
if crysio_subdir.exists():
    print(f"crysio/ subdirectory exists with contents: {list(crysio_subdir.iterdir())}")
else:
    print("WARNING: crysio/ subdirectory not found!")

version = get_version()
print(f"Using version: {version}")

setup(
    name="crysio",
    version=version,
    author="Dafa, Abdullah Hasan",
    author_email="dafa.abdullahhasan@gmail.com",
    description="Crystal I/O toolkit for materials science ML",
    
    # Use find_packages() for automatic package discovery
    # This will find crysio.core, crysio.api, crysio.utils, etc.
    packages=find_packages(),
    
    # No package_dir needed - use standard nested structure
    
    package_data={
        'crysio': [
            'data/sample_structures/*',
            'data/atomic_properties/*',
        ],
    },
    include_package_data=True,
    
    python_requires=">=3.8",
    install_requires=get_requirements(),
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12", 
        "Programming Language :: Python :: 3.13",
    ],
    
    license="MIT",
    zip_safe=False,
)