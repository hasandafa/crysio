#!/usr/bin/env python3
"""
Setup script for CrysIO package.
Enhanced for materials science and machine learning applications.
"""

from setuptools import setup, find_packages
from pathlib import Path

def get_version():
    """Read version from crysio/__init__.py (nested structure)"""
    # Try crysio/__init__.py first (standard structure)
    init_file = Path(__file__).parent / "crysio" / "__init__.py"
    
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
    
    # Fallback to root __init__.py if crysio/__init__.py doesn't exist
    root_init = Path(__file__).parent / "__init__.py"
    if root_init.exists():
        try:
            with open(root_init, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("__version__"):
                        if '"' in line:
                            return line.split('"')[1]
                        elif "'" in line:
                            return line.split("'")[1]
        except Exception as e:
            print(f"Error reading version from {root_init}: {e}")
    
    # Final fallback
    return "0.3.1"

def get_long_description():
    """Read long description from README.md"""
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        try:
            with open(readme_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    
    return "Crystal I/O toolkit for preprocessing and visualizing crystal structures for machine learning applications"

def get_requirements():
    """Read requirements.txt with fallback to core deps"""
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        try:
            with open(req_file, "r", encoding="utf-8") as f:
                return [
                    line.strip() for line in f 
                    if line.strip() and not line.startswith("#")
                ]
        except Exception:
            pass
    
    # Fallback core dependencies for v0.3.1
    return [
        # Core scientific
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        
        # Visualization
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "seaborn>=0.11.0",
        
        # Network and API
        "requests>=2.25.0",
        "tqdm>=4.62.0",
        
        # Machine Learning
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "networkx>=2.6.0",
        
        # Materials Project API
        "mp-api>=0.33.0",
        
        # Interactive features
        "ipywidgets>=7.6.0",
        "py3dmol>=2.0.0",
        
        # Materials science (optional core)
        "pymatgen>=2023.1.0,<2024.0.0",
        "ase>=3.22.0",
        
        # Utilities
        "python-dotenv>=0.19.0",
        "h5py>=3.7.0",
        "pydantic>=1.10.0",
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
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    # Use find_packages() for automatic package discovery
    # This will find crysio.core, crysio.api, crysio.utils, etc.
    packages=find_packages(),
    
    # No package_dir needed - use standard nested structure
    
    package_data={
        'crysio': [
            'data/sample_structures/*',
            'data/atomic_properties/*',
            'visualization/templates/*',
            'tests/data/*',
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "jupyter>=1.0",
            "notebook>=6.4",
            "jupyterlab>=3.0",
            "pre-commit>=2.15.0",
        ],
        "viz": [
            "mayavi>=4.7.0",
            "vtk>=9.0.0",
            "pyvista>=0.32.0",
            "jupyter-threejs>=2.3.0",
        ],
        "extra": [
            "ase>=3.22.0",
            "pymatgen>=2023.1.0,<2024.0.0",
            "spglib>=1.16.0",
            "crystal-toolkit>=2023.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crysio=crysio.cli:main",
        ],
    },
    keywords=[
        "materials-science", 
        "crystallography", 
        "machine-learning", 
        "crystal-structure", 
        "vasp", 
        "cif", 
        "graph-neural-networks",
        "pytorch-geometric",
        "visualization",
        "materials-project"
    ],
    license="MIT",
    zip_safe=False,
)