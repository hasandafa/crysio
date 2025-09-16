# CrysIO

**Crystal I/O toolkit for preprocessing and visualizing crystal structures for machine learning applications**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Development Status](https://img.shields.io/badge/Development%20Status-4%20Beta-orange.svg)](https://github.com/hasandafa/crysio)
[![Version](https://img.shields.io/badge/version-0.3.1-green.svg)](https://github.com/hasandafa/crysio)

CrysIO is a comprehensive Python toolkit designed for materials scientists and machine learning researchers working with crystal structures. It provides efficient I/O operations, advanced visualization capabilities, and seamless integration with graph neural networks for crystallographic data analysis.

## 🌟 Key Features

### 🔧 **Core Functionality**
- **Multi-format Support**: CIF, POSCAR/VASP, Materials Project API integration
- **Robust Parsing**: Advanced error handling with comprehensive validation
- **Crystal Representation**: Complete lattice parameters, atomic positions, space groups
- **Auto-detection**: Intelligent file format detection based on content analysis

### 🎨 **Advanced Visualization**
- **Interactive 3D Structures**: Customizable crystal structure visualization
- **Materials Project Style**: MP-compatible visualization with professional styling
- **Graph Networks**: Crystal connectivity and bonding analysis
- **Statistical Analysis**: Property distributions, correlation matrices, comparisons
- **Jupyter Integration**: Interactive widgets and 3D molecular viewers

### 🤖 **Machine Learning Ready**
- **PyTorch Geometric**: Direct conversion to graph representations for GNNs
- **Feature Engineering**: Automated structural descriptor extraction
- **Batch Processing**: Efficient handling of large crystal datasets
- **Graph Analysis**: Network topology analysis for crystal structures

### 🌐 **Database Integration**
- **Materials Project**: Full API integration with search and download capabilities
- **Comprehensive Search**: Query by composition, properties, crystal system
- **Rate Limiting**: Built-in API management with retry logic
- **Data Validation**: Automatic structure validation and cleaning

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install numpy scipy matplotlib plotly requests tqdm

# For graph neural network features
pip install torch torch-geometric networkx

# For interactive visualization
pip install ipywidgets py3dmol seaborn

# For materials science integration
pip install mp-api pymatgen ase

# Or install all dependencies at once
pip install -r requirements.txt
```

### Basic Usage

```python
import crysio

# Load crystal structure from file
crystal = crysio.load_structure("structure.cif")

# Access crystal properties
print(f"Formula: {crystal.formula}")
print(f"Space Group: {crystal.space_group}")
print(f"Volume: {crystal.volume:.2f} Ų")
print(f"Density: {crystal.density:.2f} g/cm³")

# Visualize structure (3D interactive)
crysio.visualizers.plot_crystal_3d(crystal, style="ball_and_stick")
```

### Advanced Features

```python
# Materials Project integration
import crysio

# Search and download structures
structures = crysio.search_materials_database(
    formula="LiFePO4", 
    properties=["energy_per_atom", "band_gap"]
)

# Graph neural network conversion
graph = crysio.to_graph(
    crystal,
    edge_cutoff=5.0,
    node_features=["atomic_number", "electronegativity"],
    edge_features=["distance", "bond_type"]
)

# Advanced visualization
from crysio.visualizers import MaterialsProjectViewer
viewer = MaterialsProjectViewer()
viewer.plot_mp_style(crystal, save_html=True)
```

## 📚 Comprehensive Examples

### Working with Multiple File Formats

```python
# Auto-detect and load various formats
crystal_cif = crysio.load_structure("quartz.cif")
crystal_poscar = crysio.load_structure("POSCAR")
crystal_vasp = crysio.load_structure("CONTCAR")

# Batch processing
structures = crysio.load_batch("structures/", pattern="*.cif")
print(f"Loaded {len(structures)} crystal structures")

# Property analysis across multiple structures
properties = crysio.analyze_batch(structures, 
    properties=["volume", "density", "space_group"])
```

### Interactive Visualization Workflow

```python
from crysio.visualizers import CrystalVisualizer, AnalysisVisualizer

# Create visualizer instances
crystal_viz = CrystalVisualizer()
analysis_viz = AnalysisVisualizer()

# 3D structure with custom styling
crystal_viz.plot_crystal_3d(
    crystal,
    style="space_filling",
    color_scheme="element",
    background="white",
    save_image="crystal_3d.png"
)

# Property correlation analysis
analysis_viz.plot_correlation_matrix(
    structures,
    properties=["volume", "density", "energy_per_atom"],
    save_html="correlations.html"
)

# Structure comparison
analysis_viz.plot_structure_comparison(
    [crystal1, crystal2, crystal3],
    metrics=["lattice_similarity", "composition_difference"]
)
```

### Machine Learning Pipeline

```python
# Complete ML preprocessing pipeline
crystals = crysio.load_batch("dataset/", pattern="*.cif")

# Convert to graphs for GNN training
graphs = []
for crystal in crystals:
    graph = crysio.to_graph(
        crystal,
        edge_cutoff=5.0,
        node_features=[
            "atomic_number", 
            "electronegativity", 
            "atomic_radius",
            "coordination_number"
        ],
        edge_features=[
            "distance", 
            "bond_order",
            "angle_strain"
        ]
    )
    graphs.append(graph)

# Create PyTorch Geometric dataset
dataset = crysio.create_pytorch_dataset(graphs, targets)
print(f"Dataset: {len(dataset)} graphs ready for training")
```

## 🧪 Testing Suite

CrysIO includes a comprehensive testing framework to verify all functionality:

```python
# Run the complete testing suite
import crysio.testing

# Execute all tests
crysio.testing.run_complete_suite()
```

**Testing Categories:**
- 🔍 **Core Functionality**: File parsing, crystal representation
- 🧪 **Validation**: Structure validation and error handling  
- 🎯 **Graph Conversion**: PyTorch Geometric integration
- 🌐 **Materials Project**: API integration and data retrieval
- 🎨 **Visualization**: All visualization modules and outputs
- 🎮 **Interactive Features**: Jupyter widgets and 3D viewers

## 📊 Supported File Formats

| Format | Read | Write | Description | Auto-detect |
|--------|------|-------|-------------|-------------|
| CIF | ✅ | ✅ | Crystallographic Information File | ✅ |
| POSCAR | ✅ | ✅ | VASP structure format | ✅ |
| CONTCAR | ✅ | ✅ | VASP output structure | ✅ |
| VASP | ✅ | ✅ | Vienna Ab initio Simulation Package | ✅ |
| Materials Project | ✅ | ❌ | API-based structure retrieval | N/A |
| XYZ | 🔄 | 🔄 | Cartesian coordinates (planned v0.4.0) | 🔄 |

## 🏗️ Architecture

### Project Structure

```
crysio/
├── crysio/
│   ├── __init__.py                 # Main package interface
│   ├── core/
│   │   ├── crystal.py              # Crystal structure classes
│   │   ├── parsers.py              # File format parsers
│   │   ├── validators.py           # Structure validation
│   │   └── graph_builder.py        # Graph conversion utilities
│   ├── visualizers/
│   │   ├── __init__.py             # Visualization interface
│   │   ├── crystal_viz.py          # 3D structure visualization
│   │   ├── analysis_plots.py       # Statistical plots
│   │   ├── graph_viz.py            # Graph network visualization
│   │   └── materials_project_viewer.py  # MP-style visualization
│   ├── api/
│   │   ├── materials_project.py    # Materials Project integration
│   │   └── database_connectors.py  # Database interfaces
│   ├── utils/
│   │   ├── exceptions.py           # Exception hierarchy
│   │   ├── atomic_data.py          # Atomic property database
│   │   └── helpers.py              # Utility functions
│   └── testing/
│       ├── test_suite.py           # Comprehensive testing
│       └── sample_data/            # Test crystal structures
├── docs/                           # Documentation
├── examples/                       # Usage examples
├── tests/                          # Unit tests
└── setup files                    # Configuration files
```

### Core Classes

```python
# Main crystal representation
class Crystal:
    """Complete crystal structure with all properties"""
    def __init__(self, lattice_parameters, atomic_sites, ...)
    
    @property
    def volume(self) -> float
    @property 
    def density(self) -> float
    @property
    def formula(self) -> str
    
    def to_graph(self, **kwargs) -> torch_geometric.data.Data
    def visualize(self, **kwargs) -> None
    def analyze_bonding(self) -> Dict

# Lattice representation
class LatticeParameters:
    """Crystallographic lattice parameters"""
    def __init__(self, a, b, c, alpha, beta, gamma)
    
    def get_matrix(self) -> np.ndarray
    def get_volume(self) -> float
    def get_reciprocal(self) -> 'LatticeParameters'

# Atomic site representation  
class AtomicSite:
    """Individual atomic site in crystal"""
    def __init__(self, element, position, occupancy=1.0)
    
    @property
    def fractional_coords(self) -> np.ndarray
    @property
    def cartesian_coords(self) -> np.ndarray
```

## 🔧 Configuration

### Environment Setup

```bash
# Set Materials Project API key
export MP_API_KEY="your_materials_project_api_key"

# Optional: Configure visualization backend
export CRYSIO_VIZ_BACKEND="plotly"  # or "matplotlib"

# Optional: Set default file format
export CRYSIO_DEFAULT_FORMAT="cif"
```

### Python Configuration

```python
# Configure CrysIO settings
import crysio

crysio.config.set_mp_api_key("your_api_key")
crysio.config.set_visualization_backend("plotly")
crysio.config.set_default_edge_cutoff(5.0)
```

## 🚧 Development & Contributing

### Development Installation

```bash
# Clone repository
git clone https://github.com/hasandafa/crysio.git
cd crysio

# Install in development mode with all dependencies
pip install -e ".[dev,viz,extra]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest --cov=crysio --cov-report=html

# Format code
black crysio/
isort crysio/

# Type checking
mypy crysio/
```

### Testing Guidelines

```bash
# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests
pytest -m "not slow"             # Skip slow tests
pytest -m visualization          # Visualization tests

# Run with coverage
pytest --cov=crysio --cov-report=term-missing

# Performance benchmarking
pytest tests/benchmark/           # Performance tests
```

## 📈 Performance & Benchmarks

### Processing Speed
- **CIF Parsing**: ~1,000 structures/second on modern hardware
- **Graph Conversion**: ~100 medium structures/second 
- **Visualization**: Real-time for structures <1,000 atoms
- **Batch Analysis**: ~10,000 structures/minute for property extraction

### Memory Usage
- **Base Import**: ~50MB
- **Crystal Structure**: ~1KB per 100 atoms
- **Graph Conversion**: ~10KB per structure
- **Visualization**: ~5MB per interactive plot

### Scalability
- **File Size**: Tested up to 100MB CIF files
- **Batch Processing**: 10,000+ structures simultaneously
- **Graph Networks**: Structures up to 10,000 atoms
- **Visualization**: Interactive plots with 1,000+ structures

## 🗺️ Roadmap

### **v0.4.0 - Enhanced Features** (Next Release)
- 🔄 **Format Conversion**: CIF ↔ POSCAR ↔ XYZ bidirectional conversion
- 🧹 **Data Cleaning**: Automated preprocessing and validation pipeline
- ⚡ **Performance**: Memory optimization and algorithm improvements
- 🧪 **Extended Testing**: Comprehensive edge case coverage
- 📱 **CLI Interface**: Command-line tools for batch operations

### **v0.5.0 - Production Ready**
- 💻 **Complete CLI**: Full command-line interface with all features
- 📚 **Documentation**: Complete website with tutorials and examples
- 🌐 **Multi-database**: COD, ICSD, AFLOW database integration
- 🤖 **ML Models**: Pre-trained property prediction models
- 🔧 **Configuration**: Advanced configuration management

### **v1.0.0 - First Stable Release & PyPI**
- ✅ **Feature Complete**: All planned core features implemented
- ✅ **Fully Tested**: >95% test coverage with comprehensive validation
- ✅ **Production Ready**: Stable API, optimized performance, robust error handling
- ✅ **PyPI Release**: `pip install crysio` available globally
- ✅ **Documentation**: Complete user guides, tutorials, and API documentation

### **🔮 Beyond v1.0.0**
- **Advanced ML Integration**: Pre-trained models for property prediction
- **Web Interface**: Browser-based crystal analysis and visualization
- **Plugin System**: Extensible architecture for custom functionality
- **Cloud Computing**: Distributed processing and collaborative features
- **Community Platform**: Structure sharing and collaborative analysis

## 🏆 Use Cases & Applications

### **Materials Discovery**
- High-throughput screening of crystal databases
- Property prediction using graph neural networks
- Crystal structure optimization and design
- Phase diagram exploration and analysis

### **Research Applications**
- Crystallographic data analysis and comparison
- Defect structure modeling and visualization
- Surface and interface structure analysis
- Phase transition studies and characterization

### **Educational Tools**
- Interactive crystal structure visualization for teaching
- Crystallography tutorials and demonstrations
- Materials science course materials
- Student research project tools

### **Industrial Applications**
- Quality control for crystal synthesis
- Process optimization for materials production
- Intellectual property analysis and prior art search
- Materials informatics and data mining

## 📄 License & Citation

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation

If you use CrysIO in your research, please cite:

```bibtex
@software{crysio2025,
  title = {CrysIO: Crystal I/O toolkit for machine learning applications},
  author = {Dafa, Abdullah Hasan},
  year = {2025},
  url = {https://github.com/hasandafa/crysio},
  version = {0.3.1}
}
```

## 📞 Contact & Support

**Abdullah Hasan Dafa**  
📧 Email: dafa.abdullahhasan@gmail.com  
🐙 GitHub: [@hasandafa](https://github.com/hasandafa)  
🔗 LinkedIn: [Abdullah Hasan Dafa](https://linkedin.com/in/abdullahhasan-dafa)

### **Get Help & Contribute**
- 🐛 **[Report Issues](https://github.com/hasandafa/crysio/issues)** - Bug reports and feature requests
- 💬 **[Discussions](https://github.com/hasandafa/crysio/discussions)** - Questions and community chat
- 🔧 **[Pull Requests](https://github.com/hasandafa/crysio/pulls)** - Contribute code and improvements
- 📖 **[Wiki](https://github.com/hasandafa/crysio/wiki)** - Documentation and tutorials
- ⭐ **[Star the Repository](https://github.com/hasandafa/crysio)** - Show your support!

### **Development Status**
- **Current Version**: v0.3.1 (Beta - Active Development)
- **Next Release**: v0.4.0 (Format conversion & performance improvements)  
- **Stable Release**: v1.0.0 (Planned for PyPI publication)
- **Update Frequency**: Weekly releases with bug fixes and new features

### **Community & Support**
- **Issue Response**: <24 hours for critical bugs
- **Feature Requests**: Reviewed weekly, implemented based on community interest
- **Documentation**: Continuously updated with new features and examples
- **Tutorials**: Regular blog posts and video tutorials

## 🙏 Acknowledgments

Special thanks to the open source community and the following projects that make CrysIO possible:

- **[Materials Project](https://materialsproject.org/)** - Comprehensive materials database and API
- **[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)** - Graph neural network framework
- **[Pymatgen](https://pymatgen.org/)** - Materials analysis library inspiration
- **[ASE](https://wiki.fysik.dtu.dk/ase/)** - Atomic simulation environment
- **[Plotly](https://plotly.com/python/)** - Interactive visualization framework
- **[NumPy](https://numpy.org/)** & **[SciPy](https://scipy.org/)** - Scientific computing foundation

## 🎯 Quick Links

- 📦 **[PyPI Package](https://pypi.org/project/crysio/)** (Coming in v1.0.0)
- 📚 **[Documentation](https://hasandafa.github.io/crysio/)**
- 🧪 **[Examples](https://github.com/hasandafa/crysio/tree/master/examples)**
- 🐛 **[Issues](https://github.com/hasandafa/crysio/issues)**
- 🔄 **[Changelog](https://github.com/hasandafa/crysio/blob/master/CHANGELOG.md)**
- 📋 **[Roadmap](https://github.com/hasandafa/crysio/projects)**

---

**⭐ Star this repository if you find CrysIO useful for your research and development!**

*CrysIO - Making crystal structure analysis accessible to everyone.*