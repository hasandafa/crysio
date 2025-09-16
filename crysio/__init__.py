"""
Crysio - Crystal I/O Library for Machine Learning
===============================================

A comprehensive Python library for crystal structure I/O, preprocessing, 
and conversion for machine learning applications.
"""

__version__ = "0.3.1"
__author__ = "Abdullah Hasan Dafa"
__email__ = "dafa.abdullahhasan@gmail.com"

# Core imports with robust error handling
try:
    from .core.crystal import Crystal, LatticeParameters, AtomicSite
    from .core.parsers import CIFParser, POSCARParser, auto_detect_format
    from .utils.exceptions import (
        CrysioError, 
        ParsingError, 
        ValidationError, 
        ConversionError
    )
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Core module import warning: {e}")
    # Set to None if import fails - this is what causes the NoneType errors
    Crystal = None
    LatticeParameters = None  
    AtomicSite = None
    CIFParser = None
    POSCARParser = None
    auto_detect_format = None
    CORE_AVAILABLE = False

# High-level API functions
def load_structure(file_path):
    """Load crystal structure from file (CIF, POSCAR, etc.)"""
    if auto_detect_format is None:
        raise ImportError("Core parsers not available - reinstall library")
    
    file_format = auto_detect_format(file_path)
    
    if file_format == 'cif':
        if CIFParser is None:
            raise ImportError("CIF parser not available")
        parser = CIFParser()
        return parser.parse(file_path)
    elif file_format == 'poscar':
        if POSCARParser is None:
            raise ImportError("POSCAR parser not available")
        parser = POSCARParser()
        return parser.parse(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

def create_crystal(lattice_params, atomic_sites):
    """Create Crystal object from lattice parameters and atomic sites"""
    if Crystal is None:
        raise ImportError("Crystal class not available - reinstall library")
    return Crystal(lattice_params, atomic_sites)

# Validation functions
try:
    from .core.validators import validate_structure
    VALIDATION_AVAILABLE = True
except ImportError:
    def validate_structure(*args, **kwargs):
        raise ImportError("Validation module not available - install dependencies")
    VALIDATION_AVAILABLE = False

# Graph conversion functions  
try:
    from .converters.graph_builder import to_graph
    GRAPH_CONVERSION_ENABLED = True
except ImportError:
    def to_graph(*args, **kwargs):
        raise ImportError("Graph conversion requires torch and torch-geometric")
    GRAPH_CONVERSION_ENABLED = False

# Materials Project API
try:
    from .api.materials_project import (
        MaterialsProjectAPI,
        search_materials_database, 
        load_from_materials_project
    )
    MATERIALS_PROJECT_ENABLED = True
except ImportError:
    MaterialsProjectAPI = None
    def search_materials_database(*args, **kwargs):
        raise ImportError("Materials Project API requires mp-api package")
    def load_from_materials_project(*args, **kwargs):
        raise ImportError("Materials Project API requires mp-api package")
    MATERIALS_PROJECT_ENABLED = False

# FIXED: Enhanced Visualization functions with all new features
try:
    # Basic visualization functions
    from .visualizers import (
        plot_crystal_3d,
        plot_unit_cell_2d, 
        plot_lattice_parameters,
        plot_atomic_positions,
        CrystalVisualizer
    )
    
    # Materials Project-style visualization (NEW)
    from .visualizers import (
        MaterialsProjectViewer,
        plot_mp_style,
        show_mp_interactive
    )
    
    # Analysis and statistical plots
    from .visualizers import (
        plot_property_distribution,
        plot_correlation_matrix,
        plot_structure_comparison,
        plot_formation_energy,
        plot_property_scatter,
        AnalysisVisualizer,
        extract_crystal_properties
    )
    
    # Graph visualization
    from .visualizers import (
        plot_graph_network,
        plot_adjacency_matrix,
        plot_node_properties,
        plot_graph_metrics,
        analyze_graph_connectivity,
        GraphVisualizer
    )
    
    # Convenience functions
    from .visualizers import (
        visualize_crystal_mp_style,
        create_quick_viewer,
        analyze_multiple_crystals
    )
    
    VISUALIZATION_ENABLED = True
    
except ImportError as e:
    print(f"⚠️  Visualization import warning: {e}")
    
    # Basic fallback functions
    def plot_crystal_3d(*args, **kwargs):
        raise ImportError("Visualization requires matplotlib and plotly")
    def plot_unit_cell_2d(*args, **kwargs):
        raise ImportError("Visualization requires matplotlib")
    def plot_lattice_parameters(*args, **kwargs):
        raise ImportError("Visualization requires matplotlib")
    def plot_atomic_positions(*args, **kwargs):
        raise ImportError("Visualization requires matplotlib")
    
    # MP-style fallback functions
    def MaterialsProjectViewer(*args, **kwargs):
        raise ImportError("MaterialsProjectViewer requires plotly>=5.0")
    def plot_mp_style(*args, **kwargs):
        raise ImportError("MP-style visualization requires plotly>=5.0")
    def show_mp_interactive(*args, **kwargs):
        raise ImportError("Interactive visualization requires plotly>=5.0")
    
    # Analysis fallback functions
    def plot_property_distribution(*args, **kwargs):
        raise ImportError("Analysis plots require matplotlib and seaborn")
    def plot_correlation_matrix(*args, **kwargs):
        raise ImportError("Analysis plots require matplotlib and seaborn")
    def plot_structure_comparison(*args, **kwargs):
        raise ImportError("Analysis plots require matplotlib")
    def plot_formation_energy(*args, **kwargs):
        raise ImportError("Analysis plots require matplotlib")
    def plot_property_scatter(*args, **kwargs):
        raise ImportError("Analysis plots require matplotlib")
    
    # Graph visualization fallback functions
    def plot_graph_network(*args, **kwargs):
        raise ImportError("Graph visualization requires networkx and matplotlib")
    def plot_adjacency_matrix(*args, **kwargs):
        raise ImportError("Graph visualization requires networkx and matplotlib")
    def plot_node_properties(*args, **kwargs):
        raise ImportError("Graph visualization requires networkx and matplotlib")
    def plot_graph_metrics(*args, **kwargs):
        raise ImportError("Graph visualization requires networkx and matplotlib")
    def analyze_graph_connectivity(*args, **kwargs):
        raise ImportError("Graph analysis requires networkx")
    
    # Convenience fallback functions
    def visualize_crystal_mp_style(*args, **kwargs):
        raise ImportError("MP-style visualization requires plotly>=5.0")
    def create_quick_viewer(*args, **kwargs):
        raise ImportError("Visualization requires matplotlib or plotly")
    def analyze_multiple_crystals(*args, **kwargs):
        raise ImportError("Analysis requires matplotlib")
    def extract_crystal_properties(*args, **kwargs):
        raise ImportError("Analysis requires numpy and pandas")
    
    # Set classes to None
    CrystalVisualizer = None
    AnalysisVisualizer = None
    GraphVisualizer = None
    
    VISUALIZATION_ENABLED = False

# Batch processing
def batch_process(file_list, operations=None):
    """Process multiple crystal structure files"""
    if operations is None:
        operations = ['clean']
    
    results = []
    for file_path in file_list:
        try:
            structure = load_structure(file_path)
            
            # Apply operations
            if 'clean' in operations:
                # Add cleaning logic here
                pass
            if 'validate' in operations:
                validate_structure(structure)
                
            results.append(structure)
        except Exception as e:
            print(f"⚠️  Failed to process {file_path}: {e}")
            continue
    
    return results

# ENHANCED: Module status checking
def check_installation():
    """Check which features are available"""
    status = {
        'core': CORE_AVAILABLE,
        'validation': VALIDATION_AVAILABLE,
        'graph_conversion': GRAPH_CONVERSION_ENABLED,
        'materials_project': MATERIALS_PROJECT_ENABLED,
        'visualization': VISUALIZATION_ENABLED
    }
    
    print("🔍 CRYSIO INSTALLATION STATUS")
    print("=" * 40)
    for feature, available in status.items():
        icon = "✅" if available else "❌" 
        print(f"{icon} {feature.replace('_', ' ').title()}")
    
    # Additional visualization details
    if VISUALIZATION_ENABLED:
        print("\n🎨 VISUALIZATION FEATURES:")
        try:
            from .visualizers import get_available_visualizers
            vis_status = get_available_visualizers()
            for module, available in vis_status.items():
                icon = "✅" if available else "❌"
                print(f"  {icon} {module.replace('_', ' ').title()}")
        except:
            print("  📊 Basic visualization available")
    
    return status

# ENHANCED: Dependency checking
def check_dependencies():
    """Check all dependencies and provide installation guidance"""
    dependencies = {}
    
    # Core dependencies
    core_deps = ['numpy', 'scipy']
    for dep in core_deps:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            dependencies[dep] = {'available': True, 'version': version}
        except ImportError:
            dependencies[dep] = {'available': False, 'install': f'pip install {dep}'}
    
    # Visualization dependencies
    vis_deps = {
        'matplotlib': 'pip install matplotlib',
        'plotly': 'pip install plotly>=5.0',
        'seaborn': 'pip install seaborn',
        'networkx': 'pip install networkx',
        'ipywidgets': 'pip install ipywidgets>=8.0'
    }
    
    for dep, install_cmd in vis_deps.items():
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            dependencies[dep] = {'available': True, 'version': version}
        except ImportError:
            dependencies[dep] = {'available': False, 'install': install_cmd}
    
    # Graph conversion dependencies
    graph_deps = ['torch', 'torch_geometric']
    for dep in graph_deps:
        try:
            if dep == 'torch_geometric':
                import torch_geometric
                module = torch_geometric
            else:
                module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            dependencies[dep] = {'available': True, 'version': version}
        except ImportError:
            if dep == 'torch':
                dependencies[dep] = {'available': False, 'install': 'pip install torch'}
            else:
                dependencies[dep] = {'available': False, 'install': 'pip install torch-geometric'}
    
    # API dependencies
    try:
        from mp_api.client import MPRester
        dependencies['mp_api'] = {'available': True, 'version': 'installed'}
    except ImportError:
        dependencies['mp_api'] = {'available': False, 'install': 'pip install mp-api'}
    
    return dependencies

def print_dependency_status():
    """Print comprehensive dependency status"""
    deps = check_dependencies()
    
    print("\n📦 DEPENDENCY STATUS:")
    print("=" * 25)
    
    categories = {
        'Core': ['numpy', 'scipy'],
        'Visualization': ['matplotlib', 'plotly', 'seaborn', 'ipywidgets'],
        'Graph Analysis': ['networkx', 'torch', 'torch_geometric'],
        'APIs': ['mp_api']
    }
    
    for category, dep_list in categories.items():
        print(f"\n{category}:")
        for dep in dep_list:
            if dep in deps:
                info = deps[dep]
                if info['available']:
                    version = info.get('version', 'unknown')
                    print(f"  ✅ {dep}: v{version}")
                else:
                    install_cmd = info.get('install', f'pip install {dep}')
                    print(f"  ❌ {dep}: {install_cmd}")

# UPDATED: Export main API with all new functions
__all__ = [
    # Core classes
    'Crystal', 'LatticeParameters', 'AtomicSite',
    
    # Parsers  
    'CIFParser', 'POSCARParser', 'auto_detect_format',
    
    # High-level API
    'load_structure', 'create_crystal', 'batch_process',
    
    # Validation
    'validate_structure',
    
    # Graph conversion
    'to_graph',
    
    # Materials Project API
    'MaterialsProjectAPI', 'search_materials_database', 'load_from_materials_project',
    
    # Basic Visualization
    'plot_crystal_3d', 'plot_unit_cell_2d', 'plot_lattice_parameters', 'plot_atomic_positions',
    'CrystalVisualizer',
    
    # Materials Project-style Visualization (NEW)
    'MaterialsProjectViewer', 'plot_mp_style', 'show_mp_interactive',
    
    # Analysis and Statistical Plots (NEW)
    'plot_property_distribution', 'plot_correlation_matrix', 'plot_structure_comparison',
    'plot_formation_energy', 'plot_property_scatter', 'AnalysisVisualizer', 'extract_crystal_properties',
    
    # Graph Visualization (NEW)
    'plot_graph_network', 'plot_adjacency_matrix', 'plot_node_properties', 
    'plot_graph_metrics', 'analyze_graph_connectivity', 'GraphVisualizer',
    
    # Convenience Functions (NEW)
    'visualize_crystal_mp_style', 'create_quick_viewer', 'analyze_multiple_crystals',
    
    # Exceptions
    'CrysioError', 'ParsingError', 'ValidationError', 'ConversionError',
    
    # Utilities
    'check_installation', 'check_dependencies', 'print_dependency_status'
]

# Availability flags for external checking
CORE_AVAILABLE = CORE_AVAILABLE if 'CORE_AVAILABLE' in locals() else False
VALIDATION_AVAILABLE = VALIDATION_AVAILABLE if 'VALIDATION_AVAILABLE' in locals() else False