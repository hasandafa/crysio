"""
Crysio: Crystal I/O and Visualization Library

A comprehensive toolkit for crystal structure processing, analysis, and visualization,
designed for materials science research and machine learning applications.
"""

# Version info
__version__ = "0.3.0"
__author__ = "Abdullah Hasan Dafa"
__email__ = "dafa.abdullahhasan@gmail.com"

# Core functionality imports
from .core.crystal import Crystal, LatticeParameters, AtomicSite
from .core.parsers import CIFParser, POSCARParser, auto_detect_format, load_structure
from .core.validators import validate_structure, CrystalValidator

# Conversion utilities
from .converters.graph_builder import to_graph, GraphBuilder

# Visualization functionality (with graceful fallbacks)
try:
    from .visualizers import (
        plot_unit_cell_2d,
        plot_crystal_3d,
        plot_lattice_parameters,
        plot_atomic_positions,
        plot_property_distribution,
        plot_correlation_matrix,
        plot_structure_comparison,
        CrystalVisualizer,
        AnalysisVisualizer,
        check_visualization_dependencies,
        VISUALIZATION_AVAILABLE
    )
    VISUALIZATION_ENABLED = True
except ImportError:
    # Graceful fallback when visualization dependencies missing
    VISUALIZATION_ENABLED = False
    VISUALIZATION_AVAILABLE = False
    
    # Create placeholder functions that provide helpful error messages
    def _visualization_not_available(*args, **kwargs):
        raise ImportError(
            "Visualization features not available. Install with: "
            "pip install matplotlib plotly seaborn pandas"
        )
    
    plot_unit_cell_2d = _visualization_not_available
    plot_crystal_3d = _visualization_not_available
    plot_lattice_parameters = _visualization_not_available
    plot_atomic_positions = _visualization_not_available
    plot_property_distribution = _visualization_not_available
    plot_correlation_matrix = _visualization_not_available
    plot_structure_comparison = _visualization_not_available
    CrystalVisualizer = _visualization_not_available
    AnalysisVisualizer = _visualization_not_available
    
    def check_visualization_dependencies():
        print("❌ Visualization dependencies not available")
        print("Install with: pip install matplotlib plotly seaborn pandas")
        return False

# Graph visualization functionality (with graceful fallbacks)
try:
    from .visualizers.graph_viz import (
        plot_graph_network,
        plot_adjacency_matrix,
        plot_3d_graph_overlay,
        plot_graph_metrics,
        GraphVisualizer
    )
    GRAPH_VISUALIZATION_ENABLED = True
except ImportError:
    GRAPH_VISUALIZATION_ENABLED = False
    
    def _graph_visualization_not_available(*args, **kwargs):
        raise ImportError(
            "Graph visualization not available. Install with: "
            "pip install networkx matplotlib plotly"
        )
    
    plot_graph_network = _graph_visualization_not_available
    plot_adjacency_matrix = _graph_visualization_not_available
    plot_3d_graph_overlay = _graph_visualization_not_available
    plot_graph_metrics = _graph_visualization_not_available
    GraphVisualizer = _graph_visualization_not_available

# Materials Project API integration (with graceful fallbacks)
try:
    from .api.materials_project import (
        MaterialsProjectAPI,
        search_materials_database,
        load_from_materials_project
    )
    MATERIALS_PROJECT_ENABLED = True
except ImportError:
    MATERIALS_PROJECT_ENABLED = False
    MaterialsProjectAPI = None
    
    def search_materials_database(*args, **kwargs):
        raise ImportError(
            "Materials Project API not available. Install with: "
            "pip install mp-api"
        )
    
    def load_from_materials_project(*args, **kwargs):
        raise ImportError(
            "Materials Project API not available. Install with: "
            "pip install mp-api"
        )

# Exception classes
from .utils.exceptions import (
    CrysioError,
    ParsingError,
    ValidationError,
    APIError,
    ConfigurationError,
    ConversionError,
    GraphBuildingError,
    VisualizationError,
    DependencyError,
    GeometryError
)

def check_dependencies():
    """
    Check availability of optional dependencies and provide installation guidance.
    
    Returns:
        dict: Status of each dependency group
    """
    status = {
        'core': True,  # Core functionality always available
        'visualization': VISUALIZATION_AVAILABLE,
        'graph_visualization': GRAPH_VISUALIZATION_ENABLED,
        'materials_project': MATERIALS_PROJECT_ENABLED,
        'graph_processing': True  # Checked in graph_builder module
    }
    
    print("=== CRYSIO DEPENDENCY STATUS ===")
    print(f"Version: {__version__}")
    print()
    
    # Core functionality
    print("✅ Core functionality: Available")
    print("   - Crystal structure representation")
    print("   - File parsing (CIF, POSCAR)")
    print("   - Structure validation")
    print()
    
    # Graph processing
    try:
        import torch
        import torch_geometric
        print("✅ Graph processing: Available")
        print("   - PyTorch Geometric integration")
        print("   - Crystal-to-graph conversion")
        status['graph_processing'] = True
    except ImportError:
        print("⚠️  Graph processing: Partially available")
        print("   - Install PyTorch and PyTorch Geometric for full functionality")
        print("   - pip install torch torch-geometric")
        status['graph_processing'] = False
    print()
    
    # Visualization
    if status['visualization']:
        print("✅ Visualization: Available") 
        print("   - 2D/3D crystal structure plots")
        print("   - Statistical analysis charts")
        print("   - Interactive visualizations")
    else:
        print("❌ Visualization: Not available")
        print("   - Install with: pip install matplotlib plotly seaborn pandas")
    print()
    
    # Graph visualization
    if status['graph_visualization']:
        print("✅ Graph visualization: Available")
        print("   - Network topology plots")
        print("   - Adjacency matrix heatmaps")
        print("   - 3D graph overlays")
        print("   - Graph metrics analysis")
    else:
        print("❌ Graph visualization: Not available")
        print("   - Install with: pip install networkx matplotlib plotly")
    print()
    
    # Materials Project API
    if status['materials_project']:
        print("✅ Materials Project API: Available")
        print("   - Database search and download")
        print("   - Property analysis")
    else:
        print("❌ Materials Project API: Not available")
        print("   - Install with: pip install mp-api")
    print()
    
    # Overall status
    available_features = sum(status.values())
    total_features = len(status)
    print(f"📊 Overall: {available_features}/{total_features} feature groups available")
    
    if available_features == total_features:
        print("🎉 All features are available!")
    elif available_features >= 3:
        print("✅ Most features ready. Install optional dependencies for full capabilities.")
    else:
        print("⚠️  Limited functionality. Consider installing additional dependencies.")
    
    return status


def create_example_crystal():
    """
    Create an example crystal structure for testing and demonstrations.
    
    Returns:
        Crystal: Simple cubic silicon carbide structure
    """
    # Simple SiC structure for demonstration
    lattice = LatticeParameters(
        a=4.36, b=4.36, c=4.36,
        alpha=90.0, beta=90.0, gamma=90.0,
        crystal_system="cubic"
    )
    
    sites = [
        AtomicSite(element="Si", x=0.0, y=0.0, z=0.0),
        AtomicSite(element="C", x=0.25, y=0.25, z=0.25),
        AtomicSite(element="Si", x=0.5, y=0.5, z=0.0),
        AtomicSite(element="C", x=0.75, y=0.75, z=0.25),
        AtomicSite(element="Si", x=0.5, y=0.0, z=0.5),
        AtomicSite(element="C", x=0.75, y=0.25, z=0.75),
        AtomicSite(element="Si", x=0.0, y=0.5, z=0.5),
        AtomicSite(element="C", x=0.25, y=0.75, z=0.75)
    ]
    
    return Crystal(lattice_parameters=lattice, atomic_sites=sites)


# Export main classes and functions
__all__ = [
    # Version info
    '__version__',
    
    # Core classes
    'Crystal',
    'LatticeParameters', 
    'AtomicSite',
    
    # Parsing functions
    'CIFParser',
    'POSCARParser',
    'auto_detect_format',
    'load_structure',
    
    # Validation
    'validate_structure',
    'CrystalValidator',
    
    # Graph conversion
    'to_graph',
    'GraphBuilder',
    
    # Visualization (if available)
    'plot_unit_cell_2d',
    'plot_crystal_3d',
    'plot_lattice_parameters', 
    'plot_atomic_positions',
    'plot_property_distribution',
    'plot_correlation_matrix',
    'plot_structure_comparison',
    'CrystalVisualizer',
    'AnalysisVisualizer',
    'check_visualization_dependencies',
    
    # Graph visualization (if available)
    'plot_graph_network',
    'plot_adjacency_matrix',
    'plot_3d_graph_overlay',
    'plot_graph_metrics',
    'GraphVisualizer',
    
    # Materials Project API (if available)
    'MaterialsProjectAPI',
    'search_materials_database',
    'load_from_materials_project',
    
    # Utilities
    'check_dependencies',
    'create_example_crystal',
    
    # Exception classes
    'CrysioError',
    'ParsingError',
    'ValidationError',
    'APIError',
    'ConfigurationError',
    'ConversionError',
    'GraphBuildingError',
    'VisualizationError',
    'DependencyError',
    'GeometryError',
    
    # Feature flags
    'VISUALIZATION_ENABLED',
    'GRAPH_VISUALIZATION_ENABLED',
    'MATERIALS_PROJECT_ENABLED'
]