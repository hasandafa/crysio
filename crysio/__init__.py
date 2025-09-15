"""
Crysio: Crystal I/O and Visualization Library

A comprehensive toolkit for crystal structure processing, analysis, and visualization,
designed for materials science research and machine learning applications.
"""

# Version info
__version__ = "0.3.0"
__author__ = "Abdullah Hasan Dafa"
__email__ = "your.email@example.com"
__license__ = "MIT"

# Core imports with error handling
try:
    from .core.crystal import Crystal, LatticeParameters, AtomicSite
    from .core.parsers import parse_cif, parse_poscar, auto_detect_format
    
    # Create convenience functions
    def load_crystal(filepath: str) -> Crystal:
        """
        Load crystal structure from file with auto-detection.
        
        Args:
            filepath: Path to crystal structure file
            
        Returns:
            Crystal: Loaded crystal structure
        """
        return auto_detect_format(filepath)
    
    # Export main loading function
    load_structure = load_crystal  # Alias for compatibility
    
except ImportError as e:
    import warnings
    warnings.warn(f"Core crystal functionality not available: {e}")
    Crystal = None
    LatticeParameters = None
    AtomicSite = None
    load_crystal = None
    load_structure = None

# Validators
try:
    from .core.validators import validate_crystal, CrystalValidator
except ImportError:
    validate_crystal = None
    CrystalValidator = None

# Graph conversion
try:
    from .converters.graph_builder import to_graph, GraphBuilder
except ImportError:
    to_graph = None
    GraphBuilder = None

# Materials Project API
try:
    from .api.materials_project import MaterialsProjectAPI, search_materials_database, load_from_materials_project
except ImportError:
    MaterialsProjectAPI = None
    search_materials_database = None
    load_from_materials_project = None

# Visualization (new in v0.3.0)
try:
    from . import visualizers
    # Import main visualization functions
    from .visualizers import (
        plot_unit_cell_2d, plot_crystal_3d, plot_lattice_parameters,
        plot_property_distribution, plot_correlation_matrix,
        plot_graph_network, plot_adjacency_matrix
    )
except ImportError:
    import warnings
    warnings.warn("Visualization module not available. Install dependencies: pip install matplotlib plotly")
    visualizers = None

# Exception classes
try:
    from .utils.exceptions import (
        CrysioError, ParseError, ValidationError, 
        VisualizationError, DependencyError, APIError
    )
except ImportError:
    # Define minimal exception classes if utils not available
    class CrysioError(Exception):
        """Base exception for Crysio library."""
        pass
    
    class ParseError(CrysioError):
        """Exception raised for parsing errors."""
        pass
    
    class ValidationError(CrysioError):
        """Exception raised for validation errors."""
        pass

# Define what gets exported when using "from crysio import *"
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',
    
    # Core classes
    'Crystal',
    'LatticeParameters', 
    'AtomicSite',
    
    # Loading functions
    'load_crystal',
    'load_structure',  # Compatibility alias
    'parse_cif',
    'parse_poscar',
    
    # Validation
    'validate_crystal',
    'CrystalValidator',
    
    # Graph conversion
    'to_graph',
    'GraphBuilder',
    
    # Materials Project API
    'MaterialsProjectAPI',
    'search_materials_database',
    'load_from_materials_project',
    
    # Visualization module
    'visualizers',
    
    # Main visualization functions
    'plot_unit_cell_2d',
    'plot_crystal_3d', 
    'plot_lattice_parameters',
    'plot_property_distribution',
    'plot_correlation_matrix',
    'plot_graph_network',
    'plot_adjacency_matrix',
    
    # Exceptions
    'CrysioError',
    'ParseError', 
    'ValidationError',
    'VisualizationError',
    'DependencyError',
    'APIError'
]

# Remove None values from __all__
__all__ = [name for name in __all__ if globals().get(name) is not None]


def get_info():
    """Get information about Crysio installation and available features."""
    info = {
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'features': {}
    }
    
    # Check core features
    info['features']['core'] = Crystal is not None
    info['features']['validation'] = validate_crystal is not None
    info['features']['graph_conversion'] = to_graph is not None
    info['features']['materials_project_api'] = MaterialsProjectAPI is not None
    info['features']['visualization'] = visualizers is not None
    
    # Check visualization sub-features
    if visualizers is not None:
        try:
            availability = visualizers.get_available_visualizers()
            info['features']['visualization_modules'] = availability
        except:
            info['features']['visualization_modules'] = {}
    
    return info


def check_installation():
    """Check Crysio installation and print status report."""
    print(f"Crysio v{__version__} Installation Status")
    print("=" * 40)
    
    info = get_info()
    
    for feature, available in info['features'].items():
        if isinstance(available, dict):
            print(f"\n{feature.replace('_', ' ').title()}:")
            for subfeature, sub_available in available.items():
                status = "✅" if sub_available else "❌"
                print(f"  {status} {subfeature}")
        else:
            status = "✅" if available else "❌"
            print(f"{status} {feature.replace('_', ' ').title()}")
    
    # Installation recommendations
    missing_features = [k for k, v in info['features'].items() 
                       if isinstance(v, bool) and not v]
    
    if missing_features:
        print(f"\n💡 To enable missing features:")
        if 'visualization' in missing_features:
            print("   pip install matplotlib plotly seaborn networkx")
        if 'materials_project_api' in missing_features:
            print("   pip install requests")
        if 'graph_conversion' in missing_features:
            print("   pip install torch torch-geometric")
    else:
        print(f"\n🎉 All features available!")