"""
Crysio Visualization Module v0.3.1 - FIXED VERSION

Enhanced visualization capabilities with Materials Project-style interactive viewers,
building upon the solid foundation of v0.3.0 visualization system.
Compatible with actual Crysio Crystal class structure.

New in v0.3.1:
- MaterialsProjectViewer: MP-style interactive crystal visualization
- Enhanced Plotly integration with real-time controls
- Property panels and export capabilities
- Improved color schemes and styling
- Fixed compatibility with actual Crystal class structure
"""

# Version info
__version__ = "0.3.1"

# FIXED: Import core visualization functions with graceful fallbacks
try:
    from .crystal_viz import (
        plot_unit_cell_2d,
        plot_crystal_3d,
        plot_lattice_parameters,
        plot_atomic_positions,
        CrystalVisualizer
    )
    CRYSTAL_VIZ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: crystal_viz import failed: {e}")
    CRYSTAL_VIZ_AVAILABLE = False

try:
    from .analysis_plots import (
        plot_property_distribution,
        plot_correlation_matrix,
        plot_structure_comparison,
        plot_formation_energy,
        plot_property_scatter,
        AnalysisVisualizer,
        extract_crystal_properties  # ADDED: Helper function for Crystal objects
    )
    ANALYSIS_PLOTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: analysis_plots import failed: {e}")
    ANALYSIS_PLOTS_AVAILABLE = False

try:
    from .graph_viz import (
        plot_graph_network,
        plot_adjacency_matrix,
        plot_node_properties,
        plot_graph_metrics,
        analyze_graph_connectivity,
        GraphVisualizer
    )
    GRAPH_VIZ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: graph_viz import failed: {e}")
    GRAPH_VIZ_AVAILABLE = False

# FIXED: Materials Project-style visualization with proper error handling
try:
    from .materials_project_viewer import (
        MaterialsProjectViewer,
        plot_mp_style,
        show_mp_interactive
    )
    MP_VIEWER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: materials_project_viewer import failed: {e}")
    MP_VIEWER_AVAILABLE = False

# Build dynamic __all__ list based on availability
__all__ = []

# v0.3.0 functions
if CRYSTAL_VIZ_AVAILABLE:
    __all__.extend([
        'plot_unit_cell_2d',
        'plot_crystal_3d', 
        'plot_lattice_parameters',
        'plot_atomic_positions',
        'CrystalVisualizer'
    ])

if ANALYSIS_PLOTS_AVAILABLE:
    __all__.extend([
        'plot_property_distribution',
        'plot_correlation_matrix',
        'plot_structure_comparison',
        'plot_formation_energy',
        'plot_property_scatter',
        'AnalysisVisualizer',
        'extract_crystal_properties'
    ])

if GRAPH_VIZ_AVAILABLE:
    __all__.extend([
        'plot_graph_network',
        'plot_adjacency_matrix',
        'plot_node_properties', 
        'plot_graph_metrics',
        'analyze_graph_connectivity',
        'GraphVisualizer'
    ])

# v0.3.1 functions
if MP_VIEWER_AVAILABLE:
    __all__.extend([
        'MaterialsProjectViewer',
        'plot_mp_style',
        'show_mp_interactive'
    ])

# Availability flags
__all__.extend([
    'CRYSTAL_VIZ_AVAILABLE',
    'ANALYSIS_PLOTS_AVAILABLE', 
    'GRAPH_VIZ_AVAILABLE',
    'MP_VIEWER_AVAILABLE'
])


def get_available_visualizers():
    """
    Get information about available visualization modules.
    
    Returns:
        dict: Dictionary with availability status of each visualization module
    """
    return {
        'crystal_viz': CRYSTAL_VIZ_AVAILABLE,
        'analysis_plots': ANALYSIS_PLOTS_AVAILABLE,
        'graph_viz': GRAPH_VIZ_AVAILABLE,
        'materials_project_viewer': MP_VIEWER_AVAILABLE
    }


def check_dependencies():
    """
    Check visualization dependencies and provide installation guidance.
    
    Returns:
        dict: Status of required and optional dependencies
    """
    dependencies = {}
    
    # Core dependencies
    try:
        import matplotlib
        dependencies['matplotlib'] = {'available': True, 'version': matplotlib.__version__}
    except ImportError:
        dependencies['matplotlib'] = {'available': False, 'install': 'pip install matplotlib'}
    
    try:
        import plotly
        dependencies['plotly'] = {'available': True, 'version': plotly.__version__}
    except ImportError:
        dependencies['plotly'] = {'available': False, 'install': 'pip install plotly>=5.0'}
    
    # Enhanced dependencies for v0.3.1
    try:
        import ipywidgets
        dependencies['ipywidgets'] = {'available': True, 'version': ipywidgets.__version__}
    except ImportError:
        dependencies['ipywidgets'] = {'available': False, 'install': 'pip install ipywidgets>=8.0'}
    
    # Optional dependencies
    try:
        import seaborn
        dependencies['seaborn'] = {'available': True, 'version': seaborn.__version__}
    except ImportError:
        dependencies['seaborn'] = {'available': False, 'install': 'pip install seaborn'}
    
    try:
        import networkx
        dependencies['networkx'] = {'available': True, 'version': networkx.__version__}
    except ImportError:
        dependencies['networkx'] = {'available': False, 'install': 'pip install networkx'}
    
    # Graph analysis dependencies
    try:
        import torch
        dependencies['torch'] = {'available': True, 'version': torch.__version__}
    except ImportError:
        dependencies['torch'] = {'available': False, 'install': 'pip install torch'}
    
    try:
        import torch_geometric
        dependencies['torch_geometric'] = {'available': True, 'version': torch_geometric.__version__}
    except ImportError:
        dependencies['torch_geometric'] = {'available': False, 'install': 'pip install torch-geometric'}
    
    return dependencies


def print_status():
    """Print comprehensive status of visualization system."""
    print("Crysio Visualization System v0.3.1 (FIXED)")
    print("=" * 50)
    
    # Module availability
    print("\nVisualization Modules:")
    availability = get_available_visualizers()
    for module, available in availability.items():
        status = "✅" if available else "❌"
        print(f"  {status} {module.replace('_', ' ').title()}")
    
    # Dependency status
    print("\nDependency Status:")
    deps = check_dependencies()
    for dep, info in deps.items():
        if info['available']:
            version = info.get('version', 'unknown')
            print(f"  ✅ {dep}: v{version}")
        else:
            install_cmd = info.get('install', 'unknown')
            print(f"  ❌ {dep}: {install_cmd}")
    
    # Feature recommendations
    print("\nRecommended Setup for Full Features:")
    if not deps.get('plotly', {}).get('available'):
        print("  📦 pip install plotly>=5.0  # For interactive MP-style visualization")
    if not deps.get('ipywidgets', {}).get('available'):
        print("  📦 pip install ipywidgets>=8.0  # For Jupyter controls")
    
    # Usage examples
    if MP_VIEWER_AVAILABLE:
        print("\n🎯 Quick Start (v0.3.1):")
        print("  import crysio")
        print("  crystal = crysio.load_structure('structure.cif')")
        print("  crysio.visualizers.show_mp_interactive(crystal)")
    
    # Compatibility note
    print("\n🔧 Compatibility:")
    print("  ✅ Fixed Crystal class compatibility")
    print("  ✅ Proper lattice_parameters access")
    print("  ✅ Corrected import paths")
    print("  ✅ Site.position coordinate handling")


def visualize_crystal_mp_style(crystal, **kwargs):
    """
    Visualize crystal structure in Materials Project style.
    
    This is a convenience function that automatically chooses the best
    available visualization method based on the environment.
    
    Args:
        crystal: Crystal structure to visualize
        **kwargs: Additional visualization parameters
        
    Examples:
        >>> import crysio
        >>> crystal = crysio.load_structure("structure.cif")
        >>> crysio.visualizers.visualize_crystal_mp_style(crystal)
    """
    if MP_VIEWER_AVAILABLE:
        # Use new MP-style viewer (v0.3.1)
        show_mp_interactive(crystal, **kwargs)
    elif CRYSTAL_VIZ_AVAILABLE:
        # Fallback to basic 3D visualization (v0.3.0)
        import warnings
        warnings.warn("MaterialsProjectViewer not available. Using basic 3D visualization.")
        plot_crystal_3d(crystal, **kwargs)
    else:
        raise ImportError(
            "No visualization methods available. "
            "Install dependencies: pip install plotly matplotlib"
        )


def create_quick_viewer(crystal):
    """
    Create the best available viewer for the crystal structure.
    
    Args:
        crystal: Crystal structure to visualize
        
    Returns:
        Viewer object (MaterialsProjectViewer if available, else CrystalVisualizer)
        
    Examples:
        >>> viewer = crysio.visualizers.create_quick_viewer(crystal)
        >>> viewer.show_interactive()
    """
    if MP_VIEWER_AVAILABLE:
        return MaterialsProjectViewer(crystal)
    elif CRYSTAL_VIZ_AVAILABLE:
        return CrystalVisualizer(crystal)
    else:
        raise ImportError("No viewer classes available")


def upgrade_to_mp_style(existing_plot_call):
    """
    Helper function to upgrade existing v0.3.0 plotting calls to v0.3.1 MP-style.
    
    Args:
        existing_plot_call: String representation of existing plot call
        
    Returns:
        String with suggested v0.3.1 equivalent
        
    Examples:
        >>> upgrade_to_mp_style("plot_crystal_3d(crystal)")
        'show_mp_interactive(crystal)  # Enhanced MP-style with controls'
    """
    upgrades = {
        'plot_crystal_3d': 'show_mp_interactive',
        'plot_unit_cell_2d': 'MaterialsProjectViewer(crystal).show_interactive()',
        'CrystalVisualizer': 'MaterialsProjectViewer'
    }
    
    for old, new in upgrades.items():
        if old in existing_plot_call:
            return existing_plot_call.replace(old, new) + "  # Enhanced v0.3.1 MP-style"
    
    return existing_plot_call + "  # No direct v0.3.1 equivalent"


# ADDED: Convenience function for crystal structure analysis
def analyze_multiple_crystals(crystals, labels=None, properties=None):
    """
    Analyze multiple crystal structures and generate comparative visualizations.
    
    Args:
        crystals: List of Crystal objects
        labels: Labels for each crystal (optional)
        properties: List of properties to analyze (optional)
        
    Returns:
        dict: Analysis results and generated plots
        
    Examples:
        >>> crystals = [crystal1, crystal2, crystal3]
        >>> results = crysio.visualizers.analyze_multiple_crystals(crystals)
    """
    if not ANALYSIS_PLOTS_AVAILABLE:
        raise ImportError("Analysis plots not available. Install matplotlib and seaborn.")
    
    # Extract properties for analysis
    if properties is None:
        properties = ['volume', 'density', 'a', 'b', 'c']
    
    data = extract_crystal_properties(crystals, properties)
    
    results = {
        'data': data,
        'crystals': crystals,
        'labels': labels or [f'Crystal {i+1}' for i in range(len(crystals))],
        'plots': {}
    }
    
    # Generate comparison plots
    try:
        for prop in properties:
            if prop in data and len(data[prop]) > 1:
                fig = plot_structure_comparison(crystals, prop, labels)
                results['plots'][f'{prop}_comparison'] = fig
    except Exception as e:
        print(f"Warning: Could not generate comparison plots: {e}")
    
    # Generate correlation matrix if enough properties
    try:
        if len([p for p in properties if p in data]) >= 2:
            fig = plot_correlation_matrix(data)
            results['plots']['correlation_matrix'] = fig
    except Exception as e:
        print(f"Warning: Could not generate correlation matrix: {e}")
    
    return results


# Show import warnings for missing modules
if not MP_VIEWER_AVAILABLE and not CRYSTAL_VIZ_AVAILABLE:
    import warnings
    warnings.warn(
        "No visualization modules available. Install dependencies:\n"
        "pip install plotly matplotlib  # For basic visualization\n"
        "pip install plotly ipywidgets  # For full MP-style experience"
    )
elif not MP_VIEWER_AVAILABLE:
    import warnings
    warnings.warn(
        "MaterialsProjectViewer not available. Install for enhanced features:\n"
        "pip install plotly>=5.0 ipywidgets>=8.0"
    )