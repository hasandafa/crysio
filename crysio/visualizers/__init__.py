"""
Crysio Visualization Module

This module provides comprehensive visualization capabilities for crystal structures,
including 2D plots, 3D interactive viewers, graph network analysis, and statistical charts.
"""

# Import core visualization functions with graceful fallbacks
# Use absolute imports to avoid relative import issues

# Check for required dependencies
MATPLOTLIB_AVAILABLE = False
PLOTLY_AVAILABLE = False
SEABORN_AVAILABLE = False
NETWORKX_AVAILABLE = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import seaborn
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import networkx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Crystal visualization functions
CRYSTAL_VIZ_AVAILABLE = False
if MATPLOTLIB_AVAILABLE:
    try:
        from crysio.visualizers.crystal_viz import (
            plot_unit_cell_2d,
            plot_crystal_3d,
            plot_lattice_parameters,
            plot_atomic_positions,
            CrystalVisualizer
        )
        CRYSTAL_VIZ_AVAILABLE = True
    except ImportError:
        try:
            # Fallback to direct import without relative paths
            import sys
            import os
            current_dir = os.path.dirname(__file__)
            sys.path.insert(0, current_dir)
            
            from crystal_viz import (
                plot_unit_cell_2d,
                plot_crystal_3d,
                plot_lattice_parameters,
                plot_atomic_positions,
                CrystalVisualizer
            )
            CRYSTAL_VIZ_AVAILABLE = True
        except ImportError:
            CRYSTAL_VIZ_AVAILABLE = False

# Analysis plots functions
ANALYSIS_PLOTS_AVAILABLE = False
if MATPLOTLIB_AVAILABLE:
    try:
        from crysio.visualizers.analysis_plots import (
            plot_property_distribution,
            plot_correlation_matrix,
            plot_structure_comparison,
            plot_formation_energy,
            plot_property_scatter,
            AnalysisVisualizer
        )
        ANALYSIS_PLOTS_AVAILABLE = True
    except ImportError:
        try:
            # Fallback to direct import
            from analysis_plots import (
                plot_property_distribution,
                plot_correlation_matrix,
                plot_structure_comparison,
                plot_formation_energy,
                plot_property_scatter,
                AnalysisVisualizer
            )
            ANALYSIS_PLOTS_AVAILABLE = True
        except ImportError:
            ANALYSIS_PLOTS_AVAILABLE = False

# Graph visualization functions
GRAPH_VIZ_AVAILABLE = False
if MATPLOTLIB_AVAILABLE and NETWORKX_AVAILABLE:
    try:
        from crysio.visualizers.graph_viz import (
            plot_graph_network,
            plot_adjacency_matrix,
            plot_node_properties,
            plot_graph_metrics,
            analyze_graph_connectivity,
            GraphVisualizer
        )
        GRAPH_VIZ_AVAILABLE = True
    except ImportError:
        try:
            # Fallback to direct import
            from graph_viz import (
                plot_graph_network,
                plot_adjacency_matrix,
                plot_node_properties,
                plot_graph_metrics,
                analyze_graph_connectivity,
                GraphVisualizer
            )
            GRAPH_VIZ_AVAILABLE = True
        except ImportError:
            GRAPH_VIZ_AVAILABLE = False

# Build __all__ list dynamically based on what's available
__all__ = []

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
        'AnalysisVisualizer'
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

# Availability flags
__all__.extend([
    'CRYSTAL_VIZ_AVAILABLE',
    'ANALYSIS_PLOTS_AVAILABLE', 
    'GRAPH_VIZ_AVAILABLE',
    'MATPLOTLIB_AVAILABLE',
    'PLOTLY_AVAILABLE',
    'SEABORN_AVAILABLE',
    'NETWORKX_AVAILABLE'
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
        'graph_viz': GRAPH_VIZ_AVAILABLE
    }


def check_dependencies():
    """
    Check visualization dependencies and provide installation guidance.
    
    Returns:
        dict: Status of required and optional dependencies
    """
    dependencies = {}
    
    # Required dependencies
    if MATPLOTLIB_AVAILABLE:
        import matplotlib
        dependencies['matplotlib'] = {'available': True, 'version': matplotlib.__version__}
    else:
        dependencies['matplotlib'] = {'available': False, 'install': 'pip install matplotlib'}
    
    # Optional dependencies
    if PLOTLY_AVAILABLE:
        import plotly
        dependencies['plotly'] = {'available': True, 'version': plotly.__version__}
    else:
        dependencies['plotly'] = {'available': False, 'install': 'pip install plotly'}
    
    if SEABORN_AVAILABLE:
        import seaborn
        dependencies['seaborn'] = {'available': True, 'version': seaborn.__version__}
    else:
        dependencies['seaborn'] = {'available': False, 'install': 'pip install seaborn'}
    
    if NETWORKX_AVAILABLE:
        import networkx
        dependencies['networkx'] = {'available': True, 'version': networkx.__version__}
    else:
        dependencies['networkx'] = {'available': False, 'install': 'pip install networkx'}
    
    return dependencies


def print_status():
    """Print status of visualization modules."""
    print("Crysio Visualization Module Status")
    print("=" * 40)
    
    availability = get_available_visualizers()
    for module, available in availability.items():
        status = "✅" if available else "❌"
        print(f"{status} {module}: {available}")
    
    print("\nDependency Status:")
    deps = check_dependencies()
    for dep, info in deps.items():
        if info['available']:
            version = info.get('version', 'unknown')
            print(f"✅ {dep}: v{version}")
        else:
            install_cmd = info.get('install', 'unknown')
            print(f"❌ {dep}: {install_cmd}")


# Show import warnings if modules failed to load
if not CRYSTAL_VIZ_AVAILABLE and MATPLOTLIB_AVAILABLE:
    import warnings
    warnings.warn("Crystal visualization module could not be loaded. Check crystal_viz.py for errors.")

if not ANALYSIS_PLOTS_AVAILABLE and MATPLOTLIB_AVAILABLE:
    import warnings
    warnings.warn("Analysis plots module could not be loaded. Check analysis_plots.py for errors.")

if not GRAPH_VIZ_AVAILABLE and MATPLOTLIB_AVAILABLE and NETWORKX_AVAILABLE:
    import warnings
    warnings.warn("Graph visualization module could not be loaded. Check graph_viz.py for errors.")