"""
Crysio Visualization Module

This module provides comprehensive visualization capabilities for crystal structures,
including 2D plots, 3D interactive viewers, and statistical analysis charts.
"""

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    plt = None
    go = None

def check_visualization_dependencies():
    """Check if all visualization dependencies are available."""
    missing = []
    
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    try:
        import plotly
    except ImportError:
        missing.append("plotly")
    
    try:
        import seaborn
    except ImportError:
        missing.append("seaborn")
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    try:
        import networkx
    except ImportError:
        missing.append("networkx")
    
    if missing:
        print(f"⚠️  Missing visualization dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All visualization dependencies available")
    return True

# Core visualization functions
from .crystal_viz import (
    plot_unit_cell_2d,
    plot_crystal_3d,
    plot_lattice_parameters,
    plot_atomic_positions,
    CrystalVisualizer
)

from .analysis_plots import (
    plot_property_distribution,
    plot_correlation_matrix,
    plot_structure_comparison,
    AnalysisVisualizer
)

# Graph visualization functions (with graceful fallbacks)
try:
    from .graph_viz import (
        plot_graph_network,
        plot_adjacency_matrix,
        plot_3d_graph_overlay,
        plot_graph_metrics,
        GraphVisualizer
    )
    GRAPH_VIZ_AVAILABLE = True
except ImportError:
    GRAPH_VIZ_AVAILABLE = False
    
    def _graph_viz_not_available(*args, **kwargs):
        raise ImportError(
            "Graph visualization not available. Install with: "
            "pip install networkx"
        )
    
    plot_graph_network = _graph_viz_not_available
    plot_adjacency_matrix = _graph_viz_not_available
    plot_3d_graph_overlay = _graph_viz_not_available
    plot_graph_metrics = _graph_viz_not_available
    GraphVisualizer = _graph_viz_not_available

# Export main classes and functions
__all__ = [
    # Core visualization
    'plot_unit_cell_2d',
    'plot_crystal_3d', 
    'plot_lattice_parameters',
    'plot_atomic_positions',
    'CrystalVisualizer',
    
    # Analysis plots
    'plot_property_distribution',
    'plot_correlation_matrix', 
    'plot_structure_comparison',
    'AnalysisVisualizer',
    
    # Graph visualization
    'plot_graph_network',
    'plot_adjacency_matrix',
    'plot_3d_graph_overlay',
    'plot_graph_metrics',
    'GraphVisualizer',
    
    # Utilities
    'check_visualization_dependencies',
    'VISUALIZATION_AVAILABLE',
    'GRAPH_VIZ_AVAILABLE'
]