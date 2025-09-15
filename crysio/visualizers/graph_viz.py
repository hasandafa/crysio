"""
Graph Network Visualization Module

Provides visualization capabilities for crystal structure graphs generated
from the graph_builder module, including network topology, properties,
and analysis metrics.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
import warnings

# Import dependencies with graceful fallbacks
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Import PyTorch Geometric with fallback
try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False

# Import Crysio modules - use try/except for robust imports
try:
    from crysio.utils.exceptions import DependencyError, VisualizationError
except ImportError:
    try:
        from ..utils.exceptions import DependencyError, VisualizationError
    except ImportError:
        # Define minimal exception classes
        class DependencyError(Exception):
            pass
        class VisualizationError(Exception):
            pass


class GraphVisualizer:
    """
    Comprehensive graph network visualizer for crystal structures.
    """
    
    def __init__(self, backend: str = 'matplotlib'):
        """
        Initialize the graph visualizer.
        
        Args:
            backend: Visualization backend ('matplotlib' or 'plotly')
        """
        self.backend = backend
        self._validate_backend()
        
        # Default node colors for different elements
        self.element_colors = {
            'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00',
            'B': '#FFB5B5', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
            'F': '#90E050', 'Ne': '#B3E3F5', 'Na': '#AB5CF2', 'Mg': '#8AFF00',
            'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000', 'S': '#FFFF30',
            'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4', 'Ca': '#3DFF00',
            'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050', 'Cu': '#C88033'
        }
        
        # Network layout algorithms
        self.layout_algorithms = {
            'spring': nx.spring_layout if NETWORKX_AVAILABLE else None,
            'circular': nx.circular_layout if NETWORKX_AVAILABLE else None,
            'kamada_kawai': nx.kamada_kawai_layout if NETWORKX_AVAILABLE else None,
            'spectral': nx.spectral_layout if NETWORKX_AVAILABLE else None,
            'random': nx.random_layout if NETWORKX_AVAILABLE else None
        }
    
    def _validate_backend(self):
        """Validate that the requested backend is available."""
        if self.backend == 'matplotlib' and not MATPLOTLIB_AVAILABLE:
            raise DependencyError("Matplotlib not available. Install with: pip install matplotlib")
        elif self.backend == 'plotly' and not PLOTLY_AVAILABLE:
            raise DependencyError("Plotly not available. Install with: pip install plotly")
        elif self.backend not in ['matplotlib', 'plotly']:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def get_element_color(self, element: str) -> str:
        """Get color for an element."""
        return self.element_colors.get(element, '#FF69B4')  # Default pink


def plot_graph_network(graph_data: Union[Data, nx.Graph, Dict], 
                      layout: str = 'spring',
                      node_color_property: Optional[str] = None,
                      edge_color_property: Optional[str] = None,
                      figsize: Tuple[int, int] = (12, 8),
                      backend: str = 'matplotlib') -> Any:
    """
    Plot crystal structure as a network graph.
    
    Args:
        graph_data: Graph data (PyTorch Geometric Data, NetworkX Graph, or dict)
        layout: Network layout algorithm ('spring', 'circular', 'kamada_kawai', etc.)
        node_color_property: Property to use for node coloring
        edge_color_property: Property to use for edge coloring
        figsize: Figure size for matplotlib
        backend: Visualization backend
        
    Returns:
        Figure object
    """
    visualizer = GraphVisualizer(backend)
    
    # Convert to NetworkX graph for easier handling
    G = _convert_to_networkx(graph_data)
    
    if backend == 'matplotlib':
        return _plot_network_mpl(G, layout, node_color_property, edge_color_property, 
                               figsize, visualizer)
    elif backend == 'plotly':
        return _plot_network_plotly(G, layout, node_color_property, edge_color_property, 
                                  visualizer)


def _convert_to_networkx(graph_data: Union[Data, nx.Graph, Dict]) -> nx.Graph:
    """Convert various graph formats to NetworkX."""
    if isinstance(graph_data, nx.Graph):
        return graph_data
    
    elif PYTORCH_GEOMETRIC_AVAILABLE and isinstance(graph_data, Data):
        # Convert PyTorch Geometric Data to NetworkX
        G = nx.Graph()
        
        # Add nodes with features
        num_nodes = graph_data.x.size(0) if graph_data.x is not None else graph_data.num_nodes
        for i in range(num_nodes):
            node_attrs = {'id': i}
            if graph_data.x is not None and i < graph_data.x.size(0):
                # Add node features as attributes
                features = graph_data.x[i].numpy() if hasattr(graph_data.x[i], 'numpy') else graph_data.x[i]
                for j, feat in enumerate(features):
                    node_attrs[f'feature_{j}'] = float(feat)
            G.add_node(i, **node_attrs)
        
        # Add edges
        if graph_data.edge_index is not None:
            edge_index = graph_data.edge_index.numpy() if hasattr(graph_data.edge_index, 'numpy') else graph_data.edge_index
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                edge_attrs = {}
                if graph_data.edge_attr is not None and i < graph_data.edge_attr.size(0):
                    edge_features = graph_data.edge_attr[i].numpy() if hasattr(graph_data.edge_attr[i], 'numpy') else graph_data.edge_attr[i]
                    for j, feat in enumerate(edge_features):
                        edge_attrs[f'feature_{j}'] = float(feat)
                G.add_edge(int(src), int(dst), **edge_attrs)
        
        return G
    
    elif isinstance(graph_data, dict):
        # Convert dictionary format to NetworkX
        G = nx.Graph()
        
        # Add nodes
        if 'nodes' in graph_data:
            for node_data in graph_data['nodes']:
                if isinstance(node_data, dict):
                    node_id = node_data.get('id', len(G.nodes))
                    G.add_node(node_id, **{k: v for k, v in node_data.items() if k != 'id'})
                else:
                    G.add_node(node_data)
        
        # Add edges
        if 'edges' in graph_data:
            for edge_data in graph_data['edges']:
                if isinstance(edge_data, (list, tuple)) and len(edge_data) >= 2:
                    src, dst = edge_data[0], edge_data[1]
                    edge_attrs = edge_data[2] if len(edge_data) > 2 else {}
                    G.add_edge(src, dst, **edge_attrs)
                elif isinstance(edge_data, dict):
                    src = edge_data.get('source') or edge_data.get('src')
                    dst = edge_data.get('target') or edge_data.get('dst')
                    if src is not None and dst is not None:
                        edge_attrs = {k: v for k, v in edge_data.items() 
                                    if k not in ['source', 'target', 'src', 'dst']}
                        G.add_edge(src, dst, **edge_attrs)
        
        return G
    
    else:
        raise TypeError(f"Unsupported graph data type: {type(graph_data)}")


def _plot_network_mpl(G: nx.Graph, layout: str, node_color_property: Optional[str], 
                     edge_color_property: Optional[str], figsize: Tuple[int, int], 
                     visualizer: GraphVisualizer):
    """Matplotlib implementation of network plot."""
    if not NETWORKX_AVAILABLE:
        raise DependencyError("NetworkX required for graph visualization. Install with: pip install networkx")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get layout positions
    if layout in visualizer.layout_algorithms and visualizer.layout_algorithms[layout]:
        try:
            pos = visualizer.layout_algorithms[layout](G)
        except:
            warnings.warn(f"Layout '{layout}' failed, using spring layout")
            pos = nx.spring_layout(G)
    else:
        warnings.warn(f"Unknown layout '{layout}', using spring layout")
        pos = nx.spring_layout(G)
    
    # Determine node colors
    if node_color_property and node_color_property in list(G.nodes(data=True))[0][1]:
        node_colors = [data.get(node_color_property, 0) for _, data in G.nodes(data=True)]
        node_cmap = 'viridis'
    else:
        # Color by element if available, otherwise use default
        node_colors = []
        for _, data in G.nodes(data=True):
            element = data.get('element', data.get('atom_type', 'C'))
            node_colors.append(visualizer.get_element_color(element))
        node_cmap = None
    
    # Determine edge colors
    if edge_color_property and G.edges():
        edge_data = list(G.edges(data=True))
        if edge_data and edge_color_property in edge_data[0][2]:
            edge_colors = [data.get(edge_color_property, 0) for _, _, data in G.edges(data=True)]
            edge_cmap = 'plasma'
        else:
            edge_colors = 'gray'
            edge_cmap = None
    else:
        edge_colors = 'gray'
        edge_cmap = None
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, 
                          alpha=0.8, ax=ax, cmap=node_cmap)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.6, 
                          ax=ax, edge_cmap=edge_cmap)
    
    # Add node labels if not too many nodes
    if len(G.nodes) <= 50:
        labels = {}
        for node, data in G.nodes(data=True):
            element = data.get('element', data.get('atom_type', str(node)))
            labels[node] = element
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title(f'Crystal Structure Network Graph ({layout} layout)')
    ax.axis('off')
    
    # Add graph statistics
    stats_text = f'Nodes: {len(G.nodes)}\nEdges: {len(G.edges)}\nDensity: {nx.density(G):.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def _plot_network_plotly(G: nx.Graph, layout: str, node_color_property: Optional[str],
                        edge_color_property: Optional[str], visualizer: GraphVisualizer):
    """Plotly implementation of network plot."""
    if not NETWORKX_AVAILABLE:
        raise DependencyError("NetworkX required for graph visualization. Install with: pip install networkx")
    
    # Get layout positions
    if layout in visualizer.layout_algorithms and visualizer.layout_algorithms[layout]:
        try:
            pos = visualizer.layout_algorithms[layout](G)
        except:
            pos = nx.spring_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Extract node positions
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    # Extract edge positions
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='gray'),
                           hoverinfo='none', mode='lines', name='Bonds')
    
    # Determine node colors and info
    node_colors = []
    node_text = []
    for node, data in G.nodes(data=True):
        element = data.get('element', data.get('atom_type', 'C'))
        node_colors.append(visualizer.get_element_color(element))
        
        # Create hover text
        info_text = f'Node: {node}<br>Element: {element}'
        for key, value in data.items():
            if key not in ['element', 'atom_type']:
                info_text += f'<br>{key}: {value}'
        node_text.append(info_text)
    
    # Create node trace
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                           marker=dict(size=15, color=node_colors, line=dict(width=1, color='black')),
                           text=[data.get('element', data.get('atom_type', str(node))) 
                                for node, data in G.nodes(data=True)],
                           textposition="middle center",
                           textfont=dict(size=8),
                           hovertext=node_text,
                           hoverinfo='text',
                           name='Atoms')
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title=f'Crystal Structure Network Graph ({layout} layout)',
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text=f"Nodes: {len(G.nodes)}, Edges: {len(G.edges)}, Density: {nx.density(G):.3f}",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1000, height=800
    )
    
    return fig


def plot_adjacency_matrix(graph_data: Union[Data, nx.Graph, Dict],
                         figsize: Tuple[int, int] = (8, 6),
                         backend: str = 'matplotlib') -> Any:
    """
    Plot adjacency matrix of the crystal structure graph.
    
    Args:
        graph_data: Graph data
        figsize: Figure size for matplotlib
        backend: Visualization backend
        
    Returns:
        Figure object
    """
    visualizer = GraphVisualizer(backend)
    G = _convert_to_networkx(graph_data)
    
    if backend == 'matplotlib':
        return _plot_adjacency_mpl(G, figsize)
    elif backend == 'plotly':
        return _plot_adjacency_plotly(G)


def _plot_adjacency_mpl(G: nx.Graph, figsize: Tuple[int, int]):
    """Matplotlib implementation of adjacency matrix plot."""
    if not NETWORKX_AVAILABLE:
        raise DependencyError("NetworkX required for adjacency matrix. Install with: pip install networkx")
    
    # Get adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if SEABORN_AVAILABLE:
        sns.heatmap(adj_matrix, cmap='Blues', square=True, ax=ax, 
                   cbar_kws={'shrink': 0.8}, xticklabels=False, yticklabels=False)
    else:
        im = ax.imshow(adj_matrix, cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
    
    ax.set_title('Adjacency Matrix')
    ax.set_xlabel('Node Index')
    ax.set_ylabel('Node Index')
    
    # Add statistics
    density = nx.density(G)
    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes) if len(G.nodes) > 0 else 0
    stats_text = f'Density: {density:.3f}\nAvg Degree: {avg_degree:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def _plot_adjacency_plotly(G: nx.Graph):
    """Plotly implementation of adjacency matrix plot."""
    if not NETWORKX_AVAILABLE:
        raise DependencyError("NetworkX required for adjacency matrix. Install with: pip install networkx")
    
    # Get adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()
    
    fig = go.Figure(data=go.Heatmap(
        z=adj_matrix,
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title='Adjacency Matrix',
        xaxis_title='Node Index',
        yaxis_title='Node Index',
        width=600, height=600
    )
    
    return fig


def plot_node_properties(graph_data: Union[Data, nx.Graph, Dict], 
                        property_name: str,
                        figsize: Tuple[int, int] = (10, 6),
                        backend: str = 'matplotlib') -> Any:
    """
    Plot distribution of node properties.
    
    Args:
        graph_data: Graph data
        property_name: Name of node property to plot
        figsize: Figure size for matplotlib
        backend: Visualization backend
        
    Returns:
        Figure object
    """
    visualizer = GraphVisualizer(backend)
    G = _convert_to_networkx(graph_data)
    
    # Extract property values
    property_values = []
    for _, data in G.nodes(data=True):
        if property_name in data:
            property_values.append(data[property_name])
    
    if not property_values:
        raise ValueError(f"Property '{property_name}' not found in node data")
    
    if backend == 'matplotlib':
        return _plot_node_properties_mpl(property_values, property_name, figsize)
    elif backend == 'plotly':
        return _plot_node_properties_plotly(property_values, property_name)


def _plot_node_properties_mpl(property_values: List[float], property_name: str, 
                             figsize: Tuple[int, int]):
    """Matplotlib implementation of node properties plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1.hist(property_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel(property_name)
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of {property_name}')
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    if SEABORN_AVAILABLE:
        sns.boxplot(y=property_values, ax=ax2, color='lightcoral')
    else:
        ax2.boxplot(property_values, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightcoral', alpha=0.7))
    
    ax2.set_ylabel(property_name)
    ax2.set_title(f'Box Plot of {property_name}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def _plot_node_properties_plotly(property_values: List[float], property_name: str):
    """Plotly implementation of node properties plot."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'Distribution of {property_name}', f'Box Plot of {property_name}']
    )
    
    # Histogram
    fig.add_trace(go.Histogram(x=property_values, name='Distribution'), row=1, col=1)
    
    # Box plot
    fig.add_trace(go.Box(y=property_values, name='Box Plot'), row=1, col=2)
    
    fig.update_layout(
        title_text=f'Node Property Analysis: {property_name}',
        showlegend=False,
        width=1000, height=500
    )
    
    return fig


def plot_graph_metrics(graph_data: Union[Data, nx.Graph, Dict],
                      figsize: Tuple[int, int] = (12, 8),
                      backend: str = 'matplotlib') -> Any:
    """
    Plot various graph metrics and statistics.
    
    Args:
        graph_data: Graph data
        figsize: Figure size for matplotlib
        backend: Visualization backend
        
    Returns:
        Figure object
    """
    visualizer = GraphVisualizer(backend)
    G = _convert_to_networkx(graph_data)
    
    if backend == 'matplotlib':
        return _plot_graph_metrics_mpl(G, figsize)
    elif backend == 'plotly':
        return _plot_graph_metrics_plotly(G)


def _plot_graph_metrics_mpl(G: nx.Graph, figsize: Tuple[int, int]):
    """Matplotlib implementation of graph metrics plot."""
    if not NETWORKX_AVAILABLE:
        raise DependencyError("NetworkX required for graph metrics. Install with: pip install networkx")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Degree distribution
    degrees = [d for n, d in G.degree()]
    ax1.hist(degrees, bins=max(10, len(set(degrees))), alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Degree Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Clustering coefficient distribution
    if len(G.nodes) > 0:
        clustering_coeffs = list(nx.clustering(G).values())
        ax2.hist(clustering_coeffs, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Clustering Coefficient')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Clustering Coefficient Distribution')
        ax2.grid(True, alpha=0.3)
    
    # Betweenness centrality (for smaller graphs)
    if len(G.nodes) <= 100:
        betweenness = list(nx.betweenness_centrality(G).values())
        ax3.hist(betweenness, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('Betweenness Centrality')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Betweenness Centrality Distribution')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Graph too large\nfor betweenness\ncentrality', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Betweenness Centrality (Skipped)')
    
    # Graph summary statistics
    stats = {
        'Nodes': len(G.nodes),
        'Edges': len(G.edges),
        'Density': nx.density(G),
        'Avg Clustering': nx.average_clustering(G) if len(G.nodes) > 0 else 0,
    }
    
    # Try to compute average shortest path (expensive for large graphs)
    if len(G.nodes) <= 50 and nx.is_connected(G):
        try:
            stats['Avg Path Length'] = nx.average_shortest_path_length(G)
        except:
            stats['Avg Path Length'] = 'N/A'
    else:
        stats['Avg Path Length'] = 'N/A (too large/disconnected)'
    
    # Plot statistics as text
    ax4.axis('off')
    stats_text = '\n'.join([f'{k}: {v}' for k, v in stats.items()])
    ax4.text(0.1, 0.8, stats_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax4.set_title('Graph Statistics')
    
    plt.tight_layout()
    return fig


def _plot_graph_metrics_plotly(G: nx.Graph):
    """Plotly implementation of graph metrics plot."""
    if not NETWORKX_AVAILABLE:
        raise DependencyError("NetworkX required for graph metrics. Install with: pip install networkx")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Degree Distribution', 'Clustering Coefficient Distribution',
                       'Betweenness Centrality', 'Graph Statistics'],
        specs=[[{'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'table'}]]
    )
    
    # Degree distribution
    degrees = [d for n, d in G.degree()]
    fig.add_trace(go.Histogram(x=degrees, name='Degree Distribution'), row=1, col=1)
    
    # Clustering coefficient distribution
    if len(G.nodes) > 0:
        clustering_coeffs = list(nx.clustering(G).values())
        fig.add_trace(go.Histogram(x=clustering_coeffs, name='Clustering Coefficient'), row=1, col=2)
    
    # Betweenness centrality (for smaller graphs)
    if len(G.nodes) <= 100:
        betweenness = list(nx.betweenness_centrality(G).values())
        fig.add_trace(go.Histogram(x=betweenness, name='Betweenness Centrality'), row=2, col=1)
    
    # Graph statistics table
    stats_data = [
        ['Nodes', len(G.nodes)],
        ['Edges', len(G.edges)],
        ['Density', f'{nx.density(G):.3f}'],
        ['Avg Clustering', f'{nx.average_clustering(G):.3f}' if len(G.nodes) > 0 else '0.000'],
    ]
    
    fig.add_trace(go.Table(
        header=dict(values=['Metric', 'Value']),
        cells=dict(values=list(zip(*stats_data)))
    ), row=2, col=2)
    
    fig.update_layout(
        title_text='Graph Network Analysis',
        showlegend=False,
        width=1200, height=800
    )
    
    return fig


def analyze_graph_connectivity(graph_data: Union[Data, nx.Graph, Dict]) -> Dict[str, Any]:
    """
    Analyze connectivity properties of the crystal structure graph.
    
    Args:
        graph_data: Graph data
        
    Returns:
        Dictionary with connectivity analysis results
    """
    if not NETWORKX_AVAILABLE:
        raise DependencyError("NetworkX required for connectivity analysis. Install with: pip install networkx")
    
    G = _convert_to_networkx(graph_data)
    
    analysis = {
        'num_nodes': len(G.nodes),
        'num_edges': len(G.edges),
        'density': nx.density(G),
        'is_connected': nx.is_connected(G),
        'num_connected_components': nx.number_connected_components(G),
    }
    
    if len(G.nodes) > 0:
        analysis['average_degree'] = sum(dict(G.degree()).values()) / len(G.nodes)
        analysis['average_clustering'] = nx.average_clustering(G)
        
        # For smaller graphs, compute more expensive metrics
        if len(G.nodes) <= 100:
            analysis['diameter'] = nx.diameter(G) if nx.is_connected(G) else None
            analysis['radius'] = nx.radius(G) if nx.is_connected(G) else None
            
            if nx.is_connected(G):
                analysis['average_shortest_path_length'] = nx.average_shortest_path_length(G)
        
        # Degree statistics
        degrees = list(dict(G.degree()).values())
        analysis['degree_stats'] = {
            'min': min(degrees),
            'max': max(degrees),
            'mean': np.mean(degrees),
            'std': np.std(degrees)
        }
    
    return analysis