"""
Analysis and Statistical Plotting Module

Provides statistical visualization capabilities for crystal structure analysis,
including property distributions, correlations, and comparative analysis.
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
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import Crysio modules - use try/except for robust imports
try:
    from crysio.core.crystal import Crystal
except ImportError:
    try:
        from ..core.crystal import Crystal
    except ImportError:
        # Define minimal Crystal class for testing
        Crystal = None

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


class AnalysisVisualizer:
    """
    Statistical analysis and plotting for crystal structures.
    """
    
    def __init__(self, backend: str = 'matplotlib'):
        """
        Initialize the analysis visualizer.
        
        Args:
            backend: Visualization backend ('matplotlib' or 'plotly')
        """
        self.backend = backend
        self._validate_backend()
        
        # Set default style
        if self.backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
            if SEABORN_AVAILABLE:
                sns.set_style("whitegrid")
            plt.style.use('default')
    
    def _validate_backend(self):
        """Validate that the requested backend is available."""
        if self.backend == 'matplotlib' and not MATPLOTLIB_AVAILABLE:
            raise DependencyError("Matplotlib not available. Install with: pip install matplotlib")
        elif self.backend == 'plotly' and not PLOTLY_AVAILABLE:
            raise DependencyError("Plotly not available. Install with: pip install plotly")
        elif self.backend not in ['matplotlib', 'plotly']:
            raise ValueError(f"Unknown backend: {self.backend}")


def plot_property_distribution(data: Dict[str, List[float]], property_name: str,
                             bins: int = 30, figsize: Tuple[int, int] = (10, 6),
                             backend: str = 'matplotlib') -> Any:
    """
    Plot distribution of crystal properties.
    
    Args:
        data: Dictionary with property values
        property_name: Name of the property to plot
        bins: Number of histogram bins
        figsize: Figure size for matplotlib
        backend: Visualization backend
        
    Returns:
        Figure object
    """
    visualizer = AnalysisVisualizer(backend)
    
    if property_name not in data:
        raise ValueError(f"Property '{property_name}' not found in data")
    
    values = np.array(data[property_name])
    
    if backend == 'matplotlib':
        return _plot_distribution_mpl(values, property_name, bins, figsize)
    elif backend == 'plotly':
        return _plot_distribution_plotly(values, property_name, bins)


def _plot_distribution_mpl(values: np.ndarray, property_name: str, 
                          bins: int, figsize: Tuple[int, int]):
    """Matplotlib implementation of distribution plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    n, bins_edges, patches = ax1.hist(values, bins=bins, alpha=0.7, color='skyblue', 
                                     edgecolor='black', linewidth=0.5)
    ax1.set_xlabel(property_name)
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of {property_name}')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_val = np.mean(values)
    std_val = np.std(values)
    median_val = np.median(values)
    
    stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMedian: {median_val:.3f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Box plot
    if SEABORN_AVAILABLE:
        sns.boxplot(y=values, ax=ax2, color='lightcoral')
    else:
        ax2.boxplot(values, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightcoral', alpha=0.7))
    
    ax2.set_ylabel(property_name)
    ax2.set_title(f'Box Plot of {property_name}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def _plot_distribution_plotly(values: np.ndarray, property_name: str, bins: int):
    """Plotly implementation of distribution plot."""
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'Distribution of {property_name}', f'Box Plot of {property_name}'],
        specs=[[{'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=values, nbinsx=bins, name='Distribution', 
                    marker_color='skyblue', opacity=0.7),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=values, name='Box Plot', marker_color='lightcoral'),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text=property_name, row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text=property_name, row=1, col=2)
    
    fig.update_layout(
        title_text=f'Statistical Analysis of {property_name}',
        showlegend=False,
        width=1000, height=500
    )
    
    return fig


def plot_correlation_matrix(data: Dict[str, List[float]], properties: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (10, 8), backend: str = 'matplotlib') -> Any:
    """
    Plot correlation matrix of crystal properties.
    
    Args:
        data: Dictionary with property values
        properties: List of properties to include (None for all)
        figsize: Figure size for matplotlib
        backend: Visualization backend
        
    Returns:
        Figure object
    """
    visualizer = AnalysisVisualizer(backend)
    
    # Select properties
    if properties is None:
        properties = list(data.keys())
    
    # Check that all properties exist
    missing = [prop for prop in properties if prop not in data]
    if missing:
        raise ValueError(f"Properties not found in data: {missing}")
    
    # Create correlation matrix
    import pandas as pd
    df = pd.DataFrame({prop: data[prop] for prop in properties})
    corr_matrix = df.corr()
    
    if backend == 'matplotlib':
        return _plot_correlation_mpl(corr_matrix, figsize)
    elif backend == 'plotly':
        return _plot_correlation_plotly(corr_matrix)


def _plot_correlation_mpl(corr_matrix, figsize: Tuple[int, int]):
    """Matplotlib implementation of correlation matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if SEABORN_AVAILABLE:
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
    else:
        im = ax.imshow(corr_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Coefficient')
        
        # Add text annotations
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        # Set ticks and labels
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.index)
    
    ax.set_title('Property Correlation Matrix')
    plt.tight_layout()
    return fig


def _plot_correlation_plotly(corr_matrix):
    """Plotly implementation of correlation matrix."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Property Correlation Matrix',
        xaxis_title='Properties',
        yaxis_title='Properties',
        width=800, height=800
    )
    
    return fig


def plot_structure_comparison(crystals: List[Crystal], property_name: str,
                            labels: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (10, 6),
                            backend: str = 'matplotlib') -> Any:
    """
    Compare a property across multiple crystal structures.
    
    Args:
        crystals: List of crystal structures
        property_name: Property to compare
        labels: Labels for each structure
        figsize: Figure size for matplotlib
        backend: Visualization backend
        
    Returns:
        Figure object
    """
    visualizer = AnalysisVisualizer(backend)
    
    if labels is None:
        labels = [f'Structure {i+1}' for i in range(len(crystals))]
    
    if len(labels) != len(crystals):
        raise ValueError("Number of labels must match number of crystals")
    
    # Extract property values
    values = []
    for crystal in crystals:
        if property_name == 'volume':
            values.append(crystal.lattice_parameters.volume)
        elif property_name == 'density':
            values.append(crystal.density)
        elif property_name == 'num_atoms':
            values.append(len(crystal.atomic_sites))
        elif property_name in ['a', 'b', 'c']:
            values.append(getattr(crystal.lattice_parameters, property_name))
        elif property_name in ['alpha', 'beta', 'gamma']:
            values.append(getattr(crystal.lattice_parameters, property_name))
        else:
            raise ValueError(f"Unknown property: {property_name}")
    
    if backend == 'matplotlib':
        return _plot_comparison_mpl(values, labels, property_name, figsize)
    elif backend == 'plotly':
        return _plot_comparison_plotly(values, labels, property_name)


def _plot_comparison_mpl(values: List[float], labels: List[str], 
                        property_name: str, figsize: Tuple[int, int]):
    """Matplotlib implementation of structure comparison."""
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
                f'{value:.3f}', ha='center', va='bottom')
    
    ax.set_xlabel('Crystal Structures')
    ax.set_ylabel(property_name.capitalize())
    ax.set_title(f'Comparison of {property_name.capitalize()}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def _plot_comparison_plotly(values: List[float], labels: List[str], property_name: str):
    """Plotly implementation of structure comparison."""
    fig = go.Figure(data=[
        go.Bar(x=labels, y=values, 
               marker_color=px.colors.qualitative.Set3[:len(labels)],
               text=[f'{v:.3f}' for v in values],
               textposition='auto')
    ])
    
    fig.update_layout(
        title=f'Comparison of {property_name.capitalize()}',
        xaxis_title='Crystal Structures',
        yaxis_title=property_name.capitalize(),
        width=800, height=600
    )
    
    return fig


def plot_formation_energy(data: Dict[str, List[float]], composition_col: str = 'composition',
                         energy_col: str = 'formation_energy_per_atom',
                         figsize: Tuple[int, int] = (10, 6),
                         backend: str = 'matplotlib') -> Any:
    """
    Plot formation energy vs composition.
    
    Args:
        data: Dictionary with formation energy data
        composition_col: Column name for composition
        energy_col: Column name for formation energy
        figsize: Figure size for matplotlib
        backend: Visualization backend
        
    Returns:
        Figure object
    """
    visualizer = AnalysisVisualizer(backend)
    
    if composition_col not in data or energy_col not in data:
        raise ValueError(f"Required columns not found: {composition_col}, {energy_col}")
    
    compositions = data[composition_col]
    energies = data[energy_col]
    
    if backend == 'matplotlib':
        return _plot_formation_energy_mpl(compositions, energies, figsize)
    elif backend == 'plotly':
        return _plot_formation_energy_plotly(compositions, energies)


def _plot_formation_energy_mpl(compositions: List[str], energies: List[float], 
                              figsize: Tuple[int, int]):
    """Matplotlib implementation of formation energy plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color points by energy
    scatter = ax.scatter(range(len(compositions)), energies, 
                        c=energies, cmap='RdYlBu_r', alpha=0.7, s=50)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Formation Energy (eV/atom)')
    
    ax.set_xlabel('Material Index')
    ax.set_ylabel('Formation Energy (eV/atom)')
    ax.set_title('Formation Energy by Composition')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Stability threshold')
    ax.legend()
    
    plt.tight_layout()
    return fig


def _plot_formation_energy_plotly(compositions: List[str], energies: List[float]):
    """Plotly implementation of formation energy plot."""
    fig = go.Figure(data=go.Scatter(
        x=list(range(len(compositions))),
        y=energies,
        mode='markers',
        marker=dict(
            color=energies,
            colorscale='RdYlBu_r',
            size=8,
            colorbar=dict(title="Formation Energy (eV/atom)")
        ),
        text=compositions,
        hovertemplate='<b>%{text}</b><br>Energy: %{y:.3f} eV/atom<extra></extra>'
    ))
    
    # Add stability threshold line
    fig.add_hline(y=0, line_dash="dash", line_color="black", 
                  annotation_text="Stability threshold")
    
    fig.update_layout(
        title='Formation Energy by Composition',
        xaxis_title='Material Index',
        yaxis_title='Formation Energy (eV/atom)',
        width=1000, height=600
    )
    
    return fig


def plot_property_scatter(data: Dict[str, List[float]], x_property: str, y_property: str,
                         color_property: Optional[str] = None,
                         figsize: Tuple[int, int] = (8, 6),
                         backend: str = 'matplotlib') -> Any:
    """
    Create scatter plot of two properties.
    
    Args:
        data: Dictionary with property values
        x_property: Property for x-axis
        y_property: Property for y-axis
        color_property: Property for color coding (optional)
        figsize: Figure size for matplotlib
        backend: Visualization backend
        
    Returns:
        Figure object
    """
    visualizer = AnalysisVisualizer(backend)
    
    # Check required properties exist
    required = [x_property, y_property]
    if color_property:
        required.append(color_property)
    
    missing = [prop for prop in required if prop not in data]
    if missing:
        raise ValueError(f"Properties not found in data: {missing}")
    
    x_values = data[x_property]
    y_values = data[y_property]
    c_values = data[color_property] if color_property else None
    
    if backend == 'matplotlib':
        return _plot_scatter_mpl(x_values, y_values, c_values, 
                                x_property, y_property, color_property, figsize)
    elif backend == 'plotly':
        return _plot_scatter_plotly(x_values, y_values, c_values,
                                   x_property, y_property, color_property)


def _plot_scatter_mpl(x_values: List[float], y_values: List[float], 
                     c_values: Optional[List[float]], x_property: str, 
                     y_property: str, color_property: Optional[str], 
                     figsize: Tuple[int, int]):
    """Matplotlib implementation of scatter plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if c_values:
        scatter = ax.scatter(x_values, y_values, c=c_values, cmap='viridis', 
                           alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_property)
    else:
        ax.scatter(x_values, y_values, alpha=0.7, s=50, color='skyblue',
                  edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(x_property)
    ax.set_ylabel(y_property)
    ax.set_title(f'{y_property} vs {x_property}')
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient if both are numeric
    try:
        corr_coeff = np.corrcoef(x_values, y_values)[0, 1]
        ax.text(0.02, 0.98, f'r = {corr_coeff:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    except:
        pass
    
    plt.tight_layout()
    return fig


def _plot_scatter_plotly(x_values: List[float], y_values: List[float],
                        c_values: Optional[List[float]], x_property: str,
                        y_property: str, color_property: Optional[str]):
    """Plotly implementation of scatter plot."""
    if c_values:
        fig = go.Figure(data=go.Scatter(
            x=x_values, y=y_values,
            mode='markers',
            marker=dict(
                color=c_values,
                colorscale='viridis',
                size=8,
                colorbar=dict(title=color_property) if color_property else None,
                line=dict(width=1, color='black')
            ),
            hovertemplate=f'<b>{x_property}</b>: %{{x}}<br><b>{y_property}</b>: %{{y}}<br>' +
                         (f'<b>{color_property}</b>: %{{marker.color}}<br>' if color_property else '') +
                         '<extra></extra>'
        ))
    else:
        fig = go.Figure(data=go.Scatter(
            x=x_values, y=y_values,
            mode='markers',
            marker=dict(color='skyblue', size=8, line=dict(width=1, color='black')),
            hovertemplate=f'<b>{x_property}</b>: %{{x}}<br><b>{y_property}</b>: %{{y}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'{y_property} vs {x_property}',
        xaxis_title=x_property,
        yaxis_title=y_property,
        width=800, height=600
    )
    
    return fig