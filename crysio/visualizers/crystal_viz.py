"""
Crystal Structure Visualization Module - FIXED VERSION

Provides 2D and 3D visualization capabilities for crystal structures,
including unit cells, atomic positions, and lattice parameters.
Compatible with actual Crysio Crystal class structure.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
import warnings

# Import dependencies with graceful fallbacks
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# FIXED: Use direct imports instead of relative
try:
    from crysio.core.crystal import Crystal
    from crysio.utils.exceptions import DependencyError, VisualizationError
except ImportError:
    Crystal = None
    class DependencyError(Exception): pass
    class VisualizationError(Exception): pass


def create_lattice_matrix(lattice_params):
    """Create lattice matrix from lattice parameters."""
    a, b, c = lattice_params.a, lattice_params.b, lattice_params.c
    alpha, beta, gamma = np.radians([lattice_params.alpha, lattice_params.beta, lattice_params.gamma])
    
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    
    # Volume calculation
    volume = a * b * c * np.sqrt(1 + 2*cos_alpha*cos_beta*cos_gamma - 
                                 cos_alpha**2 - cos_beta**2 - cos_gamma**2)
    
    # Lattice vectors as columns
    lattice_matrix = np.array([
        [a, b * cos_gamma, c * cos_beta],
        [0, b * sin_gamma, c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma],
        [0, 0, volume / (a * b * sin_gamma)]
    ])
    
    return lattice_matrix


class CrystalVisualizer:
    """Comprehensive crystal structure visualizer with 2D and 3D capabilities."""
    
    def __init__(self, backend: str = 'matplotlib'):
        """Initialize the crystal visualizer."""
        self.backend = backend
        self._validate_backend()
        
        # Color mapping for elements
        self.element_colors = {
            'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00',
            'B': '#FFB5B5', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
            'F': '#90E050', 'Ne': '#B3E3F5', 'Na': '#AB5CF2', 'Mg': '#8AFF00',
            'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000', 'S': '#FFFF30',
            'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4', 'Ca': '#3DFF00',
            'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V': '#A6A6AB', 'Cr': '#8A99C7',
            'Mn': '#9C7AC7', 'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050',
            'Cu': '#C88033', 'Zn': '#7D80B0', 'Ga': '#C28F8F', 'Ge': '#668F8F',
            'Ce': '#FFFFC7', 'Pr': '#D9FFC7', 'Nd': '#C7FFC7'
        }
        
        # Default atomic radii (in Angstroms)
        self.atomic_radii = {
            'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84,
            'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
            'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07,
            'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76,
            'Fe': 1.26, 'Co': 1.25, 'Ni': 1.24, 'Cu': 1.28, 'Zn': 1.33,
            'Ga': 1.22, 'Ge': 1.22, 'Ce': 1.81, 'Pr': 1.82, 'Nd': 1.81
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
    
    def get_atomic_radius(self, element: str) -> float:
        """Get atomic radius for an element."""
        return self.atomic_radii.get(element, 1.0)  # Default 1.0 Angstrom


def plot_unit_cell_2d(crystal: Crystal, plane: str = 'xy', figsize: Tuple[int, int] = (8, 6), 
                     backend: str = 'matplotlib') -> Any:
    """Plot 2D projection of unit cell."""
    visualizer = CrystalVisualizer(backend)
    
    if backend == 'matplotlib':
        return _plot_unit_cell_2d_mpl(crystal, plane, figsize, visualizer)
    elif backend == 'plotly':
        return _plot_unit_cell_2d_plotly(crystal, plane, visualizer)


def _plot_unit_cell_2d_mpl(crystal: Crystal, plane: str, figsize: Tuple[int, int], 
                          visualizer: CrystalVisualizer):
    """Matplotlib implementation of 2D unit cell plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get projection indices
    plane_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    if plane not in plane_map:
        raise ValueError(f"Invalid plane: {plane}. Use 'xy', 'xz', or 'yz'")
    
    x_idx, y_idx = plane_map[plane]
    
    # FIXED: Use lattice_parameters instead of lattice
    lattice_matrix = create_lattice_matrix(crystal.lattice_parameters)
    
    # Unit cell corners
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]
    ])
    
    # Transform to Cartesian coordinates
    cart_corners = corners @ lattice_matrix
    
    # Plot unit cell edges
    ax.plot(cart_corners[:5, x_idx], cart_corners[:5, y_idx], 'k-', linewidth=2, alpha=0.7, label='Unit cell')
    ax.plot(cart_corners[5:, x_idx], cart_corners[5:, y_idx], 'k-', linewidth=2, alpha=0.7)
    
    # Connect top and bottom faces
    for i in range(4):
        ax.plot([cart_corners[i, x_idx], cart_corners[i+5, x_idx]], 
                [cart_corners[i, y_idx], cart_corners[i+5, y_idx]], 'k-', linewidth=2, alpha=0.7)
    
    # Plot atoms
    plotted_elements = set()
    for site in crystal.atomic_sites:
        # FIXED: Use site.position instead of site.fractional_coords
        cart_pos = np.array(site.position) @ lattice_matrix
        
        color = visualizer.get_element_color(site.element)
        radius = visualizer.get_atomic_radius(site.element) * 200  # Scale for visualization
        
        # Add to legend only once per element
        label = site.element if site.element not in plotted_elements else ""
        if label:
            plotted_elements.add(site.element)
        
        ax.scatter(cart_pos[x_idx], cart_pos[y_idx], c=color, s=radius, 
                  alpha=0.8, edgecolors='black', linewidth=0.5, label=label)
    
    # Legend
    if plotted_elements:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_xlabel(f'{["x", "y", "z"][x_idx]} (Å)')
    ax.set_ylabel(f'{["x", "y", "z"][y_idx]} (Å)')
    ax.set_title(f'Unit Cell Projection ({plane.upper()} plane)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def _plot_unit_cell_2d_plotly(crystal: Crystal, plane: str, visualizer: CrystalVisualizer):
    """Plotly implementation of 2D unit cell plot."""
    # Get projection indices
    plane_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    if plane not in plane_map:
        raise ValueError(f"Invalid plane: {plane}. Use 'xy', 'xz', or 'yz'")
    
    x_idx, y_idx = plane_map[plane]
    axis_labels = ['x', 'y', 'z']
    
    fig = go.Figure()
    
    # FIXED: Use lattice_parameters instead of lattice
    lattice_matrix = create_lattice_matrix(crystal.lattice_parameters)
    
    # Unit cell corners
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]
    ])
    
    cart_corners = corners @ lattice_matrix
    
    # Add unit cell edges
    fig.add_trace(go.Scatter(
        x=cart_corners[:5, x_idx], y=cart_corners[:5, y_idx],
        mode='lines', line=dict(color='black', width=2),
        name='Unit cell', showlegend=True
    ))
    
    # Plot atoms
    elements_plotted = set()
    for site in crystal.atomic_sites:
        # FIXED: Use site.position instead of site.fractional_coords
        cart_pos = np.array(site.position) @ lattice_matrix
        color = visualizer.get_element_color(site.element)
        radius = visualizer.get_atomic_radius(site.element) * 20
        
        show_legend = site.element not in elements_plotted
        elements_plotted.add(site.element)
        
        fig.add_trace(go.Scatter(
            x=[cart_pos[x_idx]], y=[cart_pos[y_idx]],
            mode='markers',
            marker=dict(color=color, size=radius, line=dict(color='black', width=1)),
            name=site.element,
            showlegend=show_legend
        ))
    
    fig.update_layout(
        title=f'Unit Cell Projection ({plane.upper()} plane)',
        xaxis_title=f'{axis_labels[x_idx]} (Å)',
        yaxis_title=f'{axis_labels[y_idx]} (Å)',
        showlegend=True,
        width=800, height=600
    )
    
    return fig


def plot_crystal_3d(crystal: Crystal, supercell: Tuple[int, int, int] = (1, 1, 1),
                   figsize: Tuple[int, int] = (10, 8), backend: str = 'matplotlib') -> Any:
    """Plot 3D crystal structure."""
    visualizer = CrystalVisualizer(backend)
    
    if backend == 'matplotlib':
        return _plot_crystal_3d_mpl(crystal, supercell, figsize, visualizer)
    elif backend == 'plotly':
        return _plot_crystal_3d_plotly(crystal, supercell, visualizer)


def _plot_crystal_3d_mpl(crystal: Crystal, supercell: Tuple[int, int, int], 
                        figsize: Tuple[int, int], visualizer: CrystalVisualizer):
    """Matplotlib implementation of 3D crystal plot."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # FIXED: Use lattice_parameters instead of lattice
    lattice_matrix = create_lattice_matrix(crystal.lattice_parameters)
    
    # Generate supercell atoms
    nx, ny, nz = supercell
    elements_plotted = set()
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for site in crystal.atomic_sites:
                    # FIXED: Use site.position instead of site.fractional_coords
                    frac_coords = np.array(site.position) + np.array([i, j, k])
                    cart_pos = frac_coords @ lattice_matrix
                    
                    color = visualizer.get_element_color(site.element)
                    radius = visualizer.get_atomic_radius(site.element) * 100
                    
                    show_label = site.element not in elements_plotted
                    elements_plotted.add(site.element)
                    
                    ax.scatter(cart_pos[0], cart_pos[1], cart_pos[2], 
                             c=color, s=radius, alpha=0.8, edgecolors='black',
                             label=site.element if show_label else "")
    
    # Plot unit cell edges for each cell in supercell
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                origin = np.array([i, j, k]) @ lattice_matrix
                _draw_unit_cell_edges_3d(ax, origin, lattice_matrix)
    
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    ax.set_zlabel('z (Å)')
    ax.set_title(f'3D Crystal Structure ({nx}×{ny}×{nz} supercell)')
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    return fig


def _plot_crystal_3d_plotly(crystal: Crystal, supercell: Tuple[int, int, int], 
                           visualizer: CrystalVisualizer):
    """Plotly implementation of 3D crystal plot."""
    fig = go.Figure()
    
    # FIXED: Use lattice_parameters instead of lattice
    lattice_matrix = create_lattice_matrix(crystal.lattice_parameters)
    nx, ny, nz = supercell
    
    # Generate supercell atoms
    elements_plotted = set()
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for site in crystal.atomic_sites:
                    # FIXED: Use site.position instead of site.fractional_coords
                    frac_coords = np.array(site.position) + np.array([i, j, k])
                    cart_pos = frac_coords @ lattice_matrix
                    
                    color = visualizer.get_element_color(site.element)
                    radius = visualizer.get_atomic_radius(site.element) * 10
                    
                    show_legend = site.element not in elements_plotted
                    elements_plotted.add(site.element)
                    
                    fig.add_trace(go.Scatter3d(
                        x=[cart_pos[0]], y=[cart_pos[1]], z=[cart_pos[2]],
                        mode='markers',
                        marker=dict(color=color, size=radius, line=dict(color='black', width=1)),
                        name=site.element,
                        showlegend=show_legend
                    ))
    
    # Add unit cell edges
    _add_unit_cell_edges_3d_plotly(fig, lattice_matrix, supercell)
    
    fig.update_layout(
        title=f'3D Crystal Structure ({nx}×{ny}×{nz} supercell)',
        scene=dict(
            xaxis_title='x (Å)',
            yaxis_title='y (Å)',
            zaxis_title='z (Å)',
            aspectmode='data'
        ),
        showlegend=True,
        width=1000, height=800
    )
    
    return fig


def _draw_unit_cell_edges_3d(ax, origin: np.ndarray, lattice_matrix: np.ndarray):
    """Draw unit cell edges in 3D matplotlib plot."""
    # Define unit cell vertices
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
    ])
    
    # Transform to Cartesian
    cart_vertices = (vertices @ lattice_matrix) + origin
    
    # Define edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    # Draw edges
    for edge in edges:
        start, end = edge
        ax.plot3D([cart_vertices[start, 0], cart_vertices[end, 0]],
                  [cart_vertices[start, 1], cart_vertices[end, 1]],
                  [cart_vertices[start, 2], cart_vertices[end, 2]], 'k-', alpha=0.3)


def _add_unit_cell_edges_3d_plotly(fig, lattice_matrix: np.ndarray, supercell: Tuple[int, int, int]):
    """Add unit cell edges to Plotly 3D plot."""
    nx, ny, nz = supercell
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                origin = np.array([i, j, k]) @ lattice_matrix
                
                # Unit cell vertices
                vertices = np.array([
                    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
                ])
                
                cart_vertices = (vertices @ lattice_matrix) + origin
                
                # Define edges
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
                    [4, 5], [5, 6], [6, 7], [7, 4],  # Top
                    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
                ]
                
                # Add edges
                for edge in edges:
                    start, end = edge
                    fig.add_trace(go.Scatter3d(
                        x=[cart_vertices[start, 0], cart_vertices[end, 0]],
                        y=[cart_vertices[start, 1], cart_vertices[end, 1]],
                        z=[cart_vertices[start, 2], cart_vertices[end, 2]],
                        mode='lines',
                        line=dict(color='black', width=2),
                        showlegend=False
                    ))


def plot_lattice_parameters(crystal: Crystal, figsize: Tuple[int, int] = (10, 6)) -> Any:
    """Plot lattice parameters as bar chart."""
    if not MATPLOTLIB_AVAILABLE:
        raise DependencyError("Matplotlib required for lattice parameter plots")
    
    # FIXED: Use lattice_parameters instead of lattice
    lattice = crystal.lattice_parameters
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Lattice lengths
    lengths = [lattice.a, lattice.b, lattice.c]
    labels = ['a', 'b', 'c']
    colors = ['red', 'green', 'blue']
    
    bars1 = ax1.bar(labels, lengths, color=colors, alpha=0.7)
    ax1.set_ylabel('Length (Å)')
    ax1.set_title('Lattice Parameters (Lengths)')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, length in zip(bars1, lengths):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(lengths),
                f'{length:.3f}', ha='center', va='bottom')
    
    # Lattice angles
    angles = [lattice.alpha, lattice.beta, lattice.gamma]
    angle_labels = ['α', 'β', 'γ']
    
    bars2 = ax2.bar(angle_labels, angles, color=['orange', 'purple', 'brown'], alpha=0.7)
    ax2.set_ylabel('Angle (°)')
    ax2.set_title('Lattice Parameters (Angles)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, angle in zip(bars2, angles):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{angle:.1f}°', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def plot_atomic_positions(crystal: Crystal, projection: str = '3d', 
                         figsize: Tuple[int, int] = (8, 6)) -> Any:
    """Plot atomic positions in crystal structure."""
    if not MATPLOTLIB_AVAILABLE:
        raise DependencyError("Matplotlib required for atomic position plots")
    
    visualizer = CrystalVisualizer('matplotlib')
    
    if projection == '3d':
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        elements_plotted = set()
        for site in crystal.atomic_sites:
            color = visualizer.get_element_color(site.element)
            radius = visualizer.get_atomic_radius(site.element) * 100
            
            show_label = site.element not in elements_plotted
            elements_plotted.add(site.element)
            
            # FIXED: Use site.position instead of site.fractional_coords
            ax.scatter(site.position[0], site.position[1], 
                      site.position[2], c=color, s=radius, alpha=0.8,
                      edgecolors='black', label=site.element if show_label else "")
        
        ax.set_xlabel('Fractional x')
        ax.set_ylabel('Fractional y')
        ax.set_zlabel('Fractional z')
        ax.set_title('Atomic Positions (Fractional Coordinates)')
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
    else:  # 2D projection
        fig, ax = plt.subplots(figsize=figsize)
        
        elements_plotted = set()
        for site in crystal.atomic_sites:
            color = visualizer.get_element_color(site.element)
            radius = visualizer.get_atomic_radius(site.element) * 200
            
            show_label = site.element not in elements_plotted
            elements_plotted.add(site.element)
            
            # FIXED: Use site.position instead of site.fractional_coords
            ax.scatter(site.position[0], site.position[1], 
                      c=color, s=radius, alpha=0.8, edgecolors='black',
                      label=site.element if show_label else "")
        
        ax.set_xlabel('Fractional x')
        ax.set_ylabel('Fractional y')
        ax.set_title('Atomic Positions (xy projection)')
        ax.grid(True, alpha=0.3)
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    return fig