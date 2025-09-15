"""
Crystal Structure Visualization Module

Provides 2D and 3D visualization capabilities for crystal structures,
including unit cells, atomic positions, and lattice parameters.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path

# Optional dependencies with graceful fallbacks
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    patches = None
    Axes3D = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None
    make_subplots = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

from ..core.crystal import Crystal, AtomicSite, LatticeParameters
from ..utils.exceptions import VisualizationError, DependencyError


# Atomic colors for visualization (CPK color scheme)
ATOMIC_COLORS = {
    'H': '#FFFFFF',  'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00',
    'B': '#FFB5B5',  'C': '#909090',  'N': '#3050F8',  'O': '#FF0D0D',
    'F': '#90E050',  'Ne': '#B3E3F5', 'Na': '#AB5CF2', 'Mg': '#8AFF00',
    'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000',  'S': '#FFFF30',
    'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4',  'Ca': '#3DFF00',
    'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V': '#A6A6AB',  'Cr': '#8A99C7',
    'Mn': '#9C7AC7', 'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050',
    'Cu': '#C88033', 'Zn': '#7D80B0', 'Ga': '#C28F8F', 'Ge': '#668F8F',
    'As': '#BD80E3', 'Se': '#FFA100', 'Br': '#A62929', 'Kr': '#5CB8D1'
}

# Atomic radii (in Angstroms) for visualization
ATOMIC_RADII = {
    'H': 0.31,   'He': 0.28,  'Li': 1.28,  'Be': 0.96,  'B': 0.84,
    'C': 0.76,   'N': 0.71,   'O': 0.66,   'F': 0.57,   'Ne': 0.58,
    'Na': 1.66,  'Mg': 1.41,  'Al': 1.21,  'Si': 1.11,  'P': 1.07,
    'S': 1.05,   'Cl': 1.02,  'Ar': 1.06,  'K': 2.03,   'Ca': 1.76,
    'Sc': 1.70,  'Ti': 1.60,  'V': 1.53,   'Cr': 1.39,  'Mn': 1.39,
    'Fe': 1.32,  'Co': 1.26,  'Ni': 1.24,  'Cu': 1.32,  'Zn': 1.22
}


def _check_dependencies(required: List[str]) -> None:
    """Check if required visualization dependencies are available."""
    missing = []
    
    if 'matplotlib' in required and not MATPLOTLIB_AVAILABLE:
        missing.append('matplotlib')
    if 'plotly' in required and not PLOTLY_AVAILABLE:
        missing.append('plotly')
    if 'seaborn' in required and not SEABORN_AVAILABLE:
        missing.append('seaborn')
    
    if missing:
        raise DependencyError(
            f"Missing required dependencies: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}",
            missing_deps=missing,
            feature="visualization"
        )


def plot_unit_cell_2d(
    crystal: Crystal,
    plane: str = 'xy',
    show_atoms: bool = True,
    show_bonds: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> Any:
    """
    Plot 2D projection of crystal unit cell.
    
    Args:
        crystal: Crystal structure to visualize
        plane: Projection plane ('xy', 'xz', 'yz')
        show_atoms: Whether to show atomic positions
        show_bonds: Whether to show bonds (if available)
        figsize: Figure size as (width, height)
        save_path: Path to save the figure
        
    Returns:
        matplotlib figure object
        
    Raises:
        DependencyError: If matplotlib is not available
        VisualizationError: If visualization fails
    """
    _check_dependencies(['matplotlib'])
    
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get lattice vectors
        lattice = crystal.lattice_parameters
        a, b, c = lattice.a, lattice.b, lattice.c
        alpha, beta, gamma = np.radians([lattice.alpha, lattice.beta, lattice.gamma])
        
        # Calculate unit cell vectors in Cartesian coordinates
        if plane == 'xy':
            # Project onto xy-plane
            v1 = np.array([a, 0])
            v2 = np.array([b * np.cos(gamma), b * np.sin(gamma)])
            xlabel, ylabel = 'X (Å)', 'Y (Å)'
            coord_indices = (0, 1)
        elif plane == 'xz':
            # Project onto xz-plane  
            v1 = np.array([a, 0])
            v2 = np.array([c * np.cos(beta), c * np.sin(beta)])
            xlabel, ylabel = 'X (Å)', 'Z (Å)'
            coord_indices = (0, 2)
        elif plane == 'yz':
            # Project onto yz-plane
            v1 = np.array([b * np.cos(gamma), b * np.sin(gamma)])
            v2 = np.array([c * np.cos(alpha), c * np.sin(alpha)])
            xlabel, ylabel = 'Y (Å)', 'Z (Å)'
            coord_indices = (1, 2)
        else:
            raise VisualizationError(f"Invalid plane '{plane}'. Use 'xy', 'xz', or 'yz'", viz_type="2D unit cell")
        
        # Draw unit cell boundary
        unit_cell = np.array([[0, 0], v1, v1 + v2, v2, [0, 0]])
        ax.plot(unit_cell[:, 0], unit_cell[:, 1], 'k-', linewidth=2, label='Unit Cell')
        
        # Plot atoms if requested
        if show_atoms and crystal.atomic_sites:
            for site in crystal.atomic_sites:
                # Convert fractional to Cartesian coordinates
                frac_coords = np.array([site.x, site.y, site.z])
                cart_coords = crystal._fractional_to_cartesian(frac_coords)
                
                # Project to 2D plane
                x, y = cart_coords[coord_indices[0]], cart_coords[coord_indices[1]]
                
                # Get atomic properties
                color = ATOMIC_COLORS.get(site.element, '#808080')
                radius = ATOMIC_RADII.get(site.element, 1.0) * 100  # Scale for visibility
                
                # Plot atom
                circle = plt.Circle((x, y), radius, color=color, alpha=0.7, 
                                  edgecolor='black', linewidth=0.5)
                ax.add_patch(circle)
                
                # Add element label
                ax.text(x, y, site.element, ha='center', va='center', 
                       fontsize=8, fontweight='bold')
        
        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Crystal Structure - {plane.upper()} Projection\n'
                    f'{crystal.lattice_parameters.crystal_system}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    except Exception as e:
        raise VisualizationError(f"Failed to create 2D unit cell plot: {str(e)}", viz_type="2D unit cell")


def plot_crystal_3d(
    crystal: Crystal,
    show_unit_cell: bool = True,
    show_atoms: bool = True,
    show_bonds: bool = False,
    supercell: Tuple[int, int, int] = (1, 1, 1),
    width: int = 800,
    height: int = 600,
    save_path: Optional[str] = None
) -> Any:
    """
    Create interactive 3D visualization of crystal structure.
    
    Args:
        crystal: Crystal structure to visualize
        show_unit_cell: Whether to show unit cell boundaries
        show_atoms: Whether to show atoms
        show_bonds: Whether to show bonds
        supercell: Supercell dimensions (nx, ny, nz)
        width: Plot width in pixels
        height: Plot height in pixels
        save_path: Path to save as HTML file
        
    Returns:
        plotly figure object
        
    Raises:
        DependencyError: If plotly is not available
        VisualizationError: If visualization fails
    """
    _check_dependencies(['plotly'])
    
    try:
        fig = go.Figure()
        
        # Get lattice vectors in Cartesian coordinates
        lattice = crystal.lattice_parameters
        a, b, c = lattice.a, lattice.b, lattice.c
        alpha, beta, gamma = np.radians([lattice.alpha, lattice.beta, lattice.gamma])
        
        # Calculate Cartesian lattice vectors
        ax = a
        ay = 0
        az = 0
        
        bx = b * np.cos(gamma)
        by = b * np.sin(gamma)
        bz = 0
        
        cx = c * np.cos(beta)
        cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        cz = c * np.sqrt(1 - np.cos(beta)**2 - 
                        ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma))**2)
        
        lattice_vectors = np.array([
            [ax, ay, az],
            [bx, by, bz], 
            [cx, cy, cz]
        ])
        
        # Create supercell if requested
        nx, ny, nz = supercell
        
        if show_unit_cell:
            # Draw unit cell boundaries for each cell in supercell
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        origin = i * lattice_vectors[0] + j * lattice_vectors[1] + k * lattice_vectors[2]
                        
                        # Define unit cell edges
                        edges = [
                            [origin, origin + lattice_vectors[0]],
                            [origin, origin + lattice_vectors[1]],
                            [origin, origin + lattice_vectors[2]],
                            [origin + lattice_vectors[0], origin + lattice_vectors[0] + lattice_vectors[1]],
                            [origin + lattice_vectors[0], origin + lattice_vectors[0] + lattice_vectors[2]],
                            [origin + lattice_vectors[1], origin + lattice_vectors[1] + lattice_vectors[0]],
                            [origin + lattice_vectors[1], origin + lattice_vectors[1] + lattice_vectors[2]],
                            [origin + lattice_vectors[2], origin + lattice_vectors[2] + lattice_vectors[0]],
                            [origin + lattice_vectors[2], origin + lattice_vectors[2] + lattice_vectors[1]],
                            [origin + lattice_vectors[0] + lattice_vectors[1], 
                             origin + lattice_vectors[0] + lattice_vectors[1] + lattice_vectors[2]],
                            [origin + lattice_vectors[0] + lattice_vectors[2],
                             origin + lattice_vectors[0] + lattice_vectors[1] + lattice_vectors[2]],
                            [origin + lattice_vectors[1] + lattice_vectors[2],
                             origin + lattice_vectors[0] + lattice_vectors[1] + lattice_vectors[2]]
                        ]
                        
                        # Add edges to plot
                        for edge in edges:
                            fig.add_trace(go.Scatter3d(
                                x=[edge[0][0], edge[1][0]],
                                y=[edge[0][1], edge[1][1]], 
                                z=[edge[0][2], edge[1][2]],
                                mode='lines',
                                line=dict(color='black', width=2),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
        
        # Plot atoms if requested
        if show_atoms and crystal.atomic_sites:
            # Group atoms by element for legend
            elements = {}
            
            for site in crystal.atomic_sites:
                if site.element not in elements:
                    elements[site.element] = {'x': [], 'y': [], 'z': [], 'text': []}
                
                # Generate positions for all supercell replications
                for i in range(nx):
                    for j in range(ny):
                        for k in range(nz):
                            # Fractional coordinates
                            frac_coords = np.array([site.x, site.y, site.z])
                            
                            # Convert to Cartesian
                            cart_coords = (frac_coords[0] * lattice_vectors[0] + 
                                         frac_coords[1] * lattice_vectors[1] + 
                                         frac_coords[2] * lattice_vectors[2])
                            
                            # Add supercell offset
                            cart_coords += (i * lattice_vectors[0] + 
                                          j * lattice_vectors[1] + 
                                          k * lattice_vectors[2])
                            
                            elements[site.element]['x'].append(cart_coords[0])
                            elements[site.element]['y'].append(cart_coords[1])
                            elements[site.element]['z'].append(cart_coords[2])
                            elements[site.element]['text'].append(
                                f"{site.element}<br>({i},{j},{k})<br>"
                                f"Fractional: ({site.x:.3f}, {site.y:.3f}, {site.z:.3f})"
                            )
            
            # Add atoms to plot by element
            for element, coords in elements.items():
                color = ATOMIC_COLORS.get(element, '#808080')
                size = ATOMIC_RADII.get(element, 1.0) * 20  # Scale for visibility
                
                fig.add_trace(go.Scatter3d(
                    x=coords['x'],
                    y=coords['y'],
                    z=coords['z'],
                    mode='markers',
                    marker=dict(
                        size=size,
                        color=color,
                        opacity=0.8,
                        line=dict(color='black', width=1)
                    ),
                    text=coords['text'],
                    hovertemplate='%{text}<extra></extra>',
                    name=element
                ))
        
        # Update layout
        fig.update_layout(
            title=f'3D Crystal Structure - {crystal.lattice_parameters.crystal_system}',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data'
            ),
            width=width,
            height=height,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
        
    except Exception as e:
        raise VisualizationError(f"Failed to create 3D crystal plot: {str(e)}", viz_type="3D crystal")


def plot_lattice_parameters(
    crystal: Crystal,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> Any:
    """
    Plot lattice parameters as bar charts.
    
    Args:
        crystal: Crystal structure to visualize
        figsize: Figure size as (width, height)
        save_path: Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    _check_dependencies(['matplotlib'])
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        lattice = crystal.lattice_parameters
        
        # Lattice lengths
        lengths = [lattice.a, lattice.b, lattice.c]
        length_labels = ['a', 'b', 'c']
        colors1 = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars1 = ax1.bar(length_labels, lengths, color=colors1, alpha=0.7, edgecolor='black')
        ax1.set_title('Lattice Parameters (Å)')
        ax1.set_ylabel('Length (Å)')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, lengths):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Lattice angles
        angles = [lattice.alpha, lattice.beta, lattice.gamma]
        angle_labels = ['α', 'β', 'γ']
        colors2 = ['#96CEB4', '#FFEAA7', '#DDA0DD']
        
        bars2 = ax2.bar(angle_labels, angles, color=colors2, alpha=0.7, edgecolor='black')
        ax2.set_title('Lattice Angles (°)')
        ax2.set_ylabel('Angle (°)')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, angles):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Crystal system info
        ax3.text(0.5, 0.7, f'Crystal System: {lattice.crystal_system}', 
                ha='center', va='center', fontsize=16, fontweight='bold',
                transform=ax3.transAxes)
        ax3.text(0.5, 0.5, f'Space Group: {lattice.space_group or "Unknown"}',
                ha='center', va='center', fontsize=14,
                transform=ax3.transAxes)
        ax3.text(0.5, 0.3, f'Volume: {crystal.volume:.2f} Å³',
                ha='center', va='center', fontsize=14,
                transform=ax3.transAxes)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # Unit cell dimensions comparison
        if SEABORN_AVAILABLE:
            import seaborn as sns
            ax4.set_style("whitegrid")
        
        # Radar-like comparison (simplified as bar chart)
        params = ['a (Å)', 'b (Å)', 'c (Å)', 'α (°)', 'β (°)', 'γ (°)']
        values = [lattice.a, lattice.b, lattice.c, lattice.alpha/10, lattice.beta/10, lattice.gamma/10]
        colors3 = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        bars3 = ax4.bar(range(len(params)), values, color=colors3, alpha=0.7, edgecolor='black')
        ax4.set_title('Normalized Parameters Overview')
        ax4.set_ylabel('Normalized Value')
        ax4.set_xticks(range(len(params)))
        ax4.set_xticklabels(params, rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    except Exception as e:
        raise VisualizationError(f"Failed to create lattice parameters plot: {str(e)}", viz_type="lattice parameters")


def plot_atomic_positions(
    crystal: Crystal,
    projection: str = '3d',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> Any:
    """
    Plot atomic positions in the unit cell.
    
    Args:
        crystal: Crystal structure to visualize
        projection: '3d' for 3D scatter, '2d' for 2D projections
        figsize: Figure size as (width, height)
        save_path: Path to save the figure
        
    Returns:
        matplotlib figure object
    """
    _check_dependencies(['matplotlib'])
    
    if not crystal.atomic_sites:
        raise VisualizationError("No atomic sites available for visualization", viz_type="atomic positions")
    
    try:
        if projection == '3d':
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Group atoms by element
            elements = {}
            for site in crystal.atomic_sites:
                if site.element not in elements:
                    elements[site.element] = {'x': [], 'y': [], 'z': []}
                elements[site.element]['x'].append(site.x)
                elements[site.element]['y'].append(site.y)
                elements[site.element]['z'].append(site.z)
            
            # Plot atoms by element
            for element, coords in elements.items():
                color = ATOMIC_COLORS.get(element, '#808080')
                size = ATOMIC_RADII.get(element, 1.0) * 100
                
                ax.scatter(coords['x'], coords['y'], coords['z'],
                          c=color, s=size, alpha=0.7, 
                          edgecolors='black', linewidth=0.5,
                          label=element)
            
            ax.set_xlabel('X (fractional)')
            ax.set_ylabel('Y (fractional)')
            ax.set_zlabel('Z (fractional)')
            ax.set_title('Atomic Positions in Unit Cell')
            ax.legend()
            
        else:  # 2D projections
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            
            # Group atoms by element
            elements = {}
            for site in crystal.atomic_sites:
                if site.element not in elements:
                    elements[site.element] = {'x': [], 'y': [], 'z': []}
                elements[site.element]['x'].append(site.x)
                elements[site.element]['y'].append(site.y)
                elements[site.element]['z'].append(site.z)
            
            # XY projection
            for element, coords in elements.items():
                color = ATOMIC_COLORS.get(element, '#808080')
                size = ATOMIC_RADII.get(element, 1.0) * 50
                ax1.scatter(coords['x'], coords['y'], c=color, s=size, 
                           alpha=0.7, edgecolors='black', linewidth=0.5, label=element)
            ax1.set_xlabel('X (fractional)')
            ax1.set_ylabel('Y (fractional)')
            ax1.set_title('XY Projection')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # XZ projection
            for element, coords in elements.items():
                color = ATOMIC_COLORS.get(element, '#808080')
                size = ATOMIC_RADII.get(element, 1.0) * 50
                ax2.scatter(coords['x'], coords['z'], c=color, s=size,
                           alpha=0.7, edgecolors='black', linewidth=0.5, label=element)
            ax2.set_xlabel('X (fractional)')
            ax2.set_ylabel('Z (fractional)')
            ax2.set_title('XZ Projection')
            ax2.grid(True, alpha=0.3)
            
            # YZ projection
            for element, coords in elements.items():
                color = ATOMIC_COLORS.get(element, '#808080')
                size = ATOMIC_RADII.get(element, 1.0) * 50
                ax3.scatter(coords['y'], coords['z'], c=color, s=size,
                           alpha=0.7, edgecolors='black', linewidth=0.5, label=element)
            ax3.set_xlabel('Y (fractional)')
            ax3.set_ylabel('Z (fractional)')
            ax3.set_title('YZ Projection')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    except Exception as e:
        raise VisualizationError(f"Failed to create atomic positions plot: {str(e)}", viz_type="atomic positions")


class CrystalVisualizer:
    """
    Comprehensive crystal structure visualization class.
    
    Provides methods for creating various types of crystal structure
    visualizations including 2D projections, 3D interactive plots,
    and parameter analysis charts.
    """
    
    def __init__(self, crystal: Crystal):
        """
        Initialize visualizer with crystal structure.
        
        Args:
            crystal: Crystal structure to visualize
        """
        self.crystal = crystal
        
    def plot_all(
        self,
        save_dir: Optional[str] = None,
        interactive: bool = True,
        show_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Create all available visualizations.
        
        Args:
            save_dir: Directory to save plots
            interactive: Whether to create interactive plots
            show_plots: Whether to display plots
            
        Returns:
            Dictionary of created figures
        """
        figures = {}
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
        
        try:
            # 2D unit cell projections
            for plane in ['xy', 'xz', 'yz']:
                fig = plot_unit_cell_2d(
                    self.crystal, 
                    plane=plane,
                    save_path=str(save_dir / f'unit_cell_{plane}.png') if save_dir else None
                )
                figures[f'unit_cell_{plane}'] = fig
                if show_plots:
                    plt.show()
            
            # 3D interactive plot
            if interactive and PLOTLY_AVAILABLE:
                fig = plot_crystal_3d(
                    self.crystal,
                    save_path=str(save_dir / 'crystal_3d.html') if save_dir else None
                )
                figures['crystal_3d'] = fig
                if show_plots:
                    fig.show()
            
            # Lattice parameters
            fig = plot_lattice_parameters(
                self.crystal,
                save_path=str(save_dir / 'lattice_parameters.png') if save_dir else None
            )
            figures['lattice_parameters'] = fig
            if show_plots:
                plt.show()
            
            # Atomic positions
            for projection in ['3d', '2d']:
                fig = plot_atomic_positions(
                    self.crystal,
                    projection=projection,
                    save_path=str(save_dir / f'atomic_positions_{projection}.png') if save_dir else None
                )
                figures[f'atomic_positions_{projection}'] = fig
                if show_plots:
                    plt.show()
            
            return figures
            
        except Exception as e:
            raise VisualizationError(f"Failed to create all visualizations: {str(e)}", viz_type="batch visualization")
    
    def summary_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a text summary of crystal properties.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Summary report as string
        """
        lattice = self.crystal.lattice_parameters
        
        report = f"""
CRYSTAL STRUCTURE ANALYSIS REPORT
================================

Crystal System: {lattice.crystal_system}
Space Group: {lattice.space_group or 'Unknown'}

Lattice Parameters:
  a = {lattice.a:.4f} Å
  b = {lattice.b:.4f} Å  
  c = {lattice.c:.4f} Å
  α = {lattice.alpha:.2f}°
  β = {lattice.beta:.2f}°
  γ = {lattice.gamma:.2f}°

Unit Cell Volume: {self.crystal.volume:.4f} Å³

Atomic Sites: {len(self.crystal.atomic_sites) if self.crystal.atomic_sites else 0}
"""
        
        if self.crystal.atomic_sites:
            # Element composition
            elements = {}
            for site in self.crystal.atomic_sites:
                elements[site.element] = elements.get(site.element, 0) + 1
            
            report += "\nElement Composition:\n"
            for element, count in sorted(elements.items()):
                percentage = (count / len(self.crystal.atomic_sites)) * 100
                report += f"  {element}: {count} atoms ({percentage:.1f}%)\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report