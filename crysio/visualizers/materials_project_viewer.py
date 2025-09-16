"""
Materials Project-Style Interactive Crystal Viewer - FIXED VERSION
Compatible with actual Crysio v0.3.1 Crystal class structure
"""

import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple
import warnings

# Import dependencies with graceful fallbacks
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")

try:
    import ipywidgets as widgets
    from IPython.display import display, HTML
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

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
    
    # Volume calculation for triclinic system
    volume = a * b * c * np.sqrt(1 + 2*cos_alpha*cos_beta*cos_gamma - 
                                 cos_alpha**2 - cos_beta**2 - cos_gamma**2)
    
    # Lattice vectors as columns
    lattice_matrix = np.array([
        [a, b * cos_gamma, c * cos_beta],
        [0, b * sin_gamma, c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma],
        [0, 0, volume / (a * b * sin_gamma)]
    ])
    
    return lattice_matrix


def get_crystal_elements(crystal):
    """Extract unique elements from crystal structure."""
    try:
        return crystal.get_elements()
    except AttributeError:
        return list(set(site.element for site in crystal.atomic_sites))


def get_crystal_system(lattice_params):
    """Determine crystal system from lattice parameters."""
    a, b, c = lattice_params.a, lattice_params.b, lattice_params.c
    alpha, beta, gamma = lattice_params.alpha, lattice_params.beta, lattice_params.gamma
    
    tol = 1e-5
    
    if (abs(a - b) < tol and abs(b - c) < tol and 
        abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol):
        return "cubic"
    elif (abs(a - b) < tol and abs(alpha - 90) < tol and 
          abs(beta - 90) < tol and abs(gamma - 90) < tol):
        return "tetragonal"
    elif (abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol):
        return "orthorhombic"
    elif (abs(a - b) < tol and abs(alpha - 90) < tol and 
          abs(beta - 90) < tol and abs(gamma - 120) < tol):
        return "hexagonal"
    elif (abs(a - b) < tol and abs(b - c) < tol and 
          abs(alpha - beta) < tol and abs(beta - gamma) < tol):
        return "trigonal"
    elif (abs(alpha - 90) < tol and abs(gamma - 90) < tol):
        return "monoclinic"
    else:
        return "triclinic"


class MaterialsProjectViewer:
    """Materials Project-style interactive crystal structure viewer."""
    
    def __init__(self, crystal):
        """Initialize Materials Project-style viewer."""
        if not PLOTLY_AVAILABLE:
            raise DependencyError(
                "Plotly required for MaterialsProjectViewer. "
                "Install with: pip install plotly>=5.0"
            )
        
        if crystal is None:
            raise ValueError("Crystal object cannot be None")
        
        self.crystal = crystal
        self.bonding_algorithm = 'distance_cutoff'
        self.color_scheme = 'materials_project'
        self.show_unit_cell = True
        self.show_bonds = True
        self.show_labels = False
        self.supercell_size = (1, 1, 1)
        
        # Materials Project color scheme
        self.mp_colors = {
            'H': '#FFFFFF',   'He': '#D9FFFF',  'Li': '#CC80FF',  'Be': '#C2FF00',
            'B': '#FFB5B5',   'C': '#909090',   'N': '#3050F8',   'O': '#FF0D0D',
            'F': '#90E050',   'Ne': '#B3E3F5',  'Na': '#AB5CF2',  'Mg': '#8AFF00',
            'Al': '#BFA6A6',  'Si': '#F0C8A0',  'P': '#FF8000',   'S': '#FFFF30',
            'Cl': '#1FF01F',  'Ar': '#80D1E3',  'K': '#8F40D4',   'Ca': '#3DFF00',
            'Sc': '#E6E6E6',  'Ti': '#BFC2C7',  'V': '#A6A6AB',   'Cr': '#8A99C7',
            'Mn': '#9C7AC7',  'Fe': '#E06633',  'Co': '#F090A0',  'Ni': '#50D050',
            'Cu': '#C88033',  'Zn': '#7D80B0',  'Ga': '#C28F8F',  'Ge': '#668F8F',
            'Ce': '#FFFFC7',  'Pr': '#D9FFC7',  'Nd': '#C7FFC7'
        }
        
        # MP-style atomic radii scaling
        self.mp_radii_scale = {
            'H': 0.6,   'Li': 1.8,  'C': 1.0,   'N': 0.9,   'O': 0.8,
            'F': 0.7,   'Na': 2.2,  'Mg': 1.6,  'Al': 1.4,  'Si': 1.3,
            'P': 1.2,   'S': 1.2,   'Cl': 1.2,  'K': 2.8,   'Ca': 2.0,
            'Fe': 1.4,  'Co': 1.4,  'Ni': 1.4,  'Cu': 1.4,  'Zn': 1.3,
            'Ga': 1.3,  'Ge': 1.2,  'Ce': 1.8,  'Pr': 1.8,  'Nd': 1.8
        }
    
    def get_mp_color(self, element: str) -> str:
        """Get Materials Project-style color for element."""
        return self.mp_colors.get(element, '#808080')
    
    def get_mp_radius(self, element: str) -> float:
        """Get Materials Project-style radius scaling for element."""
        return self.mp_radii_scale.get(element, 1.0)
    
    def create_mp_style_plot(self) -> go.Figure:
        """Create Materials Project-style 3D crystal structure plot."""
        fig = go.Figure()
        
        # FIXED: Use lattice_parameters instead of lattice
        lattice_matrix = create_lattice_matrix(self.crystal.lattice_parameters)
        nx, ny, nz = self.supercell_size
        
        # Generate supercell atoms
        atom_data = self._generate_supercell_atoms(lattice_matrix, (nx, ny, nz))
        
        # Add atoms as 3D scatter
        if atom_data['x']:
            fig.add_trace(go.Scatter3d(
                x=atom_data['x'],
                y=atom_data['y'], 
                z=atom_data['z'],
                mode='markers',
                marker=dict(
                    size=[r * 15 for r in atom_data['radii']],
                    color=atom_data['colors'],
                    line=dict(color='black', width=1),
                    opacity=0.9
                ),
                text=atom_data['hover_text'],
                hovertemplate='<b>%{text}</b><extra></extra>',
                name='Atoms',
                showlegend=False
            ))
        
        # Add bonds if enabled
        if self.show_bonds:
            self._add_bonds_to_plot(fig, lattice_matrix)
        
        # Add unit cell if enabled
        if self.show_unit_cell:
            self._add_unit_cell_to_plot(fig, lattice_matrix)
        
        # Apply MP layout
        self._apply_mp_layout(fig)
        
        return fig
    
    def _generate_supercell_atoms(self, lattice_matrix: np.ndarray, 
                                 supercell: Tuple[int, int, int]) -> Dict:
        """Generate atomic data for supercell visualization."""
        nx, ny, nz = supercell
        atom_data = {
            'x': [], 'y': [], 'z': [],
            'elements': [], 'colors': [], 'radii': [], 'hover_text': []
        }
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for idx, site in enumerate(self.crystal.atomic_sites):
                        # FIXED: Use site.position instead of site.fractional_coords
                        frac_coords = np.array(site.position) + np.array([i, j, k])
                        cart_pos = frac_coords @ lattice_matrix
                        
                        atom_data['x'].append(cart_pos[0])
                        atom_data['y'].append(cart_pos[1])
                        atom_data['z'].append(cart_pos[2])
                        atom_data['elements'].append(site.element)
                        atom_data['colors'].append(self.get_mp_color(site.element))
                        atom_data['radii'].append(self.get_mp_radius(site.element))
                        
                        # Create hover text
                        hover_info = [
                            f"{site.element}{idx+1}",
                            f"Position: ({cart_pos[0]:.2f}, {cart_pos[1]:.2f}, {cart_pos[2]:.2f})",
                            f"Fractional: ({site.position[0]:.3f}, {site.position[1]:.3f}, {site.position[2]:.3f})",
                            f"Cell: ({i}, {j}, {k})"
                        ]
                        atom_data['hover_text'].append('<br>'.join(hover_info))
        
        return atom_data
    
    def _add_bonds_to_plot(self, fig: go.Figure, lattice_matrix: np.ndarray):
        """Add bonds using distance-based algorithm."""
        bond_cutoffs = {
            'H': 1.2, 'C': 1.8, 'N': 1.8, 'O': 1.8, 'F': 1.8,
            'Na': 2.8, 'Mg': 2.4, 'Al': 2.6, 'Si': 2.4, 'P': 2.4,
            'S': 2.4, 'Cl': 2.4, 'K': 3.2, 'Ca': 2.8, 'Fe': 2.6,
            'Co': 2.5, 'Ni': 2.4, 'Cu': 2.6, 'Zn': 2.5, 'Ga': 2.5,
            'Ge': 2.4, 'Ce': 3.0, 'Pr': 3.0, 'Nd': 3.0
        }
        
        nx, ny, nz = self.supercell_size
        positions = []
        elements = []
        
        # Collect all atomic positions
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for site in self.crystal.atomic_sites:
                        frac_coords = np.array(site.position) + np.array([i, j, k])
                        cart_pos = frac_coords @ lattice_matrix
                        positions.append(cart_pos)
                        elements.append(site.element)
        
        # Find bonds
        bond_lines = {'x': [], 'y': [], 'z': []}
        
        for i, (pos1, elem1) in enumerate(zip(positions, elements)):
            for j, (pos2, elem2) in enumerate(zip(positions, elements)):
                if i < j:
                    distance = np.linalg.norm(pos1 - pos2)
                    max_cutoff = max(
                        bond_cutoffs.get(elem1, 2.5),
                        bond_cutoffs.get(elem2, 2.5)
                    )
                    
                    if distance < max_cutoff:
                        bond_lines['x'].extend([pos1[0], pos2[0], None])
                        bond_lines['y'].extend([pos1[1], pos2[1], None])
                        bond_lines['z'].extend([pos1[2], pos2[2], None])
        
        # Add bond lines to plot
        if bond_lines['x']:
            fig.add_trace(go.Scatter3d(
                x=bond_lines['x'],
                y=bond_lines['y'],
                z=bond_lines['z'],
                mode='lines',
                line=dict(color='gray', width=3),
                hoverinfo='skip',
                name='Bonds',
                showlegend=False
            ))
    
    def _add_unit_cell_to_plot(self, fig: go.Figure, lattice_matrix: np.ndarray):
        """Add unit cell boundaries."""
        nx, ny, nz = self.supercell_size
        
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    if i < nx or j < ny or k < nz:
                        self._add_cell_edges(fig, np.array([i, j, k]), lattice_matrix)
    
    def _add_cell_edges(self, fig: go.Figure, cell_origin: np.ndarray, 
                       lattice_matrix: np.ndarray):
        """Add unit cell edges for a specific cell."""
        origin = cell_origin @ lattice_matrix
        a, b, c = lattice_matrix[:, 0], lattice_matrix[:, 1], lattice_matrix[:, 2]
        
        # Unit cell vertices
        vertices = [
            origin,                    # 0: (0,0,0)
            origin + a,               # 1: (1,0,0)
            origin + a + b,           # 2: (1,1,0)
            origin + b,               # 3: (0,1,0)
            origin + c,               # 4: (0,0,1)
            origin + a + c,           # 5: (1,0,1)
            origin + a + b + c,       # 6: (1,1,1)
            origin + b + c            # 7: (0,1,1)
        ]
        
        # Unit cell edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        # Add each edge
        for edge in edges:
            start_vertex, end_vertex = vertices[edge[0]], vertices[edge[1]]
            fig.add_trace(go.Scatter3d(
                x=[start_vertex[0], end_vertex[0]],
                y=[start_vertex[1], end_vertex[1]],
                z=[start_vertex[2], end_vertex[2]],
                mode='lines',
                line=dict(color='black', width=2),
                hoverinfo='skip',
                showlegend=False,
                name='Unit Cell'
            ))
    
    def _apply_mp_layout(self, fig: go.Figure):
        """Apply Materials Project-style layout."""
        crystal_system = get_crystal_system(self.crystal.lattice_parameters)
        
        atom_data = self._generate_supercell_atoms(
            create_lattice_matrix(self.crystal.lattice_parameters), 
            self.supercell_size
        )
        
        # Calculate center and range
        if atom_data['x']:
            x_center = (max(atom_data['x']) + min(atom_data['x'])) / 2
            y_center = (max(atom_data['y']) + min(atom_data['y'])) / 2
            z_center = (max(atom_data['z']) + min(atom_data['z'])) / 2
            
            range_x = max(atom_data['x']) - min(atom_data['x'])
            range_y = max(atom_data['y']) - min(atom_data['y'])
            range_z = max(atom_data['z']) - min(atom_data['z'])
            max_range = max(range_x, range_y, range_z)
        else:
            x_center = y_center = z_center = 0
            max_range = 10
        
        # Apply MP-style layout
        fig.update_layout(
            scene=dict(
                bgcolor='white',
                xaxis=dict(visible=False, showgrid=False, zeroline=False),
                yaxis=dict(visible=False, showgrid=False, zeroline=False),
                zaxis=dict(visible=False, showgrid=False, zeroline=False),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='cube'
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            title=dict(
                text=f"{self.crystal.formula} ({crystal_system})",
                x=0.5,
                font=dict(size=16, color='#333333')
            ),
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
    
    def create_property_panel(self) -> str:
        """Create Materials Project-style property panel."""
        crystal = self.crystal
        
        num_atoms = len(crystal.atomic_sites)
        elements = get_crystal_elements(crystal)
        crystal_system = get_crystal_system(crystal.lattice_parameters)
        
        # Generate HTML panel
        html = f"""
        <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; 
                    padding: 20px; margin: 10px 0; font-family: -apple-system, BlinkMacSystemFont, 
                    'Segoe UI', Roboto, sans-serif; font-size: 14px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05); max-width: 400px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        margin: -20px -20px 20px -20px; padding: 15px 20px; 
                        border-radius: 8px 8px 0 0; color: white;">
                <h2 style="margin: 0; font-size: 20px; font-weight: 600;">{crystal.formula}</h2>
                <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 12px;">
                    {crystal_system.title()} Crystal System
                </p>
            </div>
            
            <div style="display: grid; gap: 12px;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <div style="background: white; padding: 12px; border-radius: 6px; border: 1px solid #e9ecef;">
                        <div style="font-weight: 600; color: #495057; font-size: 12px; margin-bottom: 4px;">VOLUME</div>
                        <div style="font-size: 16px; color: #212529;">{crystal.volume:.2f} Ų</div>
                    </div>
                    <div style="background: white; padding: 12px; border-radius: 6px; border: 1px solid #e9ecef;">
                        <div style="font-weight: 600; color: #495057; font-size: 12px; margin-bottom: 4px;">DENSITY</div>
                        <div style="font-size: 16px; color: #212529;">{crystal.density:.2f} g/cm³</div>
                    </div>
                </div>
                
                <div style="background: white; padding: 12px; border-radius: 6px; border: 1px solid #e9ecef;">
                    <div style="font-weight: 600; color: #495057; font-size: 12px; margin-bottom: 8px;">LATTICE PARAMETERS</div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 13px;">
                        <div>a = {crystal.lattice_parameters.a:.3f} Å</div>
                        <div>α = {crystal.lattice_parameters.alpha:.1f}°</div>
                        <div>b = {crystal.lattice_parameters.b:.3f} Å</div>
                        <div>β = {crystal.lattice_parameters.beta:.1f}°</div>
                        <div>c = {crystal.lattice_parameters.c:.3f} Å</div>
                        <div>γ = {crystal.lattice_parameters.gamma:.1f}°</div>
                    </div>
                </div>
                
                <div style="background: white; padding: 12px; border-radius: 6px; border: 1px solid #e9ecef;">
                    <div style="font-weight: 600; color: #495057; font-size: 12px; margin-bottom: 8px;">COMPOSITION</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                        {self._generate_element_badges(elements)}
                    </div>
                    <div style="margin-top: 8px; font-size: 13px; color: #6c757d;">
                        {num_atoms} atoms in unit cell
                    </div>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_element_badges(self, elements: List[str]) -> str:
        """Generate colored element badges."""
        badges = []
        for element in elements:
            color = self.get_mp_color(element)
            badges.append(f"""
                <span style="background: {color}; 
                            color: {'white' if element in ['N', 'C'] else 'black'};
                            padding: 4px 8px; border-radius: 12px; font-size: 12px; 
                            font-weight: 600; display: inline-block;">{element}</span>
            """)
        return ''.join(badges)
    
    def show_interactive(self, supercell: Tuple[int, int, int] = (1, 1, 1), 
                        controls: bool = True) -> None:
        """Display Materials Project-style interactive viewer."""
        self.supercell_size = supercell
        
        # Display property panel
        if JUPYTER_AVAILABLE:
            property_html = self.create_property_panel()
            display(HTML(property_html))
        
        # Create and display plot
        fig = self.create_mp_style_plot()
        fig.show()


# Convenience functions
def plot_mp_style(crystal, **kwargs) -> go.Figure:
    """Quick Materials Project-style plot."""
    viewer = MaterialsProjectViewer(crystal)
    return viewer.create_mp_style_plot()


def show_mp_interactive(crystal, **kwargs) -> None:
    """Quick interactive Materials Project-style viewer."""
    viewer = MaterialsProjectViewer(crystal)
    viewer.show_interactive(**kwargs)