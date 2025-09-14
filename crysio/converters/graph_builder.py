"""
Graph builder module for converting crystal structures to PyTorch Geometric graphs.

This module provides functionality to convert crystal structures into graph representations
suitable for Graph Neural Networks, particularly using PyTorch Geometric.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass

try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints when PyTorch is not available
    class Data:
        pass

from ..core.crystal import Crystal
from ..utils.exceptions import GraphBuildingError, ConversionError


@dataclass
class GraphConfig:
    """Configuration for graph building parameters."""
    
    # Edge construction parameters
    cutoff_radius: float = 5.0  # Maximum distance for edge creation (Å)
    max_neighbors: Optional[int] = 12  # Maximum number of neighbors per node
    self_loops: bool = False  # Include self-connections
    
    # Node feature parameters
    use_atomic_number: bool = True
    use_electronegativity: bool = True
    use_atomic_radius: bool = True
    use_coordination_number: bool = True
    use_oxidation_state: bool = False
    
    # Edge feature parameters
    use_distance: bool = True
    use_bond_angles: bool = False
    use_relative_positions: bool = True
    
    # Graph construction options
    periodic_boundary: bool = True  # Consider periodic boundary conditions
    supercell_expansion: Tuple[int, int, int] = (1, 1, 1)  # Expand unit cell for better connectivity


class GraphBuilder:
    """
    Convert crystal structures to PyTorch Geometric graphs.
    
    This class handles the conversion of crystal structures into graph representations
    suitable for Graph Neural Networks. It supports various node and edge features
    and handles periodic boundary conditions.
    
    Attributes:
        config (GraphConfig): Configuration for graph building
        atomic_properties (Dict): Database of atomic properties
        
    Examples:
        >>> builder = GraphBuilder()
        >>> graph = builder.build_graph(crystal_structure)
        >>> print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
        
        >>> # Custom configuration
        >>> config = GraphConfig(cutoff_radius=6.0, use_bond_angles=True)
        >>> builder = GraphBuilder(config)
        >>> graph = builder.build_graph(crystal_structure)
    """
    
    def __init__(self, config: Optional[GraphConfig] = None):
        """
        Initialize GraphBuilder with configuration.
        
        Args:
            config: GraphConfig object with building parameters
            
        Raises:
            ImportError: If PyTorch Geometric is not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and PyTorch Geometric are required for graph building. "
                "Install with: pip install torch torch-geometric"
            )
        
        self.config = config or GraphConfig()
        self.atomic_properties = self._load_atomic_properties()
    
    def build_graph(self, crystal: Crystal, **kwargs) -> Data:
        """
        Convert crystal structure to PyTorch Geometric graph.
        
        Args:
            crystal: Crystal structure to convert
            **kwargs: Override configuration parameters
            
        Returns:
            Data: PyTorch Geometric data object
            
        Raises:
            GraphBuildingError: If graph construction fails
        """
        try:
            # Update config with any kwargs
            config = self._update_config(**kwargs)
            
            # Expand supercell if needed
            if config.supercell_expansion != (1, 1, 1):
                crystal = crystal.supercell(*config.supercell_expansion)
            
            # Get atomic positions and types
            positions = self._get_atomic_positions(crystal)
            node_features = self._build_node_features(crystal, config)
            
            # Build edges and edge features
            edge_index, edge_attr = self._build_edges(crystal, positions, config)
            
            # Create PyTorch Geometric data object
            graph_data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=positions,
                # Additional metadata
                formula=crystal.formula,
                space_group=crystal.space_group,
                lattice_params=torch.tensor([
                    crystal.lattice.a, crystal.lattice.b, crystal.lattice.c,
                    crystal.lattice.alpha, crystal.lattice.beta, crystal.lattice.gamma
                ], dtype=torch.float32),
                volume=torch.tensor([crystal.volume], dtype=torch.float32),
                num_atoms=torch.tensor([crystal.num_atoms], dtype=torch.long)
            )
            
            return graph_data
            
        except Exception as e:
            raise GraphBuildingError(f"Failed to build graph: {str(e)}", crystal.formula)
    
    def _get_atomic_positions(self, crystal: Crystal) -> torch.Tensor:
        """Get atomic positions as tensor."""
        cartesian_coords = crystal.to_cartesian_coordinates()
        positions = torch.tensor(cartesian_coords, dtype=torch.float32)
        return positions
    
    def _build_node_features(self, crystal: Crystal, config: GraphConfig) -> torch.Tensor:
        """
        Build node features for each atom.
        
        Args:
            crystal: Crystal structure
            config: Graph configuration
            
        Returns:
            torch.Tensor: Node features [num_nodes, num_features]
        """
        features = []
        
        for site in crystal.sites:
            atom_features = []
            element = site.element
            
            # Atomic number (essential for most applications)
            if config.use_atomic_number:
                atomic_num = self.atomic_properties.get(element, {}).get('atomic_number', 0)
                atom_features.append(atomic_num)
            
            # Electronegativity
            if config.use_electronegativity:
                electronegativity = self.atomic_properties.get(element, {}).get('electronegativity', 0.0)
                atom_features.append(electronegativity)
            
            # Atomic radius
            if config.use_atomic_radius:
                atomic_radius = self.atomic_properties.get(element, {}).get('atomic_radius', 0.0)
                atom_features.append(atomic_radius)
            
            # Coordination number (calculated)
            if config.use_coordination_number:
                coord_num = self._calculate_coordination_number(crystal, site, config.cutoff_radius)
                atom_features.append(coord_num)
            
            # Oxidation state (if available)
            if config.use_oxidation_state and site.oxidation_state is not None:
                atom_features.append(site.oxidation_state)
            elif config.use_oxidation_state:
                atom_features.append(0.0)  # Default value
            
            features.append(atom_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _build_edges(self, crystal: Crystal, positions: torch.Tensor, config: GraphConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edge indices and edge features.
        
        Args:
            crystal: Crystal structure
            positions: Atomic positions
            config: Graph configuration
            
        Returns:
            Tuple of (edge_index, edge_attr)
        """
        edge_indices = []
        edge_features = []
        
        num_atoms = len(crystal.sites)
        
        # Calculate distance matrix
        if config.periodic_boundary:
            distances, edge_pairs = self._calculate_periodic_distances(crystal, positions, config.cutoff_radius)
        else:
            distances, edge_pairs = self._calculate_distances(positions, config.cutoff_radius)
        
        # Build edges within cutoff radius
        for (i, j), distance in zip(edge_pairs, distances):
            if i == j and not config.self_loops:
                continue
            
            edge_indices.append([i, j])
            
            # Edge features
            edge_feat = []
            
            if config.use_distance:
                edge_feat.append(distance)
            
            if config.use_relative_positions:
                relative_pos = positions[j] - positions[i]
                edge_feat.extend(relative_pos.tolist())
            
            edge_features.append(edge_feat)
        
        # Limit number of neighbors if specified
        if config.max_neighbors is not None:
            edge_indices, edge_features = self._limit_neighbors(edge_indices, edge_features, config.max_neighbors)
        
        # Convert to tensors
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(edge_features[0]) if edge_features else 1), dtype=torch.float32)
        
        return edge_index, edge_attr
    
    def _calculate_periodic_distances(self, crystal: Crystal, positions: torch.Tensor, cutoff: float) -> Tuple[List[float], List[Tuple[int, int]]]:
        """
        Calculate distances considering periodic boundary conditions.
        
        Args:
            crystal: Crystal structure
            positions: Atomic positions
            cutoff: Distance cutoff
            
        Returns:
            Tuple of (distances, edge_pairs)
        """
        distances = []
        edge_pairs = []
        
        lattice_matrix = torch.tensor(crystal.lattice.lattice_matrix(), dtype=torch.float32)
        num_atoms = positions.shape[0]
        
        # Generate neighboring unit cells to check
        cell_offsets = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    cell_offsets.append([i, j, k])
        
        for atom_i in range(num_atoms):
            for atom_j in range(num_atoms):
                min_distance = float('inf')
                
                # Check all periodic images
                for offset in cell_offsets:
                    offset_vector = torch.tensor(offset, dtype=torch.float32) @ lattice_matrix
                    translated_pos = positions[atom_j] + offset_vector
                    distance = torch.norm(positions[atom_i] - translated_pos).item()
                    
                    if distance < min_distance:
                        min_distance = distance
                
                if min_distance <= cutoff:
                    distances.append(min_distance)
                    edge_pairs.append((atom_i, atom_j))
        
        return distances, edge_pairs
    
    def _calculate_distances(self, positions: torch.Tensor, cutoff: float) -> Tuple[List[float], List[Tuple[int, int]]]:
        """
        Calculate distances without periodic boundary conditions.
        
        Args:
            positions: Atomic positions
            cutoff: Distance cutoff
            
        Returns:
            Tuple of (distances, edge_pairs)
        """
        distances = []
        edge_pairs = []
        
        num_atoms = positions.shape[0]
        
        for i in range(num_atoms):
            for j in range(num_atoms):
                distance = torch.norm(positions[i] - positions[j]).item()
                
                if distance <= cutoff:
                    distances.append(distance)
                    edge_pairs.append((i, j))
        
        return distances, edge_pairs
    
    def _calculate_coordination_number(self, crystal: Crystal, site, cutoff: float) -> float:
        """Calculate coordination number for an atomic site."""
        site_index = crystal.sites.index(site)
        positions = self._get_atomic_positions(crystal)
        
        distances, edge_pairs = self._calculate_periodic_distances(crystal, positions, cutoff)
        
        coord_num = 0
        for (i, j), distance in zip(edge_pairs, distances):
            if i == site_index and j != site_index:
                coord_num += 1
        
        return float(coord_num)
    
    def _limit_neighbors(self, edge_indices: List, edge_features: List, max_neighbors: int) -> Tuple[List, List]:
        """Limit number of neighbors per node."""
        # Group edges by source node
        node_edges = {}
        for idx, (i, j) in enumerate(edge_indices):
            if i not in node_edges:
                node_edges[i] = []
            node_edges[i].append((idx, j))
        
        # Keep only closest neighbors
        limited_edge_indices = []
        limited_edge_features = []
        
        for node, edges in node_edges.items():
            # Sort by distance (assuming distance is first edge feature)
            edges.sort(key=lambda x: edge_features[x[0]][0] if edge_features[x[0]] else 0)
            
            # Keep top max_neighbors
            for idx, (edge_idx, target) in enumerate(edges[:max_neighbors]):
                limited_edge_indices.append([node, target])
                limited_edge_features.append(edge_features[edge_idx])
        
        return limited_edge_indices, limited_edge_features
    
    def _update_config(self, **kwargs) -> GraphConfig:
        """Update configuration with keyword arguments."""
        config_dict = {
            'cutoff_radius': kwargs.get('cutoff_radius', self.config.cutoff_radius),
            'max_neighbors': kwargs.get('max_neighbors', self.config.max_neighbors),
            'self_loops': kwargs.get('self_loops', self.config.self_loops),
            'use_atomic_number': kwargs.get('use_atomic_number', self.config.use_atomic_number),
            'use_electronegativity': kwargs.get('use_electronegativity', self.config.use_electronegativity),
            'use_atomic_radius': kwargs.get('use_atomic_radius', self.config.use_atomic_radius),
            'use_coordination_number': kwargs.get('use_coordination_number', self.config.use_coordination_number),
            'use_oxidation_state': kwargs.get('use_oxidation_state', self.config.use_oxidation_state),
            'use_distance': kwargs.get('use_distance', self.config.use_distance),
            'use_bond_angles': kwargs.get('use_bond_angles', self.config.use_bond_angles),
            'use_relative_positions': kwargs.get('use_relative_positions', self.config.use_relative_positions),
            'periodic_boundary': kwargs.get('periodic_boundary', self.config.periodic_boundary),
            'supercell_expansion': kwargs.get('supercell_expansion', self.config.supercell_expansion),
        }
        return GraphConfig(**config_dict)
    
    def _load_atomic_properties(self) -> Dict[str, Dict[str, float]]:
        """Load atomic properties database."""
        # Basic atomic properties for common elements
        properties = {
            'H': {'atomic_number': 1, 'electronegativity': 2.20, 'atomic_radius': 0.37},
            'He': {'atomic_number': 2, 'electronegativity': 0.00, 'atomic_radius': 0.32},
            'Li': {'atomic_number': 3, 'electronegativity': 0.98, 'atomic_radius': 1.34},
            'Be': {'atomic_number': 4, 'electronegativity': 1.57, 'atomic_radius': 0.90},
            'B': {'atomic_number': 5, 'electronegativity': 2.04, 'atomic_radius': 0.82},
            'C': {'atomic_number': 6, 'electronegativity': 2.55, 'atomic_radius': 0.77},
            'N': {'atomic_number': 7, 'electronegativity': 3.04, 'atomic_radius': 0.75},
            'O': {'atomic_number': 8, 'electronegativity': 3.44, 'atomic_radius': 0.73},
            'F': {'atomic_number': 9, 'electronegativity': 3.98, 'atomic_radius': 0.71},
            'Ne': {'atomic_number': 10, 'electronegativity': 0.00, 'atomic_radius': 0.69},
            'Na': {'atomic_number': 11, 'electronegativity': 0.93, 'atomic_radius': 1.54},
            'Mg': {'atomic_number': 12, 'electronegativity': 1.31, 'atomic_radius': 1.30},
            'Al': {'atomic_number': 13, 'electronegativity': 1.61, 'atomic_radius': 1.18},
            'Si': {'atomic_number': 14, 'electronegativity': 1.90, 'atomic_radius': 1.11},
            'P': {'atomic_number': 15, 'electronegativity': 2.19, 'atomic_radius': 1.06},
            'S': {'atomic_number': 16, 'electronegativity': 2.58, 'atomic_radius': 1.02},
            'Cl': {'atomic_number': 17, 'electronegativity': 3.16, 'atomic_radius': 0.99},
            'Ar': {'atomic_number': 18, 'electronegativity': 0.00, 'atomic_radius': 0.97},
            'K': {'atomic_number': 19, 'electronegativity': 0.82, 'atomic_radius': 1.96},
            'Ca': {'atomic_number': 20, 'electronegativity': 1.00, 'atomic_radius': 1.74},
            'Sc': {'atomic_number': 21, 'electronegativity': 1.36, 'atomic_radius': 1.44},
            'Ti': {'atomic_number': 22, 'electronegativity': 1.54, 'atomic_radius': 1.36},
            'V': {'atomic_number': 23, 'electronegativity': 1.63, 'atomic_radius': 1.25},
            'Cr': {'atomic_number': 24, 'electronegativity': 1.66, 'atomic_radius': 1.27},
            'Mn': {'atomic_number': 25, 'electronegativity': 1.55, 'atomic_radius': 1.39},
            'Fe': {'atomic_number': 26, 'electronegativity': 1.83, 'atomic_radius': 1.25},
            'Co': {'atomic_number': 27, 'electronegativity': 1.88, 'atomic_radius': 1.26},
            'Ni': {'atomic_number': 28, 'electronegativity': 1.91, 'atomic_radius': 1.21},
            'Cu': {'atomic_number': 29, 'electronegativity': 1.90, 'atomic_radius': 1.38},
            'Zn': {'atomic_number': 30, 'electronegativity': 1.65, 'atomic_radius': 1.31},
            'Ga': {'atomic_number': 31, 'electronegativity': 1.81, 'atomic_radius': 1.26},
            'Ge': {'atomic_number': 32, 'electronegativity': 2.01, 'atomic_radius': 1.22},
            'As': {'atomic_number': 33, 'electronegativity': 2.18, 'atomic_radius': 1.19},
            'Se': {'atomic_number': 34, 'electronegativity': 2.55, 'atomic_radius': 1.16},
            'Br': {'atomic_number': 35, 'electronegativity': 2.96, 'atomic_radius': 1.14},
            'Kr': {'atomic_number': 36, 'electronegativity': 0.00, 'atomic_radius': 1.10},
        }
        
        return properties
    
    def get_graph_statistics(self, graph_data: Data) -> Dict[str, Any]:
        """
        Get statistics about the generated graph.
        
        Args:
            graph_data: PyTorch Geometric data object
            
        Returns:
            Dict with graph statistics
        """
        stats = {
            'num_nodes': graph_data.num_nodes,
            'num_edges': graph_data.num_edges,
            'num_node_features': graph_data.num_node_features,
            'num_edge_features': graph_data.num_edge_features if graph_data.edge_attr is not None else 0,
            'average_degree': graph_data.num_edges / graph_data.num_nodes if graph_data.num_nodes > 0 else 0,
            'formula': graph_data.formula if hasattr(graph_data, 'formula') else 'Unknown',
            'space_group': graph_data.space_group if hasattr(graph_data, 'space_group') else 'Unknown',
        }
        
        if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
            # Distance statistics (assuming first edge feature is distance)
            distances = graph_data.edge_attr[:, 0]
            stats.update({
                'min_distance': float(torch.min(distances)),
                'max_distance': float(torch.max(distances)),
                'mean_distance': float(torch.mean(distances)),
            })
        
        return stats