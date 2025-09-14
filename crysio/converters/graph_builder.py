"""
Graph conversion module for crystal structures.

This module converts crystal structures to PyTorch Geometric graphs for use in
Graph Neural Networks (GNNs) and other graph-based machine learning applications.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from ..core.crystal import Crystal
from ..utils.exceptions import ConversionError

# Check for PyTorch and PyTorch Geometric availability
try:
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints
    class Data:
        pass
    torch = None
    torch_geometric = None


class GraphBuildingError(ConversionError):
    """Exception raised during graph building process."""
    pass


@dataclass
class GraphConfig:
    """Configuration for graph building."""
    cutoff_radius: float = 5.0  # Angstrom
    max_neighbors: int = 12
    include_edge_features: bool = True
    include_node_features: bool = True
    periodic_boundary: bool = True
    

class GraphBuilder:
    """
    Convert crystal structures to PyTorch Geometric graphs.
    
    This class provides functionality to convert Crystal objects into graph
    representations suitable for Graph Neural Networks.
    
    Examples:
        >>> builder = GraphBuilder()
        >>> graph = builder.build(crystal_structure)
        >>> print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
    """
    
    def __init__(self, config: Optional[GraphConfig] = None):
        """
        Initialize GraphBuilder.
        
        Args:
            config: Configuration for graph building parameters
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and PyTorch Geometric are required for graph conversion. "
                "Install with: pip install torch torch-geometric"
            )
        
        self.config = config or GraphConfig()
        
    def build(self, crystal: Crystal, **kwargs) -> Data:
        """
        Convert crystal structure to PyTorch Geometric graph.
        
        Args:
            crystal: Crystal structure to convert
            **kwargs: Override config parameters
            
        Returns:
            torch_geometric.data.Data: Graph representation
            
        Raises:
            GraphBuildingError: If graph building fails
        """
        try:
            # Override config with kwargs
            config = GraphConfig(
                cutoff_radius=kwargs.get('cutoff_radius', self.config.cutoff_radius),
                max_neighbors=kwargs.get('max_neighbors', self.config.max_neighbors),
                include_edge_features=kwargs.get('include_edge_features', self.config.include_edge_features),
                include_node_features=kwargs.get('include_node_features', self.config.include_node_features),
                periodic_boundary=kwargs.get('periodic_boundary', self.config.periodic_boundary)
            )
            
            # Extract atomic positions and elements
            positions, elements = self._extract_structure_info(crystal)
            
            # Build node features
            node_features = self._build_node_features(elements) if config.include_node_features else None
            
            # Build edges with periodic boundary conditions
            edge_indices, edge_features = self._build_edges(
                crystal, positions, config
            )
            
            # Create PyTorch Geometric Data object
            graph_data = Data(
                x=node_features,
                edge_index=edge_indices,
                edge_attr=edge_features if config.include_edge_features else None,
                pos=torch.tensor(positions, dtype=torch.float32)
            )
            
            # Add metadata
            graph_data.formula = crystal.formula
            graph_data.num_atoms = crystal.num_atoms
            graph_data.lattice_params = self._get_lattice_params(crystal)
            
            return graph_data
            
        except Exception as e:
            raise GraphBuildingError(f"Failed to build graph: {str(e)}")
    
    def _extract_structure_info(self, crystal: Crystal) -> Tuple[np.ndarray, List[str]]:
        """Extract positions and elements from crystal."""
        positions = []
        elements = []
        
        # Convert lattice parameters to matrix
        lattice_matrix = np.array(crystal.lattice.lattice_matrix())
        
        for site in crystal.sites:
            # Convert fractional coordinates to Cartesian
            cartesian_pos = lattice_matrix @ site.position
            positions.append(cartesian_pos)
            elements.append(site.element)
        
        return np.array(positions), elements
    
    def _build_node_features(self, elements: List[str]) -> torch.Tensor:
        """Build node features based on atomic properties."""
        # Simple atomic number encoding
        atomic_numbers = []
        for element in elements:
            atomic_num = self._get_atomic_number(element)
            atomic_numbers.append(atomic_num)
        
        # Convert to one-hot encoding (simplified - could be more sophisticated)
        max_atomic_num = max(atomic_numbers)
        node_features = torch.zeros(len(elements), max_atomic_num)
        
        for i, atomic_num in enumerate(atomic_numbers):
            if atomic_num > 0:
                node_features[i, atomic_num - 1] = 1.0
        
        return node_features
    
    def _build_edges(self, crystal: Crystal, positions: np.ndarray, 
                    config: GraphConfig) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build edge indices and features."""
        from scipy.spatial.distance import cdist
        
        edge_indices = []
        edge_features = []
        
        # Calculate distances
        distances = cdist(positions, positions)
        
        # Handle periodic boundary conditions if enabled
        if config.periodic_boundary:
            positions_extended = self._extend_with_periodic_images(
                crystal, positions, config.cutoff_radius
            )
            distances_extended = cdist(positions, positions_extended)
            distances = np.minimum(distances, distances_extended[:, :len(positions)])
        
        # Build edges based on cutoff radius
        for i in range(len(positions)):
            # Find neighbors within cutoff
            neighbors = np.where(
                (distances[i] < config.cutoff_radius) & 
                (distances[i] > 0.1)  # Exclude self
            )[0]
            
            # Limit to max_neighbors
            if len(neighbors) > config.max_neighbors:
                neighbor_distances = distances[i][neighbors]
                sorted_indices = np.argsort(neighbor_distances)
                neighbors = neighbors[sorted_indices[:config.max_neighbors]]
            
            # Add edges
            for j in neighbors:
                edge_indices.append([i, j])
                
                if config.include_edge_features:
                    # Distance as edge feature
                    distance = distances[i, j]
                    edge_features.append([distance])
        
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features, dtype=torch.float32) if edge_features else None
        
        return edge_indices, edge_features
    
    def _extend_with_periodic_images(self, crystal: Crystal, positions: np.ndarray, 
                                   cutoff: float) -> np.ndarray:
        """Generate periodic images for boundary conditions."""
        lattice_matrix = np.array(crystal.lattice.lattice_matrix())
        
        # Simple approach: extend by one unit cell in each direction
        extended_positions = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # Skip original unit cell
                    
                    translation = dx * lattice_matrix[0] + dy * lattice_matrix[1] + dz * lattice_matrix[2]
                    translated_positions = positions + translation
                    extended_positions.extend(translated_positions)
        
        return np.vstack([positions, np.array(extended_positions)])
    
    def _get_atomic_number(self, element: str) -> int:
        """Get atomic number for element."""
        # Simple mapping - could be more comprehensive
        atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Fe': 26, 'Co': 27,
            'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34
        }
        return atomic_numbers.get(element, 0)
    
    def _get_lattice_params(self, crystal: Crystal) -> Dict[str, float]:
        """Extract lattice parameters as metadata."""
        return {
            'a': crystal.lattice.a,
            'b': crystal.lattice.b,
            'c': crystal.lattice.c,
            'alpha': crystal.lattice.alpha,
            'beta': crystal.lattice.beta,
            'gamma': crystal.lattice.gamma,
            'volume': crystal.volume
        }


def to_graph(crystal: Crystal, **kwargs) -> Data:
    """
    Convenience function to convert crystal to graph.
    
    Args:
        crystal: Crystal structure to convert
        **kwargs: Configuration parameters for GraphBuilder
        
    Returns:
        torch_geometric.data.Data: Graph representation
        
    Examples:
        >>> graph = to_graph(crystal, cutoff_radius=4.0, max_neighbors=8)
        >>> print(f"Converted crystal with {graph.num_nodes} nodes")
    """
    builder = GraphBuilder()
    return builder.build(crystal, **kwargs)