"""
Crysio: Crystal I/O toolkit for preprocessing and visualizing 
crystal structures for machine learning applications.
"""

__version__ = "0.2.0"
__author__ = "Dafa, Abdullah Hasan"
__email__ = "dafa.abdullahhasan@gmail.com"

# Import dependencies first
from pathlib import Path

# Core imports with better error handling
try:
    from .core.crystal import Crystal
    from .core.parsers import CIFParser, POSCARParser, auto_detect_format, get_parser
except ImportError:
    # Fallback for direct execution or import issues
    try:
        from crysio.core.crystal import Crystal
        from crysio.core.parsers import CIFParser, POSCARParser, auto_detect_format, get_parser
    except ImportError as e:
        print(f"Warning: Could not import core modules: {e}")
        Crystal = None
        CIFParser = None
        POSCARParser = None
        auto_detect_format = None
        get_parser = None

# Utils imports with error handling
try:
    from .utils.exceptions import (
        CrysioError,
        ParsingError, 
        ValidationError,
        ConversionError,
        APIError,
        GraphBuildingError,
        VisualizationError,
        ConfigurationError
    )
except ImportError:
    try:
        from crysio.utils.exceptions import (
            CrysioError,
            ParsingError, 
            ValidationError,
            ConversionError,
            APIError,
            GraphBuildingError,
            VisualizationError,
            ConfigurationError
        )
    except ImportError as e:
        print(f"Warning: Could not import exception classes: {e}")
        CrysioError = Exception
        ParsingError = Exception
        ValidationError = Exception
        ConversionError = Exception
        APIError = Exception
        GraphBuildingError = Exception
        VisualizationError = Exception
        ConfigurationError = Exception


def load_structure(filepath_or_structure, format=None):
    """
    Load crystal structure from file or string.
    
    Args:
        filepath_or_structure: Path to structure file or structure string
        format: File format ('cif', 'poscar', 'xyz'). Auto-detected if None.
        
    Returns:
        Crystal: Loaded crystal structure
        
    Examples:
        >>> structure = crysio.load_structure("example.cif")
        >>> structure = crysio.load_structure("POSCAR", format="poscar")
    """
    if auto_detect_format is None or get_parser is None:
        raise ImportError("Parser modules not available")
        
    if format is None:
        format = auto_detect_format(filepath_or_structure)
    
    parser = get_parser(format)
    return parser.parse(filepath_or_structure)


def clean(structure, **kwargs):
    """
    Clean and validate crystal structure.
    
    Args:
        structure: Crystal structure to clean
        **kwargs: Additional cleaning options
        
    Returns:
        Crystal: Cleaned crystal structure
        
    Examples:
        >>> clean_structure = crysio.clean(structure)
        >>> clean_structure = crysio.clean(structure, remove_duplicates=True)
        
    Note:
        Currently returns structure as-is. Full cleaning implementation coming soon.
    """
    print("Warning: Structure cleaning not yet implemented. Returning original structure.")
    return structure


def to_graph(structure, **kwargs):
    """
    Convert crystal structure to PyTorch Geometric graph.
    
    Args:
        structure: Crystal structure to convert
        **kwargs: Graph building options (cutoff_radius, edge_features, etc.)
        
    Returns:
        torch_geometric.data.Data: Graph representation
        
    Examples:
        >>> graph = crysio.to_graph(structure)
        >>> graph = crysio.to_graph(structure, cutoff_radius=5.0)
        
    Note:
        Graph conversion implementation coming soon.
    """
    raise NotImplementedError("Graph conversion not yet implemented. Coming soon!")


def batch_process(structures, operations=['clean', 'validate'], **kwargs):
    """
    Process multiple structures in batch.
    
    Args:
        structures: List of crystal structures or file paths
        operations: List of operations to perform
        **kwargs: Additional processing options
        
    Returns:
        List[Crystal]: Processed structures
        
    Examples:
        >>> files = ["struct1.cif", "struct2.cif", "struct3.cif"]  
        >>> processed = crysio.batch_process(files)
        >>> processed = crysio.batch_process(structures, operations=['clean', 'to_graph'])
    """
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback if tqdm not available
        def tqdm(iterable, desc=""):
            return iterable
    
    results = []
    for structure in tqdm(structures, desc="Processing structures"):
        try:
            # Load if filepath
            if isinstance(structure, (str, Path)):
                structure = load_structure(structure)
                
            # Apply operations (currently limited)
            if 'clean' in operations:
                structure = clean(structure)
            if 'validate' in operations:
                print("Warning: Validation not yet implemented")
            if 'to_graph' in operations:
                print("Warning: Graph conversion not yet implemented")
                
            results.append(structure)
        except Exception as e:
            print(f"Error processing structure: {e}")
            continue
            
    return results


# Visualization namespace - placeholder implementation
class _VisualizationManager:
    """Manager class for visualization functions - placeholder implementation."""
    
    def __init__(self):
        pass  # Don't print warning on every import
    
    def ball_and_stick_3d(self, structure, **kwargs):
        """Placeholder for 3D ball-and-stick visualization."""
        raise NotImplementedError("3D visualization not yet implemented. Coming soon!")
    
    def interactive_3d(self, structure, **kwargs):
        """Placeholder for interactive 3D visualization."""
        raise NotImplementedError("Interactive visualization not yet implemented. Coming soon!")
    
    def property_correlation_heatmap(self, dataset, **kwargs):
        """Placeholder for property correlation analysis."""
        raise NotImplementedError("Property analysis plots not yet implemented. Coming soon!")
    
    def data_quality_dashboard(self, dataset, **kwargs):
        """Placeholder for data quality assessment."""
        raise NotImplementedError("Data quality visualization not yet implemented. Coming soon!")


# Global visualization instance
visualize = _VisualizationManager()


# Convenience imports for direct access
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    
    # High-level functions
    "load_structure",
    "clean", 
    "to_graph",
    "batch_process",
    
    # Visualization
    "visualize",
    
    # Core classes (if available)
    "Crystal",
    "CIFParser",
    "POSCARParser",
    "auto_detect_format",
    "get_parser",
    
    # Exceptions (if available)
    "CrysioError",
    "ParsingError",
    "ValidationError", 
    "ConversionError",
    "APIError",
    "GraphBuildingError",
    "VisualizationError",
    "ConfigurationError",
]