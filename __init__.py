# crysio/__init__.py
"""
Crysio: Crystal I/O toolkit for preprocessing and visualizing 
crystal structures for machine learning applications.
"""

from pathlib import Path

__version__ = "0.1.0"
__author__ = "Dafa, Abdullah Hasan"
__email__ = "dafa.abdullahhasan@gmail.com"

# Core imports
from .core.crystal import Crystal
from .core.parsers import CIFParser, POSCARParser
from .core.cleaners import StructureCleaner
from .core.validators import StructureValidator

# Converter imports  
from .converters.graph_builder import GraphBuilder
from .converters.format_converter import FormatConverter

# Visualizer imports
from .visualizers.crystal_viz import Crystal2DVisualizer, Crystal3DVisualizer, InteractiveCrystalViz
from .visualizers.analysis_plots import StructureAnalysisPlots, PropertyCorrelationPlots

# API imports
from .api.materials_project import MaterialsProjectAPI

# Utils imports
from .utils.config import Config
from .utils.exceptions import (
    CrysioError,
    ParsingError, 
    ValidationError,
    ConversionError
)


# High-level convenience functions
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
    from .core.parsers import auto_detect_format, get_parser
    
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
    """
    cleaner = StructureCleaner(**kwargs)
    validator = StructureValidator()
    
    cleaned = cleaner.clean(structure)
    validator.validate(cleaned)
    return cleaned


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
    """
    graph_builder = GraphBuilder(**kwargs)
    return graph_builder.build_graph(structure)


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
    from tqdm import tqdm
    
    results = []
    for structure in tqdm(structures, desc="Processing structures"):
        try:
            # Load if filepath
            if isinstance(structure, (str, Path)):
                structure = load_structure(structure)
                
            # Apply operations
            if 'clean' in operations:
                structure = clean(structure)
            if 'validate' in operations:
                validator = StructureValidator()
                validator.validate(structure)
            if 'to_graph' in operations:
                structure = to_graph(structure, **kwargs)
                
            results.append(structure)
        except Exception as e:
            print(f"Error processing structure: {e}")
            continue
            
    return results


# Visualization namespace
class _VisualizationManager:
    """Manager class for visualization functions."""
    
    def __init__(self):
        self._2d = None
        self._3d = None
        self._interactive = None
        self._analysis = None
    
    @property
    def crystal_2d(self):
        if self._2d is None:
            self._2d = Crystal2DVisualizer()
        return self._2d
    
    @property
    def crystal_3d(self):
        if self._3d is None:
            self._3d = Crystal3DVisualizer()
        return self._3d
    
    @property
    def interactive(self):
        if self._interactive is None:
            self._interactive = InteractiveCrystalViz()
        return self._interactive
    
    @property
    def analysis(self):
        if self._analysis is None:
            self._analysis = StructureAnalysisPlots()
        return self._analysis
    
    def ball_and_stick_3d(self, structure, **kwargs):
        """Quick access to 3D ball-and-stick visualization."""
        return self.crystal_3d.ball_and_stick_model(structure, **kwargs)
    
    def interactive_3d(self, structure, **kwargs):
        """Quick access to interactive 3D visualization."""
        return self.interactive.interactive_3d_viewer(structure, **kwargs)
    
    def property_correlation_heatmap(self, dataset, **kwargs):
        """Quick access to property correlation analysis."""
        return self.analysis.property_vs_composition(dataset, **kwargs)
    
    def data_quality_dashboard(self, dataset, **kwargs):
        """Quick access to data quality assessment."""
        return self.analysis.missing_data_heatmap(dataset, **kwargs)


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
    
    # Core classes
    "Crystal",
    "CIFParser",
    "POSCARParser", 
    "StructureCleaner",
    "StructureValidator",
    
    # Converters
    "GraphBuilder",
    "FormatConverter",
    
    # Visualizers
    "Crystal2DVisualizer",
    "Crystal3DVisualizer", 
    "InteractiveCrystalViz",
    "StructureAnalysisPlots",
    "PropertyCorrelationPlots",
    
    # API
    "MaterialsProjectAPI",
    
    # Utils
    "Config",
    "CrysioError",
    "ParsingError",
    "ValidationError", 
    "ConversionError",
]