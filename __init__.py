"""
Crysio: Crystal I/O toolkit for preprocessing and visualizing crystal structures 
for machine learning applications.

This library provides comprehensive tools for:
- Loading and parsing crystal structures (CIF, POSCAR, XYZ)
- Converting structures to PyTorch Geometric graphs
- Structure validation and cleaning
- Integration with Materials Project database
- Batch processing and analysis

Version 0.2.1 - Bug Fix Release
===============================
Fixed POSCAR parsing bug for Materials Project format files.
"""

__version__ = "0.2.1"
__author__ = "Dafa, Abdullah Hasan"
__email__ = "dafa.abdullahhasan@gmail.com"
__license__ = "MIT"

# Core imports
from .core.crystal import (
    Crystal,
    LatticeParameters,
    AtomicSite
)

from .core.parsers import (
    CIFParser,
    POSCARParser,
    auto_detect_format,
    get_parser
)

from .core.validators import (
    CrystalValidator,
    validate_structure,
    ValidationLevel
)

# Converter imports
from .converters.graph_builder import (
    GraphBuilder,
    to_graph
)

# API imports
from .api.materials_project import (
    MaterialsProjectAPI,
    search_materials_database,
    load_from_materials_project
)

# Exception imports  
from .utils.exceptions import (
    CrysioError,
    ParsingError,
    ValidationError,
    ConversionError,
    APIError
)

# Main API functions
def load_structure(filepath_or_content, format=None):
    """
    Load crystal structure from file or content string.
    
    Args:
        filepath_or_content: Path to structure file or content as string
        format: File format ('cif', 'poscar', etc.). If None, auto-detect.
        
    Returns:
        Crystal: Loaded crystal structure
        
    Examples:
        >>> structure = crysio.load_structure("example.cif")
        >>> structure = crysio.load_structure("POSCAR", format="poscar")
    """
    if format is None:
        format = auto_detect_format(filepath_or_content)
    
    parser = get_parser(format)
    return parser.parse(filepath_or_content)


def clean_structure(structure, validation_level="medium"):
    """
    Clean and validate crystal structure.
    
    Args:
        structure: Crystal structure to clean
        validation_level: Validation strictness ("basic", "medium", "strict")
        
    Returns:
        Crystal: Cleaned crystal structure
        
    Examples:
        >>> clean_struct = crysio.clean_structure(structure)
    """
    validator = CrystalValidator(validation_level=ValidationLevel.from_string(validation_level))
    
    # Validate structure
    is_valid, issues = validator.validate_crystal(structure)
    
    if not is_valid:
        # Apply fixes for common issues
        # This would be expanded in future versions
        pass
    
    return structure


def batch_process(structures_or_paths, format=None, validation_level="medium", 
                 progress=True, n_workers=1):
    """
    Process multiple crystal structures in batch.
    
    Args:
        structures_or_paths: List of file paths or Crystal objects
        format: File format if loading from paths
        validation_level: Validation strictness
        progress: Show progress bar
        n_workers: Number of parallel workers
        
    Returns:
        List[Crystal]: Processed crystal structures
        
    Examples:
        >>> structures = crysio.batch_process(["file1.cif", "file2.cif"])
    """
    from tqdm import tqdm
    
    processed = []
    iterator = structures_or_paths
    
    if progress:
        iterator = tqdm(structures_or_paths, desc="Processing structures")
    
    for item in iterator:
        try:
            if isinstance(item, Crystal):
                structure = item
            else:
                structure = load_structure(item, format=format)
            
            clean_struct = clean_structure(structure, validation_level=validation_level)
            processed.append(clean_struct)
            
        except Exception as e:
            if progress:
                iterator.set_postfix({"error": str(e)[:30]})
            # Could add error handling options here
            continue
    
    return processed


# Convenience aliases
load = load_structure
parse = load_structure

# Version info
version_info = tuple(int(x) for x in __version__.split('.'))

# All public API
__all__ = [
    # Version info
    '__version__',
    'version_info',
    
    # Core classes
    'Crystal',
    'LatticeParameters', 
    'AtomicSite',
    
    # Parsers
    'CIFParser',
    'POSCARParser',
    'auto_detect_format',
    'get_parser',
    
    # Validators
    'CrystalValidator',
    'validate_structure',
    'ValidationLevel',
    
    # Converters
    'GraphBuilder',
    'to_graph',
    
    # API
    'MaterialsProjectAPI',
    'search_materials_database',
    'load_from_materials_project',
    
    # Exceptions
    'CrysioError',
    'ParsingError', 
    'ValidationError',
    'ConversionError',
    'APIError',
    
    # Main functions
    'load_structure',
    'clean_structure',
    'batch_process',
    'load',
    'parse',
]