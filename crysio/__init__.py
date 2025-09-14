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

# FIXED: Import what actually exists in validators.py
from .core.validators import (
    StructureValidator,
    ValidationResult,
    BaseValidator,
    LatticeValidator,
    AtomicPositionValidator,
    CompositionValidator
)

# Converter imports - only import if modules exist
try:
    from .converters.graph_builder import (
        GraphBuilder,
        to_graph
    )
except ImportError:
    # GraphBuilder not available, define dummy functions
    GraphBuilder = None
    def to_graph(*args, **kwargs):
        raise ImportError("GraphBuilder not available. Install torch-geometric for graph conversion.")

# API imports - only import if modules exist
try:
    from .api.materials_project import (
        MaterialsProjectAPI,
        search_materials_database,
        load_from_materials_project
    )
except ImportError:
    # API not available, define dummy functions
    MaterialsProjectAPI = None
    def search_materials_database(*args, **kwargs):
        raise ImportError("Materials Project API not available.")
    def load_from_materials_project(*args, **kwargs):
        raise ImportError("Materials Project API not available.")

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

# FIXED: Create validate_structure function using StructureValidator
def validate_structure(structure, validation_level="medium"):
    """
    Validate crystal structure using StructureValidator.
    
    Args:
        structure: Crystal structure to validate
        validation_level: Currently not implemented, kept for compatibility
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_issues)
        
    Examples:
        >>> is_valid, issues = crysio.validate_structure(structure)
    """
    validator = StructureValidator()
    result = validator.validate(structure)
    
    # Convert ValidationResult to tuple format
    issues = []
    if hasattr(result, 'errors'):
        issues.extend(result.errors)
    if hasattr(result, 'warnings'):
        issues.extend(result.warnings)
    
    return result.is_valid, issues

def clean_structure(structure, validation_level="medium"):
    """
    Clean and validate crystal structure.
    
    Args:
        structure: Crystal structure to clean
        validation_level: Validation strictness (placeholder)
        
    Returns:
        Crystal: Input structure (cleaning not implemented yet)
        
    Examples:
        >>> clean_struct = crysio.clean_structure(structure)
    """
    # For now, just validate and return original structure
    # Cleaning functionality to be implemented
    is_valid, issues = validate_structure(structure, validation_level)
    
    if not is_valid:
        print("Validation issues found:", issues)
    
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
    try:
        from tqdm import tqdm
        iterator = tqdm(structures_or_paths, desc="Processing structures") if progress else structures_or_paths
    except ImportError:
        iterator = structures_or_paths
    
    processed = []
    
    for item in iterator:
        try:
            if isinstance(item, Crystal):
                structure = item
            else:
                structure = load_structure(item, format=format)
            
            clean_struct = clean_structure(structure, validation_level=validation_level)
            processed.append(clean_struct)
            
        except Exception as e:
            if progress and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({"error": str(e)[:30]})
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
    
    # Validators (FIXED)
    'StructureValidator',
    'ValidationResult',
    'BaseValidator',
    'LatticeValidator',
    'AtomicPositionValidator',
    'CompositionValidator',
    'validate_structure',
    
    # Converters (optional)
    'GraphBuilder',
    'to_graph',
    
    # API (optional)
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
    'validate_structure',
    'load',
    'parse',
]