"""
Core module for Crysio library.

This module provides the fundamental classes and functions for crystal structure
representation, parsing, cleaning, and validation.
"""

from .crystal import Crystal, LatticeParameters, AtomicSite
from .parsers import CIFParser, POSCARParser, BaseParser, auto_detect_format, get_parser

# Import placeholder classes (to be implemented)
# from .cleaners import StructureCleaner
# from .validators import StructureValidator

__all__ = [
    # Core structure classes
    'Crystal',
    'LatticeParameters', 
    'AtomicSite',
    
    # Parser classes
    'BaseParser',
    'CIFParser',
    'POSCARParser',
    
    # Utility functions
    'auto_detect_format',
    'get_parser',
    
    # Placeholder for future classes
    # 'StructureCleaner',
    # 'StructureValidator',
]