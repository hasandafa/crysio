"""
Utilities module for Crysio library.

This module provides utility functions, configuration management,
exception classes, and helper functions.
"""

from .exceptions import (
    CrysioError,
    ParsingError,
    ValidationError, 
    ConversionError,
    APIError,
    GraphBuildingError,
    VisualizationError,
    ConfigurationError
)

# Import placeholder classes (to be implemented)
# from .config import Config
# from .helpers import *

__all__ = [
    # Exception classes
    'CrysioError',
    'ParsingError', 
    'ValidationError',
    'ConversionError',
    'APIError',
    'GraphBuildingError',
    'VisualizationError',
    'ConfigurationError',
    
    # Placeholder for future classes
    # 'Config',
]