"""
Custom exception classes for Crysio library.

This module defines all custom exceptions used throughout the Crysio library
to provide clear error messages and proper error handling.
"""

from typing import Optional, Any


class CrysioError(Exception):
    """
    Base exception class for all Crysio-related errors.
    
    All other custom exceptions in Crysio should inherit from this base class.
    """
    pass


class ParsingError(CrysioError):
    """
    Raised when there's an error parsing crystal structure files.
    
    This exception is raised when file format is invalid, corrupted,
    or contains unexpected data that cannot be parsed.
    
    Attributes:
        filename: The name of the file that caused the error
        line_number: The line number where the error occurred (if applicable)
        message: Detailed error message
    """
    
    def __init__(
        self, 
        message: str, 
        filename: Optional[str] = None,
        line_number: Optional[int] = None
    ):
        self.filename = filename
        self.line_number = line_number
        self.message = message
        
        # Construct detailed error message
        error_msg = f"Parsing Error: {message}"
        if filename:
            error_msg += f" (File: {filename}"
            if line_number:
                error_msg += f", Line: {line_number}"
            error_msg += ")"
            
        super().__init__(error_msg)


class ValidationError(CrysioError):
    """
    Raised when crystal structure validation fails.
    
    This exception is raised when a crystal structure doesn't meet
    expected physical or mathematical constraints.
    
    Attributes:
        structure_id: Identifier for the structure that failed validation
        validation_type: Type of validation that failed
        message: Detailed error message
    """
    
    def __init__(
        self, 
        message: str,
        structure_id: Optional[str] = None,
        validation_type: Optional[str] = None
    ):
        self.structure_id = structure_id
        self.validation_type = validation_type
        self.message = message
        
        error_msg = f"Validation Error"
        if validation_type:
            error_msg += f" ({validation_type})"
        error_msg += f": {message}"
        if structure_id:
            error_msg += f" (Structure: {structure_id})"
            
        super().__init__(error_msg)


class ConversionError(CrysioError):
    """
    Raised when there's an error converting between different formats or representations.
    
    This includes errors in:
    - File format conversion (CIF → POSCAR, etc.)
    - Structure → Graph conversion
    - Coordinate transformations
    
    Attributes:
        from_format: Source format or representation
        to_format: Target format or representation  
        message: Detailed error message
    """
    
    def __init__(
        self, 
        message: str,
        from_format: Optional[str] = None,
        to_format: Optional[str] = None
    ):
        self.from_format = from_format
        self.to_format = to_format
        self.message = message
        
        error_msg = f"Conversion Error"
        if from_format and to_format:
            error_msg += f" ({from_format} → {to_format})"
        error_msg += f": {message}"
        
        super().__init__(error_msg)


class APIError(CrysioError):
    """
    Raised when there's an error with external API calls.
    
    This includes errors from Materials Project API, database connections,
    network issues, authentication problems, etc.
    
    Attributes:
        api_name: Name of the API that caused the error
        status_code: HTTP status code (if applicable)
        response: API response content (if available)
        message: Detailed error message
    """
    
    def __init__(
        self, 
        message: str,
        api_name: Optional[str] = None,
        status_code: Optional[int] = None,
        response: Optional[Any] = None
    ):
        self.api_name = api_name
        self.status_code = status_code
        self.response = response
        self.message = message
        
        error_msg = f"API Error"
        if api_name:
            error_msg += f" ({api_name})"
        if status_code:
            error_msg += f" [Status: {status_code}]"
        error_msg += f": {message}"
        
        super().__init__(error_msg)


class GraphBuildingError(ConversionError):
    """
    Specialized conversion error for graph building operations.
    
    Raised when there's an error converting crystal structures to
    graph representations for Graph Neural Networks.
    """
    
    def __init__(self, message: str, structure_id: Optional[str] = None):
        self.structure_id = structure_id
        error_msg = f"Graph Building Error: {message}"
        if structure_id:
            error_msg += f" (Structure: {structure_id})"
        
        # Call ConversionError with from_format and to_format
        super(ConversionError, self).__init__(error_msg)


class VisualizationError(CrysioError):
    """
    Raised when there's an error in visualization operations.
    
    This includes errors in plotting, 3D rendering, interactive
    visualizations, or saving visualization outputs.
    
    Attributes:
        viz_type: Type of visualization that failed
        message: Detailed error message
    """
    
    def __init__(self, message: str, viz_type: Optional[str] = None):
        self.viz_type = viz_type
        self.message = message
        
        error_msg = f"Visualization Error"
        if viz_type:
            error_msg += f" ({viz_type})"
        error_msg += f": {message}"
        
        super().__init__(error_msg)


class ConfigurationError(CrysioError):
    """
    Raised when there's an error in configuration or setup.
    
    This includes missing API keys, invalid configuration files,
    missing dependencies, or environment setup issues.
    
    Attributes:
        config_item: The configuration item that caused the error
        message: Detailed error message
    """
    
    def __init__(self, message: str, config_item: Optional[str] = None):
        self.config_item = config_item
        self.message = message
        
        error_msg = f"Configuration Error"
        if config_item:
            error_msg += f" ({config_item})"
        error_msg += f": {message}"
        
        super().__init__(error_msg)