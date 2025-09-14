"""
API module for Crysio library.

This module provides interfaces to external materials databases and web services
for fetching crystal structures, properties, and other materials data.
"""

from .materials_project import (
    MaterialsProjectAPI, 
    MaterialsProjectConfig, 
    quick_search, 
    download_stable_materials,
    search_materials_database,  # ADDED: Missing function
    load_from_materials_project  # ADDED: Missing function
)

__all__ = [
    # Materials Project API
    'MaterialsProjectAPI',
    'MaterialsProjectConfig',
    'quick_search',
    'download_stable_materials',
    'search_materials_database',      # ADDED
    'load_from_materials_project',    # ADDED
]