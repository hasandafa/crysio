"""
API module for Crysio library.

This module provides interfaces to external materials databases and web services
for fetching crystal structures, properties, and other materials data.
"""

from .materials_project import (
    MaterialsProjectAPI, 
    MaterialsProjectConfig, 
    quick_search, 
    download_stable_materials
)

__all__ = [
    # Materials Project API
    'MaterialsProjectAPI',
    'MaterialsProjectConfig',
    'quick_search',
    'download_stable_materials',
]