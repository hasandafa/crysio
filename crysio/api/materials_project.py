"""
Materials Project API integration using the modern mp-api client.

This module provides interfaces to the Materials Project database using the
official mp-api package with MPRester client.
"""

import os
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    from mp_api.client import MPRester
    from emmet.core.summary import HasProps
    MP_API_AVAILABLE = True
except ImportError:
    MP_API_AVAILABLE = False
    MPRester = None
    HasProps = None

from ..core.crystal import Crystal, LatticeParameters, AtomicSite
from ..utils.exceptions import APIError, ParsingError, ConfigurationError


@dataclass
class MaterialsProjectConfig:
    """Configuration for Materials Project API."""
    api_key: str
    use_document_model: bool = True  # Use Pydantic models for responses
    monty_decode: bool = True        # Decode pymatgen objects
    chunk_size: int = 1000          # Chunk size for large queries


class MaterialsProjectAPI:
    """
    Modern interface to Materials Project database using mp-api.
    
    This class provides methods to search, retrieve, and process materials data
    from the Materials Project database using the official mp-api client package.
    
    Attributes:
        config (MaterialsProjectConfig): API configuration
        
    Examples:
        >>> mp_api = MaterialsProjectAPI("your_api_key_here")
        >>> structure = mp_api.get_structure_by_material_id("mp-149")
        >>> print(f"Structure: {structure.formula}")
        
        >>> # Search with properties
        >>> materials = mp_api.search_materials(
        ...     elements=["Li", "Fe", "P", "O"], 
        ...     properties=["material_id", "formation_energy_per_atom"]
        ... )
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[MaterialsProjectConfig] = None):
        """
        Initialize Materials Project API client.
        
        Args:
            api_key: Materials Project API key
            config: Custom configuration (overrides api_key if provided)
            
        Raises:
            ImportError: If mp-api package is not installed
            ConfigurationError: If no API key is provided
        """
        if not MP_API_AVAILABLE:
            raise ImportError(
                "mp-api package is required for Materials Project integration. "
                "Install with: pip install mp-api"
            )
        
        if config is not None:
            self.config = config
        elif api_key is not None:
            self.config = MaterialsProjectConfig(api_key=api_key)
        else:
            # Try to load from environment
            api_key = self._load_api_key()
            if api_key is None:
                raise ConfigurationError(
                    "Materials Project API key required. "
                    "Provide via api_key parameter, MP_API_KEY environment variable, "
                    "or configure through pymatgen"
                )
            self.config = MaterialsProjectConfig(api_key=api_key)
    
    def get_structure_by_material_id(self, material_id: str) -> Crystal:
        """
        Get crystal structure by Materials Project ID.
        
        Args:
            material_id: Materials Project ID (e.g., 'mp-149')
            
        Returns:
            Crystal: Crystal structure
            
        Raises:
            APIError: If API request fails
        """
        try:
            with MPRester(self.config.api_key) as mpr:
                # Use the shortcut method for single structure
                pymatgen_structure = mpr.get_structure_by_material_id(material_id)
                
                if pymatgen_structure is None:
                    raise APIError(f"No structure found for {material_id}", 'Materials Project')
                
                return self._convert_pymatgen_structure(pymatgen_structure, material_id)
                
        except Exception as e:
            if isinstance(e, APIError):
                raise
            raise APIError(f"Failed to fetch structure {material_id}: {str(e)}", 'Materials Project')
    
    def get_structures_by_formula(self, formula: str, limit: int = 100) -> List[Tuple[str, Crystal]]:
        """
        Get crystal structures by chemical formula.
        
        Args:
            formula: Chemical formula (e.g., 'LiFePO4', 'Si')
            limit: Maximum number of results
            
        Returns:
            List of (material_id, Crystal) tuples
            
        Raises:
            APIError: If API request fails
        """
        structures = []
        
        try:
            with MPRester(self.config.api_key) as mpr:
                # Search for materials with the given formula
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=["material_id", "structure", "formula_pretty"]
                )
                
                # Limit results
                docs = docs[:limit]
                
                for doc in docs:
                    try:
                        crystal = self._convert_pymatgen_structure(
                            doc.structure, 
                            str(doc.material_id)
                        )
                        structures.append((str(doc.material_id), crystal))
                    except Exception as e:
                        print(f"Warning: Could not convert structure for {doc.material_id}: {e}")
                        continue
            
            return structures
            
        except Exception as e:
            raise APIError(f"Failed to search formula {formula}: {str(e)}", 'Materials Project')
    
    def search_materials(self, 
                        formula: Optional[str] = None,
                        elements: Optional[List[str]] = None,
                        exclude_elements: Optional[List[str]] = None,
                        properties: Optional[List[str]] = None,
                        limit: int = 100,
                        **kwargs) -> List[Dict[str, Any]]:
        """
        Search materials in the Materials Project database.
        
        Args:
            formula: Chemical formula to search for
            elements: List of required elements
            exclude_elements: List of elements to exclude
            properties: List of properties to retrieve
            limit: Maximum number of results
            **kwargs: Additional search criteria (band_gap, formation_energy_per_atom, etc.)
            
        Returns:
            List of material data dictionaries
            
        Raises:
            APIError: If API request fails
        """
        try:
            with MPRester(self.config.api_key) as mpr:
                # Build search parameters
                search_params = {}
                
                if formula:
                    search_params['formula'] = formula
                if elements:
                    search_params['elements'] = elements
                if exclude_elements:
                    search_params['exclude_elements'] = exclude_elements
                
                # Add additional search criteria
                search_params.update(kwargs)
                
                # Determine fields to retrieve
                if properties:
                    fields = ['material_id'] + properties
                else:
                    fields = None  # Get all available fields
                
                if fields:
                    search_params['fields'] = fields
                
                # Execute search
                docs = mpr.materials.summary.search(**search_params)
                
                # Limit results
                docs = docs[:limit]
                
                # Convert to dictionaries
                results = []
                for doc in docs:
                    result = doc.model_dump() if hasattr(doc, 'model_dump') else doc.dict()
                    results.append(result)
                
                return results
                
        except Exception as e:
            raise APIError(f"Materials search failed: {str(e)}", 'Materials Project')
    
    def get_material_properties(self, material_id: str, properties: List[str]) -> Dict[str, Any]:
        """
        Get specific properties for a material.
        
        Args:
            material_id: Materials Project ID
            properties: List of property names to retrieve
            
        Returns:
            Dictionary of property values
            
        Raises:
            APIError: If API request fails
        """
        try:
            with MPRester(self.config.api_key) as mpr:
                docs = mpr.materials.summary.search(
                    material_ids=[material_id],
                    fields=['material_id'] + properties
                )
                
                if not docs:
                    raise APIError(f"No data found for {material_id}", 'Materials Project')
                
                result = docs[0].model_dump() if hasattr(docs[0], 'model_dump') else docs[0].dict()
                return result
                
        except Exception as e:
            if isinstance(e, APIError):
                raise
            raise APIError(f"Failed to get properties for {material_id}: {str(e)}", 'Materials Project')
    
    def search_by_criteria(self, 
                          band_gap_range: Optional[Tuple[float, float]] = None,
                          formation_energy_range: Optional[Tuple[float, float]] = None,
                          stability_range: Optional[Tuple[float, float]] = None,
                          crystal_system: Optional[str] = None,
                          space_group: Optional[int] = None,
                          has_properties: Optional[List[str]] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search materials by specific criteria.
        
        Args:
            band_gap_range: (min, max) band gap in eV
            formation_energy_range: (min, max) formation energy per atom in eV
            stability_range: (min, max) energy above hull in eV
            crystal_system: Crystal system (cubic, tetragonal, etc.)
            space_group: Space group number
            has_properties: List of required properties (e.g., ['dielectric', 'piezoelectric'])
            limit: Maximum number of results
            
        Returns:
            List of material data dictionaries
        """
        search_params = {}
        
        # Band gap range
        if band_gap_range:
            search_params['band_gap'] = band_gap_range
        
        # Formation energy range
        if formation_energy_range:
            search_params['formation_energy_per_atom'] = formation_energy_range
        
        # Stability range (energy above hull)
        if stability_range:
            search_params['energy_above_hull'] = stability_range
        
        # Crystal system
        if crystal_system:
            search_params['crystal_system'] = crystal_system
        
        # Space group
        if space_group:
            search_params['spacegroup'] = space_group
        
        # Properties filter using HasProps
        if has_properties and HasProps:
            props_enum = []
            for prop in has_properties:
                if hasattr(HasProps, prop.upper()):
                    props_enum.append(getattr(HasProps, prop.upper()))
            if props_enum:
                search_params['has_props'] = props_enum
        
        # Define fields to retrieve
        fields = [
            'material_id', 'formula_pretty', 'structure',
            'formation_energy_per_atom', 'band_gap', 'energy_above_hull',
            'crystal_system', 'spacegroup', 'volume', 'density'
        ]
        
        return self.search_materials(fields=fields, limit=limit, **search_params)
    
    def download_structures_batch(self, 
                                 material_ids: List[str], 
                                 output_dir: Optional[str] = None,
                                 include_properties: bool = True) -> Dict[str, Crystal]:
        """
        Download multiple crystal structures in batch.
        
        Args:
            material_ids: List of Materials Project IDs
            output_dir: Optional directory to save structures as files
            include_properties: Include additional properties in metadata
            
        Returns:
            Dictionary mapping material_id to Crystal structure
            
        Raises:
            APIError: If API requests fail
        """
        structures = {}
        output_path = Path(output_dir) if output_dir else None
        
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            with MPRester(self.config.api_key) as mpr:
                # Get structures in batch
                fields = ['material_id', 'structure', 'formula_pretty']
                if include_properties:
                    fields.extend([
                        'formation_energy_per_atom', 'band_gap', 'energy_above_hull',
                        'crystal_system', 'spacegroup', 'volume', 'density'
                    ])
                
                docs = mpr.materials.summary.search(
                    material_ids=material_ids,
                    fields=fields
                )
                
                print(f"Retrieved {len(docs)} structures from Materials Project")
                
                for doc in docs:
                    material_id = str(doc.material_id)
                    
                    try:
                        # Convert structure
                        crystal = self._convert_pymatgen_structure(
                            doc.structure, 
                            material_id,
                            additional_metadata=doc.model_dump() if hasattr(doc, 'model_dump') else doc.dict()
                        )
                        structures[material_id] = crystal
                        
                        # Optionally save to file
                        if output_path:
                            filename = output_path / f"{material_id}.json"
                            self._save_structure_json(crystal, filename)
                        
                    except Exception as e:
                        print(f"Error processing {material_id}: {e}")
                        continue
            
            return structures
            
        except Exception as e:
            raise APIError(f"Batch download failed: {str(e)}", 'Materials Project')
    
    def get_phase_diagram_data(self, chemsys: str) -> Dict[str, Any]:
        """
        Get phase diagram data for a chemical system.
        
        Args:
            chemsys: Chemical system (e.g., 'Li-Fe-P-O')
            
        Returns:
            Dictionary with phase diagram data
        """
        try:
            with MPRester(self.config.api_key) as mpr:
                # Get entries for phase diagram
                entries = mpr.get_entries_in_chemsys(chemsys)
                
                return {
                    'chemical_system': chemsys,
                    'num_entries': len(entries),
                    'entries': [entry.as_dict() for entry in entries[:10]]  # Limit for size
                }
                
        except Exception as e:
            raise APIError(f"Failed to get phase diagram for {chemsys}: {str(e)}", 'Materials Project')
    
    def _convert_pymatgen_structure(self, 
                                   pymatgen_structure, 
                                   material_id: str,
                                   additional_metadata: Optional[Dict] = None) -> Crystal:
        """Convert pymatgen Structure to Crysio Crystal object."""
        try:
            # Extract lattice parameters
            lattice = pymatgen_structure.lattice
            lattice_params = LatticeParameters(
                a=lattice.a,
                b=lattice.b,
                c=lattice.c,
                alpha=lattice.alpha,
                beta=lattice.beta,
                gamma=lattice.gamma
            )
            
            # Extract atomic sites
            sites = []
            for i, site in enumerate(pymatgen_structure.sites):
                # Get the most abundant species (handle disorder)
                species = site.species
                if hasattr(species, 'most_common'):
                    element = str(species.most_common()[0][0])
                    occupancy = species.most_common()[0][1]
                else:
                    element = str(species)
                    occupancy = 1.0
                
                # Clean element symbol (remove charge)
                element = element.split('+')[0].split('-')[0]
                
                atomic_site = AtomicSite(
                    element=element,
                    position=site.frac_coords,
                    occupancy=occupancy,
                    label=f"{element}{i+1}"
                )
                sites.append(atomic_site)
            
            # Prepare metadata
            metadata = {
                'material_id': material_id,
                'source': 'Materials Project',
                'pymatgen_spacegroup': pymatgen_structure.get_space_group_info()[0]
            }
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Create Crystal object
            crystal = Crystal(
                lattice=lattice_params,
                sites=sites,
                formula=pymatgen_structure.composition.reduced_formula,
                space_group=pymatgen_structure.get_space_group_info()[0],
                metadata=metadata
            )
            
            return crystal
            
        except Exception as e:
            raise ParsingError(f"Failed to convert pymatgen structure: {str(e)}", material_id)
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment."""
        return os.getenv('MP_API_KEY')
    
    def _save_structure_json(self, crystal: Crystal, filename: Path):
        """Save crystal structure as JSON file."""
        import json
        
        data = {
            'material_id': crystal.metadata.get('material_id', 'unknown'),
            'formula': crystal.formula,
            'space_group': crystal.space_group,
            'lattice': {
                'a': crystal.lattice.a,
                'b': crystal.lattice.b,
                'c': crystal.lattice.c,
                'alpha': crystal.lattice.alpha,
                'beta': crystal.lattice.beta,
                'gamma': crystal.lattice.gamma,
                'volume': crystal.volume
            },
            'sites': [
                {
                    'element': site.element,
                    'position': site.position.tolist(),
                    'occupancy': site.occupancy,
                    'label': site.label
                }
                for site in crystal.sites
            ],
            'metadata': crystal.metadata
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API status and database version information."""
        try:
            with MPRester(self.config.api_key) as mpr:
                # Test API with a simple query
                docs = mpr.materials.summary.search(
                    material_ids=["mp-149"], 
                    fields=["material_id"]
                )
                
                # Get database version if possible
                try:
                    db_version = mpr.get_database_version()
                except:
                    db_version = "Unknown"
                
                return {
                    'status': 'active',
                    'api_client': 'mp-api (modern)',
                    'database_version': db_version,
                    'api_key_valid': len(docs) > 0,
                    'test_query_result': f"Found {len(docs)} materials"
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'api_key_valid': False
            }


# Utility functions for common operations
def quick_search(formula: str, api_key: Optional[str] = None, limit: int = 10) -> List[Tuple[str, Crystal]]:
    """
    Quick search function for getting structures by formula.
    
    Args:
        formula: Chemical formula
        api_key: Materials Project API key (optional if configured)
        limit: Maximum number of results
        
    Returns:
        List of (material_id, Crystal) tuples
    """
    mp_api = MaterialsProjectAPI(api_key)
    return mp_api.get_structures_by_formula(formula, limit=limit)


def download_stable_materials(elements: List[str], 
                             api_key: Optional[str] = None, 
                             output_dir: str = "structures") -> Dict[str, Crystal]:
    """
    Download stable materials containing specific elements.
    
    Args:
        elements: List of required elements
        api_key: Materials Project API key (optional if configured)
        output_dir: Directory to save structures
        
    Returns:
        Dictionary of material_id -> Crystal structures
    """
    mp_api = MaterialsProjectAPI(api_key)
    
    # Search for stable materials (energy_above_hull ≤ 0.01 eV)
    results = mp_api.search_by_criteria(
        stability_range=(0.0, 0.01),  # Nearly stable materials
        limit=1000
    )
    
    # Filter by elements
    filtered_ids = []
    for result in results:
        material_elements = set()
        if 'formula_pretty' in result:
            # Simple element extraction from formula
            import re
            elements_in_formula = re.findall(r'[A-Z][a-z]?', result['formula_pretty'])
            material_elements = set(elements_in_formula)
        
        if all(elem in material_elements for elem in elements):
            filtered_ids.append(result['material_id'])
    
    print(f"Found {len(filtered_ids)} stable materials with elements {elements}")
    
    return mp_api.download_structures_batch(filtered_ids[:100], output_dir)  # Limit for safety


def search_battery_materials(api_key: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Search for materials relevant to battery applications.
    
    Args:
        api_key: Materials Project API key
        limit: Maximum number of results
        
    Returns:
        List of material data
    """
    mp_api = MaterialsProjectAPI(api_key)
    
    # Search for materials with Li and reasonable band gap for electrodes
    return mp_api.search_by_criteria(
        elements=['Li'],
        band_gap_range=(0.0, 4.0),  # Semiconductors and metals
        stability_range=(0.0, 0.1),  # Stable materials
        limit=limit
    )