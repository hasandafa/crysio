"""
Crystal structure file parsers for various formats.

This module provides parsers for different crystallographic file formats
including CIF (Crystallographic Information File) and POSCAR (VASP format).
"""

import re
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod

from .crystal import Crystal, LatticeParameters, AtomicSite
from ..utils.exceptions import ParsingError, ValidationError


class BaseParser(ABC):
    """
    Abstract base class for all structure file parsers.
    
    This class defines the common interface that all parsers must implement.
    """
    
    @abstractmethod
    def parse(self, filepath_or_content: Union[str, Path]) -> Crystal:
        """
        Parse crystal structure from file or content string.
        
        Args:
            filepath_or_content: Path to file or file content as string
            
        Returns:
            Crystal: Parsed crystal structure
            
        Raises:
            ParsingError: If parsing fails
        """
        pass
    
    @abstractmethod
    def validate_format(self, content: str) -> bool:
        """
        Validate if content matches expected file format.
        
        Args:
            content: File content as string
            
        Returns:
            bool: True if format is valid, False otherwise
        """
        pass


class CIFParser(BaseParser):
    """
    Parser for CIF (Crystallographic Information File) format.
    
    CIF is a standard text file format for representing crystallographic information,
    widely used in materials science and crystallography databases.
    
    Supported CIF features:
    - Cell parameters (_cell_length_*, _cell_angle_*)
    - Atomic coordinates (_atom_site_*)
    - Space group information (_space_group_*)
    - Chemical formula (_chemical_formula_*)
    - Symmetry operations (_symmetry_equiv_pos_*)
    
    Examples:
        >>> parser = CIFParser()
        >>> crystal = parser.parse("structure.cif")
        >>> print(crystal.formula)
    """
    
    def __init__(self):
        self.current_file = None
        self.current_line = 0
        
        # CIF data tags we're interested in
        self.cell_tags = {
            '_cell_length_a': 'a',
            '_cell_length_b': 'b', 
            '_cell_length_c': 'c',
            '_cell_angle_alpha': 'alpha',
            '_cell_angle_beta': 'beta',
            '_cell_angle_gamma': 'gamma'
        }
        
        self.atom_site_tags = {
            '_atom_site_label': 'label',
            '_atom_site_type_symbol': 'element',
            '_atom_site_fract_x': 'x',
            '_atom_site_fract_y': 'y', 
            '_atom_site_fract_z': 'z',
            '_atom_site_occupancy': 'occupancy',
            '_atom_site_B_iso_or_equiv': 'b_iso',
            '_atom_site_U_iso_or_equiv': 'u_iso'
        }
    
    def parse(self, filepath_or_content: Union[str, Path]) -> Crystal:
        """
        Parse CIF file and return Crystal structure.
        
        Args:
            filepath_or_content: Path to CIF file or CIF content as string
            
        Returns:
            Crystal: Parsed crystal structure
            
        Raises:
            ParsingError: If CIF format is invalid or required data is missing
        """
        try:
            # Read content
            if isinstance(filepath_or_content, (str, Path)) and Path(filepath_or_content).exists():
                self.current_file = str(filepath_or_content)
                with open(filepath_or_content, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                self.current_file = "string_input"
                content = str(filepath_or_content)
            
            # Validate CIF format
            if not self.validate_format(content):
                raise ParsingError("Invalid CIF format", self.current_file)
            
            # Parse CIF data
            cif_data = self._parse_cif_content(content)
            
            # Extract crystal structure information
            lattice = self._extract_lattice_parameters(cif_data)
            sites = self._extract_atomic_sites(cif_data)
            space_group = self._extract_space_group(cif_data)
            formula = self._extract_formula(cif_data)
            
            # Create Crystal object
            crystal = Crystal(
                lattice=lattice,
                sites=sites,
                space_group=space_group,
                formula=formula,
                metadata=self._extract_metadata(cif_data)
            )
            
            return crystal
            
        except Exception as e:
            if isinstance(e, ParsingError):
                raise
            else:
                raise ParsingError(f"Unexpected error during CIF parsing: {str(e)}", self.current_file)
    
    def validate_format(self, content: str) -> bool:
        """
        Validate CIF format by checking for essential CIF markers.
        
        Args:
            content: CIF file content
            
        Returns:
            bool: True if valid CIF format
        """
        # Check for CIF data block
        if not re.search(r'^data_', content, re.MULTILINE):
            return False
            
        # Check for at least some cell parameters
        cell_params = ['_cell_length_a', '_cell_length_b', '_cell_length_c']
        if not any(param in content for param in cell_params):
            return False
            
        return True
    
    def _parse_cif_content(self, content: str) -> Dict[str, Any]:
        """
        Parse CIF content into structured data dictionary.
        
        Args:
            content: CIF file content
            
        Returns:
            Dict containing parsed CIF data
        """
        data = {}
        lines = content.split('\n')
        current_data_block = None
        current_loop = None
        loop_headers = []
        loop_data = []
        
        for i, line in enumerate(lines):
            self.current_line = i + 1
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Data block
            if line.startswith('data_'):
                current_data_block = line[5:]
                if current_data_block not in data:
                    data[current_data_block] = {}
                continue
            
            # Loop start
            if line.startswith('loop_'):
                current_loop = True
                loop_headers = []
                loop_data = []
                continue
            
            # Loop header (starts with _)
            if current_loop and line.startswith('_'):
                loop_headers.append(line)
                continue
            
            # Loop data or end of loop
            if current_loop:
                if line.startswith('_') or line.startswith('data_') or line.startswith('loop_'):
                    # End of current loop
                    self._process_loop_data(data, current_data_block, loop_headers, loop_data)
                    current_loop = None
                    # Process this line again
                    if line.startswith('_'):
                        # Single value assignment
                        self._parse_single_value(data, current_data_block, line)
                else:
                    # Loop data line
                    loop_data.append(line)
                continue
            
            # Single value assignment
            if line.startswith('_'):
                self._parse_single_value(data, current_data_block, line)
                continue
        
        # Process final loop if exists
        if current_loop and loop_headers:
            self._process_loop_data(data, current_data_block, loop_headers, loop_data)
        
        return data
    
    def _parse_single_value(self, data: Dict, data_block: str, line: str):
        """Parse single CIF tag-value pair."""
        parts = line.split(None, 1)  # Split on first whitespace
        if len(parts) >= 2:
            tag = parts[0]
            value = parts[1].strip("'\"")  # Remove quotes
            
            # Remove uncertainty values in parentheses
            value = re.sub(r'\([^)]*\)', '', value).strip()
            
            if data_block:
                data[data_block][tag] = value
            else:
                data[tag] = value
    
    def _process_loop_data(self, data: Dict, data_block: str, headers: List[str], loop_data: List[str]):
        """Process CIF loop data."""
        if not headers or not loop_data:
            return
        
        # Parse loop data lines
        all_values = []
        for line in loop_data:
            # Handle quoted strings and split properly
            values = self._split_cif_line(line)
            all_values.extend(values)
        
        # Organize into rows
        num_cols = len(headers)
        if len(all_values) % num_cols != 0:
            raise ParsingError(f"Inconsistent loop data: {len(all_values)} values for {num_cols} columns", 
                             self.current_file, self.current_line)
        
        num_rows = len(all_values) // num_cols
        
        # Create structured data
        loop_dict = {}
        for i, header in enumerate(headers):
            loop_dict[header] = []
            for j in range(num_rows):
                value = all_values[j * num_cols + i]
                # Remove uncertainty values
                value = re.sub(r'\([^)]*\)', '', value).strip("'\"")
                loop_dict[header].append(value)
        
        # Store in data dictionary
        target = data[data_block] if data_block else data
        target.update(loop_dict)
    
    def _split_cif_line(self, line: str) -> List[str]:
        """Split CIF line handling quoted strings properly."""
        values = []
        current_value = ""
        in_quotes = False
        quote_char = None
        
        i = 0
        while i < len(line):
            char = line[i]
            
            if not in_quotes:
                if char in ["'", '"']:
                    in_quotes = True
                    quote_char = char
                elif char.isspace():
                    if current_value:
                        values.append(current_value)
                        current_value = ""
                else:
                    current_value += char
            else:
                if char == quote_char:
                    in_quotes = False
                    quote_char = None
                else:
                    current_value += char
            
            i += 1
        
        if current_value:
            values.append(current_value)
        
        return values
    
    def _extract_lattice_parameters(self, cif_data: Dict) -> LatticeParameters:
        """Extract lattice parameters from CIF data."""
        # Get first data block
        if not cif_data:
            raise ParsingError("No data blocks found in CIF", self.current_file)
        
        # Find data block with cell parameters
        data_block = None
        for block_name, block_data in cif_data.items():
            if any(tag in block_data for tag in self.cell_tags.keys()):
                data_block = block_data
                break
        
        if not data_block:
            raise ParsingError("No lattice parameters found in CIF", self.current_file)
        
        # Extract cell parameters
        try:
            params = {}
            for cif_tag, param_name in self.cell_tags.items():
                if cif_tag in data_block:
                    value_str = str(data_block[cif_tag]).strip()
                    # Parse numeric value (handle uncertainties)
                    params[param_name] = self._parse_numeric_value(value_str)
                else:
                    raise ParsingError(f"Missing required parameter: {cif_tag}", self.current_file)
            
            return LatticeParameters(
                a=params['a'],
                b=params['b'], 
                c=params['c'],
                alpha=params['alpha'],
                beta=params['beta'],
                gamma=params['gamma']
            )
            
        except (ValueError, KeyError) as e:
            raise ParsingError(f"Invalid lattice parameters: {str(e)}", self.current_file)
    
    def _extract_atomic_sites(self, cif_data: Dict) -> List[AtomicSite]:
        """Extract atomic sites from CIF data."""
        # Find data block with atom sites
        data_block = None
        for block_name, block_data in cif_data.items():
            if any(tag in block_data for tag in self.atom_site_tags.keys()):
                data_block = block_data
                break
        
        if not data_block:
            raise ParsingError("No atomic site data found in CIF", self.current_file)
        
        # Check required atom site data
        required_tags = ['_atom_site_type_symbol', '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z']
        missing_tags = [tag for tag in required_tags if tag not in data_block]
        if missing_tags:
            raise ParsingError(f"Missing required atom site tags: {missing_tags}", self.current_file)
        
        # Extract atomic sites
        sites = []
        num_sites = len(data_block['_atom_site_type_symbol'])
        
        for i in range(num_sites):
            try:
                # Element symbol
                element = data_block['_atom_site_type_symbol'][i].strip()
                # Remove charge/oxidation state info (e.g., 'Fe3+' -> 'Fe')
                element = re.sub(r'[0-9+-]+', '', element)
                
                # Fractional coordinates
                x = self._parse_numeric_value(data_block['_atom_site_fract_x'][i])
                y = self._parse_numeric_value(data_block['_atom_site_fract_y'][i])
                z = self._parse_numeric_value(data_block['_atom_site_fract_z'][i])
                position = np.array([x, y, z])
                
                # Optional parameters
                occupancy = 1.0
                if '_atom_site_occupancy' in data_block:
                    occupancy = self._parse_numeric_value(data_block['_atom_site_occupancy'][i])
                
                label = None
                if '_atom_site_label' in data_block:
                    label = data_block['_atom_site_label'][i].strip()
                
                # Create atomic site
                site = AtomicSite(
                    element=element,
                    position=position,
                    occupancy=occupancy,
                    label=label
                )
                sites.append(site)
                
            except (ValueError, IndexError) as e:
                raise ParsingError(f"Error parsing atom site {i}: {str(e)}", self.current_file)
        
        return sites
    
    def _parse_numeric_value(self, value_str: str) -> float:
        """Parse numeric value from CIF, handling uncertainties."""
        # Remove uncertainty in parentheses
        clean_value = re.sub(r'\([^)]*\)', '', str(value_str)).strip()
        
        # Handle special cases
        if clean_value in ['.', '?']:
            raise ValueError(f"Undefined or missing value: {value_str}")
        
        try:
            return float(clean_value)
        except ValueError:
            raise ValueError(f"Cannot convert to float: {value_str}")
    
    def _extract_space_group(self, cif_data: Dict) -> Optional[str]:
        """Extract space group information from CIF data."""
        # Look for space group tags
        space_group_tags = [
            '_space_group_name_H-M_alt',
            '_space_group_name_H-M',
            '_symmetry_space_group_name_H-M',
            '_symmetry_space_group_name_Hall'
        ]
        
        for block_data in cif_data.values():
            for tag in space_group_tags:
                if tag in block_data:
                    return str(block_data[tag]).strip("'\"")
        
        return None
    
    def _extract_formula(self, cif_data: Dict) -> Optional[str]:
        """Extract chemical formula from CIF data."""
        formula_tags = [
            '_chemical_formula_sum',
            '_chemical_formula_structural',
            '_chemical_formula_moiety'
        ]
        
        for block_data in cif_data.values():
            for tag in formula_tags:
                if tag in block_data:
                    return str(block_data[tag]).strip("'\"")
        
        return None
    
    def _extract_metadata(self, cif_data: Dict) -> Dict[str, Any]:
        """Extract additional metadata from CIF data."""
        metadata = {}
        
        # Common metadata tags
        metadata_tags = {
            '_chemical_name_common': 'common_name',
            '_chemical_name_mineral': 'mineral_name',
            '_cell_volume': 'cell_volume',
            '_cell_formula_units_Z': 'formula_units_Z',
            '_diffrn_ambient_temperature': 'temperature',
            '_diffrn_ambient_pressure': 'pressure'
        }
        
        for block_data in cif_data.values():
            for cif_tag, meta_key in metadata_tags.items():
                if cif_tag in block_data:
                    value = block_data[cif_tag]
                    try:
                        # Try to parse as number
                        metadata[meta_key] = self._parse_numeric_value(value)
                    except ValueError:
                        # Keep as string
                        metadata[meta_key] = str(value).strip("'\"")
        
        return metadata


class POSCARParser(BaseParser):
    """
    Parser for POSCAR/CONTCAR format (VASP).
    
    POSCAR is the input file format for the Vienna Ab initio Simulation Package (VASP)
    and contains crystal structure information in a specific format.
    
    POSCAR format structure:
    1. Comment line
    2. Scaling factor
    3. Lattice vectors (3 lines)
    4. Element names (optional, VASP 5+)
    5. Number of atoms per element
    6. Coordinate type (Direct/Cartesian)
    7. Atomic coordinates
    
    Examples:
        >>> parser = POSCARParser()
        >>> crystal = parser.parse("POSCAR")
        >>> print(crystal.lattice.a)
    """
    
    def __init__(self):
        self.current_file = None
        self.current_line = 0
    
    def parse(self, filepath_or_content: Union[str, Path]) -> Crystal:
        """
        Parse POSCAR file and return Crystal structure.
        
        Args:
            filepath_or_content: Path to POSCAR file or content as string
            
        Returns:
            Crystal: Parsed crystal structure
            
        Raises:
            ParsingError: If POSCAR format is invalid
        """
        try:
            # Read content
            if isinstance(filepath_or_content, (str, Path)) and Path(filepath_or_content).exists():
                self.current_file = str(filepath_or_content)
                with open(filepath_or_content, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            else:
                self.current_file = "string_input"
                lines = str(filepath_or_content).split('\n')
            
            # Clean lines
            lines = [line.strip() for line in lines if line.strip()]
            
            if not self.validate_format('\n'.join(lines)):
                raise ParsingError("Invalid POSCAR format", self.current_file)
            
            # Parse POSCAR sections
            comment = self._parse_comment(lines[0])
            scaling_factor = self._parse_scaling_factor(lines[1])
            lattice_vectors = self._parse_lattice_vectors(lines[2:5], scaling_factor)
            elements, atom_counts, line_offset = self._parse_elements_and_counts(lines[5:])
            coordinate_type = self._parse_coordinate_type(lines[5 + line_offset])
            atomic_positions = self._parse_atomic_positions(
                lines[6 + line_offset:], elements, atom_counts, coordinate_type, lattice_vectors
            )
            
            # Create Crystal object
            crystal = Crystal(
                lattice=self._vectors_to_lattice_params(lattice_vectors),
                sites=atomic_positions,
                formula=self._generate_formula_from_composition(elements, atom_counts),
                metadata={
                    'comment': comment,
                    'scaling_factor': scaling_factor,
                    'coordinate_type': coordinate_type
                }
            )
            
            return crystal
            
        except Exception as e:
            if isinstance(e, ParsingError):
                raise
            else:
                raise ParsingError(f"Unexpected error during POSCAR parsing: {str(e)}", self.current_file)
    
    def validate_format(self, content: str) -> bool:
        """
        Validate POSCAR format.
        
        Args:
            content: POSCAR file content
            
        Returns:
            bool: True if valid POSCAR format
        """
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Must have at least 8 lines
        if len(lines) < 8:
            return False
        
        # Line 2 should be a number (scaling factor)
        try:
            float(lines[1])
        except ValueError:
            return False
        
        # Lines 3-5 should be lattice vectors (3 numbers each)
        for i in range(2, 5):
            try:
                values = lines[i].split()
                if len(values) != 3:
                    return False
                [float(x) for x in values]
            except (ValueError, IndexError):
                return False
        
        return True
    
    def _parse_comment(self, line: str) -> str:
        """Parse comment line."""
        return line.strip()
    
    def _parse_scaling_factor(self, line: str) -> float:
        """Parse scaling factor."""
        try:
            return float(line.strip())
        except ValueError:
            raise ParsingError(f"Invalid scaling factor: {line}", self.current_file, 2)
    
    def _parse_lattice_vectors(self, lines: List[str], scaling: float) -> np.ndarray:
        """Parse lattice vectors and apply scaling."""
        vectors = np.zeros((3, 3))
        
        for i, line in enumerate(lines):
            try:
                values = [float(x) for x in line.split()]
                if len(values) != 3:
                    raise ValueError(f"Expected 3 values, got {len(values)}")
                vectors[i] = np.array(values) * scaling
            except ValueError as e:
                raise ParsingError(f"Invalid lattice vector on line {i+3}: {str(e)}", self.current_file, i+3)
        
        return vectors
    
    def _parse_elements_and_counts(self, lines: List[str]) -> Tuple[List[str], List[int], int]:
        """
        Parse element names and atom counts.
        
        Returns:
            Tuple of (elements, counts, line_offset)
        """
        # Try to parse first line as element names (VASP 5+ format)
        first_line = lines[0].split()
        
        # Check if first line contains element symbols
        if all(item.isalpha() and len(item) <= 2 for item in first_line):
            # VASP 5+ format with element names
            elements = first_line
            try:
                counts = [int(x) for x in lines[1].split()]
                return elements, counts, 1
            except ValueError:
                raise ParsingError("Invalid atom counts", self.current_file)
        else:
            # VASP 4 format without element names
            try:
                counts = [int(x) for x in first_line]
                # Generate generic element names
                elements = [f"El{i+1}" for i in range(len(counts))]
                return elements, counts, 0
            except ValueError:
                raise ParsingError("Invalid atom counts", self.current_file)
    
    def _parse_coordinate_type(self, line: str) -> str:
        """Parse coordinate type (Direct/Cartesian)."""
        coord_type = line.strip().lower()
        if coord_type.startswith('d'):
            return 'Direct'
        elif coord_type.startswith('c') or coord_type.startswith('k'):
            return 'Cartesian'
        else:
            raise ParsingError(f"Unknown coordinate type: {line}", self.current_file)
    
    def _parse_atomic_positions(
        self, 
        lines: List[str], 
        elements: List[str], 
        counts: List[int], 
        coord_type: str,
        lattice_vectors: np.ndarray
    ) -> List[AtomicSite]:
        """Parse atomic positions."""
        sites = []
        line_idx = 0
        
        for element, count in zip(elements, counts):
            for i in range(count):
                if line_idx >= len(lines):
                    raise ParsingError("Not enough atomic positions", self.current_file)
                
                try:
                    coords = [float(x) for x in lines[line_idx].split()[:3]]
                    position = np.array(coords)
                    
                    # Convert Cartesian to fractional if needed
                    if coord_type == 'Cartesian':
                        # Convert to fractional coordinates
                        lattice_matrix = lattice_vectors.T
                        position = np.linalg.solve(lattice_matrix, position)
                    
                    site = AtomicSite(
                        element=element,
                        position=position,
                        occupancy=1.0,
                        label=f"{element}{i+1}"
                    )
                    sites.append(site)
                    
                except (ValueError, IndexError) as e:
                    raise ParsingError(f"Invalid atomic position on line {line_idx}: {str(e)}", 
                                     self.current_file)
                
                line_idx += 1
        
        return sites
    
    def _vectors_to_lattice_params(self, vectors: np.ndarray) -> LatticeParameters:
        """Convert lattice vectors to lattice parameters."""
        a_vec, b_vec, c_vec = vectors
        
        # Calculate lengths
        a = np.linalg.norm(a_vec)
        b = np.linalg.norm(b_vec)
        c = np.linalg.norm(c_vec)
        
        # Calculate angles
        alpha = math.degrees(math.acos(np.clip(np.dot(b_vec, c_vec) / (b * c), -1, 1)))
        beta = math.degrees(math.acos(np.clip(np.dot(a_vec, c_vec) / (a * c), -1, 1)))
        gamma = math.degrees(math.acos(np.clip(np.dot(a_vec, b_vec) / (a * b), -1, 1)))
        
        return LatticeParameters(a, b, c, alpha, beta, gamma)
    
    def _generate_formula_from_composition(self, elements: List[str], counts: List[int]) -> str:
        """Generate chemical formula from element composition."""
        formula_parts = []
        for element, count in zip(elements, counts):
            if count == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}{count}")
        return ''.join(formula_parts)


# Utility functions for auto-detection and parser selection
def auto_detect_format(filepath_or_content: Union[str, Path]) -> str:
    """
    Auto-detect file format based on content or filename.
    
    Args:
        filepath_or_content: Path to file or file content
        
    Returns:
        str: Detected format ('cif', 'poscar', 'xyz', etc.)
        
    Raises:
        ParsingError: If format cannot be detected
    """
    # Check filename extension first
    if isinstance(filepath_or_content, (str, Path)):
        path = Path(filepath_or_content)
        if path.exists():
            # Check file extension
            ext = path.suffix.lower()
            if ext == '.cif':
                return 'cif'
            elif ext in ['.vasp', '.poscar', '.contcar']:
                return 'poscar'
            elif path.name.upper() in ['POSCAR', 'CONTCAR']:
                return 'poscar'
            
            # Read content for analysis
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = str(filepath_or_content)
    else:
        content = str(filepath_or_content)
    
    # Analyze content
    if re.search(r'^data_', content, re.MULTILINE):
        return 'cif'
    elif '_cell_length_' in content or '_atom_site_' in content:
        return 'cif'
    else:
        # Try POSCAR validation
        poscar_parser = POSCARParser()
        if poscar_parser.validate_format(content):
            return 'poscar'
    
    raise ParsingError("Cannot detect file format")


def get_parser(format_name: str) -> BaseParser:
    """
    Get parser instance for specified format.
    
    Args:
        format_name: Format name ('cif', 'poscar', etc.)
        
    Returns:
        BaseParser: Parser instance
        
    Raises:
        ValueError: If format is not supported
    """
    format_name = format_name.lower()
    
    if format_name == 'cif':
        return CIFParser()
    elif format_name in ['poscar', 'vasp', 'contcar']:
        return POSCARParser()
    else:
        raise ValueError(f"Unsupported format: {format_name}")