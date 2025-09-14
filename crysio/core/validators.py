"""
Structure validation module for crystal structures.

This module provides validation tools to ensure crystal structures are
physically reasonable and mathematically consistent.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .crystal import Crystal, LatticeParameters, AtomicSite
from ..utils.exceptions import ValidationError


@dataclass
class ValidationResult:
    """Result of structure validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    
    def add_error(self, message: str):
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add validation warning."""
        self.warnings.append(message)


class BaseValidator(ABC):
    """Base class for structure validators."""
    
    @abstractmethod
    def validate(self, crystal: Crystal) -> ValidationResult:
        """Validate crystal structure."""
        pass


class LatticeValidator(BaseValidator):
    """Validator for lattice parameters."""
    
    def __init__(self, 
                 min_length: float = 0.1,
                 max_length: float = 100.0,
                 min_angle: float = 10.0,
                 max_angle: float = 170.0):
        """
        Initialize lattice validator.
        
        Args:
            min_length: Minimum lattice parameter (Å)
            max_length: Maximum lattice parameter (Å)
            min_angle: Minimum lattice angle (degrees)
            max_angle: Maximum lattice angle (degrees)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_angle = min_angle
        self.max_angle = max_angle
    
    def validate(self, crystal: Crystal) -> ValidationResult:
        """Validate lattice parameters."""
        result = ValidationResult(True, [], [], {})
        lattice = crystal.lattice
        
        # Check lattice lengths
        lengths = [lattice.a, lattice.b, lattice.c]
        for i, length in enumerate(lengths):
            if length < self.min_length:
                result.add_error(f"Lattice parameter {['a', 'b', 'c'][i]} too small: {length:.3f} Å")
            elif length > self.max_length:
                result.add_error(f"Lattice parameter {['a', 'b', 'c'][i]} too large: {length:.3f} Å")
        
        # Check lattice angles
        angles = [lattice.alpha, lattice.beta, lattice.gamma]
        for i, angle in enumerate(angles):
            if angle < self.min_angle:
                result.add_error(f"Lattice angle {['α', 'β', 'γ'][i]} too small: {angle:.1f}°")
            elif angle > self.max_angle:
                result.add_error(f"Lattice angle {['α', 'β', 'γ'][i]} too large: {angle:.1f}°")
        
        # Check angle sum constraint for valid lattice
        if sum(angles) >= 360.0:
            result.add_error(f"Sum of lattice angles too large: {sum(angles):.1f}° (must be < 360°)")
        
        # Calculate volume and check if positive
        try:
            volume = lattice.volume()
            if volume <= 0:
                result.add_error(f"Invalid lattice volume: {volume:.3f} Å³")
            result.metrics['volume'] = volume
        except Exception as e:
            result.add_error(f"Cannot calculate lattice volume: {str(e)}")
        
        # Crystal system identification
        result.metrics['crystal_system'] = self._identify_crystal_system(lattice)
        
        return result
    
    def _identify_crystal_system(self, lattice: LatticeParameters) -> str:
        """Identify crystal system based on lattice parameters."""
        a, b, c = lattice.a, lattice.b, lattice.c
        alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
        
        tol = 1e-3  # Tolerance for floating point comparison
        angle_tol = 1.0  # Tolerance for angles in degrees
        
        # Check if lengths are equal within tolerance
        a_eq_b = abs(a - b) < tol
        b_eq_c = abs(b - c) < tol
        a_eq_c = abs(a - c) < tol
        
        # Check if angles are 90 degrees
        alpha_90 = abs(alpha - 90) < angle_tol
        beta_90 = abs(beta - 90) < angle_tol
        gamma_90 = abs(gamma - 90) < angle_tol
        
        # Check if angles are 120 degrees
        gamma_120 = abs(gamma - 120) < angle_tol
        
        # Crystal system identification
        if a_eq_b and b_eq_c and alpha_90 and beta_90 and gamma_90:
            return "Cubic"
        elif alpha_90 and beta_90 and gamma_90:
            if a_eq_b:
                return "Tetragonal"
            else:
                return "Orthorhombic"
        elif a_eq_b and alpha_90 and beta_90 and gamma_120:
            return "Hexagonal"
        elif a_eq_b and alpha == beta and gamma_90:
            return "Trigonal"
        elif alpha_90 and gamma_90:
            return "Monoclinic"
        else:
            return "Triclinic"


class AtomicPositionValidator(BaseValidator):
    """Validator for atomic positions."""
    
    def __init__(self, 
                 min_distance: float = 0.5,
                 check_overlaps: bool = True,
                 occupancy_tolerance: float = 1e-3):
        """
        Initialize atomic position validator.
        
        Args:
            min_distance: Minimum allowed distance between atoms (Å)
            check_overlaps: Whether to check for atomic overlaps
            occupancy_tolerance: Tolerance for occupancy sum validation
        """
        self.min_distance = min_distance
        self.check_overlaps = check_overlaps
        self.occupancy_tolerance = occupancy_tolerance
    
    def validate(self, crystal: Crystal) -> ValidationResult:
        """Validate atomic positions."""
        result = ValidationResult(True, [], [], {})
        
        # Check for empty structure
        if not crystal.sites:
            result.add_error("Structure contains no atoms")
            return result
        
        # Validate individual atomic sites
        for i, site in enumerate(crystal.sites):
            self._validate_site(site, i, result)
        
        # Check for atomic overlaps
        if self.check_overlaps:
            overlaps = self._check_atomic_overlaps(crystal)
            if overlaps:
                for overlap in overlaps:
                    result.add_error(f"Atoms too close: {overlap}")
        
        # Calculate coordination statistics
        coord_stats = self._calculate_coordination_statistics(crystal)
        result.metrics.update(coord_stats)
        
        return result
    
    def _validate_site(self, site: AtomicSite, index: int, result: ValidationResult):
        """Validate individual atomic site."""
        # Check element symbol
        if not site.element or not site.element.strip():
            result.add_error(f"Atom {index}: Empty element symbol")
        
        # Check fractional coordinates
        for i, coord in enumerate(site.position):
            if coord < 0 or coord > 1:
                result.add_warning(f"Atom {index}: Fractional coordinate {['x', 'y', 'z'][i]} = {coord:.3f} outside [0,1]")
        
        # Check occupancy
        if site.occupancy < 0 or site.occupancy > 1:
            result.add_error(f"Atom {index}: Invalid occupancy {site.occupancy:.3f}")
        elif site.occupancy < self.occupancy_tolerance:
            result.add_warning(f"Atom {index}: Very low occupancy {site.occupancy:.3f}")
    
    def _check_atomic_overlaps(self, crystal: Crystal) -> List[str]:
        """Check for overlapping atoms."""
        overlaps = []
        cartesian_coords = crystal.to_cartesian_coordinates()
        
        for i in range(len(crystal.sites)):
            for j in range(i + 1, len(crystal.sites)):
                distance = np.linalg.norm(cartesian_coords[i] - cartesian_coords[j])
                
                if distance < self.min_distance:
                    element_i = crystal.sites[i].element
                    element_j = crystal.sites[j].element
                    overlaps.append(f"{element_i}{i+1}-{element_j}{j+1} distance: {distance:.3f} Å")
        
        return overlaps
    
    def _calculate_coordination_statistics(self, crystal: Crystal) -> Dict[str, Any]:
        """Calculate coordination environment statistics."""
        cartesian_coords = crystal.to_cartesian_coordinates()
        coordination_numbers = []
        
        # Calculate coordination numbers (using 3.5 Å cutoff)
        cutoff = 3.5
        for i in range(len(crystal.sites)):
            coord_num = 0
            for j in range(len(crystal.sites)):
                if i != j:
                    distance = np.linalg.norm(cartesian_coords[i] - cartesian_coords[j])
                    if distance <= cutoff:
                        coord_num += 1
            coordination_numbers.append(coord_num)
        
        if coordination_numbers:
            return {
                'coordination_numbers': coordination_numbers,
                'average_coordination': float(np.mean(coordination_numbers)),
                'min_coordination': int(np.min(coordination_numbers)),
                'max_coordination': int(np.max(coordination_numbers)),
            }
        else:
            return {}


class CompositionValidator(BaseValidator):
    """Validator for chemical composition."""
    
    def __init__(self, check_charge_neutrality: bool = True):
        """
        Initialize composition validator.
        
        Args:
            check_charge_neutrality: Whether to check charge neutrality
        """
        self.check_charge_neutrality = check_charge_neutrality
        self.common_oxidation_states = self._load_oxidation_states()
    
    def validate(self, crystal: Crystal) -> ValidationResult:
        """Validate chemical composition."""
        result = ValidationResult(True, [], [], {})
        
        # Analyze composition
        composition = self._analyze_composition(crystal)
        result.metrics['composition'] = composition
        
        # Check for common element symbols
        invalid_elements = self._check_element_symbols(crystal)
        if invalid_elements:
            for element in invalid_elements:
                result.add_warning(f"Unusual element symbol: {element}")
        
        # Check charge neutrality if oxidation states available
        if self.check_charge_neutrality:
            charge_balance = self._check_charge_neutrality(crystal)
            if charge_balance is not None:
                result.metrics['total_charge'] = charge_balance
                if abs(charge_balance) > 1e-3:
                    result.add_error(f"Structure not charge neutral: total charge = {charge_balance:.3f}")
        
        return result
    
    def _analyze_composition(self, crystal: Crystal) -> Dict[str, int]:
        """Analyze chemical composition."""
        composition = {}
        for site in crystal.sites:
            element = site.element
            if element in composition:
                composition[element] += 1
            else:
                composition[element] = 1
        return composition
    
    def _check_element_symbols(self, crystal: Crystal) -> List[str]:
        """Check for valid element symbols."""
        # Common element symbols (simplified periodic table)
        valid_elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
        }
        
        invalid_elements = []
        for site in crystal.sites:
            if site.element not in valid_elements:
                invalid_elements.append(site.element)
        
        return list(set(invalid_elements))  # Remove duplicates
    
    def _check_charge_neutrality(self, crystal: Crystal) -> Optional[float]:
        """Check charge neutrality if oxidation states are available."""
        total_charge = 0.0
        has_oxidation_states = False
        
        for site in crystal.sites:
            if site.oxidation_state is not None:
                total_charge += site.oxidation_state * site.occupancy
                has_oxidation_states = True
        
        return total_charge if has_oxidation_states else None
    
    def _load_oxidation_states(self) -> Dict[str, List[int]]:
        """Load common oxidation states for elements."""
        # Common oxidation states for elements
        return {
            'H': [-1, 1],
            'Li': [1],
            'Be': [2],
            'B': [3],
            'C': [-4, 2, 4],
            'N': [-3, 3, 5],
            'O': [-2],
            'F': [-1],
            'Na': [1],
            'Mg': [2],
            'Al': [3],
            'Si': [4],
            'P': [-3, 3, 5],
            'S': [-2, 4, 6],
            'Cl': [-1, 1, 3, 5, 7],
            'K': [1],
            'Ca': [2],
            'Ti': [2, 3, 4],
            'V': [2, 3, 4, 5],
            'Cr': [2, 3, 6],
            'Mn': [2, 3, 4, 6, 7],
            'Fe': [2, 3],
            'Co': [2, 3],
            'Ni': [2],
            'Cu': [1, 2],
            'Zn': [2],
        }


class StructureValidator:
    """
    Main structure validator that combines multiple validation checks.
    
    This class orchestrates various validation checks for crystal structures
    to ensure they are physically reasonable and mathematically consistent.
    
    Examples:
        >>> validator = StructureValidator()
        >>> result = validator.validate(crystal)
        >>> if result.is_valid:
        ...     print("Structure is valid")
        >>> else:
        ...     print("Validation errors:", result.errors)
    """
    
    def __init__(self, 
                 include_lattice: bool = True,
                 include_positions: bool = True,
                 include_composition: bool = True,
                 strict_mode: bool = False):
        """
        Initialize structure validator.
        
        Args:
            include_lattice: Include lattice parameter validation
            include_positions: Include atomic position validation
            include_composition: Include composition validation
            strict_mode: Use strict validation criteria
        """
        self.validators = []
        
        if include_lattice:
            if strict_mode:
                self.validators.append(LatticeValidator(min_length=1.0, max_length=50.0))
            else:
                self.validators.append(LatticeValidator())
        
        if include_positions:
            if strict_mode:
                self.validators.append(AtomicPositionValidator(min_distance=1.0))
            else:
                self.validators.append(AtomicPositionValidator())
        
        if include_composition:
            self.validators.append(CompositionValidator())
    
    def validate(self, crystal: Crystal) -> ValidationResult:
        """
        Validate crystal structure using all configured validators.
        
        Args:
            crystal: Crystal structure to validate
            
        Returns:
            ValidationResult: Combined validation results
            
        Raises:
            ValidationError: If validation fails critically
        """
        try:
            combined_result = ValidationResult(True, [], [], {})
            
            for validator in self.validators:
                result = validator.validate(crystal)
                
                # Combine results
                combined_result.errors.extend(result.errors)
                combined_result.warnings.extend(result.warnings)
                combined_result.metrics.update(result.metrics)
                
                if not result.is_valid:
                    combined_result.is_valid = False
            
            return combined_result
            
        except Exception as e:
            raise ValidationError(f"Validation failed: {str(e)}", crystal.formula)
    
    def validate_and_raise(self, crystal: Crystal):
        """
        Validate structure and raise exception if invalid.
        
        Args:
            crystal: Crystal structure to validate
            
        Raises:
            ValidationError: If structure is invalid
        """
        result = self.validate(crystal)
        if not result.is_valid:
            error_msg = "; ".join(result.errors)
            raise ValidationError(f"Structure validation failed: {error_msg}", crystal.formula)
    
    def get_validation_summary(self, crystal: Crystal) -> str:
        """
        Get human-readable validation summary.
        
        Args:
            crystal: Crystal structure to validate
            
        Returns:
            str: Formatted validation summary
        """
        result = self.validate(crystal)
        
        summary = [f"Validation Summary for {crystal.formula}"]
        summary.append("=" * 40)
        
        if result.is_valid:
            summary.append("✅ Structure is VALID")
        else:
            summary.append("❌ Structure is INVALID")
        
        if result.errors:
            summary.append(f"\nErrors ({len(result.errors)}):")
            for error in result.errors:
                summary.append(f"  • {error}")
        
        if result.warnings:
            summary.append(f"\nWarnings ({len(result.warnings)}):")
            for warning in result.warnings:
                summary.append(f"  • {warning}")
        
        if result.metrics:
            summary.append("\nMetrics:")
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    summary.append(f"  • {key}: {value}")
                elif isinstance(value, dict):
                    summary.append(f"  • {key}: {value}")
        
        return "\n".join(summary)