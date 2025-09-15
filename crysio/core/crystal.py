"""
Core Crystal class for representing crystal structures.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class LatticeParameters:
    """Lattice parameters for crystal structure."""
    a: float  # Lattice parameter a (Å)
    b: float  # Lattice parameter b (Å)  
    c: float  # Lattice parameter c (Å)
    alpha: float  # Angle alpha (degrees)
    beta: float   # Angle beta (degrees) 
    gamma: float  # Angle gamma (degrees)
    
    def volume(self) -> float:
        """Calculate unit cell volume."""
        # Convert angles to radians
        alpha_rad = np.radians(self.alpha)
        beta_rad = np.radians(self.beta)
        gamma_rad = np.radians(self.gamma)
        
        # Calculate volume using formula
        volume = self.a * self.b * self.c * np.sqrt(
            1 + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
            - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2
        )
        return volume
    
    def lattice_matrix(self) -> np.ndarray:
        """
        Generate the 3x3 lattice matrix from parameters.
        
        Returns:
            np.ndarray: 3x3 lattice matrix where columns are lattice vectors
        """
        # Convert angles to radians
        alpha_rad = np.radians(self.alpha)
        beta_rad = np.radians(self.beta)
        gamma_rad = np.radians(self.gamma)
        
        # Calculate lattice matrix
        cos_alpha = np.cos(alpha_rad)
        cos_beta = np.cos(beta_rad) 
        cos_gamma = np.cos(gamma_rad)
        sin_gamma = np.sin(gamma_rad)
        
        # Lattice vectors as columns
        matrix = np.array([
            [self.a, self.b * cos_gamma, self.c * cos_beta],
            [0, self.b * sin_gamma, self.c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma],
            [0, 0, self.c * self.volume() / (self.a * self.b * sin_gamma)]
        ])
        
        return matrix
    
    # Compatibility method for visualization modules
    def get_lattice_matrix(self) -> np.ndarray:
        """Compatibility method - alias for lattice_matrix.""" 
        return self.lattice_matrix()
    
    @property
    def crystal_system(self) -> str:
        """Determine crystal system based on lattice parameters."""
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        
        # Tolerance for comparison
        tol = 1e-5
        angle_tol = 0.1  # degrees
        
        # Helper functions
        def is_equal(x, y, tolerance=tol):
            return abs(x - y) < tolerance
        
        def is_90(angle, tolerance=angle_tol):
            return abs(angle - 90.0) < tolerance
        
        def is_120(angle, tolerance=angle_tol):
            return abs(angle - 120.0) < tolerance
        
        # Cubic: a = b = c, α = β = γ = 90°
        if is_equal(a, b) and is_equal(b, c) and is_90(alpha) and is_90(beta) and is_90(gamma):
            return "cubic"
        
        # Tetragonal: a = b ≠ c, α = β = γ = 90°
        elif is_equal(a, b) and not is_equal(a, c) and is_90(alpha) and is_90(beta) and is_90(gamma):
            return "tetragonal"
        
        # Orthorhombic: a ≠ b ≠ c, α = β = γ = 90°
        elif not is_equal(a, b) and not is_equal(b, c) and not is_equal(a, c) and is_90(alpha) and is_90(beta) and is_90(gamma):
            return "orthorhombic"
        
        # Hexagonal: a = b ≠ c, α = β = 90°, γ = 120°
        elif is_equal(a, b) and not is_equal(a, c) and is_90(alpha) and is_90(beta) and is_120(gamma):
            return "hexagonal"
        
        # Trigonal/Rhombohedral: a = b = c, α = β = γ ≠ 90°
        elif is_equal(a, b) and is_equal(b, c) and is_equal(alpha, beta) and is_equal(beta, gamma) and not is_90(alpha):
            return "trigonal"
        
        # Monoclinic: a ≠ b ≠ c, α = γ = 90° ≠ β
        elif not is_equal(a, b) and not is_equal(b, c) and is_90(alpha) and not is_90(beta) and is_90(gamma):
            return "monoclinic"
        
        # Triclinic: a ≠ b ≠ c, α ≠ β ≠ γ ≠ 90°
        else:
            return "triclinic"


@dataclass 
class AtomicSite:
    """Represents an atomic site in crystal structure."""
    element: str                    # Element symbol (e.g., 'Fe', 'O')
    position: np.ndarray           # Fractional coordinates [x, y, z]
    occupancy: float = 1.0         # Site occupancy (0.0 to 1.0)
    oxidation_state: Optional[float] = None  # Oxidation state
    magnetic_moment: Optional[float] = None  # Magnetic moment
    thermal_displacement: Optional[Dict[str, float]] = None  # Thermal parameters
    label: Optional[str] = None    # Atom label/name
    
    def __post_init__(self):
        """Validate and process atomic site data."""
        # Ensure position is numpy array
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position)
        
        # Validate fractional coordinates (should be between 0 and 1)
        if not (0 <= self.position[0] <= 1 and 0 <= self.position[1] <= 1 and 0 <= self.position[2] <= 1):
            # Wrap coordinates to unit cell
            self.position = self.position % 1.0
    
    def cartesian_position(self, lattice_matrix: np.ndarray) -> np.ndarray:
        """
        Convert fractional coordinates to cartesian coordinates.
        
        Args:
            lattice_matrix: 3x3 lattice matrix
            
        Returns:
            np.ndarray: Cartesian coordinates [x, y, z] in Å
        """
        return lattice_matrix @ self.position
    
    # Compatibility property for visualization modules
    @property  
    def fractional_coords(self) -> np.ndarray:
        """Compatibility property - alias for position."""
        return self.position


class Crystal:
    """
    Main class for representing crystal structures.
    
    This class encapsulates all information about a crystal structure including
    lattice parameters, atomic positions, and metadata. It provides methods
    for structure manipulation, analysis, and conversion.
    
    Attributes:
        lattice (LatticeParameters): Crystal lattice parameters
        sites (List[AtomicSite]): List of atomic sites in the structure
        space_group (str): Space group symbol (optional)
        formula (str): Chemical formula (optional) 
        metadata (Dict): Additional metadata
        
    Examples:
        >>> lattice = LatticeParameters(5.0, 5.0, 5.0, 90, 90, 90)  # Cubic
        >>> sites = [AtomicSite('Na', [0, 0, 0]), AtomicSite('Cl', [0.5, 0.5, 0.5])]
        >>> crystal = Crystal(lattice, sites)
        >>> print(crystal.formula)  # 'NaCl'
    """
    
    def __init__(
        self, 
        lattice: LatticeParameters,
        sites: List[AtomicSite],
        space_group: Optional[str] = None,
        formula: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.lattice = lattice
        self.sites = sites
        self.space_group = space_group
        self.formula = formula or self._generate_formula()
        self.metadata = metadata or {}
        
    def _generate_formula(self) -> str:
        """Generate chemical formula from atomic sites."""
        from collections import Counter
        
        # Count elements
        element_counts = Counter(site.element for site in self.sites)
        
        # Sort elements (metals first, then non-metals alphabetically)
        def sort_key(item):
            element, count = item
            # Simple heuristic: assume metals have lower atomic numbers for common elements
            metal_priority = {'Li': 1, 'Na': 2, 'K': 3, 'Ca': 4, 'Mg': 5, 'Al': 6, 'Fe': 7, 'Ti': 8}
            return metal_priority.get(element, 100 + ord(element[0]))
        
        sorted_elements = sorted(element_counts.items(), key=sort_key)
        
        # Build formula string
        formula_parts = []
        for element, count in sorted_elements:
            if count == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}{count}")
                
        return ''.join(formula_parts)
    
    @property
    def num_atoms(self) -> int:
        """Number of atoms in the unit cell."""
        return len(self.sites)
    
    @property
    def volume(self) -> float:
        """Unit cell volume in Å³."""
        return self.lattice.volume()
    
    @property
    def density(self) -> float:
        """
        Crystal density in g/cm³ (approximate).
        
        Note: This is a rough approximation based on atomic masses.
        """
        # Atomic masses (simplified)
        atomic_masses = {
            'H': 1.008, 'Li': 6.94, 'C': 12.01, 'N': 14.01, 'O': 15.999,
            'Na': 22.99, 'Mg': 24.31, 'Al': 26.98, 'Si': 28.085, 'P': 30.97,
            'S': 32.065, 'Cl': 35.453, 'K': 39.10, 'Ca': 40.078, 'Fe': 55.845,
            'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38, 'Ti': 47.867
        }
        
        # Calculate total mass
        total_mass = sum(atomic_masses.get(site.element, 50.0) for site in self.sites)
        
        # Convert to g/cm³ (using Avogadro's number and unit conversion)
        volume_cm3 = self.volume * 1e-24  # Å³ to cm³
        density = (total_mass * 1.66054e-24) / volume_cm3  # amu to g conversion
        
        return density
    
    # Compatibility properties for visualization modules
    @property
    def atomic_sites(self) -> List[AtomicSite]:
        """Compatibility property - alias for sites."""
        return self.sites

    @property
    def lattice_parameters(self) -> LatticeParameters:
        """Compatibility property - alias for lattice."""
        return self.lattice

    @property
    def composition(self) -> str:
        """Compatibility property - alias for formula."""
        return self.formula
    
    def get_elements(self) -> List[str]:
        """Get list of unique elements in the structure."""
        return list(set(site.element for site in self.sites))
    
    def get_sites_by_element(self, element: str) -> List[AtomicSite]:
        """Get all atomic sites for a specific element."""
        return [site for site in self.sites if site.element == element]
    
    def to_cartesian_coordinates(self) -> List[np.ndarray]:
        """
        Convert all atomic positions to cartesian coordinates.
        
        Returns:
            List[np.ndarray]: List of cartesian positions in Å
        """
        lattice_matrix = self.lattice.lattice_matrix()
        return [site.cartesian_position(lattice_matrix) for site in self.sites]
    
    def supercell(self, nx: int = 2, ny: int = 2, nz: int = 2) -> 'Crystal':
        """
        Generate a supercell by replicating the unit cell.
        
        Args:
            nx, ny, nz: Number of replications in each direction
            
        Returns:
            Crystal: New crystal structure representing the supercell
        """
        # Scale lattice parameters
        new_lattice = LatticeParameters(
            a=self.lattice.a * nx,
            b=self.lattice.b * ny, 
            c=self.lattice.c * nz,
            alpha=self.lattice.alpha,
            beta=self.lattice.beta,
            gamma=self.lattice.gamma
        )
        
        # Replicate atomic sites
        new_sites = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for site in self.sites:
                        new_position = np.array([
                            (site.position[0] + i) / nx,
                            (site.position[1] + j) / ny, 
                            (site.position[2] + k) / nz
                        ])
                        
                        new_site = AtomicSite(
                            element=site.element,
                            position=new_position,
                            occupancy=site.occupancy,
                            oxidation_state=site.oxidation_state,
                            magnetic_moment=site.magnetic_moment,
                            thermal_displacement=site.thermal_displacement,
                            label=site.label
                        )
                        new_sites.append(new_site)
        
        return Crystal(
            lattice=new_lattice,
            sites=new_sites, 
            space_group=self.space_group,
            formula=self.formula,
            metadata={**self.metadata, 'supercell': (nx, ny, nz)}
        )
    
    def __str__(self) -> str:
        """String representation of the crystal structure."""
        return (f"Crystal({self.formula})\n"
                f"  Space Group: {self.space_group or 'Unknown'}\n" 
                f"  Lattice: a={self.lattice.a:.3f} b={self.lattice.b:.3f} c={self.lattice.c:.3f}\n"
                f"           α={self.lattice.alpha:.1f}° β={self.lattice.beta:.1f}° γ={self.lattice.gamma:.1f}°\n"
                f"  Volume: {self.volume:.3f} Å³\n"
                f"  Atoms: {self.num_atoms}")
    
    def __repr__(self) -> str:
        return f"Crystal(formula='{self.formula}', num_atoms={self.num_atoms})"