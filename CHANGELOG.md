# Changelog

All notable changes to crysio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2025-09-14

### Fixed
- **POSCAR Parser Bug**: Fixed critical parsing error for Materials Project POSCAR format
  - `_parse_elements_and_counts()` method now correctly handles element name vs atom count lines
  - Added robust element symbol detection using `is_element_symbol()` helper function
  - Fixed line offset calculation that was causing "Unknown coordinate type: 4 4" error
  - Added `_clean_poscar_line()` method to handle charge states (Si4+, C4-) from Materials Project
- **Coordinate Processing**: Improved handling of POSCAR files with extra columns
  - Only first 3 coordinates (x, y, z) are now extracted from atomic position lines
  - Charge state information is automatically stripped during parsing
- **Error Messages**: Enhanced error reporting with more specific parsing error descriptions

### Changed
- Updated parser validation to be more permissive with Materials Project format variations
- Improved element symbol detection for better VASP 4/5+ format compatibility

### Documentation
- Added detailed docstring comments explaining the POSCAR parsing fix
- Updated parser class documentation with v0.2.1 bug fix information

## [0.2.0] - 2025-09-13

### Added
- **Graph Neural Network Integration**: Complete PyTorch Geometric conversion with configurable node and edge features
- **Structure Validation Framework**: Comprehensive validation for lattice parameters, atomic positions, and composition  
- **Materials Project API Integration**: Direct access to 150,000+ crystal structures and properties
- **Batch Processing Pipeline**: Efficient processing of multiple structures with progress tracking
- **Coordination Environment Analysis**: Automatic calculation of coordination numbers and bond statistics

#### New Modules
- `crysio.converters.graph_builder`: Crystal-to-graph conversion with periodic boundary handling
- `crysio.core.validators`: Multi-layer structure validation system  
- `crysio.api.materials_project`: Full API integration with rate limiting and error recovery

#### API Enhancements  
- `crysio.to_graph()`: Convert crystal structures to PyTorch Geometric graphs
- `crysio.validate_structure()`: Validate crystal structure integrity
- `crysio.load_from_materials_project()`: Load structures directly from Materials Project
- `crysio.search_materials_database()`: Search Materials Project by formula or criteria

#### Technical Improvements
- Robust error handling with specialized exception types
- Periodic boundary condition support for graph construction  
- Automatic atomic property database (electronegativity, atomic radius)
- Rate limiting and retry logic for API requests
- Enhanced batch processing with progress indicators

### Dependencies
- **torch**: Required for graph conversion functionality
- **torch-geometric**: PyTorch Geometric for GNN integration  
- **requests**: HTTP client for Materials Project API
- **tqdm**: Progress bars for batch processing

## [0.1.0] - 2025-09-12

### Added
- **Initial Release**: Core crystal structure representation and file parsing
- **File Format Support**: CIF and POSCAR/VASP format parsers with auto-detection
- **Crystal Structure Classes**: Complete `Crystal`, `LatticeParameters`, and `AtomicSite` classes
- **Basic Analysis**: Crystal property calculations (volume, density, composition)
- **Exception Handling**: Comprehensive exception hierarchy for robust error handling

#### Core Features  
- Basic crystal structure representation (`Crystal`, `LatticeParameters`, `AtomicSite`)
- File format parsers for CIF and POSCAR formats
- Auto-detection of file formats
- Basic exception handling framework  
- Simple crystal property calculations (volume, density, composition)

#### Modules
- `crysio.core.crystal`: Core crystal structure classes
- `crysio.core.parsers`: CIF and POSCAR file parsers  
- `crysio.utils.exceptions`: Basic exception handling

#### API
- `crysio.load_structure()`: Load crystal structures from files
- Basic crystal analysis methods (supercell generation, coordinate conversion)

---

## Version Compatibility

### Breaking Changes
- **None**: Version 0.2.1 is fully backward compatible with 0.2.0 and 0.1.0
- All existing code continues to work without modification
- New functionality is additive only

### Migration Guide
- **0.1.0 → 0.2.1**: No changes required, all APIs remain the same
- **0.2.0 → 0.2.1**: No changes required, this is a bug fix release

---

## Development Status

- **0.1.0**: Initial foundation - Core parsers and crystal representation
- **0.2.0**: Feature expansion - GNN integration, validation, and Materials Project API  
- **0.2.1**: Bug fix - Materials Project POSCAR parsing issue resolved
- **0.3.0**: (Planned) Visualization and advanced analysis tools
- **1.0.0**: (Planned) Stable API with comprehensive testing and documentation