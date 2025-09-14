# Changelog

All notable changes to crysio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2025-09-14

### Fixed
- **Materials Project API Search Logic**: Improved query detection to properly distinguish between elements and chemical formulas
  - `search_materials_database()` now correctly handles "SiC" as formula, not element "S", "i", "C"
  - Better detection logic: single elements (≤2 chars, uppercase) vs formulas (compounds like "Al2O3")
  - Fixed API request failures caused by incorrect query categorization
- **Invalid Field Names**: Updated field specifications to match current Materials Project API
  - Replaced deprecated `'crystal_system'`, `'spacegroup'` fields with valid `'symmetry'` field
  - Fixed "invalid fields requested" errors in `search_by_criteria()` function
  - Updated batch download operations with correct field names
- **Graph Conversion Display**: Fixed batch information display bug in graph conversion output
  - Resolved `'NoneType' object has no attribute 'shape'` error when displaying single graphs
  - Added proper handling for single graph vs batch graph scenarios
- **Code Structure**: Removed duplicate functions and improved code organization
  - Eliminated duplicate `search_materials_database()` and `load_from_materials_project()` functions
  - Cleaner module structure and better error handling

### Improved
- Enhanced query detection algorithm for more accurate Materials Project searches
- Better error messages for API field validation issues
- More robust handling of PyTorch Geometric graph attributes
- Improved code documentation with v0.2.2 specific fixes

### Technical Details
- Query detection now uses `len(query) <= 2 and query.isalpha() and query.isupper()` for elements
- All other queries treated as chemical formulas for better accuracy
- Field names aligned with Materials Project API v2024 specification
- Graph display code now checks `batch is not None` before accessing shape attribute

## [0.2.1] - 2025-09-13

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

## [0.2.0] - 2025-09-12

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

## [0.1.0] - 2025-09-11

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
- **None**: Version 0.2.2 is fully backward compatible with 0.2.1, 0.2.0 and 0.1.0
- All existing code continues to work without modification
- New functionality is additive only

### Migration Guide
- **0.1.0 → 0.2.2**: No changes required, all APIs remain the same
- **0.2.0 → 0.2.2**: No changes required, these are bug fix releases
- **0.2.1 → 0.2.2**: No changes required, this is a bug fix release

---

## Development Status

- **0.1.0**: Initial foundation - Core parsers and crystal representation
- **0.2.0**: Feature expansion - GNN integration, validation, and Materials Project API  
- **0.2.1**: Bug fix - Materials Project POSCAR parsing issue resolved
- **0.2.2**: Bug fix - Materials Project API search and field validation issues resolved
- **0.3.0**: (Planned) Visualization and advanced analysis tools
- **1.0.0**: (Planned) Stable API with comprehensive testing and documentation