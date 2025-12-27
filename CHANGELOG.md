# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Evaluation metrics**: `precision`, `recall`, `f_score`, `confusion_matrix` for record linkage evaluation
- **RapidFuzz-compatible functions**: `partial_ratio`, `token_sort_ratio`, `token_set_ratio`, `wratio`, `ratio`, `extract`, `extract_one`
- **Polars integration**: `match_series`, `dedupe_series`, `match_dataframe`, `fuzzy_join` for DataFrame operations
- **Algorithm enum**: Type-safe algorithm selection via `Algorithm` enum
- **`#[must_use]` attributes**: Added to all pure functions to catch unused return values
- **Parallel threshold**: Batch operations now use sequential processing for inputs < 100 items to avoid thread pool overhead
- **Hashable result types**: `SearchResult` and `MatchResult` now support `__eq__` and `__hash__` for use in sets and as dict keys

### Changed

- **API Breaking**: Removed `min_similarity` parameter from `NgramIndex` and `HybridIndex` constructors
  - Use `min_similarity` in search methods instead
- **API Breaking**: Renamed test parameter `min_score` to `min_similarity` for consistency
- **API Breaking**: `hamming_similarity` now raises `ValueError` instead of returning `None` for unequal-length strings
  - Consistent with `hamming()` behavior

### Documentation

- Added time/space complexity documentation to all major algorithms
- Added complexity notes to index classes (BkTree, NgramIndex, HybridIndex)
- Added complexity documentation to find_duplicates function
- Added algorithm selection guide to `Algorithm` enum docstring

### Fixed

- Improved documentation for `normalize` vs `case_insensitive` parameters
- Fixed test consistency for SchemaIndex parameter naming

## [0.1.0] - 2024-01-XX

### Added

- Initial release
- Core similarity algorithms: Levenshtein, Damerau-Levenshtein, Jaro, Jaro-Winkler, Hamming, N-gram, Soundex, Metaphone, LCS, Cosine
- Indexing structures: BK-tree, N-gram Index, Hybrid Index
- Schema-based multi-field matching with SchemaIndex
- Deduplication with graph-based clustering
- Parallel batch processing with Rayon
- Python bindings via PyO3
- Comprehensive type stubs for IDE support

[Unreleased]: https://github.com/username/fuzzyrust/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/username/fuzzyrust/releases/tag/v0.1.0
