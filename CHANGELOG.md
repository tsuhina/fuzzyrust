# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-01-04

### Breaking Changes

- **Removed deprecated top-level functions** - Use new `batch` and `polars` submodules instead
  - Removed: `fuzzy_join()` → Use `polars.df_join()`
  - Removed: `fuzzy_dedupe_rows()` → Use `polars.df_dedupe()`
  - Removed: `match_dataframe()` → Use `polars.df_match_pairs()`
  - Removed: `match_series()` → Use `polars.series_match()`
  - Removed: `dedupe_series()` → Use `polars.series_dedupe()`
  - Removed: `batch_best_match()` → Use `polars.series_best_match()`
  - Removed: `dedupe_snm()` → Use `polars.df_dedupe_snm()`
  - Removed: `find_similar_pairs()` → Use `polars.df_find_pairs()`
  - Removed: `match_records_batch()` → Use `polars.df_match_records()`
  - Removed: `find_best_matches()` → Use `batch.best_matches()`
  - Removed: `find_duplicates()` → Use `batch.deduplicate()`
  - Removed: `batch_similarity_pairs()` → Use `batch.pairwise()`
  - Removed: `cdist()` → Use `batch.similarity_matrix()`

- **Removed all 21 `_ci` suffix functions** - Use the `normalize` parameter instead
  - Removed: `levenshtein_ci`, `levenshtein_similarity_ci`, `jaro_ci`, `jaro_winkler_ci`, `damerau_levenshtein_ci`, `ngram_similarity_ci`, `ngram_jaccard_ci`, `cosine_similarity_ci`, `lcs_similarity_ci`, `lcs_similarity_max_ci`, `hamming_similarity_ci`, `soundex_similarity_ci`, `metaphone_similarity_ci`, `longest_common_substring_ci`, `longest_common_subsequence_ci`, and more
  - Migration: Use base function with `normalize="lowercase"` parameter (e.g., `levenshtein("Hello", "HELLO", normalize="lowercase")`)

- **Renamed `batch.distance_matrix()` to `batch.similarity_matrix()`** - Better reflects that it returns similarity scores (0.0-1.0), not distances
  - The old name `distance_matrix()` still works but is deprecated and will be removed in a future version

- **Changed `series_best_match()` default** - `normalize` now defaults to `None` instead of `"lowercase"`
  - This aligns with other Polars functions which default to case-sensitive comparison
  - For case-insensitive matching, explicitly pass `normalize="lowercase"`

- **Deprecated `batch_levenshtein()` and `batch_jaro_winkler()`** - Use `batch_similarity()` instead
  - These functions now emit DeprecationWarning and will be removed in a future version
  - Migration: `batch_similarity(strings, query, algorithm="levenshtein")` or `algorithm="jaro_winkler"`

- **Parameter renamed**: `threshold` → `min_similarity` for consistency
  - Affected functions: `batch.deduplicate()`, `polars.series_dedupe()`, `polars.df_dedupe()`, `polars.df_dedupe_snm()`, `polars.df_find_pairs()`

- **`ngram` algorithm now defaults to trigram (n=3)** instead of bigram (n=2)
  - Users who want bigram behavior can set `ngram_size=2` or use `algorithm="bigram"`
  - This aligns with common industry practice and improves match quality

### Added

- **New `batch` module**: Consolidated batch operations under `fuzzyrust.batch`
  - `batch.similarity()` - Compute similarity for multiple strings against a query
  - `batch.best_matches()` - Find best matches from a list
  - `batch.deduplicate()` - Group similar strings together
  - `batch.pairwise()` - Element-wise similarity between two lists
  - `batch.similarity_matrix()` - Full pairwise similarity matrix

- **New `polars` module`: Reorganized Polars API under `fuzzyrust.polars`
  - DataFrame functions: `df_join`, `df_dedupe`, `df_match_pairs`, `df_dedupe_snm`, `df_find_pairs`, `df_match_records`
  - Series functions: `series_similarity`, `series_best_match`, `series_match`, `series_dedupe`

- **Algorithm enum support in batch module**: All batch functions now accept `Algorithm` enum in addition to string names

- **Jaccard similarity algorithm**: Added `jaccard` algorithm support across all APIs
  - Available in batch operations, Polars expressions, and Series/DataFrame functions
  - Uses n-gram based Jaccard coefficient for string similarity

- **Polars threshold validation**: `is_similar()` expressions now validate threshold is in [0.0, 1.0] range

- **New `optimal_string_alignment()` functions**: Added `optimal_string_alignment()` and `optimal_string_alignment_similarity()` functions for the restricted Damerau-Levenshtein (OSA) algorithm. This algorithm is slightly faster than the true Damerau-Levenshtein but doesn't allow multiple edits on the same substring.

### Fixed

- **Fixed NaN comparison panic in ShardedNgramIndex**: Sorting results with NaN similarity values no longer panics. NaN values are now treated as equal during comparison.
- **Fixed SIMD similarity values outside [0,1] range**: The `levenshtein_similarity_simd()` function now properly clamps results to the valid [0.0, 1.0] range when byte-based SIMD operations produce values that would otherwise exceed bounds for Unicode strings.
- **Fixed `best_match()` missing exact matches with large target lists**: The native Polars plugin's indexed search could miss exact string matches when targets shared common prefixes (e.g., searching for "item_1" in a list containing "item_1", "item_10", "item_100" might return "item_10" instead of the exact match). Now checks for exact string matches first.
- **Added native plugin support to `best_match()` expression method**: The `.fuzzy.best_match()` expression method now uses the native Polars plugin when available, providing 10-50x performance improvement for column-to-column operations.
- **Added native plugin support to `distance()` expression method**: The `.fuzzy.distance()` expression method now uses the native plugin for Levenshtein distance column-to-column operations.
- **Added `ngram_size` and `case_insensitive` parameters to `best_match()` and `best_match_score()`**: These expression methods now support all the same parameters as other expression methods like `similarity()` and `is_similar()`.
- **Added `HAMMING` to `Algorithm` enum**: Users can now use `Algorithm.HAMMING` in addition to the string `"hamming"`.
- **Added type stubs for `batch` and `polars` submodules**: Improved IDE autocompletion and type checking for batch and polars APIs.
- **Fixed Damerau-Levenshtein to use true algorithm**: The `damerau_levenshtein()` function now uses the "true" Damerau-Levenshtein algorithm which allows multiple edits on the same substring. This matches the behavior of jellyfish and other reference implementations. The previous "restricted" (Optimal String Alignment) variant is now available as `optimal_string_alignment()` for users who need backward compatibility or the slightly faster restricted algorithm.
- **Fixed potential panic in Jaro grapheme mode**: Added bounds check in transposition counting loop
- **Fixed Soundex/Metaphone empty string handling**: Two identical empty strings now return 1.0 similarity instead of 0.0
- **Fixed Levenshtein SIMD similarity calculation**: Now uses character count instead of byte length for proper Unicode handling
- **Added missing jaccard algorithm**: Now exposed in Polars expression namespace and series operations
- Fixed struct field naming in native Polars plugin `best_match_score()` function
- Removed unused `combine_scores` parameter from `df_join()` (was never implemented)
- Consolidated `UnionFind` and algorithm validation utilities to avoid code duplication

### Changed

- **API reorganization**: Functions moved to dedicated submodules for better organization
  - Batch operations: `fuzzyrust.batch.*`
  - Polars operations: `fuzzyrust.polars.*` (import as `from fuzzyrust import polars as frp`)
  - Core functions remain at top level: `fuzzyrust.jaro_winkler_similarity()`, etc.

## [0.2.1] - 2024-12-15

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

## [0.1.0] - 2024-10-01

### Added

- Initial release
- Core similarity algorithms: Levenshtein, Damerau-Levenshtein, Jaro, Jaro-Winkler, Hamming, N-gram, Soundex, Metaphone, LCS, Cosine
- Indexing structures: BK-tree, N-gram Index, Hybrid Index
- Schema-based multi-field matching with SchemaIndex
- Deduplication with graph-based clustering
- Parallel batch processing with Rayon
- Python bindings via PyO3
- Comprehensive type stubs for IDE support

[Unreleased]: https://github.com/tsuhina/fuzzyrust/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/tsuhina/fuzzyrust/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/tsuhina/fuzzyrust/compare/v0.1.0...v0.2.1
[0.1.0]: https://github.com/tsuhina/fuzzyrust/releases/tag/v0.1.0
