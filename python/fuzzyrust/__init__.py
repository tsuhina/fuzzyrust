"""
FuzzyRust - High-performance string similarity library

A Python library powered by Rust for fast fuzzy string matching,
designed for searching through messy data with typos and variations.

Example usage:
    >>> import fuzzyrust as fr

    # Simple similarity
    >>> fr.jaro_winkler_similarity("hello", "hallo")
    0.88

    # Find best matches (returns MatchResult objects)
    >>> matches = fr.find_best_matches(["apple", "apply", "banana"], "appel")
    >>> [(m.text, m.score) for m in matches]
    [('apple', 0.93), ('apply', 0.84)]

    # Use BK-tree for efficient fuzzy search (returns SearchResult objects)
    >>> tree = fr.BkTree()
    >>> tree.add_all(["hello", "hallo", "hullo", "world"])
    >>> results = tree.search("helo", max_distance=2)
    >>> [(r.text, r.score) for r in results]
    [('hello', 0.8), ('hallo', 0.6), ('hullo', 0.6)]
"""

from importlib.metadata import version as _get_version

# Register the .fuzzy expression namespace
import fuzzyrust.expr  # noqa: F401

# Import polars subpackage for `from fuzzyrust import polars` style
from fuzzyrust import polars
from fuzzyrust._core import (
    AlgorithmComparison,
    AlgorithmError,
    # Evaluation metrics
    ConfusionMatrixResult,
    DeduplicationResult,
    # Custom exceptions
    FuzzyRustError,
    MatchResult,
    # Schema-based multi-field matching
    Schema,
    SchemaBuilder,
    SchemaError,
    SchemaIndex,
    SchemaSearchResult,
    # Result types
    SearchResult,
    # Similarity classes
    TfIdfCosine,
    ValidationError,
    batch_jaro_winkler,
    # Batch processing
    batch_levenshtein,
    batch_similarity,
    batch_similarity_pairs,
    bigram_similarity,
    bigram_similarity_ci,
    cdist,
    # Multi-algorithm comparison
    compare_algorithms,
    confusion_matrix,
    cosine_similarity_chars,
    cosine_similarity_chars_ci,
    cosine_similarity_ngrams,
    cosine_similarity_ngrams_ci,
    cosine_similarity_words,
    cosine_similarity_words_ci,
    damerau_levenshtein,
    damerau_levenshtein_bounded,
    damerau_levenshtein_ci,
    damerau_levenshtein_similarity,
    damerau_levenshtein_similarity_ci,
    double_metaphone,
    double_metaphone_match,
    double_metaphone_similarity,
    extract,
    extract_ngrams,
    extract_one,
    f_score,
    find_best_matches,
    find_duplicate_pairs,
    # Deduplication
    find_duplicates,
    hamming,
    hamming_ci,
    hamming_distance_padded,
    hamming_similarity,
    hamming_similarity_ci,
    jaro_similarity,
    jaro_similarity_ci,
    jaro_similarity_grapheme,
    jaro_winkler_similarity,
    jaro_winkler_similarity_ci,
    jaro_winkler_similarity_grapheme,
    lcs_length,
    lcs_length_ci,
    lcs_similarity,
    lcs_similarity_ci,
    lcs_similarity_max,
    lcs_string,
    lcs_string_ci,
    # Distance functions
    levenshtein,
    levenshtein_bounded,
    # Case-insensitive variants
    levenshtein_ci,
    # Grapheme cluster mode functions
    levenshtein_grapheme,
    # SIMD-accelerated functions
    levenshtein_simd,
    levenshtein_simd_bounded,
    levenshtein_similarity,
    levenshtein_similarity_ci,
    levenshtein_similarity_grapheme,
    levenshtein_similarity_simd,
    longest_common_substring,
    longest_common_substring_ci,
    longest_common_substring_length,
    metaphone,
    metaphone_match,
    metaphone_similarity,
    metaphone_similarity_ci,
    ngram_jaccard,
    ngram_jaccard_ci,
    ngram_profile_similarity,
    ngram_similarity,
    ngram_similarity_ci,
    normalize_pair,
    # Normalization
    normalize_string,
    # RapidFuzz-compatible convenience functions
    partial_ratio,
    precision,
    qratio,
    qwratio,
    ratio,
    recall,
    soundex,
    soundex_match,
    soundex_similarity,
    soundex_similarity_ci,
    token_set_ratio,
    token_sort_ratio,
    trigram_similarity,
    trigram_similarity_ci,
    wratio,
)
from fuzzyrust._core import (
    IndexError as FuzzyIndexError,  # Avoid shadowing built-in IndexError
)
from fuzzyrust._core import (
    # Index classes
    PyBkTree as BkTree,
)
from fuzzyrust._core import (
    PyHybridIndex as HybridIndex,
)
from fuzzyrust._core import (
    PyNgramIndex as NgramIndex,
)
from fuzzyrust._core import (
    # Sharded index classes
    PyShardedBkTree as ShardedBkTree,
)
from fuzzyrust._core import (
    PyShardedNgramIndex as ShardedNgramIndex,
)
from fuzzyrust._core import (
    # Thread-safe index classes
    PyThreadSafeBkTree as ThreadSafeBkTree,
)
from fuzzyrust._core import (
    PyThreadSafeNgramIndex as ThreadSafeNgramIndex,
)
from fuzzyrust.enums import Algorithm, NormalizationMode
from fuzzyrust.index import FuzzyIndex

# -----------------------------------------------------------------------------
# Polars Integration - Batch API (polars_api)
# -----------------------------------------------------------------------------
# High-performance batch operations for large datasets.
# Uses vectorized processing and Sorted Neighborhood Method (SNM).
# Best for large datasets (100K+ rows) where performance is critical.
# See: fuzzyrust.polars_api module docstring for details.
from fuzzyrust.polars_api import (
    batch_best_match,
    dedupe_snm,
    find_similar_pairs,
    match_records_batch,
)

# -----------------------------------------------------------------------------
# Polars Integration - High-Level API (polars_ext)
# -----------------------------------------------------------------------------
# User-friendly functions for common fuzzy matching operations.
# Best for small to medium datasets (< 100K rows) and simple use cases.
# See: fuzzyrust.polars_ext module docstring for details.
from fuzzyrust.polars_ext import (
    dedupe_series,
    fuzzy_dedupe_rows,
    fuzzy_join,
    match_dataframe,
    match_series,
)

__version__ = _get_version("fuzzyrust")
__all__ = [
    # Version
    "__version__",
    # Custom exceptions
    "FuzzyRustError",
    "ValidationError",
    "AlgorithmError",
    "FuzzyIndexError",
    "SchemaError",
    # Result types
    "SearchResult",
    "MatchResult",
    "DeduplicationResult",
    "AlgorithmComparison",
    "SchemaSearchResult",
    # Enums
    "Algorithm",
    "NormalizationMode",
    # Distance/similarity functions
    "levenshtein",
    "levenshtein_bounded",
    "levenshtein_similarity",
    "damerau_levenshtein",
    "damerau_levenshtein_bounded",
    "damerau_levenshtein_similarity",
    "jaro_similarity",
    "jaro_winkler_similarity",
    "hamming",
    "hamming_ci",
    "hamming_distance_padded",
    "hamming_similarity",
    "hamming_similarity_ci",
    "ngram_similarity",
    "ngram_jaccard",
    "bigram_similarity",
    "trigram_similarity",
    "ngram_profile_similarity",
    "extract_ngrams",
    "soundex",
    "soundex_match",
    "soundex_similarity",
    "metaphone",
    "metaphone_match",
    "metaphone_similarity",
    "double_metaphone",
    "double_metaphone_match",
    "double_metaphone_similarity",
    # Grapheme cluster mode
    "levenshtein_grapheme",
    "levenshtein_similarity_grapheme",
    "jaro_similarity_grapheme",
    "jaro_winkler_similarity_grapheme",
    # SIMD-accelerated functions
    "levenshtein_simd",
    "levenshtein_simd_bounded",
    "levenshtein_similarity_simd",
    "lcs_length",
    "lcs_length_ci",
    "lcs_string",
    "lcs_string_ci",
    "lcs_similarity",
    "lcs_similarity_max",
    "longest_common_substring_length",
    "longest_common_substring",
    "longest_common_substring_ci",
    "cosine_similarity_chars",
    "cosine_similarity_words",
    "cosine_similarity_ngrams",
    # Case-insensitive variants
    "levenshtein_ci",
    "levenshtein_similarity_ci",
    "damerau_levenshtein_ci",
    "damerau_levenshtein_similarity_ci",
    "jaro_similarity_ci",
    "jaro_winkler_similarity_ci",
    "ngram_similarity_ci",
    "ngram_jaccard_ci",
    "cosine_similarity_chars_ci",
    "cosine_similarity_words_ci",
    "cosine_similarity_ngrams_ci",
    "bigram_similarity_ci",
    "trigram_similarity_ci",
    "lcs_similarity_ci",
    "soundex_similarity_ci",
    "metaphone_similarity_ci",
    # Normalization
    "normalize_string",
    "normalize_pair",
    # RapidFuzz-compatible convenience functions
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "wratio",
    "ratio",
    "qratio",
    "qwratio",
    "extract",
    "extract_one",
    # Batch processing
    "batch_levenshtein",
    "batch_jaro_winkler",
    "batch_similarity_pairs",
    "cdist",
    "find_best_matches",
    # Deduplication
    "find_duplicates",
    "find_duplicate_pairs",
    # Evaluation metrics
    "ConfusionMatrixResult",
    "precision",
    "recall",
    "f_score",
    "confusion_matrix",
    # Multi-algorithm comparison
    "compare_algorithms",
    # Similarity classes
    "TfIdfCosine",
    # Index classes
    "BkTree",
    "NgramIndex",
    "HybridIndex",
    # Thread-safe index classes
    "ThreadSafeBkTree",
    "ThreadSafeNgramIndex",
    # Sharded index classes
    "ShardedBkTree",
    "ShardedNgramIndex",
    # Schema-based multi-field matching
    "Schema",
    "SchemaBuilder",
    "SchemaIndex",
    # Polars Integration - High-Level API (polars_ext)
    # User-friendly functions for small/medium datasets
    "match_series",
    "dedupe_series",
    "match_dataframe",
    "fuzzy_join",
    "fuzzy_dedupe_rows",
    # Polars Integration - Batch API (polars_api)
    # High-performance functions for large datasets (100K+ rows)
    "batch_similarity",
    "batch_best_match",
    "dedupe_snm",
    "match_records_batch",
    "find_similar_pairs",
    # Index classes (high-level)
    "FuzzyIndex",
    # Polars subpackage
    "polars",
]


# Convenience aliases
edit_distance = levenshtein
similarity = jaro_winkler_similarity
