"""
FuzzyRust - High-performance string similarity library

A Python library powered by Rust for fast fuzzy string matching,
designed for searching through messy data with typos and variations.

Example usage:
    >>> import fuzzyrust as fr

    # Simple similarity
    >>> fr.jaro_winkler_similarity("hello", "hallo")
    0.88

    # Batch operations (via batch submodule)
    >>> from fuzzyrust import batch
    >>> matches = batch.best_matches(["apple", "apply", "banana"], "appel")
    >>> [(m.text, m.score) for m in matches]
    [('apple', 0.93), ('apply', 0.84)]

    # Polars integration (via polars submodule)
    >>> from fuzzyrust import polars as frp
    >>> import polars as pl
    >>> left = pl.DataFrame({"name": ["Apple Inc"]})
    >>> right = pl.DataFrame({"company": ["Apple", "Google"]})
    >>> result = frp.df_join(left, right, left_on="name", right_on="company")

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
from fuzzyrust import batch, polars

# -----------------------------------------------------------------------------
# Submodule Access
# -----------------------------------------------------------------------------
# Polars functions: fuzzyrust.polars (df_join, series_similarity, etc.)
# Batch operations: fuzzyrust.batch (similarity, best_matches, deduplicate, etc.)
# See submodule docstrings for details.
# -----------------------------------------------------------------------------
from fuzzyrust._core import (
    AlgorithmComparison,
    AlgorithmError,
    ConfusionMatrixResult,
    DeduplicationResult,
    # --- Exceptions ---
    FuzzyRustError,
    MatchResult,
    # --- Schema-based multi-field matching ---
    Schema,
    SchemaBuilder,
    SchemaError,
    SchemaIndex,
    SchemaSearchResult,
    # --- Result types ---
    SearchResult,
    # --- Similarity classes ---
    TfIdfCosine,
    ValidationError,
    # --- Legacy batch functions (use fuzzyrust.batch module instead) ---
    batch_jaro_winkler,  # noqa: F401
    batch_levenshtein,  # noqa: F401
    batch_similarity,  # noqa: F401
    bigram_similarity,
    # --- Multi-algorithm comparison ---
    compare_algorithms,
    confusion_matrix,
    # --- Cosine similarity ---
    cosine_similarity_chars,
    cosine_similarity_ngrams,
    cosine_similarity_words,
    damerau_levenshtein,
    damerau_levenshtein_bounded,
    damerau_levenshtein_similarity,
    double_metaphone,
    double_metaphone_match,
    double_metaphone_similarity,
    extract,
    extract_ngrams,
    extract_one,
    f_score,
    find_duplicate_pairs,
    find_duplicates,
    hamming,
    hamming_distance_padded,
    hamming_similarity,
    jaro_similarity,
    jaro_similarity_grapheme,
    jaro_winkler_similarity,
    jaro_winkler_similarity_grapheme,
    # --- LCS functions ---
    lcs_length,
    lcs_similarity,
    lcs_similarity_max,
    lcs_string,
    # --- Distance functions ---
    levenshtein,
    levenshtein_bounded,
    levenshtein_grapheme,
    levenshtein_simd,
    levenshtein_simd_bounded,
    levenshtein_similarity,
    levenshtein_similarity_grapheme,
    levenshtein_similarity_simd,
    longest_common_substring,
    longest_common_substring_length,
    metaphone,
    metaphone_match,
    metaphone_similarity,
    ngram_jaccard,
    ngram_profile_similarity,
    # --- N-gram functions ---
    ngram_similarity,
    normalize_pair,
    # --- Normalization ---
    normalize_string,
    # --- Optimal String Alignment (restricted Damerau-Levenshtein) ---
    optimal_string_alignment,
    optimal_string_alignment_similarity,
    partial_ratio,
    # --- Evaluation metrics ---
    precision,
    qratio,
    qwratio,
    # --- RapidFuzz-compatible functions ---
    ratio,
    recall,
    # --- Phonetic functions ---
    soundex,
    soundex_match,
    soundex_similarity,
    token_set_ratio,
    token_sort_ratio,
    trigram_similarity,
    wratio,
)
from fuzzyrust._core import (
    IndexError as FuzzyIndexError,  # Avoid shadowing built-in IndexError
)
from fuzzyrust._core import (
    # --- Index classes ---
    PyBkTree as BkTree,
)
from fuzzyrust._core import (
    PyHybridIndex as HybridIndex,
)
from fuzzyrust._core import (
    PyNgramIndex as NgramIndex,
)
from fuzzyrust._core import (
    # --- Sharded index classes ---
    PyShardedBkTree as ShardedBkTree,
)
from fuzzyrust._core import (
    PyShardedNgramIndex as ShardedNgramIndex,
)
from fuzzyrust._core import (
    # --- Thread-safe index classes ---
    PyThreadSafeBkTree as ThreadSafeBkTree,
)
from fuzzyrust._core import (
    PyThreadSafeNgramIndex as ThreadSafeNgramIndex,
)
from fuzzyrust.enums import Algorithm, NormalizationMode
from fuzzyrust.index import FuzzyIndex

__version__ = _get_version("fuzzyrust")
__all__ = [
    # Version
    "__version__",
    # Submodules (preferred API)
    "batch",  # Batch operations: batch.similarity(), batch.best_matches(), etc.
    "polars",  # Polars integration: polars.df_join(), polars.series_similarity(), etc.
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
    "optimal_string_alignment",
    "optimal_string_alignment_similarity",
    "jaro_similarity",
    "jaro_winkler_similarity",
    "hamming",
    "hamming_distance_padded",
    "hamming_similarity",
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
    "lcs_string",
    "lcs_similarity",
    "lcs_similarity_max",
    "longest_common_substring_length",
    "longest_common_substring",
    "cosine_similarity_chars",
    "cosine_similarity_words",
    "cosine_similarity_ngrams",
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
    # Deduplication
    "find_duplicate_pairs",
    "find_duplicates",
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
    # Index classes (high-level)
    "FuzzyIndex",
]


# Convenience aliases
edit_distance = levenshtein
similarity = jaro_winkler_similarity
