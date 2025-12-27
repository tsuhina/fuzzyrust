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

from fuzzyrust._core import (
    AlgorithmComparison,
    # Evaluation metrics
    ConfusionMatrixResult,
    DeduplicationResult,
    MatchResult,
    # Schema-based multi-field matching
    Schema,
    SchemaBuilder,
    SchemaIndex,
    SchemaSearchResult,
    # Result types
    SearchResult,
    # Similarity classes
    TfIdfCosine,
    batch_jaro_winkler,
    # Batch processing
    batch_levenshtein,
    bigram_similarity,
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
    damerau_levenshtein_ci,
    damerau_levenshtein_similarity,
    damerau_levenshtein_similarity_ci,
    extract,
    extract_ngrams,
    extract_one,
    f_score,
    find_best_matches,
    # Deduplication
    find_duplicates,
    hamming,
    hamming_distance_padded,
    hamming_similarity,
    jaro_similarity,
    jaro_similarity_ci,
    jaro_winkler_similarity,
    jaro_winkler_similarity_ci,
    lcs_length,
    lcs_similarity,
    lcs_similarity_max,
    lcs_string,
    # Distance functions
    levenshtein,
    # Case-insensitive variants
    levenshtein_ci,
    levenshtein_similarity,
    levenshtein_similarity_ci,
    longest_common_substring,
    longest_common_substring_length,
    metaphone,
    metaphone_match,
    metaphone_similarity,
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
    ratio,
    recall,
    soundex,
    soundex_match,
    soundex_similarity,
    token_set_ratio,
    token_sort_ratio,
    trigram_similarity,
    wratio,
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
from fuzzyrust.enums import Algorithm, NormalizationMode

# Polars integration (optional)
try:
    from fuzzyrust.polars_ext import (
        POLARS_AVAILABLE,
        dedupe_series,
        fuzzy_join,
        match_dataframe,
        match_series,
    )
except ImportError:
    POLARS_AVAILABLE = False
    match_series = None  # type: ignore
    dedupe_series = None  # type: ignore
    match_dataframe = None  # type: ignore
    fuzzy_join = None  # type: ignore

__version__ = "0.1.1"
__all__ = [
    # Version
    "__version__",
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
    "levenshtein_similarity",
    "damerau_levenshtein",
    "damerau_levenshtein_similarity",
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
    "lcs_length",
    "lcs_string",
    "lcs_similarity",
    "lcs_similarity_max",
    "longest_common_substring_length",
    "longest_common_substring",
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
    # Normalization
    "normalize_string",
    "normalize_pair",
    # RapidFuzz-compatible convenience functions
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "wratio",
    "ratio",
    "extract",
    "extract_one",
    # Batch processing
    "batch_levenshtein",
    "batch_jaro_winkler",
    "find_best_matches",
    # Deduplication
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
    # Schema-based multi-field matching
    "Schema",
    "SchemaBuilder",
    "SchemaIndex",
    # Polars integration (optional)
    "POLARS_AVAILABLE",
    "match_series",
    "dedupe_series",
    "match_dataframe",
    "fuzzy_join",
]


# Convenience aliases
edit_distance = levenshtein
similarity = jaro_winkler_similarity
