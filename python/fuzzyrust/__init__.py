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
    # Result types
    SearchResult,
    MatchResult,
    DeduplicationResult,
    AlgorithmComparison,
    SchemaSearchResult,

    # Distance functions
    levenshtein,
    levenshtein_similarity,
    damerau_levenshtein,
    damerau_levenshtein_similarity,
    jaro_similarity,
    jaro_winkler_similarity,
    hamming,
    ngram_similarity,
    ngram_jaccard,
    extract_ngrams,
    soundex,
    soundex_match,
    metaphone,
    metaphone_match,
    lcs_length,
    lcs_string,
    lcs_similarity,
    longest_common_substring_length,
    longest_common_substring,
    cosine_similarity_chars,
    cosine_similarity_words,
    cosine_similarity_ngrams,

    # Case-insensitive variants
    levenshtein_ci,
    levenshtein_similarity_ci,
    damerau_levenshtein_ci,
    damerau_levenshtein_similarity_ci,
    jaro_similarity_ci,
    jaro_winkler_similarity_ci,
    ngram_similarity_ci,

    # Normalization
    normalize_string,
    normalize_pair,

    # Batch processing
    batch_levenshtein,
    batch_jaro_winkler,
    find_best_matches,

    # Deduplication
    find_duplicates,

    # Multi-algorithm comparison
    compare_algorithms,

    # Index classes
    PyBkTree as BkTree,
    PyNgramIndex as NgramIndex,
    PyHybridIndex as HybridIndex,

    # Schema-based multi-field matching
    Schema,
    SchemaBuilder,
    SchemaIndex,
)

from fuzzyrust.enums import Algorithm, NormalizationMode

__version__ = "0.1.0"  # Initial PyPI release
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
    "ngram_similarity",
    "ngram_jaccard",
    "extract_ngrams",
    "soundex",
    "soundex_match",
    "metaphone",
    "metaphone_match",
    "lcs_length",
    "lcs_string",
    "lcs_similarity",
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

    # Normalization
    "normalize_string",
    "normalize_pair",

    # Batch processing
    "batch_levenshtein",
    "batch_jaro_winkler",
    "find_best_matches",

    # Deduplication
    "find_duplicates",

    # Multi-algorithm comparison
    "compare_algorithms",

    # Index classes
    "BkTree",
    "NgramIndex",
    "HybridIndex",

    # Schema-based multi-field matching
    "Schema",
    "SchemaBuilder",
    "SchemaIndex",
]


# Convenience aliases
edit_distance = levenshtein
similarity = jaro_winkler_similarity
