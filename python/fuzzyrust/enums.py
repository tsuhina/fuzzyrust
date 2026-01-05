"""Enums for fuzzyrust API."""

from enum import Enum


class Algorithm(str, Enum):
    """Available similarity algorithms.

    This enum provides type-safe algorithm selection for search operations.
    String values are preserved for backward compatibility.

    Example:
        >>> from fuzzyrust import Algorithm, find_best_matches
        >>> matches = find_best_matches(
        ...     ["apple", "apply", "banana"],
        ...     "appel",
        ...     algorithm=Algorithm.JARO_WINKLER,
        ...     limit=2
        ... )
    """

    LEVENSHTEIN = "levenshtein"
    """Classic edit distance (insertions, deletions, substitutions)"""

    DAMERAU_LEVENSHTEIN = "damerau_levenshtein"
    """Edit distance including transpositions (e.g., 'ca' -> 'ac' is 1 edit)"""

    DAMERAU = "damerau"
    """Alias for DAMERAU_LEVENSHTEIN"""

    JARO = "jaro"
    """Jaro similarity, good for short strings"""

    JARO_WINKLER = "jaro_winkler"
    """Jaro-Winkler similarity with prefix weighting, excellent for names"""

    NGRAM = "ngram"
    """N-gram similarity (Sørensen-Dice coefficient), default n=3 (trigram)"""

    BIGRAM = "bigram"
    """Bigram similarity (n=2)"""

    TRIGRAM = "trigram"
    """Trigram similarity (n=3)"""

    JACCARD = "jaccard"
    """Jaccard similarity (n-gram based)"""

    LCS = "lcs"
    """Longest Common Subsequence similarity"""

    COSINE = "cosine"
    """Cosine similarity (character-level)"""

    HAMMING = "hamming"
    """Hamming distance (for equal-length strings)"""


class NormalizationMode(str, Enum):
    """String normalization modes.

    Used by normalization utilities to control how strings are preprocessed
    before comparison.

    Example:
        >>> from fuzzyrust import normalize_string, NormalizationMode
        >>> normalize_string("  Hello, World!  ", NormalizationMode.STRICT)
        'helloworld'
    """

    LOWERCASE = "lowercase"
    """Convert to lowercase only"""

    UNICODE_NFKD = "unicode_nfkd"
    """Apply Unicode NFKD normalization"""

    REMOVE_PUNCTUATION = "remove_punctuation"
    """Remove punctuation characters"""

    REMOVE_WHITESPACE = "remove_whitespace"
    """Remove all whitespace"""

    STRICT = "strict"
    """Apply all normalizations: lowercase + NFKD + remove punctuation + remove whitespace"""


__all__ = ["Algorithm", "NormalizationMode"]
