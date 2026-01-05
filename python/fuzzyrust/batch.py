"""Batch operations API for FuzzyRust.

This module provides a clean, consolidated API for list-based batch operations
on strings. All functions are thin wrappers around the optimized Rust
implementations, providing parallel processing via Rayon.

Example usage:
    >>> import fuzzyrust.batch as batch

    # Compute similarity of query against all strings
    >>> results = batch.similarity(["hello", "hallo", "world"], "helo")
    >>> [(r.text, r.score) for r in results]
    [('hello', 0.91), ('hallo', 0.73), ('world', 0.40)]

    # Find top N best matches
    >>> matches = batch.best_matches(["apple", "apply", "banana"], "appel", limit=2)
    >>> [(m.text, m.score) for m in matches]
    [('apple', 0.93), ('apply', 0.84)]

    # Find duplicate groups
    >>> result = batch.deduplicate(["hello", "Hello", "HELLO", "world"])
    >>> result.groups
    [['hello', 'Hello', 'HELLO']]

    # Pairwise similarity between aligned lists
    >>> scores = batch.pairwise(["hello", "world"], ["hallo", "word"])
    >>> scores
    [0.88, 0.75]

    # Full similarity matrix
    >>> matrix = batch.similarity_matrix(["hello", "world"], ["hallo", "word", "help"])
    >>> # matrix[0] = similarities of "hello" with each choice
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fuzzyrust._core import (
    DeduplicationResult,
    MatchResult,
)
from fuzzyrust._core import (
    batch_similarity as _batch_similarity,
)
from fuzzyrust._core import (
    batch_similarity_pairs as _batch_similarity_pairs,
)
from fuzzyrust._core import (
    cdist as _cdist,
)
from fuzzyrust._core import (
    find_best_matches as _find_best_matches,
)
from fuzzyrust._core import (
    find_duplicates as _find_duplicates,
)
from fuzzyrust._utils import normalize_algorithm

if TYPE_CHECKING:
    from fuzzyrust.enums import Algorithm

__all__ = [
    "similarity",
    "best_matches",
    "deduplicate",
    "pairwise",
    "similarity_matrix",
    "distance_matrix",  # Deprecated alias for similarity_matrix
]


def similarity(
    strings: list[str],
    query: str,
    algorithm: str | Algorithm = "jaro_winkler",
) -> list[MatchResult]:
    """Compute similarity of a query against all strings.

    Computes the similarity score between the query and each string in the
    input list using the specified algorithm. Results are returned in the
    same order as the input strings.

    Args:
        strings: List of strings to compare against the query.
        query: The query string to match.
        algorithm: Similarity algorithm to use (string or Algorithm enum). Options:
            - "levenshtein": Normalized Levenshtein similarity
            - "damerau_levenshtein": Normalized Damerau-Levenshtein similarity
            - "jaro": Jaro similarity
            - "jaro_winkler": Jaro-Winkler similarity (default)
            - "ngram": N-gram similarity (default n=3, trigram)
            - "bigram": Bigram (n=2) similarity
            - "trigram": Trigram (n=3) similarity
            - "jaccard": Jaccard similarity (n-gram based)
            - "lcs": Longest common subsequence similarity
            - "cosine": Character-level cosine similarity

    Returns:
        List of MatchResult objects in the same order as input strings.
        Each result has `text`, `score`, and `id` fields where `id` is
        the original index in the input list.

    Example:
        >>> results = similarity(["hello", "hallo", "world"], "helo")
        >>> for r in results:
        ...     print(f"{r.text}: {r.score:.2f}")
        hello: 0.91
        hallo: 0.73
        world: 0.40
    """
    algo = normalize_algorithm(algorithm)
    return _batch_similarity(strings, query, algo)


def best_matches(
    strings: list[str],
    query: str,
    algorithm: str | Algorithm = "jaro_winkler",
    limit: int = 5,
    min_similarity: float = 0.0,
) -> list[MatchResult]:
    """Find top N best matches for a query from a list of strings.

    Computes similarity scores for all strings against the query, filters
    by minimum similarity, sorts by score descending, and returns the top
    matches up to the specified limit.

    Args:
        strings: List of strings to search.
        query: The query string to match.
        algorithm: Similarity algorithm to use (string or Algorithm enum). Options:
            - "levenshtein": Normalized Levenshtein similarity
            - "damerau_levenshtein": Normalized Damerau-Levenshtein similarity
            - "jaro": Jaro similarity
            - "jaro_winkler": Jaro-Winkler similarity (default)
            - "ngram": N-gram similarity (default n=3, trigram)
            - "bigram": Bigram (n=2) similarity
            - "trigram": Trigram (n=3) similarity
            - "jaccard": Jaccard similarity (n-gram based)
            - "lcs": Longest common subsequence similarity
            - "cosine": Character-level cosine similarity
        limit: Maximum number of results to return (default: 5).
        min_similarity: Minimum similarity score to include in results
            (default: 0.0, meaning all results are included).

    Returns:
        List of MatchResult objects sorted by score descending.
        Each result has `text`, `score`, and `id` fields.

    Example:
        >>> matches = best_matches(["apple", "apply", "banana"], "appel", limit=2)
        >>> for m in matches:
        ...     print(f"{m.text}: {m.score:.2f}")
        apple: 0.93
        apply: 0.84
    """
    algo = normalize_algorithm(algorithm)
    return _find_best_matches(
        strings, query, algorithm=algo, limit=limit, min_similarity=min_similarity
    )


def deduplicate(
    strings: list[str],
    algorithm: str | Algorithm = "jaro_winkler",
    min_similarity: float = 0.8,
    normalize: str | None = None,
) -> DeduplicationResult:
    """Find duplicate groups in a list of strings.

    Identifies groups of similar strings using the specified similarity
    algorithm and min_similarity. Strings with similarity >= min_similarity are
    grouped together using Union-Find clustering.

    For large datasets (> 2000 items), automatically uses Sorted Neighborhood
    Method (SNM) for O(N log N) performance instead of O(N^2) brute force.

    Args:
        strings: List of strings to deduplicate.
        algorithm: Similarity algorithm to use. Options:
            - "levenshtein": Normalized Levenshtein similarity
            - "damerau_levenshtein": Normalized Damerau-Levenshtein similarity
            - "jaro": Jaro similarity
            - "jaro_winkler": Jaro-Winkler similarity (default)
            - "ngram": N-gram similarity (default n=3, trigram)
            - "bigram": Bigram (n=2) similarity
            - "trigram": Trigram (n=3) similarity
            - "jaccard": Jaccard similarity (n-gram based)
            - "lcs": Longest common subsequence similarity
            - "cosine": Character-level cosine similarity
        min_similarity: Minimum similarity score to consider strings as duplicates
            (default: 0.8).
        normalize: Optional normalization mode to apply before comparison:
            - None: No normalization (default)
            - "lowercase": Convert to lowercase
            - "unicode_nfkd": Apply Unicode NFKD normalization
            - "remove_punctuation": Remove ASCII punctuation
            - "remove_whitespace": Remove all whitespace
            - "strict": Apply all normalizations

    Returns:
        DeduplicationResult with:
            - groups: List of duplicate groups (each group is a list of strings)
            - unique: List of strings that have no duplicates
            - total_duplicates: Total count of duplicate strings found

    Example:
        >>> result = deduplicate(["hello", "Hello", "HELLO", "world"], normalize="lowercase")
        >>> result.groups
        [['hello', 'Hello', 'HELLO']]
        >>> result.unique
        ['world']
        >>> result.total_duplicates
        3
    """
    # Map None to "none" for the Rust function which expects a string
    norm_mode = normalize if normalize is not None else "none"
    algo = normalize_algorithm(algorithm)
    return _find_duplicates(
        strings,
        algorithm=algo,
        min_similarity=min_similarity,
        normalize=norm_mode,
    )


def pairwise(
    left: list[str],
    right: list[str],
    algorithm: str | Algorithm = "jaro_winkler",
) -> list[float]:
    """Compute pairwise similarity between two equal-length lists.

    Takes two lists of strings and computes the similarity for each
    corresponding pair (left[i], right[i]). Uses parallel processing
    via Rayon for optimal performance.

    Args:
        left: First list of strings.
        right: Second list of strings (must be same length as left).
        algorithm: Similarity algorithm to use (string or Algorithm enum). Options:
            - "levenshtein": Normalized Levenshtein similarity
            - "jaro": Jaro similarity
            - "jaro_winkler": Jaro-Winkler similarity (default)
            - "ngram": Trigram (n=3) similarity
            - "cosine": Character-level cosine similarity

    Returns:
        List of similarity scores (0.0 to 1.0), one for each pair.

    Raises:
        ValueError: If left and right have different lengths.

    Example:
        >>> scores = pairwise(["hello", "world"], ["hallo", "word"])
        >>> scores
        [0.88, 0.75]
    """
    algo = normalize_algorithm(algorithm)
    results = _batch_similarity_pairs(left, right, algorithm=algo)
    # Convert Optional[float] to float (None shouldn't occur with valid algorithm)
    return [score if score is not None else 0.0 for score in results]


def similarity_matrix(
    queries: list[str],
    choices: list[str],
    algorithm: str | Algorithm = "levenshtein",
) -> list[list[float]]:
    """Compute similarity matrix between all queries and all choices.

    Similar to scipy.spatial.distance.cdist, this function computes the
    similarity between every pair of strings from queries and choices,
    returning a 2D matrix.

    Uses parallel processing via Rayon for optimal performance on large
    inputs.

    Args:
        queries: First list of strings (rows of output matrix).
        choices: Second list of strings (columns of output matrix).
        algorithm: Similarity algorithm to use (string or Algorithm enum). Options:
            - "levenshtein": Normalized Levenshtein similarity (default)
            - "jaro": Jaro similarity
            - "jaro_winkler": Jaro-Winkler similarity
            - "ngram": Trigram (n=3) similarity
            - "cosine": Character-level cosine similarity
            - "damerau": Normalized Damerau-Levenshtein similarity

    Returns:
        2D list where result[i][j] is the similarity between queries[i]
        and choices[j].

    Example:
        >>> matrix = similarity_matrix(["hello", "world"], ["hallo", "word", "help"])
        >>> # matrix[0] = [sim("hello", "hallo"), sim("hello", "word"), sim("hello", "help")]
        >>> # matrix[1] = [sim("world", "hallo"), sim("world", "word"), sim("world", "help")]
        >>> len(matrix)
        2
        >>> len(matrix[0])
        3
    """
    algo = normalize_algorithm(algorithm)
    return _cdist(queries, choices, scorer=algo)


def distance_matrix(
    queries: list[str],
    choices: list[str],
    algorithm: str | Algorithm = "levenshtein",
) -> list[list[float]]:
    """Deprecated: Use similarity_matrix() instead.

    This function has been renamed to similarity_matrix() to better reflect
    that it returns similarity scores (0.0-1.0), not distances.

    .. deprecated:: 0.3.0
        Use :func:`similarity_matrix` instead.
    """
    import warnings

    warnings.warn(
        "distance_matrix() is deprecated and will be removed in a future version. "
        "Use similarity_matrix() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return similarity_matrix(queries, choices, algorithm)
