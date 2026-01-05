"""Type stubs for fuzzyrust.batch module.

Batch operations API for FuzzyRust. All functions are thin wrappers around
the optimized Rust implementations, providing parallel processing via Rayon.
"""

from typing import Union

from fuzzyrust import DeduplicationResult, MatchResult
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
    algorithm: Union[str, Algorithm] = "jaro_winkler",
) -> list[MatchResult]:
    """Compute similarity of a query against all strings.

    Args:
        strings: List of strings to compare against the query.
        query: The query string to match.
        algorithm: Similarity algorithm to use (string or Algorithm enum).

    Returns:
        List of MatchResult objects in the same order as input strings.
    """
    ...

def best_matches(
    strings: list[str],
    query: str,
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    limit: int = 5,
    min_similarity: float = 0.0,
) -> list[MatchResult]:
    """Find top N best matches for a query from a list of strings.

    Args:
        strings: List of strings to search.
        query: The query string to match.
        algorithm: Similarity algorithm to use (string or Algorithm enum).
        limit: Maximum number of results to return (default: 5).
        min_similarity: Minimum similarity score to include in results.

    Returns:
        List of MatchResult objects sorted by score descending.
    """
    ...

def deduplicate(
    strings: list[str],
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.8,
    normalize: str | None = None,
) -> DeduplicationResult:
    """Find duplicate groups in a list of strings.

    Args:
        strings: List of strings to deduplicate.
        algorithm: Similarity algorithm to use.
        min_similarity: Minimum similarity score to consider strings as duplicates.
        normalize: Optional normalization mode ("lowercase", "unicode_nfkd",
            "remove_punctuation", "remove_whitespace", "strict", or None).

    Returns:
        DeduplicationResult with groups, unique, and total_duplicates fields.
    """
    ...

def pairwise(
    left: list[str],
    right: list[str],
    algorithm: Union[str, Algorithm] = "jaro_winkler",
) -> list[float]:
    """Compute pairwise similarity between two equal-length lists.

    Args:
        left: First list of strings.
        right: Second list of strings (must be same length as left).
        algorithm: Similarity algorithm to use.

    Returns:
        List of similarity scores (0.0 to 1.0), one for each pair.

    Raises:
        ValueError: If left and right have different lengths.
    """
    ...

def similarity_matrix(
    queries: list[str],
    choices: list[str],
    algorithm: Union[str, Algorithm] = "levenshtein",
) -> list[list[float]]:
    """Compute similarity matrix between all queries and all choices.

    Args:
        queries: First list of strings (rows of output matrix).
        choices: Second list of strings (columns of output matrix).
        algorithm: Similarity algorithm to use.

    Returns:
        2D list where result[i][j] is the similarity between queries[i]
        and choices[j].
    """
    ...

def distance_matrix(
    queries: list[str],
    choices: list[str],
    algorithm: Union[str, Algorithm] = "levenshtein",
) -> list[list[float]]:
    """Deprecated: Use similarity_matrix() instead.

    .. deprecated:: 0.3.0
        Use :func:`similarity_matrix` instead.
    """
    ...
