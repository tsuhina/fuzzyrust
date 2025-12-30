"""Internal utilities for fuzzyrust."""

from typing import Union

from fuzzyrust.enums import Algorithm

# Valid algorithm names (lowercase)
VALID_ALGORITHMS = frozenset({
    "levenshtein",
    "damerau_levenshtein",
    "jaro",
    "jaro_winkler",
    "ngram",
    "bigram",
    "trigram",
    "lcs",
    "cosine",
    "jaccard",  # Also valid for some functions
})


def normalize_algorithm(algorithm: Union[str, Algorithm]) -> str:
    """Convert Algorithm enum to string, or validate string algorithm name.

    Args:
        algorithm: Either an Algorithm enum value or a string algorithm name.

    Returns:
        Lowercase string algorithm name.

    Raises:
        ValueError: If the algorithm name is not recognized.
        TypeError: If algorithm is not a string or Algorithm enum.

    Example:
        >>> normalize_algorithm(Algorithm.JARO_WINKLER)
        'jaro_winkler'
        >>> normalize_algorithm("levenshtein")
        'levenshtein'
    """
    if isinstance(algorithm, Algorithm):
        return algorithm.value

    if isinstance(algorithm, str):
        algo_lower = algorithm.lower()
        # Check against known algorithms
        if algo_lower in VALID_ALGORITHMS:
            return algo_lower
        # Also check enum values for case-insensitive matching
        enum_values = {a.value for a in Algorithm}
        if algo_lower in enum_values:
            return algo_lower
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. "
            f"Valid options: {sorted(VALID_ALGORITHMS | enum_values)}"
        )

    raise TypeError(
        f"algorithm must be str or Algorithm enum, got {type(algorithm).__name__}"
    )


__all__ = ["normalize_algorithm", "VALID_ALGORITHMS"]
