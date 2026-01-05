"""Internal utilities for fuzzyrust."""

from typing import Union

from fuzzyrust.enums import Algorithm

# Valid algorithm names (lowercase)
VALID_ALGORITHMS = frozenset(
    {
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
        "hamming",
    }
)


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
        # Normalize aliases
        if algo_lower == "damerau":
            algo_lower = "damerau_levenshtein"
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

    raise TypeError(f"algorithm must be str or Algorithm enum, got {type(algorithm).__name__}")


class UnionFind:
    """Union-Find data structure for efficient clustering.

    This data structure supports efficient union and find operations
    with path compression and union by rank optimizations.

    Args:
        n: Number of elements in the disjoint set.

    Example:
        >>> uf = UnionFind(5)
        >>> uf.union(0, 1)
        >>> uf.union(2, 3)
        >>> uf.find(0) == uf.find(1)
        True
        >>> uf.find(0) == uf.find(2)
        False
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find the root of element x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Union the sets containing x and y using union by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


__all__ = ["normalize_algorithm", "VALID_ALGORITHMS", "UnionFind"]
