"""FuzzyIndex for efficient batch fuzzy matching operations.

This module provides a high-level interface for building reusable fuzzy
indices from Polars Series or Python lists, enabling efficient repeated
searches without rebuilding the index.

Warning:
    This class is NOT thread-safe. Create separate instances per thread
    for concurrent operations.
"""

import pickle
from pathlib import Path
from typing import List, Literal, Optional, Union

import polars as pl

import fuzzyrust as fr


class FuzzyIndex:
    """
    A reusable fuzzy matching index for efficient batch operations.

    FuzzyIndex wraps fuzzyrust's underlying index structures (NgramIndex,
    BKTree, or HybridIndex) and provides a convenient API for building
    indices from Polars Series and performing batch searches.

    The index can be persisted to disk and reloaded for later use.

    Warning:
        This class is NOT thread-safe. Create separate instances for each
        thread when using in concurrent applications.

    Example:
        >>> import polars as pl
        >>> from fuzzyrust import FuzzyIndex
        >>>
        >>> # Build index from a Series
        >>> names = pl.Series(["Apple Inc", "Microsoft Corp", "Google LLC"])
        >>> index = FuzzyIndex.from_series(names, algorithm="ngram")
        >>>
        >>> # Search for similar strings
        >>> queries = pl.Series(["Apple", "Microsft"])
        >>> results = index.search_series(queries, min_similarity=0.6)
        >>>
        >>> # Save for later reuse
        >>> index.save("names_index.pkl")
        >>> index = FuzzyIndex.load("names_index.pkl")
    """

    def __init__(
        self,
        items: List[str],
        algorithm: Literal["ngram", "bktree", "hybrid"] = "ngram",
        ngram_size: int = 3,
    ):
        """
        Create a FuzzyIndex from a list of strings.

        Args:
            items: List of strings to index
            algorithm: Index algorithm to use:
                - "ngram": N-gram based inverted index (fast for fuzzy matching)
                - "bktree": BK-tree for metric space (good for edit distance)
                - "hybrid": Combines ngram and bktree for best of both
            ngram_size: Size of n-grams (only for "ngram" and "hybrid")
        """
        self._items = items
        self._algorithm = algorithm
        self._ngram_size = ngram_size
        self._index = self._build_index()

    def _build_index(self):
        """Build the underlying fuzzyrust index."""
        if self._algorithm == "ngram":
            index = fr.NgramIndex(ngram_size=self._ngram_size)
        elif self._algorithm == "bktree":
            index = fr.BkTree()
        elif self._algorithm == "hybrid":
            index = fr.HybridIndex(ngram_size=self._ngram_size)
        else:
            raise ValueError(f"Unknown algorithm: {self._algorithm}")

        for item in self._items:
            index.add(item)

        return index

    @classmethod
    def from_series(
        cls,
        series: "pl.Series",
        algorithm: Literal["ngram", "bktree", "hybrid"] = "ngram",
        ngram_size: int = 3,
    ) -> "FuzzyIndex":
        """
        Create a FuzzyIndex from a Polars Series.

        Args:
            series: Polars Series of strings to index
            algorithm: Index algorithm ("ngram", "bktree", or "hybrid")
            ngram_size: Size of n-grams (only for "ngram" and "hybrid")

        Returns:
            FuzzyIndex instance

        Example:
            >>> names = pl.Series(["Apple", "Microsoft", "Google"])
            >>> index = FuzzyIndex.from_series(names, algorithm="ngram")
        """
        items = [str(x) if x is not None else "" for x in series.to_list()]
        return cls(items, algorithm=algorithm, ngram_size=ngram_size)

    @classmethod
    def from_dataframe(
        cls,
        df: "pl.DataFrame",
        column: str,
        algorithm: Literal["ngram", "bktree", "hybrid"] = "ngram",
        ngram_size: int = 3,
    ) -> "FuzzyIndex":
        """
        Create a FuzzyIndex from a DataFrame column.

        Args:
            df: Polars DataFrame
            column: Column name to index
            algorithm: Index algorithm ("ngram", "bktree", or "hybrid")
            ngram_size: Size of n-grams (only for "ngram" and "hybrid")

        Returns:
            FuzzyIndex instance
        """
        return cls.from_series(df[column], algorithm=algorithm, ngram_size=ngram_size)

    def search(
        self,
        query: str,
        min_similarity: float = 0.0,
        limit: int = 10,
    ) -> List[fr.SearchResult]:
        """
        Search the index for strings similar to the query.

        Args:
            query: Query string to search for
            min_similarity: Minimum similarity score (0.0 to 1.0)
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects with text, score, and index
        """
        if self._algorithm == "bktree":
            # BkTree uses max_distance (edit distance) instead of min_similarity.
            # The relationship is: similarity = 1 - (distance / max(len(query), len(target)))
            # Rearranging: distance = (1 - similarity) * max_len
            #
            # The challenge: we don't know target lengths until we search.
            # Solution: Use a conservative max_distance that won't miss matches,
            # then filter by the actual similarity score (which BkTree calculates correctly).
            #
            # For worst case (short query, long target), we need generous max_distance.
            # We use the maximum string length in the index as a better estimate.
            query_len = len(query)

            # Estimate max target length: use index size heuristic or a generous multiplier
            # If we have items, sample to estimate max length (or just be generous)
            if self._items:
                # Use a sample to estimate max length (avoid scanning all items)
                sample_size = min(100, len(self._items))
                sample = self._items[:sample_size]
                estimated_max_len = max(len(s) for s in sample) if sample else query_len
                # Be generous: assume some items might be longer than our sample
                estimated_max_len = max(estimated_max_len, query_len * 3)
            else:
                estimated_max_len = query_len * 3

            # Calculate max_distance from similarity threshold
            # distance = (1 - min_similarity) * max_len
            max_dist = max(1, int((1 - min_similarity) * estimated_max_len))

            # Cap max_distance to avoid expensive full-tree scans
            # But ensure we get at least some results for very high similarity thresholds
            max_dist = max(max_dist, 2)  # At least allow 2 edits
            max_dist = min(max_dist, 50)  # Cap at 50 to avoid performance issues

            results = self._index.search(query, max_distance=max_dist)

            # Filter results by actual similarity score (calculated correctly by BkTree)
            # and apply limit
            filtered = [r for r in results if r.score >= min_similarity]
            # Sort by score descending (BkTree sorts by distance, we want highest similarity first)
            filtered.sort(key=lambda r: r.score, reverse=True)
            if limit:
                filtered = filtered[:limit]
            return filtered
        else:
            return self._index.search(query, min_similarity=min_similarity, limit=limit)

    def search_series(
        self,
        queries: "pl.Series",
        min_similarity: float = 0.0,
        limit: int = 1,
        include_query: bool = True,
    ) -> "pl.DataFrame":
        """
        Search for each query in a Series, returning a DataFrame of results.

        Args:
            queries: Series of query strings
            min_similarity: Minimum similarity score (0.0 to 1.0)
            limit: Maximum matches per query (default: 1 for best match only)
            include_query: Include query column in results

        Returns:
            DataFrame with columns:
            - query_idx: Index of the query in the input Series
            - query: The query string (if include_query=True)
            - match: The matched string from the index
            - match_idx: Index of the match in the original indexed data
            - score: Similarity score

        Example:
            >>> index = FuzzyIndex.from_series(targets)
            >>> results = index.search_series(queries, min_similarity=0.7)
            >>> # Join back with original data
            >>> matched = queries.to_frame("query").join(
            ...     results, on="query", how="left"
            ... )
        """
        query_list = queries.to_list()
        rows = []

        for query_idx, query in enumerate(query_list):
            if query is None:
                continue

            matches = self.search(str(query), min_similarity=min_similarity, limit=limit)

            for match in matches:
                row = {
                    "query_idx": query_idx,
                    "match": match.text,
                    "match_idx": match.id,
                    "score": match.score,
                }
                if include_query:
                    row["query"] = str(query)
                rows.append(row)

        if not rows:
            schema = {"query_idx": pl.Int64, "match": pl.Utf8, "match_idx": pl.Int64, "score": pl.Float64}
            if include_query:
                schema["query"] = pl.Utf8
            return pl.DataFrame(schema=schema)

        # Reorder columns for nicer output
        df = pl.DataFrame(rows)
        if include_query:
            return df.select(["query_idx", "query", "match", "match_idx", "score"])
        return df.select(["query_idx", "match", "match_idx", "score"])

    def batch_search(
        self,
        queries: List[str],
        min_similarity: float = 0.0,
        limit: int = 1,
    ) -> List[List[fr.SearchResult]]:
        """
        Search for multiple queries, returning results for each.

        Args:
            queries: List of query strings
            min_similarity: Minimum similarity score (0.0 to 1.0)
            limit: Maximum matches per query

        Returns:
            List of lists, where each inner list contains SearchResult
            objects for the corresponding query
        """
        return [self.search(q, min_similarity=min_similarity, limit=limit) for q in queries]

    def get_items(self) -> List[str]:
        """Return the list of indexed items."""
        return self._items.copy()

    def __len__(self) -> int:
        """Return the number of indexed items."""
        return len(self._items)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the index to a file.

        Args:
            path: File path to save to (typically .pkl extension)

        Example:
            >>> index.save("my_index.pkl")
        """
        data = {
            "items": self._items,
            "algorithm": self._algorithm,
            "ngram_size": self._ngram_size,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FuzzyIndex":
        """
        Load an index from a file.

        Args:
            path: File path to load from

        Returns:
            FuzzyIndex instance

        Example:
            >>> index = FuzzyIndex.load("my_index.pkl")
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        return cls(
            items=data["items"],
            algorithm=data["algorithm"],
            ngram_size=data["ngram_size"],
        )

    def __repr__(self) -> str:
        return f"FuzzyIndex(algorithm={self._algorithm!r}, size={len(self._items)})"
