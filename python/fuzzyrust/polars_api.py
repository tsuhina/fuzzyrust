"""Optimized batch Polars API for large-scale fuzzy matching.

This module provides high-performance, batch-optimized functions for fuzzy
matching on large Polars DataFrames. It uses vectorized operations and the
Sorted Neighborhood Method (SNM) for scalable deduplication.

When to Use This Module (polars_api)
------------------------------------
- Large datasets (100K+ rows) where performance is critical
- Batch similarity computation between columns
- Scalable deduplication using O(N log N) SNM algorithm
- Fine-grained control over matching parameters (window size, limits)
- When you need to process millions of comparisons efficiently

Performance Characteristics
---------------------------
- 10-40x faster than row-by-row processing for datasets > 10K rows
- Uses Rayon parallelization internally for batch operations
- SNM-based deduplication: O(N log N) vs O(N^2) for naive pairwise comparison
- Minimizes Python/Rust boundary crossings with batch calls

For Simpler Use Cases
---------------------
For small to medium datasets or when ease-of-use is preferred, consider
``fuzzyrust.polars_ext`` which provides:

- ``fuzzy_join()``: User-friendly fuzzy join with automatic best-match selection
- ``fuzzy_dedupe_rows()``: Simple deduplication with keep strategy options
- ``match_dataframe()``: Find similar pairs for manual review

Functions in This Module
------------------------
- ``batch_similarity()``: Compute similarity between two Series in a single call
- ``batch_best_match()``: Find best matches for each query against a target list
- ``dedupe_snm()``: Deduplicate using Sorted Neighborhood Method
- ``match_records_batch()``: Match records from two DataFrames with batch processing
- ``find_similar_pairs()``: Find all similar pairs in a DataFrame (SNM or full)

Example Usage
-------------
>>> import polars as pl
>>> import fuzzyrust as fr
>>>
>>> # Batch similarity between two columns (10-40x faster than row-by-row)
>>> df = df.with_columns(
...     score=fr.batch_similarity(df["col_a"], df["col_b"])
... )
>>>
>>> # Efficient deduplication with SNM for large datasets
>>> result = fr.dedupe_snm(df, columns=["name"], window_size=10)
>>> unique_df = result.filter(pl.col("_is_canonical"))
>>>
>>> # Batch category matching
>>> categories = ["Electronics", "Clothing", "Food", "Home"]
>>> df = df.with_columns(
...     category=fr.batch_best_match(df["raw_category"], categories)
... )

Algorithm Selection Guide
-------------------------
+---------------------+----------------+------------------+
| Algorithm           | Speed          | Best For         |
+=====================+================+==================+
| jaro_winkler        | Fast           | Names, short text|
+---------------------+----------------+------------------+
| levenshtein         | Medium         | Typos, OCR errors|
+---------------------+----------------+------------------+
| ngram               | Fast           | Long text, fuzzy |
+---------------------+----------------+------------------+
| cosine              | Fast           | Document sim.    |
+---------------------+----------------+------------------+

See Also
--------
- ``fuzzyrust.polars_ext``: High-level DataFrame operations for simpler use cases
- ``fuzzyrust.expr``: Polars expression namespace for column operations
- ``fuzzyrust.HybridIndex``: Index structure for repeated searches
"""

from typing import Literal, Optional, Union

import polars as pl

from fuzzyrust._utils import normalize_algorithm
from fuzzyrust.enums import Algorithm


# Backend detection (import-time, not per-call)
class _BackendState:
    """Encapsulates backend state to avoid global variables."""

    value: Optional[str] = None


_backend_state = _BackendState()


def _detect_backend() -> str:
    """Detect available backend at import time.

    Returns "batch" for now. Plugin backend will be added when
    pyo3-polars version compatibility is resolved.
    """
    if _backend_state.value is not None:
        return _backend_state.value

    # For now, only batch backend is available
    # Plugin backend requires pyo3-polars which has version conflicts
    _backend_state.value = "batch"
    return _backend_state.value


class UnionFind:
    """Union-Find data structure for efficient clustering."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def batch_similarity(
    left: "pl.Series",
    right: "pl.Series",
    algorithm: Union[str, Algorithm] = "jaro_winkler",
) -> "pl.Series":
    """
    Compute similarity between two Series in a single batch call.

    This is significantly faster than row-by-row processing for large datasets
    as it minimizes Python/Rust boundary crossings.

    Args:
        left: First string Series
        right: Second string Series (must be same length as left)
        algorithm: Similarity algorithm to use
            Options: "levenshtein", "jaro", "jaro_winkler", "ngram", "cosine"

    Returns:
        Float64 Series with similarity scores (0.0 to 1.0)

    Example:
        >>> df = pl.DataFrame({"a": ["hello", "world"], "b": ["hallo", "word"]})
        >>> df = df.with_columns(score=fr.batch_similarity(df["a"], df["b"]))

    See Also:
        match_series: Match each query against all targets (returns all matches)
        batch_best_match: Find best match from a list of choices
        jaro_winkler_similarity: Single-pair similarity computation
    """
    import fuzzyrust as fr

    algo = normalize_algorithm(algorithm)

    if len(left) != len(right):
        raise ValueError("Series must have equal length")

    left_list = left.to_list()
    right_list = right.to_list()

    # Use algorithm-specific batch function
    algo_map = {
        "levenshtein": fr.levenshtein_similarity,
        "jaro": fr.jaro_similarity,
        "jaro_winkler": fr.jaro_winkler_similarity,
        "ngram": lambda a, b: fr.ngram_similarity(a, b, ngram_size=3),
        "cosine": fr.cosine_similarity_chars,
    }

    if algo not in algo_map:
        raise ValueError(f"Unknown algorithm: {algo}. Valid: {list(algo_map.keys())}")

    # Handle None values by replacing with empty strings for the batch call
    left_strs = [str(a) if a is not None else "" for a in left_list]
    right_strs = [str(b) if b is not None else "" for b in right_list]

    # Use parallel batch processing from Rust
    raw_scores = fr.batch_similarity_pairs(left_strs, right_strs, algo)

    # Convert None values back for rows where input was None
    scores = []
    for i, (a, b) in enumerate(zip(left_list, right_list)):
        if a is None or b is None:
            scores.append(None)
        else:
            scores.append(raw_scores[i])

    return pl.Series("similarity", scores, dtype=pl.Float64)


def batch_best_match(
    queries: "pl.Series",
    targets: list[str],
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.0,
    limit: int = 1,
    normalize: Optional[str] = "lowercase",
) -> "pl.Series":
    """
    Find best matches for each query in a single batch operation.

    Uses HybridIndex with batch_search for optimal performance on large datasets.

    Args:
        queries: Series of query strings
        targets: List of target strings to match against
        algorithm: Similarity algorithm to use (string or Algorithm enum)
        min_similarity: Minimum similarity threshold (0.0 to 1.0)
        limit: Maximum matches per query (default 1 for best match only)
        normalize: Optional normalization mode. Use "lowercase" for case-insensitive
            comparison (default). Use None for case-sensitive comparison.

    Returns:
        Series with best matching target (or null if below min_similarity)

    Example:
        >>> categories = ["Electronics", "Clothing", "Food", "Home"]
        >>> df = df.with_columns(
        ...     category=fr.batch_best_match(df["raw_category"], categories)
        ... )

    See Also:
        batch_similarity: Compute pairwise similarity between aligned Series
        match_series: Match queries against targets with full result details
        HybridIndex: Index structure for repeated best-match searches
    """
    import fuzzyrust as fr

    algo = normalize_algorithm(algorithm)

    # Build index for efficient batch search
    index = fr.HybridIndex()
    index.add_all(targets)

    query_list = queries.to_list()
    query_strs = [str(q) if q is not None else "" for q in query_list]

    # Single batch call to Rust
    results = index.batch_search(
        queries=query_strs,
        algorithm=algo,
        min_similarity=min_similarity,
        limit=limit,
        normalize=normalize,
    )

    # Extract best matches
    matches = []
    for i, query in enumerate(query_list):
        if query is None:
            matches.append(None)
        elif results[i]:
            matches.append(results[i][0].text)
        else:
            matches.append(None)

    return pl.Series("best_match", matches, dtype=pl.Utf8)


def dedupe_snm(
    df: "pl.DataFrame",
    columns: list[str],
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.85,
    window_size: int = 10,
    keep: Literal["first", "last", "most_complete"] = "first",
) -> "pl.DataFrame":
    """
    Deduplicate DataFrame using Sorted Neighborhood Method (SNM).

    SNM provides O(N log N) complexity compared to O(N²) for naive pairwise
    comparison, making it practical for datasets with 100K+ rows.

    The algorithm:
    1. Sorts records by a blocking key (concatenated columns)
    2. Compares each record only with neighbors within a window
    3. Clusters similar records using Union-Find

    Args:
        df: DataFrame to deduplicate
        columns: Columns to use for similarity matching
        algorithm: Similarity algorithm to use
        min_similarity: Minimum score to consider as duplicates
        window_size: Number of neighbors to compare (larger = more accurate, slower)
        keep: Strategy for selecting canonical row:
            - "first": Keep first row by original index
            - "last": Keep last row by original index
            - "most_complete": Keep row with fewest null/empty values

    Returns:
        DataFrame with added columns:
        - _group_id: Integer group ID for duplicate clusters (null for unique)
        - _is_canonical: True for the row to keep in each group

    Example:
        >>> result = fr.dedupe_snm(
        ...     df,
        ...     columns=["name", "email"],
        ...     min_similarity=0.8,
        ...     window_size=20,
        ... )
        >>> unique_df = result.filter(pl.col("_is_canonical"))

    See Also:
        fuzzy_dedupe_rows: Full O(N^2) deduplication with multi-column algorithms
        dedupe_series: Simple deduplication for a single Series
        find_similar_pairs: Find pairs without clustering (for review)
        find_duplicate_pairs: Low-level SNM pair finding
    """
    import fuzzyrust as fr

    algo = normalize_algorithm(algorithm)
    n = len(df)
    if n == 0:
        return df.with_columns(
            pl.lit(None).cast(pl.Int64).alias("_group_id"),
            pl.lit(True).alias("_is_canonical"),
        )

    # Create blocking key by concatenating columns
    items = []
    for row in df.iter_rows(named=True):
        key = " ".join(str(row[col]) if row[col] is not None else "" for col in columns)
        items.append(key)

    # Use SNM to find duplicate pairs - single Rust call
    pairs = fr.find_duplicate_pairs(
        items=items,
        algorithm=algo,
        min_similarity=min_similarity,
        window_size=window_size,
    )

    # Build clusters using Union-Find
    uf = UnionFind(n)
    for i, j, _score in pairs:
        uf.union(i, j)

    # Group rows by cluster root
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # Assign group IDs and determine canonical rows
    group_ids: list[Optional[int]] = [None] * n
    is_canonical: list[bool] = [True] * n

    group_counter = 0
    for members in clusters.values():
        if len(members) == 1:
            continue

        for idx in members:
            group_ids[idx] = group_counter

        if keep == "first":
            canonical_idx = min(members)
        elif keep == "last":
            canonical_idx = max(members)
        else:  # most_complete

            def completeness(idx: int) -> int:
                count = 0
                row = df.row(idx, named=True)
                for col in df.columns:
                    val = row[col]
                    if val is not None and str(val).strip() != "":
                        count += 1
                return count

            canonical_idx = max(members, key=completeness)

        for idx in members:
            if idx != canonical_idx:
                is_canonical[idx] = False

        group_counter += 1

    return df.with_columns(
        pl.Series("_group_id", group_ids, dtype=pl.Int64),
        pl.Series("_is_canonical", is_canonical),
    )


def match_records_batch(
    queries_df: "pl.DataFrame",
    targets_df: "pl.DataFrame",
    columns: list[str],
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.0,
    weights: Optional[dict[str, float]] = None,
    algorithms: Optional[dict[str, Union[str, Algorithm]]] = None,
    limit: int = 1,
) -> "pl.DataFrame":
    """
    Match records from queries DataFrame against targets using batch processing.

    Uses SchemaIndex.batch_search for optimal performance with a single
    Python/Rust boundary crossing regardless of query count.

    Args:
        queries_df: DataFrame containing query records
        targets_df: DataFrame containing target records to match against
        columns: List of columns to use for matching
        algorithm: Default similarity algorithm
        min_similarity: Minimum combined score
        weights: Optional per-column weights
        algorithms: Optional per-column algorithms
        limit: Maximum matches per query (default 1)

    Returns:
        DataFrame with columns:
        - query_idx: Index of query row
        - target_idx: Index of best matching target row
        - score: Combined similarity score
        - Plus all columns from both DataFrames

    Example:
        >>> matches = fr.match_records_batch(
        ...     customers_df,
        ...     vendors_df,
        ...     columns=["name", "address"],
        ...     algorithms={"name": "jaro_winkler", "address": "ngram"},
        ...     weights={"name": 2.0, "address": 1.0},
        ...     min_similarity=0.7,
        ... )

    See Also:
        fuzzy_join: High-level fuzzy join with automatic DataFrame construction
        match_dataframe: Find similar pairs within a single DataFrame
        SchemaIndex: Low-level multi-field matching index
    """
    import fuzzyrust as fr

    default_algo = normalize_algorithm(algorithm)

    # Build schema
    builder = fr.SchemaBuilder()
    for col in columns:
        weight = (weights or {}).get(col, 1.0)
        col_algo = (algorithms or {}).get(col, default_algo)
        col_algo = normalize_algorithm(col_algo) if col_algo != default_algo else default_algo
        builder.add_field(
            name=col,
            field_type="short_text",
            algorithm=col_algo,
            weight=weight,
        )
    schema = builder.build()
    index = fr.SchemaIndex(schema)

    # Add all target records
    target_records = []
    for row in targets_df.iter_rows(named=True):
        record = {col: str(row[col]) if row[col] is not None else "" for col in columns}
        target_records.append(record)
        index.add(record)

    # Build query records
    query_records = []
    for row in queries_df.iter_rows(named=True):
        record = {col: str(row[col]) if row[col] is not None else "" for col in columns}
        query_records.append(record)

    # Single batch call to Rust
    all_results = index.batch_search(
        queries=query_records,
        min_similarity=min_similarity,
        limit=limit,
        min_field_similarity=0.0,
    )

    # Build result DataFrame
    rows = []
    for query_idx, matches in enumerate(all_results):
        if not matches:
            continue

        best = matches[0]
        target_idx = best.id
        score = best.score

        row_data = {
            "query_idx": query_idx,
            "target_idx": target_idx,
            "score": score,
        }

        # Add query columns with _query suffix
        query_row = queries_df.row(query_idx, named=True)
        for col in queries_df.columns:
            row_data[f"{col}_query"] = query_row[col]

        # Add target columns with _target suffix
        target_row = targets_df.row(target_idx, named=True)
        for col in targets_df.columns:
            row_data[f"{col}_target"] = target_row[col]

        rows.append(row_data)

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows)


def find_similar_pairs(
    df: "pl.DataFrame",
    columns: list[str],
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.8,
    weights: Optional[dict[str, float]] = None,
    algorithms: Optional[dict[str, Union[str, Algorithm]]] = None,
    method: Literal["snm", "full"] = "snm",
    window_size: int = 10,
) -> "pl.DataFrame":
    """
    Find all pairs of similar rows in a DataFrame.

    Supports two methods:
    - "snm": Sorted Neighborhood Method - O(N log N), for large datasets
    - "full": Full pairwise comparison - O(N²), more accurate but slower

    Args:
        df: DataFrame to search for similar pairs
        columns: Columns to use for similarity matching
        algorithm: Default similarity algorithm
        min_similarity: Minimum combined score
        weights: Optional per-column weights
        algorithms: Optional per-column algorithms
        method: "snm" for scalable, "full" for exhaustive
        window_size: Window size for SNM method

    Returns:
        DataFrame with columns:
        - idx_a: Index of first row
        - idx_b: Index of second row
        - score: Similarity score
        - Plus columns showing values from both rows

    Example:
        >>> # Fast SNM for large datasets
        >>> pairs = fr.find_similar_pairs(
        ...     df, columns=["name", "email"], method="snm", window_size=20
        ... )
        >>>
        >>> # Full comparison for small datasets or maximum accuracy
        >>> pairs = fr.find_similar_pairs(
        ...     df, columns=["name"], method="full"
        ... )

    See Also:
        match_dataframe: Full pairwise comparison with per-column algorithms
        dedupe_snm: Cluster similar rows into groups
        fuzzy_dedupe_rows: Deduplication with keep strategy selection
        find_duplicate_pairs: Low-level SNM pair finding
    """
    import fuzzyrust as fr

    default_algo = normalize_algorithm(algorithm)
    n = len(df)
    if n < 2:
        return pl.DataFrame()

    if method == "snm":
        # Use Sorted Neighborhood Method
        items = []
        for row in df.iter_rows(named=True):
            combined_weights = []
            for col in columns:
                val = str(row[col]) if row[col] is not None else ""
                combined_weights.append(val)
            items.append(" ".join(combined_weights))

        # Get pairs from SNM
        pairs = fr.find_duplicate_pairs(
            items=items,
            algorithm=default_algo,
            min_similarity=min_similarity,
            window_size=window_size,
        )

        results = []
        for i, j, score in pairs:
            row_data = {"idx_a": i, "idx_b": j, "score": score}
            row_i = df.row(i, named=True)
            row_j = df.row(j, named=True)
            for col in columns:
                row_data[f"{col}_a"] = row_i[col]
                row_data[f"{col}_b"] = row_j[col]
            results.append(row_data)

        return pl.DataFrame(results)

    else:
        # Full pairwise comparison using SchemaIndex
        builder = fr.SchemaBuilder()
        for col in columns:
            weight = (weights or {}).get(col, 1.0)
            col_algo = (algorithms or {}).get(col, default_algo)
            col_algo = normalize_algorithm(col_algo) if col_algo != default_algo else default_algo
            builder.add_field(
                name=col,
                field_type="short_text",
                algorithm=col_algo,
                weight=weight,
            )
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        records = []
        for row in df.iter_rows(named=True):
            record = {col: str(row[col]) if row[col] is not None else "" for col in columns}
            records.append(record)
            index.add(record)

        # Use batch_search for all records
        all_results = index.batch_search(
            queries=records,
            min_similarity=min_similarity,
            limit=n,  # Get all matches
            min_field_similarity=0.0,
        )

        results = []
        seen_pairs = set()

        for i, matches in enumerate(all_results):
            for match in matches:
                j = match.id
                if i == j:
                    continue

                pair = (min(i, j), max(i, j))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                row_data = {"idx_a": i, "idx_b": j, "score": match.score}
                row_i = df.row(i, named=True)
                row_j = df.row(j, named=True)
                for col in columns:
                    row_data[f"{col}_a"] = row_i[col]
                    row_data[f"{col}_b"] = row_j[col]
                results.append(row_data)

        return pl.DataFrame(results)
