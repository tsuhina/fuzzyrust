"""High-level Polars DataFrame operations for FuzzyRust.

This module provides user-friendly, high-level functions for common fuzzy
matching operations on Polars DataFrames and Series. It is designed for
ease-of-use and covers the most common use cases.

When to Use This Module (polars_ext)
------------------------------------
- Small to medium datasets (< 100K rows)
- Simple fuzzy joins between two DataFrames
- DataFrame deduplication with Union-Find clustering
- Multi-column matching with per-column algorithms and weights
- When ease-of-use is more important than raw performance

For Larger Datasets
-------------------
For datasets with 100K+ rows, consider using the optimized batch API in
``fuzzyrust.polars_api`` which provides:

- ``batch_similarity()``: Vectorized pairwise similarity (10-40x faster)
- ``batch_best_match()``: Batch matching against a list of choices
- ``dedupe_snm()``: O(N log N) deduplication using Sorted Neighborhood Method
- ``find_similar_pairs()``: Scalable pair-finding with SNM option

Functions in This Module
------------------------
- ``match_series()``: Match query Series against target Series
- ``dedupe_series()``: Deduplicate a Series, grouping similar values
- ``match_dataframe()``: Find similar row pairs within a DataFrame
- ``fuzzy_join()``: Fuzzy join two DataFrames (single or multi-column)
- ``fuzzy_dedupe_rows()``: Deduplicate DataFrame rows with clustering

Example Usage
-------------
>>> import polars as pl
>>> import fuzzyrust as fr
>>>
>>> # Fuzzy join two DataFrames
>>> left = pl.DataFrame({"name": ["Apple Inc", "Microsoft Corp"]})
>>> right = pl.DataFrame({"company": ["Apple", "Microsoft", "Google"]})
>>> result = fr.fuzzy_join(left, right, left_on="name", right_on="company")
>>>
>>> # Find and remove duplicates in a DataFrame
>>> df = pl.DataFrame({
...     "name": ["John Smith", "Jon Smith", "Jane Doe"],
...     "email": ["john@test.com", "jon@test.com", "jane@test.com"],
... })
>>> deduped = fr.fuzzy_dedupe_rows(df, columns=["name", "email"], min_similarity=0.8)
>>> unique_rows = deduped.filter(pl.col("_is_canonical"))

See Also
--------
- ``fuzzyrust.polars_api``: Optimized batch operations for large datasets
- ``fuzzyrust.expr``: Polars expression namespace for column operations
- ``fuzzyrust.FuzzyIndex``: High-level index wrapper for search operations
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import polars as pl

from fuzzyrust._utils import normalize_algorithm
from fuzzyrust.enums import Algorithm


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


def match_series(
    query_series: "pl.Series",
    target_series: "pl.Series",
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.0,
) -> "pl.DataFrame":
    """
    Match each value in query_series against all values in target_series.

    For each query, finds all target values above the similarity threshold.

    Args:
        query_series: Series of query strings
        target_series: Series of target strings to match against
        algorithm: Similarity algorithm to use (string or Algorithm enum)
            Options: "levenshtein", "jaro", "jaro_winkler", "ngram", etc.
        min_similarity: Minimum similarity threshold (0.0 to 1.0)

    Returns:
        DataFrame with columns: query_idx, query, target_idx, target, score

    Example:
        >>> queries = pl.Series(["apple", "banana"])
        >>> targets = pl.Series(["appel", "banan", "cherry"])
        >>> result = match_series(queries, targets, min_similarity=0.7)
        >>> print(result)

    See Also:
        batch_similarity: Compute pairwise similarity between aligned Series
        batch_best_match: Find best match for each query from a target list
        match_records_batch: Batch matching for multi-column records
    """
    import fuzzyrust as fr

    algo = normalize_algorithm(algorithm)
    queries = query_series.to_list()
    targets = target_series.to_list()

    results = []
    for query_idx, query in enumerate(queries):
        if query is None:
            continue
        matches = fr.find_best_matches(
            targets,
            str(query),
            algorithm=algo,
            limit=len(targets),
            min_similarity=min_similarity,
        )
        for target_idx, target in enumerate(targets):
            if target is None:
                continue
            # Find score for this target
            for match in matches:
                if match.text == str(target):
                    results.append(
                        {
                            "query_idx": query_idx,
                            "query": str(query),
                            "target_idx": target_idx,
                            "target": str(target),
                            "score": match.score,
                        }
                    )
                    break

    return pl.DataFrame(results)


def dedupe_series(
    series: "pl.Series",
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.85,
    normalize: str = "lowercase",
) -> "pl.DataFrame":
    """
    Deduplicate a Series, grouping similar values together.

    Args:
        series: Series of strings to deduplicate
        algorithm: Similarity algorithm to use (string or Algorithm enum)
        min_similarity: Minimum similarity to consider as duplicates (default: 0.85)
        normalize: Normalization mode: "none", "lowercase", "unicode_nfkd",
            "remove_punctuation", "remove_whitespace", "strict"

    Returns:
        DataFrame with columns:
        - value: The original string value
        - group_id: Group identifier (None for unique values)
        - is_canonical: True for the first (canonical) value in each group

    Example:
        >>> series = pl.Series(["hello", "helo", "world", "HELLO"])
        >>> result = dedupe_series(series, min_similarity=0.8)
        >>> print(result.filter(pl.col("group_id").is_not_null()))

    See Also:
        fuzzy_dedupe_rows: Deduplicate DataFrame rows with multi-column matching
        dedupe_snm: O(N log N) deduplication using Sorted Neighborhood Method
        find_duplicates: Low-level deduplication returning DuplicationResult
    """
    import fuzzyrust as fr

    algo = normalize_algorithm(algorithm)
    items = [str(x) if x is not None else "" for x in series.to_list()]
    result = fr.find_duplicates(
        items,
        algorithm=algo,
        min_similarity=min_similarity,
        normalize=normalize,
    )

    rows = []
    for group_id, group in enumerate(result.groups):
        for i, item in enumerate(group):
            rows.append(
                {
                    "value": item,
                    "group_id": group_id,
                    "is_canonical": i == 0,
                }
            )

    for item in result.unique:
        rows.append(
            {
                "value": item,
                "group_id": None,
                "is_canonical": True,
            }
        )

    return pl.DataFrame(rows)


def match_dataframe(
    df: "pl.DataFrame",
    columns: List[str],
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
    algorithms: Optional[Dict[str, Union[str, Algorithm]]] = None,
) -> "pl.DataFrame":
    """
    Find similar rows within a DataFrame based on multiple columns.

    Uses SchemaIndex for multi-column matching. Returns pairs of rows
    that are similar based on the specified columns.

    Args:
        df: DataFrame to search for duplicates
        columns: List of column names to use for matching
        algorithm: Default similarity algorithm (string or Algorithm enum)
        min_similarity: Minimum combined score (0.0 to 1.0)
        weights: Optional dict mapping column names to weights
        algorithms: Optional dict mapping column names to algorithms.
            Overrides the default algorithm for specific columns.
            Options: "levenshtein", "damerau_levenshtein", "jaro_winkler",
            "ngram", "jaccard", "cosine", "exact_match"

    Returns:
        DataFrame with columns:
        - idx_a: Index of first row in pair
        - idx_b: Index of second row in pair
        - score: Combined similarity score
        - Plus columns showing values from both rows (col_a, col_b)

    Example:
        >>> df = pl.DataFrame({
        ...     "name": ["John Smith", "Jon Smith", "Jane Doe"],
        ...     "email": ["john@test.com", "jon@test.com", "jane@test.com"],
        ...     "phone": ["555-1234", "555-1234", "555-9999"]
        ... })
        >>> # Use different algorithms for different fields
        >>> result = match_dataframe(
        ...     df,
        ...     columns=["name", "email", "phone"],
        ...     algorithms={"name": "jaro_winkler", "email": "levenshtein", "phone": "exact_match"},
        ...     weights={"name": 2.0, "email": 1.0, "phone": 1.0},
        ...     min_similarity=0.7
        ... )

    See Also:
        find_similar_pairs: Scalable pair-finding with SNM option for large datasets
        fuzzy_dedupe_rows: Group similar rows and select canonical representatives
        fuzzy_join: Join two DataFrames on fuzzy-matched columns
    """
    import fuzzyrust as fr

    default_algo = normalize_algorithm(algorithm)

    # Build schema with per-column algorithms
    builder = fr.SchemaBuilder()
    for col in columns:
        weight = (weights or {}).get(col, 1.0)
        col_algo_raw = (algorithms or {}).get(col, default_algo)
        col_algorithm = normalize_algorithm(col_algo_raw)
        builder.add_field(
            name=col,
            field_type="short_text",
            algorithm=col_algorithm,
            weight=weight,
        )
    schema = builder.build()
    index = fr.SchemaIndex(schema)

    # Add all records
    records = []
    for row in df.iter_rows(named=True):
        record = {col: str(row[col]) if row[col] is not None else "" for col in columns}
        records.append(record)
        index.add(record)

    # Self-match to find similar pairs
    results = []
    seen_pairs = set()

    for i, record in enumerate(records):
        query = {col: record[col] for col in columns}
        matches = index.search(query, min_similarity=min_similarity)

        for match in matches:
            j = match.id
            if i == j:  # Skip self-match
                continue

            # Avoid duplicate pairs (i,j) and (j,i)
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            row_data = {
                "idx_a": i,
                "idx_b": j,
                "score": match.score,
            }
            for col in columns:
                row_data[f"{col}_a"] = record[col]
                row_data[f"{col}_b"] = records[j][col]

            results.append(row_data)

    return pl.DataFrame(results)


# Type alias for column configuration in multi-column fuzzy_join
ColumnConfig = Tuple[str, str, Dict[str, Union[str, float]]]


def fuzzy_join(
    left: "pl.DataFrame",
    right: "pl.DataFrame",
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    on: Optional[List[Union[Tuple[str, str], ColumnConfig]]] = None,
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.8,
    how: Literal["inner", "left"] = "inner",
    combine_scores: Literal["weighted_average", "min", "max", "harmonic_mean"] = "weighted_average",
) -> "pl.DataFrame":
    """
    Fuzzy join two DataFrames based on string similarity.

    Similar to pandas merge or polars join, but matches based on
    string similarity rather than exact equality. Supports both single-column
    and multi-column matching with per-column algorithms and weights.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        left_on: Column name in left DataFrame (for single-column join)
        right_on: Column name in right DataFrame (for single-column join)
        on: List of column pairs for multi-column join. Each element can be:
            - Tuple of (left_col, right_col) using default algorithm
            - Tuple of (left_col, right_col, config_dict) where config_dict has:
                - "algorithm": Algorithm name or Algorithm enum
                - "weight": Column weight for scoring (default: 1.0)
        algorithm: Default similarity algorithm (string or Algorithm enum)
        min_similarity: Minimum combined similarity threshold for a match
        how: Join type - "inner" (default) or "left"
        combine_scores: How to combine scores for multi-column matching:
            - "weighted_average": Weighted average of scores (default)
            - "min": Minimum score across all columns
            - "max": Maximum score across all columns
            - "harmonic_mean": Harmonic mean of scores

    Returns:
        Joined DataFrame with all columns from both DataFrames
        plus a 'fuzzy_score' column indicating match quality.

    Example (single-column):
        >>> left = pl.DataFrame({"name": ["Apple Inc", "Microsoft"]})
        >>> right = pl.DataFrame({"company": ["Apple", "Microsft", "Google"]})
        >>> result = fuzzy_join(left, right, left_on="name", right_on="company")

    Example (multi-column):
        >>> left = pl.DataFrame({
        ...     "name": ["John Smith", "Jane Doe"],
        ...     "city": ["New York", "Los Angeles"]
        ... })
        >>> right = pl.DataFrame({
        ...     "customer": ["Jon Smith", "Jane Do"],
        ...     "location": ["New York", "LA"]
        ... })
        >>> result = fuzzy_join(
        ...     left, right,
        ...     on=[
        ...         ("name", "customer", {"algorithm": "jaro_winkler", "weight": 2.0}),
        ...         ("city", "location", {"algorithm": "levenshtein", "weight": 1.0}),
        ...     ],
        ...     min_similarity=0.7,
        ...     combine_scores="weighted_average"
        ... )

    See Also:
        match_records_batch: Batch matching for large-scale record linkage
        match_dataframe: Find similar pairs within a single DataFrame
        fuzzy_dedupe_rows: Deduplicate rows in a DataFrame
    """
    import fuzzyrust as fr

    default_algo = normalize_algorithm(algorithm)

    # Normalize input to multi-column format
    if on is not None:
        # Multi-column mode
        column_configs: List[Tuple[str, str, str, float]] = []
        for item in on:
            if len(item) == 2:
                left_col, right_col = item
                column_configs.append((left_col, right_col, default_algo, 1.0))
            else:
                left_col, right_col, config = item
                col_algo_raw = config.get("algorithm", default_algo)
                col_algo = normalize_algorithm(col_algo_raw)
                col_weight = config.get("weight", 1.0)
                column_configs.append((left_col, right_col, col_algo, col_weight))
    elif left_on is not None and right_on is not None:
        # Single-column mode (backwards compatible)
        column_configs = [(left_on, right_on, default_algo, 1.0)]
    else:
        raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")

    # Build schema for multi-column matching
    builder = fr.SchemaBuilder()
    for left_col, right_col, col_algo, col_weight in column_configs:
        # Use left column name as field name
        builder.add_field(
            name=left_col,
            field_type="short_text",
            algorithm=col_algo,
            weight=col_weight,
        )
    schema = builder.build()
    index = fr.SchemaIndex(schema)

    # Index all right rows
    right_records = []
    for row in right.iter_rows(named=True):
        record = {}
        for left_col, right_col, _, _ in column_configs:
            val = row[right_col]
            record[left_col] = str(val) if val is not None else ""
        right_records.append(record)
        index.add(record)

    # For each left row, find best match in right
    match_info: List[Tuple[int, Optional[int], float]] = []

    for left_idx, left_row in enumerate(left.iter_rows(named=True)):
        # Build query from left row
        query = {}
        has_null = False
        for left_col, _, _, _ in column_configs:
            val = left_row[left_col]
            if val is None:
                has_null = True
            query[left_col] = str(val) if val is not None else ""

        if has_null and how != "left":
            continue
        if has_null and how == "left":
            match_info.append((left_idx, None, 0.0))
            continue

        # Search for matches
        matches = index.search(query, min_similarity=min_similarity)

        if matches:
            # Take best match
            best = matches[0]
            match_info.append((left_idx, best.id, best.score))
        elif how == "left":
            match_info.append((left_idx, None, 0.0))

    # Build result DataFrame
    if not match_info:
        # Return empty DataFrame with correct schema
        result_schema = {}
        for col in left.columns:
            result_schema[col] = left[col].dtype
        for col in right.columns:
            if col not in result_schema:
                result_schema[col] = right[col].dtype
        result_schema["fuzzy_score"] = pl.Float64
        return pl.DataFrame(schema=result_schema)

    rows = []
    for left_idx, right_idx, score in match_info:
        row = {}
        # Add left columns
        for col in left.columns:
            row[col] = left[col][left_idx]

        # Add right columns
        if right_idx is not None:
            for col in right.columns:
                if col in row:
                    row[f"{col}_right"] = right[col][right_idx]
                else:
                    row[col] = right[col][right_idx]
        else:
            for col in right.columns:
                if col in row:
                    row[f"{col}_right"] = None
                else:
                    row[col] = None

        row["fuzzy_score"] = score
        rows.append(row)

    return pl.DataFrame(rows)


def fuzzy_dedupe_rows(
    df: "pl.DataFrame",
    columns: List[str],
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.85,
    weights: Optional[Dict[str, float]] = None,
    algorithms: Optional[Dict[str, Union[str, Algorithm]]] = None,
    keep: Literal["first", "last", "most_complete"] = "first",
) -> "pl.DataFrame":
    """
    Deduplicate a DataFrame by finding and grouping similar rows.

    Uses multi-column fuzzy matching to identify duplicate rows and adds
    columns indicating group membership and which row to keep.

    Args:
        df: DataFrame to deduplicate
        columns: List of column names to use for similarity matching
        algorithm: Default similarity algorithm (string or Algorithm enum)
        min_similarity: Minimum combined score to consider as duplicates (0.0 to 1.0)
        weights: Optional dict mapping column names to weights
        algorithms: Optional dict mapping column names to algorithms.
            Options: "levenshtein", "damerau_levenshtein", "jaro_winkler",
            "ngram", "jaccard", "cosine", "exact_match"
        keep: Strategy for selecting canonical row in each group:
            - "first": Keep the first row (by original index)
            - "last": Keep the last row (by original index)
            - "most_complete": Keep the row with fewest null/empty values

    Returns:
        Original DataFrame with added columns:
        - _group_id: Integer group ID for duplicate clusters (null for unique rows)
        - _is_canonical: True for the row to keep in each group

    Example:
        >>> df = pl.DataFrame({
        ...     "name": ["John Smith", "Jon Smith", "Jane Doe", "John Smyth"],
        ...     "email": ["john@test.com", "jon@test.com", "jane@test.com", "john@test.com"],
        ...     "phone": ["555-1234", "555-1234", "555-9999", "555-1234"]
        ... })
        >>> result = fuzzy_dedupe_rows(
        ...     df,
        ...     columns=["name", "email"],
        ...     algorithms={"name": "jaro_winkler", "email": "levenshtein"},
        ...     min_similarity=0.7
        ... )
        >>> # Get only unique/canonical rows
        >>> unique_df = result.filter(pl.col("_is_canonical"))

    See Also:
        dedupe_snm: O(N log N) deduplication using Sorted Neighborhood Method
        dedupe_series: Deduplicate a single Series of strings
        match_dataframe: Find similar pairs without grouping
        find_similar_pairs: Scalable pair-finding with SNM option
    """
    import fuzzyrust as fr

    default_algo = normalize_algorithm(algorithm)

    n = len(df)
    if n == 0:
        return df.with_columns(
            pl.lit(None).cast(pl.Int64).alias("_group_id"),
            pl.lit(True).alias("_is_canonical"),
        )

    # Build schema with per-column algorithms
    builder = fr.SchemaBuilder()
    for col in columns:
        weight = (weights or {}).get(col, 1.0)
        col_algo_raw = (algorithms or {}).get(col, default_algo)
        col_algorithm = normalize_algorithm(col_algo_raw)
        builder.add_field(
            name=col,
            field_type="short_text",
            algorithm=col_algorithm,
            weight=weight,
        )
    schema = builder.build()
    index = fr.SchemaIndex(schema)

    # Add all records to index
    records = []
    for row in df.iter_rows(named=True):
        record = {col: str(row[col]) if row[col] is not None else "" for col in columns}
        records.append(record)
        index.add(record)

    # Find all matching pairs and build clusters using Union-Find
    uf = UnionFind(n)
    for i, record in enumerate(records):
        query = {col: record[col] for col in columns}
        matches = index.search(query, min_similarity=min_similarity)
        for match in matches:
            j = match.id
            if i != j:
                uf.union(i, j)

    # Group rows by their cluster root
    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # Assign group IDs and determine canonical rows
    group_ids: List[Optional[int]] = [None] * n
    is_canonical: List[bool] = [True] * n

    group_counter = 0
    for root, members in clusters.items():
        if len(members) == 1:
            # Unique row - no group ID, is canonical
            continue

        # Assign group ID to all members
        for idx in members:
            group_ids[idx] = group_counter

        # Determine canonical row based on keep strategy
        if keep == "first":
            canonical_idx = min(members)
        elif keep == "last":
            canonical_idx = max(members)
        else:  # most_complete
            # Count non-null, non-empty values for each row
            def completeness(idx: int) -> int:
                count = 0
                row = df.row(idx, named=True)
                for col in df.columns:
                    val = row[col]
                    if val is not None and str(val).strip() != "":
                        count += 1
                return count

            canonical_idx = max(members, key=completeness)

        # Mark non-canonical rows
        for idx in members:
            if idx != canonical_idx:
                is_canonical[idx] = False

        group_counter += 1

    # Add columns to DataFrame
    return df.with_columns(
        pl.Series("_group_id", group_ids, dtype=pl.Int64),
        pl.Series("_is_canonical", is_canonical),
    )
