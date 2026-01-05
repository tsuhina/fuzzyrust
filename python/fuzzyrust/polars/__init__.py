"""Polars integration for FuzzyRust.

This module provides fuzzy string matching capabilities for Polars DataFrames
and Series. It consolidates all Polars-related functionality into a single,
coherent API with clear naming conventions.

API Overview
------------

**Series Operations** (pl.Series -> pl.Series or pl.DataFrame):
    - ``series_similarity()``: Compute pairwise similarity between two aligned Series
    - ``series_best_match()``: Find best match for each query from a target list
    - ``series_dedupe()``: Deduplicate a Series, returns DataFrame with group assignments
    - ``series_match()``: Match each query against all targets

**DataFrame Operations** (pl.DataFrame -> pl.DataFrame):
    - ``df_join()``: Fuzzy join two DataFrames
    - ``df_dedupe()``: Deduplicate DataFrame rows with Union-Find clustering
    - ``df_dedupe_snm()``: Deduplicate using Sorted Neighborhood Method (large datasets)
    - ``df_match_pairs()``: Find similar pairs within a DataFrame
    - ``df_find_pairs()``: Find all similar pairs using SNM or full comparison
    - ``df_match_records()``: Batch match records from two DataFrames

**Expression Namespace** (``.fuzzy``):
    Access fuzzy operations directly on Polars expressions:
    ``pl.col("name").fuzzy.similarity("John")``

Performance Guide
-----------------

+------------------+----------------------------+-----------------------------+
| Dataset Size     | Recommended Functions      | Notes                       |
+==================+============================+=============================+
| < 10K rows       | Any function               | All perform well            |
+------------------+----------------------------+-----------------------------+
| 10K - 100K rows  | df_dedupe, df_join         | Standard functions work     |
+------------------+----------------------------+-----------------------------+
| > 100K rows      | df_dedupe_snm, df_find_pairs| Use SNM for O(N log N)     |
+------------------+----------------------------+-----------------------------+
| > 1M rows        | df_dedupe_snm with large   | Increase window_size        |
|                  | window_size                | for accuracy                |
+------------------+----------------------------+-----------------------------+

Examples
--------
>>> import polars as pl
>>> import fuzzyrust.polars as frp

# Series operations
>>> left = pl.Series(["hello", "world"])
>>> right = pl.Series(["hallo", "word"])
>>> scores = frp.series_similarity(left, right)

# DataFrame fuzzy join
>>> left_df = pl.DataFrame({"name": ["Apple Inc", "Microsoft Corp"]})
>>> right_df = pl.DataFrame({"company": ["Apple", "Microsoft", "Google"]})
>>> result = frp.df_join(left_df, right_df, on="name", right_on="company")

# Large-scale deduplication with SNM
>>> df = pl.DataFrame({"name": ["John Smith", "Jon Smith", "Jane Doe"] * 100000})
>>> deduped = frp.df_dedupe_snm(df, columns=["name"], window_size=20)
>>> unique_df = deduped.filter(pl.col("_is_canonical"))

See Also
--------
- ``fuzzyrust.expr``: Polars expression namespace for column operations
- ``fuzzyrust.FuzzyIndex``: High-level index wrapper for search operations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable

# Import the expression namespace to ensure it's registered
import fuzzyrust.expr as _expr  # noqa: F401

# Import utilities from _utils (avoid duplication)
from fuzzyrust._utils import (
    VALID_ALGORITHMS,
    UnionFind,
)
from fuzzyrust._utils import (
    normalize_algorithm as _normalize_algorithm,
)
from fuzzyrust.expr import FuzzyExprNamespace

if TYPE_CHECKING:
    from fuzzyrust.enums import Algorithm

# Type alias for DataFrame or LazyFrame
DataFrameType = Union[pl.DataFrame, pl.LazyFrame]

# Type alias for column configuration in multi-column joins
ColumnConfig = tuple[str, str, dict[str, Union[str, float]]]


def _ensure_dataframe(df: DataFrameType) -> pl.DataFrame:
    """Convert LazyFrame to DataFrame if needed.

    Args:
        df: DataFrame or LazyFrame to convert.

    Returns:
        Collected DataFrame.
    """
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    return df


def _validate_schema_options(
    df: pl.DataFrame,
    columns: list[str],
    algorithms: dict[str, str | Algorithm] | None = None,
    weights: dict[str, float] | None = None,
) -> None:
    """Validate schema options against DataFrame columns.

    Args:
        df: DataFrame to validate against
        columns: List of column names that should exist in DataFrame
        algorithms: Optional dict mapping column names to algorithms
        weights: Optional dict mapping column names to weights

    Raises:
        ValueError: If columns don't exist or dict keys don't match columns
    """
    from fuzzyrust.enums import Algorithm

    df_columns = set(df.columns)
    column_set = set(columns)

    # Validate columns exist in DataFrame
    missing = column_set - df_columns
    if missing:
        available = sorted(df_columns)
        raise ValueError(
            f"Columns not found in DataFrame: {sorted(missing)}. Available columns: {available}"
        )

    # Validate algorithm dict keys match specified columns
    if algorithms:
        invalid_keys = set(algorithms.keys()) - column_set
        if invalid_keys:
            raise ValueError(
                f"Algorithm specified for columns not in 'columns' list: {sorted(invalid_keys)}. "
                f"Valid columns: {sorted(column_set)}"
            )

        # Validate algorithm values are valid
        for col, algo in algorithms.items():
            if isinstance(algo, str):
                algo_lower = algo.lower()
                # Check against valid algorithms (allow Algorithm enum too)
                enum_values = {a.value for a in Algorithm}
                if algo_lower not in VALID_ALGORITHMS and algo_lower not in enum_values:
                    valid_list = sorted(VALID_ALGORITHMS | enum_values)
                    raise ValueError(
                        f"Invalid algorithm '{algo}' for column '{col}'. "
                        f"Valid algorithms: {valid_list}"
                    )

    # Validate weight dict keys match specified columns
    if weights:
        invalid_keys = set(weights.keys()) - column_set
        if invalid_keys:
            raise ValueError(
                f"Weight specified for columns not in 'columns' list: {sorted(invalid_keys)}. "
                f"Valid columns: {sorted(column_set)}"
            )


# =============================================================================
# Series Operations
# =============================================================================


def series_similarity(
    left: pl.Series,
    right: pl.Series,
    algorithm: str | Algorithm = "jaro_winkler",
    ngram_size: int = 3,
) -> pl.Series:
    """Compute pairwise similarity between two aligned Series.

    Computes similarity scores between corresponding elements of two Series.
    Both Series must have the same length.

    This function is optimized for batch processing, minimizing Python/Rust
    boundary crossings for better performance on large datasets.

    Args:
        left: First string Series.
        right: Second string Series (must be same length as left).
        algorithm: Similarity algorithm to use. Options:
            - "levenshtein": Edit distance based similarity
            - "damerau_levenshtein" or "damerau": Edit distance with transpositions
            - "jaro": Jaro similarity (good for short strings)
            - "jaro_winkler": Jaro-Winkler (favors matching prefixes)
            - "ngram": N-gram based similarity (uses ngram_size parameter)
            - "bigram": Bigram similarity (n=2)
            - "trigram": Trigram similarity (n=3)
            - "cosine": Character-level cosine similarity
            - "lcs": Longest common subsequence similarity
            - "hamming": Hamming similarity (for equal-length strings)
        ngram_size: N-gram size for ngram algorithm (default: 3).

    Returns:
        Float64 Series with similarity scores (0.0 to 1.0).
        Returns None for rows where either input is None.

    Raises:
        ValueError: If Series have different lengths or unknown algorithm.

    Example:
        >>> import polars as pl
        >>> import fuzzyrust.polars as frp
        >>>
        >>> left = pl.Series(["hello", "world", "foo"])
        >>> right = pl.Series(["hallo", "word", "bar"])
        >>> scores = frp.series_similarity(left, right)
        >>> print(scores)
        shape: (3,)
        Series: 'similarity' [f64]
        [0.933333, 0.916667, 0.0]

    See Also:
        series_best_match: Find best match from a list of choices.
        series_match: Match each query against all targets.
    """
    import fuzzyrust as fr

    algo = _normalize_algorithm(algorithm)
    # Normalize "damerau" alias to "damerau_levenshtein"
    if algo == "damerau":
        algo = "damerau_levenshtein"

    if len(left) != len(right):
        raise ValueError("Series must have equal length")

    left_list = left.to_list()
    right_list = right.to_list()

    # Use algorithm-specific batch function
    algo_map = {
        "levenshtein": fr.levenshtein_similarity,
        "damerau_levenshtein": fr.damerau_levenshtein_similarity,
        "jaro": fr.jaro_similarity,
        "jaro_winkler": fr.jaro_winkler_similarity,
        "ngram": lambda a, b: fr.ngram_similarity(a, b, ngram_size=ngram_size),
        "bigram": lambda a, b: fr.ngram_similarity(a, b, ngram_size=2),
        "trigram": lambda a, b: fr.ngram_similarity(a, b, ngram_size=3),
        "jaccard": lambda a, b: fr.ngram_jaccard(a, b, ngram_size=ngram_size),
        "cosine": fr.cosine_similarity_chars,
        "lcs": fr.lcs_similarity,
        "hamming": fr.hamming_similarity,
    }

    if algo not in algo_map:
        raise ValueError(f"Unknown algorithm: {algo}. Valid: {list(algo_map.keys())}")

    # Handle None values by replacing with empty strings for the batch call
    left_strs = [str(a) if a is not None else "" for a in left_list]
    right_strs = [str(b) if b is not None else "" for b in right_list]

    # Use parallel batch processing from Rust
    raw_scores = fr.batch.pairwise(left_strs, right_strs, algo)

    # Convert None values back for rows where input was None
    scores = []
    for i, (a, b) in enumerate(zip(left_list, right_list)):
        if a is None or b is None:
            scores.append(None)
        else:
            scores.append(raw_scores[i])

    return pl.Series("similarity", scores, dtype=pl.Float64)


def series_best_match(
    queries: pl.Series,
    targets: list[str],
    algorithm: str | Algorithm = "jaro_winkler",
    min_similarity: float = 0.0,
    limit: int = 1,
    normalize: str | None = None,
) -> pl.Series:
    """Find best match for each query from a target list.

    For each value in the queries Series, finds the most similar string
    from the targets list. Uses HybridIndex internally for efficient
    batch searching.

    Args:
        queries: Series of query strings to match.
        targets: List of target strings to match against.
        algorithm: Similarity algorithm to use. Options:
            - "levenshtein", "jaro", "jaro_winkler", "ngram", "cosine"
        min_similarity: Minimum similarity threshold (0.0 to 1.0).
            Returns None for queries with no match above threshold.
        limit: Maximum matches per query (default 1 for best match only).
        normalize: Normalization mode for comparison:
            - None: Case-sensitive comparison (default)
            - "lowercase": Case-insensitive comparison

    Returns:
        Utf8 Series with best matching target for each query.
        Returns None for queries with no match above min_similarity.

    Example:
        >>> import polars as pl
        >>> import fuzzyrust.polars as frp
        >>>
        >>> categories = ["Electronics", "Clothing", "Food", "Home & Garden"]
        >>> queries = pl.Series(["Elecronics", "cloths", "foods", "xyz"])
        >>> matches = frp.series_best_match(queries, categories, min_similarity=0.6)
        >>> print(matches)
        shape: (4,)
        Series: 'best_match' [str]
        ["Electronics", "Clothing", "Food", null]

    See Also:
        series_similarity: Compute pairwise similarity between aligned Series.
        df_join: Fuzzy join two DataFrames with automatic best-match selection.
    """
    import fuzzyrust as fr

    algo = _normalize_algorithm(algorithm)

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


def series_dedupe(
    series: pl.Series,
    algorithm: str | Algorithm = "jaro_winkler",
    min_similarity: float = 0.85,
    normalize: str = "lowercase",
) -> pl.DataFrame:
    """Deduplicate a Series, grouping similar values together.

    Finds groups of similar strings in the Series and returns a DataFrame
    with group assignments. The first value in each group is marked as
    canonical (the representative value to keep).

    Args:
        series: Series of strings to deduplicate.
        algorithm: Similarity algorithm to use.
        min_similarity: Minimum similarity to consider as duplicates (0.0 to 1.0).
        normalize: Normalization mode before comparison:
            - "none": No normalization
            - "lowercase": Convert to lowercase (default)
            - "unicode_nfkd": Unicode NFKD normalization
            - "remove_punctuation": Remove punctuation
            - "remove_whitespace": Collapse whitespace
            - "strict": All normalizations combined

    Returns:
        DataFrame with columns:
            - value: The original string value
            - group_id: Integer group ID (None for unique values)
            - is_canonical: True for the representative value in each group

    Example:
        >>> import polars as pl
        >>> import fuzzyrust.polars as frp
        >>>
        >>> series = pl.Series(["hello", "helo", "world", "HELLO", "wrld"])
        >>> result = frp.series_dedupe(series, min_similarity=0.8)
        >>> print(result)
        shape: (5, 3)
        +-------+----------+--------------+
        | value | group_id | is_canonical |
        +-------+----------+--------------+
        | hello | 0        | true         |
        | helo  | 0        | false        |
        | HELLO | 0        | false        |
        | world | 1        | true         |
        | wrld  | 1        | false        |
        +-------+----------+--------------+

    See Also:
        df_dedupe: Deduplicate DataFrame rows with multi-column matching.
        df_dedupe_snm: O(N log N) deduplication for large datasets.
    """
    import fuzzyrust as fr

    algo = _normalize_algorithm(algorithm)
    items = [str(x) if x is not None else "" for x in series.to_list()]
    result = fr.batch.deduplicate(
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

    rows.extend({"value": item, "group_id": None, "is_canonical": True} for item in result.unique)

    return pl.DataFrame(rows)


def series_match(
    queries: pl.Series,
    targets: pl.Series,
    algorithm: str | Algorithm = "jaro_winkler",
    min_similarity: float = 0.0,
) -> pl.DataFrame:
    """Match each query against all targets.

    For each value in the queries Series, finds all values in the targets
    Series that meet the similarity threshold. Returns all matching pairs.

    Args:
        queries: Series of query strings.
        targets: Series of target strings to match against.
        algorithm: Similarity algorithm to use.
        min_similarity: Minimum similarity threshold (0.0 to 1.0).

    Returns:
        DataFrame with columns:
            - query_idx: Index of the query in the queries Series
            - query: The query string value
            - target_idx: Index of the target in the targets Series
            - target: The target string value
            - score: Similarity score between query and target

    Example:
        >>> import polars as pl
        >>> import fuzzyrust.polars as frp
        >>>
        >>> queries = pl.Series(["apple", "banana"])
        >>> targets = pl.Series(["appel", "banan", "cherry", "aple"])
        >>> matches = frp.series_match(queries, targets, min_similarity=0.7)
        >>> print(matches)
        shape: (4, 5)
        +-----------+--------+------------+--------+-------+
        | query_idx | query  | target_idx | target | score |
        +-----------+--------+------------+--------+-------+
        | 0         | apple  | 0          | appel  | 0.933 |
        | 0         | apple  | 3          | aple   | 0.900 |
        | 1         | banana | 1          | banan  | 0.967 |
        +-----------+--------+------------+--------+-------+

    See Also:
        series_best_match: Find only the best match for each query.
        series_similarity: Compute pairwise similarity between aligned Series.
    """
    import fuzzyrust as fr

    algo = _normalize_algorithm(algorithm)
    query_list = queries.to_list()
    target_list = targets.to_list()

    # Pre-convert targets to strings and build index for O(1) lookup
    target_strs = [str(t) if t is not None else None for t in target_list]

    results = []
    for query_idx, query in enumerate(query_list):
        if query is None:
            continue
        query_str = str(query)
        matches = fr.batch.best_matches(
            target_list,
            query_str,
            algorithm=algo,
            limit=len(target_list),
            min_similarity=min_similarity,
        )

        # Build dict for O(1) lookup: text -> (score, distance)
        # This avoids O(k) iteration for each target
        match_dict = {match.text: match.score for match in matches}

        # Single pass through targets: O(m) instead of O(m*k)
        for target_idx, target_str in enumerate(target_strs):
            if target_str is None:
                continue
            if target_str in match_dict:
                results.append(
                    {
                        "query_idx": query_idx,
                        "query": query_str,
                        "target_idx": target_idx,
                        "target": target_str,
                        "score": match_dict[target_str],
                    }
                )

    return pl.DataFrame(results)


# =============================================================================
# DataFrame Operations
# =============================================================================


def df_join(
    left: DataFrameType,
    right: DataFrameType,
    on: str | list[tuple[str, str] | ColumnConfig] | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    algorithm: str | Algorithm = "jaro_winkler",
    min_similarity: float = 0.8,
    how: Literal["inner", "left"] = "inner",
) -> pl.DataFrame:
    """Fuzzy join two DataFrames based on string similarity.

    Similar to Polars join, but matches based on string similarity rather than
    exact equality. Supports both single-column and multi-column matching with
    per-column algorithms and weights.

    Note:
        This function requires eager evaluation. If LazyFrames are passed,
        they will be automatically collected before processing.

    Args:
        left: Left DataFrame or LazyFrame.
        right: Right DataFrame or LazyFrame.
        on: Column specification for joining. Can be:
            - A string: Single column name (same in both DataFrames)
            - A list of tuples for multi-column join:
                - (left_col, right_col): Use default algorithm
                - (left_col, right_col, config): Use custom config with:
                    - "algorithm": Algorithm name
                    - "weight": Column weight for scoring
        left_on: Column name in left DataFrame (for single-column join).
            Mutually exclusive with 'on'.
        right_on: Column name in right DataFrame (for single-column join).
            Mutually exclusive with 'on'.
        algorithm: Default similarity algorithm.
        min_similarity: Minimum combined similarity threshold for a match.
        how: Join type:
            - "inner": Only rows with matches (default)
            - "left": All left rows, with None for unmatched

    Returns:
        Joined DataFrame with all columns from both DataFrames plus a
        'fuzzy_score' column indicating match quality.

    Example (single-column):
        >>> import polars as pl
        >>> import fuzzyrust.polars as frp
        >>>
        >>> left = pl.DataFrame({"name": ["Apple Inc", "Microsoft Corp"]})
        >>> right = pl.DataFrame({"company": ["Apple", "Microsft", "Google"]})
        >>> result = frp.df_join(left, right, left_on="name", right_on="company")

    Example (multi-column):
        >>> left = pl.DataFrame({
        ...     "name": ["John Smith", "Jane Doe"],
        ...     "city": ["New York", "Los Angeles"]
        ... })
        >>> right = pl.DataFrame({
        ...     "customer": ["Jon Smith", "Jane Do"],
        ...     "location": ["New York", "LA"]
        ... })
        >>> result = frp.df_join(
        ...     left, right,
        ...     on=[
        ...         ("name", "customer", {"algorithm": "jaro_winkler", "weight": 2.0}),
        ...         ("city", "location", {"algorithm": "levenshtein", "weight": 1.0}),
        ...     ],
        ...     min_similarity=0.7,
        ... )

    See Also:
        df_match_records: Batch matching for large-scale record linkage.
        df_match_pairs: Find similar pairs within a single DataFrame.
    """
    import fuzzyrust as fr

    left = _ensure_dataframe(left)
    right = _ensure_dataframe(right)
    default_algo = _normalize_algorithm(algorithm)

    # Handle 'on' parameter for single column case
    if isinstance(on, str):
        left_on = on
        right_on = on
        on = None

    # Normalize input to multi-column format
    if on is not None:
        # Multi-column mode
        column_configs: list[tuple[str, str, str, float]] = []
        for item in on:
            if len(item) == 2:
                left_col, right_col = item
                column_configs.append((left_col, right_col, default_algo, 1.0))
            else:
                left_col, right_col, config = item
                col_algo_raw = config.get("algorithm", default_algo)
                col_algo = _normalize_algorithm(col_algo_raw)
                col_weight = config.get("weight", 1.0)
                column_configs.append((left_col, right_col, col_algo, col_weight))
    elif left_on is not None and right_on is not None:
        # Single-column mode (backwards compatible)
        column_configs = [(left_on, right_on, default_algo, 1.0)]
    else:
        raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")

    # Validate column existence
    left_columns = set(left.columns)
    right_columns = set(right.columns)
    for left_col, right_col, _, _ in column_configs:
        if left_col not in left_columns:
            raise ValueError(
                f"Column '{left_col}' not found in left DataFrame. "
                f"Available columns: {sorted(left_columns)}"
            )
        if right_col not in right_columns:
            raise ValueError(
                f"Column '{right_col}' not found in right DataFrame. "
                f"Available columns: {sorted(right_columns)}"
            )

    # Build schema for multi-column matching
    builder = fr.SchemaBuilder()
    for left_col, _right_col, col_algo, col_weight in column_configs:
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
    match_info: list[tuple[int, int | None, float]] = []

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


def df_dedupe(
    df: DataFrameType,
    columns: list[str],
    algorithm: str | Algorithm = "jaro_winkler",
    min_similarity: float = 0.85,
    weights: dict[str, float] | None = None,
    algorithms: dict[str, str | Algorithm] | None = None,
    keep: Literal["first", "last", "most_complete"] = "first",
) -> pl.DataFrame:
    """Deduplicate DataFrame rows using fuzzy matching.

    Uses multi-column fuzzy matching to identify duplicate rows and groups
    them using Union-Find clustering. Adds columns indicating group membership
    and which row to keep.

    For datasets larger than 100K rows, consider using ``df_dedupe_snm()``
    which provides O(N log N) complexity.

    Note:
        This function requires eager evaluation. If a LazyFrame is passed,
        it will be automatically collected before processing.

    Args:
        df: DataFrame or LazyFrame to deduplicate.
        columns: List of column names to use for similarity matching.
        algorithm: Default similarity algorithm.
        min_similarity: Minimum combined score to consider as duplicates (0.0 to 1.0).
        weights: Optional dict mapping column names to weights.
            Higher weights give more importance to that column.
        algorithms: Optional dict mapping column names to algorithms.
            Overrides the default algorithm for specific columns.
        keep: Strategy for selecting canonical row in each group:
            - "first": Keep the first row by original index (default)
            - "last": Keep the last row by original index
            - "most_complete": Keep the row with fewest null/empty values

    Returns:
        Original DataFrame with added columns:
            - _group_id: Integer group ID for duplicate clusters (None for unique)
            - _is_canonical: True for the row to keep in each group

    Example:
        >>> import polars as pl
        >>> import fuzzyrust.polars as frp
        >>>
        >>> df = pl.DataFrame({
        ...     "name": ["John Smith", "Jon Smith", "Jane Doe", "John Smyth"],
        ...     "email": ["john@test.com", "jon@test.com", "jane@test.com", "john@test.com"],
        ... })
        >>> result = frp.df_dedupe(
        ...     df,
        ...     columns=["name", "email"],
        ...     algorithms={"name": "jaro_winkler", "email": "levenshtein"},
        ...     min_similarity=0.7,
        ... )
        >>> # Get only unique/canonical rows
        >>> unique_df = result.filter(pl.col("_is_canonical"))

    See Also:
        df_dedupe_snm: O(N log N) deduplication for large datasets.
        series_dedupe: Deduplicate a single Series.
        df_match_pairs: Find similar pairs without grouping.
    """
    import fuzzyrust as fr

    df = _ensure_dataframe(df)

    # Validate inputs
    _validate_schema_options(df, columns, algorithms=algorithms, weights=weights)

    default_algo = _normalize_algorithm(algorithm)

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
        col_algorithm = _normalize_algorithm(col_algo_raw)
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
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # Assign group IDs and determine canonical rows
    group_ids: list[int | None] = [None] * n
    is_canonical: list[bool] = [True] * n

    group_counter = 0
    for members in clusters.values():
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


def df_dedupe_snm(
    df: DataFrameType,
    columns: list[str],
    algorithm: str | Algorithm = "jaro_winkler",
    min_similarity: float = 0.85,
    window_size: int = 10,
    keep: Literal["first", "last", "most_complete"] = "first",
    blocking_key: str | Callable[[pl.DataFrame], pl.Series] | None = None,
    algorithms: dict[str, str | Algorithm] | None = None,
    weights: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Deduplicate DataFrame using Sorted Neighborhood Method (SNM).

    SNM provides O(N log N) complexity compared to O(N^2) for naive pairwise
    comparison, making it practical for datasets with 100K+ rows.

    The algorithm:
        1. Sorts records by a blocking key (concatenated columns)
        2. Compares each record only with neighbors within a window
        3. Clusters similar records using Union-Find

    Note:
        This function requires eager evaluation. If a LazyFrame is passed,
        it will be automatically collected before processing.

    Args:
        df: DataFrame or LazyFrame to deduplicate.
        columns: Columns to use for similarity matching and sorting.
        algorithm: Default similarity algorithm (used when not specified in algorithms dict).
        min_similarity: Minimum score to consider as duplicates (0.0 to 1.0).
        window_size: Number of neighbors to compare.
            - Larger values = more accurate but slower
            - Typical range: 5-50
            - Default: 10
        keep: Strategy for selecting canonical row:
            - "first": Keep first row by original index (default)
            - "last": Keep last row by original index
            - "most_complete": Keep row with fewest null/empty values
        blocking_key: Optional blocking key for partitioning data before comparison.
            Can be either:
            - A string: Column name to use as blocking key
            - A callable: Function that takes DataFrame and returns a Series
            Records with the same blocking key are compared together.
            Records with different blocking keys are never compared.
        algorithms: Optional dict mapping column names to algorithms.
            Overrides the default algorithm for specific columns.
            Example: {"name": "jaro_winkler", "code": "levenshtein"}
        weights: Optional dict mapping column names to weights.
            Higher weights give more importance to that column.
            Example: {"name": 2.0, "code": 1.0}

    Returns:
        DataFrame with added columns:
            - _group_id: Integer group ID for duplicate clusters (None for unique)
            - _is_canonical: True for the row to keep in each group

    Example:
        >>> import polars as pl
        >>> import fuzzyrust.polars as frp
        >>>
        >>> # Large dataset with potential duplicates
        >>> df = pl.DataFrame({
        ...     "name": ["John Smith", "Jon Smith", "Jane Doe"],
        ...     "code": ["ABC-123", "ABC-124", "XYZ-999"],
        ...     "category": ["A", "A", "B"],
        ... })
        >>> # Basic usage
        >>> result = frp.df_dedupe_snm(
        ...     df,
        ...     columns=["name", "code"],
        ...     min_similarity=0.8,
        ...     window_size=20,
        ... )
        >>>
        >>> # With per-column algorithms and weights
        >>> result = frp.df_dedupe_snm(
        ...     df,
        ...     columns=["name", "code"],
        ...     algorithms={"name": "jaro_winkler", "code": "levenshtein"},
        ...     weights={"name": 2.0, "code": 1.0},
        ...     min_similarity=0.8,
        ... )
        >>>
        >>> # With blocking key (only compare within same category)
        >>> result = frp.df_dedupe_snm(
        ...     df,
        ...     columns=["name"],
        ...     blocking_key="category",
        ...     min_similarity=0.8,
        ... )
        >>>
        >>> # With custom blocking key function
        >>> result = frp.df_dedupe_snm(
        ...     df,
        ...     columns=["name"],
        ...     blocking_key=lambda df: df["name"].str.slice(0, 1),
        ...     min_similarity=0.8,
        ... )
        >>> unique_df = result.filter(pl.col("_is_canonical"))

    See Also:
        df_dedupe: Full O(N^2) deduplication with multi-column algorithms.
        df_find_pairs: Find pairs without clustering (for review).
    """
    import fuzzyrust as fr

    df = _ensure_dataframe(df)

    # Validate inputs
    _validate_schema_options(df, columns, algorithms=algorithms, weights=weights)

    default_algo = _normalize_algorithm(algorithm)
    n = len(df)
    if n == 0:
        return df.with_columns(
            pl.lit(None).cast(pl.Int64).alias("_group_id"),
            pl.lit(True).alias("_is_canonical"),
        )

    # Handle blocking
    if blocking_key is not None:
        return _df_dedupe_snm_with_blocking(
            df,
            columns,
            default_algo,
            min_similarity,
            window_size,
            keep,
            blocking_key,
            algorithms,
            weights,
        )

    # Check if we need weighted multi-field comparison
    use_weighted = algorithms is not None or weights is not None

    if use_weighted:
        # Use SchemaIndex for weighted multi-field comparison
        return _df_dedupe_snm_weighted(
            df, columns, default_algo, min_similarity, window_size, keep, algorithms, weights
        )

    # Simple single-algorithm mode: concatenate columns and compare
    items = []
    for row in df.iter_rows(named=True):
        key = " ".join(str(row[col]) if row[col] is not None else "" for col in columns)
        items.append(key)

    # Use SNM to find duplicate pairs - single Rust call
    pairs = fr.find_duplicate_pairs(
        items=items,
        algorithm=default_algo,
        min_similarity=min_similarity,
        window_size=window_size,
    )

    return _build_dedup_result(df, pairs, keep)


def _df_dedupe_snm_weighted(
    df: pl.DataFrame,
    columns: list[str],
    default_algo: str,
    min_similarity: float,
    window_size: int,
    keep: str,
    algorithms: dict[str, str | Algorithm] | None,
    weights: dict[str, float] | None,
) -> pl.DataFrame:
    """SNM deduplication with per-column algorithms and weights."""
    import fuzzyrust as fr

    n = len(df)

    # Build schema with per-column algorithms and weights
    builder = fr.SchemaBuilder()
    for col in columns:
        weight = (weights or {}).get(col, 1.0)
        col_algo_raw = (algorithms or {}).get(col, default_algo)
        col_algorithm = _normalize_algorithm(col_algo_raw)
        builder.add_field(
            name=col,
            field_type="short_text",
            algorithm=col_algorithm,
            weight=weight,
        )

    # Get similarity function based on schema
    def compute_weighted_similarity(record_a: dict, record_b: dict) -> float:
        """Compute weighted similarity between two records."""
        total_weight = 0.0
        weighted_sum = 0.0

        for col in columns:
            val_a = record_a.get(col, "")
            val_b = record_b.get(col, "")

            # Skip if either value is empty
            if not val_a or not val_b:
                continue

            weight = (weights or {}).get(col, 1.0)
            col_algo_raw = (algorithms or {}).get(col, default_algo)
            col_algo = _normalize_algorithm(col_algo_raw)

            # Compute similarity using the specified algorithm
            score = _compute_similarity(val_a, val_b, col_algo)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight

    # Extract records for comparison
    records = []
    for row in df.iter_rows(named=True):
        record = {col: str(row[col]) if row[col] is not None else "" for col in columns}
        records.append(record)

    # Create blocking key for sorting (concatenated columns)
    sort_keys = []
    for record in records:
        key = " ".join(record.get(col, "") for col in columns)
        sort_keys.append(key)

    # Sort indices by blocking key
    sorted_indices = sorted(range(n), key=lambda i: sort_keys[i])

    # SNM: Compare within window
    pairs = []
    for pos, i in enumerate(sorted_indices):
        # Compare with next window_size neighbors
        for offset in range(1, min(window_size + 1, n - pos)):
            j = sorted_indices[pos + offset]
            score = compute_weighted_similarity(records[i], records[j])
            if score >= min_similarity:
                pairs.append((i, j, score))

    return _build_dedup_result(df, pairs, keep)


def _compute_similarity(a: str, b: str, algorithm: str) -> float:
    """Compute similarity between two strings using specified algorithm."""
    import fuzzyrust as fr

    algo_map = {
        "levenshtein": fr.levenshtein_similarity,
        "damerau_levenshtein": fr.damerau_levenshtein_similarity,
        "jaro": fr.jaro_similarity,
        "jaro_winkler": fr.jaro_winkler_similarity,
        "ngram": lambda a, b: fr.ngram_similarity(a, b, ngram_size=3),
        "bigram": lambda a, b: fr.ngram_similarity(a, b, ngram_size=2),
        "trigram": lambda a, b: fr.ngram_similarity(a, b, ngram_size=3),
        "jaccard": lambda a, b: fr.ngram_jaccard(a, b, ngram_size=3),
        "cosine": fr.cosine_similarity_chars,
        "lcs": fr.lcs_similarity,
        "hamming": fr.hamming_similarity,
    }

    func = algo_map.get(algorithm, fr.jaro_winkler_similarity)
    return func(a, b)


def _df_dedupe_snm_with_blocking(
    df: pl.DataFrame,
    columns: list[str],
    default_algo: str,
    min_similarity: float,
    window_size: int,
    keep: str,
    blocking_key: str | Callable[[pl.DataFrame], pl.Series],
    algorithms: dict[str, str | Algorithm] | None,
    weights: dict[str, float] | None,
) -> pl.DataFrame:
    """Process deduplication with blocking."""
    n = len(df)

    # Generate blocking key column
    if isinstance(blocking_key, str):
        if blocking_key not in df.columns:
            raise ValueError(
                f"Blocking key column '{blocking_key}' not found in DataFrame. "
                f"Available columns: {sorted(df.columns)}"
            )
        block_col = df[blocking_key]
    else:
        block_col = blocking_key(df)

    # Add blocking column and row index
    df_with_block = df.with_columns(
        [
            block_col.alias("_block_key"),
            pl.arange(0, n, eager=True).alias("_original_idx"),
        ]
    )

    # Process each block
    all_group_ids: list[int | None] = [None] * n
    all_is_canonical: list[bool] = [True] * n
    group_counter = 0

    for block_value in df_with_block["_block_key"].unique().to_list():
        if block_value is None:
            # Records with null blocking key are treated as unique
            continue

        block_mask = df_with_block["_block_key"] == block_value
        block_df = df_with_block.filter(block_mask)
        original_indices = block_df["_original_idx"].to_list()

        if len(block_df) < 2:
            # Single record in block - no duplicates possible
            continue

        # Drop helper columns for processing
        block_df_clean = block_df.drop(["_block_key", "_original_idx"])

        # Run dedup on block (without blocking_key to avoid recursion)
        block_result = df_dedupe_snm(
            block_df_clean,
            columns=columns,
            algorithm=default_algo,
            min_similarity=min_similarity,
            window_size=window_size,
            keep=keep,
            blocking_key=None,
            algorithms=algorithms,
            weights=weights,
        )

        # Map block results back to original indices
        block_group_ids = block_result["_group_id"].to_list()
        block_is_canonical = block_result["_is_canonical"].to_list()

        # Remap group IDs to global numbering
        local_to_global: dict[int, int] = {}
        for block_idx, orig_idx in enumerate(original_indices):
            local_group = block_group_ids[block_idx]
            if local_group is not None:
                if local_group not in local_to_global:
                    local_to_global[local_group] = group_counter
                    group_counter += 1
                all_group_ids[orig_idx] = local_to_global[local_group]
            all_is_canonical[orig_idx] = block_is_canonical[block_idx]

    return df.with_columns(
        pl.Series("_group_id", all_group_ids, dtype=pl.Int64),
        pl.Series("_is_canonical", all_is_canonical),
    )


def _build_dedup_result(
    df: pl.DataFrame,
    pairs: list[tuple[int, int, float]],
    keep: str,
) -> pl.DataFrame:
    """Build deduplication result from pairs using Union-Find clustering."""
    n = len(df)

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
    group_ids: list[int | None] = [None] * n
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


def df_match_pairs(
    df: DataFrameType,
    columns: list[str],
    algorithm: str | Algorithm = "jaro_winkler",
    min_similarity: float = 0.8,
    weights: dict[str, float] | None = None,
    algorithms: dict[str, str | Algorithm] | None = None,
) -> pl.DataFrame:
    """Find similar pairs within a DataFrame.

    Compares all rows against each other and returns pairs that meet the
    similarity threshold. Useful for reviewing potential duplicates before
    deduplication.

    Note:
        This has O(N^2) complexity. For large datasets (>10K rows),
        consider using ``df_find_pairs()`` with method="snm".

    Note:
        This function requires eager evaluation. If a LazyFrame is passed,
        it will be automatically collected before processing.

    Args:
        df: DataFrame or LazyFrame to search for similar pairs.
        columns: List of column names to use for matching.
        algorithm: Default similarity algorithm.
        min_similarity: Minimum combined score (0.0 to 1.0).
        weights: Optional dict mapping column names to weights.
        algorithms: Optional dict mapping column names to algorithms.

    Returns:
        DataFrame with columns:
            - idx_a: Index of first row in pair
            - idx_b: Index of second row in pair
            - score: Combined similarity score
            - {col}_a, {col}_b: Values from both rows for each column

    Example:
        >>> import polars as pl
        >>> import fuzzyrust.polars as frp
        >>>
        >>> df = pl.DataFrame({
        ...     "name": ["John Smith", "Jon Smith", "Jane Doe"],
        ...     "email": ["john@test.com", "jon@test.com", "jane@test.com"],
        ... })
        >>> pairs = frp.df_match_pairs(
        ...     df,
        ...     columns=["name", "email"],
        ...     algorithms={"name": "jaro_winkler", "email": "levenshtein"},
        ...     min_similarity=0.7,
        ... )
        >>> print(pairs)

    See Also:
        df_find_pairs: Scalable pair-finding with SNM option.
        df_dedupe: Group similar rows and select canonical representatives.
    """
    df = _ensure_dataframe(df)
    import fuzzyrust as fr

    # Validate inputs
    _validate_schema_options(df, columns, algorithms=algorithms, weights=weights)

    default_algo = _normalize_algorithm(algorithm)

    # Build schema with per-column algorithms
    builder = fr.SchemaBuilder()
    for col in columns:
        weight = (weights or {}).get(col, 1.0)
        col_algo_raw = (algorithms or {}).get(col, default_algo)
        col_algorithm = _normalize_algorithm(col_algo_raw)
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


def df_find_pairs(
    df: DataFrameType,
    columns: list[str],
    algorithm: str | Algorithm = "jaro_winkler",
    min_similarity: float = 0.8,
    weights: dict[str, float] | None = None,
    algorithms: dict[str, str | Algorithm] | None = None,
    method: Literal["snm", "full"] = "snm",
    window_size: int = 10,
    min_field_similarity: float = 0.0,
) -> pl.DataFrame:
    """Find all similar pairs in a DataFrame.

    Supports two methods:
        - "snm": Sorted Neighborhood Method - O(N log N), for large datasets
        - "full": Full pairwise comparison - O(N^2), more accurate but slower

    Note:
        This function requires eager evaluation. If a LazyFrame is passed,
        it will be automatically collected before processing.

    Args:
        df: DataFrame or LazyFrame to search for similar pairs.
        columns: Columns to use for similarity matching.
        algorithm: Default similarity algorithm.
        min_similarity: Minimum combined score (0.0 to 1.0).
        weights: Optional per-column weights (only used with method="full").
        algorithms: Optional per-column algorithms (only used with method="full").
        method: Comparison method:
            - "snm": Sorted Neighborhood Method (default) - faster for large data
            - "full": Full pairwise comparison - more thorough but O(N^2)
        window_size: Window size for SNM method. Only used when method="snm".
        min_field_similarity: Minimum similarity threshold for individual fields
            in multi-column matching (only used with method="full"). Default 0.0.

    Returns:
        DataFrame with columns:
            - idx_a: Index of first row
            - idx_b: Index of second row
            - score: Similarity score
            - {col}_a, {col}_b: Values from both rows for each column

    Example:
        >>> import polars as pl
        >>> import fuzzyrust.polars as frp
        >>>
        >>> # Fast SNM for large datasets
        >>> pairs = frp.df_find_pairs(
        ...     df,
        ...     columns=["name", "email"],
        ...     method="snm",
        ...     window_size=20,
        ... )
        >>>
        >>> # Full comparison for small datasets or maximum accuracy
        >>> pairs = frp.df_find_pairs(
        ...     df,
        ...     columns=["name"],
        ...     method="full",
        ... )

    See Also:
        df_match_pairs: Full pairwise comparison with per-column algorithms.
        df_dedupe_snm: Cluster similar rows into groups.
    """
    import fuzzyrust as fr

    df = _ensure_dataframe(df)
    default_algo = _normalize_algorithm(algorithm)
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
            col_algo = _normalize_algorithm(col_algo) if col_algo != default_algo else default_algo
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
            min_field_similarity=min_field_similarity,
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


def df_match_records(
    queries: DataFrameType,
    targets: DataFrameType,
    columns: list[str],
    algorithm: str | Algorithm = "jaro_winkler",
    min_similarity: float = 0.0,
    weights: dict[str, float] | None = None,
    algorithms: dict[str, str | Algorithm] | None = None,
    limit: int = 1,
    min_field_similarity: float = 0.0,
) -> pl.DataFrame:
    """Batch match records from two DataFrames.

    Finds the best matching record in targets for each record in queries.
    Uses SchemaIndex with batch_search for optimal performance with a single
    Python/Rust boundary crossing regardless of query count.

    Note:
        This function requires eager evaluation. If LazyFrames are passed,
        they will be automatically collected before processing.

    Args:
        queries: DataFrame or LazyFrame containing query records.
        targets: DataFrame or LazyFrame containing target records to match against.
        columns: List of columns to use for matching (must exist in both DataFrames).
        algorithm: Default similarity algorithm.
        min_similarity: Minimum combined score to return a match.
        weights: Optional per-column weights.
        algorithms: Optional per-column algorithms.
        limit: Maximum matches per query (default 1).
        min_field_similarity: Minimum similarity threshold for individual fields
            in multi-column matching. Default 0.0.

    Returns:
        DataFrame with columns:
            - query_idx: Index of query row
            - target_idx: Index of best matching target row
            - score: Combined similarity score
            - {col}_query: Query values for each column
            - {col}_target: Target values for each column

    Example:
        >>> import polars as pl
        >>> import fuzzyrust.polars as frp
        >>>
        >>> customers = pl.DataFrame({
        ...     "name": ["John Smith", "Jane Doe"],
        ...     "address": ["123 Main St", "456 Oak Ave"],
        ... })
        >>> vendors = pl.DataFrame({
        ...     "name": ["Jon Smith", "J. Doe"],
        ...     "address": ["123 Main Street", "456 Oak Avenue"],
        ... })
        >>> matches = frp.df_match_records(
        ...     customers,
        ...     vendors,
        ...     columns=["name", "address"],
        ...     algorithms={"name": "jaro_winkler", "address": "ngram"},
        ...     weights={"name": 2.0, "address": 1.0},
        ...     min_similarity=0.7,
        ... )

    See Also:
        df_join: High-level fuzzy join with automatic DataFrame construction.
        df_match_pairs: Find similar pairs within a single DataFrame.
    """
    import fuzzyrust as fr

    queries_df = _ensure_dataframe(queries)
    targets_df = _ensure_dataframe(targets)
    default_algo = _normalize_algorithm(algorithm)

    # Build schema
    builder = fr.SchemaBuilder()
    for col in columns:
        weight = (weights or {}).get(col, 1.0)
        col_algo = (algorithms or {}).get(col, default_algo)
        col_algo = _normalize_algorithm(col_algo) if col_algo != default_algo else default_algo
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
        min_field_similarity=min_field_similarity,
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


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Series operations
    "series_similarity",
    "series_best_match",
    "series_dedupe",
    "series_match",
    # DataFrame operations
    "df_join",
    "df_dedupe",
    "df_dedupe_snm",
    "df_match_pairs",
    "df_find_pairs",
    "df_match_records",
    # Expression namespace
    "FuzzyExprNamespace",
    # Utility classes
    "UnionFind",
]
