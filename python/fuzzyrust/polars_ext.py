"""Polars integration for FuzzyRust.

This module provides convenience functions for working with Polars
DataFrames and Series for fuzzy matching and deduplication.
"""

from typing import Dict, List, Optional

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None  # type: ignore


def _check_polars():
    """Raise ImportError if polars is not available."""
    if not POLARS_AVAILABLE:
        raise ImportError(
            "polars is required for this function. Install it with: pip install polars"
        )


def match_series(
    query_series: "pl.Series",
    target_series: "pl.Series",
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.0,
) -> "pl.DataFrame":
    """
    Match each value in query_series against all values in target_series.

    For each query, finds all target values above the similarity threshold.

    Args:
        query_series: Series of query strings
        target_series: Series of target strings to match against
        algorithm: Similarity algorithm to use (default: "jaro_winkler")
            Options: "levenshtein", "jaro", "jaro_winkler", "ngram", etc.
        min_similarity: Minimum similarity threshold (0.0 to 1.0)

    Returns:
        DataFrame with columns: query_idx, query, target_idx, target, score

    Example:
        >>> queries = pl.Series(["apple", "banana"])
        >>> targets = pl.Series(["appel", "banan", "cherry"])
        >>> result = match_series(queries, targets, min_similarity=0.7)
        >>> print(result)
    """
    _check_polars()
    import fuzzyrust as fr

    queries = query_series.to_list()
    targets = target_series.to_list()

    results = []
    for query_idx, query in enumerate(queries):
        if query is None:
            continue
        matches = fr.find_best_matches(
            targets,
            str(query),
            algorithm=algorithm,
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
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.85,
    normalize: str = "lowercase",
) -> "pl.DataFrame":
    """
    Deduplicate a Series, grouping similar values together.

    Args:
        series: Series of strings to deduplicate
        algorithm: Similarity algorithm to use (default: "jaro_winkler")
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
    """
    _check_polars()
    import fuzzyrust as fr

    items = [str(x) if x is not None else "" for x in series.to_list()]
    result = fr.find_duplicates(
        items,
        algorithm=algorithm,
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
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
) -> "pl.DataFrame":
    """
    Find similar rows within a DataFrame based on multiple columns.

    Uses SchemaIndex for multi-column matching. Returns pairs of rows
    that are similar based on the specified columns.

    Args:
        df: DataFrame to search for duplicates
        columns: List of column names to use for matching
        algorithm: Similarity algorithm (default: "jaro_winkler")
        min_similarity: Minimum combined score (0.0 to 1.0)
        weights: Optional dict mapping column names to weights

    Returns:
        DataFrame with columns:
        - idx_a: Index of first row in pair
        - idx_b: Index of second row in pair
        - score: Combined similarity score
        - Plus columns showing values from both rows (col_a, col_b)

    Example:
        >>> df = pl.DataFrame({
        ...     "name": ["John Smith", "Jon Smith", "Jane Doe"],
        ...     "city": ["NYC", "New York", "Boston"]
        ... })
        >>> result = match_dataframe(df, ["name", "city"], min_similarity=0.7)
    """
    _check_polars()
    import fuzzyrust as fr

    # Build schema
    builder = fr.SchemaBuilder()
    for col in columns:
        weight = (weights or {}).get(col, 1.0)
        builder.add_field(
            name=col,
            field_type="short_text",
            algorithm=algorithm,
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


def fuzzy_join(
    left: "pl.DataFrame",
    right: "pl.DataFrame",
    left_on: str,
    right_on: str,
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.8,
    how: str = "inner",
) -> "pl.DataFrame":
    """
    Fuzzy join two DataFrames based on string similarity.

    Similar to pandas merge or polars join, but matches based on
    string similarity rather than exact equality.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        left_on: Column name in left DataFrame to join on
        right_on: Column name in right DataFrame to join on
        algorithm: Similarity algorithm (default: "jaro_winkler")
        min_similarity: Minimum similarity threshold for a match
        how: Join type - "inner" (default), "left", or "cross"

    Returns:
        Joined DataFrame with all columns from both DataFrames
        plus a 'fuzzy_score' column indicating match quality.

    Example:
        >>> left = pl.DataFrame({"name": ["Apple Inc", "Microsoft"]})
        >>> right = pl.DataFrame({"company": ["Apple", "Microsft", "Google"]})
        >>> result = fuzzy_join(left, right, "name", "company", min_similarity=0.7)
    """
    _check_polars()
    import fuzzyrust as fr

    left_values = left[left_on].to_list()
    right_values = right[right_on].to_list()

    # For each left value, find best match in right
    match_info = []

    for left_idx, left_val in enumerate(left_values):
        if left_val is None:
            if how == "left":
                match_info.append((left_idx, None, 0.0))
            continue

        result = fr.extract_one(
            str(left_val),
            [str(v) if v is not None else "" for v in right_values],
            score_cutoff=min_similarity,
        )

        if result is not None:
            # Find the index in right
            for right_idx, right_val in enumerate(right_values):
                if str(right_val) == result.text:
                    match_info.append((left_idx, right_idx, result.score))
                    break
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
