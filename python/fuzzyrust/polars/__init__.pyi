"""Type stubs for fuzzyrust.polars module.

Polars integration for FuzzyRust. Provides fuzzy string matching capabilities
for Polars DataFrames and Series.
"""

from typing import Literal, Union

import polars as pl

from fuzzyrust.enums import Algorithm

# Type alias for DataFrame or LazyFrame
DataFrameType = Union[pl.DataFrame, pl.LazyFrame]

# Type alias for column configuration in multi-column joins
ColumnConfig = tuple[str, str, dict[str, Union[str, float]]]

# =============================================================================
# Series Operations
# =============================================================================

def series_similarity(
    left: pl.Series,
    right: pl.Series,
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    ngram_size: int = 3,
) -> pl.Series:
    """Compute pairwise similarity between two aligned Series.

    Args:
        left: First string Series.
        right: Second string Series (must be same length as left).
        algorithm: Similarity algorithm to use.
        ngram_size: N-gram size for ngram algorithm (default: 3).

    Returns:
        Float64 Series with similarity scores (0.0 to 1.0).

    Raises:
        ValueError: If Series have different lengths or unknown algorithm.
    """
    ...

def series_best_match(
    queries: pl.Series,
    targets: list[str],
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.0,
    limit: int = 1,
    normalize: str | None = None,
) -> pl.Series:
    """Find best match for each query from a target list.

    Args:
        queries: Series of query strings to match.
        targets: List of target strings to match against.
        algorithm: Similarity algorithm to use.
        min_similarity: Minimum similarity threshold (0.0 to 1.0).
        limit: Maximum matches per query (default 1).
        normalize: Normalization mode for comparison.

    Returns:
        Utf8 Series with best matching target for each query.
    """
    ...

def series_dedupe(
    series: pl.Series,
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.85,
    normalize: str = "lowercase",
) -> pl.DataFrame:
    """Deduplicate a Series, grouping similar values together.

    Args:
        series: Series of strings to deduplicate.
        algorithm: Similarity algorithm to use.
        min_similarity: Minimum similarity to consider as duplicates.
        normalize: Normalization mode before comparison.

    Returns:
        DataFrame with columns: value, group_id, is_canonical.
    """
    ...

def series_match(
    queries: pl.Series,
    targets: pl.Series,
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.0,
) -> pl.DataFrame:
    """Match each query against all targets.

    Args:
        queries: Series of query strings.
        targets: Series of target strings to match against.
        algorithm: Similarity algorithm to use.
        min_similarity: Minimum similarity threshold.

    Returns:
        DataFrame with columns: query_idx, query, target_idx, target, score.
    """
    ...

# =============================================================================
# DataFrame Operations
# =============================================================================

def df_join(
    left: DataFrameType,
    right: DataFrameType,
    on: str | list[tuple[str, str] | ColumnConfig] | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.8,
    how: Literal["inner", "left"] = "inner",
) -> pl.DataFrame:
    """Fuzzy join two DataFrames based on string similarity.

    Args:
        left: Left DataFrame or LazyFrame.
        right: Right DataFrame or LazyFrame.
        on: Column specification for joining.
        left_on: Column name in left DataFrame.
        right_on: Column name in right DataFrame.
        algorithm: Default similarity algorithm.
        min_similarity: Minimum combined similarity threshold.
        how: Join type ("inner" or "left").

    Returns:
        Joined DataFrame with fuzzy_score column.
    """
    ...

def df_dedupe(
    df: DataFrameType,
    columns: list[str],
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.85,
    weights: dict[str, float] | None = None,
    algorithms: dict[str, Union[str, Algorithm]] | None = None,
    keep: Literal["first", "last", "most_complete"] = "first",
) -> pl.DataFrame:
    """Deduplicate DataFrame rows using fuzzy matching.

    Args:
        df: DataFrame or LazyFrame to deduplicate.
        columns: List of column names to use for similarity matching.
        algorithm: Default similarity algorithm.
        min_similarity: Minimum combined score to consider as duplicates.
        weights: Optional dict mapping column names to weights.
        algorithms: Optional dict mapping column names to algorithms.
        keep: Strategy for selecting canonical row in each group.

    Returns:
        Original DataFrame with _group_id and _is_canonical columns.
    """
    ...

def df_dedupe_snm(
    df: DataFrameType,
    columns: list[str],
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.85,
    window_size: int = 10,
    keep: Literal["first", "last", "most_complete"] = "first",
) -> pl.DataFrame:
    """Deduplicate DataFrame using Sorted Neighborhood Method (SNM).

    Args:
        df: DataFrame or LazyFrame to deduplicate.
        columns: Columns to use for similarity matching and sorting.
        algorithm: Similarity algorithm to use.
        min_similarity: Minimum score to consider as duplicates.
        window_size: Number of neighbors to compare.
        keep: Strategy for selecting canonical row.

    Returns:
        DataFrame with _group_id and _is_canonical columns.
    """
    ...

def df_match_pairs(
    df: DataFrameType,
    columns: list[str],
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.8,
    weights: dict[str, float] | None = None,
    algorithms: dict[str, Union[str, Algorithm]] | None = None,
) -> pl.DataFrame:
    """Find similar pairs within a DataFrame.

    Args:
        df: DataFrame or LazyFrame to search for similar pairs.
        columns: List of column names to use for matching.
        algorithm: Default similarity algorithm.
        min_similarity: Minimum combined score.
        weights: Optional dict mapping column names to weights.
        algorithms: Optional dict mapping column names to algorithms.

    Returns:
        DataFrame with columns: idx_a, idx_b, score, and {col}_a, {col}_b.
    """
    ...

def df_find_pairs(
    df: DataFrameType,
    columns: list[str],
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.8,
    weights: dict[str, float] | None = None,
    algorithms: dict[str, Union[str, Algorithm]] | None = None,
    method: Literal["snm", "full"] = "snm",
    window_size: int = 10,
    min_field_similarity: float = 0.0,
) -> pl.DataFrame:
    """Find all similar pairs in a DataFrame.

    Args:
        df: DataFrame or LazyFrame to search for similar pairs.
        columns: Columns to use for similarity matching.
        algorithm: Default similarity algorithm.
        min_similarity: Minimum combined score.
        weights: Optional per-column weights (only used with method="full").
        algorithms: Optional per-column algorithms (only used with method="full").
        method: Comparison method ("snm" or "full").
        window_size: Window size for SNM method.
        min_field_similarity: Minimum field similarity threshold.

    Returns:
        DataFrame with columns: idx_a, idx_b, score, and {col}_a, {col}_b.
    """
    ...

def df_match_records(
    queries: DataFrameType,
    targets: DataFrameType,
    columns: list[str],
    algorithm: Union[str, Algorithm] = "jaro_winkler",
    min_similarity: float = 0.0,
    weights: dict[str, float] | None = None,
    algorithms: dict[str, Union[str, Algorithm]] | None = None,
    limit: int = 1,
    min_field_similarity: float = 0.0,
) -> pl.DataFrame:
    """Batch match records from two DataFrames.

    Args:
        queries: DataFrame or LazyFrame containing query records.
        targets: DataFrame or LazyFrame containing target records.
        columns: List of columns to use for matching.
        algorithm: Default similarity algorithm.
        min_similarity: Minimum combined score to return a match.
        weights: Optional per-column weights.
        algorithms: Optional per-column algorithms.
        limit: Maximum matches per query (default 1).
        min_field_similarity: Minimum field similarity threshold.

    Returns:
        DataFrame with query_idx, target_idx, score, and value columns.
    """
    ...

# =============================================================================
# Exports
# =============================================================================

__all__: list[str]
