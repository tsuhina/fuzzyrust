"""
Polars integration for FuzzyRust.

This module provides fuzzy string matching capabilities for Polars DataFrames
and Series, with three levels of functionality:

Levels:
    1. **Expression Namespace** (`.fuzzy`) - Per-row operations
       Use this for column-wise fuzzy operations in Polars expressions.
       Example: `df.with_columns(score=pl.col("name").fuzzy.similarity("John"))`

    2. **DataFrame Functions** - Joins, deduplication
       Use this for fuzzy joins between DataFrames and deduplication.
       Example: `fuzzy_join(left_df, right_df, "name", "company")`

    3. **Batch API** - 100K+ row optimization
       Use this for high-performance batch operations on large datasets.
       Example: `dedupe_snm(df, columns=["name"], window_size=100)`

Examples:
    >>> import polars as pl
    >>> import fuzzyrust.polars as frp  # or: from fuzzyrust import polars as frp

    # Expression namespace (registered automatically when importing fuzzyrust)
    >>> import fuzzyrust
    >>> df = pl.DataFrame({"name": ["John", "Jon", "Jane"]})
    >>> df.with_columns(score=pl.col("name").fuzzy.similarity("John"))

    # DataFrame functions
    >>> left = pl.DataFrame({"name": ["Apple Inc"]})
    >>> right = pl.DataFrame({"company": ["Apple"]})
    >>> frp.fuzzy_join(left, right, "name", "company", min_similarity=0.5)

    # Batch API for large datasets
    >>> df = pl.DataFrame({"name": ["John Smith", "Jon Smith", "Jane Doe"] * 10000})
    >>> deduped = frp.dedupe_snm(df, columns=["name"], min_similarity=0.8)
"""

# Expression namespace is registered on import
# (importing fuzzyrust.expr handles this)
import fuzzyrust.expr as _expr  # noqa: F401

# DataFrame functions for joins and deduplication
from fuzzyrust.polars_ext import (
    dedupe_series,
    fuzzy_dedupe_rows,
    fuzzy_join,
    match_dataframe,
    match_series,
)

# High-performance batch API for large datasets
from fuzzyrust.polars_api import (
    batch_best_match,
    batch_similarity,
    dedupe_snm,
    find_similar_pairs,
    match_records_batch,
)

__all__ = [
    # DataFrame functions
    "match_series",
    "dedupe_series",
    "match_dataframe",
    "fuzzy_join",
    "fuzzy_dedupe_rows",
    # Batch API
    "batch_similarity",
    "batch_best_match",
    "dedupe_snm",
    "match_records_batch",
    "find_similar_pairs",
]
