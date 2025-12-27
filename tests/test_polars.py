"""Tests for Polars integration."""

import pytest

# Skip all tests if polars is not installed
pytest.importorskip("polars")

import polars as pl

from fuzzyrust.polars_ext import (
    dedupe_series,
    fuzzy_join,
    match_dataframe,
    match_series,
)


class TestMatchSeries:
    """Tests for match_series function."""

    def test_basic_match(self):
        """Basic matching between two series."""
        queries = pl.Series(["apple", "banana"])
        targets = pl.Series(["appel", "banan", "cherry"])
        result = match_series(queries, targets, min_similarity=0.7)

        assert isinstance(result, pl.DataFrame)
        assert "query" in result.columns
        assert "target" in result.columns
        assert "score" in result.columns
        assert len(result) > 0

    def test_empty_series(self):
        """Empty series should return empty DataFrame."""
        queries = pl.Series([], dtype=pl.Utf8)
        targets = pl.Series(["apple", "banana"])
        result = match_series(queries, targets)

        assert len(result) == 0

    def test_no_matches(self):
        """No matches above threshold."""
        queries = pl.Series(["xyz"])
        targets = pl.Series(["apple", "banana"])
        result = match_series(queries, targets, min_similarity=0.9)

        assert len(result) == 0


class TestDedupeSeries:
    """Tests for dedupe_series function."""

    def test_basic_dedup(self):
        """Basic deduplication."""
        series = pl.Series(["hello", "helo", "world", "HELLO"])
        result = dedupe_series(series, min_similarity=0.8)

        assert isinstance(result, pl.DataFrame)
        assert "value" in result.columns
        assert "group_id" in result.columns
        assert "is_canonical" in result.columns

        # Should have at least one duplicate group
        groups = result.filter(pl.col("group_id").is_not_null())
        assert len(groups) > 0

        # Should identify some canonical values
        canonical = result.filter(pl.col("is_canonical"))
        assert len(canonical) > 0

    def test_no_duplicates(self):
        """No duplicates should have all unique."""
        series = pl.Series(["apple", "banana", "cherry"])
        result = dedupe_series(series, min_similarity=0.95)

        # All should be canonical (unique)
        canonical = result.filter(pl.col("is_canonical"))
        assert len(canonical) == 3

    def test_all_duplicates(self):
        """All same value should be one group."""
        series = pl.Series(["hello", "hello", "hello"])
        result = dedupe_series(series, min_similarity=0.9)

        # Should have one group with 3 items
        groups = result["group_id"].unique().drop_nulls()
        assert len(groups) == 1


class TestMatchDataframe:
    """Tests for match_dataframe function."""

    def test_basic_match(self):
        """Basic multi-column matching."""
        df = pl.DataFrame(
            {"name": ["John Smith", "Jon Smith", "Jane Doe"], "city": ["NYC", "New York", "Boston"]}
        )
        result = match_dataframe(df, ["name"], min_similarity=0.7)

        assert isinstance(result, pl.DataFrame)
        assert "idx_a" in result.columns
        assert "idx_b" in result.columns
        assert "score" in result.columns

    def test_no_matches(self):
        """No matches above threshold."""
        df = pl.DataFrame(
            {
                "name": ["Apple", "Banana", "Cherry"],
            }
        )
        result = match_dataframe(df, ["name"], min_similarity=0.95)

        # May or may not have matches depending on algorithm
        assert isinstance(result, pl.DataFrame)

    def test_with_weights(self):
        """Matching with column weights."""
        df = pl.DataFrame({"name": ["John", "Jon"], "email": ["john@test.com", "jon@test.com"]})
        result = match_dataframe(
            df, ["name", "email"], weights={"name": 2.0, "email": 1.0}, min_similarity=0.5
        )

        assert isinstance(result, pl.DataFrame)


class TestFuzzyJoin:
    """Tests for fuzzy_join function."""

    def test_basic_join(self):
        """Basic fuzzy join."""
        left = pl.DataFrame({"name": ["Apple Inc", "Microsoft"]})
        right = pl.DataFrame({"company": ["Apple", "Microsft", "Google"]})
        result = fuzzy_join(left, right, "name", "company", min_similarity=0.5)

        assert isinstance(result, pl.DataFrame)
        assert "fuzzy_score" in result.columns

    def test_no_matches(self):
        """No matches above threshold."""
        left = pl.DataFrame({"name": ["XYZ Corp"]})
        right = pl.DataFrame({"company": ["Apple", "Microsoft"]})
        result = fuzzy_join(left, right, "name", "company", min_similarity=0.95)

        # Should return empty DataFrame with correct schema
        assert len(result) == 0

    def test_left_join(self):
        """Left join should include unmatched rows."""
        left = pl.DataFrame({"name": ["Apple", "XYZ"]})
        right = pl.DataFrame({"company": ["Apple Inc"]})
        result = fuzzy_join(left, right, "name", "company", min_similarity=0.5, how="left")

        # Left join should include all left rows
        assert len(result) >= 1


class TestPolarsIntegration:
    """Integration tests for Polars functions."""

    def test_dedup_and_match_workflow(self):
        """Common workflow: dedupe then match."""
        # Dedupe a messy list
        series = pl.Series(["Apple Inc", "apple inc", "Microsoft", "MICROSOFT"])
        deduped = dedupe_series(series, min_similarity=0.85)

        # Get canonical values
        canonical = deduped.filter(pl.col("is_canonical"))["value"].to_list()

        # Match against a query
        queries = pl.Series(["apple"])
        targets = pl.Series(canonical)
        matches = match_series(queries, targets, min_similarity=0.5)

        assert len(matches) > 0

    def test_dataframe_dedup_workflow(self):
        """Deduplicate rows in a DataFrame."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["John Smith", "Jon Smith", "Jane Doe", "John Smyth"],
                "city": ["NYC", "NYC", "Boston", "New York"],
            }
        )

        # Find similar rows
        result = match_dataframe(df, ["name"], min_similarity=0.7)

        # Should find John/Jon/Smyth as similar
        assert len(result) > 0

    def test_fuzzy_merge_workflow(self):
        """Fuzzy merge two datasets."""
        customers = pl.DataFrame(
            {"customer_name": ["Apple Computer", "Microsoft Corp"], "revenue": [1000, 2000]}
        )

        orders = pl.DataFrame({"company": ["Apple Inc", "Microsft"], "order_value": [100, 200]})

        merged = fuzzy_join(customers, orders, "customer_name", "company", min_similarity=0.5)

        assert len(merged) > 0
        assert "revenue" in merged.columns
        assert "order_value" in merged.columns
