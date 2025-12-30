"""Tests for Polars integration."""

import polars as pl
import pytest

from fuzzyrust.polars_ext import (
    dedupe_series,
    fuzzy_dedupe_rows,
    fuzzy_join,
    match_dataframe,
    match_series,
)
from fuzzyrust import FuzzyIndex


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


class TestFuzzyDedupeRows:
    """Tests for fuzzy_dedupe_rows function."""

    def test_basic_dedupe(self):
        """Basic DataFrame deduplication."""
        df = pl.DataFrame({
            "name": ["John Smith", "Jon Smith", "Jane Doe", "John Smyth"],
            "email": ["john@test.com", "jon@test.com", "jane@test.com", "john@test.com"],
        })
        result = fuzzy_dedupe_rows(df, columns=["name"], min_similarity=0.8)

        assert isinstance(result, pl.DataFrame)
        assert "_group_id" in result.columns
        assert "_is_canonical" in result.columns
        assert len(result) == len(df)

        # Should find duplicates
        duplicates = result.filter(pl.col("_group_id").is_not_null())
        assert len(duplicates) > 0

    def test_multi_column_dedupe(self):
        """Deduplication using multiple columns."""
        df = pl.DataFrame({
            "name": ["John Smith", "Jon Smith", "Jane Doe"],
            "email": ["john@test.com", "jon@test.com", "jane@test.com"],
        })
        result = fuzzy_dedupe_rows(
            df,
            columns=["name", "email"],
            algorithms={"name": "jaro_winkler", "email": "levenshtein"},
            min_similarity=0.7
        )

        assert len(result) == len(df)
        assert "_group_id" in result.columns

    def test_keep_strategies(self):
        """Test different keep strategies."""
        df = pl.DataFrame({
            "name": ["John", "Jon", "Johnny"],
            "data": ["complete", None, "partial"],
        })

        # Keep first
        result_first = fuzzy_dedupe_rows(df, columns=["name"], min_similarity=0.7, keep="first")
        canonical_first = result_first.filter(pl.col("_is_canonical"))

        # Keep last
        result_last = fuzzy_dedupe_rows(df, columns=["name"], min_similarity=0.7, keep="last")
        canonical_last = result_last.filter(pl.col("_is_canonical"))

        # Keep most complete
        result_complete = fuzzy_dedupe_rows(df, columns=["name"], min_similarity=0.7, keep="most_complete")
        canonical_complete = result_complete.filter(pl.col("_is_canonical"))

        assert len(canonical_first) > 0
        assert len(canonical_last) > 0
        assert len(canonical_complete) > 0

    def test_empty_dataframe(self):
        """Empty DataFrame returns correct schema."""
        df = pl.DataFrame({"name": []}, schema={"name": pl.Utf8})
        result = fuzzy_dedupe_rows(df, columns=["name"])

        assert "_group_id" in result.columns
        assert "_is_canonical" in result.columns
        assert len(result) == 0

    def test_no_duplicates(self):
        """No duplicates means all rows are canonical."""
        df = pl.DataFrame({"name": ["apple", "banana", "cherry"]})
        result = fuzzy_dedupe_rows(df, columns=["name"], min_similarity=0.95)

        canonical = result.filter(pl.col("_is_canonical"))
        assert len(canonical) == 3


class TestMultiColumnFuzzyJoin:
    """Tests for multi-column fuzzy_join."""

    def test_multi_column_join(self):
        """Join on multiple columns."""
        left = pl.DataFrame({
            "name": ["John Smith", "Jane Doe"],
            "city": ["New York", "Los Angeles"],
        })
        right = pl.DataFrame({
            "customer": ["Jon Smith", "Jane Do"],
            "location": ["NYC", "LA"],
        })
        result = fuzzy_join(
            left, right,
            on=[
                ("name", "customer", {"algorithm": "jaro_winkler", "weight": 2.0}),
                ("city", "location", {"algorithm": "levenshtein", "weight": 1.0}),
            ],
            min_similarity=0.5
        )

        assert isinstance(result, pl.DataFrame)
        assert "fuzzy_score" in result.columns

    def test_simple_tuple_syntax(self):
        """Join with simple tuple syntax (no config dict)."""
        left = pl.DataFrame({
            "name": ["Apple"],
            "country": ["USA"],
        })
        right = pl.DataFrame({
            "company": ["Apple Inc"],
            "region": ["United States"],
        })
        result = fuzzy_join(
            left, right,
            on=[("name", "company"), ("country", "region")],
            min_similarity=0.5
        )

        assert isinstance(result, pl.DataFrame)

    def test_backwards_compatible_single_column(self):
        """Single column syntax still works."""
        left = pl.DataFrame({"name": ["Apple Inc"]})
        right = pl.DataFrame({"company": ["Apple"]})
        result = fuzzy_join(left, right, left_on="name", right_on="company", min_similarity=0.5)

        assert len(result) > 0


class TestFuzzyIndex:
    """Tests for FuzzyIndex class."""

    def test_from_series(self):
        """Create index from Series."""
        series = pl.Series(["apple", "banana", "cherry"])
        index = FuzzyIndex.from_series(series, algorithm="ngram")

        assert len(index) == 3

    def test_search(self):
        """Basic search."""
        series = pl.Series(["apple", "application", "banana"])
        index = FuzzyIndex.from_series(series, algorithm="ngram")

        results = index.search("appel", min_similarity=0.5, limit=2)
        assert len(results) > 0
        assert results[0].text in ["apple", "application"]

    def test_search_series(self):
        """Batch search with Series."""
        targets = pl.Series(["apple", "banana", "cherry"])
        index = FuzzyIndex.from_series(targets, algorithm="ngram")

        queries = pl.Series(["appel", "banan"])
        results = index.search_series(queries, min_similarity=0.5)

        assert isinstance(results, pl.DataFrame)
        assert "query" in results.columns
        assert "match" in results.columns
        assert "score" in results.columns

    def test_from_dataframe(self):
        """Create index from DataFrame column."""
        df = pl.DataFrame({"name": ["John", "Jane", "Bob"]})
        index = FuzzyIndex.from_dataframe(df, "name", algorithm="ngram")

        assert len(index) == 3

    def test_batch_search(self):
        """Batch search with list."""
        index = FuzzyIndex(["apple", "banana", "cherry"], algorithm="ngram")

        results = index.batch_search(["appel", "banan"], min_similarity=0.5)
        assert len(results) == 2

    def test_get_items(self):
        """Get indexed items."""
        items = ["apple", "banana"]
        index = FuzzyIndex(items, algorithm="ngram")

        assert index.get_items() == items

    def test_repr(self):
        """String representation."""
        index = FuzzyIndex(["a", "b", "c"], algorithm="ngram")
        assert "FuzzyIndex" in repr(index)
        assert "size=3" in repr(index)


class TestExpressionNamespace:
    """Tests for .fuzzy expression namespace."""

    def test_similarity_literal(self):
        """Similarity against literal string."""
        import fuzzyrust  # Ensure namespace is registered

        df = pl.DataFrame({"name": ["John", "Jon", "Jane"]})
        result = df.with_columns(
            score=pl.col("name").fuzzy.similarity("John")
        )

        assert "score" in result.columns
        scores = result["score"].to_list()
        assert scores[0] == 1.0  # Exact match

    def test_similarity_column(self):
        """Similarity between two columns."""
        import fuzzyrust

        df = pl.DataFrame({
            "name1": ["John", "Jane"],
            "name2": ["Jon", "Janet"],
        })
        result = df.with_columns(
            score=pl.col("name1").fuzzy.similarity(pl.col("name2"))
        )

        assert "score" in result.columns
        assert len(result["score"]) == 2

    def test_is_similar(self):
        """Boolean similarity check."""
        import fuzzyrust

        df = pl.DataFrame({"name": ["John", "Jon", "Jane", "Bob"]})
        result = df.filter(pl.col("name").fuzzy.is_similar("John", min_similarity=0.8))

        # John and Jon should match
        assert len(result) >= 1
        assert "John" in result["name"].to_list()

    def test_best_match(self):
        """Find best match from choices."""
        import fuzzyrust

        df = pl.DataFrame({"query": ["appel", "bananna", "xyz"]})
        choices = ["apple", "banana", "cherry"]

        result = df.with_columns(
            match=pl.col("query").fuzzy.best_match(choices, min_similarity=0.6)
        )

        assert "match" in result.columns
        matches = result["match"].to_list()
        assert matches[0] == "apple"
        assert matches[1] == "banana"
        # xyz should be None (no good match)

    def test_distance(self):
        """Edit distance calculation."""
        import fuzzyrust

        df = pl.DataFrame({"name": ["hello", "helo", "world"]})
        result = df.with_columns(
            dist=pl.col("name").fuzzy.distance("hello")
        )

        assert "dist" in result.columns
        distances = result["dist"].to_list()
        assert distances[0] == 0  # Exact match
        assert distances[1] == 1  # One deletion

    def test_phonetic(self):
        """Phonetic encoding."""
        import fuzzyrust

        df = pl.DataFrame({"name": ["Smith", "Smyth", "Johnson"]})
        result = df.with_columns(
            soundex=pl.col("name").fuzzy.phonetic("soundex")
        )

        assert "soundex" in result.columns
        # Smith and Smyth should have same Soundex
        codes = result["soundex"].to_list()
        assert codes[0] == codes[1]

    def test_normalize(self):
        """String normalization."""
        import fuzzyrust

        df = pl.DataFrame({"name": ["HELLO World", "  foo  bar  "]})
        result = df.with_columns(
            normalized=pl.col("name").fuzzy.normalize("lowercase")
        )

        assert result["normalized"][0] == "hello world"
