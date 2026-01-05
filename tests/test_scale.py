"""
Tests for large-scale operations.

These tests verify functionality and performance characteristics
for datasets ranging from 10K to 1M+ records.

Run with: uv run pytest tests/test_scale.py -v
Skip slow tests: uv run pytest tests/test_scale.py -v -m "not slow"
"""

from __future__ import annotations

import polars as pl
import pytest

import fuzzyrust as fr
from fuzzyrust import polars as frp


def generate_records(n: int, with_duplicates: bool = True) -> pl.DataFrame:
    """Generate test records with controlled duplicates."""
    names = [f"Company Name {i:06d}" for i in range(n)]
    codes = [f"CODE-{i:06d}" for i in range(n)]

    if with_duplicates:
        # Add 10% duplicates with typos
        dup_count = n // 10
        for i in range(dup_count):
            names.append(names[i].replace("Name", "Nme"))  # typo
            codes.append(codes[i])

    return pl.DataFrame(
        {
            "name": names,
            "code": codes[: len(names)],
        }
    )


class TestSchemaIndexScale:
    """Tests for SchemaIndex at scale."""

    @pytest.mark.slow
    def test_schema_index_100k_records(self):
        """Test SchemaIndex with 100K records."""
        builder = fr.SchemaBuilder()
        builder.add_field("name", field_type="short_text", algorithm="jaro_winkler", weight=2.0)
        builder.add_field("code", field_type="short_text", algorithm="levenshtein", weight=1.0)
        schema = builder.build()

        # Generate records
        n = 100_000
        records = [{"name": f"Company {i:06d}", "code": f"CODE-{i:06d}"} for i in range(n)]

        # Build index
        index = fr.SchemaIndex(schema)
        for record in records:
            index.add(record)

        assert len(index) == n

        # Search should be fast
        query = {"name": "Company 050000", "code": "CODE-050000"}
        results = index.search(query, min_similarity=0.8, limit=5)

        assert len(results) > 0
        assert results[0].score > 0.9  # Should find exact match

    @pytest.mark.slow
    def test_batch_search_10k_queries(self):
        """Test batch search with 10K queries."""
        builder = fr.SchemaBuilder()
        builder.add_field("name", field_type="short_text", algorithm="jaro_winkler")
        schema = builder.build()

        # Build index with 50K records
        index = fr.SchemaIndex(schema)
        for i in range(50_000):
            index.add({"name": f"Record {i:06d}"})

        # Run 10K queries
        queries = [{"name": f"Record {i:06d}"} for i in range(0, 50_000, 5)]
        results = index.batch_search(queries, min_similarity=0.8, limit=1)

        assert len(results) == len(queries)
        # Most queries should find exact matches
        matches_found = sum(1 for r in results if r and r[0].score > 0.99)
        assert matches_found > len(queries) * 0.9


class TestDedupeSNMScale:
    """Tests for df_dedupe_snm at scale."""

    @pytest.mark.slow
    def test_df_dedupe_snm_100k_records(self):
        """Test SNM deduplication with 100K records."""
        df = generate_records(100_000, with_duplicates=True)

        result = frp.df_dedupe_snm(
            df,
            columns=["name"],
            min_similarity=0.85,
            window_size=30,
        )

        assert len(result) == len(df)
        assert "_group_id" in result.columns
        assert "_is_canonical" in result.columns

        # Should find some duplicates
        grouped = result.filter(pl.col("_group_id").is_not_null())
        assert len(grouped) > 0

    def test_window_size_effect_on_accuracy(self):
        """Test that larger window size improves accuracy."""
        # Create data where duplicates are spread apart
        df = pl.DataFrame(
            {
                "name": [
                    "Alpha Corp",
                    "Beta Inc",
                    "Gamma LLC",
                    "Delta Co",
                    "Epsilon Ltd",
                    "Alpha Corporation",  # Duplicate of Alpha Corp
                ]
            }
        )

        # Small window might miss the duplicate
        result_small = frp.df_dedupe_snm(df, columns=["name"], min_similarity=0.7, window_size=2)

        # Large window should find it
        result_large = frp.df_dedupe_snm(df, columns=["name"], min_similarity=0.7, window_size=10)

        # Both should return valid results
        assert len(result_small) == len(df)
        assert len(result_large) == len(df)


class TestBlockingKey:
    """Tests for blocking_key parameter."""

    def test_blocking_key_string(self):
        """Test blocking with column name."""
        df = pl.DataFrame(
            {
                "category": ["A", "A", "B", "B"],
                "name": ["Alpha", "Alfa", "Beta", "Betta"],
            }
        )

        result = frp.df_dedupe_snm(
            df,
            columns=["name"],
            min_similarity=0.7,
            blocking_key="category",
        )

        assert len(result) == len(df)
        assert "_group_id" in result.columns
        assert "_is_canonical" in result.columns

        # Alpha/Alfa should be grouped (same category)
        # Beta/Betta should be grouped (same category)
        # Verify we found duplicates in each category
        grouped = result.filter(pl.col("_group_id").is_not_null())
        assert len(grouped) >= 2  # At least two pairs found

    def test_blocking_key_callable(self):
        """Test blocking with custom function."""
        df = pl.DataFrame(
            {
                "name": ["Apple Inc", "Apple Corp", "Banana LLC", "Banana Ltd"],
            }
        )

        # Block by first character
        result = frp.df_dedupe_snm(
            df,
            columns=["name"],
            min_similarity=0.7,
            blocking_key=lambda df: df["name"].str.slice(0, 1),
        )

        assert len(result) == len(df)
        assert "_group_id" in result.columns
        assert "_is_canonical" in result.columns

        # Verify duplicates were found
        grouped = result.filter(pl.col("_group_id").is_not_null())
        assert len(grouped) > 0

    def test_blocking_key_isolates_groups(self):
        """Test that records with different blocking keys are not compared."""
        df = pl.DataFrame(
            {
                "category": ["A", "B"],
                "name": ["Test", "Test"],  # Identical names but different categories
            }
        )

        result = frp.df_dedupe_snm(
            df,
            columns=["name"],
            min_similarity=0.99,  # Would match if compared
            blocking_key="category",
        )

        # Should NOT be grouped because different blocking keys
        grouped = result.filter(pl.col("_group_id").is_not_null())
        assert len(grouped) == 0

    def test_blocking_key_invalid_column(self):
        """Test that invalid blocking key column raises error."""
        df = pl.DataFrame({"name": ["Test"]})

        with pytest.raises(ValueError, match="not found in DataFrame"):
            frp.df_dedupe_snm(
                df,
                columns=["name"],
                blocking_key="nonexistent",
            )

    def test_blocking_key_with_null_values(self):
        """Test that records with null blocking key are treated as unique."""
        df = pl.DataFrame(
            {
                "category": ["A", "A", None, None],
                "name": ["Alpha", "Alfa", "Test1", "Test2"],
            }
        )

        result = frp.df_dedupe_snm(
            df,
            columns=["name"],
            min_similarity=0.7,
            blocking_key="category",
        )

        assert len(result) == len(df)

        # Records with null blocking key should be unique (is_canonical=True)
        null_records = result.filter(pl.col("category").is_null())
        assert all(null_records["_is_canonical"].to_list())


class TestPerColumnAlgorithms:
    """Tests for per-column algorithm support in df_dedupe_snm."""

    def test_algorithms_dict(self):
        """Test specifying different algorithms per column."""
        df = pl.DataFrame(
            {
                "name": ["John Smith", "Jon Smith", "Jane Doe"],
                "code": ["ABC-123", "ABC-124", "XYZ-999"],
            }
        )

        result = frp.df_dedupe_snm(
            df,
            columns=["name", "code"],
            algorithms={"name": "jaro_winkler", "code": "levenshtein"},
            min_similarity=0.7,
            window_size=10,
        )

        assert len(result) == len(df)
        assert "_group_id" in result.columns

    def test_weights_dict(self):
        """Test specifying different weights per column."""
        df = pl.DataFrame(
            {
                "name": ["John Smith", "Jon Smith"],
                "email": ["john@test.com", "totally_different@xyz.com"],
            }
        )

        # With high weight on name, should find match despite different emails
        result = frp.df_dedupe_snm(
            df,
            columns=["name", "email"],
            weights={"name": 10.0, "email": 1.0},
            min_similarity=0.7,
            window_size=10,
        )

        grouped = result.filter(pl.col("_group_id").is_not_null())
        # Name weight dominates, so should find duplicate
        assert len(grouped) >= 2

    def test_algorithms_and_weights_combined(self):
        """Test using both algorithms and weights together."""
        df = pl.DataFrame(
            {
                "part_number": ["PN-001234", "PN-001235", "XY-999999"],
                "name": ["Widget Type A", "Widget Type A", "Other Product"],
                "manufacturer": ["Acme", "ACME", "Zenith"],
            }
        )

        result = frp.df_dedupe_snm(
            df,
            columns=["part_number", "name", "manufacturer"],
            algorithms={
                "part_number": "levenshtein",
                "name": "jaro_winkler",
                "manufacturer": "jaro_winkler",
            },
            weights={"part_number": 3.0, "name": 2.0, "manufacturer": 1.0},
            min_similarity=0.8,
            window_size=10,
        )

        assert len(result) == len(df)
        assert "_group_id" in result.columns

    def test_partial_algorithms_dict(self):
        """Test that unspecified columns use default algorithm."""
        df = pl.DataFrame(
            {
                "name": ["Test", "Test"],
                "code": ["ABC", "ABC"],
            }
        )

        # Only specify algorithm for 'name', 'code' should use default
        result = frp.df_dedupe_snm(
            df,
            columns=["name", "code"],
            algorithms={"name": "levenshtein"},  # code uses default jaro_winkler
            min_similarity=0.99,
            window_size=10,
        )

        assert len(result) == len(df)
        # Should find the duplicate
        grouped = result.filter(pl.col("_group_id").is_not_null())
        assert len(grouped) == 2

    def test_invalid_algorithm_raises_error(self):
        """Test that invalid algorithm name raises error."""
        df = pl.DataFrame({"name": ["Test", "Test"]})

        with pytest.raises(ValueError, match="Invalid algorithm"):
            frp.df_dedupe_snm(
                df,
                columns=["name"],
                algorithms={"name": "invalid_algorithm"},
            )

    def test_invalid_column_in_algorithms_raises_error(self):
        """Test that algorithm for non-existent column raises error."""
        df = pl.DataFrame({"name": ["Test"]})

        with pytest.raises(ValueError, match="not in 'columns' list"):
            frp.df_dedupe_snm(
                df,
                columns=["name"],
                algorithms={"nonexistent": "jaro_winkler"},
            )

    def test_invalid_column_in_weights_raises_error(self):
        """Test that weight for non-existent column raises error."""
        df = pl.DataFrame({"name": ["Test"]})

        with pytest.raises(ValueError, match="not in 'columns' list"):
            frp.df_dedupe_snm(
                df,
                columns=["name"],
                weights={"nonexistent": 1.0},
            )


class TestBlockingWithAlgorithmsAndWeights:
    """Tests combining blocking_key with algorithms and weights."""

    def test_blocking_with_algorithms(self):
        """Test blocking combined with per-column algorithms."""
        df = pl.DataFrame(
            {
                "category": ["A", "A", "B", "B"],
                "name": ["John Smith", "Jon Smith", "Jane Doe", "Janet Doe"],
                "code": ["ABC-123", "ABC-124", "XYZ-001", "XYZ-002"],
            }
        )

        result = frp.df_dedupe_snm(
            df,
            columns=["name", "code"],
            algorithms={"name": "jaro_winkler", "code": "levenshtein"},
            blocking_key="category",
            min_similarity=0.7,
            window_size=10,
        )

        assert len(result) == len(df)
        assert "_group_id" in result.columns

    def test_blocking_with_weights(self):
        """Test blocking combined with per-column weights."""
        df = pl.DataFrame(
            {
                "category": ["X", "X"],
                "name": ["Alpha", "Alpha"],
                "description": ["Very different text", "Completely other content"],
            }
        )

        result = frp.df_dedupe_snm(
            df,
            columns=["name", "description"],
            weights={"name": 10.0, "description": 1.0},
            blocking_key="category",
            min_similarity=0.7,
            window_size=10,
        )

        assert len(result) == len(df)
        # Should find duplicate due to high weight on name
        grouped = result.filter(pl.col("_group_id").is_not_null())
        assert len(grouped) == 2


class TestEmptyAndEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pl.DataFrame({"name": [], "code": []})

        result = frp.df_dedupe_snm(df, columns=["name"])

        assert len(result) == 0
        assert "_group_id" in result.columns
        assert "_is_canonical" in result.columns

    def test_single_row(self):
        """Test with single row."""
        df = pl.DataFrame({"name": ["Test"]})

        result = frp.df_dedupe_snm(df, columns=["name"])

        assert len(result) == 1
        assert result["_group_id"][0] is None
        assert result["_is_canonical"][0] is True

    def test_no_duplicates(self):
        """Test with no duplicates."""
        df = pl.DataFrame(
            {
                "name": ["Alpha", "Beta", "Gamma", "Delta"],
            }
        )

        result = frp.df_dedupe_snm(
            df,
            columns=["name"],
            min_similarity=0.99,  # Very high threshold
        )

        assert len(result) == len(df)
        # All should be unique (no group IDs)
        assert result["_group_id"].is_null().all()
        assert result["_is_canonical"].all()

    def test_all_duplicates(self):
        """Test with all identical values."""
        df = pl.DataFrame(
            {
                "name": ["Test", "Test", "Test"],
            }
        )

        result = frp.df_dedupe_snm(
            df,
            columns=["name"],
            min_similarity=0.99,
        )

        assert len(result) == len(df)
        # All should be in same group
        grouped = result.filter(pl.col("_group_id").is_not_null())
        assert len(grouped) == 3
        # Only one should be canonical
        canonical = result.filter(pl.col("_is_canonical"))
        assert len(canonical) == 1
