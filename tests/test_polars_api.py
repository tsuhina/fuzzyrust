"""Tests for the optimized Polars API (polars_api.py)."""

import polars as pl
import pytest

import fuzzyrust as fr
from fuzzyrust.polars_api import batch_similarity as batch_similarity_polars


class TestBatchSimilarity:
    """Tests for batch_similarity function."""

    def test_basic_similarity(self):
        """Test basic similarity computation."""
        left = pl.Series(["hello", "world", "test"])
        right = pl.Series(["hallo", "word", "test"])

        result = batch_similarity_polars(left, right)

        assert len(result) == 3
        # Batch similarity uses jaro-winkler by default
        # hello vs hallo: high similarity (1 char different in middle)
        assert 0.85 <= result[0] <= 0.95, f"Expected 0.85-0.95 for hello vs hallo, got {result[0]}"
        # world vs word: very high similarity (missing 'l')
        assert 0.90 <= result[1] <= 1.0, f"Expected 0.90-1.0 for world vs word, got {result[1]}"
        # test vs test: identical
        assert result[2] == 1.0, f"Expected 1.0 for test vs test, got {result[2]}"

    def test_different_algorithms(self):
        """Test with different algorithms."""
        left = pl.Series(["kitten", "saturday"])
        right = pl.Series(["sitting", "sunday"])

        # Expected score ranges for kitten/sitting and saturday/sunday by algorithm
        # Actual values: kitten/sitting, saturday/sunday
        #   levenshtein: 0.5714, 0.6250
        #   jaro: 0.7460, 0.7528
        #   jaro_winkler: 0.7460, 0.7775
        #   ngram: 0.1176, 0.4444
        #   cosine: 0.7462, 0.7746
        expected_ranges = {
            # (kitten/sitting min, kitten/sitting max, saturday/sunday min, saturday/sunday max)
            "levenshtein": (0.55, 0.60, 0.60, 0.65),  # edit distance based
            "jaro": (0.72, 0.77, 0.73, 0.78),  # character matching
            "jaro_winkler": (0.72, 0.77, 0.75, 0.80),  # prefix bonus
            "ngram": (0.10, 0.15, 0.42, 0.47),  # ngram overlap (lower scores)
            "cosine": (0.72, 0.77, 0.75, 0.80),  # vector similarity
        }

        for algo in ["levenshtein", "jaro", "jaro_winkler", "ngram", "cosine"]:
            result = batch_similarity_polars(left, right, algorithm=algo)
            assert len(result) == 2, f"Expected 2 results for algorithm {algo}"

            # All similarity scores must be in [0, 1] range
            for i, s in enumerate(result):
                if s is not None:
                    assert 0.0 <= s <= 1.0, f"Algorithm {algo} returned invalid score {s} at index {i}"

            # Check algorithm-specific expected ranges for kitten/sitting
            ks_min, ks_max, ss_min, ss_max = expected_ranges[algo]
            assert ks_min <= result[0] <= ks_max, (
                f"Algorithm {algo}: kitten/sitting score {result[0]} outside expected range [{ks_min}, {ks_max}]"
            )
            assert ss_min <= result[1] <= ss_max, (
                f"Algorithm {algo}: saturday/sunday score {result[1]} outside expected range [{ss_min}, {ss_max}]"
            )

    def test_handles_nulls(self):
        """Test null handling."""
        left = pl.Series(["hello", None, "test"])
        right = pl.Series(["hallo", "world", None])

        result = batch_similarity_polars(left, right)

        # hello vs hallo: high similarity (default algorithm may vary)
        assert result[0] is not None, "Expected non-null score for hello vs hallo"
        assert 0.8 <= result[0] <= 0.95, f"Expected 0.8-0.95 for hello vs hallo, got {result[0]}"
        assert result[1] is None, "Expected None when left value is None"
        assert result[2] is None, "Expected None when right value is None"

    def test_unequal_length_raises(self):
        """Test that unequal length series raise error."""
        left = pl.Series(["hello", "world"])
        right = pl.Series(["hallo"])

        with pytest.raises(ValueError, match="equal length"):
            batch_similarity_polars(left, right)


class TestBatchBestMatch:
    """Tests for batch_best_match function."""

    def test_basic_matching(self):
        """Test basic best match finding."""
        queries = pl.Series(["appel", "banan", "cheery"])
        targets = ["apple", "banana", "cherry", "grape"]

        result = fr.batch_best_match(queries, targets, min_similarity=0.7)

        assert result[0] == "apple"
        assert result[1] == "banana"
        assert result[2] == "cherry"

    def test_min_similarity_filtering(self):
        """Test that min_similarity filters out low matches."""
        queries = pl.Series(["xyz", "apple"])
        targets = ["apple", "banana"]

        result = fr.batch_best_match(queries, targets, min_similarity=0.8)

        assert result[0] is None  # xyz has no good match
        assert result[1] == "apple"  # exact match

    def test_handles_nulls(self):
        """Test null handling."""
        queries = pl.Series(["apple", None])
        targets = ["apple", "banana"]

        result = fr.batch_best_match(queries, targets)

        # "apple" should match itself exactly
        assert result[0] == "apple", f"Expected 'apple' as best match, got {result[0]}"
        assert result[1] is None, "Expected None when query is None"


class TestDedupeSNM:
    """Tests for dedupe_snm function using Sorted Neighborhood Method."""

    def test_finds_duplicates(self):
        """Test that duplicates are found."""
        df = pl.DataFrame({
            "name": ["John Smith", "Jon Smith", "Jane Doe", "John Smyth"],
            "id": [1, 2, 3, 4],
        })

        result = fr.dedupe_snm(
            df,
            columns=["name"],
            min_similarity=0.8,
            window_size=10,
        )

        assert "_group_id" in result.columns
        assert "_is_canonical" in result.columns

        # John Smith, Jon Smith, and John Smyth should be grouped together
        # At min_similarity=0.8, these similar names should form at least one group
        grouped = result.filter(pl.col("_group_id").is_not_null())
        assert len(grouped) >= 2, f"Expected at least 2 grouped rows (John/Jon variants), got {len(grouped)}"
        # Verify the grouped names are the expected similar ones
        grouped_names = set(grouped["name"].to_list())
        assert any(name in grouped_names for name in ["John Smith", "Jon Smith", "John Smyth"]), \
            f"Expected Smith variants to be grouped, but got {grouped_names}"

    def test_keep_first(self):
        """Test keep='first' strategy."""
        df = pl.DataFrame({
            "name": ["John Smith", "Jon Smith", "John Smyth"],
        })

        result = fr.dedupe_snm(df, columns=["name"], keep="first")
        canonical = result.filter(pl.col("_is_canonical"))

        # First row should be canonical
        assert canonical["name"][0] == "John Smith"

    def test_keep_last(self):
        """Test keep='last' strategy."""
        df = pl.DataFrame({
            "name": ["John Smith", "Jon Smith", "John Smyth"],
        })

        result = fr.dedupe_snm(df, columns=["name"], keep="last")
        canonical_rows = result.filter(
            pl.col("_is_canonical") & pl.col("_group_id").is_not_null()
        )

        if len(canonical_rows) > 0:
            # Last row in the group should be canonical
            assert canonical_rows["name"][0] in ["Jon Smith", "John Smyth"]

    def test_window_size_effect(self):
        """Test that window_size affects results."""
        # Create data where duplicates are far apart in sorted order
        df = pl.DataFrame({
            "name": ["aaa", "zzz duplicate", "zzz duplicte"],  # duplicates far in alphabet
        })

        # Small window might miss distant duplicates
        small_window = fr.dedupe_snm(df, columns=["name"], window_size=2)
        large_window = fr.dedupe_snm(df, columns=["name"], window_size=10)

        # Both should return valid DataFrames with required columns
        assert "_group_id" in small_window.columns, "small_window missing _group_id column"
        assert "_is_canonical" in small_window.columns, "small_window missing _is_canonical column"
        assert "_group_id" in large_window.columns, "large_window missing _group_id column"
        assert "_is_canonical" in large_window.columns, "large_window missing _is_canonical column"

        # Both should have same number of rows as input
        assert len(small_window) == len(df), "small_window row count mismatch"
        assert len(large_window) == len(df), "large_window row count mismatch"

        # Both should find the zzz duplicates since they're adjacent after sorting
        small_grouped = small_window.filter(pl.col("_group_id").is_not_null())
        large_grouped = large_window.filter(pl.col("_group_id").is_not_null())
        assert len(small_grouped) >= 2, "Expected at least 2 grouped rows for zzz duplicates"
        assert len(large_grouped) >= 2, "Expected at least 2 grouped rows for zzz duplicates"

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pl.DataFrame({"name": []})
        result = fr.dedupe_snm(df, columns=["name"])

        assert len(result) == 0
        assert "_group_id" in result.columns
        assert "_is_canonical" in result.columns


class TestMatchRecordsBatch:
    """Tests for match_records_batch function."""

    def test_basic_matching(self):
        """Test basic record matching."""
        queries = pl.DataFrame({
            "name": ["Jon Smith", "Jane Do"],
            "city": ["New York", "LA"],
        })
        targets = pl.DataFrame({
            "name": ["John Smith", "Jane Doe", "Bob Wilson"],
            "city": ["New York", "Los Angeles", "Chicago"],
        })

        result = fr.match_records_batch(
            queries,
            targets,
            columns=["name", "city"],
            min_similarity=0.5,
        )

        assert "query_idx" in result.columns
        assert "target_idx" in result.columns
        assert "score" in result.columns

    def test_with_weights(self):
        """Test that weights affect scoring."""
        queries = pl.DataFrame({"name": ["Jon Smith"], "email": ["jon@test.com"]})
        targets = pl.DataFrame({
            "name": ["John Smith", "Jon Jones"],
            "email": ["john@test.com", "jon@test.com"],
        })

        # Heavy weight on email should prefer exact email match
        result = fr.match_records_batch(
            queries,
            targets,
            columns=["name", "email"],
            weights={"name": 1.0, "email": 10.0},
            min_similarity=0.5,
        )

        assert len(result) > 0, "Expected at least one match"
        # With 10x weight on email, exact email match "jon@test.com" should win
        # Target index 1 has "jon@test.com" which matches query exactly
        assert result["target_idx"][0] == 1, \
            f"Expected target_idx 1 (Jon Jones with exact email), got {result['target_idx'][0]}"

    def test_per_column_algorithms(self):
        """Test per-column algorithm specification."""
        queries = pl.DataFrame({"name": ["Jon"], "code": ["ABC123"]})
        targets = pl.DataFrame({"name": ["John"], "code": ["ABC124"]})

        result = fr.match_records_batch(
            queries,
            targets,
            columns=["name", "code"],
            algorithms={"name": "jaro_winkler", "code": "levenshtein"},
            min_similarity=0.5,
        )

        # Should find at least one match since "Jon" is similar to "John" and "ABC123" to "ABC124"
        assert len(result) >= 1, "Expected at least one match with per-column algorithms"
        assert "query_idx" in result.columns
        assert "target_idx" in result.columns
        assert "score" in result.columns


class TestFindSimilarPairs:
    """Tests for find_similar_pairs function."""

    def test_snm_method(self):
        """Test SNM method finds similar pairs."""
        df = pl.DataFrame({
            "name": ["John Smith", "Jon Smith", "Jane Doe", "John Smyth"],
        })

        result = fr.find_similar_pairs(
            df,
            columns=["name"],
            method="snm",
            min_similarity=0.8,
        )

        assert "idx_a" in result.columns
        assert "idx_b" in result.columns
        assert "score" in result.columns

        # Should find at least some pairs among the Johns
        assert len(result) >= 1

    def test_full_method(self):
        """Test full pairwise comparison method."""
        df = pl.DataFrame({
            "name": ["hello", "hallo", "world"],
        })

        result = fr.find_similar_pairs(
            df,
            columns=["name"],
            method="full",
            min_similarity=0.8,
        )

        # hello and hallo should be similar
        assert len(result) >= 1

    def test_no_duplicate_pairs(self):
        """Test that pairs are not duplicated (i,j) and (j,i)."""
        df = pl.DataFrame({
            "name": ["hello", "hallo"],
        })

        result = fr.find_similar_pairs(
            df,
            columns=["name"],
            method="full",
            min_similarity=0.5,
        )

        # Should only have one pair, not two
        if len(result) > 0:
            pairs = set()
            for row in result.iter_rows(named=True):
                pair = (min(row["idx_a"], row["idx_b"]), max(row["idx_a"], row["idx_b"]))
                assert pair not in pairs, "Duplicate pair found"
                pairs.add(pair)


class TestIntegration:
    """Integration tests for the new API."""

    def test_full_workflow(self):
        """Test a complete deduplication workflow."""
        # Create sample data with duplicates
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["John Smith", "Jon Smith", "Jane Doe", "John Smyth", "Bob Wilson"],
            "email": ["john@test.com", "jon@test.com", "jane@test.com", "john@test.com", "bob@test.com"],
        })

        # Step 1: Find duplicates using SNM
        deduped = fr.dedupe_snm(
            df,
            columns=["name", "email"],
            min_similarity=0.7,
            window_size=10,
        )

        # Step 2: Get unique records
        unique = deduped.filter(pl.col("_is_canonical"))

        # Should have fewer unique records than original
        assert len(unique) <= len(df)

    def test_batch_vs_rowwise_consistency(self):
        """Test that batch operations are consistent with row-wise."""
        left = pl.Series(["hello", "world", "test"])
        right = pl.Series(["hallo", "word", "testing"])

        batch_result = batch_similarity_polars(left, right, algorithm="jaro_winkler")

        # Compare with row-wise computation
        for i in range(len(left)):
            expected = fr.jaro_winkler_similarity(left[i], right[i])
            assert abs(batch_result[i] - expected) < 0.001


class TestPerformance:
    """Performance-related tests (not benchmarks, just sanity checks)."""

    def test_handles_medium_dataset(self):
        """Test that the API handles medium-sized datasets."""
        n = 1000
        df = pl.DataFrame({
            "name": [f"Name {i % 100}" for i in range(n)],
        })

        result = fr.dedupe_snm(
            df,
            columns=["name"],
            min_similarity=0.9,
            window_size=20,
        )

        assert len(result) == n

    def test_batch_search_performance(self):
        """Test that batch operations complete in reasonable time."""
        queries = pl.Series([f"query {i}" for i in range(100)])
        targets = [f"target {i}" for i in range(100)]

        result = fr.batch_best_match(queries, targets, min_similarity=0.0)

        assert len(result) == 100
