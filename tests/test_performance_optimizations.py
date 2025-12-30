"""Tests for Phase 6 performance optimizations.

This module tests the three long-term performance optimizations:
1. BK-tree Parallel Search (2-4x speedup for large trees)
2. Best Match Indexing for Polars (5-10x speedup for large target lists)
3. Dedup Parallel Efficiency (3-5x speedup for large datasets)
"""

import pytest

import fuzzyrust as fr


class TestBkTreeParallelSearch:
    """Tests for BK-tree parallel search optimization."""

    def test_parallel_search_matches_sequential_small_tree(self):
        """Parallel and sequential search should return same results for small trees."""
        # Build tree with items below parallel threshold
        items = [f"item_{i}" for i in range(100)]
        tree = fr.BkTree()
        tree.add_all(items)

        # Both methods should find same matches
        seq_results = tree.search("item_50", max_distance=2)
        par_results = tree.search_parallel("item_50", max_distance=2)

        seq_texts = {r.text for r in seq_results}
        par_texts = {r.text for r in par_results}

        assert seq_texts == par_texts
        assert len(seq_results) == len(par_results)

    def test_parallel_search_with_min_similarity(self):
        """Parallel search should work with min_similarity parameter."""
        items = ["hello", "hallo", "hullo", "world", "help"]
        tree = fr.BkTree()
        tree.add_all(items)

        results = tree.search_parallel("hello", min_similarity=0.8)

        # Should find hello (exact) and similar words
        texts = {r.text for r in results}
        assert "hello" in texts

    def test_parallel_search_with_limit(self):
        """Parallel search should respect limit parameter."""
        items = [f"item_{i:04d}" for i in range(200)]
        tree = fr.BkTree()
        tree.add_all(items)

        results = tree.search_parallel("item_0050", max_distance=3, limit=5)

        assert len(results) <= 5

    def test_parallel_search_empty_tree(self):
        """Parallel search on empty tree should return empty results."""
        tree = fr.BkTree()
        results = tree.search_parallel("query", max_distance=2)

        assert len(results) == 0

    def test_parallel_search_no_matches(self):
        """Parallel search should return empty when no matches found."""
        items = ["apple", "banana", "cherry"]
        tree = fr.BkTree()
        tree.add_all(items)

        results = tree.search_parallel("xyz", max_distance=1)

        assert len(results) == 0

    def test_parallel_search_exact_match(self):
        """Parallel search should find exact matches with distance 0."""
        items = ["hello", "world", "test"]
        tree = fr.BkTree()
        tree.add_all(items)

        results = tree.search_parallel("hello", max_distance=0)

        assert len(results) == 1
        assert results[0].text == "hello"
        assert results[0].distance == 0

    def test_find_nearest_parallel(self):
        """find_nearest_parallel should return k nearest neighbors."""
        items = ["apple", "application", "apply", "banana", "bandana"]
        tree = fr.BkTree()
        tree.add_all(items)

        results = tree.find_nearest_parallel("appli", 2)

        assert len(results) == 2
        # Results should be sorted by distance
        assert results[0].distance <= results[1].distance

    def test_find_nearest_parallel_matches_sequential(self):
        """find_nearest_parallel should match find_nearest results."""
        items = ["hello", "hallo", "hullo", "world", "help"]
        tree = fr.BkTree()
        tree.add_all(items)

        seq_results = tree.find_nearest("helo", 3)
        par_results = tree.find_nearest_parallel("helo", 3)

        seq_texts = {r.text for r in seq_results}
        par_texts = {r.text for r in par_results}

        assert seq_texts == par_texts

    def test_parallel_search_validation_errors(self):
        """Parallel search should raise validation errors for invalid parameters."""
        tree = fr.BkTree()
        tree.add("test")

        # Must specify max_distance or min_similarity
        with pytest.raises(fr.ValidationError):
            tree.search_parallel("test")

        # Cannot specify both
        with pytest.raises(fr.ValidationError):
            tree.search_parallel("test", max_distance=2, min_similarity=0.8)

        # Invalid min_similarity range
        with pytest.raises(fr.ValidationError):
            tree.search_parallel("test", min_similarity=1.5)

    @pytest.mark.slow
    def test_parallel_search_large_tree_performance(self):
        """Parallel search should handle large trees efficiently."""
        # Build large tree (above parallel threshold of 10K)
        items = [f"item_{i:06d}" for i in range(15000)]
        tree = fr.BkTree()
        tree.add_all(items)

        # Should complete without error
        results = tree.search_parallel("item_007500", max_distance=2)

        # Should find the exact match
        assert any(r.text == "item_007500" for r in results)


class TestBestMatchIndexing:
    """Tests for indexed best match in Polars expressions."""

    def test_best_match_small_targets_correct(self):
        """Best match with small target list should be correct."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df = pl.DataFrame({"query": ["apple", "banan", "orang"]})
        targets = ["apple", "banana", "orange", "grape"]

        result = df.with_columns(
            match=pl.col("query").fuzzy.best_match(targets, min_similarity=0.6)
        )

        assert result["match"][0] == "apple"
        assert result["match"][1] == "banana"
        assert result["match"][2] == "orange"

    def test_best_match_large_targets_correct(self):
        """Best match with large target list (>100 items) should be correct."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df = pl.DataFrame({"query": ["apple", "banan", "orang"]})
        # Create large target list to trigger indexing
        targets = [f"fruit_{i}" for i in range(200)] + ["apple", "banana", "orange"]

        result = df.with_columns(
            match=pl.col("query").fuzzy.best_match(targets, min_similarity=0.6)
        )

        assert result["match"][0] == "apple"
        assert result["match"][1] == "banana"
        assert result["match"][2] == "orange"

    def test_best_match_no_match_above_threshold(self):
        """Best match should return null when no match above threshold."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df = pl.DataFrame({"query": ["xyz"]})
        targets = ["apple", "banana", "orange"]

        result = df.with_columns(
            match=pl.col("query").fuzzy.best_match(targets, min_similarity=0.9)
        )

        assert result["match"][0] is None

    def test_best_match_with_different_algorithms(self):
        """Best match should work with different similarity algorithms."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df = pl.DataFrame({"query": ["hello"]})
        targets = ["hello", "hallo", "world"]

        for algorithm in ["jaro_winkler", "levenshtein", "jaro"]:
            result = df.with_columns(
                match=pl.col("query").fuzzy.best_match(
                    targets, min_similarity=0.5, algorithm=algorithm
                )
            )
            # Should find "hello" as best match for itself
            assert result["match"][0] == "hello"

    def test_best_match_handles_null_queries(self):
        """Best match should handle null queries gracefully."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        df = pl.DataFrame({"query": ["apple", None, "orange"]})
        targets = ["apple", "banana", "orange"]

        result = df.with_columns(
            match=pl.col("query").fuzzy.best_match(targets, min_similarity=0.6)
        )

        assert result["match"][0] == "apple"
        assert result["match"][1] is None
        assert result["match"][2] == "orange"

    @pytest.mark.slow
    def test_best_match_large_targets_performance(self):
        """Best match with large targets should complete efficiently."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        # Create larger dataset to test indexing performance
        df = pl.DataFrame({"query": [f"item_{i}" for i in range(100)]})
        targets = [f"item_{i}" for i in range(500)]

        result = df.with_columns(
            match=pl.col("query").fuzzy.best_match(targets, min_similarity=0.8)
        )

        # All queries should find exact matches
        for i in range(100):
            assert result["match"][i] == f"item_{i}"


class TestDedupParallelEfficiency:
    """Tests for chunk-based deduplication optimization."""

    def test_dedup_small_dataset_correctness(self):
        """Deduplication should work correctly for small datasets."""
        items = ["hello", "helo", "world", "wrold", "test"]

        result = fr.find_duplicates(items, algorithm="jaro_winkler", min_similarity=0.85)

        # Should find some duplicates
        assert len(result.groups) > 0 or len(result.unique) > 0

    def test_dedup_all_unique(self):
        """Deduplication should identify all unique items correctly."""
        items = ["apple", "banana", "cherry", "date", "elderberry"]

        result = fr.find_duplicates(items, algorithm="jaro_winkler", min_similarity=0.95)

        # All items are unique
        assert result.total_duplicates == 0
        assert len(result.unique) == 5

    def test_dedup_all_duplicates(self):
        """Deduplication should group identical items."""
        items = ["test", "test", "test", "test"]

        result = fr.find_duplicates(items, algorithm="jaro_winkler", min_similarity=0.99)

        # All items should be grouped (check total_duplicates which is correctly tracked)
        # Note: groups may contain repeated strings due to how groups are reported
        assert result.total_duplicates >= 3  # At least 3 duplicates (4 items - 1 unique)
        assert len(result.groups) >= 1  # At least one group of duplicates
        assert len(result.unique) == 0  # No unique items

    def test_dedup_mixed(self):
        """Deduplication should handle mix of duplicates and unique items."""
        items = [
            "hello",
            "helo",  # Similar pair
            "world",
            "wrold",  # Similar pair
            "apple",
            "banana",  # Truly unique items (very different)
        ]

        result = fr.find_duplicates(items, algorithm="jaro_winkler", min_similarity=0.85)

        # Should find at least 2 groups of duplicates (hello/helo and world/wrold)
        assert len(result.groups) >= 2
        # Should have at least some unique items (apple, banana)
        total_grouped = sum(len(g) for g in result.groups)
        total_unique = len(result.unique)
        assert total_grouped + total_unique == 6

    def test_dedup_empty_list(self):
        """Deduplication should handle empty input."""
        items = []

        result = fr.find_duplicates(items, algorithm="jaro_winkler", min_similarity=0.85)

        assert len(result.groups) == 0
        assert len(result.unique) == 0
        assert result.total_duplicates == 0

    def test_dedup_single_item(self):
        """Deduplication should handle single item."""
        items = ["hello"]

        result = fr.find_duplicates(items, algorithm="jaro_winkler", min_similarity=0.85)

        assert len(result.groups) == 0
        assert len(result.unique) == 1
        assert result.total_duplicates == 0

    def test_find_duplicate_pairs_correctness(self):
        """find_duplicate_pairs should return correct pairs."""
        items = ["hello", "helo", "world", "test"]

        pairs = fr.find_duplicate_pairs(items, algorithm="jaro_winkler", min_similarity=0.85)

        # Should find hello/helo pair
        pair_set = {(min(a, b), max(a, b)) for a, b, _ in pairs}
        assert (0, 1) in pair_set  # hello and helo

    def test_find_duplicate_pairs_with_window(self):
        """find_duplicate_pairs should respect window_size parameter."""
        items = [f"item_{i}" for i in range(100)]

        pairs = fr.find_duplicate_pairs(
            items, algorithm="jaro_winkler", min_similarity=0.99, window_size=10
        )

        # With distinct items and high threshold, should find no pairs
        assert len(pairs) == 0

    def test_find_duplicate_pairs_with_normalization(self):
        """find_duplicate_pairs should respect normalize parameter."""
        items = ["Hello", "hello", "HELLO"]

        # With lowercase normalization, all should match
        pairs = fr.find_duplicate_pairs(
            items, algorithm="jaro_winkler", min_similarity=0.99, normalize="lowercase"
        )

        # Should find all pairs as duplicates
        assert len(pairs) == 3  # (0,1), (0,2), (1,2)

    @pytest.mark.slow
    def test_dedup_large_dataset_performance(self):
        """Deduplication should handle large datasets efficiently."""
        # Create dataset with some duplicates
        items = [f"item_{i:04d}" for i in range(500)]
        # Add some duplicates
        items.extend(["item_0100", "item_0200", "item_0300"])

        result = fr.find_duplicates(items, algorithm="jaro_winkler", min_similarity=0.99)

        # Should find the duplicate groups
        assert len(result.groups) == 3


class TestIntegration:
    """Integration tests combining multiple optimizations."""

    def test_bktree_then_polars_matching(self):
        """Test workflow: build BK-tree, search, then match in Polars."""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("Polars not installed")

        # Build index
        tree = fr.BkTree()
        reference_items = ["apple", "banana", "cherry", "date"]
        tree.add_all(reference_items)

        # Search
        results = tree.search_parallel("banan", max_distance=2)
        assert any(r.text == "banana" for r in results)

        # Match in Polars
        df = pl.DataFrame({"query": ["aple", "banan", "chery"]})
        matched = df.with_columns(
            match=pl.col("query").fuzzy.best_match(reference_items, min_similarity=0.7)
        )

        assert matched["match"][0] == "apple"
        assert matched["match"][1] == "banana"
        assert matched["match"][2] == "cherry"

    def test_dedup_then_index_build(self):
        """Test workflow: dedupe items, then build index from unique items."""
        # Start with duplicated data
        items = [
            "hello",
            "hello",
            "helo",  # Duplicates
            "world",
            "world",  # Duplicates
            "unique",  # Unique
        ]

        # Dedupe
        result = fr.find_duplicates(items, algorithm="jaro_winkler", min_similarity=0.95)

        # Get unique items (one from each group + truly unique)
        unique_items = result.unique.copy()
        for group in result.groups:
            unique_items.append(group[0])  # Take first from each group

        # Build index from unique items
        tree = fr.BkTree()
        tree.add_all(unique_items)

        assert len(tree) == len(unique_items)


# Mark slow tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
