"""Memory leak tests for fuzzyrust.

These tests verify that repeated operations don't cause memory leaks.
Uses tracemalloc for memory tracking.
"""

import gc
import tracemalloc

import pytest


class TestBkTreeMemory:
    """Memory tests for BkTree operations."""

    @pytest.mark.slow
    def test_bktree_repeated_creation_no_leak(self):
        """Verify that creating and destroying BkTrees doesn't leak memory."""
        import fuzzyrust as fr

        gc.collect()
        tracemalloc.start()

        # Create and destroy many trees
        for _ in range(50):
            tree = fr.BkTree()
            for i in range(500):
                tree.add(f"item_{i}")
            del tree
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak should be reasonable (< 50MB for this workload)
        assert peak < 50_000_000, f"Peak memory {peak / 1_000_000:.1f}MB exceeds 50MB limit"

    @pytest.mark.slow
    def test_bktree_repeated_search_no_leak(self):
        """Verify that repeated searches don't leak memory."""
        import fuzzyrust as fr

        tree = fr.BkTree()
        for i in range(1000):
            tree.add(f"item_{i}")

        gc.collect()
        tracemalloc.start()

        # Perform many searches
        for _ in range(1000):
            results = tree.search("item_500", 2)
            del results

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory growth should be minimal for search operations
        assert peak < 10_000_000, f"Peak memory {peak / 1_000_000:.1f}MB exceeds 10MB limit"

    @pytest.mark.slow
    def test_bktree_deletion_no_leak(self):
        """Verify that deletion operations don't leak memory."""
        import fuzzyrust as fr

        gc.collect()
        tracemalloc.start()

        for _ in range(20):
            tree = fr.BkTree()
            for i in range(500):
                tree.add(f"item_{i}")

            # Delete half the items
            for i in range(0, 500, 2):
                tree.remove(i)

            # Compact to rebuild
            tree.compact()
            del tree
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 30_000_000, f"Peak memory {peak / 1_000_000:.1f}MB exceeds 30MB limit"

    @pytest.mark.slow
    def test_bktree_serialization_no_leak(self):
        """Verify that serialization/deserialization cycles don't leak memory."""
        import fuzzyrust as fr

        tree = fr.BkTree()
        for i in range(500):
            tree.add(f"item_{i}")

        gc.collect()
        tracemalloc.start()

        # Serialize and deserialize many times
        for _ in range(100):
            data = tree.to_bytes()
            restored = fr.BkTree.from_bytes(data)
            del data
            del restored
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 20_000_000, f"Peak memory {peak / 1_000_000:.1f}MB exceeds 20MB limit"


class TestNgramIndexMemory:
    """Memory tests for NgramIndex operations."""

    @pytest.mark.slow
    def test_ngram_index_repeated_creation_no_leak(self):
        """Verify that creating and destroying NgramIndex doesn't leak memory."""
        import fuzzyrust as fr

        gc.collect()
        tracemalloc.start()

        for _ in range(50):
            index = fr.NgramIndex(ngram_size=3)
            for i in range(500):
                index.add(f"item_{i}")
            del index
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 50_000_000, f"Peak memory {peak / 1_000_000:.1f}MB exceeds 50MB limit"

    @pytest.mark.slow
    def test_ngram_index_repeated_search_no_leak(self):
        """Verify that repeated searches don't leak memory."""
        import fuzzyrust as fr

        index = fr.NgramIndex(ngram_size=3)
        for i in range(1000):
            index.add(f"item_{i}")

        gc.collect()
        tracemalloc.start()

        for _ in range(1000):
            results = index.search("item_500", min_similarity=0.5)
            del results

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 10_000_000, f"Peak memory {peak / 1_000_000:.1f}MB exceeds 10MB limit"

    @pytest.mark.slow
    def test_ngram_index_serialization_no_leak(self):
        """Verify that serialization/deserialization cycles don't leak memory."""
        import fuzzyrust as fr

        index = fr.NgramIndex(ngram_size=3)
        for i in range(500):
            index.add(f"item_{i}")

        gc.collect()
        tracemalloc.start()

        for _ in range(100):
            data = index.to_bytes()
            restored = fr.NgramIndex.from_bytes(data)
            del data
            del restored
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 20_000_000, f"Peak memory {peak / 1_000_000:.1f}MB exceeds 20MB limit"


class TestHybridIndexMemory:
    """Memory tests for HybridIndex operations."""

    @pytest.mark.slow
    def test_hybrid_index_repeated_creation_no_leak(self):
        """Verify that creating and destroying HybridIndex doesn't leak memory."""
        import fuzzyrust as fr

        gc.collect()
        tracemalloc.start()

        for _ in range(30):
            index = fr.HybridIndex()
            for i in range(300):
                index.add(f"item_{i}")
            del index
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 100_000_000, f"Peak memory {peak / 1_000_000:.1f}MB exceeds 100MB limit"


class TestBatchOperationsMemory:
    """Memory tests for batch operations."""

    @pytest.mark.slow
    def test_batch_levenshtein_no_leak(self):
        """Verify that repeated batch operations don't leak memory."""
        import fuzzyrust as fr

        strings = [f"string_{i}" for i in range(1000)]

        gc.collect()
        tracemalloc.start()

        for _ in range(100):
            results = fr.batch_levenshtein(strings, "query_string")
            del results
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 20_000_000, f"Peak memory {peak / 1_000_000:.1f}MB exceeds 20MB limit"

    @pytest.mark.slow
    def test_find_best_matches_no_leak(self):
        """Verify that find_best_matches doesn't leak memory."""
        import fuzzyrust as fr

        strings = [f"string_{i}" for i in range(1000)]

        gc.collect()
        tracemalloc.start()

        for _ in range(100):
            results = fr.find_best_matches(strings, "query_string", limit=10)
            del results
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 20_000_000, f"Peak memory {peak / 1_000_000:.1f}MB exceeds 20MB limit"


class TestSearchResultMemory:
    """Memory tests for SearchResult handling."""

    @pytest.mark.slow
    def test_search_results_with_data_no_leak(self):
        """Verify that search results with user data don't leak memory."""
        import fuzzyrust as fr

        tree = fr.BkTree()
        for i in range(500):
            # add_with_data takes numeric data (u64)
            tree.add_with_data(f"item_{i}", i)

        gc.collect()
        tracemalloc.start()

        for _ in range(500):
            results = tree.search("item_100", 2)
            for r in results:
                _ = r.text
                _ = r.data
            del results
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 30_000_000, f"Peak memory {peak / 1_000_000:.1f}MB exceeds 30MB limit"


class TestStringMemory:
    """Memory tests for string handling."""

    @pytest.mark.slow
    def test_unicode_strings_no_leak(self):
        """Verify that Unicode string operations don't leak memory."""
        import fuzzyrust as fr

        gc.collect()
        tracemalloc.start()

        for _ in range(1000):
            _ = fr.levenshtein("日本語テキスト", "日本語テスト")
            _ = fr.jaro_winkler_similarity("émile zola", "emile zolà")
            _ = fr.ngram_similarity("München", "Muenchen", 2)
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 5_000_000, f"Peak memory {peak / 1_000_000:.1f}MB exceeds 5MB limit"

    @pytest.mark.slow
    def test_long_strings_no_leak(self):
        """Verify that long string operations don't leak memory."""
        import fuzzyrust as fr

        long_a = "a" * 1000
        long_b = "b" * 1000

        gc.collect()
        tracemalloc.start()

        for _ in range(100):
            _ = fr.levenshtein(long_a, long_b)
            _ = fr.jaro_winkler_similarity(long_a, long_b)
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 10_000_000, f"Peak memory {peak / 1_000_000:.1f}MB exceeds 10MB limit"
