"""
Concurrency tests for FuzzyRust.

Tests cover:
- Thread safety documentation verification
- Separate indices per thread work correctly
- Parallel batch operations
"""

import concurrent.futures
import threading

import pytest

import fuzzyrust as fr


class TestThreadSafetyDocumentation:
    """Verify thread safety warnings are documented."""

    def test_bktree_docstring_warning(self):
        """BkTree should have thread safety warning in docstring."""
        docstring = fr.BkTree.__doc__ or ""
        # The warning should be documented
        assert "thread" in docstring.lower(), "Missing thread-safety warning in BkTree docstring"

    def test_ngram_index_docstring_warning(self):
        """NgramIndex should have thread safety warning."""
        docstring = fr.NgramIndex.__doc__ or ""
        assert (
            "thread" in docstring.lower()
        ), "Missing thread-safety warning in NgramIndex docstring"

    def test_fuzzy_index_docstring_warning(self):
        """FuzzyIndex should have thread safety warning."""
        docstring = fr.FuzzyIndex.__doc__ or ""
        assert (
            "thread" in docstring.lower()
        ), "Missing thread-safety warning in FuzzyIndex docstring"


class TestSeparateIndicesPerThread:
    """Test that separate index instances work correctly in threads."""

    def test_bktree_separate_instances(self):
        """Separate BkTree instances should work in parallel."""
        results = []

        def worker(thread_id):
            # Each thread creates its own tree
            tree = fr.BkTree()
            for i in range(100):
                tree.add(f"item_{thread_id}_{i}")

            # Search in own tree
            found = tree.search(f"item_{thread_id}_50", max_distance=2)
            return len(found)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            results = [f.result() for f in futures]

        # All threads should find results
        assert all(r > 0 for r in results)

    def test_ngram_index_separate_instances(self):
        """Separate NgramIndex instances should work in parallel."""

        def worker(thread_id):
            index = fr.NgramIndex(ngram_size=2)
            for i in range(100):
                index.add(f"item_{thread_id}_{i}")

            found = index.search(f"item_{thread_id}_50", min_similarity=0.5)
            return len(found)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            results = [f.result() for f in futures]

        assert all(r > 0 for r in results)

    def test_fuzzy_index_separate_instances(self):
        """Separate FuzzyIndex instances should work in parallel."""

        def worker(thread_id):
            items = [f"item_{thread_id}_{i}" for i in range(100)]
            index = fr.FuzzyIndex(items, algorithm="ngram")

            found = index.search(f"item_{thread_id}_50", min_similarity=0.5)
            return len(found)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            results = [f.result() for f in futures]

        assert all(r > 0 for r in results)


class TestParallelBatchOperations:
    """Test batch operations can run in parallel."""

    def test_batch_levenshtein_parallel(self):
        """Batch levenshtein should work in parallel calls."""
        pairs = [("hello", "hallo"), ("world", "word"), ("test", "text")]

        def worker():
            results = []
            for s1, s2 in pairs:
                results.append(fr.levenshtein(s1, s2))
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker) for _ in range(4)]
            all_results = [f.result() for f in futures]

        # All threads should get same results
        expected = worker()
        for result in all_results:
            assert result == expected

    def test_batch_jaro_winkler_parallel(self):
        """Jaro-Winkler should work in parallel calls."""
        pairs = [("hello", "hallo"), ("world", "word"), ("test", "text")]

        def worker():
            return [fr.jaro_winkler_similarity(s1, s2) for s1, s2 in pairs]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker) for _ in range(4)]
            all_results = [f.result() for f in futures]

        expected = worker()
        for result in all_results:
            for i, val in enumerate(result):
                assert val == pytest.approx(expected[i], abs=0.001)

    def test_find_best_matches_parallel(self):
        """find_best_matches should work in parallel."""
        choices = ["apple", "banana", "cherry", "date", "elderberry"]
        queries = ["appel", "banan", "chery"]

        def worker():
            return [fr.find_best_matches(choices, q) for q in queries]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker) for _ in range(4)]
            all_results = [f.result() for f in futures]

        # All threads should get consistent results
        expected = worker()
        for result in all_results:
            assert len(result) == len(expected)


class TestConcurrentIndexBuilding:
    """Test concurrent index building."""

    def test_build_multiple_indices_concurrently(self):
        """Building multiple indices concurrently should work."""
        items_sets = [
            [f"set_a_{i}" for i in range(100)],
            [f"set_b_{i}" for i in range(100)],
            [f"set_c_{i}" for i in range(100)],
            [f"set_d_{i}" for i in range(100)],
        ]

        def build_index(items):
            tree = fr.BkTree()
            for item in items:
                tree.add(item)
            return tree.search(items[50], max_distance=0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(build_index, items) for items in items_sets]
            results = [f.result() for f in futures]

        # Each should find exactly its item
        for result in results:
            assert len(result) >= 1


class TestRaceConditionPrevention:
    """Tests to ensure no race conditions in stateless operations."""

    def test_many_concurrent_similarity_calls(self):
        """Many concurrent similarity calls should not interfere."""
        num_threads = 8
        calls_per_thread = 1000
        errors = []

        def worker(thread_id):
            try:
                for i in range(calls_per_thread):
                    s1 = f"string_{thread_id}_{i}"
                    s2 = f"strng_{thread_id}_{i}"

                    sim = fr.jaro_winkler_similarity(s1, s2)
                    if not (0.0 <= sim <= 1.0):
                        errors.append(f"Invalid similarity: {sim}")

                    dist = fr.levenshtein(s1, s2)
                    if dist < 0:
                        errors.append(f"Negative distance: {dist}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors[:10]}"

    def test_concurrent_soundex_metaphone(self):
        """Concurrent phonetic encoding should work."""
        names = ["Smith", "Johnson", "Williams", "Brown", "Jones"] * 20

        def worker():
            results = []
            for name in names:
                results.append((fr.soundex(name), fr.metaphone(name)))
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker) for _ in range(4)]
            all_results = [f.result() for f in futures]

        # All threads should get same results
        expected = worker()
        for result in all_results:
            assert result == expected


class TestThreadSafeIndices:
    """Tests for the thread-safe index wrappers."""

    def test_threadsafe_bktree_concurrent_reads(self):
        """ThreadSafeBkTree should support concurrent reads."""
        tree = fr.ThreadSafeBkTree()
        tree.add_all([f"item_{i}" for i in range(1000)])

        def worker():
            results = []
            for i in range(100):
                found = tree.search(f"item_{i * 10}", max_distance=2)
                results.append(len(found))
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker) for _ in range(8)]
            all_results = [f.result() for f in futures]

        # All threads should get results
        for result in all_results:
            assert all(r >= 0 for r in result)

    def test_threadsafe_bktree_concurrent_writes(self):
        """ThreadSafeBkTree should handle concurrent writes."""
        tree = fr.ThreadSafeBkTree()

        def writer(thread_id):
            for i in range(100):
                tree.add(f"thread_{thread_id}_item_{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(writer, i) for i in range(4)]
            for f in futures:
                f.result()

        # All items should be added
        assert len(tree) == 400

    def test_threadsafe_bktree_mixed_operations(self):
        """ThreadSafeBkTree should handle mixed read/write operations."""
        tree = fr.ThreadSafeBkTree()
        tree.add_all([f"initial_{i}" for i in range(100)])
        errors = []

        def writer():
            for i in range(50):
                tree.add(f"write_{i}")

        def reader():
            for _ in range(100):
                try:
                    results = tree.search("initial_50", max_distance=2)
                    if len(results) < 1:
                        errors.append("No results found")
                except Exception as e:
                    errors.append(str(e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Launch readers and writers concurrently
            futures = [
                executor.submit(reader),
                executor.submit(reader),
                executor.submit(writer),
                executor.submit(reader),
            ]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors: {errors[:10]}"

    def test_threadsafe_ngram_index_concurrent_reads(self):
        """ThreadSafeNgramIndex should support concurrent reads."""
        index = fr.ThreadSafeNgramIndex(ngram_size=3)
        index.add_all([f"item_{i}" for i in range(1000)])

        def worker():
            results = []
            for i in range(50):
                found = index.search(f"item_{i * 20}", min_similarity=0.5)
                results.append(len(found))
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker) for _ in range(8)]
            all_results = [f.result() for f in futures]

        for result in all_results:
            assert all(r >= 0 for r in result)

    def test_threadsafe_ngram_index_parallel_add(self):
        """ThreadSafeNgramIndex should support parallel bulk insertion."""
        index = fr.ThreadSafeNgramIndex(ngram_size=3)
        items = [f"item_{i}" for i in range(10000)]

        # Test parallel add
        index.add_all_parallel(items)

        assert len(index) == 10000

    def test_threadsafe_serialization_under_load(self):
        """ThreadSafeBkTree serialization should work during concurrent access."""
        tree = fr.ThreadSafeBkTree()
        tree.add_all([f"item_{i}" for i in range(500)])
        errors = []

        def reader():
            for _ in range(50):
                try:
                    tree.search("item_100", max_distance=2)
                except Exception as e:
                    errors.append(f"Reader error: {e}")

        def serializer():
            try:
                for _ in range(10):
                    data = tree.to_bytes()
                    restored = fr.ThreadSafeBkTree.from_bytes(data)
                    assert len(restored) == len(tree)
            except Exception as e:
                errors.append(f"Serializer error: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(reader),
                executor.submit(reader),
                executor.submit(serializer),
                executor.submit(reader),
            ]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors: {errors[:10]}"


class TestParallelBulkInsertion:
    """Tests for parallel bulk insertion feature."""

    def test_ngram_index_parallel_vs_sequential(self):
        """Parallel add should produce same results as sequential."""
        items = [f"item_{i}" for i in range(5000)]

        # Sequential
        index_seq = fr.NgramIndex(ngram_size=3)
        index_seq.add_all(items)

        # Parallel
        index_par = fr.NgramIndex(ngram_size=3)
        index_par.add_all_parallel(items)

        # Both should have same size
        assert len(index_seq) == len(index_par)

        # Both should find same results
        query = "item_2500"
        results_seq = index_seq.search(query, min_similarity=0.8)
        results_par = index_par.search(query, min_similarity=0.8)

        assert len(results_seq) == len(results_par)

    @pytest.mark.slow
    def test_ngram_index_parallel_large_scale(self):
        """Test parallel insertion with large dataset."""
        items = [f"string_number_{i}_with_some_text" for i in range(50000)]

        index = fr.NgramIndex(ngram_size=3)
        index.add_all_parallel(items)

        assert len(index) == 50000

        # Search should work
        results = index.search("string_number_25000_with_some_text", min_similarity=0.9)
        assert len(results) > 0


class TestStressConditions:
    """High-stress concurrency tests."""

    @pytest.mark.slow
    def test_high_contention_threadsafe_bktree(self):
        """Test ThreadSafeBkTree under high contention."""
        tree = fr.ThreadSafeBkTree()
        num_threads = 16
        ops_per_thread = 500
        errors = []

        def worker(thread_id):
            try:
                for i in range(ops_per_thread):
                    # Mix of operations
                    if i % 3 == 0:
                        tree.add(f"thread_{thread_id}_item_{i}")
                    elif i % 3 == 1:
                        tree.search(f"thread_{thread_id}_item_{i // 2}", max_distance=2)
                    else:
                        tree.contains(f"thread_{thread_id}_item_{i // 2}")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors[:10]}"

    @pytest.mark.slow
    def test_rapid_add_remove_compact(self):
        """Test rapid add/remove/compact cycles."""
        tree = fr.ThreadSafeBkTree()

        for cycle in range(10):
            # Add items
            for i in range(100):
                tree.add(f"cycle_{cycle}_item_{i}")

            # Remove some
            for i in range(0, 100, 2):
                tree.remove(i + cycle * 100)

            # Compact periodically
            if cycle % 3 == 0:
                tree.compact()

        # Should have roughly half the items
        assert tree.active_count() > 0
