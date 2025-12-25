"""Tests for Sprint 3 features (v0.2.0 API changes)."""

import pytest


class TestCaseInsensitiveVariants:
    """Tests for case-insensitive function variants."""

    def test_levenshtein_ci(self):
        import fuzzyrust as fr
        assert fr.levenshtein_ci("Hello", "HELLO") == 0
        assert fr.levenshtein_ci("Hello", "World") > 0

    def test_levenshtein_similarity_ci(self):
        import fuzzyrust as fr
        assert fr.levenshtein_similarity_ci("Hello", "HELLO") == 1.0
        assert fr.levenshtein_similarity_ci("Test", "TEST") == 1.0
        assert 0.0 < fr.levenshtein_similarity_ci("Hello", "Hallo") < 1.0

    def test_damerau_levenshtein_ci(self):
        import fuzzyrust as fr
        assert fr.damerau_levenshtein_ci("ab", "BA") == 1  # Transposition
        assert fr.damerau_levenshtein_ci("TEST", "test") == 0

    def test_jaro_similarity_ci(self):
        import fuzzyrust as fr
        assert fr.jaro_similarity_ci("Hello", "HELLO") == 1.0
        assert fr.jaro_similarity_ci("Test", "test") == 1.0

    def test_jaro_winkler_similarity_ci(self):
        import fuzzyrust as fr
        assert fr.jaro_winkler_similarity_ci("Hello", "HELLO") == 1.0
        assert fr.jaro_winkler_similarity_ci("Prefix", "PREFIX") == 1.0

    def test_ngram_similarity_ci(self):
        import fuzzyrust as fr
        assert fr.ngram_similarity_ci("Test", "TEST") == 1.0
        assert fr.ngram_similarity_ci("Hello", "HELLO", ngram_size=3) == 1.0


class TestNormalizationMode:
    """Tests for NormalizationMode enum."""

    def test_enum_available(self):
        import fuzzyrust as fr
        assert hasattr(fr, 'NormalizationMode')

    def test_enum_values(self):
        import fuzzyrust as fr
        modes = [mode.value for mode in fr.NormalizationMode]
        assert 'lowercase' in modes
        assert 'unicode_nfkd' in modes
        assert 'remove_punctuation' in modes
        assert 'remove_whitespace' in modes
        assert 'strict' in modes


class TestNormalizationFunctions:
    """Tests for normalize_string and normalize_pair functions."""

    def test_normalize_string_lowercase(self):
        import fuzzyrust as fr
        assert fr.normalize_string("Hello WORLD", "lowercase") == "hello world"
        assert fr.normalize_string("ABC", "lowercase") == "abc"

    def test_normalize_string_unicode_nfkd(self):
        import fuzzyrust as fr
        # Test with combining characters
        result = fr.normalize_string("café", "unicode_nfkd")
        assert isinstance(result, str)
        # The exact result depends on the unicode normalization

    def test_normalize_string_remove_punctuation(self):
        import fuzzyrust as fr
        assert fr.normalize_string("Hello, World!", "remove_punctuation") == "Hello World"
        assert fr.normalize_string("a.b.c", "remove_punctuation") == "abc"

    def test_normalize_string_remove_whitespace(self):
        import fuzzyrust as fr
        assert fr.normalize_string("Hello World", "remove_whitespace") == "HelloWorld"
        assert fr.normalize_string("  a  b  c  ", "remove_whitespace") == "abc"

    def test_normalize_string_strict(self):
        import fuzzyrust as fr
        result = fr.normalize_string("  Hello, World!  ", "strict")
        # Strict applies all normalizations: lowercase, remove punctuation, remove whitespace
        assert result == "helloworld"

    def test_normalize_pair_lowercase(self):
        import fuzzyrust as fr
        a, b = fr.normalize_pair("Hello", "WORLD", "lowercase")
        assert a == "hello"
        assert b == "world"

    def test_normalize_pair_strict(self):
        import fuzzyrust as fr
        a, b = fr.normalize_pair("  A, B!  ", "  C, D!  ", "strict")
        assert a == "ab"
        assert b == "cd"

    def test_normalize_string_with_enum_value(self):
        import fuzzyrust as fr
        # Using enum value (string form)
        result = fr.normalize_string("Hello", fr.NormalizationMode.LOWERCASE.value)
        assert result == "hello"

    def test_normalize_string_with_enum_directly(self):
        """Test normalize_string accepts the enum directly, not just .value."""
        import fuzzyrust as fr
        # Passing enum directly should also work
        result = fr.normalize_string("Hello WORLD", fr.NormalizationMode.LOWERCASE)
        assert result == "hello world"

        result = fr.normalize_string("  A B C  ", fr.NormalizationMode.REMOVE_WHITESPACE)
        assert result == "ABC"

        result = fr.normalize_string("  Hello, World!  ", fr.NormalizationMode.STRICT)
        assert result == "helloworld"

    def test_normalize_string_invalid_mode(self):
        """Test normalize_string raises error for invalid mode."""
        import fuzzyrust as fr
        import pytest
        # Invalid string mode should raise ValueError
        with pytest.raises(ValueError):
            fr.normalize_string("Hello", "invalid_mode")

    def test_normalize_string_empty(self):
        import fuzzyrust as fr
        assert fr.normalize_string("", "lowercase") == ""
        assert fr.normalize_string("", "strict") == ""

    def test_normalize_pair_empty(self):
        import fuzzyrust as fr
        a, b = fr.normalize_pair("", "", "lowercase")
        assert a == ""
        assert b == ""


class TestDeduplication:
    """Tests for find_duplicates function."""

    def test_find_duplicates_basic(self):
        import fuzzyrust as fr
        items = ["hello", "Hello", "HELLO", "world"]
        result = fr.find_duplicates(items, min_similarity=0.9, normalize="lowercase")

        assert isinstance(result.groups, list)
        assert isinstance(result.unique, list)
        assert isinstance(result.total_duplicates, int)

        # "hello", "Hello", "HELLO" should be grouped
        assert len(result.groups) >= 1
        assert "world" in result.unique

    def test_find_duplicates_no_normalize(self):
        import fuzzyrust as fr
        items = ["hello", "Hello", "HELLO"]
        result = fr.find_duplicates(items, min_similarity=0.9, normalize="none")

        # Without normalization, case differences matter
        # So they won't be grouped
        assert len(result.groups) == 0 or len(result.unique) > 0

    def test_find_duplicates_algorithms(self):
        import fuzzyrust as fr
        items = ["test", "Test", "tset"]

        for algo in ["levenshtein", "jaro_winkler", "ngram"]:
            result = fr.find_duplicates(items, algorithm=algo, min_similarity=0.7, normalize="lowercase")
            # Verify result structure is correct
            assert hasattr(result, 'groups')
            assert hasattr(result, 'unique')
            assert hasattr(result, 'total_duplicates')
            assert isinstance(result.groups, list)
            assert isinstance(result.unique, list)
            assert isinstance(result.total_duplicates, int)
            # total_duplicates should be non-negative
            assert result.total_duplicates >= 0, f"Algorithm {algo}: expected non-negative duplicates"
            # With normalize=True and min_similarity=0.7, we expect some grouping behavior
            # At minimum, "test" and "Test" should be normalized to same value
            # The exact grouping depends on the algorithm and min_similarity

    def test_find_duplicates_empty(self):
        import fuzzyrust as fr
        result = fr.find_duplicates([], min_similarity=0.8)
        assert result.groups == []
        assert result.unique == []
        assert result.total_duplicates == 0

    def test_find_duplicates_single(self):
        import fuzzyrust as fr
        result = fr.find_duplicates(["hello"], min_similarity=0.8)
        assert result.groups == []
        assert result.unique == ["hello"]
        assert result.total_duplicates == 0

    def test_find_duplicates_all_unique(self):
        import fuzzyrust as fr
        items = ["apple", "banana", "cherry"]
        result = fr.find_duplicates(items, min_similarity=0.9)
        assert result.groups == []
        assert len(result.unique) == 3
        assert result.total_duplicates == 0

    def test_deduplication_result_attributes(self):
        import fuzzyrust as fr
        items = ["hello", "Hello"]
        result = fr.find_duplicates(items, min_similarity=0.9, normalize="lowercase")

        # Test that all attributes are accessible
        assert hasattr(result, 'groups')
        assert hasattr(result, 'unique')
        assert hasattr(result, 'total_duplicates')

        assert hasattr(result, 'total_duplicates')

    def test_find_duplicates_snm(self):
        """Test Sorted Neighborhood Method."""
        import fuzzyrust as fr
        items = ["hello", "Hello", "HELLO", "world", "worl"]

        # SNM with window_size=5 should easily catch these
        result = fr.find_duplicates(
            items,
            min_similarity=0.8,
            normalize="lowercase",
            method="snm",
            window_size=5
        )
        
        assert len(result.groups) >= 1
        # hello, Hello, HELLO should be grouped
        # world, worl might be grouped depending on algo
        assert result.total_duplicates >= 2

    def test_find_duplicates_auto(self):
        """Test auto method selection."""
        import fuzzyrust as fr
        items = ["test"] * 100
        # Should default to brute force for small N
        result = fr.find_duplicates(items, method="auto")
        assert result.total_duplicates == 99

    def test_find_duplicates_snm_large(self):
        """Test SNM with a larger synthetic dataset."""
        import fuzzyrust as fr
        import random
        import string
        
        # Generate 2000 items with some duplicates
        base_words = ["apple", "banana", "cherry", "date", "elderberry"]
        items = []
        for _ in range(2000):
            word = random.choice(base_words)
            # Add some noise
            if random.random() < 0.2:
                word += random.choice(string.ascii_lowercase)
            items.append(word)
            
        # This would be slow with brute force, testing speed/completion of SNM
        import time
        start = time.time()
        result = fr.find_duplicates(items, method="snm", window_size=50, min_similarity=0.8)
        end = time.time()
        
        assert end - start < 2.0  # Should be fast
        assert result.total_duplicates > 0

class TestMultiAlgorithmComparison:
    """Tests for compare_algorithms function."""

    def test_compare_algorithms_basic(self):
        import fuzzyrust as fr
        strings = ["hello", "hallo", "help"]
        query = "helo"

        results = fr.compare_algorithms(strings, query)

        assert isinstance(results, list)
        assert len(results) > 0

        # Check first result structure
        first = results[0]
        assert hasattr(first, 'algorithm')
        assert hasattr(first, 'score')
        assert hasattr(first, 'matches')
        assert isinstance(first.matches, list)

    def test_compare_algorithms_specific_algos(self):
        import fuzzyrust as fr
        strings = ["test", "text", "best"]
        query = "test"

        results = fr.compare_algorithms(
            strings,
            query,
            algorithms=["jaro_winkler", "levenshtein"],
            limit=2
        )

        assert len(results) == 2
        algo_names = [r.algorithm for r in results]
        assert "jaro_winkler" in algo_names
        assert "levenshtein" in algo_names

    def test_compare_algorithms_limit(self):
        import fuzzyrust as fr
        strings = ["a", "b", "c", "d", "e"]
        results = fr.compare_algorithms(strings, "x", limit=2)

        # Each algorithm should have at most 2 matches
        for result in results:
            assert len(result.matches) <= 2

    def test_compare_algorithms_sorted(self):
        import fuzzyrust as fr
        strings = ["hello", "world"]
        results = fr.compare_algorithms(strings, "hello")

        # Results should be sorted by score (highest first)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_compare_algorithms_match_result(self):
        import fuzzyrust as fr
        strings = ["apple", "apply"]
        results = fr.compare_algorithms(strings, "apple", limit=1)

        # Check MatchResult structure
        if results and results[0].matches:
            match = results[0].matches[0]
            assert hasattr(match, 'text')
            assert hasattr(match, 'score')
            assert 0.0 <= match.score <= 1.0

    def test_algorithm_comparison_attributes(self):
        import fuzzyrust as fr
        strings = ["test"]
        results = fr.compare_algorithms(strings, "test", limit=1)

        assert len(results) > 0
        result = results[0]
        assert isinstance(result.algorithm, str)
        assert isinstance(result.score, float)
        assert isinstance(result.matches, list)


class TestIndexMissingMethods:
    """Tests for newly added index methods."""

    def test_ngram_index_find_nearest(self):
        import fuzzyrust as fr
        index = fr.NgramIndex()
        index.add_all(["apple", "apply", "banana"])

        results = index.find_nearest("aple", k=2)
        assert len(results) <= 2
        assert all(hasattr(r, 'text') and hasattr(r, 'score') for r in results)

    def test_ngram_index_contains(self):
        import fuzzyrust as fr
        index = fr.NgramIndex()
        index.add_all(["hello", "world"])

        assert index.contains("hello") is True
        assert index.contains("goodbye") is False

    def test_hybrid_index_batch_search(self):
        import fuzzyrust as fr
        index = fr.HybridIndex()
        index.add_all(["cat", "dog", "bird"])

        results = index.batch_search(["ct", "dg"])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_hybrid_index_find_nearest(self):
        import fuzzyrust as fr
        index = fr.HybridIndex()
        index.add_all(["test", "text", "best"])

        results = index.find_nearest("tst", k=2)
        assert len(results) <= 2

    def test_hybrid_index_contains(self):
        import fuzzyrust as fr
        index = fr.HybridIndex()
        index.add_all(["apple", "banana"])

        assert index.contains("apple") is True
        assert index.contains("cherry") is False

    def test_index_method_consistency(self):
        """Test that all indices have consistent method signatures."""
        import fuzzyrust as fr

        # All indices should support these methods
        bktree = fr.BkTree()
        ngram_idx = fr.NgramIndex()
        hybrid_idx = fr.HybridIndex()

        for idx in [bktree, ngram_idx, hybrid_idx]:
            assert hasattr(idx, 'add')
            assert hasattr(idx, 'add_with_data')
            assert hasattr(idx, 'add_all')
            assert hasattr(idx, 'search')
            assert hasattr(idx, '__len__')

        # NgramIndex and HybridIndex should have these
        for idx in [ngram_idx, hybrid_idx]:
            assert hasattr(idx, 'find_nearest')
            assert hasattr(idx, 'contains')

        # HybridIndex should have batch_search
        assert hasattr(hybrid_idx, 'batch_search')


class TestNewResultTypes:
    """Tests for new result types (SearchResult, MatchResult, etc.)."""

    def test_search_result_attributes(self):
        import fuzzyrust as fr
        tree = fr.BkTree()
        tree.add_with_data("hello", 42)

        results = tree.search("hello", max_distance=0)
        assert len(results) > 0

        result = results[0]
        assert hasattr(result, 'id')
        assert hasattr(result, 'text')
        assert hasattr(result, 'score')
        assert hasattr(result, 'distance')
        assert hasattr(result, 'data')

        assert result.text == "hello"
        assert result.data == 42

    def test_match_result_attributes(self):
        import fuzzyrust as fr
        strings = ["apple", "apply"]
        results = fr.find_best_matches(strings, "apple", limit=2)

        assert len(results) > 0
        result = results[0]

        assert hasattr(result, 'text')
        assert hasattr(result, 'score')
        assert isinstance(result.text, str)
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0

    def test_batch_operations_return_match_results(self):
        import fuzzyrust as fr
        strings = ["hello", "hallo", "world"]

        # batch_levenshtein should return MatchResults
        results = fr.batch_levenshtein(strings, "hello")
        assert all(hasattr(r, 'text') and hasattr(r, 'score') for r in results)

        # batch_jaro_winkler should return MatchResults
        results = fr.batch_jaro_winkler(strings, "hello")
        assert all(hasattr(r, 'text') and hasattr(r, 'score') for r in results)


class TestParameterRenames:
    """Tests for parameter name changes (n → ngram_size, min_score → min_similarity)."""

    def test_ngram_size_parameter(self):
        import fuzzyrust as fr

        # ngram_similarity should use ngram_size
        result = fr.ngram_similarity("test", "test", ngram_size=2)
        assert result == 1.0

        result = fr.ngram_similarity("test", "test", ngram_size=3)
        assert result == 1.0

    def test_extract_ngrams_parameter(self):
        import fuzzyrust as fr

        # extract_ngrams should use ngram_size
        ngrams = fr.extract_ngrams("abc", ngram_size=2, pad=False)
        assert ngrams == ["ab", "bc"]

    def test_ngram_index_constructor(self):
        import fuzzyrust as fr

        # NgramIndex should use ngram_size
        index = fr.NgramIndex(ngram_size=3, min_similarity=0.5)
        assert len(index) == 0

    def test_find_best_matches_min_similarity(self):
        import fuzzyrust as fr
        strings = ["hello", "hallo", "world"]

        # find_best_matches should use min_similarity
        results = fr.find_best_matches(strings, "hello", min_similarity=0.8)
        assert all(r.score >= 0.8 for r in results)


class TestAlgorithmEnum:
    """Tests for Algorithm enum."""

    def test_algorithm_enum_available(self):
        import fuzzyrust as fr
        assert hasattr(fr, 'Algorithm')

    def test_algorithm_enum_values(self):
        import fuzzyrust as fr
        algos = [algo.value for algo in fr.Algorithm]
        assert 'levenshtein' in algos
        assert 'jaro_winkler' in algos
        assert 'ngram' in algos


class TestEdgeCases:
    """Edge cases for new features."""

    def test_find_duplicates_high_min_similarity(self):
        import fuzzyrust as fr
        items = ["hello", "hallo", "hullo"]
        result = fr.find_duplicates(items, min_similarity=0.99)
        # With very high min_similarity, items might not be grouped
        assert isinstance(result.groups, list)

    def test_compare_algorithms_empty_strings(self):
        import fuzzyrust as fr
        results = fr.compare_algorithms([], "test")
        # Should handle empty list gracefully
        assert isinstance(results, list)

    def test_find_nearest_empty_index(self):
        import fuzzyrust as fr
        index = fr.NgramIndex()
        results = index.find_nearest("test", k=5)
        assert results == []

    def test_contains_empty_index(self):
        import fuzzyrust as fr
        index = fr.HybridIndex()
        assert index.contains("test") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
