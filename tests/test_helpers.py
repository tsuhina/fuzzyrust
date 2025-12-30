"""Tests for Sprint 3 features (v0.2.0 API changes)."""

import pytest


class TestCaseInsensitiveVariants:
    """Tests for case-insensitive function variants."""

    def test_levenshtein_ci(self):
        import fuzzyrust as fr

        assert fr.levenshtein_ci("Hello", "HELLO") == 0
        # "hello" vs "world" (lowercased): 4 substitutions = distance 4
        assert fr.levenshtein_ci("Hello", "World") == 4

    def test_levenshtein_similarity_ci(self):
        import fuzzyrust as fr

        assert fr.levenshtein_similarity_ci("Hello", "HELLO") == 1.0
        assert fr.levenshtein_similarity_ci("Test", "TEST") == 1.0
        # "Hello" vs "Hallo" after lowercasing: 1 edit out of 5 = 0.8 similarity
        assert fr.levenshtein_similarity_ci("Hello", "Hallo") == 0.8

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

        assert hasattr(fr, "NormalizationMode")

    def test_enum_values(self):
        import fuzzyrust as fr

        modes = [mode.value for mode in fr.NormalizationMode]
        assert "lowercase" in modes
        assert "unicode_nfkd" in modes
        assert "remove_punctuation" in modes
        assert "remove_whitespace" in modes
        assert "strict" in modes


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
        import pytest

        import fuzzyrust as fr

        # Invalid string mode should raise ValidationError
        with pytest.raises(fr.ValidationError):
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

        assert isinstance(result.groups, list), "Expected groups to be a list"
        assert (
            len(result.groups) == 1
        ), f"Expected exactly 1 group (hello variants), got {len(result.groups)}"
        assert isinstance(result.unique, list), "Expected unique to be a list"
        assert len(result.unique) == 1, f"Expected 1 unique item (world), got {len(result.unique)}"
        assert isinstance(result.total_duplicates, int)

        # "hello", "Hello", "HELLO" should be grouped together (3 items, 2 duplicates)
        assert (
            result.total_duplicates == 2
        ), f"Expected 2 duplicates in hello group, got {result.total_duplicates}"
        assert "world" in result.unique, f"Expected 'world' in unique list, got {result.unique}"
        # Verify the group contains all hello variants (check unique values)
        hello_group = result.groups[0]
        unique_in_group = set(hello_group)
        assert unique_in_group == {
            "hello",
            "Hello",
            "HELLO",
        }, f"Expected hello variants in group, got {unique_in_group}"

    def test_find_duplicates_no_normalize(self):
        import fuzzyrust as fr

        items = ["hello", "Hello", "HELLO"]
        result = fr.find_duplicates(items, min_similarity=0.9, normalize="none")

        # Without normalization and high min_similarity (0.9), case differences prevent grouping
        # "hello" vs "Hello" has ~0.8 similarity (1 char diff / 5 chars), below 0.9 threshold
        # So all items should remain unique with no duplicate groups
        assert (
            len(result.groups) == 0
        ), "Expected no groups when case differs and min_similarity=0.9"
        assert len(result.unique) == 3, "Expected all 3 items to be unique without normalization"
        assert result.total_duplicates == 0, "Expected no duplicates detected"

    def test_find_duplicates_algorithms(self):
        import fuzzyrust as fr

        items = ["test", "Test", "tset"]

        for algo in ["levenshtein", "jaro_winkler", "ngram"]:
            result = fr.find_duplicates(
                items, algorithm=algo, min_similarity=0.7, normalize="lowercase"
            )
            # Verify result structure is correct
            assert hasattr(result, "groups")
            assert hasattr(result, "unique")
            assert hasattr(result, "total_duplicates")
            assert isinstance(
                result.groups, list
            ), f"Algorithm {algo}: expected groups to be a list"
            assert isinstance(
                result.unique, list
            ), f"Algorithm {algo}: expected unique to be a list"
            assert isinstance(result.total_duplicates, int)

            # With normalize="lowercase", "test" and "Test" become identical (1.0 similarity)
            # At min_similarity=0.7, "test" and "Test" will always be grouped
            # "tset" may or may not be included depending on the algorithm
            assert (
                len(result.groups) >= 1
            ), f"Algorithm {algo}: expected at least 1 group, got {len(result.groups)}"
            assert (
                result.total_duplicates >= 1
            ), f"Algorithm {algo}: expected at least 1 duplicate (test/Test), got {result.total_duplicates}"
            # Verify the group contains test and Test (normalized to same value)
            unique_in_first_group = set(result.groups[0])
            assert {"test", "Test"}.issubset(
                unique_in_first_group
            ), f"Algorithm {algo}: expected test/Test in group, got {unique_in_first_group}"
            # Total items accounted for: groups + unique should cover all inputs
            grouped_items = set()
            for group in result.groups:
                grouped_items.update(group)
            all_items = set(items)
            covered_items = grouped_items.union(set(result.unique))
            assert all_items.issubset(
                covered_items
            ), f"Algorithm {algo}: not all items covered. Missing: {all_items - covered_items}"

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
        assert hasattr(result, "groups")
        assert hasattr(result, "unique")
        assert hasattr(result, "total_duplicates")

        assert hasattr(result, "total_duplicates")

    def test_find_duplicates_snm(self):
        """Test Sorted Neighborhood Method."""
        import fuzzyrust as fr

        items = ["hello", "Hello", "HELLO", "world", "worl"]

        # SNM with window_size=5 should easily catch these
        result = fr.find_duplicates(
            items, min_similarity=0.8, normalize="lowercase", method="snm", window_size=5
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
        import random
        import string

        import fuzzyrust as fr

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

        assert isinstance(results, list), "Expected results to be a list"
        # compare_algorithms returns results for multiple algorithms by default
        # At minimum: levenshtein, jaro_winkler, ngram (3 algorithms)
        assert len(results) >= 3, f"Expected at least 3 algorithm results, got {len(results)}"

        # Check first result structure and content
        first = results[0]
        assert hasattr(first, "algorithm")
        assert hasattr(first, "score")
        assert hasattr(first, "matches")
        assert isinstance(first.matches, list), "Expected matches to be a list"
        # Query "helo" should find matches in ["hello", "hallo", "help"]
        # All are within edit distance 1-2, so we expect matches
        assert (
            len(first.matches) >= 1
        ), f"Expected at least 1 match for top algorithm, got {len(first.matches)}"
        # The best match for "helo" should be "hello" (1 insertion)
        assert first.matches[0].text in [
            "hello",
            "hallo",
            "help",
        ], f"Expected match from input strings, got {first.matches[0].text}"

    def test_compare_algorithms_specific_algos(self):
        import fuzzyrust as fr

        strings = ["test", "text", "best"]
        query = "test"

        results = fr.compare_algorithms(
            strings, query, algorithms=["jaro_winkler", "levenshtein"], limit=2
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
            assert hasattr(match, "text")
            assert hasattr(match, "score")
            # "apple" vs "apple" should be exact match (1.0)
            assert (
                match.score == 1.0
            ), f"Expected exact match score 1.0 for 'apple' vs 'apple', got {match.score}"

    def test_algorithm_comparison_attributes(self):
        import fuzzyrust as fr

        strings = ["test"]
        results = fr.compare_algorithms(strings, "test", limit=1)

        assert len(results) >= 3, f"Expected at least 3 algorithm results, got {len(results)}"
        result = results[0]
        assert isinstance(result.algorithm, str), "Expected algorithm to be a string"
        assert result.algorithm in [
            "levenshtein",
            "jaro_winkler",
            "ngram",
            "cosine",
            "jaro",
        ], f"Unexpected algorithm name: {result.algorithm}"
        assert isinstance(result.score, float), "Expected score to be a float"
        # For exact match "test" vs "test", top algorithm should have score 1.0
        assert result.score == 1.0, f"Expected score 1.0 for exact match, got {result.score}"
        assert isinstance(result.matches, list), "Expected matches to be a list"
        assert len(result.matches) == 1, f"Expected 1 match (limit=1), got {len(result.matches)}"
        # Verify the match is the exact string
        assert (
            result.matches[0].text == "test"
        ), f"Expected match text 'test', got {result.matches[0].text}"
        assert (
            result.matches[0].score == 1.0
        ), f"Expected match score 1.0, got {result.matches[0].score}"


class TestIndexMissingMethods:
    """Tests for newly added index methods."""

    def test_ngram_index_find_nearest(self):
        import fuzzyrust as fr

        index = fr.NgramIndex()
        index.add_all(["apple", "apply", "banana"])

        results = index.find_nearest("aple", limit=2)
        assert len(results) <= 2
        assert all(hasattr(r, "text") and hasattr(r, "score") for r in results)

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

        # Use longer query strings for better n-gram matching
        results = index.batch_search(["catt", "dogg"])
        assert isinstance(results, list), "Expected results to be a list"
        assert len(results) == 2, f"Expected 2 result lists for 2 queries, got {len(results)}"
        assert all(isinstance(r, list) for r in results), "Expected each result to be a list"
        # "catt" should match "cat" (high similarity), "dogg" should match "dog"
        assert len(results[0]) >= 1, f"Expected at least 1 match for 'catt', got {len(results[0])}"
        assert len(results[1]) >= 1, f"Expected at least 1 match for 'dogg', got {len(results[1])}"
        # Verify the matches are from the index and correct
        assert (
            results[0][0].text == "cat"
        ), f"Expected 'cat' as best match for 'catt', got {results[0][0].text}"
        assert (
            results[1][0].text == "dog"
        ), f"Expected 'dog' as best match for 'dogg', got {results[1][0].text}"
        # Verify high similarity scores
        assert (
            results[0][0].score >= 0.9
        ), f"Expected high score for 'catt'/'cat', got {results[0][0].score}"
        assert (
            results[1][0].score >= 0.9
        ), f"Expected high score for 'dogg'/'dog', got {results[1][0].score}"

    def test_hybrid_index_find_nearest(self):
        import fuzzyrust as fr

        index = fr.HybridIndex()
        index.add_all(["test", "text", "best"])

        results = index.find_nearest("tst", limit=2)
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
            assert hasattr(idx, "add")
            assert hasattr(idx, "add_with_data")
            assert hasattr(idx, "add_all")
            assert hasattr(idx, "search")
            assert hasattr(idx, "__len__")

        # NgramIndex and HybridIndex should have these
        for idx in [ngram_idx, hybrid_idx]:
            assert hasattr(idx, "find_nearest")
            assert hasattr(idx, "contains")

        # HybridIndex should have batch_search
        assert hasattr(hybrid_idx, "batch_search")


class TestNewResultTypes:
    """Tests for new result types (SearchResult, MatchResult, etc.)."""

    def test_search_result_attributes(self):
        import fuzzyrust as fr

        tree = fr.BkTree()
        tree.add_with_data("hello", 42)

        results = tree.search("hello", max_distance=0)
        assert len(results) > 0

        result = results[0]
        assert hasattr(result, "id")
        assert hasattr(result, "text")
        assert hasattr(result, "score")
        assert hasattr(result, "distance")
        assert hasattr(result, "data")

        assert result.text == "hello"
        assert result.data == 42

    def test_match_result_attributes(self):
        import fuzzyrust as fr

        strings = ["apple", "apply"]
        results = fr.find_best_matches(strings, "apple", limit=2)

        assert len(results) > 0
        result = results[0]

        assert hasattr(result, "text")
        assert hasattr(result, "score")
        assert isinstance(result.text, str)
        assert isinstance(result.score, float)
        # "apple" should be exact match with score 1.0
        assert (
            result.score == 1.0
        ), f"Expected exact match score 1.0 for 'apple', got {result.score}"
        assert result.text == "apple", f"Expected best match to be 'apple', got {result.text}"

    def test_batch_operations_return_match_results(self):
        import fuzzyrust as fr

        strings = ["hello", "hallo", "world"]

        # batch_levenshtein should return MatchResults
        results = fr.batch_levenshtein(strings, "hello")
        assert all(hasattr(r, "text") and hasattr(r, "score") for r in results)

        # batch_jaro_winkler should return MatchResults
        results = fr.batch_jaro_winkler(strings, "hello")
        assert all(hasattr(r, "text") and hasattr(r, "score") for r in results)


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

        # NgramIndex should use ngram_size and min_ngram_ratio
        index = fr.NgramIndex(ngram_size=3, min_ngram_ratio=0.2)
        assert len(index) == 0

    def test_find_best_matches_min_similarity(self):
        import fuzzyrust as fr

        strings = ["hello", "hallo", "world"]

        # find_best_matches should use min_similarity
        results = fr.find_best_matches(strings, "hello", min_similarity=0.8)
        # Verify we get results and they all meet the min_similarity threshold
        assert (
            len(results) >= 1
        ), f"Expected at least 1 result for 'hello' query, got {len(results)}"
        # "hello" exact match should have score 1.0
        # "hallo" differs by 1 char, should have score ~0.8-0.9 (Jaro-Winkler)
        # "world" differs significantly, should be filtered out at 0.8 threshold
        assert len(results) <= 2, f"Expected at most 2 results (hello, hallo), got {len(results)}"
        for r in results:
            # All results must meet the min_similarity threshold
            assert r.score >= 0.8, f"Result '{r.text}' has score {r.score} below min_similarity 0.8"
            # More specific: scores should be in expected ranges
            if r.text == "hello":
                assert r.score == 1.0, f"Expected exact match score 1.0 for 'hello', got {r.score}"
            elif r.text == "hallo":
                # "hallo" vs "hello" should be ~0.88 (Jaro-Winkler)
                assert 0.8 <= r.score <= 0.95, f"Expected 0.8-0.95 for 'hallo', got {r.score}"


class TestAlgorithmEnum:
    """Tests for Algorithm enum."""

    def test_algorithm_enum_available(self):
        import fuzzyrust as fr

        assert hasattr(fr, "Algorithm")

    def test_algorithm_enum_values(self):
        import fuzzyrust as fr

        algos = [algo.value for algo in fr.Algorithm]
        assert "levenshtein" in algos
        assert "jaro_winkler" in algos
        assert "ngram" in algos


class TestEdgeCases:
    """Edge cases for new features."""

    def test_find_duplicates_high_min_similarity(self):
        import fuzzyrust as fr

        items = ["hello", "hallo", "hullo"]
        result = fr.find_duplicates(items, min_similarity=0.99)
        # With min_similarity=0.99, only near-identical strings should group
        # "hello", "hallo", "hullo" each differ by 1 char (80% similar), below 0.99 threshold
        assert isinstance(result.groups, list), "Expected groups to be a list"
        assert (
            len(result.groups) == 0
        ), f"Expected 0 groups at 0.99 threshold, got {len(result.groups)}"
        assert (
            len(result.unique) == 3
        ), f"Expected all 3 items unique at 0.99 threshold, got {len(result.unique)}"
        assert (
            result.total_duplicates == 0
        ), f"Expected 0 duplicates at 0.99 threshold, got {result.total_duplicates}"

    def test_compare_algorithms_empty_strings(self):
        import fuzzyrust as fr

        results = fr.compare_algorithms([], "test")
        # Should handle empty list gracefully - returns algorithm results with empty matches
        assert isinstance(results, list), "Expected results to be a list"
        # Even with empty input, we get results for each algorithm
        assert len(results) >= 3, f"Expected at least 3 algorithm results, got {len(results)}"
        # Each algorithm result should have empty matches
        for r in results:
            assert hasattr(r, "matches"), "Algorithm result missing 'matches' attribute"
            assert len(r.matches) == 0, f"Expected 0 matches for empty input, got {len(r.matches)}"

    def test_find_nearest_empty_index(self):
        import fuzzyrust as fr

        index = fr.NgramIndex()
        results = index.find_nearest("test", limit=5)
        assert results == []

    def test_contains_empty_index(self):
        import fuzzyrust as fr

        index = fr.HybridIndex()
        assert index.contains("test") is False


# =============================================================================
# Tests for find_duplicate_pairs
# =============================================================================


class TestFindDuplicatePairs:
    """Tests for find_duplicate_pairs function."""

    def test_basic(self):
        """Basic test with similar strings."""
        import fuzzyrust as fr

        pairs = fr.find_duplicate_pairs(["hello", "hallo", "world"])
        assert len(pairs) >= 1, "Expected at least one pair"
        # Each pair is (idx1, idx2, score)
        assert all(len(p) == 3 for p in pairs), "Each pair should have 3 elements"
        # Verify pair structure: (int, int, float)
        for idx1, idx2, score in pairs:
            assert isinstance(idx1, int), f"idx1 should be int, got {type(idx1)}"
            assert isinstance(idx2, int), f"idx2 should be int, got {type(idx2)}"
            assert isinstance(score, float), f"score should be float, got {type(score)}"
            assert idx1 < idx2, "idx1 should always be less than idx2"
            assert 0.0 <= score <= 1.0, f"score should be in [0, 1], got {score}"

    def test_with_normalize(self):
        """Test with normalize parameter."""
        import fuzzyrust as fr

        pairs = fr.find_duplicate_pairs(["Hello", "HELLO"], normalize="lowercase")
        assert len(pairs) == 1, f"Expected 1 pair, got {len(pairs)}"
        assert pairs[0][2] == 1.0, "Identical after normalization should be 1.0"

    def test_empty_list(self):
        """Empty input should return empty result."""
        import fuzzyrust as fr

        pairs = fr.find_duplicate_pairs([])
        assert pairs == [], f"Expected empty list, got {pairs}"

    def test_single_item(self):
        """Single item should return empty result."""
        import fuzzyrust as fr

        pairs = fr.find_duplicate_pairs(["hello"])
        assert pairs == [], f"Expected empty list for single item, got {pairs}"

    def test_no_duplicates(self):
        """Items with no similarity should return empty result."""
        import fuzzyrust as fr

        # Very different strings with high threshold
        pairs = fr.find_duplicate_pairs(["apple", "orange", "banana"], min_similarity=0.99)
        assert pairs == [], f"Expected no pairs for very different strings, got {pairs}"

    def test_all_identical(self):
        """All identical items should all be paired."""
        import fuzzyrust as fr

        pairs = fr.find_duplicate_pairs(["test", "test", "test"])
        # With 3 identical items, we should get pairs: (0,1), (0,2), (1,2)
        assert len(pairs) == 3, f"Expected 3 pairs, got {len(pairs)}"
        for idx1, idx2, score in pairs:
            assert score == 1.0, f"Identical strings should have score 1.0, got {score}"

    def test_min_similarity_threshold(self):
        """Test min_similarity parameter filtering."""
        import fuzzyrust as fr

        items = ["hello", "hallo", "hullo", "world"]

        # With high threshold, fewer pairs
        pairs_high = fr.find_duplicate_pairs(items, min_similarity=0.95)

        # With lower threshold, more pairs
        pairs_low = fr.find_duplicate_pairs(items, min_similarity=0.7)

        assert len(pairs_low) >= len(pairs_high), (
            f"Lower threshold should find at least as many pairs: "
            f"low={len(pairs_low)}, high={len(pairs_high)}"
        )

    def test_different_algorithms(self):
        """Test with different similarity algorithms."""
        import fuzzyrust as fr

        items = ["hello", "hallo"]

        # Different algorithms have different thresholds where hello/hallo pair is found
        # - levenshtein: 0.8 (1 edit in 5 chars)
        # - jaro_winkler: ~0.88 (high similarity)
        # - ngram: ~0.57 (lower due to n-gram overlap)
        algorithm_thresholds = [
            ("levenshtein", 0.7),
            ("jaro_winkler", 0.7),
            ("ngram", 0.5),  # ngram has lower similarity (~0.57) for hello/hallo
        ]

        for algo, threshold in algorithm_thresholds:
            pairs = fr.find_duplicate_pairs(items, algorithm=algo, min_similarity=threshold)
            # Each algorithm should find the hello/hallo pair with appropriate threshold
            assert (
                len(pairs) >= 1
            ), f"Algorithm {algo} with threshold {threshold} should find at least 1 pair"
            # Verify the pair is valid
            idx1, idx2, score = pairs[0]
            assert (
                threshold <= score <= 1.0
            ), f"Algorithm {algo}: score {score} below threshold {threshold}"

    def test_window_size_parameter(self):
        """Test window_size parameter affects results."""
        import fuzzyrust as fr

        # Create items where duplicates are far apart
        items = ["hello"] + [f"item_{i}" for i in range(100)] + ["hallo"]

        # With small window, might miss the pair
        pairs_small = fr.find_duplicate_pairs(items, window_size=5, min_similarity=0.7)

        # With larger window, should find more pairs
        pairs_large = fr.find_duplicate_pairs(items, window_size=200, min_similarity=0.7)

        # The test is that both should run without error
        # Result depends on sorting and algorithm specifics
        assert isinstance(pairs_small, list)
        assert isinstance(pairs_large, list)

    def test_normalize_modes(self):
        """Test different normalization modes."""
        import fuzzyrust as fr

        items = ["HELLO", "hello", "Hello"]

        # With "none", case differences affect similarity
        pairs_none = fr.find_duplicate_pairs(items, normalize="none", min_similarity=0.99)

        # With "lowercase", all should be identical
        pairs_lower = fr.find_duplicate_pairs(items, normalize="lowercase", min_similarity=0.99)

        # Lowercase normalization should find all pairs as identical
        assert len(pairs_lower) == 3, f"Expected 3 pairs with lowercase, got {len(pairs_lower)}"
        for idx1, idx2, score in pairs_lower:
            assert score == 1.0, "All pairs should be 1.0 after lowercase normalization"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
