"""Tests for batch processing functions.

This module tests the batch similarity computation functions provided by fuzzyrust,
including batch_levenshtein, batch_jaro_winkler, find_best_matches, and more.
"""

import pytest

import fuzzyrust as fr


class TestBatchProcessing:
    """Tests for batch processing functions."""

    def test_batch_levenshtein(self):
        strings = ["hello", "hallo", "hullo", "world"]
        results = fr.batch_levenshtein(strings, "hello")
        # Results are now MatchResult objects
        assert results[0].score == 1.0  # hello (exact match)
        assert results[0].text == "hello"
        assert results[1].text == "hallo"
        assert results[3].text == "world"

    def test_batch_jaro_winkler(self):
        strings = ["hello", "hallo", "world"]
        results = fr.batch_jaro_winkler(strings, "hello")
        # Results are now MatchResult objects
        assert results[0].score == 1.0
        assert results[1].score > results[2].score

    def test_find_best_matches(self):
        strings = ["apple", "apply", "banana", "application"]
        results = fr.find_best_matches(strings, "appel", limit=2, min_similarity=0.0)
        assert len(results) == 2
        # Results are now MatchResult objects
        assert results[0].text == "apple"  # Best match
        assert results[0].score >= results[1].score  # Sorted by score


class TestBatchEdgeCases:
    """Edge case tests for batch operations."""

    def test_batch_levenshtein_empty_list(self):
        """Test batch_levenshtein with empty list."""
        result = fr.batch_levenshtein([], "hello")
        assert result == []

    def test_batch_levenshtein_empty_query(self):
        """Test batch_levenshtein with empty query."""
        strings = ["hello", "world"]
        result = fr.batch_levenshtein(strings, "")
        # Empty string vs non-empty = distance equals length, normalized similarity = 0.0
        assert [r.score for r in result] == [0.0, 0.0]

    def test_batch_levenshtein_empty_strings_in_list(self):
        """Test batch_levenshtein with empty strings in list."""
        strings = ["", "a", ""]
        result = fr.batch_levenshtein(strings, "a")
        # "" vs "a" = distance 1, normalized = 0.0; "a" vs "a" = exact match = 1.0
        assert [r.score for r in result] == [0.0, 1.0, 0.0]

    def test_batch_jaro_winkler_empty_list(self):
        """Test batch_jaro_winkler with empty list."""
        result = fr.batch_jaro_winkler([], "hello")
        assert result == []

    def test_batch_jaro_winkler_unicode(self):
        """Test batch_jaro_winkler with Unicode strings."""
        strings = ["cafe", "cafe", "caf"]
        result = fr.batch_jaro_winkler(strings, "cafe")
        assert len(result) == 3
        # batch_jaro_winkler returns MatchResult objects with .score attribute
        # All strings are similar to "cafe" (short strings with common prefix "caf")
        # "cafe" vs "cafe" = 1.0 (identical)
        # "cafe" vs "cafe" should be high (~0.93+ due to 1 char difference in 4 chars)
        # "caf" vs "cafe" should be high (~0.93+ due to length difference)
        for r in result:
            assert 0.85 <= r.score <= 1.0, (
                f"Expected score in [0.85, 1.0] for {r.text!r}, got {r.score}"
            )

    def test_find_best_matches_empty_list(self):
        """Test find_best_matches with empty list."""
        result = fr.find_best_matches([], "hello")
        assert result == []

    def test_find_best_matches_limit_zero(self):
        """Test find_best_matches with limit=0."""
        strings = ["hello", "world"]
        result = fr.find_best_matches(strings, "hello", limit=0)
        assert result == []

    def test_find_best_matches_min_score_one(self):
        """Test find_best_matches with min_similarity=1.0 (exact matches only)."""
        strings = ["hello", "hallo", "hello"]
        result = fr.find_best_matches(strings, "hello", min_similarity=1.0)
        # Note: results are deduplicated, so only one "hello" match
        assert len(result) >= 1
        # Results are MatchResult objects
        assert all(r.score == 1.0 for r in result)

    def test_find_best_matches_all_algorithms(self):
        """Test find_best_matches with all valid algorithms."""
        strings = ["test", "text", "best"]
        algorithms = [
            "levenshtein",
            "damerau_levenshtein",
            "jaro",
            "jaro_winkler",
            "ngram",
            "bigram",
            "trigram",
            "lcs",
            "cosine",
        ]

        for algo in algorithms:
            result = fr.find_best_matches(strings, "test", algorithm=algo, min_similarity=0.0)
            # With min_similarity=0.0, all 3 strings should be returned
            assert len(result) == 3, f"Expected 3 results for algorithm {algo}, got {len(result)}"
            # Results are MatchResult objects with scores in valid range
            for r in result:
                assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of range for algorithm {algo}"


class TestBatchSimilarityPairs:
    """Tests for batch_similarity_pairs function."""

    def test_identical_strings(self):
        """Identical string pairs should have similarity 1.0."""
        result = fr.batch_similarity_pairs(["hello", "world"], ["hello", "world"], "levenshtein")
        assert result == [1.0, 1.0], f"Expected [1.0, 1.0], got {result}"

    def test_different_strings(self):
        """Different string pairs should have low similarity."""
        result = fr.batch_similarity_pairs(["hello"], ["world"], "jaro_winkler")
        assert len(result) == 1
        assert 0.0 <= result[0] <= 0.5, f"Expected low similarity, got {result[0]}"

    def test_empty_lists(self):
        """Empty input lists should return empty result list."""
        result = fr.batch_similarity_pairs([], [], "levenshtein")
        assert result == [], f"Expected empty list, got {result}"

    def test_various_algorithms(self):
        """All supported algorithms should work correctly."""
        for algo in ["levenshtein", "jaro", "jaro_winkler", "ngram", "cosine"]:
            result = fr.batch_similarity_pairs(["test"], ["test"], algo)
            assert result[0] == 1.0, f"Algorithm {algo} failed for identical strings"

    def test_mixed_similarity(self):
        """Test with pairs having different similarity levels."""
        left = ["hello", "world", "test"]
        right = ["hello", "word", "testing"]
        result = fr.batch_similarity_pairs(left, right, "levenshtein")

        assert len(result) == 3
        assert result[0] == 1.0, "First pair (hello, hello) should be 1.0"
        assert 0.5 < result[1] < 1.0, "Second pair (world, word) should have medium similarity"
        assert 0.5 < result[2] < 1.0, "Third pair (test, testing) should have medium similarity"

    def test_unicode_strings(self):
        """Test with Unicode strings."""
        result = fr.batch_similarity_pairs(["cafe", "hello"], ["cafe", "hello"], "jaro_winkler")
        assert result == [1.0, 1.0]


class TestConvenienceAliases:
    """Tests for convenience aliases."""

    def test_edit_distance_alias(self):
        """Test that edit_distance is an alias for levenshtein."""
        assert fr.edit_distance("kitten", "sitting") == fr.levenshtein("kitten", "sitting")

    def test_similarity_alias(self):
        """Test that similarity is an alias for jaro_winkler_similarity."""
        assert fr.similarity("hello", "hallo") == fr.jaro_winkler_similarity("hello", "hallo")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
