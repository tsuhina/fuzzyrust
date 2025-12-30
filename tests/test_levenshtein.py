"""Tests for edit distance algorithms: Levenshtein, Damerau-Levenshtein, and Hamming.

This module tests the core edit distance algorithms provided by fuzzyrust,
including their standard and case-insensitive variants.
"""

import pytest

import fuzzyrust as fr


class TestLevenshtein:
    """Tests for Levenshtein distance functions."""

    def test_identical_strings(self):
        assert fr.levenshtein("hello", "hello") == 0, "Identical strings should have distance 0"
        assert fr.levenshtein("", "") == 0, "Two empty strings should have distance 0"

    def test_empty_strings(self):
        assert fr.levenshtein("hello", "") == 5, "Distance to empty string equals string length"
        assert fr.levenshtein("", "hello") == 5, "Distance from empty string equals target length"

    def test_classic_examples(self):
        # kitten -> sitten (s for k) -> sittin (i for e) -> sitting (g added) = 3 edits
        assert fr.levenshtein("kitten", "sitting") == 3, "kitten->sitting requires 3 edits"
        # saturday -> sunday requires 3 edits
        assert fr.levenshtein("saturday", "sunday") == 3, "saturday->sunday requires 3 edits"

    def test_unicode(self):
        # cafe with accent vs without: 1 character difference
        assert fr.levenshtein("caf\u00e9", "cafe") == 1, "Accent difference is 1 edit"
        # 3 chars vs 2 chars: 1 deletion
        assert fr.levenshtein("\u65e5\u672c\u8a9e", "\u65e5\u672c") == 1, "Removing one Japanese character is 1 edit"

    def test_max_distance(self):
        # Should return max_distance + 1 when distance exceeds threshold
        # (RapidFuzz compatible behavior - changed from usize::MAX in v0.2.0)
        result = fr.levenshtein("abcdef", "ghijkl", max_distance=3)
        assert result == 4, "Distance exceeding max_distance returns max_distance + 1"

        # Should return actual distance when within threshold
        assert fr.levenshtein("abc", "abd", max_distance=2) == 1, \
            "Distance within max_distance returns actual distance"

        # Test the bounded variant that returns None when exceeded
        assert fr.levenshtein_bounded("abc", "abd", max_distance=2) == 1
        assert fr.levenshtein_bounded("abcdef", "ghijkl", max_distance=3) is None

    def test_similarity(self):
        assert fr.levenshtein_similarity("hello", "hello") == 1.0, \
            "Identical strings should have similarity 1.0"
        assert fr.levenshtein_similarity("hello", "") == 0.0, \
            "Comparing to empty string should have similarity 0.0"
        # "hello" vs "hallo": 1 edit out of 5 chars = 0.8 similarity
        assert fr.levenshtein_similarity("hello", "hallo") == 0.8, \
            "1 edit in 5 chars = 0.8 similarity"


class TestDamerauLevenshtein:
    """Tests for Damerau-Levenshtein distance."""

    def test_transposition(self):
        # Damerau counts transposition as 1 edit
        assert fr.damerau_levenshtein("ab", "ba") == 1
        assert fr.damerau_levenshtein("ca", "ac") == 1

        # Regular Levenshtein would count this as 2
        assert fr.levenshtein("ab", "ba") == 2

    def test_similarity(self):
        # "hello" vs "ehllo": 1 transposition out of 5 chars = 0.8 similarity
        sim = fr.damerau_levenshtein_similarity("hello", "ehllo")
        assert sim == 0.8, f"Expected 0.8 for single transposition, got {sim}"


class TestDamerauLevenshteinSimilarityCi:
    """Tests for damerau_levenshtein_similarity_ci function."""

    def test_case_insensitive(self):
        """Upper vs lower case should have similarity 1.0."""
        assert fr.damerau_levenshtein_similarity_ci("HELLO", "hello") == 1.0

    def test_mixed_case(self):
        """Mixed case strings should be treated as equal."""
        assert fr.damerau_levenshtein_similarity_ci("HeLLo", "hElLO") == 1.0

    def test_empty_strings(self):
        """Empty strings should have similarity 1.0."""
        assert fr.damerau_levenshtein_similarity_ci("", "") == 1.0

    def test_one_empty(self):
        """Non-empty vs empty string should have similarity 0.0."""
        assert fr.damerau_levenshtein_similarity_ci("hello", "") == 0.0
        assert fr.damerau_levenshtein_similarity_ci("", "hello") == 0.0

    def test_transposition(self):
        """Transposition should be counted correctly."""
        # "ab" and "BA" after case normalization are "ab" and "ba"
        # Transposition is 1 edit, similarity = 1 - 1/2 = 0.5
        sim = fr.damerau_levenshtein_similarity_ci("ab", "BA")
        assert sim == 0.5, f"Expected 0.5 for transposition, got {sim}"


class TestHamming:
    """Tests for Hamming distance."""

    def test_equal_length(self):
        assert fr.hamming("karolin", "kathrin") == 3
        assert fr.hamming("abc", "abc") == 0

    def test_unequal_length_raises(self):
        with pytest.raises(fr.ValidationError):
            fr.hamming("abc", "ab")


class TestHammingVariants:
    """Tests for Hamming distance variants."""

    def test_hamming_padded_equal_length(self):
        """Padded Hamming should work on equal-length strings."""
        assert fr.hamming_distance_padded("abc", "axc") == 1
        assert fr.hamming_distance_padded("abc", "abc") == 0

    def test_hamming_padded_unequal_length(self):
        """Padded Hamming should work on unequal-length strings."""
        # "ab" becomes "ab " when padded to match "abc"
        result = fr.hamming_distance_padded("ab", "abc")
        assert result >= 1

    def test_hamming_similarity_equal_length(self):
        """Hamming similarity should work for equal-length strings."""
        assert fr.hamming_similarity("abc", "abc") == 1.0, "Identical strings should have similarity 1.0"
        # "abc" vs "axc": 1 difference out of 3 = 2/3 similarity
        sim = fr.hamming_similarity("abc", "axc")
        expected = 2.0 / 3.0  # 2 matches out of 3 characters
        assert abs(sim - expected) < 1e-10, f"Expected similarity ~{expected:.4f}, got {sim}"

    def test_hamming_similarity_unequal_length(self):
        """Hamming similarity should raise ValidationError for unequal lengths."""
        with pytest.raises(fr.ValidationError):
            fr.hamming_similarity("abc", "ab")
        with pytest.raises(fr.ValidationError):
            fr.hamming_similarity("ab", "abc")


class TestVeryLongStringProtection:
    """Tests verifying very long strings don't cause DoS."""

    def test_levenshtein_long_strings(self):
        """Test that Levenshtein handles long strings without hanging."""
        long_a = "a" * 1000
        long_b = "b" * 1000
        # Should complete without hanging (space-efficient algorithm)
        result = fr.levenshtein(long_a, long_b)
        assert result == 1000

    def test_damerau_levenshtein_long_strings(self):
        """Test Damerau-Levenshtein with long strings."""
        long_a = "a" * 1000
        long_b = "b" * 1000
        # Should complete without hanging
        result = fr.damerau_levenshtein(long_a, long_b)
        assert isinstance(result, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
