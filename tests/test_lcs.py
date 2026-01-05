"""Tests for Longest Common Subsequence and Substring algorithms.

This module tests the LCS-based similarity functions provided by fuzzyrust,
including length, string extraction, and similarity measures.
"""

import pytest

import fuzzyrust as fr


class TestLCS:
    """Tests for Longest Common Subsequence."""

    def test_lcs_length(self):
        assert fr.lcs_length("ABCDGH", "AEDFHR") == 3  # ADH
        assert fr.lcs_length("AGGTAB", "GXTXAYB") == 4  # GTAB

    def test_lcs_string(self):
        assert fr.lcs_string("ABCDGH", "AEDFHR") == "ADH"

    def test_longest_common_substring(self):
        assert fr.longest_common_substring("abcdef", "zbcdf") == "bcd"
        assert fr.longest_common_substring_length("abcdef", "zbcdf") == 3


class TestLCSEdgeCases:
    """Tests for LCS edge cases."""

    def test_lcs_empty_strings(self):
        """Test LCS with empty strings."""
        assert fr.lcs_length("", "") == 0
        assert fr.lcs_length("abc", "") == 0
        assert fr.lcs_length("", "abc") == 0
        assert fr.lcs_string("", "abc") == ""

    def test_lcs_no_common(self):
        """Test LCS with no common subsequence."""
        assert fr.lcs_length("abc", "xyz") == 0
        assert fr.lcs_string("abc", "xyz") == ""

    def test_longest_common_substring_empty(self):
        """Test longest common substring with empty strings."""
        assert fr.longest_common_substring_length("", "") == 0
        assert fr.longest_common_substring("abc", "") == ""


class TestLcsAlternative:
    """Tests for LCS alternative metrics."""

    def test_lcs_similarity_max_identical(self):
        """Identical strings should have similarity 1.0."""
        assert fr.lcs_similarity_max("abc", "abc") == 1.0

    def test_lcs_similarity_max_partial(self):
        """Partial matches should have proportional similarity."""
        # LCS of "abc" and "ab" is "ab" (length 2)
        # lcs_similarity_max = 2 / max(3, 2) = 2/3
        sim = fr.lcs_similarity_max("abc", "ab")
        assert abs(sim - 2 / 3) < 0.01

    def test_lcs_similarity_max_different(self):
        """Completely different strings should have similarity 0.0."""
        assert fr.lcs_similarity_max("abc", "xyz") == 0.0


class TestLcsSimilarityWithLowercase:
    """Tests for lcs_similarity with manual lowercasing."""

    def test_case_insensitive(self):
        """Upper vs lower case should have similarity 1.0 after lowercasing."""
        a, b = "HELLO", "hello"
        assert fr.lcs_similarity(a.lower(), b.lower()) == 1.0

    def test_mixed_case(self):
        """Mixed case strings should be treated as equal after lowercasing."""
        a, b = "HeLLo", "hElLO"
        assert fr.lcs_similarity(a.lower(), b.lower()) == 1.0

    def test_empty_strings(self):
        """Empty strings should have similarity 1.0."""
        a, b = "", ""
        assert fr.lcs_similarity(a.lower(), b.lower()) == 1.0

    def test_one_empty(self):
        """Non-empty vs empty string should have similarity 0.0."""
        assert fr.lcs_similarity("hello".lower(), "".lower()) == 0.0
        assert fr.lcs_similarity("".lower(), "hello".lower()) == 0.0

    def test_partial_match(self):
        """Partial matches should have appropriate similarity."""
        a, b = "ABCD", "axbxcxd"
        sim = fr.lcs_similarity(a.lower(), b.lower())
        assert 0.0 < sim < 1.0


class TestLCSLongStrings:
    """Tests for LCS with long strings."""

    def test_lcs_length_long_strings(self):
        """Test LCS length with long strings (space-efficient)."""
        long_a = "a" * 1000
        long_b = "a" * 500 + "b" * 500
        result = fr.lcs_length(long_a, long_b)
        assert result == 500

    def test_lcs_string_very_long_raises_error(self):
        """Test lcs_string raises ValidationError for very long strings to prevent DoS."""
        # Strings longer than 10000 should raise ValidationError (protection)
        long_a = "a" * 15000
        long_b = "a" * 15000
        with pytest.raises(fr.ValidationError):
            fr.lcs_string(long_a, long_b)

    def test_longest_common_substring_very_long_raises_error(self):
        """Test longest_common_substring raises ValidationError for very long strings."""
        long_a = "a" * 15000
        long_b = "a" * 15000
        with pytest.raises(fr.ValidationError):
            fr.longest_common_substring(long_a, long_b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
