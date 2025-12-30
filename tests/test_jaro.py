"""Tests for Jaro and Jaro-Winkler similarity algorithms.

This module tests the Jaro and Jaro-Winkler similarity functions provided by fuzzyrust,
including parameter validation and case-insensitive variants.
"""

import pytest

import fuzzyrust as fr


class TestJaro:
    """Tests for Jaro and Jaro-Winkler similarity."""

    def test_jaro_identical(self):
        assert fr.jaro_similarity("hello", "hello") == 1.0
        assert fr.jaro_similarity("", "") == 1.0

    def test_jaro_different(self):
        assert fr.jaro_similarity("abc", "xyz") == 0.0

    def test_jaro_classic_examples(self):
        # Classic MARTHA/MARHTA example
        sim = fr.jaro_similarity("MARTHA", "MARHTA")
        assert 0.94 < sim < 0.95

    def test_jaro_winkler_prefix_boost(self):
        jaro = fr.jaro_similarity("MARTHA", "MARHTA")
        jaro_winkler = fr.jaro_winkler_similarity("MARTHA", "MARHTA")
        # Jaro-Winkler should be higher due to common prefix
        assert jaro_winkler > jaro

    def test_jaro_winkler_params(self):
        # Higher prefix weight should increase similarity for common prefixes
        default = fr.jaro_winkler_similarity("prefix_test", "prefix_best")
        higher = fr.jaro_winkler_similarity("prefix_test", "prefix_best", prefix_weight=0.2)
        assert higher >= default


class TestJaroWinklerValidation:
    """Tests for Jaro-Winkler prefix_weight validation."""

    def test_prefix_weight_too_high_raises_error(self):
        """Test that prefix_weight > 0.25 raises ValidationError."""
        # Prefix weight > 0.25 should raise ValidationError
        with pytest.raises(fr.ValidationError, match="prefix_weight must be in range"):
            fr.jaro_winkler_similarity("hello", "hello", prefix_weight=1.0)
        with pytest.raises(fr.ValidationError, match="prefix_weight must be in range"):
            fr.jaro_winkler_similarity("hello", "hello", prefix_weight=0.26)

    def test_prefix_weight_negative_raises_error(self):
        """Test that negative prefix_weight raises ValidationError."""
        # Negative weight should raise ValidationError
        with pytest.raises(fr.ValidationError, match="prefix_weight must be in range"):
            fr.jaro_winkler_similarity("hello", "hallo", prefix_weight=-1.0)
        with pytest.raises(fr.ValidationError, match="prefix_weight must be in range"):
            fr.jaro_winkler_similarity("hello", "hallo", prefix_weight=-0.01)

    def test_prefix_weight_valid_range(self):
        """Test prefix_weight within valid range produces expected boost."""
        jaro = fr.jaro_similarity("prefix_test", "prefix_best")
        jaro_winkler = fr.jaro_winkler_similarity("prefix_test", "prefix_best", prefix_weight=0.1)
        # With common prefix, Jaro-Winkler should be higher
        assert jaro_winkler >= jaro

    def test_prefix_weight_boundary_values(self):
        """Test prefix_weight at boundary values (0 and 0.25)."""
        # With prefix_weight=0.0, Jaro-Winkler equals Jaro for identical strings (1.0)
        sim_zero = fr.jaro_winkler_similarity("hello", "hello", prefix_weight=0.0)
        assert sim_zero == 1.0, f"Identical strings should have similarity 1.0, got {sim_zero}"
        # With prefix_weight=0.25 (max allowed), identical strings still have similarity 1.0
        sim_max = fr.jaro_winkler_similarity("hello", "hello", prefix_weight=0.25)
        assert sim_max == 1.0, f"Identical strings should have similarity 1.0, got {sim_max}"


class TestJaroLongStrings:
    """Tests for Jaro similarity with long strings."""

    def test_jaro_similarity_long_strings(self):
        """Test Jaro similarity with long strings."""
        long_a = "a" * 1000
        long_b = "b" * 1000
        result = fr.jaro_similarity(long_a, long_b)
        assert isinstance(result, float), f"Expected float, got {type(result).__name__}"
        # "aaa..." and "bbb..." have no matching characters, so Jaro similarity is 0.0
        assert result == 0.0, f"Expected Jaro similarity of 0.0 for completely different strings, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
