"""Tests for phonetic algorithms: Soundex, Metaphone, Double Metaphone.

This module tests the phonetic encoding and similarity functions provided by fuzzyrust,
including their case-insensitive variants.
"""

import pytest

import fuzzyrust as fr


class TestPhonetic:
    """Tests for phonetic algorithms."""

    def test_soundex(self):
        assert fr.soundex("Robert") == "R163"
        assert fr.soundex("Rupert") == "R163"

    def test_soundex_match(self):
        assert fr.soundex_match("Robert", "Rupert")
        assert fr.soundex_match("Smith", "Smyth")
        assert not fr.soundex_match("Robert", "Rubin")

    def test_metaphone(self):
        assert fr.metaphone("phone") == "FN"

    def test_metaphone_match(self):
        assert fr.metaphone_match("phone", "fone")
        assert fr.metaphone_match("Stephen", "Steven")


class TestPhoneticEdgeCases:
    """Edge case tests for phonetic algorithms."""

    def test_soundex_empty_string(self):
        """Test Soundex with empty string."""
        result = fr.soundex("")
        # Empty string should return empty or default code
        assert isinstance(result, str)

    def test_soundex_single_char(self):
        """Test Soundex with single character."""
        result = fr.soundex("A")
        assert isinstance(result, str)
        assert len(result) == 4  # Soundex always returns 4 chars
        assert result.startswith("A")

    def test_metaphone_empty_string(self):
        """Test Metaphone with empty string."""
        result = fr.metaphone("")
        assert result == ""

    def test_metaphone_single_char(self):
        """Test Metaphone with single character."""
        result = fr.metaphone("A")
        assert isinstance(result, str)

    def test_soundex_numbers_only(self):
        """Test Soundex with numbers only."""
        result = fr.soundex("123")
        assert isinstance(result, str)

    def test_metaphone_unicode(self):
        """Test Metaphone with Unicode characters."""
        result = fr.metaphone("cafe")
        assert isinstance(result, str)


class TestPhoneticSimilarity:
    """Tests for phonetic similarity functions."""

    def test_soundex_similarity_identical_codes(self):
        """Same Soundex codes should have similarity 1.0."""
        # Robert and Rupert both have Soundex code R163
        assert fr.soundex_similarity("Robert", "Rupert") == 1.0

    def test_soundex_similarity_partial(self):
        """Different Soundex codes should have partial similarity."""
        sim = fr.soundex_similarity("Robert", "Richard")
        assert 0.0 < sim < 1.0

    def test_soundex_similarity_different(self):
        """Completely different names should have low similarity."""
        sim = fr.soundex_similarity("Smith", "Johnson")
        assert sim < 0.5

    def test_metaphone_similarity_identical(self):
        """Same Metaphone codes should have similarity 1.0."""
        # Stephen and Steven have same Metaphone code
        assert fr.metaphone_similarity("Stephen", "Steven") == 1.0

    def test_metaphone_similarity_partial(self):
        """Similar sounds should have high but not perfect similarity."""
        sim = fr.metaphone_similarity("John", "Jon")
        assert sim > 0.8


class TestSoundexSimilarityCi:
    """Tests for soundex_similarity_ci function."""

    def test_case_insensitive(self):
        """Upper vs lower case should have similarity 1.0."""
        assert fr.soundex_similarity_ci("ROBERT", "robert") == 1.0

    def test_mixed_case(self):
        """Mixed case strings should be treated as equal."""
        assert fr.soundex_similarity_ci("RoBeRt", "rObErT") == 1.0

    def test_empty_strings(self):
        """Empty strings return 0.0 for phonetic algorithms (no phonetic content)."""
        # Phonetic algorithms return 0.0 for empty strings (no phonetic content to compare)
        assert fr.soundex_similarity_ci("", "") == 0.0

    def test_one_empty(self):
        """Non-empty vs empty string should have similarity 0.0."""
        assert fr.soundex_similarity_ci("hello", "") == 0.0
        assert fr.soundex_similarity_ci("", "hello") == 0.0

    def test_same_soundex_different_case(self):
        """Words with same Soundex code should have high similarity."""
        # Robert and Rupert have same Soundex code R163
        assert fr.soundex_similarity_ci("ROBERT", "rupert") == 1.0


class TestMetaphoneSimilarityCi:
    """Tests for metaphone_similarity_ci function."""

    def test_case_insensitive(self):
        """Upper vs lower case should have similarity 1.0."""
        assert fr.metaphone_similarity_ci("PHONE", "phone") == 1.0

    def test_mixed_case(self):
        """Mixed case strings should be treated as equal."""
        assert fr.metaphone_similarity_ci("PhOnE", "pHoNe") == 1.0

    def test_empty_strings(self):
        """Empty strings return 0.0 for phonetic algorithms (no phonetic content)."""
        # Phonetic algorithms return 0.0 for empty strings (no phonetic content to compare)
        assert fr.metaphone_similarity_ci("", "") == 0.0

    def test_one_empty(self):
        """Non-empty vs empty string should have similarity 0.0."""
        assert fr.metaphone_similarity_ci("hello", "") == 0.0
        assert fr.metaphone_similarity_ci("", "hello") == 0.0

    def test_same_metaphone_different_case(self):
        """Words with same Metaphone code should have similarity 1.0."""
        # Stephen and Steven have same Metaphone code
        assert fr.metaphone_similarity_ci("STEPHEN", "steven") == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
