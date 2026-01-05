"""Tests for string normalization functions and parameters.

This module tests the normalization modes provided by fuzzyrust,
including lowercase, strict, remove_punctuation, remove_whitespace, and unicode_nfkd.
"""

import pytest

import fuzzyrust as fr


class TestNormalizeParameter:
    """Tests for the new normalize parameter on similarity functions."""

    def test_levenshtein_normalize_lowercase(self):
        """levenshtein with normalize='lowercase' should be case-insensitive."""
        # Without normalization
        assert fr.levenshtein("Hello", "HELLO") == 4  # 4 substitutions
        # With lowercase normalization
        assert fr.levenshtein("Hello", "HELLO", normalize="lowercase") == 0

    def test_levenshtein_similarity_normalize(self):
        """levenshtein_similarity with normalize should work."""
        assert fr.levenshtein_similarity("Hello", "HELLO", normalize="lowercase") == 1.0
        assert fr.levenshtein_similarity("Hello", "HELLO") < 1.0  # Without normalization

    def test_damerau_levenshtein_normalize(self):
        """damerau_levenshtein with normalize should work."""
        assert fr.damerau_levenshtein("Hello", "HELLO", normalize="lowercase") == 0
        assert fr.damerau_levenshtein("Hello", "HELLO") > 0

    def test_damerau_levenshtein_similarity_normalize(self):
        """damerau_levenshtein_similarity with normalize should work."""
        assert fr.damerau_levenshtein_similarity("Hello", "HELLO", normalize="lowercase") == 1.0

    def test_jaro_similarity_normalize(self):
        """jaro_similarity with normalize should work."""
        assert fr.jaro_similarity("MARTHA", "martha", normalize="lowercase") == 1.0

    def test_jaro_winkler_similarity_normalize(self):
        """jaro_winkler_similarity with normalize should work."""
        assert fr.jaro_winkler_similarity("Hello", "HELLO", normalize="lowercase") == 1.0
        # Works with other parameters too
        assert (
            fr.jaro_winkler_similarity("Hello", "HELLO", prefix_weight=0.1, normalize="lowercase")
            == 1.0
        )

    def test_ngram_similarity_normalize(self):
        """ngram_similarity with normalize should work."""
        assert fr.ngram_similarity("Hello", "HELLO", normalize="lowercase") == 1.0
        # Works with other parameters too
        assert fr.ngram_similarity("Hello", "HELLO", ngram_size=3, normalize="lowercase") == 1.0

    def test_ngram_jaccard_normalize(self):
        """ngram_jaccard with normalize should work."""
        assert fr.ngram_jaccard("Hello", "HELLO", normalize="lowercase") == 1.0

    def test_cosine_similarity_chars_normalize(self):
        """cosine_similarity_chars with normalize should work."""
        assert fr.cosine_similarity_chars("ABC", "abc", normalize="lowercase") == 1.0

    def test_cosine_similarity_words_normalize(self):
        """cosine_similarity_words with normalize should work."""
        assert (
            fr.cosine_similarity_words("HELLO WORLD", "hello world", normalize="lowercase") == 1.0
        )

    def test_cosine_similarity_ngrams_normalize(self):
        """cosine_similarity_ngrams with normalize should work."""
        assert fr.cosine_similarity_ngrams("Hello", "HELLO", normalize="lowercase") == 1.0

    def test_normalize_strict_mode(self):
        """Test strict normalization mode (combines all normalizations)."""
        # Strict mode: lowercase + remove punctuation + remove whitespace + NFKD
        assert (
            fr.levenshtein_similarity("  Hello, World!  ", "helloworld", normalize="strict") == 1.0
        )

    def test_normalize_remove_punctuation(self):
        """Test remove_punctuation normalization mode."""
        # Removes punctuation but keeps case and whitespace
        assert (
            fr.levenshtein_similarity(
                "Hello, World!", "Hello World", normalize="remove_punctuation"
            )
            == 1.0
        )

    def test_normalize_remove_whitespace(self):
        """Test remove_whitespace normalization mode."""
        assert (
            fr.levenshtein_similarity("hello world", "helloworld", normalize="remove_whitespace")
            == 1.0
        )

    def test_normalize_invalid_mode_raises_error(self):
        """Invalid normalize mode should raise ValidationError."""
        with pytest.raises(fr.ValidationError, match="Unknown normalization mode"):
            fr.levenshtein_similarity("hello", "world", normalize="invalid_mode")

    def test_normalize_lowercase_gives_expected_results(self):
        """normalize='lowercase' should make comparisons case-insensitive."""
        # levenshtein_similarity with normalize="lowercase" treats differently-cased strings as identical
        assert fr.levenshtein_similarity("Hello", "HELLO", normalize="lowercase") == 1.0

        # jaro_winkler_similarity with normalize="lowercase" treats differently-cased strings as identical
        assert fr.jaro_winkler_similarity("Hello", "HELLO", normalize="lowercase") == 1.0

        # ngram_similarity with normalize="lowercase" treats differently-cased strings as identical
        assert fr.ngram_similarity("Hello", "HELLO", normalize="lowercase") == 1.0

    def test_normalize_none_is_default(self):
        """normalize=None should give same results as no normalization."""
        # Explicit None should be the same as not passing the parameter
        assert fr.levenshtein_similarity(
            "Hello", "HELLO", normalize=None
        ) == fr.levenshtein_similarity("Hello", "HELLO")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
