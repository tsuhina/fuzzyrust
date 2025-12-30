"""Tests for N-gram, bigram, trigram, and cosine similarity algorithms.

This module tests the n-gram-based similarity functions provided by fuzzyrust,
including Dice coefficient, Jaccard index, cosine similarity, and TF-IDF.
"""

import pytest

import fuzzyrust as fr


class TestNgram:
    """Tests for N-gram similarity."""

    def test_extract_ngrams(self):
        ngrams = fr.extract_ngrams("abc", ngram_size=2, pad=False)
        # Use set comparison to avoid order dependency
        assert set(ngrams) == {"ab", "bc"}
        assert len(ngrams) == 2

        ngrams_padded = fr.extract_ngrams("abc", ngram_size=2, pad=True)
        assert len(ngrams_padded) == 4  # " a", "ab", "bc", "c "

    def test_similarity(self):
        assert fr.ngram_similarity("abc", "abc") == 1.0
        assert fr.ngram_similarity("abc", "xyz") == 0.0

        # Partial similarity - "night" and "nacht" with bigram padding (ngram_size=2):
        # " n", "ni", "ig", "gh", "ht", "t " vs " n", "na", "ac", "ch", "ht", "t "
        # Intersection: {" n", "ht", "t "} = 3, Total: 6+6=12
        # Sorensen-Dice = 2*3/12 = 0.5
        sim = fr.ngram_similarity("night", "nacht", ngram_size=2)
        assert 0.49 < sim < 0.51


class TestNgramEdgeCases:
    """Tests for n-gram edge cases including validation."""

    def test_ngram_n_zero_raises_error(self):
        """Test that ngram_size=0 raises ValidationError."""
        with pytest.raises(fr.ValidationError, match="ngram_size must be at least 1"):
            fr.ngram_similarity("hello", "hello", ngram_size=0)
        with pytest.raises(fr.ValidationError, match="ngram_size must be at least 1"):
            fr.ngram_similarity("abc", "xyz", ngram_size=0)

    def test_ngram_short_strings(self):
        """Test n-gram behavior with strings shorter than ngram_size."""
        # Short strings without padding return empty n-grams
        ngrams = fr.extract_ngrams("a", ngram_size=3, pad=False)
        assert ngrams == []

    def test_ngram_jaccard_n_zero_raises_error(self):
        """Test Jaccard similarity with ngram_size=0 raises ValidationError."""
        with pytest.raises(fr.ValidationError, match="ngram_size must be at least 1"):
            fr.ngram_jaccard("hello", "hello", ngram_size=0)

    def test_cosine_ngrams_n_zero_raises_error(self):
        """Test cosine n-gram similarity with ngram_size=0 raises ValidationError."""
        with pytest.raises(fr.ValidationError, match="ngram_size must be at least 1"):
            fr.cosine_similarity_ngrams("hello", "hello", ngram_size=0)

    def test_extract_ngrams_n_zero_raises_error(self):
        """Test extract_ngrams with ngram_size=0 raises ValidationError."""
        with pytest.raises(fr.ValidationError, match="ngram_size must be at least 1"):
            fr.extract_ngrams("hello", ngram_size=0)

    def test_ngram_profile_n_zero_raises_error(self):
        """Test ngram_profile_similarity with ngram_size=0 raises ValidationError."""
        with pytest.raises(fr.ValidationError, match="ngram_size must be at least 1"):
            fr.ngram_profile_similarity("hello", "hello", ngram_size=0)

    def test_ngram_ci_variants_n_zero_raises_error(self):
        """Test case-insensitive ngram functions with ngram_size=0 raise ValidationError."""
        with pytest.raises(fr.ValidationError, match="ngram_size must be at least 1"):
            fr.ngram_similarity_ci("hello", "HELLO", ngram_size=0)
        with pytest.raises(fr.ValidationError, match="ngram_size must be at least 1"):
            fr.ngram_jaccard_ci("hello", "HELLO", ngram_size=0)
        with pytest.raises(fr.ValidationError, match="ngram_size must be at least 1"):
            fr.cosine_similarity_ngrams_ci("hello", "HELLO", ngram_size=0)


class TestNgramConvenience:
    """Tests for n-gram convenience functions."""

    def test_bigram_similarity_identical(self):
        """Identical strings should have similarity 1.0."""
        assert fr.bigram_similarity("hello", "hello") == 1.0

    def test_bigram_similarity_similar(self):
        """Similar strings should have high similarity."""
        sim = fr.bigram_similarity("hello", "hallo")
        assert 0.0 < sim < 1.0

    def test_trigram_similarity(self):
        """Trigram similarity should work correctly."""
        sim = fr.trigram_similarity("hello", "hallo")
        assert 0.0 < sim < 1.0
        # Trigram is typically stricter than bigram
        bigram = fr.bigram_similarity("hello", "hallo")
        assert sim <= bigram or abs(sim - bigram) < 0.2

    def test_ngram_profile_similarity_identical(self):
        """Identical strings should have profile similarity 1.0."""
        assert fr.ngram_profile_similarity("abab", "abab", 2) == 1.0

    def test_ngram_profile_counts_frequency(self):
        """Profile similarity should account for n-gram frequency."""
        # Profile similarity uses frequency counts
        sim = fr.ngram_profile_similarity("aaa", "aa", 2)
        assert 0.0 < sim < 1.0


class TestBigramSimilarityCi:
    """Tests for bigram_similarity_ci function."""

    def test_case_insensitive(self):
        """Upper vs lower case should have similarity 1.0."""
        assert fr.bigram_similarity_ci("HELLO", "hello") == 1.0

    def test_mixed_case(self):
        """Mixed case strings should be treated as equal."""
        assert fr.bigram_similarity_ci("HeLLo", "hElLO") == 1.0

    def test_empty_strings(self):
        """Empty strings should have similarity 1.0."""
        assert fr.bigram_similarity_ci("", "") == 1.0

    def test_one_empty(self):
        """Non-empty vs empty string should have similarity 0.0."""
        assert fr.bigram_similarity_ci("hello", "") == 0.0
        assert fr.bigram_similarity_ci("", "hello") == 0.0

    def test_similar_strings(self):
        """Similar strings should have reasonable similarity."""
        sim = fr.bigram_similarity_ci("HELLO", "hallo")
        assert 0.0 < sim < 1.0


class TestTrigramSimilarityCi:
    """Tests for trigram_similarity_ci function."""

    def test_case_insensitive(self):
        """Upper vs lower case should have similarity 1.0."""
        assert fr.trigram_similarity_ci("HELLO", "hello") == 1.0

    def test_mixed_case(self):
        """Mixed case strings should be treated as equal."""
        assert fr.trigram_similarity_ci("HeLLo", "hElLO") == 1.0

    def test_empty_strings(self):
        """Empty strings should have similarity 1.0."""
        assert fr.trigram_similarity_ci("", "") == 1.0

    def test_one_empty(self):
        """Non-empty vs empty string should have similarity 0.0."""
        assert fr.trigram_similarity_ci("hello", "") == 0.0
        assert fr.trigram_similarity_ci("", "hello") == 0.0

    def test_similar_strings(self):
        """Similar strings should have reasonable similarity."""
        sim = fr.trigram_similarity_ci("HELLO", "hallo")
        assert 0.0 < sim < 1.0


class TestCosine:
    """Tests for Cosine similarity."""

    def test_cosine_chars(self):
        assert fr.cosine_similarity_chars("abc", "abc") == 1.0
        assert fr.cosine_similarity_chars("abc", "def") == 0.0

    def test_cosine_words(self):
        a = "the quick brown fox"
        b = "the quick brown dog"
        # 3 common words (the, quick, brown) out of 4 unique, cosine = 3/4 = 0.75
        sim = fr.cosine_similarity_words(a, b)
        assert sim == 0.75, f"Expected 0.75 for 3 out of 4 common words, got {sim}"


class TestTfIdfCosine:
    """Tests for TF-IDF weighted cosine similarity."""

    def test_tfidf_basic(self):
        """Basic TF-IDF similarity test."""
        tfidf = fr.TfIdfCosine()
        tfidf.add_documents(["hello world", "hello there", "world news"])
        sim = tfidf.similarity("hello world", "hello there")
        assert 0.0 < sim < 1.0

    def test_tfidf_identical(self):
        """Identical strings should have similarity 1.0."""
        tfidf = fr.TfIdfCosine()
        tfidf.add_document("test document")
        assert tfidf.similarity("hello", "hello") == 1.0

    def test_tfidf_different(self):
        """Completely different strings should have low similarity."""
        tfidf = fr.TfIdfCosine()
        tfidf.add_documents(["apple banana", "cherry grape"])
        sim = tfidf.similarity("apple", "cherry")
        assert sim == 0.0  # No common words

    def test_tfidf_num_documents(self):
        """num_documents should return correct count."""
        tfidf = fr.TfIdfCosine()
        assert tfidf.num_documents() == 0
        tfidf.add_document("first")
        assert tfidf.num_documents() == 1
        tfidf.add_documents(["second", "third"])
        assert tfidf.num_documents() == 3

    def test_tfidf_rare_words_weighted_higher(self):
        """Rare words should contribute more to similarity than common words."""
        tfidf = fr.TfIdfCosine()
        # "the" appears in most docs, "quantum" appears in one
        tfidf.add_documents(
            ["the quick brown fox", "the lazy dog", "the fast cat", "quantum physics theory"]
        )
        # Comparing strings with common word "the"
        sim_common = tfidf.similarity("the cat", "the dog")
        # Comparing strings with rare word "quantum"
        sim_rare = tfidf.similarity("quantum theory", "quantum physics")
        # Rare words should give higher similarity
        assert sim_rare > sim_common


class TestCaseInsensitiveFunctions:
    """Tests for case-insensitive (_ci) function variants related to n-grams."""

    def test_ngram_jaccard_ci(self):
        """ngram_jaccard_ci should be case-insensitive."""
        # Case-sensitive would give different results
        assert fr.ngram_jaccard_ci("Hello", "hello") == fr.ngram_jaccard("hello", "hello")
        assert fr.ngram_jaccard_ci("WORLD", "world") == 1.0

    def test_cosine_similarity_chars_ci(self):
        """cosine_similarity_chars_ci should be case-insensitive."""
        assert fr.cosine_similarity_chars_ci("ABC", "abc") == 1.0
        assert fr.cosine_similarity_chars_ci("Hello", "HELLO") == 1.0

    def test_cosine_similarity_words_ci(self):
        """cosine_similarity_words_ci should be case-insensitive."""
        assert fr.cosine_similarity_words_ci("The Quick Fox", "the quick fox") == 1.0
        assert fr.cosine_similarity_words_ci("HELLO WORLD", "hello world") == 1.0

    def test_cosine_similarity_ngrams_ci(self):
        """cosine_similarity_ngrams_ci should be case-insensitive."""
        assert fr.cosine_similarity_ngrams_ci("Hello", "HELLO") == 1.0
        assert fr.cosine_similarity_ngrams_ci("WORLD", "world") == 1.0


class TestNgramLongStrings:
    """Tests for n-gram similarity with long strings."""

    def test_ngram_similarity_long_strings(self):
        """Test n-gram similarity with long strings."""
        long_a = "a" * 1000
        long_b = "a" * 1000
        result = fr.ngram_similarity(long_a, long_b, ngram_size=2)
        assert result == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
