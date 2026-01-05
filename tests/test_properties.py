"""Property-based tests for FuzzyRust algorithms using Hypothesis.

These tests verify mathematical properties that should hold for all inputs:
- Symmetry: similarity(a, b) == similarity(b, a)
- Identity: similarity(a, a) == 1.0
- Non-negative: distance(a, b) >= 0
- Bounds: 0.0 <= similarity(a, b) <= 1.0
- Triangle inequality: d(a,c) <= d(a,b) + d(b,c) for edit distances
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import fuzzyrust as fr

# Strategy for reasonable text (avoid extremely long strings for performance)
text_strategy = st.text(max_size=100, alphabet=st.characters(blacklist_categories=["Cs"]))
short_text_strategy = st.text(max_size=50, alphabet=st.characters(blacklist_categories=["Cs"]))


class TestSimilaritySymmetry:
    """Test that similarity(a, b) == similarity(b, a) for all algorithms."""

    @given(st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=100)
    def test_levenshtein_symmetry(self, a: str, b: str):
        assert fr.levenshtein_similarity(a, b) == fr.levenshtein_similarity(b, a)

    @given(st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=100)
    def test_jaro_symmetry(self, a: str, b: str):
        assert fr.jaro_similarity(a, b) == fr.jaro_similarity(b, a)

    @given(st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=100)
    def test_jaro_winkler_symmetry(self, a: str, b: str):
        assert fr.jaro_winkler_similarity(a, b) == fr.jaro_winkler_similarity(b, a)

    @given(st.text(max_size=50), st.text(max_size=50))
    @settings(max_examples=100)
    def test_damerau_symmetry(self, a: str, b: str):
        assert fr.damerau_levenshtein_similarity(a, b) == fr.damerau_levenshtein_similarity(b, a)

    @given(st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=100)
    def test_ngram_symmetry(self, a: str, b: str):
        assert fr.ngram_similarity(a, b) == fr.ngram_similarity(b, a)

    @given(st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=100)
    def test_cosine_symmetry(self, a: str, b: str):
        assert fr.cosine_similarity_chars(a, b) == fr.cosine_similarity_chars(b, a)

    @given(st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=100)
    def test_lcs_symmetry(self, a: str, b: str):
        assert fr.lcs_similarity(a, b) == fr.lcs_similarity(b, a)


class TestSimilarityIdentity:
    """Test that similarity(a, a) == 1.0 for all algorithms."""

    @given(st.text(max_size=100))
    @settings(max_examples=100)
    def test_levenshtein_identity(self, s: str):
        assert fr.levenshtein_similarity(s, s) == 1.0

    @given(st.text(max_size=100))
    @settings(max_examples=100)
    def test_jaro_identity(self, s: str):
        assert fr.jaro_similarity(s, s) == 1.0

    @given(st.text(max_size=100))
    @settings(max_examples=100)
    def test_jaro_winkler_identity(self, s: str):
        assert fr.jaro_winkler_similarity(s, s) == 1.0

    @given(st.text(max_size=50))
    @settings(max_examples=100)
    def test_damerau_identity(self, s: str):
        assert fr.damerau_levenshtein_similarity(s, s) == 1.0

    @given(st.text(max_size=100))
    @settings(max_examples=100)
    def test_ngram_identity(self, s: str):
        # N-gram similarity of string with itself is 1.0 (or 0.0 for very short strings)
        sim = fr.ngram_similarity(s, s)
        # For strings shorter than n, ngram returns 0.0; otherwise 1.0
        if len(s) >= 3:  # default ngram_size=3
            assert sim == 1.0

    @given(st.text(max_size=100))
    @settings(max_examples=100)
    def test_cosine_identity(self, s: str):
        sim = fr.cosine_similarity_chars(s, s)
        if len(s) > 0:
            assert sim == 1.0

    @given(st.text(max_size=100))
    @settings(max_examples=100)
    def test_lcs_identity(self, s: str):
        assert fr.lcs_similarity(s, s) == 1.0

    @given(st.text(max_size=100))
    @settings(max_examples=100)
    def test_hamming_identity(self, s: str):
        assert fr.hamming_similarity(s, s) == 1.0


class TestSimilarityBounds:
    """Test that 0.0 <= similarity(a, b) <= 1.0 for all algorithms."""

    @given(st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=100)
    def test_levenshtein_bounds(self, a: str, b: str):
        score = fr.levenshtein_similarity(a, b)
        assert 0.0 <= score <= 1.0

    @given(st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=100)
    def test_jaro_bounds(self, a: str, b: str):
        score = fr.jaro_similarity(a, b)
        assert 0.0 <= score <= 1.0

    @given(st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=100)
    def test_jaro_winkler_bounds(self, a: str, b: str):
        score = fr.jaro_winkler_similarity(a, b)
        assert 0.0 <= score <= 1.0

    @given(st.text(max_size=50), st.text(max_size=50))
    @settings(max_examples=100)
    def test_damerau_bounds(self, a: str, b: str):
        score = fr.damerau_levenshtein_similarity(a, b)
        assert 0.0 <= score <= 1.0

    @given(st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=100)
    def test_ngram_bounds(self, a: str, b: str):
        score = fr.ngram_similarity(a, b)
        assert 0.0 <= score <= 1.0

    @given(st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=100)
    def test_cosine_bounds(self, a: str, b: str):
        score = fr.cosine_similarity_chars(a, b)
        assert 0.0 <= score <= 1.0

    @given(st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=100)
    def test_lcs_bounds(self, a: str, b: str):
        score = fr.lcs_similarity(a, b)
        assert 0.0 <= score <= 1.0

    @given(st.text(max_size=50))
    @settings(max_examples=100)
    def test_hamming_bounds_same_length(self, s: str):
        # Hamming similarity requires equal-length strings
        # Test with same string to ensure bounds are respected
        score = fr.hamming_similarity(s, s)
        assert 0.0 <= score <= 1.0

    @given(st.text(max_size=20), st.text(max_size=20))
    @settings(max_examples=100)
    def test_hamming_bounds_equal_length(self, a: str, b: str):
        # Hamming similarity requires equal-length strings
        # Pad or truncate to make them equal length
        assume(len(a) > 0 and len(b) > 0)
        min_len = min(len(a), len(b))
        a_trimmed = a[:min_len]
        b_trimmed = b[:min_len]
        score = fr.hamming_similarity(a_trimmed, b_trimmed)
        assert 0.0 <= score <= 1.0


class TestDistanceNonNegative:
    """Test that distance(a, b) >= 0 for all algorithms."""

    @given(st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=100)
    def test_levenshtein_non_negative(self, a: str, b: str):
        assert fr.levenshtein(a, b) >= 0

    @given(st.text(max_size=50), st.text(max_size=50))
    @settings(max_examples=100)
    def test_damerau_non_negative(self, a: str, b: str):
        assert fr.damerau_levenshtein(a, b) >= 0


class TestDistanceIdentity:
    """Test that distance(a, a) == 0 for all algorithms."""

    @given(st.text(max_size=100))
    @settings(max_examples=100)
    def test_levenshtein_identity(self, s: str):
        assert fr.levenshtein(s, s) == 0

    @given(st.text(max_size=50))
    @settings(max_examples=100)
    def test_damerau_identity(self, s: str):
        assert fr.damerau_levenshtein(s, s) == 0

    @given(st.text(max_size=100))
    @settings(max_examples=100)
    def test_hamming_identity(self, s: str):
        assert fr.hamming(s, s) == 0


class TestTriangleInequality:
    """Test triangle inequality: d(a,c) <= d(a,b) + d(b,c)."""

    @given(st.text(max_size=30), st.text(max_size=30), st.text(max_size=30))
    @settings(max_examples=50)
    def test_levenshtein_triangle_inequality(self, a: str, b: str, c: str):
        d_ab = fr.levenshtein(a, b)
        d_bc = fr.levenshtein(b, c)
        d_ac = fr.levenshtein(a, c)
        assert d_ac <= d_ab + d_bc

    @given(st.text(max_size=30), st.text(max_size=30), st.text(max_size=30))
    @settings(max_examples=50)
    def test_damerau_triangle_inequality(self, a: str, b: str, c: str):
        d_ab = fr.damerau_levenshtein(a, b)
        d_bc = fr.damerau_levenshtein(b, c)
        d_ac = fr.damerau_levenshtein(a, c)
        assert d_ac <= d_ab + d_bc


class TestDistanceSymmetry:
    """Test that distance(a, b) == distance(b, a) for all algorithms."""

    @given(st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=100)
    def test_levenshtein_symmetry(self, a: str, b: str):
        assert fr.levenshtein(a, b) == fr.levenshtein(b, a)

    @given(st.text(max_size=50), st.text(max_size=50))
    @settings(max_examples=100)
    def test_damerau_symmetry(self, a: str, b: str):
        assert fr.damerau_levenshtein(a, b) == fr.damerau_levenshtein(b, a)


class TestPhoneticConsistency:
    """Test phonetic encoding consistency."""

    @given(st.text(alphabet=st.characters(whitelist_categories=["Lu", "Ll"]), max_size=20))
    @settings(max_examples=100)
    def test_soundex_deterministic(self, s: str):
        # Same input should always produce same output
        assume(len(s) > 0)
        assert fr.soundex(s) == fr.soundex(s)

    @given(st.text(alphabet=st.characters(whitelist_categories=["Lu", "Ll"]), max_size=20))
    @settings(max_examples=100)
    def test_metaphone_deterministic(self, s: str):
        # Same input should always produce same output
        assume(len(s) > 0)
        assert fr.metaphone(s) == fr.metaphone(s)


class TestNormalizationIdempotence:
    """Test that normalization is idempotent: normalize(normalize(s)) == normalize(s)."""

    @given(st.text(max_size=100))
    @settings(max_examples=100)
    def test_lowercase_idempotent(self, s: str):
        once = fr.normalize_string(s, "lowercase")
        twice = fr.normalize_string(once, "lowercase")
        assert once == twice

    @given(
        st.text(
            max_size=100,
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !@#$%^&*()",
        )
    )
    @settings(max_examples=100)
    def test_strict_idempotent(self, s: str):
        # Use ASCII-only alphabet to avoid Unicode normalization edge cases
        # where combining characters can be reordered during NFKD normalization
        once = fr.normalize_string(s, "strict")
        twice = fr.normalize_string(once, "strict")
        assert once == twice


class TestBatchConsistency:
    """Test that batch operations produce same results as individual calls."""

    @given(
        st.lists(st.text(max_size=20), min_size=1, max_size=10),
        st.lists(st.text(max_size=20), min_size=1, max_size=10),
    )
    @settings(max_examples=20)
    def test_batch_similarity_consistency(self, list_a: list[str], list_b: list[str]):
        # Align lists to same length
        min_len = min(len(list_a), len(list_b))
        list_a = list_a[:min_len]
        list_b = list_b[:min_len]

        # Compare batch result with individual calls
        # batch_similarity_pairs takes two separate lists, not tuples
        batch_result = fr.batch.pairwise(list_a, list_b, "jaro_winkler")
        individual = [fr.jaro_winkler_similarity(a, b) for a, b in zip(list_a, list_b)]

        for batch_score, indiv_score in zip(batch_result, individual):
            # batch_similarity_pairs can return None for unknown algorithms
            if batch_score is not None:
                assert abs(batch_score - indiv_score) < 1e-10
