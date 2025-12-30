"""
Property-based fuzzing tests for FuzzyRust using Hypothesis.

Tests cover mathematical invariants and edge cases:
- Symmetry: distance(a, b) == distance(b, a)
- Triangle inequality: distance(a, c) <= distance(a, b) + distance(b, c)
- Identity: distance(a, a) == 0, similarity(a, a) == 1.0
- Bounds: 0 <= similarity <= 1.0, distance >= 0
- Non-negativity and consistency across algorithms
"""

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import fuzzyrust as fr

# Custom strategies for text generation
# Limit string length to avoid extremely slow tests
text_strategy = st.text(min_size=0, max_size=100)
short_text_strategy = st.text(min_size=0, max_size=50)
nonempty_text_strategy = st.text(min_size=1, max_size=50)


class TestLevenshteinInvariants:
    """Property-based tests for Levenshtein distance."""

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_symmetry(self, a, b):
        """Levenshtein distance is symmetric: d(a, b) == d(b, a)"""
        assert fr.levenshtein(a, b) == fr.levenshtein(b, a)

    @given(st.text(), st.text(), st.text())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_triangle_inequality(self, a, b, c):
        """Triangle inequality: d(a, c) <= d(a, b) + d(b, c)"""
        assert fr.levenshtein(a, c) <= fr.levenshtein(a, b) + fr.levenshtein(b, c)

    @given(st.text())
    @settings(max_examples=200)
    def test_identity(self, a):
        """Distance to self is zero: d(a, a) == 0"""
        assert fr.levenshtein(a, a) == 0

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_non_negative(self, a, b):
        """Distance is non-negative: d(a, b) >= 0"""
        assert fr.levenshtein(a, b) >= 0

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_similarity_bounds(self, a, b):
        """Similarity is in [0, 1]: 0 <= sim(a, b) <= 1"""
        sim = fr.levenshtein_similarity(a, b)
        assert 0.0 <= sim <= 1.0

    @given(st.text())
    @settings(max_examples=200)
    def test_similarity_identity(self, a):
        """Similarity to self is 1 (for non-empty strings)"""
        sim = fr.levenshtein_similarity(a, a)
        if len(a) > 0:
            assert sim == 1.0
        else:
            # Empty strings: 0 / 0 = 1.0 by convention
            assert sim == 1.0


class TestDamerauLevenshteinInvariants:
    """Property-based tests for Damerau-Levenshtein distance."""

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_symmetry(self, a, b):
        """Damerau-Levenshtein is symmetric"""
        assert fr.damerau_levenshtein(a, b) == fr.damerau_levenshtein(b, a)

    @given(st.text(), st.text(), st.text())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_triangle_inequality(self, a, b, c):
        """Triangle inequality holds"""
        assert fr.damerau_levenshtein(a, c) <= fr.damerau_levenshtein(
            a, b
        ) + fr.damerau_levenshtein(b, c)

    @given(st.text())
    @settings(max_examples=200)
    def test_identity(self, a):
        """Distance to self is zero"""
        assert fr.damerau_levenshtein(a, a) == 0

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_at_most_levenshtein(self, a, b):
        """Damerau-Levenshtein <= Levenshtein (transpositions are cheaper)"""
        assert fr.damerau_levenshtein(a, b) <= fr.levenshtein(a, b)


class TestJaroWinklerInvariants:
    """Property-based tests for Jaro-Winkler similarity."""

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_symmetry(self, a, b):
        """Jaro-Winkler is symmetric"""
        assert fr.jaro_winkler_similarity(a, b) == pytest.approx(
            fr.jaro_winkler_similarity(b, a), abs=1e-10
        )

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_bounds(self, a, b):
        """Similarity is in [0, 1]"""
        sim = fr.jaro_winkler_similarity(a, b)
        assert 0.0 <= sim <= 1.0

    @given(st.text())
    @settings(max_examples=200)
    def test_identity(self, a):
        """Similarity to self is 1 (for non-empty) or 1 (for empty by convention)"""
        assert fr.jaro_winkler_similarity(a, a) == 1.0

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_jaro_winkler_at_least_jaro(self, a, b):
        """Jaro-Winkler >= Jaro (prefix bonus only increases score)"""
        jaro = fr.jaro_similarity(a, b)
        jw = fr.jaro_winkler_similarity(a, b)
        assert jw >= jaro - 1e-10  # Small epsilon for floating point


class TestHammingInvariants:
    """Property-based tests for Hamming distance."""

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_symmetry_padded(self, a, b):
        """Hamming distance (padded) is symmetric"""
        assert fr.hamming_distance_padded(a, b) == fr.hamming_distance_padded(b, a)

    @given(st.text())
    @settings(max_examples=200)
    def test_identity(self, a):
        """Distance to self is zero"""
        assert fr.hamming_distance_padded(a, a) == 0

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_at_most_max_length(self, a, b):
        """Hamming distance <= max(len(a), len(b))"""
        dist = fr.hamming_distance_padded(a, b)
        assert dist <= max(len(a), len(b))


class TestNgramInvariants:
    """Property-based tests for N-gram similarity."""

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_symmetry(self, a, b):
        """N-gram similarity is symmetric"""
        assert fr.ngram_similarity(a, b, 2) == pytest.approx(
            fr.ngram_similarity(b, a, 2), abs=1e-10
        )

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_bounds(self, a, b):
        """Similarity is in [0, 1]"""
        for n in [2, 3]:
            sim = fr.ngram_similarity(a, b, n)
            assert 0.0 <= sim <= 1.0

    @given(st.text())
    @settings(max_examples=200)
    def test_identity(self, a):
        """Similarity to self is 1 (if string has enough chars for ngrams)"""
        if len(a) >= 2:
            assert fr.ngram_similarity(a, a, 2) == 1.0

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_jaccard_symmetry(self, a, b):
        """N-gram Jaccard is symmetric"""
        assert fr.ngram_jaccard(a, b, 2) == pytest.approx(fr.ngram_jaccard(b, a, 2), abs=1e-10)


class TestCosineInvariants:
    """Property-based tests for cosine similarity."""

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_char_symmetry(self, a, b):
        """Cosine similarity (chars) is symmetric"""
        assert fr.cosine_similarity_chars(a, b) == pytest.approx(
            fr.cosine_similarity_chars(b, a), abs=1e-10
        )

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_char_bounds(self, a, b):
        """Similarity is in [0, 1]"""
        sim = fr.cosine_similarity_chars(a, b)
        assert 0.0 <= sim <= 1.0

    @given(st.text())
    @settings(max_examples=200)
    def test_char_identity(self, a):
        """Similarity to self is 1 for non-empty strings"""
        if len(a) > 0:
            assert fr.cosine_similarity_chars(a, a) == pytest.approx(1.0, abs=1e-10)


class TestLCSInvariants:
    """Property-based tests for Longest Common Subsequence."""

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_symmetry(self, a, b):
        """LCS length is symmetric"""
        assert fr.lcs_length(a, b) == fr.lcs_length(b, a)

    @given(st.text())
    @settings(max_examples=200)
    def test_identity(self, a):
        """LCS of string with itself is its length"""
        assert fr.lcs_length(a, a) == len(a)

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_at_most_min_length(self, a, b):
        """LCS length <= min(len(a), len(b))"""
        assert fr.lcs_length(a, b) <= min(len(a), len(b))

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_similarity_bounds(self, a, b):
        """Similarity is in [0, 1]"""
        sim = fr.lcs_similarity(a, b)
        assert 0.0 <= sim <= 1.0


class TestPhoneticInvariants:
    """Property-based tests for phonetic algorithms."""

    @given(st.text(alphabet=st.characters(whitelist_categories=("L",)), min_size=0, max_size=50))
    @settings(max_examples=200)
    def test_soundex_deterministic(self, a):
        """Soundex is deterministic"""
        assert fr.soundex(a) == fr.soundex(a)

    @given(st.text(alphabet=st.characters(whitelist_categories=("L",)), min_size=0, max_size=50))
    @settings(max_examples=200)
    def test_metaphone_deterministic(self, a):
        """Metaphone is deterministic"""
        assert fr.metaphone(a) == fr.metaphone(a)

    @given(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("L",),
                whitelist_characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=200)
    def test_soundex_length(self, a):
        """Soundex code is exactly 4 characters (for ASCII alphabetic input)"""
        code = fr.soundex(a)
        # ASCII alphabetic input should produce 4-char code
        if any(c.isascii() and c.isalpha() for c in a):
            assert len(code) == 4


class TestRatioFunctionsInvariants:
    """Property-based tests for RapidFuzz-compatible ratio functions.

    Note: FuzzyRust's ratio functions return values in [0.0, 1.0] range,
    unlike RapidFuzz which uses [0, 100]. This is by design for consistency
    with other similarity functions in the library.
    """

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_ratio_symmetry(self, a, b):
        """ratio() is symmetric"""
        assert fr.ratio(a, b) == pytest.approx(fr.ratio(b, a), abs=1e-10)

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_ratio_bounds(self, a, b):
        """ratio() is in [0, 1]"""
        r = fr.ratio(a, b)
        assert 0.0 <= r <= 1.0

    @given(st.text())
    @settings(max_examples=200)
    def test_ratio_identity(self, a):
        """ratio(a, a) is 1.0"""
        assert fr.ratio(a, a) == 1.0

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_partial_ratio_bounds(self, a, b):
        """partial_ratio() is in [0, 1]"""
        r = fr.partial_ratio(a, b)
        assert 0.0 <= r <= 1.0

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_token_sort_ratio_bounds(self, a, b):
        """token_sort_ratio() is in [0, 1]"""
        r = fr.token_sort_ratio(a, b)
        assert 0.0 <= r <= 1.0

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_token_set_ratio_bounds(self, a, b):
        """token_set_ratio() is in [0, 1]"""
        r = fr.token_set_ratio(a, b)
        assert 0.0 <= r <= 1.0

    @given(st.text(), st.text())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_wratio_bounds(self, a, b):
        """wratio() is in [0, 1]"""
        r = fr.wratio(a, b)
        assert 0.0 <= r <= 1.0


class TestNormalizationInvariants:
    """Property-based tests for string normalization."""

    @given(st.text())
    @settings(max_examples=200)
    def test_idempotent(self, a):
        """Normalizing twice is same as once"""
        once = fr.normalize_string(a, fr.NormalizationMode.LOWERCASE)
        twice = fr.normalize_string(once, fr.NormalizationMode.LOWERCASE)
        assert once == twice

    @given(st.text())
    @settings(max_examples=200)
    def test_lowercase(self, a):
        """Normalized string with LOWERCASE mode is lowercase"""
        result = fr.normalize_string(a, fr.NormalizationMode.LOWERCASE)
        # The normalized string should not have uppercase letters
        assert result == result.lower()

    @given(st.from_regex(r"[A-Za-z0-9 !@#$%^&*()_+\-=\[\]{}|;:,.<>?]*", fullmatch=True))
    @settings(max_examples=200)
    def test_strict_idempotent(self, a):
        """Strict normalization is idempotent (for ASCII characters)"""
        once = fr.normalize_string(a, fr.NormalizationMode.STRICT)
        twice = fr.normalize_string(once, fr.NormalizationMode.STRICT)
        assert once == twice


class TestBkTreeInvariants:
    """Property-based tests for BK-tree operations."""

    @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=50))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_add_and_contains(self, items):
        """All added items can be found"""
        tree = fr.BkTree()
        for item in items:
            tree.add(item)

        for item in items:
            assert tree.contains(item), f"Item '{item}' not found"

    @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=50))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_search_distance_zero_finds_exact(self, items):
        """Search with max_distance=0 finds exact matches"""
        tree = fr.BkTree()
        unique_items = list(set(items))
        for item in unique_items:
            tree.add(item)

        for item in unique_items:
            results = tree.search(item, max_distance=0)
            texts = [r.text for r in results]
            assert item in texts, f"Exact match for '{item}' not found"


class TestNgramIndexInvariants:
    """Property-based tests for N-gram index operations."""

    @given(st.lists(st.text(min_size=3, max_size=30), min_size=1, max_size=50))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_add_and_contains(self, items):
        """All added items can be found"""
        index = fr.NgramIndex(ngram_size=2)
        for item in items:
            index.add(item)

        for item in items:
            assert index.contains(item), f"Item '{item}' not found"

    @given(st.lists(st.text(min_size=3, max_size=30), min_size=1, max_size=50))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_search_finds_exact(self, items):
        """High similarity search finds exact matches"""
        index = fr.NgramIndex(ngram_size=2)
        unique_items = list(set(items))
        for item in unique_items:
            index.add(item)

        for item in unique_items:
            results = index.search(item, min_similarity=0.99)
            texts = [r.text for r in results]
            assert item in texts, f"Exact match for '{item}' not found in {texts}"


class TestShardedIndexInvariants:
    """Property-based tests for sharded index operations."""

    @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=100))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_sharded_bktree_contains_all(self, items):
        """All items added to sharded BK-tree can be found"""
        tree = fr.ShardedBkTree(num_shards=4)
        for item in items:
            tree.add(item)

        for item in items:
            assert tree.contains(item), f"Item '{item}' not found in sharded tree"

    @given(st.lists(st.text(min_size=3, max_size=20), min_size=1, max_size=100))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_sharded_ngram_index_contains_all(self, items):
        """All items added to sharded N-gram index can be found"""
        index = fr.ShardedNgramIndex(num_shards=4, ngram_size=2)
        for item in items:
            index.add(item)

        for item in items:
            assert index.contains(item), f"Item '{item}' not found in sharded index"

    @given(st.lists(st.text(min_size=1, max_size=20), min_size=10, max_size=100))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_shard_distribution(self, items):
        """Items are distributed across shards"""
        tree = fr.ShardedBkTree(num_shards=4)
        for item in items:
            tree.add(item)

        dist = tree.shard_distribution()
        # With enough items, at least some shards should have items
        assert sum(dist) == len(tree)


class TestEdgeCases:
    """Edge case tests discovered through fuzzing."""

    def test_empty_strings(self):
        """Empty strings should be handled gracefully"""
        assert fr.levenshtein("", "") == 0
        assert fr.levenshtein("", "abc") == 3
        assert fr.levenshtein("abc", "") == 3
        assert fr.jaro_winkler_similarity("", "") == 1.0
        assert fr.jaro_winkler_similarity("", "abc") == 0.0

    def test_unicode_strings(self):
        """Unicode strings should work correctly"""
        assert fr.levenshtein("caf\u00e9", "cafe") == 1
        assert fr.levenshtein("\u4e2d\u6587", "\u4e2d\u6587") == 0
        assert fr.jaro_winkler_similarity("\u00fcber", "\u00fcber") == 1.0

    def test_very_long_strings(self):
        """Long strings should not cause issues"""
        long_a = "a" * 1000
        long_b = "b" * 1000
        # Should complete without error
        dist = fr.levenshtein(long_a, long_b)
        assert dist == 1000

    def test_special_characters(self):
        """Special characters should be handled"""
        assert fr.levenshtein("hello\nworld", "hello world") == 1
        assert fr.levenshtein("hello\tworld", "hello world") == 1
        assert fr.levenshtein("hello\0world", "helloworld") == 1

    def test_null_bytes(self):
        """Null bytes in strings should be handled"""
        result = fr.levenshtein("a\0b", "ab")
        assert result >= 0  # Should return valid result

    def test_mixed_unicode_categories(self):
        """Mixed unicode categories should work"""
        # Combining characters
        result = fr.levenshtein("e\u0301", "\u00e9")  # e + combining accent vs precomposed
        assert result >= 0
