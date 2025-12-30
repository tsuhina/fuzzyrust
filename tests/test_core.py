"""Core tests for fuzzyrust: property-based, benchmarks, threading, memory, and edge cases.

This module contains cross-cutting tests that don't fit cleanly into algorithm-specific
test modules:
- Property-based tests using Hypothesis
- Performance benchmarks using pytest-benchmark
- Thread safety tests
- Memory stress tests
- Unicode edge cases
- Error handling tests
- General edge cases

Algorithm-specific tests are now in separate modules:
- test_levenshtein.py: Levenshtein, Damerau-Levenshtein, Hamming
- test_jaro.py: Jaro, Jaro-Winkler
- test_ngram.py: N-gram, bigram, trigram, cosine similarity
- test_phonetic.py: Soundex, Metaphone
- test_lcs.py: Longest Common Subsequence/Substring
- test_batch.py: Batch processing functions
- test_indexes.py: BK-tree, N-gram index, Hybrid index
- test_normalization.py: String normalization modes
- test_fuzz.py: RapidFuzz-compatible functions
- test_schema.py: Schema-based multi-field matching
"""

import concurrent.futures
import string
import threading

import pytest


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_strings(self):
        import fuzzyrust as fr

        assert fr.levenshtein("", "") == 0
        assert fr.jaro_similarity("", "") == 1.0
        assert fr.ngram_similarity("", "") == 1.0

    def test_single_chars(self):
        import fuzzyrust as fr

        assert fr.levenshtein("a", "b") == 1
        assert fr.levenshtein("a", "a") == 0

    def test_moderately_long_strings(self):
        """Test with 1000 character strings - moderate length."""
        import fuzzyrust as fr

        a = "a" * 1000
        b = "a" * 999 + "b"
        assert fr.levenshtein(a, b) == 1

    def test_special_characters(self):
        import fuzzyrust as fr

        assert fr.levenshtein("hello!", "hello?") == 1
        assert fr.levenshtein("a@b.com", "a@b.org") == 3

    def test_case_sensitivity(self):
        import fuzzyrust as fr

        # These algorithms are case-sensitive
        assert fr.levenshtein("Hello", "hello") == 1
        assert fr.jaro_winkler_similarity("Hello", "hello") < 1.0


class TestUnicodeEdgeCases:
    """Tests for Unicode edge cases."""

    def test_emoji(self):
        """Test handling of emoji characters."""
        import fuzzyrust as fr

        # Single emoji - should be treated as one character
        assert fr.levenshtein("\U0001F44B", "\U0001F44B") == 0
        assert fr.levenshtein("\U0001F44B", "\U0001F590") == 1

        # Emoji in strings
        assert fr.levenshtein("hello \U0001F44B", "hello \U0001F44B") == 0
        assert fr.levenshtein("hello \U0001F44B", "hello \U0001F590") == 1
        assert fr.levenshtein("hello \U0001F44B", "hello") == 2  # space + emoji

        # Multiple emoji
        assert fr.levenshtein("\U0001F600\U0001F603\U0001F604", "\U0001F600\U0001F603\U0001F604") == 0
        assert fr.levenshtein("\U0001F600\U0001F603\U0001F604", "\U0001F600\U0001F603\U0001F601") == 1

        # Emoji similarity: 7 chars, 6 match = high jaro-winkler score
        sim = fr.jaro_winkler_similarity("hello \U0001F44B", "hello \U0001F590")
        # Jaro-Winkler for 6/7 matching chars with common prefix "hello "
        assert 0.90 <= sim <= 0.95, f"Expected Jaro-Winkler ~0.93 for 6/7 matching chars, got {sim}"

        # Complex emoji (with skin tone modifiers) - treated as single unit
        # Note: This tests handling of multi-codepoint emoji
        assert fr.levenshtein("\U0001F44B\U0001F3FB", "\U0001F44B\U0001F3FB") == 0

    def test_combining_characters(self):
        """Test handling of combining characters."""
        import fuzzyrust as fr

        # Precomposed vs decomposed forms may differ
        # This tests that we handle multi-byte UTF-8 correctly
        # cafe with accent vs without
        assert fr.levenshtein("caf\u00e9", "cafe") == 1

    def test_cjk_characters(self):
        """Test handling of CJK characters."""
        import fuzzyrust as fr

        # Japanese: nihon vs nippon (3 chars vs 2 chars)
        assert fr.levenshtein("\u65e5\u672c\u8a9e", "\u65e5\u672c") == 1
        # Chinese characters
        assert fr.levenshtein("\u4f60\u597d", "\u4f60\u597d\u5417") == 1

    def test_rtl_text(self):
        """Test handling of right-to-left text."""
        import fuzzyrust as fr

        # Hebrew text: shalom vs shalom
        assert fr.levenshtein("\u05e9\u05dc\u05d5\u05dd", "\u05e9\u05dc\u05d5\u05dd") == 0
        # Arabic text: marhaba vs marhaba with one char different
        assert fr.levenshtein("\u0645\u0631\u062d\u0628\u0627", "\u0645\u0631\u062d\u0628") == 1

    def test_zwj_emoji_sequences(self):
        """Test handling of Zero-Width Joiner emoji sequences."""
        import fuzzyrust as fr

        # Family emoji (man+woman+girl+boy) is a ZWJ sequence
        family = "\U0001F468\u200d\U0001F469\u200d\U0001F467\u200d\U0001F466"
        # Same family should have distance 0
        assert fr.levenshtein(family, family) == 0
        # Different family composition
        couple = "\U0001F468\u200d\U0001F469\u200d\U0001F466"
        # These are different sequences
        assert fr.levenshtein(family, couple) > 0
        # Similarity should still work
        sim = fr.jaro_winkler_similarity(family, family)
        assert sim == 1.0

    def test_regional_indicator_flags(self):
        """Test handling of flag emoji (regional indicator symbols)."""
        import fuzzyrust as fr

        # US flag vs UK flag (both are pairs of regional indicators)
        us_flag = "\U0001F1FA\U0001F1F8"
        uk_flag = "\U0001F1EC\U0001F1E7"
        assert fr.levenshtein(us_flag, us_flag) == 0
        # Different flags should have some distance
        assert fr.levenshtein(us_flag, uk_flag) > 0

    def test_zero_width_characters(self):
        """Test handling of zero-width characters."""
        import fuzzyrust as fr

        # Zero-width space (U+200B)
        zwsp = "\u200b"
        # String with and without ZWSP
        with_zwsp = f"hello{zwsp}world"
        without_zwsp = "helloworld"
        # These differ by one character (the ZWSP)
        assert fr.levenshtein(with_zwsp, without_zwsp) == 1

    def test_unicode_normalization_forms(self):
        """Test handling of different Unicode normalization forms."""
        import unicodedata

        import fuzzyrust as fr

        # e as precomposed (NFC) vs decomposed (NFD)
        nfc = unicodedata.normalize("NFC", "caf\u00e9")
        nfd = unicodedata.normalize("NFD", "caf\u00e9")
        # With unicode_nfkd normalization, these should be identical
        assert fr.levenshtein(nfc, nfd, normalize="unicode_nfkd") == 0
        # Without normalization, NFD has extra combining character (5 vs 4 codepoints)
        # Jaro-Winkler handles this with reasonable similarity (~0.848)
        sim = fr.jaro_winkler_similarity(nfc, nfd)
        assert 0.84 <= sim <= 0.90, f"Expected Jaro-Winkler 0.84-0.90 for NFC vs NFD forms, got {sim}"

    def test_surrogates_and_supplementary_planes(self):
        """Test handling of characters outside BMP (supplementary planes)."""
        import fuzzyrust as fr

        # Mathematical symbols from Plane 1
        math1 = "\U0001D400\U0001D401\U0001D402"  # Mathematical Bold Capital
        math2 = "\U0001D400\U0001D401\U0001D403"  # One different
        assert fr.levenshtein(math1, math1) == 0
        assert fr.levenshtein(math1, math2) == 1
        # Ancient scripts (e.g., Egyptian Hieroglyphs from Plane 1)
        hieroglyph = "\U00013000"
        assert fr.levenshtein(hieroglyph, hieroglyph) == 0

    def test_mixed_scripts(self):
        """Test handling of mixed script strings."""
        import fuzzyrust as fr

        # Mix of Latin, Cyrillic, Greek
        mixed1 = "Hello \u041c\u0438\u0440 \u03b1\u03b2\u03b3"
        mixed2 = "Hello \u041c\u0438\u0440 \u03b1\u03b2\u03b4"
        assert fr.levenshtein(mixed1, mixed2) == 1
        # Similarity: 12/13 chars match with common prefix
        sim = fr.jaro_winkler_similarity(mixed1, mixed2)
        assert 0.94 <= sim <= 0.98, f"Expected Jaro-Winkler ~0.96 for 12/13 matching chars, got {sim}"


class TestErrorHandling:
    """Tests for error handling and invalid inputs."""

    def test_invalid_algorithm_find_best_matches(self):
        """Test that invalid algorithm raises AlgorithmError."""
        import fuzzyrust as fr

        with pytest.raises(fr.AlgorithmError, match="Unknown algorithm"):
            fr.find_best_matches(["hello"], "hello", algorithm="invalid_algo")

    def test_invalid_algorithm_ngram_index_search(self):
        """Test that invalid algorithm in NgramIndex.search raises AlgorithmError."""
        import fuzzyrust as fr

        index = fr.NgramIndex(ngram_size=2)
        index.add("test")
        with pytest.raises(fr.AlgorithmError, match="Unknown algorithm"):
            index.search("test", algorithm="invalid_algo")

    def test_invalid_algorithm_hybrid_index_search(self):
        """Test that invalid algorithm in HybridIndex.search raises AlgorithmError."""
        import fuzzyrust as fr

        index = fr.HybridIndex()
        index.add("test")
        with pytest.raises(fr.AlgorithmError, match="Unknown algorithm"):
            index.search("test", algorithm="invalid_algo")

    def test_hamming_unequal_length(self):
        """Test that Hamming distance raises ValidationError for unequal length strings."""
        import fuzzyrust as fr

        with pytest.raises(fr.ValidationError):
            fr.hamming("abc", "ab")
        with pytest.raises(fr.ValidationError):
            fr.hamming("a", "abc")


class TestCaseInsensitiveVariants:
    """Test cases for new case-insensitive function variants."""

    def test_hamming_ci_basic(self):
        """Test case-insensitive Hamming distance."""
        import fuzzyrust as fr

        # Same strings with different case should have distance 0
        assert fr.hamming_ci("ABC", "abc") == 0
        assert fr.hamming_ci("Hello", "hello") == 0
        assert fr.hamming_ci("HELLO", "HeLLo") == 0

        # Actual differences should be detected
        assert fr.hamming_ci("ABC", "AXC") == 1
        assert fr.hamming_ci("abc", "AXC") == 1

    def test_hamming_ci_unequal_length(self):
        """Test that hamming_ci raises ValidationError for unequal length strings."""
        import fuzzyrust as fr

        with pytest.raises(fr.ValidationError):
            fr.hamming_ci("abc", "ab")
        with pytest.raises(fr.ValidationError):
            fr.hamming_ci("A", "abc")

    def test_hamming_similarity_ci_basic(self):
        """Test case-insensitive Hamming similarity."""
        import fuzzyrust as fr

        # Same strings with different case should have similarity 1.0
        assert fr.hamming_similarity_ci("ABC", "abc") == 1.0
        assert fr.hamming_similarity_ci("Hello", "hello") == 1.0

        # Actual differences should reduce similarity
        sim = fr.hamming_similarity_ci("ABC", "AXC")
        assert 0.6 < sim < 0.7  # 2/3 match

    def test_hamming_similarity_ci_unequal_length(self):
        """Test that hamming_similarity_ci raises ValidationError for unequal length strings."""
        import fuzzyrust as fr

        with pytest.raises(fr.ValidationError):
            fr.hamming_similarity_ci("abc", "ab")

    def test_lcs_length_ci_basic(self):
        """Test case-insensitive LCS length."""
        import fuzzyrust as fr

        # Case-insensitive should find LCS regardless of case
        assert fr.lcs_length_ci("ABCDGH", "aedfhr") == 3  # ADH
        assert fr.lcs_length_ci("ABC", "abc") == 3
        assert fr.lcs_length_ci("Hello", "HELLO") == 5

        # Empty strings
        assert fr.lcs_length_ci("", "") == 0
        assert fr.lcs_length_ci("abc", "") == 0

    def test_lcs_string_ci_basic(self):
        """Test case-insensitive LCS string."""
        import fuzzyrust as fr

        # Results should be lowercase (from lowercase conversion)
        result = fr.lcs_string_ci("ABCDGH", "aedfhr")
        assert result == "adh"

        result = fr.lcs_string_ci("Hello", "HELLO")
        assert result == "hello"

        # Empty strings
        assert fr.lcs_string_ci("", "") == ""
        assert fr.lcs_string_ci("abc", "") == ""

    def test_longest_common_substring_ci_basic(self):
        """Test case-insensitive longest common substring."""
        import fuzzyrust as fr

        # Should find common substring regardless of case
        result = fr.longest_common_substring_ci("ABCDEF", "zbcdf")
        assert result == "bcd"

        result = fr.longest_common_substring_ci("Hello World", "ELLO")
        assert result == "ello"

        # Empty strings
        assert fr.longest_common_substring_ci("", "") == ""
        assert fr.longest_common_substring_ci("abc", "") == ""

    def test_ci_functions_consistency_with_lowercase(self):
        """CI functions should give same result as calling regular function on lowercase strings."""
        import fuzzyrust as fr

        a = "Hello"
        b = "WORLD"

        # Hamming requires equal length
        a_eq = "HELLO"
        b_eq = "world"

        # hamming_ci should equal hamming on lowercase
        assert fr.hamming_ci(a_eq, b_eq) == fr.hamming(a_eq.lower(), b_eq.lower())
        assert fr.hamming_similarity_ci(a_eq, b_eq) == fr.hamming_similarity(a_eq.lower(), b_eq.lower())

        # LCS functions should be consistent
        assert fr.lcs_length_ci(a, b) == fr.lcs_length(a.lower(), b.lower())
        assert fr.lcs_string_ci(a, b) == fr.lcs_string(a.lower(), b.lower())
        assert fr.longest_common_substring_ci(a, b) == fr.longest_common_substring(a.lower(), b.lower())


# =============================================================================
# Property-Based Tests using Hypothesis
# =============================================================================

try:
    from hypothesis import assume, given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Provide stub decorators when hypothesis is not installed

    def given(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def settings(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def assume(condition):
        pass

    class st:
        @staticmethod
        def text(*args, **kwargs):
            return None

        @staticmethod
        def lists(*args, **kwargs):
            return None

        @staticmethod
        def integers(*args, **kwargs):
            return None

        @staticmethod
        def floats(*args, **kwargs):
            return None


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(st.text(alphabet=string.ascii_letters + string.digits, min_size=0, max_size=100))
    def test_levenshtein_identity(self, s):
        """Levenshtein distance from a string to itself is always 0."""
        import fuzzyrust as fr

        assert fr.levenshtein(s, s) == 0

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_levenshtein_symmetry(self, a, b):
        """Levenshtein distance is symmetric: d(a,b) == d(b,a)."""
        import fuzzyrust as fr

        assert fr.levenshtein(a, b) == fr.levenshtein(b, a)

    @given(
        st.text(min_size=0, max_size=30),
        st.text(min_size=0, max_size=30),
        st.text(min_size=0, max_size=30),
    )
    def test_levenshtein_triangle_inequality(self, a, b, c):
        """Levenshtein distance satisfies triangle inequality."""
        import fuzzyrust as fr

        d_ab = fr.levenshtein(a, b)
        d_bc = fr.levenshtein(b, c)
        d_ac = fr.levenshtein(a, c)
        assert d_ac <= d_ab + d_bc

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_jaro_winkler_bounds(self, a, b):
        """Jaro-Winkler similarity is always between 0 and 1."""
        import fuzzyrust as fr

        sim = fr.jaro_winkler_similarity(a, b)
        assert 0.0 <= sim <= 1.0, f"Jaro-Winkler({a!r}, {b!r}) = {sim} is outside [0.0, 1.0]"

    @given(st.text(alphabet=string.ascii_letters, min_size=1, max_size=50))
    def test_jaro_identity(self, s):
        """Jaro similarity of identical non-empty strings is 1.0."""
        import fuzzyrust as fr

        assert fr.jaro_similarity(s, s) == 1.0

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_damerau_levenshtein_symmetry(self, a, b):
        """Damerau-Levenshtein distance is symmetric."""
        import fuzzyrust as fr

        assert fr.damerau_levenshtein(a, b) == fr.damerau_levenshtein(b, a)

    @given(st.text(min_size=0, max_size=50))
    def test_ngram_identity(self, s):
        """N-gram similarity of a string with itself is 1.0."""
        import fuzzyrust as fr

        assert fr.ngram_similarity(s, s) == 1.0

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_cosine_symmetry(self, a, b):
        """Cosine similarity is symmetric."""
        import fuzzyrust as fr

        assert abs(fr.cosine_similarity_chars(a, b) - fr.cosine_similarity_chars(b, a)) < 1e-10

    @given(
        st.text(alphabet=string.ascii_letters, min_size=1, max_size=20),
        st.text(alphabet=string.ascii_letters, min_size=1, max_size=20),
    )
    @settings(max_examples=50)
    def test_soundex_deterministic(self, a, b):
        """Soundex encoding is deterministic."""
        import fuzzyrust as fr

        # Same input always produces same output
        assert fr.soundex(a) == fr.soundex(a)
        # Match is symmetric
        assert fr.soundex_match(a, b) == fr.soundex_match(b, a)

    @given(
        st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=20),
        st.text(min_size=1, max_size=30),
    )
    @settings(max_examples=30)
    def test_batch_levenshtein_consistency(self, strings, query):
        """Batch Levenshtein matches individual calculations."""
        import fuzzyrust as fr

        batch_results = fr.batch_levenshtein(strings, query)
        individual_results = [fr.levenshtein_similarity(s, query) for s in strings]
        # batch_levenshtein returns MatchResult with .score = normalized similarity (0.0-1.0)
        assert [r.score for r in batch_results] == individual_results

    @given(
        st.text(min_size=0, max_size=30),
        st.text(min_size=0, max_size=30),
        st.text(min_size=0, max_size=30),
    )
    def test_damerau_levenshtein_triangle_inequality(self, a, b, c):
        """Damerau-Levenshtein distance satisfies triangle inequality."""
        import fuzzyrust as fr

        d_ab = fr.damerau_levenshtein(a, b)
        d_bc = fr.damerau_levenshtein(b, c)
        d_ac = fr.damerau_levenshtein(a, c)
        assert d_ac <= d_ab + d_bc

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_levenshtein_non_negativity(self, a, b):
        """Levenshtein distance is always non-negative."""
        import fuzzyrust as fr

        lev_dist = fr.levenshtein(a, b)
        dam_dist = fr.damerau_levenshtein(a, b)
        # Distance is bounded by max string length (at most replace all chars + insert/delete difference)
        max_dist = max(len(a), len(b))
        assert 0 <= lev_dist <= max_dist, f"levenshtein({a!r}, {b!r}) = {lev_dist} is outside [0, {max_dist}]"
        assert 0 <= dam_dist <= max_dist, f"damerau_levenshtein({a!r}, {b!r}) = {dam_dist} is outside [0, {max_dist}]"

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_ngram_symmetry(self, a, b):
        """N-gram similarity is symmetric."""
        import fuzzyrust as fr

        assert abs(fr.ngram_similarity(a, b) - fr.ngram_similarity(b, a)) < 1e-10

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_ngram_bounds(self, a, b):
        """N-gram similarity is always between 0 and 1."""
        import fuzzyrust as fr

        sim = fr.ngram_similarity(a, b)
        assert 0.0 <= sim <= 1.0, f"ngram_similarity({a!r}, {b!r}) = {sim} is outside [0.0, 1.0]"

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_lcs_bounds(self, a, b):
        """LCS similarity is always between 0 and 1."""
        import fuzzyrust as fr

        sim = fr.lcs_similarity(a, b)
        assert 0.0 <= sim <= 1.0, f"lcs_similarity({a!r}, {b!r}) = {sim} is outside [0.0, 1.0]"

    @given(st.text(min_size=0, max_size=50))
    def test_lcs_identity(self, s):
        """LCS similarity of a string with itself is 1.0."""
        import fuzzyrust as fr

        assert fr.lcs_similarity(s, s) == 1.0

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_lcs_symmetry(self, a, b):
        """LCS similarity is symmetric."""
        import fuzzyrust as fr

        assert abs(fr.lcs_similarity(a, b) - fr.lcs_similarity(b, a)) < 1e-10

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_jaro_symmetry(self, a, b):
        """Jaro similarity is symmetric."""
        import fuzzyrust as fr

        assert abs(fr.jaro_similarity(a, b) - fr.jaro_similarity(b, a)) < 1e-10

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_jaro_bounds(self, a, b):
        """Jaro similarity is always between 0 and 1."""
        import fuzzyrust as fr

        sim = fr.jaro_similarity(a, b)
        assert 0.0 <= sim <= 1.0, f"jaro_similarity({a!r}, {b!r}) = {sim} is outside [0.0, 1.0]"

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_levenshtein_similarity_bounds(self, a, b):
        """Levenshtein similarity is always between 0 and 1."""
        import fuzzyrust as fr

        sim = fr.levenshtein_similarity(a, b)
        assert 0.0 <= sim <= 1.0, f"levenshtein_similarity({a!r}, {b!r}) = {sim} is outside [0.0, 1.0]"

    @given(st.text(min_size=0, max_size=50))
    def test_levenshtein_similarity_identity(self, s):
        """Levenshtein similarity of a string with itself is 1.0."""
        import fuzzyrust as fr

        assert fr.levenshtein_similarity(s, s) == 1.0

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_cosine_bounds(self, a, b):
        """Cosine similarity variants are always between 0 and 1."""
        import fuzzyrust as fr

        sim_chars = fr.cosine_similarity_chars(a, b)
        assert 0.0 <= sim_chars <= 1.0, f"cosine_similarity_chars({a!r}, {b!r}) = {sim_chars} is outside [0.0, 1.0]"
        sim_ngrams = fr.cosine_similarity_ngrams(a, b)
        assert 0.0 <= sim_ngrams <= 1.0, f"cosine_similarity_ngrams({a!r}, {b!r}) = {sim_ngrams} is outside [0.0, 1.0]"

    @given(
        st.text(alphabet=string.ascii_letters + " ", min_size=1, max_size=50),
        st.text(alphabet=string.ascii_letters + " ", min_size=1, max_size=50),
    )
    @settings(max_examples=50)
    def test_cosine_words_bounds(self, a, b):
        """Cosine word similarity is between 0 and 1."""
        import fuzzyrust as fr

        sim = fr.cosine_similarity_words(a, b)
        assert 0.0 <= sim <= 1.0, f"cosine_similarity_words({a!r}, {b!r}) = {sim} is outside [0.0, 1.0]"

    @given(st.text(min_size=0, max_size=40))
    def test_ci_variants_consistency(self, s):
        """Case-insensitive variants should return same result for same-case input."""
        import fuzzyrust as fr

        lower = s.lower()
        # CI variant on lowercase should equal regular on lowercase
        assert fr.levenshtein_ci(lower, lower) == fr.levenshtein(lower, lower)
        assert abs(fr.jaro_similarity_ci(lower, lower) - fr.jaro_similarity(lower, lower)) < 1e-10

    @given(
        st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=30),
        st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=30),
    )
    @settings(max_examples=50)
    def test_ci_case_insensitivity(self, a, b):
        """Case-insensitive functions treat different cases as equivalent."""
        import fuzzyrust as fr

        upper_a, upper_b = a.upper(), b.upper()
        # CI versions should give same result regardless of case
        assert fr.levenshtein_ci(a, b) == fr.levenshtein_ci(upper_a, upper_b)
        assert abs(fr.jaro_similarity_ci(a, b) - fr.jaro_similarity_ci(upper_a, upper_b)) < 1e-10


# =============================================================================
# Performance Benchmark Tests using pytest-benchmark
# =============================================================================


class TestBenchmarks:
    """Performance benchmark tests using pytest-benchmark."""

    @pytest.fixture
    def sample_strings(self):
        """Generate sample strings for benchmarking."""
        import random

        random.seed(42)
        return [
            "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 20)))
            for _ in range(1000)
        ]

    @pytest.fixture
    def large_string_list(self):
        """Generate large list for stress testing."""
        import random

        random.seed(42)
        return ["".join(random.choices(string.ascii_lowercase, k=10)) for _ in range(10000)]

    def test_benchmark_levenshtein(self, benchmark):
        """Benchmark Levenshtein distance calculation."""
        import fuzzyrust as fr

        result = benchmark(fr.levenshtein, "kitten", "sitting")
        assert result == 3

    def test_benchmark_jaro_winkler(self, benchmark):
        """Benchmark Jaro-Winkler similarity calculation."""
        import fuzzyrust as fr

        result = benchmark(fr.jaro_winkler_similarity, "hello", "hallo")
        # "hello" and "hallo" have high similarity (common prefix "h", similar structure)
        # Expected Jaro-Winkler similarity is approximately 0.88
        assert 0.85 <= result <= 0.92, f"Expected Jaro-Winkler('hello', 'hallo') in [0.85, 0.92], got {result}"

    def test_benchmark_batch_levenshtein(self, benchmark, sample_strings):
        """Benchmark batch Levenshtein processing."""
        import fuzzyrust as fr

        result = benchmark(fr.batch_levenshtein, sample_strings, "hello")
        assert len(result) == len(sample_strings)

    def test_benchmark_batch_jaro_winkler(self, benchmark, sample_strings):
        """Benchmark batch Jaro-Winkler processing."""
        import fuzzyrust as fr

        result = benchmark(fr.batch_jaro_winkler, sample_strings, "hello")
        assert len(result) == len(sample_strings)

    def test_benchmark_find_best_matches(self, benchmark, sample_strings):
        """Benchmark find_best_matches function."""
        import fuzzyrust as fr

        result = benchmark(fr.find_best_matches, sample_strings, "hello", limit=10)
        assert len(result) <= 10

    def test_benchmark_bktree_build(self, benchmark, sample_strings):
        """Benchmark BK-tree construction."""
        import fuzzyrust as fr

        def build_tree():
            tree = fr.BkTree()
            tree.add_all(sample_strings)
            return tree

        tree = benchmark(build_tree)
        assert len(tree) == len(sample_strings)

    def test_benchmark_bktree_search(self, benchmark, sample_strings):
        """Benchmark BK-tree search."""
        import fuzzyrust as fr

        tree = fr.BkTree()
        tree.add_all(sample_strings)
        result = benchmark(tree.search, "hello", max_distance=2)
        assert isinstance(result, list), f"Expected list, got {type(result).__name__}"
        # Verify result contains SearchResult objects with valid structure
        for r in result:
            assert hasattr(r, "text") and hasattr(r, "distance"), f"SearchResult missing required attributes"
            assert r.distance <= 2, f"Result distance {r.distance} exceeds max_distance=2"

    def test_benchmark_ngram_index_build(self, benchmark, sample_strings):
        """Benchmark N-gram index construction."""
        import fuzzyrust as fr

        def build_index():
            index = fr.NgramIndex(ngram_size=2)
            index.add_all(sample_strings)
            return index

        index = benchmark(build_index)
        assert len(index) == len(sample_strings)

    def test_benchmark_ngram_index_search(self, benchmark, sample_strings):
        """Benchmark N-gram index search."""
        import fuzzyrust as fr

        index = fr.NgramIndex(ngram_size=2)
        index.add_all(sample_strings)
        result = benchmark(index.search, "hello", min_similarity=0.5)
        assert isinstance(result, list), f"Expected list, got {type(result).__name__}"
        # Verify result contains MatchResult objects with valid structure and scores
        for r in result:
            assert hasattr(r, "text") and hasattr(r, "score"), f"MatchResult missing required attributes"
            assert r.score >= 0.5, f"Result score {r.score} below min_similarity=0.5"

    def test_benchmark_soundex(self, benchmark):
        """Benchmark Soundex encoding."""
        import fuzzyrust as fr

        result = benchmark(fr.soundex, "Washington")
        assert len(result) == 4

    def test_benchmark_metaphone(self, benchmark):
        """Benchmark Metaphone encoding."""
        import fuzzyrust as fr

        result = benchmark(fr.metaphone, "Washington")
        assert len(result) > 0

    # --- Additional benchmarks for varied input sizes ---

    @pytest.fixture
    def short_string_pair(self):
        """Short strings for benchmarking."""
        return ("cat", "bat")

    @pytest.fixture
    def medium_string_pair(self):
        """Medium-length strings for benchmarking."""
        return ("the quick brown fox", "the quick brown dog")

    @pytest.fixture
    def long_string_pair(self):
        """Long strings for benchmarking."""
        import random

        random.seed(42)
        s1 = "".join(random.choices(string.ascii_lowercase + " ", k=500))
        s2 = "".join(random.choices(string.ascii_lowercase + " ", k=500))
        return (s1, s2)

    def test_benchmark_levenshtein_short(self, benchmark, short_string_pair):
        """Benchmark Levenshtein on short strings."""
        import fuzzyrust as fr

        s1, s2 = short_string_pair
        benchmark(fr.levenshtein, s1, s2)

    def test_benchmark_levenshtein_medium(self, benchmark, medium_string_pair):
        """Benchmark Levenshtein on medium strings."""
        import fuzzyrust as fr

        s1, s2 = medium_string_pair
        benchmark(fr.levenshtein, s1, s2)

    def test_benchmark_levenshtein_long(self, benchmark, long_string_pair):
        """Benchmark Levenshtein on long strings."""
        import fuzzyrust as fr

        s1, s2 = long_string_pair
        benchmark(fr.levenshtein, s1, s2)

    def test_benchmark_damerau_levenshtein(self, benchmark, medium_string_pair):
        """Benchmark Damerau-Levenshtein distance."""
        import fuzzyrust as fr

        s1, s2 = medium_string_pair
        benchmark(fr.damerau_levenshtein, s1, s2)

    def test_benchmark_jaro(self, benchmark, medium_string_pair):
        """Benchmark Jaro similarity."""
        import fuzzyrust as fr

        s1, s2 = medium_string_pair
        benchmark(fr.jaro_similarity, s1, s2)

    def test_benchmark_ngram_similarity(self, benchmark, medium_string_pair):
        """Benchmark N-gram similarity."""
        import fuzzyrust as fr

        s1, s2 = medium_string_pair
        benchmark(fr.ngram_similarity, s1, s2)

    def test_benchmark_lcs(self, benchmark, medium_string_pair):
        """Benchmark LCS similarity."""
        import fuzzyrust as fr

        s1, s2 = medium_string_pair
        benchmark(fr.lcs_similarity, s1, s2)

    def test_benchmark_cosine_chars(self, benchmark, medium_string_pair):
        """Benchmark cosine similarity (character-based)."""
        import fuzzyrust as fr

        s1, s2 = medium_string_pair
        benchmark(fr.cosine_similarity_chars, s1, s2)

    def test_benchmark_cosine_ngrams(self, benchmark, medium_string_pair):
        """Benchmark cosine similarity (n-gram-based)."""
        import fuzzyrust as fr

        s1, s2 = medium_string_pair
        benchmark(fr.cosine_similarity_ngrams, s1, s2)

    def test_benchmark_cosine_words(self, benchmark, medium_string_pair):
        """Benchmark cosine similarity (word-based)."""
        import fuzzyrust as fr

        s1, s2 = medium_string_pair
        benchmark(fr.cosine_similarity_words, s1, s2)

    # --- Index benchmarks with different sizes ---

    @pytest.fixture
    def medium_string_list(self):
        """5000 strings for medium-scale benchmarking."""
        import random

        random.seed(42)
        return [
            "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 15)))
            for _ in range(5000)
        ]

    def test_benchmark_hybrid_index_build(self, benchmark, sample_strings):
        """Benchmark HybridIndex construction."""
        import fuzzyrust as fr

        def build_index():
            index = fr.HybridIndex(ngram_size=2)
            index.add_all(sample_strings)
            return index

        index = benchmark(build_index)
        assert len(index) == len(sample_strings)

    def test_benchmark_hybrid_index_search(self, benchmark, sample_strings):
        """Benchmark HybridIndex search."""
        import fuzzyrust as fr

        index = fr.HybridIndex(ngram_size=2)
        index.add_all(sample_strings)
        result = benchmark(index.search, "hello", min_similarity=0.5)
        assert isinstance(result, list), f"Expected list, got {type(result).__name__}"
        # Verify result contains MatchResult objects with valid structure and scores
        for r in result:
            assert hasattr(r, "text") and hasattr(r, "score"), f"MatchResult missing required attributes"
            assert r.score >= 0.5, f"Result score {r.score} below min_similarity=0.5"

    def test_benchmark_find_duplicates_small(self, benchmark):
        """Benchmark find_duplicates on small dataset."""
        import fuzzyrust as fr

        strings = [f"item_{i % 20}" for i in range(100)]
        result = benchmark(fr.find_duplicates, strings, min_similarity=0.9)
        assert result.total_duplicates >= 0

    def test_benchmark_find_duplicates_medium(self, benchmark):
        """Benchmark find_duplicates on medium dataset."""
        import fuzzyrust as fr

        strings = [f"product_{i % 50}" for i in range(500)]
        result = benchmark(fr.find_duplicates, strings, min_similarity=0.9)
        assert result.total_duplicates >= 0

    # --- Multi-algorithm comparison ---

    def test_benchmark_compare_algorithms(self, benchmark, sample_strings):
        """Benchmark multi-algorithm comparison."""
        import fuzzyrust as fr

        result = benchmark(fr.compare_algorithms, sample_strings[:100], "hello")
        assert len(result) > 0


# =============================================================================
# Threading and Concurrent Access Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safety and concurrent access patterns."""

    def test_concurrent_similarity_calculations(self):
        """Test concurrent calls to similarity functions."""
        import fuzzyrust as fr

        results = []
        errors = []

        def compute_similarities(thread_id):
            try:
                for i in range(100):
                    s1 = f"string_{thread_id}_{i}"
                    s2 = f"string_{thread_id}_{i + 1}"
                    fr.levenshtein(s1, s2)
                    fr.jaro_winkler_similarity(s1, s2)
                    fr.ngram_similarity(s1, s2)
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=compute_similarities, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10

    def test_concurrent_batch_operations(self):
        """Test concurrent batch processing calls."""
        import fuzzyrust as fr

        strings = [f"word_{i}" for i in range(100)]

        def batch_operation(query):
            return fr.batch_jaro_winkler(strings, query)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            queries = [f"word_{i}" for i in range(20)]
            futures = [executor.submit(batch_operation, q) for q in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        for r in results:
            assert len(r) == 100

    def test_separate_index_instances_per_thread(self):
        """Test that separate index instances work correctly in different threads."""
        import fuzzyrust as fr

        results = {}
        errors = []

        def thread_with_index(thread_id):
            try:
                # Each thread creates its own index
                tree = fr.BkTree()
                for i in range(50):
                    tree.add(f"thread_{thread_id}_item_{i}")

                search_results = tree.search(f"thread_{thread_id}_item_25", max_distance=2)
                results[thread_id] = len(search_results)
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=thread_with_index, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 5
        for thread_id, count in results.items():
            assert count > 0

    def test_concurrent_find_best_matches(self):
        """Test find_best_matches under concurrent load."""
        import fuzzyrust as fr

        strings = [f"product_{i:04d}" for i in range(500)]

        def search(query):
            return fr.find_best_matches(strings, query, limit=5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            queries = [f"product_{i:04d}" for i in range(0, 100, 10)]
            futures = [executor.submit(search, q) for q in queries]
            results = [f.result() for f in futures]

        assert len(results) == 10
        for r in results:
            assert len(r) <= 5

    def test_parallel_index_searches_different_instances(self):
        """Test parallel searches on different NgramIndex instances."""
        import fuzzyrust as fr

        def create_and_search(suffix):
            index = fr.NgramIndex(ngram_size=2)
            items = [f"item_{suffix}_{i}" for i in range(100)]
            index.add_all(items)
            return index.search(f"item_{suffix}_50", min_similarity=0.7)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_and_search, i) for i in range(8)]
            results = [f.result() for f in futures]

        assert len(results) == 8

    @pytest.mark.slow
    def test_high_contention_similarity(self):
        """Stress test: many threads calling similarity functions simultaneously."""
        import fuzzyrust as fr

        errors = []
        results_count = [0]  # Use list for thread-safe increment
        lock = threading.Lock()

        def stress_worker(worker_id):
            try:
                for i in range(1000):
                    s1 = f"worker_{worker_id}_iteration_{i}"
                    s2 = f"worker_{worker_id}_iteration_{i + 1}"
                    # Call multiple similarity functions
                    fr.levenshtein(s1, s2)
                    fr.jaro_similarity(s1, s2)
                    fr.jaro_winkler_similarity(s1, s2)
                    fr.ngram_similarity(s1, s2)
                    fr.cosine_similarity_ngrams(s1, s2)
                with lock:
                    results_count[0] += 1
            except Exception as e:
                with lock:
                    errors.append((worker_id, e))

        # 20 threads, each doing 1000 iterations
        threads = [threading.Thread(target=stress_worker, args=(i,)) for i in range(20)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in stress test: {errors}"
        assert results_count[0] == 20

    @pytest.mark.slow
    def test_result_consistency_under_concurrency(self):
        """Verify same inputs produce same outputs under concurrent load."""
        import fuzzyrust as fr

        # Fixed test inputs
        test_pairs = [
            ("hello", "hallo"),
            ("world", "word"),
            ("python", "pthon"),
            ("fuzzy", "fuzz"),
        ]

        # Pre-compute expected results
        expected = {}
        for s1, s2 in test_pairs:
            expected[(s1, s2)] = (
                fr.levenshtein(s1, s2),
                fr.jaro_winkler_similarity(s1, s2),
                fr.ngram_similarity(s1, s2),
            )

        inconsistencies = []
        lock = threading.Lock()

        def verify_consistency(thread_id):
            for _ in range(100):
                for s1, s2 in test_pairs:
                    lev = fr.levenshtein(s1, s2)
                    jw = fr.jaro_winkler_similarity(s1, s2)
                    ng = fr.ngram_similarity(s1, s2)
                    exp_lev, exp_jw, exp_ng = expected[(s1, s2)]

                    if lev != exp_lev or jw != exp_jw or ng != exp_ng:
                        with lock:
                            inconsistencies.append(
                                {
                                    "thread": thread_id,
                                    "pair": (s1, s2),
                                    "got": (lev, jw, ng),
                                    "expected": (exp_lev, exp_jw, exp_ng),
                                }
                            )

        threads = [threading.Thread(target=verify_consistency, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(inconsistencies) == 0, f"Inconsistent results: {inconsistencies[:5]}"

    @pytest.mark.slow
    def test_rapid_fire_batch_operations(self):
        """Stress test rapid sequential and parallel batch calls."""
        import fuzzyrust as fr

        strings = [f"item_{i:04d}" for i in range(200)]
        queries = [f"item_{i:04d}" for i in range(0, 200, 5)]

        errors = []

        def batch_worker(worker_id):
            try:
                for _ in range(50):
                    results = fr.batch_jaro_winkler(strings, queries[worker_id % len(queries)])
                    assert len(results) == len(strings)
            except Exception as e:
                errors.append((worker_id, e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(batch_worker, i) for i in range(40)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Batch operation errors: {errors}"

    @pytest.mark.slow
    def test_concurrent_deduplication(self):
        """Test find_duplicates under concurrent execution."""
        import fuzzyrust as fr

        def dedupe_task(seed):
            # Each task has slightly different data
            strings = [f"duplicate_{seed}_{i % 10}" for i in range(100)]
            result = fr.find_duplicates(strings, min_similarity=0.9)
            # DeduplicationResult has groups, unique, and total_duplicates attributes
            return result.total_duplicates

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(dedupe_task, i) for i in range(16)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All tasks should complete without error
        assert len(results) == 16
        # Each should find some duplicates (since we have repeated strings)
        assert all(r > 0 for r in results)


# =============================================================================
# Memory Stress Tests
# =============================================================================


class TestMemoryStress:
    """Memory stress tests for large datasets."""

    @pytest.mark.slow
    def test_large_bktree(self):
        """Test BK-tree with large number of entries."""
        import fuzzyrust as fr

        tree = fr.BkTree()
        # Add 100,000 strings
        strings = [f"item_{i:06d}" for i in range(100_000)]
        tree.add_all(strings)

        assert len(tree) == 100_000

        # Verify search still works
        results = tree.search("item_050000", max_distance=1)
        assert len(results) > 0
        # Results are SearchResult objects
        assert any(r.text == "item_050000" for r in results)

    @pytest.mark.slow
    def test_large_ngram_index(self):
        """Test N-gram index with large number of entries."""
        import fuzzyrust as fr

        index = fr.NgramIndex(ngram_size=2)
        strings = [f"product_{i:06d}" for i in range(100_000)]
        index.add_all(strings)

        assert len(index) == 100_000

        # Verify search works
        results = index.search("product_050000", min_similarity=0.9)
        assert len(results) > 0

    @pytest.mark.slow
    def test_large_batch_processing(self):
        """Test batch processing with large input."""
        import fuzzyrust as fr

        strings = [f"string_{i:05d}" for i in range(50_000)]

        # Should handle large batch without memory issues
        results = fr.batch_levenshtein(strings, "string_25000")

        assert len(results) == 50_000
        assert results[25000].score == 1.0  # Exact match has similarity 1.0

    @pytest.mark.slow
    def test_long_strings(self):
        """Test algorithms with very long strings."""
        import fuzzyrust as fr

        long_a = "a" * 10_000
        long_b = "a" * 9_999 + "b"

        # Should handle without stack overflow or excessive memory
        dist = fr.levenshtein(long_a, long_b)
        assert dist == 1

        sim = fr.jaro_winkler_similarity(long_a, long_b)
        assert sim > 0.99

    @pytest.mark.slow
    def test_many_small_operations(self):
        """Stress test with many small operations."""
        import fuzzyrust as fr

        # 1 million small operations
        for i in range(1_000_000):
            result = fr.levenshtein("hello", "hallo")

        # Verify computation is correct after many iterations (no memory corruption)
        assert result == 1, f"Expected distance of 1 after 1M operations, got {result}"

    @pytest.mark.slow
    def test_hybrid_index_large_scale(self):
        """Test HybridIndex with large dataset."""
        import fuzzyrust as fr

        index = fr.HybridIndex(ngram_size=3)
        strings = [f"hybrid_test_{i:06d}" for i in range(50_000)]
        index.add_all(strings)

        assert len(index) == 50_000

        results = index.search("hybrid_test_025000", min_similarity=0.8, limit=10)
        assert len(results) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
