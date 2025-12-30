"""
Parameter validation and mathematical property tests for FuzzyRust.

Tests cover:
- Parameter boundary validation (min_similarity, ngram_size, etc.)
- Type validation (non-string inputs)
- Mathematical properties of similarity/distance metrics
"""

import math

import pytest

import fuzzyrust as fr


class TestParameterValidation:
    """Tests for parameter boundary validation."""

    def test_min_similarity_above_one(self):
        """min_similarity > 1.0 should raise ValidationError."""
        with pytest.raises(fr.ValidationError):
            fr.find_best_matches(["a"], "b", min_similarity=1.5)

    def test_min_similarity_negative(self):
        """Negative min_similarity should raise ValidationError."""
        with pytest.raises(fr.ValidationError):
            fr.find_best_matches(["a"], "b", min_similarity=-0.1)

    def test_min_similarity_nan(self):
        """NaN min_similarity should raise ValidationError."""
        with pytest.raises(fr.ValidationError):
            fr.find_best_matches(["a"], "b", min_similarity=float("nan"))

    def test_min_similarity_infinity(self):
        """Infinity min_similarity should raise ValidationError."""
        with pytest.raises(fr.ValidationError):
            fr.find_best_matches(["a"], "b", min_similarity=float("inf"))

    def test_min_similarity_negative_infinity(self):
        """Negative infinity min_similarity should raise ValidationError."""
        with pytest.raises(fr.ValidationError):
            fr.find_best_matches(["a"], "b", min_similarity=float("-inf"))

    def test_ngram_size_zero(self):
        """Zero ngram_size should raise ValidationError."""
        with pytest.raises(fr.ValidationError):
            fr.ngram_similarity("hello", "world", ngram_size=0)

    def test_ngram_size_negative(self):
        """Negative ngram_size should raise ValidationError."""
        # PyO3 converts negative int to unsigned, which may cause overflow error
        # or ValidationError depending on implementation
        with pytest.raises(Exception):
            fr.ngram_similarity("a", "b", ngram_size=-1)

    def test_batch_length_mismatch(self):
        """Mismatched list lengths should raise error."""
        with pytest.raises(Exception):
            fr.batch_similarity_pairs(["a", "b"], ["x"], "levenshtein")

    def test_index_min_similarity_above_one(self):
        """Index search with min_similarity > 1.0 should raise ValidationError."""
        index = fr.NgramIndex(ngram_size=2)
        index.add("hello")
        with pytest.raises(fr.ValidationError):
            index.search("hello", min_similarity=1.5)

    def test_index_min_similarity_negative(self):
        """Index search with negative min_similarity should raise ValidationError."""
        index = fr.NgramIndex(ngram_size=2)
        index.add("hello")
        with pytest.raises(fr.ValidationError):
            index.search("hello", min_similarity=-0.5)

    def test_hybrid_index_min_similarity_above_one(self):
        """HybridIndex search with min_similarity > 1.0 should raise ValidationError."""
        index = fr.HybridIndex(ngram_size=2)
        index.add("hello")
        with pytest.raises(fr.ValidationError):
            index.search("hello", min_similarity=1.1)

    def test_find_duplicates_min_similarity_above_one(self):
        """find_duplicates with min_similarity > 1.0 should raise ValidationError."""
        with pytest.raises(fr.ValidationError):
            fr.find_duplicates(["hello", "hallo"], min_similarity=1.5)

    def test_find_duplicates_min_similarity_negative(self):
        """find_duplicates with negative min_similarity should raise ValidationError."""
        with pytest.raises(fr.ValidationError):
            fr.find_duplicates(["hello", "hallo"], min_similarity=-0.1)

    def test_extract_min_similarity_above_one(self):
        """extract with min_similarity > 1.0 should return empty results.

        Note: extract() does not raise ValidationError for out-of-range min_similarity,
        instead it returns empty results since no match can have similarity > 1.0.
        This is consistent with RapidFuzz compatibility semantics.
        """
        result = fr.extract("hello", ["hello", "world"], min_similarity=1.5)
        assert result == [], "min_similarity > 1.0 should return empty results"

    def test_jaro_winkler_prefix_weight_above_one(self):
        """Jaro-Winkler with prefix_weight > 1.0 should raise ValidationError."""
        # Prefix weight > 0.25 can cause similarity > 1.0, so should be validated
        with pytest.raises(fr.ValidationError):
            fr.jaro_winkler_similarity("hello", "hallo", prefix_weight=1.5)

    def test_jaro_winkler_prefix_weight_negative(self):
        """Jaro-Winkler with negative prefix_weight should raise ValidationError."""
        with pytest.raises(fr.ValidationError):
            fr.jaro_winkler_similarity("hello", "hallo", prefix_weight=-0.1)


class TestTypeErrors:
    """Tests for type validation."""

    def test_levenshtein_integer_input(self):
        """Integer input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.levenshtein(123, "hello")  # type: ignore[arg-type]

    def test_levenshtein_list_input(self):
        """List input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.levenshtein(["hello"], "world")  # type: ignore[arg-type]

    def test_levenshtein_none_first(self):
        """None as first argument should raise TypeError."""
        with pytest.raises(TypeError):
            fr.levenshtein(None, "hello")  # type: ignore[arg-type]

    def test_levenshtein_none_second(self):
        """None as second argument should raise TypeError."""
        with pytest.raises(TypeError):
            fr.levenshtein("hello", None)  # type: ignore[arg-type]

    def test_jaro_winkler_integer_input(self):
        """Integer input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.jaro_winkler_similarity(123, "hello")  # type: ignore[arg-type]

    def test_jaro_winkler_list_input(self):
        """List input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.jaro_winkler_similarity(["hello"], "world")  # type: ignore[arg-type]

    def test_jaro_winkler_none_input(self):
        """None input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.jaro_winkler_similarity(None, "hello")  # type: ignore[arg-type]

    def test_ngram_similarity_integer_input(self):
        """Integer input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.ngram_similarity(123, "hello")  # type: ignore[arg-type]

    def test_ngram_similarity_none_input(self):
        """None input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.ngram_similarity(None, "hello")  # type: ignore[arg-type]

    def test_damerau_levenshtein_integer_input(self):
        """Integer input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.damerau_levenshtein(123, "hello")  # type: ignore[arg-type]

    def test_find_best_matches_non_list_strings(self):
        """Non-list strings argument should raise TypeError."""
        with pytest.raises(TypeError):
            fr.find_best_matches("not a list", "query")  # type: ignore[arg-type]

    def test_find_best_matches_integer_query(self):
        """Integer query should raise TypeError."""
        with pytest.raises(TypeError):
            fr.find_best_matches(["hello", "world"], 123)  # type: ignore[arg-type]

    def test_soundex_integer_input(self):
        """Integer input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.soundex(123)  # type: ignore[arg-type]

    def test_metaphone_integer_input(self):
        """Integer input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.metaphone(123)  # type: ignore[arg-type]


class TestMathematicalProperties:
    """Verify algorithms satisfy mathematical properties."""

    def test_levenshtein_triangle_inequality(self):
        """
        Triangle inequality: d(a,c) <= d(a,b) + d(b,c).

        This is a fundamental property of metric spaces.
        Levenshtein distance is a proper metric and must satisfy this.
        """
        test_cases = [
            ("kitten", "sitting", "kitchen"),
            ("hello", "hallo", "hullo"),
            ("abc", "abd", "acd"),
            ("", "test", "testing"),
            ("algorithm", "altruistic", "logarithm"),
        ]
        for a, b, c in test_cases:
            d_ac = fr.levenshtein(a, c)
            d_ab = fr.levenshtein(a, b)
            d_bc = fr.levenshtein(b, c)
            assert d_ac <= d_ab + d_bc, (
                f"Triangle inequality violated for ({a!r}, {b!r}, {c!r}): "
                f"d({a!r}, {c!r})={d_ac} > d({a!r}, {b!r})={d_ab} + d({b!r}, {c!r})={d_bc}"
            )

    def test_damerau_levenshtein_triangle_inequality(self):
        """
        Triangle inequality: d(a,c) <= d(a,b) + d(b,c).

        Damerau-Levenshtein is also a proper metric.
        """
        test_cases = [
            ("kitten", "sitting", "kitchen"),
            ("ab", "ba", "aa"),
            ("hello", "ehllo", "olleh"),
        ]
        for a, b, c in test_cases:
            d_ac = fr.damerau_levenshtein(a, c)
            d_ab = fr.damerau_levenshtein(a, b)
            d_bc = fr.damerau_levenshtein(b, c)
            assert d_ac <= d_ab + d_bc, (
                f"Triangle inequality violated for ({a!r}, {b!r}, {c!r}): "
                f"d({a!r}, {c!r})={d_ac} > d({a!r}, {b!r})={d_ab} + d({b!r}, {c!r})={d_bc}"
            )

    def test_similarity_symmetry(self):
        """
        Symmetry: sim(a,b) == sim(b,a).

        All similarity functions should be symmetric.
        """
        test_pairs = [
            ("hello", "world"),
            ("kitten", "sitting"),
            ("abc", "xyz"),
            ("", "test"),
            ("algorithm", "altruistic"),
        ]
        for a, b in test_pairs:
            # Jaro-Winkler
            jw_ab = fr.jaro_winkler_similarity(a, b)
            jw_ba = fr.jaro_winkler_similarity(b, a)
            assert jw_ab == pytest.approx(
                jw_ba, abs=1e-9
            ), f"Jaro-Winkler not symmetric for ({a!r}, {b!r}): {jw_ab} != {jw_ba}"

            # Jaro
            j_ab = fr.jaro_similarity(a, b)
            j_ba = fr.jaro_similarity(b, a)
            assert j_ab == pytest.approx(
                j_ba, abs=1e-9
            ), f"Jaro not symmetric for ({a!r}, {b!r}): {j_ab} != {j_ba}"

            # Levenshtein similarity
            lev_ab = fr.levenshtein_similarity(a, b)
            lev_ba = fr.levenshtein_similarity(b, a)
            assert lev_ab == pytest.approx(
                lev_ba, abs=1e-9
            ), f"Levenshtein similarity not symmetric for ({a!r}, {b!r}): {lev_ab} != {lev_ba}"

            # N-gram similarity
            ng_ab = fr.ngram_similarity(a, b)
            ng_ba = fr.ngram_similarity(b, a)
            assert ng_ab == pytest.approx(
                ng_ba, abs=1e-9
            ), f"N-gram similarity not symmetric for ({a!r}, {b!r}): {ng_ab} != {ng_ba}"

            # Cosine similarity (chars)
            cos_ab = fr.cosine_similarity_chars(a, b)
            cos_ba = fr.cosine_similarity_chars(b, a)
            assert cos_ab == pytest.approx(
                cos_ba, abs=1e-9
            ), f"Cosine similarity not symmetric for ({a!r}, {b!r}): {cos_ab} != {cos_ba}"

    def test_distance_symmetry(self):
        """
        Symmetry: d(a,b) == d(b,a).

        All distance functions should be symmetric.
        """
        test_pairs = [
            ("hello", "world"),
            ("kitten", "sitting"),
            ("abc", "xyz"),
            ("", "test"),
        ]
        for a, b in test_pairs:
            # Levenshtein distance
            lev_ab = fr.levenshtein(a, b)
            lev_ba = fr.levenshtein(b, a)
            assert (
                lev_ab == lev_ba
            ), f"Levenshtein distance not symmetric for ({a!r}, {b!r}): {lev_ab} != {lev_ba}"

            # Damerau-Levenshtein distance
            dam_ab = fr.damerau_levenshtein(a, b)
            dam_ba = fr.damerau_levenshtein(b, a)
            assert (
                dam_ab == dam_ba
            ), f"Damerau-Levenshtein distance not symmetric for ({a!r}, {b!r}): {dam_ab} != {dam_ba}"

    def test_identical_strings_max_similarity(self):
        """
        Identity: sim(a,a) == 1.0 for all algorithms.

        Identical strings should always have maximum similarity.
        """
        test_strings = [
            "test string",
            "hello world",
            "a",
            "algorithm",
            "",  # Empty string special case
        ]
        for s in test_strings:
            # Jaro-Winkler
            assert fr.jaro_winkler_similarity(s, s) == 1.0, f"Jaro-Winkler({s!r}, {s!r}) != 1.0"

            # Jaro
            assert fr.jaro_similarity(s, s) == 1.0, f"Jaro({s!r}, {s!r}) != 1.0"

            # Levenshtein similarity
            assert (
                fr.levenshtein_similarity(s, s) == 1.0
            ), f"Levenshtein similarity({s!r}, {s!r}) != 1.0"

            # Damerau-Levenshtein similarity
            assert (
                fr.damerau_levenshtein_similarity(s, s) == 1.0
            ), f"Damerau-Levenshtein similarity({s!r}, {s!r}) != 1.0"

            # N-gram similarity
            assert fr.ngram_similarity(s, s) == 1.0, f"N-gram similarity({s!r}, {s!r}) != 1.0"

            # Cosine similarity (chars)
            assert (
                fr.cosine_similarity_chars(s, s) == 1.0
            ), f"Cosine similarity chars({s!r}, {s!r}) != 1.0"

            # LCS similarity (skip empty strings - edge case)
            if s:
                assert fr.lcs_similarity(s, s) == 1.0, f"LCS similarity({s!r}, {s!r}) != 1.0"

    def test_identical_strings_zero_distance(self):
        """
        Identity: d(a,a) == 0 for all distance metrics.

        Identical strings should always have zero distance.
        """
        test_strings = [
            "test string",
            "hello world",
            "a",
            "algorithm",
            "",  # Empty string special case
        ]
        for s in test_strings:
            # Levenshtein
            assert fr.levenshtein(s, s) == 0, f"Levenshtein({s!r}, {s!r}) != 0"

            # Damerau-Levenshtein
            assert fr.damerau_levenshtein(s, s) == 0, f"Damerau-Levenshtein({s!r}, {s!r}) != 0"

            # Hamming (only for non-empty strings)
            if s:
                assert fr.hamming(s, s) == 0, f"Hamming({s!r}, {s!r}) != 0"

    def test_similarity_bounds(self):
        """
        Bounds: 0.0 <= sim(a,b) <= 1.0 for all algorithms.

        Similarity scores must always be in the [0, 1] range.
        """
        test_pairs = [
            ("hello", "world"),
            ("hello", "hello"),
            ("", "test"),
            ("abc", "xyz"),
            ("kitten", "sitting"),
            ("algorithm", "altruistic"),
        ]
        for a, b in test_pairs:
            # Jaro-Winkler
            jw = fr.jaro_winkler_similarity(a, b)
            assert 0.0 <= jw <= 1.0, f"Jaro-Winkler({a!r}, {b!r})={jw} out of bounds"

            # Jaro
            j = fr.jaro_similarity(a, b)
            assert 0.0 <= j <= 1.0, f"Jaro({a!r}, {b!r})={j} out of bounds"

            # Levenshtein similarity
            lev = fr.levenshtein_similarity(a, b)
            assert 0.0 <= lev <= 1.0, f"Levenshtein similarity({a!r}, {b!r})={lev} out of bounds"

            # N-gram similarity
            ng = fr.ngram_similarity(a, b)
            assert 0.0 <= ng <= 1.0, f"N-gram similarity({a!r}, {b!r})={ng} out of bounds"

            # Cosine similarity
            cos = fr.cosine_similarity_chars(a, b)
            assert 0.0 <= cos <= 1.0, f"Cosine similarity({a!r}, {b!r})={cos} out of bounds"

    def test_distance_non_negative(self):
        """
        Non-negativity: d(a,b) >= 0 for all distance metrics.

        Distance must always be non-negative.
        """
        test_pairs = [
            ("hello", "world"),
            ("hello", "hello"),
            ("", "test"),
            ("abc", "xyz"),
        ]
        for a, b in test_pairs:
            # Levenshtein
            lev = fr.levenshtein(a, b)
            assert lev >= 0, f"Levenshtein({a!r}, {b!r})={lev} is negative"

            # Damerau-Levenshtein
            dam = fr.damerau_levenshtein(a, b)
            assert dam >= 0, f"Damerau-Levenshtein({a!r}, {b!r})={dam} is negative"

    def test_lcs_length_bounds(self):
        """
        LCS length bounds: 0 <= lcs_length(a,b) <= min(len(a), len(b)).

        The longest common subsequence cannot be longer than the shorter string.
        """
        test_pairs = [
            ("hello", "world"),
            ("abc", "abcd"),
            ("", "test"),
            ("algorithm", "altruistic"),
        ]
        for a, b in test_pairs:
            lcs_len = fr.lcs_length(a, b)
            max_possible = min(len(a), len(b))
            assert (
                0 <= lcs_len <= max_possible
            ), f"LCS length({a!r}, {b!r})={lcs_len} out of bounds [0, {max_possible}]"

    def test_levenshtein_upper_bound(self):
        """
        Upper bound: d(a,b) <= max(len(a), len(b)).

        Levenshtein distance cannot exceed the length of the longer string.
        """
        test_pairs = [
            ("hello", "world"),
            ("abc", "xyz"),
            ("", "test"),
            ("a", "bcdefghij"),
        ]
        for a, b in test_pairs:
            dist = fr.levenshtein(a, b)
            upper = max(len(a), len(b))
            assert dist <= upper, f"Levenshtein({a!r}, {b!r})={dist} > max(len)={upper}"

    def test_completely_different_strings(self):
        """
        Completely different strings (no common characters) should have low similarity.
        """
        # Strings with no overlapping characters
        a = "abc"
        b = "xyz"

        # All similarity metrics should return 0.0 for completely disjoint strings
        assert fr.jaro_similarity(a, b) == 0.0, "Jaro should be 0.0 for disjoint strings"
        assert (
            fr.jaro_winkler_similarity(a, b) == 0.0
        ), "Jaro-Winkler should be 0.0 for disjoint strings"
        assert (
            fr.levenshtein_similarity(a, b) == 0.0
        ), "Levenshtein similarity should be 0.0 for equal-length disjoint strings"
        assert (
            fr.ngram_similarity(a, b) == 0.0
        ), "N-gram similarity should be 0.0 for disjoint strings"

    def test_single_character_edit_distance(self):
        """
        Single character operations should have predictable edit distances.
        """
        # Single substitution
        assert fr.levenshtein("hello", "hallo") == 1, "Single substitution should be distance 1"

        # Single insertion
        assert fr.levenshtein("hello", "helloo") == 1, "Single insertion should be distance 1"
        assert (
            fr.levenshtein("hello", "hhello") == 1
        ), "Single prefix insertion should be distance 1"

        # Single deletion
        assert fr.levenshtein("hello", "hell") == 1, "Single deletion should be distance 1"
        assert fr.levenshtein("hello", "ello") == 1, "Single prefix deletion should be distance 1"

    def test_damerau_vs_levenshtein_transposition(self):
        """
        Damerau-Levenshtein should count transpositions as single edits.

        For a transposition (adjacent character swap), Damerau should be 1,
        but regular Levenshtein should be 2.
        """
        # Simple transposition
        assert fr.levenshtein("ab", "ba") == 2, "Levenshtein: swap is 2 edits"
        assert fr.damerau_levenshtein("ab", "ba") == 1, "Damerau: swap is 1 edit"

        # Transposition in longer string
        assert fr.levenshtein("hello", "ehllo") == 2, "Levenshtein: swap in word is 2 edits"
        assert fr.damerau_levenshtein("hello", "ehllo") == 1, "Damerau: swap in word is 1 edit"
