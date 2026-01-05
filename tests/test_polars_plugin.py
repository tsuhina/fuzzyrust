"""Tests for the native Polars plugin functionality.

These tests verify that:
1. Plugin detection works correctly
2. Native plugin functions produce correct results
3. Results match the fallback (map_elements) implementation
4. Environment variable disabling works
"""

import os

import polars as pl
import pytest

import fuzzyrust as fr
from fuzzyrust._plugin import (
    _find_plugin_lib,
    is_plugin_available,
    use_native_plugin,
)


class TestPluginDetection:
    """Tests for plugin detection and availability."""

    def test_plugin_library_found(self):
        """Verify the plugin library can be found."""
        lib_path = _find_plugin_lib()
        assert lib_path is not None, "Plugin library not found"
        assert lib_path.exists(), f"Plugin library does not exist: {lib_path}"

    def test_plugin_available(self):
        """Verify the plugin is available."""
        # Reset any cached state
        use_native_plugin(True)
        assert is_plugin_available(), "Plugin should be available"

    def test_disable_via_env_var(self):
        """Test disabling plugin via environment variable."""
        # Save original state
        original_value = os.environ.get("FUZZYRUST_DISABLE_PLUGIN")

        try:
            # Disable via env var
            os.environ["FUZZYRUST_DISABLE_PLUGIN"] = "1"
            use_native_plugin(True)  # Reset to re-check

            # Check it's disabled
            # Force re-evaluation by resetting the cached value
            import fuzzyrust._plugin as plugin_module

            plugin_module._state.available = None

            assert not is_plugin_available(), "Plugin should be disabled"
        finally:
            # Restore original state
            if original_value is None:
                os.environ.pop("FUZZYRUST_DISABLE_PLUGIN", None)
            else:
                os.environ["FUZZYRUST_DISABLE_PLUGIN"] = original_value
            # Re-enable for other tests
            use_native_plugin(True)

    def test_use_native_plugin_toggle(self):
        """Test programmatic enable/disable of plugin."""
        # Should start enabled (after previous test cleanup)
        use_native_plugin(True)
        assert is_plugin_available()

        # Disable
        use_native_plugin(False)
        assert not is_plugin_available()

        # Re-enable
        use_native_plugin(True)
        assert is_plugin_available()


class TestPluginSimilarity:
    """Tests for plugin similarity functions via expression API."""

    def setup_method(self):
        """Ensure plugin is enabled for each test."""
        use_native_plugin(True)

    def test_similarity_column_to_column(self):
        """Test similarity between two columns using plugin."""
        df = pl.DataFrame(
            {
                "left": ["hello", "world", "test", "fuzzy"],
                "right": ["hallo", "word", "testing", "fuzzi"],
            }
        )

        result = df.with_columns(score=pl.col("left").fuzzy.similarity(pl.col("right")))

        assert "score" in result.columns
        assert len(result) == 4

        # All scores should be in [0, 1]
        for score in result["score"]:
            assert 0.0 <= score <= 1.0

    def test_similarity_matches_fallback(self):
        """Verify plugin results match fallback implementation."""
        df = pl.DataFrame(
            {
                "left": ["hello", "world", "test", "John Smith"],
                "right": ["hallo", "word", "testing", "Jon Smith"],
            }
        )

        # Get plugin results
        use_native_plugin(True)
        plugin_result = df.with_columns(score=pl.col("left").fuzzy.similarity(pl.col("right")))

        # Get fallback results by computing manually
        fallback_scores = [
            fr.jaro_winkler_similarity(left, right) for left, right in zip(df["left"], df["right"])
        ]

        # Compare
        for i, (plugin_score, fallback_score) in enumerate(
            zip(plugin_result["score"], fallback_scores)
        ):
            assert abs(plugin_score - fallback_score) < 1e-10, (
                f"Mismatch at row {i}: plugin={plugin_score}, fallback={fallback_score}"
            )

    def test_similarity_different_algorithms(self):
        """Test similarity with different algorithms."""
        df = pl.DataFrame(
            {
                "left": ["kitten", "saturday"],
                "right": ["sitting", "sunday"],
            }
        )

        # Test all supported algorithms including new ones
        algorithms = [
            "levenshtein",
            "damerau_levenshtein",
            "jaro",
            "jaro_winkler",
            "ngram",
            "cosine",
            "lcs",
        ]

        for algo in algorithms:
            result = df.with_columns(
                score=pl.col("left").fuzzy.similarity(pl.col("right"), algorithm=algo)
            )

            # All scores should be in [0, 1]
            for score in result["score"]:
                assert 0.0 <= score <= 1.0, f"Invalid score for {algo}: {score}"

    def test_similarity_hamming_algorithm(self):
        """Test Hamming similarity (requires equal-length strings)."""
        df = pl.DataFrame(
            {
                "left": ["hello", "world", "test"],
                "right": ["hallo", "words", "best"],
            }
        )

        result = df.with_columns(
            score=pl.col("left").fuzzy.similarity(pl.col("right"), algorithm="hamming")
        )

        # All scores should be in [0, 1]
        for score in result["score"]:
            assert 0.0 <= score <= 1.0, f"Invalid Hamming score: {score}"

        # hello vs hallo: 1 difference out of 5 = 0.8 similarity
        assert result["score"][0] == 0.8

    def test_similarity_hamming_different_lengths(self):
        """Test Hamming returns 0 for different-length strings."""
        df = pl.DataFrame(
            {
                "left": ["hello", "hi"],
                "right": ["hallo", "hello"],  # Different lengths
            }
        )

        result = df.with_columns(
            score=pl.col("left").fuzzy.similarity(pl.col("right"), algorithm="hamming")
        )

        # Same length: valid score
        assert result["score"][0] == 0.8
        # Different lengths: should return 0.0
        assert result["score"][1] == 0.0

    def test_similarity_with_nulls(self):
        """Test similarity handles null values correctly."""
        df = pl.DataFrame(
            {
                "left": ["hello", None, "test"],
                "right": ["hallo", "world", None],
            }
        )

        result = df.with_columns(score=pl.col("left").fuzzy.similarity(pl.col("right")))

        # First row should have a valid score
        assert result["score"][0] is not None
        assert 0.0 <= result["score"][0] <= 1.0


class TestPluginIsMatch:
    """Tests for plugin is_match (threshold-based similarity)."""

    def setup_method(self):
        """Ensure plugin is enabled for each test."""
        use_native_plugin(True)

    def test_is_similar_column_to_column(self):
        """Test is_similar between two columns."""
        df = pl.DataFrame(
            {
                "left": ["hello", "world", "xyz"],
                "right": ["hallo", "word", "abc"],
            }
        )

        result = df.with_columns(
            is_match=pl.col("left").fuzzy.is_similar(pl.col("right"), min_similarity=0.8)
        )

        assert "is_match" in result.columns
        assert result["is_match"].dtype == pl.Boolean

        # "hello"/"hallo" and "world"/"word" should be similar at 0.8
        # "xyz"/"abc" should not be similar
        assert result["is_match"][0] is True or result["is_match"][1] is True
        assert result["is_match"][2] is False

    def test_is_similar_threshold_filtering(self):
        """Test that threshold correctly filters matches."""
        df = pl.DataFrame(
            {
                "left": ["test", "test", "test"],
                "right": ["test", "tset", "abcd"],  # exact, typo, different
            }
        )

        result = df.with_columns(
            match_90=pl.col("left").fuzzy.is_similar(pl.col("right"), min_similarity=0.9),
            match_50=pl.col("left").fuzzy.is_similar(pl.col("right"), min_similarity=0.5),
        )

        # Exact match should pass both thresholds
        assert result["match_90"][0] is True
        assert result["match_50"][0] is True

        # Typo might fail strict threshold but pass lenient
        # Different should fail both
        assert result["match_90"][2] is False
        assert result["match_50"][2] is False


class TestPluginDistance:
    """Tests for plugin distance functions."""

    def setup_method(self):
        """Ensure plugin is enabled for each test."""
        use_native_plugin(True)

    def test_distance_column_to_column(self):
        """Test distance between two columns."""
        df = pl.DataFrame(
            {
                "left": ["hello", "world", "test"],
                "right": ["hallo", "word", "test"],
            }
        )

        result = df.with_columns(dist=pl.col("left").fuzzy.distance(pl.col("right")))

        assert "dist" in result.columns

        # hello -> hallo = 1 edit
        assert result["dist"][0] == 1
        # world -> word = 1 edit (delete 'l')
        assert result["dist"][1] == 1
        # test -> test = 0 edits
        assert result["dist"][2] == 0

    def test_distance_matches_fallback(self):
        """Verify distance results match fallback."""
        df = pl.DataFrame(
            {
                "left": ["kitten", "saturday"],
                "right": ["sitting", "sunday"],
            }
        )

        result = df.with_columns(dist=pl.col("left").fuzzy.distance(pl.col("right")))

        # Compare with direct function calls
        expected = [fr.levenshtein(df["left"][i], df["right"][i]) for i in range(len(df))]

        for i, (actual, exp) in enumerate(zip(result["dist"], expected)):
            assert actual == exp, f"Mismatch at row {i}: got {actual}, expected {exp}"


class TestPluginPhonetic:
    """Tests for plugin phonetic functions."""

    def setup_method(self):
        """Ensure plugin is enabled for each test."""
        use_native_plugin(True)

    def test_soundex(self):
        """Test Soundex encoding via plugin."""
        df = pl.DataFrame(
            {
                "name": ["Robert", "Rupert", "Smith", "Smyth"],
            }
        )

        result = df.with_columns(soundex=pl.col("name").fuzzy.phonetic("soundex"))

        assert "soundex" in result.columns

        # Robert and Rupert should have same Soundex
        assert result["soundex"][0] == result["soundex"][1]
        # Smith and Smyth should have same Soundex
        assert result["soundex"][2] == result["soundex"][3]

    def test_metaphone(self):
        """Test Metaphone encoding via plugin."""
        df = pl.DataFrame(
            {
                "word": ["phone", "Stephen", "knight"],
            }
        )

        result = df.with_columns(metaphone=pl.col("word").fuzzy.phonetic("metaphone"))

        assert "metaphone" in result.columns

        # All should have non-empty encodings
        for val in result["metaphone"]:
            assert val is not None and len(val) > 0

    def test_phonetic_matches_fallback(self):
        """Verify phonetic results match direct function calls."""
        df = pl.DataFrame(
            {
                "name": ["John", "Jane", "Katherine", "Catherine"],
            }
        )

        # Soundex
        result_soundex = df.with_columns(code=pl.col("name").fuzzy.phonetic("soundex"))
        for i, name in enumerate(df["name"]):
            expected = fr.soundex(name)
            assert result_soundex["code"][i] == expected, (
                f"Soundex mismatch for {name}: got {result_soundex['code'][i]}, expected {expected}"
            )

        # Metaphone
        result_metaphone = df.with_columns(code=pl.col("name").fuzzy.phonetic("metaphone"))
        for i, name in enumerate(df["name"]):
            expected = fr.metaphone(name)
            assert result_metaphone["code"][i] == expected, (
                f"Metaphone mismatch for {name}: got {result_metaphone['code'][i]}, expected {expected}"
            )


class TestPluginExpressionIntegration:
    """Integration tests for plugin in Polars expressions."""

    def setup_method(self):
        """Ensure plugin is enabled for each test."""
        use_native_plugin(True)

    def test_chained_operations(self):
        """Test plugin works in chained expression operations."""
        df = pl.DataFrame(
            {
                "name": ["John Smith", "Jon Smith", "Jane Doe"],
                "ref": ["John Smith", "John Smith", "John Smith"],
            }
        )

        result = df.with_columns(
            score=pl.col("name").fuzzy.similarity(pl.col("ref")),
        ).filter(pl.col("score") > 0.8)

        # John Smith exact match and Jon Smith should pass 0.8 threshold
        assert len(result) >= 2

    def test_multiple_fuzzy_columns(self):
        """Test multiple fuzzy operations in same query."""
        df = pl.DataFrame(
            {
                "name1": ["hello", "world"],
                "name2": ["hallo", "word"],
            }
        )

        result = df.with_columns(
            sim=pl.col("name1").fuzzy.similarity(pl.col("name2")),
            dist=pl.col("name1").fuzzy.distance(pl.col("name2")),
            is_match=pl.col("name1").fuzzy.is_similar(pl.col("name2"), min_similarity=0.8),
        )

        assert "sim" in result.columns
        assert "dist" in result.columns
        assert "is_match" in result.columns

    def test_lazy_frame_support(self):
        """Test plugin works with lazy DataFrames."""
        df = pl.DataFrame(
            {
                "left": ["hello", "world"],
                "right": ["hallo", "word"],
            }
        ).lazy()

        result = df.with_columns(score=pl.col("left").fuzzy.similarity(pl.col("right"))).collect()

        assert "score" in result.columns
        assert len(result) == 2


class TestPluginPerformance:
    """Sanity checks for plugin performance characteristics."""

    def setup_method(self):
        """Ensure plugin is enabled for each test."""
        use_native_plugin(True)

    def test_medium_dataset(self):
        """Test plugin handles medium-sized datasets."""
        n = 1000
        df = pl.DataFrame(
            {
                "left": [f"string_{i}" for i in range(n)],
                "right": [f"string_{i + 1}" for i in range(n)],
            }
        )

        result = df.with_columns(score=pl.col("left").fuzzy.similarity(pl.col("right")))

        assert len(result) == n
        # All scores should be valid
        for score in result["score"]:
            assert 0.0 <= score <= 1.0

    def test_plugin_vs_fallback_consistency(self):
        """Verify plugin and fallback produce identical results."""
        df = pl.DataFrame(
            {
                "left": [f"test_{i}" for i in range(100)],
                "right": [f"test_{i + 5}" for i in range(100)],
            }
        )

        # Plugin results
        use_native_plugin(True)
        plugin_result = df.with_columns(score=pl.col("left").fuzzy.similarity(pl.col("right")))[
            "score"
        ].to_list()

        # Fallback results (compute manually)
        fallback_result = [
            fr.jaro_winkler_similarity(df["left"][i], df["right"][i]) for i in range(len(df))
        ]

        # Should be identical within floating point tolerance
        for i, (p, f) in enumerate(zip(plugin_result, fallback_result)):
            assert abs(p - f) < 1e-10, f"Row {i}: plugin={p}, fallback={f}"


class TestFallbackBehavior:
    """Tests to verify fallback works when plugin is disabled."""

    def test_similarity_with_plugin_disabled(self):
        """Test similarity works when plugin is disabled."""
        use_native_plugin(False)

        df = pl.DataFrame(
            {
                "left": ["hello", "world"],
                "right": ["hallo", "word"],
            }
        )

        result = df.with_columns(score=pl.col("left").fuzzy.similarity(pl.col("right")))

        assert "score" in result.columns
        # Should still produce valid results
        for score in result["score"]:
            assert 0.0 <= score <= 1.0

        # Re-enable for other tests
        use_native_plugin(True)

    def test_literal_comparison_uses_plugin(self):
        """Test that literal string comparisons use native plugin when available."""
        df = pl.DataFrame(
            {
                "name": ["John", "Jon", "Jane"],
            }
        )

        # Literal comparison should work with native plugin (10-50x speedup)
        result = df.with_columns(score=pl.col("name").fuzzy.similarity("John"))

        assert "score" in result.columns
        # John vs John should be 1.0
        assert result["score"][0] == 1.0
        # Jon vs John should be high (Jaro-Winkler gives ~0.933)
        assert result["score"][1] > 0.9

        # Also test is_similar with literal (use higher threshold)
        result2 = df.with_columns(
            is_match=pl.col("name").fuzzy.is_similar("John", min_similarity=0.95)
        )

        assert "is_match" in result2.columns
        assert result2["is_match"][0] is True  # John == John (1.0)
        assert result2["is_match"][1] is False  # Jon vs John (~0.933, below 0.95)
        assert result2["is_match"][2] is False  # Jane vs John (below 0.95)

    def test_literal_comparison_all_algorithms(self):
        """Test literal comparisons with all supported algorithms."""
        df = pl.DataFrame(
            {
                "name": ["kitten", "sitting", "saturday"],
            }
        )

        # Test all algorithms with literal comparison
        algorithms = [
            "levenshtein",
            "damerau_levenshtein",
            "jaro",
            "jaro_winkler",
            "ngram",
            "cosine",
            "lcs",
        ]

        for algo in algorithms:
            result = df.with_columns(
                score=pl.col("name").fuzzy.similarity("kitten", algorithm=algo)
            )

            # All scores should be in [0, 1]
            for score in result["score"]:
                assert 0.0 <= score <= 1.0, f"Invalid score for {algo}: {score}"

            # First row should be 1.0 (kitten == kitten)
            assert result["score"][0] == 1.0, f"Expected 1.0 for {algo}, got {result['score'][0]}"

    def test_literal_comparison_with_nulls(self):
        """Test literal comparisons handle nulls correctly."""
        df = pl.DataFrame(
            {
                "name": ["John", None, "Jane"],
            }
        )

        result = df.with_columns(score=pl.col("name").fuzzy.similarity("John"))

        assert result["score"][0] == 1.0  # John vs John
        assert result["score"][1] is None  # null
        assert result["score"][2] < 1.0  # Jane vs John
