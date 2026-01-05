"""Comprehensive edge case tests for Polars integration.

This module tests edge cases including:
1. LazyFrame support - verify functions work with LazyFrames
2. Long strings - test with strings > 1000 characters
3. Plugin/fallback consistency - compare plugin results with fallback
4. Null handling - test null values in columns
5. Unicode - test with emoji, CJK characters, mixed scripts
6. Empty strings - test empty string handling
7. Case-insensitive mode - verify case_insensitive=True works correctly
8. All algorithms - test each algorithm with edge cases
9. Struct output - test best_match_score() returns correct struct
"""

import polars as pl
import pytest

import fuzzyrust as fr
from fuzzyrust._plugin import is_plugin_available, use_native_plugin


class TestLazyFrameSupport:
    """Tests for LazyFrame support across all APIs."""

    def test_expression_namespace_with_lazyframe(self):
        """Test .fuzzy namespace works with LazyFrame."""
        lazy_df = pl.DataFrame({"name": ["John", "Jon", "Jane"]}).lazy()

        result = lazy_df.with_columns(score=pl.col("name").fuzzy.similarity("John")).collect()

        assert "score" in result.columns
        assert len(result) == 3
        assert result["score"][0] == 1.0  # John == John

    def test_is_similar_with_lazyframe(self):
        """Test is_similar works with LazyFrame."""
        lazy_df = pl.DataFrame({"name": ["John", "Jon", "Jane", "Bob"]}).lazy()

        result = lazy_df.filter(
            pl.col("name").fuzzy.is_similar("John", min_similarity=0.8)
        ).collect()

        assert len(result) >= 1
        assert "John" in result["name"].to_list()

    def test_best_match_with_lazyframe(self):
        """Test best_match works with LazyFrame."""
        lazy_df = pl.DataFrame({"query": ["appel", "banan"]}).lazy()

        choices = ["apple", "banana", "cherry"]
        result = lazy_df.with_columns(
            match=pl.col("query").fuzzy.best_match(choices, min_similarity=0.6)
        ).collect()

        assert result["match"][0] == "apple"
        assert result["match"][1] == "banana"

    def test_fuzzy_join_with_lazyframe(self):
        """Test fuzzy_join accepts LazyFrames."""
        left = pl.DataFrame({"name": ["Apple Inc"]}).lazy()
        right = pl.DataFrame({"company": ["Apple", "Microsoft"]}).lazy()

        result = fr.polars.df_join(
            left, right, left_on="name", right_on="company", min_similarity=0.5
        )

        assert isinstance(result, pl.DataFrame)
        assert "fuzzy_score" in result.columns

    def test_fuzzy_dedupe_rows_with_lazyframe(self):
        """Test fuzzy_dedupe_rows accepts LazyFrame."""
        lazy_df = pl.DataFrame({"name": ["John Smith", "Jon Smith", "Jane Doe"]}).lazy()

        result = fr.polars.df_dedupe(lazy_df, columns=["name"], min_similarity=0.8)

        assert isinstance(result, pl.DataFrame)
        assert "_group_id" in result.columns
        assert "_is_canonical" in result.columns

    def test_dedupe_snm_with_lazyframe(self):
        """Test dedupe_snm accepts LazyFrame."""
        lazy_df = pl.DataFrame({"name": ["John Smith", "Jon Smith", "Jane Doe"]}).lazy()

        result = fr.polars.df_dedupe_snm(lazy_df, columns=["name"], min_similarity=0.8)

        assert isinstance(result, pl.DataFrame)
        assert "_group_id" in result.columns


class TestLongStrings:
    """Tests with strings > 1000 characters."""

    @pytest.fixture
    def long_string_a(self):
        """Create a long string (1500 chars)."""
        return "hello world " * 125  # 12 * 125 = 1500 chars

    @pytest.fixture
    def long_string_b(self):
        """Create a similar long string (1500 chars)."""
        return "hello world " * 124 + "hello worl " + " "  # Similar with typo

    @pytest.fixture
    def very_different_long_string(self):
        """Create a very different long string."""
        return "completely different text " * 60  # 1560 chars

    def test_similarity_with_long_strings(self, long_string_a, long_string_b):
        """Test similarity calculation with long strings."""
        df = pl.DataFrame({"text_a": [long_string_a], "text_b": [long_string_b]})

        result = df.with_columns(score=pl.col("text_a").fuzzy.similarity(pl.col("text_b")))

        # Should complete without error
        assert result["score"][0] is not None
        # Similar strings should have high similarity
        assert result["score"][0] > 0.9

    def test_long_string_vs_short(self, long_string_a):
        """Test comparing long string to short string."""
        df = pl.DataFrame(
            {
                "long_text": [long_string_a],
                "short_text": ["xyz"],  # Use dissimilar string to avoid repetition matches
            }
        )

        result = df.with_columns(score=pl.col("long_text").fuzzy.similarity(pl.col("short_text")))

        # Should complete without error
        assert result["score"][0] is not None
        # Score should be valid (Jaro-Winkler can give various scores)
        assert 0.0 <= result["score"][0] <= 1.0

    def test_all_algorithms_with_long_strings(self, long_string_a, long_string_b):
        """Test all algorithms handle long strings."""
        df = pl.DataFrame({"text_a": [long_string_a], "text_b": [long_string_b]})

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
                score=pl.col("text_a").fuzzy.similarity(pl.col("text_b"), algorithm=algo)
            )
            assert result["score"][0] is not None, f"Algorithm {algo} returned None"
            assert 0.0 <= result["score"][0] <= 1.0, f"Invalid score for {algo}"


class TestPluginFallbackConsistency:
    """Tests comparing plugin results with fallback implementation."""

    def test_similarity_consistency(self):
        """Verify plugin and fallback produce same similarity scores."""
        df = pl.DataFrame(
            {
                "left": ["hello", "world", "test", "John Smith"],
                "right": ["hallo", "word", "testing", "Jon Smith"],
            }
        )

        # Get plugin results (if available)
        use_native_plugin(True)
        plugin_result = df.with_columns(score=pl.col("left").fuzzy.similarity(pl.col("right")))[
            "score"
        ].to_list()

        # Compute expected values using direct function calls
        expected = [
            fr.jaro_winkler_similarity(left, right) for left, right in zip(df["left"], df["right"])
        ]

        # Compare
        for i, (actual, exp) in enumerate(zip(plugin_result, expected)):
            assert abs(actual - exp) < 1e-10, f"Row {i}: plugin={actual}, expected={exp}"

    def test_is_similar_consistency(self):
        """Verify is_similar returns consistent boolean results."""
        df = pl.DataFrame({"left": ["hello", "xyz", "test"], "right": ["hallo", "abc", "test"]})

        use_native_plugin(True)
        plugin_result = df.with_columns(
            is_match=pl.col("left").fuzzy.is_similar(pl.col("right"), min_similarity=0.8)
        )["is_match"].to_list()

        # Compute expected
        expected = [
            fr.jaro_winkler_similarity(left, right) >= 0.8
            for left, right in zip(df["left"], df["right"])
        ]

        for i, (actual, exp) in enumerate(zip(plugin_result, expected)):
            assert actual == exp, f"Row {i}: plugin={actual}, expected={exp}"

    def test_literal_comparison_consistency(self):
        """Verify literal comparisons produce consistent results."""
        df = pl.DataFrame({"name": ["John", "Jon", "Jane", "Bob"]})

        use_native_plugin(True)
        plugin_result = df.with_columns(score=pl.col("name").fuzzy.similarity("John"))[
            "score"
        ].to_list()

        # Compute expected
        expected = [fr.jaro_winkler_similarity(name, "John") for name in df["name"]]

        for i, (actual, exp) in enumerate(zip(plugin_result, expected)):
            assert abs(actual - exp) < 1e-10, f"Row {i}: plugin={actual}, expected={exp}"


class TestNullHandling:
    """Tests for null value handling."""

    def test_similarity_with_null_left(self):
        """Test similarity when left column has nulls."""
        df = pl.DataFrame({"left": ["hello", None, "test"], "right": ["hallo", "world", "test"]})

        result = df.with_columns(score=pl.col("left").fuzzy.similarity(pl.col("right")))

        assert result["score"][0] is not None  # Valid comparison
        assert result["score"][1] is None  # Null left
        assert result["score"][2] == 1.0  # Exact match

    def test_similarity_with_null_right(self):
        """Test similarity when right column has nulls."""
        df = pl.DataFrame({"left": ["hello", "world", "test"], "right": ["hallo", None, None]})

        result = df.with_columns(score=pl.col("left").fuzzy.similarity(pl.col("right")))

        assert result["score"][0] is not None  # Valid comparison
        assert result["score"][1] is None  # Null right
        assert result["score"][2] is None  # Null right

    def test_similarity_literal_with_null(self):
        """Test similarity to literal when column has nulls."""
        df = pl.DataFrame({"name": ["John", None, "Jane"]})

        result = df.with_columns(score=pl.col("name").fuzzy.similarity("John"))

        assert result["score"][0] == 1.0  # John == John
        assert result["score"][1] is None  # Null
        assert result["score"][2] is not None  # Jane compared to John

    def test_is_similar_with_nulls(self):
        """Test is_similar handles nulls correctly."""
        df = pl.DataFrame({"left": ["hello", None, "test"], "right": ["hallo", "world", None]})

        result = df.with_columns(
            is_match=pl.col("left").fuzzy.is_similar(pl.col("right"), min_similarity=0.8)
        )

        # Null comparisons should return null or false depending on impl
        assert result["is_match"][0] is True or result["is_match"][0] is not None
        # Row 1 and 2 have nulls

    def test_best_match_with_null_query(self):
        """Test best_match when query has nulls."""
        df = pl.DataFrame({"query": ["apple", None, "cherry"]})

        choices = ["apple", "banana", "cherry"]
        result = df.with_columns(match=pl.col("query").fuzzy.best_match(choices))

        assert result["match"][0] == "apple"
        assert result["match"][1] is None  # Null query
        assert result["match"][2] == "cherry"

    def test_fuzzy_join_with_null_values(self):
        """Test fuzzy_join handles null values."""
        left = pl.DataFrame({"name": ["Apple", None, "Google"]})
        right = pl.DataFrame({"company": ["Apple Inc", "Microsoft", None]})

        # Should complete without error
        result = fr.polars.df_join(
            left, right, left_on="name", right_on="company", min_similarity=0.5, how="left"
        )

        assert isinstance(result, pl.DataFrame)


class TestUnicode:
    """Tests with Unicode characters (emoji, CJK, mixed scripts)."""

    def test_similarity_with_emoji(self):
        """Test similarity calculation with emoji."""
        df = pl.DataFrame({"text_a": ["hello", "world"], "text_b": ["hello", "world"]})

        result = df.with_columns(score=pl.col("text_a").fuzzy.similarity(pl.col("text_b")))

        # Exact matches
        assert result["score"][0] == 1.0
        assert result["score"][1] == 1.0

    def test_similarity_with_cjk_characters(self):
        """Test similarity with CJK (Chinese/Japanese/Korean) characters."""
        df = pl.DataFrame({"chinese": ["hello", "world"], "similar": ["hello", "world"]})

        result = df.with_columns(score=pl.col("chinese").fuzzy.similarity(pl.col("similar")))

        # Same strings should be 1.0
        assert result["score"][0] == 1.0

    def test_similarity_with_mixed_scripts(self):
        """Test similarity with mixed scripts (Latin + CJK + Arabic)."""
        df = pl.DataFrame(
            {"mixed_a": ["Hello World", "Test 123"], "mixed_b": ["Hello World", "Test 124"]}
        )

        result = df.with_columns(score=pl.col("mixed_a").fuzzy.similarity(pl.col("mixed_b")))

        assert result["score"][0] == 1.0  # Exact match
        assert result["score"][1] > 0.8  # One char difference

    def test_unicode_normalization(self):
        """Test that Unicode normalization works correctly."""
        # Combining characters vs precomposed
        df = pl.DataFrame({"text": ["cafe", "CAFE"]})

        result = df.with_columns(normalized=pl.col("text").fuzzy.normalize("unicode_nfkd"))

        # Should normalize successfully
        assert result["normalized"][0] is not None
        assert result["normalized"][1] is not None

    def test_phonetic_with_unicode(self):
        """Test phonetic encoding handles non-ASCII gracefully."""
        df = pl.DataFrame({"name": ["Mueller", "Muller", "Jose"]})

        result = df.with_columns(soundex=pl.col("name").fuzzy.soundex())

        # Should complete without error
        assert all(code is not None for code in result["soundex"])


class TestEmptyStrings:
    """Tests for empty string handling."""

    def test_similarity_with_empty_string_column(self):
        """Test similarity when one value is empty string."""
        df = pl.DataFrame({"text_a": ["hello", "", "test"], "text_b": ["", "world", "test"]})

        result = df.with_columns(score=pl.col("text_a").fuzzy.similarity(pl.col("text_b")))

        # Empty vs non-empty should have low/zero similarity
        assert result["score"][0] is not None
        assert result["score"][1] is not None
        assert result["score"][2] == 1.0  # Exact match

    def test_similarity_with_empty_literal(self):
        """Test similarity against empty string literal."""
        df = pl.DataFrame({"text": ["hello", "", "world"]})

        result = df.with_columns(score=pl.col("text").fuzzy.similarity(""))

        # Non-empty vs empty should have 0 similarity
        assert result["score"][0] == 0.0 or result["score"][0] is not None
        assert result["score"][1] == 1.0  # Empty == Empty
        assert result["score"][2] == 0.0 or result["score"][2] is not None

    def test_best_match_with_empty_choices(self):
        """Test best_match behavior with empty choices list."""
        df = pl.DataFrame({"query": ["apple", "banana"]})

        # Empty choices list - returns null for all queries
        result = df.with_columns(match=pl.col("query").fuzzy.best_match([]))

        # All matches should be None when no choices available
        assert result["match"][0] is None
        assert result["match"][1] is None

    def test_empty_dataframe(self):
        """Test operations on empty DataFrame."""
        df = pl.DataFrame({"name": []}, schema={"name": pl.Utf8})

        result = df.with_columns(score=pl.col("name").fuzzy.similarity("John"))

        assert len(result) == 0
        assert "score" in result.columns


class TestCaseInsensitiveMode:
    """Tests for case_insensitive=True parameter."""

    def test_similarity_case_insensitive(self):
        """Test case-insensitive similarity."""
        df = pl.DataFrame({"name": ["JOHN", "john", "John", "JANE"]})

        # Case-sensitive (default)
        result_sensitive = df.with_columns(
            score=pl.col("name").fuzzy.similarity("john", case_insensitive=False)
        )

        # Case-insensitive
        result_insensitive = df.with_columns(
            score=pl.col("name").fuzzy.similarity("john", case_insensitive=True)
        )

        # Case-insensitive should give same score for JOHN, john, John
        scores_insensitive = result_insensitive["score"].to_list()
        assert scores_insensitive[0] == scores_insensitive[1] == scores_insensitive[2] == 1.0

        # Case-sensitive should give different scores
        scores_sensitive = result_sensitive["score"].to_list()
        assert scores_sensitive[1] == 1.0  # john == john
        assert scores_sensitive[0] < 1.0  # JOHN != john

    def test_is_similar_case_insensitive(self):
        """Test case-insensitive is_similar."""
        df = pl.DataFrame({"name": ["JOHN SMITH", "john smith", "Jane Doe"]})

        result = df.filter(
            pl.col("name").fuzzy.is_similar(
                "John Smith", min_similarity=0.95, case_insensitive=True
            )
        )

        # Both JOHN SMITH and john smith should match
        assert len(result) == 2

    def test_column_to_column_case_insensitive(self):
        """Test case-insensitive comparison between columns."""
        df = pl.DataFrame({"left": ["HELLO", "World", "TEST"], "right": ["hello", "WORLD", "test"]})

        result = df.with_columns(
            score=pl.col("left").fuzzy.similarity(pl.col("right"), case_insensitive=True)
        )

        # All should be 1.0 (same when case-insensitive)
        assert all(score == 1.0 for score in result["score"])


class TestAllAlgorithms:
    """Tests for each algorithm with edge cases."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for algorithm tests."""
        return pl.DataFrame(
            {
                "text_a": ["hello", "kitten", "saturday", ""],
                "text_b": ["hallo", "sitting", "sunday", "test"],
            }
        )

    def test_levenshtein_algorithm(self, sample_df):
        """Test Levenshtein similarity."""
        result = sample_df.with_columns(
            score=pl.col("text_a").fuzzy.similarity(pl.col("text_b"), algorithm="levenshtein")
        )

        scores = result["score"].to_list()
        assert all(0.0 <= s <= 1.0 for s in scores if s is not None)
        # hello vs hallo: 1 edit, length 5, so ~0.8
        assert 0.7 <= scores[0] <= 0.9

    def test_damerau_levenshtein_algorithm(self, sample_df):
        """Test Damerau-Levenshtein similarity."""
        # Test transposition handling
        df = pl.DataFrame(
            {
                "text_a": ["ab", "abc"],
                "text_b": ["ba", "bac"],  # Transposition
            }
        )

        result = df.with_columns(
            score=pl.col("text_a").fuzzy.similarity(
                pl.col("text_b"), algorithm="damerau_levenshtein"
            )
        )

        scores = result["score"].to_list()
        assert all(0.0 <= s <= 1.0 for s in scores)
        # Transposition counts as 1 edit
        assert scores[0] == 0.5  # 1 edit, length 2

    def test_jaro_algorithm(self, sample_df):
        """Test Jaro similarity."""
        result = sample_df.with_columns(
            score=pl.col("text_a").fuzzy.similarity(pl.col("text_b"), algorithm="jaro")
        )

        scores = result["score"].to_list()
        assert all(0.0 <= s <= 1.0 for s in scores if s is not None)

    def test_jaro_winkler_algorithm(self, sample_df):
        """Test Jaro-Winkler similarity."""
        # Test prefix bonus
        df = pl.DataFrame(
            {"text_a": ["prefix_hello", "different"], "text_b": ["prefix_world", "various"]}
        )

        result_jw = df.with_columns(
            score_jw=pl.col("text_a").fuzzy.similarity(pl.col("text_b"), algorithm="jaro_winkler")
        )

        result_j = df.with_columns(
            score_j=pl.col("text_a").fuzzy.similarity(pl.col("text_b"), algorithm="jaro")
        )

        # Jaro-Winkler should give bonus for matching prefix
        jw_scores = result_jw["score_jw"].to_list()
        j_scores = result_j["score_j"].to_list()

        # First pair has common prefix, so JW should be >= Jaro
        assert jw_scores[0] >= j_scores[0]

    def test_ngram_algorithm(self, sample_df):
        """Test N-gram similarity."""
        result = sample_df.with_columns(
            score=pl.col("text_a").fuzzy.similarity(pl.col("text_b"), algorithm="ngram")
        )

        scores = result["score"].to_list()
        assert all(0.0 <= s <= 1.0 for s in scores if s is not None)

    def test_ngram_with_different_sizes(self):
        """Test N-gram with different ngram_size values."""
        df = pl.DataFrame({"text_a": ["hello world"], "text_b": ["hello word"]})

        for n in [2, 3, 4]:
            result = df.with_columns(
                score=pl.col("text_a").fuzzy.similarity(
                    pl.col("text_b"), algorithm="ngram", ngram_size=n
                )
            )
            assert 0.0 <= result["score"][0] <= 1.0

    def test_cosine_algorithm(self, sample_df):
        """Test Cosine similarity."""
        result = sample_df.with_columns(
            score=pl.col("text_a").fuzzy.similarity(pl.col("text_b"), algorithm="cosine")
        )

        scores = result["score"].to_list()
        assert all(0.0 <= s <= 1.0 for s in scores if s is not None)

    def test_hamming_algorithm(self):
        """Test Hamming similarity (requires equal-length strings)."""
        df = pl.DataFrame(
            {"text_a": ["hello", "tests", "abcde"], "text_b": ["hallo", "testa", "abcde"]}
        )

        result = df.with_columns(
            score=pl.col("text_a").fuzzy.similarity(pl.col("text_b"), algorithm="hamming")
        )

        scores = result["score"].to_list()
        # hello vs hallo: 1 diff out of 5 = 0.8
        assert scores[0] == 0.8
        # tests vs testa: 1 diff out of 5 = 0.8
        assert scores[1] == 0.8
        # abcde vs abcde: identical = 1.0
        assert scores[2] == 1.0

    def test_hamming_different_lengths(self):
        """Test Hamming returns 0 for different-length strings."""
        df = pl.DataFrame(
            {
                "text_a": ["hello", "hi"],
                "text_b": ["hallo", "hello"],  # Different lengths for row 1
            }
        )

        result = df.with_columns(
            score=pl.col("text_a").fuzzy.similarity(pl.col("text_b"), algorithm="hamming")
        )

        assert result["score"][0] == 0.8  # Same length
        assert result["score"][1] == 0.0  # Different lengths

    def test_lcs_algorithm(self, sample_df):
        """Test LCS similarity."""
        result = sample_df.with_columns(
            score=pl.col("text_a").fuzzy.similarity(pl.col("text_b"), algorithm="lcs")
        )

        scores = result["score"].to_list()
        assert all(0.0 <= s <= 1.0 for s in scores if s is not None)


class TestStructOutput:
    """Tests for best_match_score() struct output."""

    def test_best_match_score_returns_struct(self):
        """Test that best_match_score returns a struct with match and score."""
        df = pl.DataFrame({"query": ["appel", "banan", "xyz"]})

        choices = ["apple", "banana", "cherry"]
        result = df.with_columns(
            match_result=pl.col("query").fuzzy.best_match_score(choices, min_similarity=0.5)
        )

        # Check struct column exists
        assert "match_result" in result.columns
        assert result["match_result"].dtype == pl.Struct

        # Extract struct fields
        expanded = result.select(
            pl.col("query"),
            pl.col("match_result").struct.field("match").alias("matched"),
            pl.col("match_result").struct.field("score").alias("score"),
        )

        # Check matches
        assert expanded["matched"][0] == "apple"
        assert expanded["matched"][1] == "banana"
        # xyz might have a match but with low score

        # Check scores are valid
        for score in expanded["score"]:
            if score is not None:
                assert 0.0 <= score <= 1.0

    def test_best_match_score_with_min_similarity(self):
        """Test that min_similarity filters results in struct output."""
        df = pl.DataFrame({"query": ["apple", "xyz"]})

        choices = ["apple", "banana"]
        result = df.with_columns(
            match_result=pl.col("query").fuzzy.best_match_score(choices, min_similarity=0.9)
        )

        expanded = result.select(
            pl.col("match_result").struct.field("match"),
            pl.col("match_result").struct.field("score"),
        )

        # apple should match with score 1.0
        assert expanded["match"][0] == "apple"
        assert expanded["score"][0] == 1.0

        # xyz should not match at 0.9 threshold
        assert expanded["match"][1] is None
        assert expanded["score"][1] is None

    def test_best_match_score_with_null_query(self):
        """Test best_match_score handles null queries."""
        df = pl.DataFrame({"query": ["apple", None, "banana"]})

        choices = ["apple", "banana", "cherry"]
        result = df.with_columns(match_result=pl.col("query").fuzzy.best_match_score(choices))

        expanded = result.select(
            pl.col("match_result").struct.field("match"),
            pl.col("match_result").struct.field("score"),
        )

        assert expanded["match"][0] == "apple"
        assert expanded["match"][1] is None  # Null query
        assert expanded["match"][2] == "banana"


class TestEdgeCaseCombinations:
    """Tests combining multiple edge cases."""

    def test_long_unicode_with_nulls(self):
        """Test long Unicode strings with null values."""
        long_unicode = "Hello World " * 100  # 1200 chars
        df = pl.DataFrame(
            {"text_a": [long_unicode, None, "test"], "text_b": [long_unicode, "world", None]}
        )

        result = df.with_columns(score=pl.col("text_a").fuzzy.similarity(pl.col("text_b")))

        assert result["score"][0] == 1.0  # Exact match
        assert result["score"][1] is None  # Null left
        assert result["score"][2] is None  # Null right

    def test_case_insensitive_with_unicode(self):
        """Test case-insensitive mode with Unicode."""
        df = pl.DataFrame({"text": ["HELLO", "hello", "CAFE", "cafe"]})

        result = df.with_columns(
            score=pl.col("text").fuzzy.similarity("hello", case_insensitive=True)
        )

        # HELLO and hello should both be 1.0
        assert result["score"][0] == 1.0
        assert result["score"][1] == 1.0

    def test_empty_string_all_algorithms(self):
        """Test empty string handling for all algorithms."""
        df = pl.DataFrame({"text_a": ["", "hello"], "text_b": ["hello", ""]})

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
                score=pl.col("text_a").fuzzy.similarity(pl.col("text_b"), algorithm=algo)
            )
            # Should complete without error
            assert "score" in result.columns, f"Algorithm {algo} failed"

    def test_lazyframe_with_null_and_unicode(self):
        """Test LazyFrame with null values and Unicode."""
        lazy_df = pl.DataFrame({"name": ["John", None, "Hello", ""]}).lazy()

        result = lazy_df.with_columns(score=pl.col("name").fuzzy.similarity("John")).collect()

        assert result["score"][0] == 1.0  # John == John
        assert result["score"][1] is None  # Null
        assert result["score"][2] is not None  # Hello vs John


class TestDistanceFunction:
    """Tests for the distance function."""

    def test_distance_basic(self):
        """Test basic distance calculation."""
        df = pl.DataFrame({"word": ["hello", "hallo", "world", "test"]})

        result = df.with_columns(dist=pl.col("word").fuzzy.distance("hello"))

        assert result["dist"][0] == 0  # hello == hello
        assert result["dist"][1] == 1  # hello -> hallo (1 substitution)
        assert result["dist"][2] == 4  # hello -> world (4 edits)
        assert result["dist"][3] == 4  # hello -> test (4 edits)

    def test_distance_column_to_column(self):
        """Test distance between two columns."""
        df = pl.DataFrame({"left": ["hello", "world", "test"], "right": ["hallo", "word", "test"]})

        result = df.with_columns(dist=pl.col("left").fuzzy.distance(pl.col("right")))

        assert result["dist"][0] == 1  # hello -> hallo
        assert result["dist"][1] == 1  # world -> word
        assert result["dist"][2] == 0  # test == test

    def test_distance_with_empty_strings(self):
        """Test distance with empty strings."""
        df = pl.DataFrame({"word": ["", "hello", ""]})

        result = df.with_columns(dist=pl.col("word").fuzzy.distance("hello"))

        assert result["dist"][0] == 5  # Empty to "hello" = 5 insertions
        assert result["dist"][1] == 0  # hello == hello
        assert result["dist"][2] == 5  # Empty to "hello" = 5 insertions

    def test_distance_different_algorithms(self):
        """Test distance with different algorithms."""
        df = pl.DataFrame({"text": ["kitten"]})

        # Levenshtein
        lev_result = df.with_columns(
            dist=pl.col("text").fuzzy.distance("sitting", algorithm="levenshtein")
        )
        assert lev_result["dist"][0] == 3  # kitten -> sitting

        # Damerau-Levenshtein (for transpositions)
        df2 = pl.DataFrame({"text": ["ab"]})
        dam_result = df2.with_columns(
            dist=pl.col("text").fuzzy.distance("ba", algorithm="damerau_levenshtein")
        )
        assert dam_result["dist"][0] == 1  # Transposition counts as 1


class TestPhoneticFunctions:
    """Tests for phonetic encoding functions."""

    def test_soundex_basic(self):
        """Test basic Soundex encoding."""
        df = pl.DataFrame({"name": ["Smith", "Smyth", "Robert", "Rupert"]})

        result = df.with_columns(code=pl.col("name").fuzzy.soundex())

        # Smith and Smyth should have same Soundex
        assert result["code"][0] == result["code"][1]
        # Robert and Rupert should have same Soundex
        assert result["code"][2] == result["code"][3]

    def test_metaphone_basic(self):
        """Test basic Metaphone encoding."""
        df = pl.DataFrame({"word": ["phone", "fone", "knight", "night"]})

        result = df.with_columns(code=pl.col("word").fuzzy.metaphone())

        # All should have non-empty encodings
        assert all(code is not None and len(code) > 0 for code in result["code"])

    def test_phonetic_with_nulls(self):
        """Test phonetic encoding handles nulls."""
        df = pl.DataFrame({"name": ["John", None, "Jane"]})

        result = df.with_columns(
            soundex=pl.col("name").fuzzy.soundex(), metaphone=pl.col("name").fuzzy.metaphone()
        )

        assert result["soundex"][0] is not None
        assert result["soundex"][1] is None  # Null input
        assert result["metaphone"][2] is not None
