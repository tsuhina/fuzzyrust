"""Comprehensive tests for fuzzyrust."""

import pytest
import sys


class TestLevenshtein:
    """Tests for Levenshtein distance functions."""
    
    def test_identical_strings(self):
        import fuzzyrust as fr
        assert fr.levenshtein("hello", "hello") == 0
        assert fr.levenshtein("", "") == 0
    
    def test_empty_strings(self):
        import fuzzyrust as fr
        assert fr.levenshtein("hello", "") == 5
        assert fr.levenshtein("", "hello") == 5
    
    def test_classic_examples(self):
        import fuzzyrust as fr
        assert fr.levenshtein("kitten", "sitting") == 3
        assert fr.levenshtein("saturday", "sunday") == 3
    
    def test_unicode(self):
        import fuzzyrust as fr
        assert fr.levenshtein("café", "cafe") == 1
        assert fr.levenshtein("日本語", "日本") == 1
    
    def test_max_distance(self):
        import fuzzyrust as fr
        # Should return MAX when distance exceeds threshold
        # Note: Rust's usize::MAX (2^64-1) differs from Python's sys.maxsize (2^63-1)
        result = fr.levenshtein("abcdef", "ghijkl", max_distance=3)
        assert result > 2**63  # Very large value indicating exceeded threshold

        # Should return actual distance when within threshold
        assert fr.levenshtein("abc", "abd", max_distance=2) == 1
    
    def test_similarity(self):
        import fuzzyrust as fr
        assert fr.levenshtein_similarity("hello", "hello") == 1.0
        assert fr.levenshtein_similarity("hello", "") == 0.0
        assert 0.5 < fr.levenshtein_similarity("hello", "hallo") < 1.0


class TestDamerauLevenshtein:
    """Tests for Damerau-Levenshtein distance."""
    
    def test_transposition(self):
        import fuzzyrust as fr
        # Damerau counts transposition as 1 edit
        assert fr.damerau_levenshtein("ab", "ba") == 1
        assert fr.damerau_levenshtein("ca", "ac") == 1
        
        # Regular Levenshtein would count this as 2
        assert fr.levenshtein("ab", "ba") == 2
    
    def test_similarity(self):
        import fuzzyrust as fr
        sim = fr.damerau_levenshtein_similarity("hello", "ehllo")
        assert 0.7 < sim < 1.0


class TestJaro:
    """Tests for Jaro and Jaro-Winkler similarity."""
    
    def test_jaro_identical(self):
        import fuzzyrust as fr
        assert fr.jaro_similarity("hello", "hello") == 1.0
        assert fr.jaro_similarity("", "") == 1.0
    
    def test_jaro_different(self):
        import fuzzyrust as fr
        assert fr.jaro_similarity("abc", "xyz") == 0.0
    
    def test_jaro_classic_examples(self):
        import fuzzyrust as fr
        # Classic MARTHA/MARHTA example
        sim = fr.jaro_similarity("MARTHA", "MARHTA")
        assert 0.94 < sim < 0.95
    
    def test_jaro_winkler_prefix_boost(self):
        import fuzzyrust as fr
        jaro = fr.jaro_similarity("MARTHA", "MARHTA")
        jaro_winkler = fr.jaro_winkler_similarity("MARTHA", "MARHTA")
        # Jaro-Winkler should be higher due to common prefix
        assert jaro_winkler > jaro
    
    def test_jaro_winkler_params(self):
        import fuzzyrust as fr
        # Higher prefix weight should increase similarity for common prefixes
        default = fr.jaro_winkler_similarity("prefix_test", "prefix_best")
        higher = fr.jaro_winkler_similarity("prefix_test", "prefix_best", prefix_weight=0.2)
        assert higher >= default


class TestHamming:
    """Tests for Hamming distance."""
    
    def test_equal_length(self):
        import fuzzyrust as fr
        assert fr.hamming("karolin", "kathrin") == 3
        assert fr.hamming("abc", "abc") == 0
    
    def test_unequal_length_raises(self):
        import fuzzyrust as fr
        with pytest.raises(ValueError):
            fr.hamming("abc", "ab")


class TestNgram:
    """Tests for N-gram similarity."""
    
    def test_extract_ngrams(self):
        import fuzzyrust as fr
        ngrams = fr.extract_ngrams("abc", ngram_size=2, pad=False)
        # Use set comparison to avoid order dependency
        assert set(ngrams) == {"ab", "bc"}
        assert len(ngrams) == 2

        ngrams_padded = fr.extract_ngrams("abc", ngram_size=2, pad=True)
        assert len(ngrams_padded) == 4  # " a", "ab", "bc", "c "
    
    def test_similarity(self):
        import fuzzyrust as fr
        assert fr.ngram_similarity("abc", "abc") == 1.0
        assert fr.ngram_similarity("abc", "xyz") == 0.0

        # Partial similarity - "night" and "nacht" with bigram padding:
        # " n", "ni", "ig", "gh", "ht", "t " vs " n", "na", "ac", "ch", "ht", "t "
        # Intersection: {" n", "ht", "t "} = 3, Total: 6+6=12
        # Sørensen-Dice = 2*3/12 = 0.5
        sim = fr.ngram_similarity("night", "nacht")
        assert 0.49 < sim < 0.51


class TestPhonetic:
    """Tests for phonetic algorithms."""
    
    def test_soundex(self):
        import fuzzyrust as fr
        assert fr.soundex("Robert") == "R163"
        assert fr.soundex("Rupert") == "R163"
    
    def test_soundex_match(self):
        import fuzzyrust as fr
        assert fr.soundex_match("Robert", "Rupert")
        assert fr.soundex_match("Smith", "Smyth")
        assert not fr.soundex_match("Robert", "Rubin")
    
    def test_metaphone(self):
        import fuzzyrust as fr
        assert fr.metaphone("phone") == "FN"
    
    def test_metaphone_match(self):
        import fuzzyrust as fr
        assert fr.metaphone_match("phone", "fone")
        assert fr.metaphone_match("Stephen", "Steven")


class TestLCS:
    """Tests for Longest Common Subsequence."""
    
    def test_lcs_length(self):
        import fuzzyrust as fr
        assert fr.lcs_length("ABCDGH", "AEDFHR") == 3  # ADH
        assert fr.lcs_length("AGGTAB", "GXTXAYB") == 4  # GTAB
    
    def test_lcs_string(self):
        import fuzzyrust as fr
        assert fr.lcs_string("ABCDGH", "AEDFHR") == "ADH"
    
    def test_longest_common_substring(self):
        import fuzzyrust as fr
        assert fr.longest_common_substring("abcdef", "zbcdf") == "bcd"
        assert fr.longest_common_substring_length("abcdef", "zbcdf") == 3


class TestCosine:
    """Tests for Cosine similarity."""
    
    def test_cosine_chars(self):
        import fuzzyrust as fr
        assert fr.cosine_similarity_chars("abc", "abc") == 1.0
        assert fr.cosine_similarity_chars("abc", "def") == 0.0
    
    def test_cosine_words(self):
        import fuzzyrust as fr
        a = "the quick brown fox"
        b = "the quick brown dog"
        sim = fr.cosine_similarity_words(a, b)
        assert 0.5 < sim < 1.0


class TestBatchProcessing:
    """Tests for batch processing functions."""
    
    def test_batch_levenshtein(self):
        import fuzzyrust as fr
        strings = ["hello", "hallo", "hullo", "world"]
        results = fr.batch_levenshtein(strings, "hello")
        # Results are now MatchResult objects
        assert results[0].score == 1.0  # hello (exact match)
        assert results[0].text == "hello"
        assert results[1].text == "hallo"
        assert results[3].text == "world"

    def test_batch_jaro_winkler(self):
        import fuzzyrust as fr
        strings = ["hello", "hallo", "world"]
        results = fr.batch_jaro_winkler(strings, "hello")
        # Results are now MatchResult objects
        assert results[0].score == 1.0
        assert results[1].score > results[2].score

    def test_find_best_matches(self):
        import fuzzyrust as fr
        strings = ["apple", "apply", "banana", "application"]
        results = fr.find_best_matches(strings, "appel", limit=2, min_similarity=0.0)
        assert len(results) == 2
        # Results are now MatchResult objects
        assert results[0].text == "apple"  # Best match
        assert results[0].score >= results[1].score  # Sorted by score


class TestBkTree:
    """Tests for BK-tree index."""
    
    def test_basic_operations(self):
        import fuzzyrust as fr
        tree = fr.BkTree()
        tree.add("hello")
        tree.add("hallo")
        tree.add("hullo")
        assert len(tree) == 3
        assert tree.contains("hello")
        assert not tree.contains("helloo")
    
    def test_add_all(self):
        import fuzzyrust as fr
        tree = fr.BkTree()
        tree.add_all(["hello", "world", "test"])
        assert len(tree) == 3
    
    def test_search(self):
        import fuzzyrust as fr
        tree = fr.BkTree()
        tree.add_all(["book", "books", "boo", "cook", "cake"])
        results = tree.search("book", max_distance=1)
        # Results are now SearchResult objects
        texts = [r.text for r in results]
        assert "book" in texts
        assert "books" in texts
        assert "boo" in texts
        assert "cook" in texts
        assert "cake" not in texts

    def test_find_nearest(self):
        import fuzzyrust as fr
        tree = fr.BkTree()
        tree.add_all(["apple", "application", "apply", "banana"])
        results = tree.find_nearest("appli", k=2)
        assert len(results) == 2
        # Results are SearchResult objects
        assert all(hasattr(r, 'text') and hasattr(r, 'score') for r in results)

    def test_with_data(self):
        import fuzzyrust as fr
        tree = fr.BkTree()
        tree.add_with_data("hello", 42)
        tree.add_with_data("world", 99)
        results = tree.search("hello", max_distance=0)
        # Results are SearchResult objects
        assert results[0].data == 42
    
    def test_damerau_mode(self):
        import fuzzyrust as fr
        tree_lev = fr.BkTree(use_damerau=False)
        tree_dam = fr.BkTree(use_damerau=True)
        
        tree_lev.add("ab")
        tree_dam.add("ab")
        
        # "ba" is distance 2 in Levenshtein, 1 in Damerau
        results_lev = tree_lev.search("ba", max_distance=1)
        results_dam = tree_dam.search("ba", max_distance=1)
        
        assert len(results_lev) == 0
        assert len(results_dam) == 1


class TestNgramIndex:
    """Tests for N-gram index."""

    def test_basic_operations(self):
        import fuzzyrust as fr
        index = fr.NgramIndex(ngram_size=2)
        id1 = index.add("hello")
        id2 = index.add("world")
        assert len(index) == 2
        assert id1 == 0
        assert id2 == 1

    def test_search(self):
        import fuzzyrust as fr
        index = fr.NgramIndex(ngram_size=2)
        index.add_all(["hello", "hallo", "hullo", "world", "help"])
        results = index.search("hello", algorithm="jaro_winkler", min_similarity=0.8)
        # Results are now SearchResult objects
        texts = [r.text for r in results]
        assert "hello" in texts

    def test_search_algorithms(self):
        import fuzzyrust as fr
        index = fr.NgramIndex(ngram_size=2)
        index.add_all(["test", "text", "best"])

        for algo in ["jaro_winkler", "jaro", "levenshtein", "ngram", "trigram"]:
            results = index.search("test", algorithm=algo)
            assert len(results) > 0

    def test_batch_search(self):
        import fuzzyrust as fr
        index = fr.NgramIndex(ngram_size=2)
        index.add_all(["apple", "banana", "cherry"])
        results = index.batch_search(["apple", "banan"], algorithm="jaro_winkler")
        assert len(results) == 2
        # Each result is a list of SearchResult objects
        assert all(isinstance(r, list) for r in results)

    def test_with_data(self):
        import fuzzyrust as fr
        index = fr.NgramIndex(ngram_size=2)
        index.add_with_data("product-123", 42)
        results = index.search("product-123", min_similarity=0.9)
        # Results are SearchResult objects
        assert results[0].data == 42


class TestHybridIndex:
    """Tests for Hybrid index."""
    
    def test_basic_operations(self):
        import fuzzyrust as fr
        index = fr.HybridIndex(ngram_size=3)
        index.add_all(["hello", "world", "test"])
        assert len(index) == 3
    
    def test_search(self):
        import fuzzyrust as fr
        index = fr.HybridIndex()
        index.add_all(["apple", "application", "apply", "banana"])
        results = index.search("appel", min_similarity=0.7, limit=2)
        assert len(results) <= 2


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


class TestNgramEdgeCases:
    """Tests for n-gram edge cases including n=0."""

    def test_ngram_n_zero(self):
        """Test that ngram_size=0 returns 0.0 similarity (no valid n-grams)."""
        import fuzzyrust as fr
        # ngram_size=0 should return 0.0 as no valid n-grams can be extracted
        assert fr.ngram_similarity("hello", "hello", ngram_size=0) == 0.0
        assert fr.ngram_similarity("abc", "xyz", ngram_size=0) == 0.0

    def test_ngram_short_strings(self):
        """Test n-gram behavior with strings shorter than ngram_size."""
        import fuzzyrust as fr
        # Short strings without padding return empty n-grams
        ngrams = fr.extract_ngrams("a", ngram_size=3, pad=False)
        assert ngrams == []

    def test_ngram_jaccard_n_zero(self):
        """Test Jaccard similarity with ngram_size=0."""
        import fuzzyrust as fr
        assert fr.ngram_jaccard("hello", "hello", ngram_size=0) == 0.0

    def test_cosine_ngrams_n_zero(self):
        """Test cosine n-gram similarity with ngram_size=0."""
        import fuzzyrust as fr
        assert fr.cosine_similarity_ngrams("hello", "hello", ngram_size=0) == 0.0


class TestJaroWinklerValidation:
    """Tests for Jaro-Winkler prefix_weight validation."""

    def test_prefix_weight_too_high_raises_error(self):
        """Test that prefix_weight > 0.25 raises ValueError."""
        import fuzzyrust as fr
        import pytest
        # Prefix weight > 0.25 should raise ValueError
        with pytest.raises(ValueError, match="prefix_weight must be in range"):
            fr.jaro_winkler_similarity("hello", "hello", prefix_weight=1.0)
        with pytest.raises(ValueError, match="prefix_weight must be in range"):
            fr.jaro_winkler_similarity("hello", "hello", prefix_weight=0.26)

    def test_prefix_weight_negative_raises_error(self):
        """Test that negative prefix_weight raises ValueError."""
        import fuzzyrust as fr
        import pytest
        # Negative weight should raise ValueError
        with pytest.raises(ValueError, match="prefix_weight must be in range"):
            fr.jaro_winkler_similarity("hello", "hallo", prefix_weight=-1.0)
        with pytest.raises(ValueError, match="prefix_weight must be in range"):
            fr.jaro_winkler_similarity("hello", "hallo", prefix_weight=-0.01)

    def test_prefix_weight_valid_range(self):
        """Test prefix_weight within valid range produces expected boost."""
        import fuzzyrust as fr
        jaro = fr.jaro_similarity("prefix_test", "prefix_best")
        jaro_winkler = fr.jaro_winkler_similarity("prefix_test", "prefix_best", prefix_weight=0.1)
        # With common prefix, Jaro-Winkler should be higher
        assert jaro_winkler >= jaro

    def test_prefix_weight_boundary_values(self):
        """Test prefix_weight at boundary values (0 and 0.25)."""
        import fuzzyrust as fr
        # Boundary values should work
        sim_zero = fr.jaro_winkler_similarity("hello", "hello", prefix_weight=0.0)
        assert 0.0 <= sim_zero <= 1.0
        sim_max = fr.jaro_winkler_similarity("hello", "hello", prefix_weight=0.25)
        assert 0.0 <= sim_max <= 1.0


class TestUnicodeEdgeCases:
    """Tests for Unicode edge cases."""

    def test_emoji(self):
        """Test handling of emoji characters."""
        import fuzzyrust as fr
        # Single emoji - should be treated as one character
        assert fr.levenshtein("👋", "👋") == 0
        assert fr.levenshtein("👋", "🖐") == 1

        # Emoji in strings
        assert fr.levenshtein("hello 👋", "hello 👋") == 0
        assert fr.levenshtein("hello 👋", "hello 🖐") == 1
        assert fr.levenshtein("hello 👋", "hello") == 2  # space + emoji

        # Multiple emoji
        assert fr.levenshtein("😀😃😄", "😀😃😄") == 0
        assert fr.levenshtein("😀😃😄", "😀😃😁") == 1

        # Emoji similarity
        sim = fr.jaro_winkler_similarity("hello 👋", "hello 🖐")
        assert 0.8 < sim < 1.0  # High similarity, only one char different

        # Complex emoji (with skin tone modifiers) - treated as single unit
        # Note: This tests handling of multi-codepoint emoji
        assert fr.levenshtein("👋🏻", "👋🏻") == 0

    def test_combining_characters(self):
        """Test handling of combining characters."""
        import fuzzyrust as fr
        # Precomposed vs decomposed forms may differ
        # This tests that we handle multi-byte UTF-8 correctly
        # café vs cafe (accented e)
        assert fr.levenshtein("café", "cafe") == 1

    def test_cjk_characters(self):
        """Test handling of CJK characters."""
        import fuzzyrust as fr
        # Japanese: nihon vs nippon (日本 vs 日本語)
        assert fr.levenshtein("日本", "日本語") == 1
        # Chinese characters
        assert fr.levenshtein("你好", "你好吗") == 1

    def test_rtl_text(self):
        """Test handling of right-to-left text."""
        import fuzzyrust as fr
        # Hebrew text: shalom vs shalom
        assert fr.levenshtein("שלום", "שלום") == 0
        # Arabic text: marhaba vs marhaba with one char different
        assert fr.levenshtein("مرحبا", "مرحب") == 1


class TestErrorHandling:
    """Tests for error handling and invalid inputs."""

    def test_invalid_algorithm_find_best_matches(self):
        """Test that invalid algorithm raises ValueError."""
        import fuzzyrust as fr
        with pytest.raises(ValueError, match="Unknown algorithm"):
            fr.find_best_matches(["hello"], "hello", algorithm="invalid_algo")

    def test_invalid_algorithm_ngram_index_search(self):
        """Test that invalid algorithm in NgramIndex.search raises ValueError."""
        import fuzzyrust as fr
        index = fr.NgramIndex(ngram_size=2)
        index.add("test")
        with pytest.raises(ValueError, match="Unknown algorithm"):
            index.search("test", algorithm="invalid_algo")

    def test_invalid_algorithm_hybrid_index_search(self):
        """Test that invalid algorithm in HybridIndex.search raises ValueError."""
        import fuzzyrust as fr
        index = fr.HybridIndex()
        index.add("test")
        with pytest.raises(ValueError, match="Unknown algorithm"):
            index.search("test", algorithm="invalid_algo")

    def test_hamming_unequal_length(self):
        """Test that Hamming distance raises ValueError for unequal length strings."""
        import fuzzyrust as fr
        with pytest.raises(ValueError):
            fr.hamming("abc", "ab")
        with pytest.raises(ValueError):
            fr.hamming("a", "abc")


class TestLCSEdgeCases:
    """Tests for LCS edge cases."""

    def test_lcs_empty_strings(self):
        """Test LCS with empty strings."""
        import fuzzyrust as fr
        assert fr.lcs_length("", "") == 0
        assert fr.lcs_length("abc", "") == 0
        assert fr.lcs_length("", "abc") == 0
        assert fr.lcs_string("", "abc") == ""

    def test_lcs_no_common(self):
        """Test LCS with no common subsequence."""
        import fuzzyrust as fr
        assert fr.lcs_length("abc", "xyz") == 0
        assert fr.lcs_string("abc", "xyz") == ""

    def test_longest_common_substring_empty(self):
        """Test longest common substring with empty strings."""
        import fuzzyrust as fr
        assert fr.longest_common_substring_length("", "") == 0
        assert fr.longest_common_substring("abc", "") == ""


class TestConvenienceAliases:
    """Tests for convenience aliases."""

    def test_edit_distance_alias(self):
        """Test that edit_distance is an alias for levenshtein."""
        import fuzzyrust as fr
        assert fr.edit_distance("kitten", "sitting") == fr.levenshtein("kitten", "sitting")

    def test_similarity_alias(self):
        """Test that similarity is an alias for jaro_winkler_similarity."""
        import fuzzyrust as fr
        assert fr.similarity("hello", "hallo") == fr.jaro_winkler_similarity("hello", "hallo")


# =============================================================================
# Property-Based Tests using Hypothesis
# =============================================================================

try:
    from hypothesis import given, strategies as st, settings, assume
    import string
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


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

    @given(st.text(min_size=0, max_size=30),
           st.text(min_size=0, max_size=30),
           st.text(min_size=0, max_size=30))
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
        assert 0.0 <= sim <= 1.0

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

    @given(st.text(alphabet=string.ascii_letters, min_size=1, max_size=20),
           st.text(alphabet=string.ascii_letters, min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_soundex_deterministic(self, a, b):
        """Soundex encoding is deterministic."""
        import fuzzyrust as fr
        # Same input always produces same output
        assert fr.soundex(a) == fr.soundex(a)
        # Match is symmetric
        assert fr.soundex_match(a, b) == fr.soundex_match(b, a)

    @given(st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=20),
           st.text(min_size=1, max_size=30))
    @settings(max_examples=30)
    def test_batch_levenshtein_consistency(self, strings, query):
        """Batch Levenshtein matches individual calculations."""
        import fuzzyrust as fr
        batch_results = fr.batch_levenshtein(strings, query)
        individual_results = [fr.levenshtein_similarity(s, query) for s in strings]
        # batch_levenshtein returns MatchResult with .score = normalized similarity (0.0-1.0)
        assert [r.score for r in batch_results] == individual_results


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
        return [''.join(random.choices(string.ascii_lowercase, k=random.randint(5, 20)))
                for _ in range(1000)]

    @pytest.fixture
    def large_string_list(self):
        """Generate large list for stress testing."""
        import random
        random.seed(42)
        return [''.join(random.choices(string.ascii_lowercase, k=10))
                for _ in range(10000)]

    def test_benchmark_levenshtein(self, benchmark):
        """Benchmark Levenshtein distance calculation."""
        import fuzzyrust as fr
        result = benchmark(fr.levenshtein, "kitten", "sitting")
        assert result == 3

    def test_benchmark_jaro_winkler(self, benchmark):
        """Benchmark Jaro-Winkler similarity calculation."""
        import fuzzyrust as fr
        result = benchmark(fr.jaro_winkler_similarity, "hello", "hallo")
        assert 0.0 <= result <= 1.0

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
        assert isinstance(result, list)

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
        assert isinstance(result, list)

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


# =============================================================================
# Threading and Concurrent Access Tests
# =============================================================================

import threading
import concurrent.futures


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
                    s2 = f"string_{thread_id}_{i+1}"
                    fr.levenshtein(s1, s2)
                    fr.jaro_winkler_similarity(s1, s2)
                    fr.ngram_similarity(s1, s2)
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=compute_similarities, args=(i,))
                   for i in range(10)]

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

        threads = [threading.Thread(target=thread_with_index, args=(i,))
                   for i in range(5)]

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
            fr.levenshtein("hello", "hallo")

        # If we get here without memory leak, test passes
        assert True

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


# =============================================================================
# Batch Operation Edge Case Tests
# =============================================================================

class TestBatchEdgeCases:
    """Edge case tests for batch operations."""

    def test_batch_levenshtein_empty_list(self):
        """Test batch_levenshtein with empty list."""
        import fuzzyrust as fr
        result = fr.batch_levenshtein([], "hello")
        assert result == []

    def test_batch_levenshtein_empty_query(self):
        """Test batch_levenshtein with empty query."""
        import fuzzyrust as fr
        strings = ["hello", "world"]
        result = fr.batch_levenshtein(strings, "")
        # Empty string vs non-empty = distance equals length, normalized similarity = 0.0
        assert [r.score for r in result] == [0.0, 0.0]

    def test_batch_levenshtein_empty_strings_in_list(self):
        """Test batch_levenshtein with empty strings in list."""
        import fuzzyrust as fr
        strings = ["", "a", ""]
        result = fr.batch_levenshtein(strings, "a")
        # "" vs "a" = distance 1, normalized = 0.0; "a" vs "a" = exact match = 1.0
        assert [r.score for r in result] == [0.0, 1.0, 0.0]

    def test_batch_jaro_winkler_empty_list(self):
        """Test batch_jaro_winkler with empty list."""
        import fuzzyrust as fr
        result = fr.batch_jaro_winkler([], "hello")
        assert result == []

    def test_batch_jaro_winkler_unicode(self):
        """Test batch_jaro_winkler with Unicode strings."""
        import fuzzyrust as fr
        strings = ["cafe", "café", "caf"]
        result = fr.batch_jaro_winkler(strings, "café")
        assert len(result) == 3
        # batch_jaro_winkler returns MatchResult objects with .score attribute
        assert all(0.0 <= r.score <= 1.0 for r in result)

    def test_find_best_matches_empty_list(self):
        """Test find_best_matches with empty list."""
        import fuzzyrust as fr
        result = fr.find_best_matches([], "hello")
        assert result == []

    def test_find_best_matches_limit_zero(self):
        """Test find_best_matches with limit=0."""
        import fuzzyrust as fr
        strings = ["hello", "world"]
        result = fr.find_best_matches(strings, "hello", limit=0)
        assert result == []

    def test_find_best_matches_min_score_one(self):
        """Test find_best_matches with min_similarity=1.0 (exact matches only)."""
        import fuzzyrust as fr
        strings = ["hello", "hallo", "hello"]
        result = fr.find_best_matches(strings, "hello", min_similarity=1.0)
        # Note: results are deduplicated, so only one "hello" match
        assert len(result) >= 1
        # Results are MatchResult objects
        assert all(r.score == 1.0 for r in result)

    def test_find_best_matches_all_algorithms(self):
        """Test find_best_matches with all valid algorithms."""
        import fuzzyrust as fr
        strings = ["test", "text", "best"]
        algorithms = ["levenshtein", "damerau_levenshtein", "jaro",
                      "jaro_winkler", "ngram", "bigram", "trigram", "lcs", "cosine"]

        for algo in algorithms:
            result = fr.find_best_matches(strings, "test", algorithm=algo, min_similarity=0.0)
            assert len(result) >= 0
            # Results are MatchResult objects
            assert all(0.0 <= r.score <= 1.0 for r in result)

    def test_ngram_index_batch_search_empty_queries(self):
        """Test NgramIndex.batch_search with empty query list."""
        import fuzzyrust as fr
        index = fr.NgramIndex(ngram_size=2)
        index.add_all(["hello", "world"])
        result = index.batch_search([])
        assert result == []

    def test_bktree_search_max_distance_zero(self):
        """Test BK-tree search with max_distance=0 (exact matches only)."""
        import fuzzyrust as fr
        tree = fr.BkTree()
        tree.add_all(["hello", "hallo", "hello"])
        result = tree.search("hello", max_distance=0)
        # Note: BK-tree may store duplicates separately depending on implementation
        assert len(result) >= 1
        # Results are SearchResult objects
        assert all(r.text == "hello" and r.distance == 0 for r in result)

    def test_bktree_find_nearest_k_zero(self):
        """Test BK-tree find_nearest with k=0."""
        import fuzzyrust as fr
        tree = fr.BkTree()
        tree.add_all(["hello", "world"])
        result = tree.find_nearest("hello", k=0)
        assert result == []

    def test_bktree_find_nearest_large_k(self):
        """Test BK-tree find_nearest with k larger than tree size."""
        import fuzzyrust as fr
        tree = fr.BkTree()
        tree.add_all(["hello", "world"])
        result = tree.find_nearest("hello", k=100)
        assert len(result) == 2  # Can only return what exists


class TestPhoneticEdgeCases:
    """Edge case tests for phonetic algorithms."""

    def test_soundex_empty_string(self):
        """Test Soundex with empty string."""
        import fuzzyrust as fr
        result = fr.soundex("")
        # Empty string should return empty or default code
        assert isinstance(result, str)

    def test_soundex_single_char(self):
        """Test Soundex with single character."""
        import fuzzyrust as fr
        result = fr.soundex("A")
        assert isinstance(result, str)
        assert len(result) == 4  # Soundex always returns 4 chars
        assert result.startswith("A")

    def test_metaphone_empty_string(self):
        """Test Metaphone with empty string."""
        import fuzzyrust as fr
        result = fr.metaphone("")
        assert result == ""

    def test_metaphone_single_char(self):
        """Test Metaphone with single character."""
        import fuzzyrust as fr
        result = fr.metaphone("A")
        assert isinstance(result, str)

    def test_soundex_numbers_only(self):
        """Test Soundex with numbers only."""
        import fuzzyrust as fr
        result = fr.soundex("123")
        assert isinstance(result, str)

    def test_metaphone_unicode(self):
        """Test Metaphone with Unicode characters."""
        import fuzzyrust as fr
        result = fr.metaphone("café")
        assert isinstance(result, str)


class TestVeryLongStringProtection:
    """Tests verifying very long strings don't cause DoS."""

    def test_levenshtein_long_strings(self):
        """Test that Levenshtein handles long strings without hanging."""
        import fuzzyrust as fr
        long_a = "a" * 1000
        long_b = "b" * 1000
        # Should complete without hanging (space-efficient algorithm)
        result = fr.levenshtein(long_a, long_b)
        assert result == 1000

    def test_damerau_levenshtein_long_strings(self):
        """Test Damerau-Levenshtein with long strings."""
        import fuzzyrust as fr
        long_a = "a" * 1000
        long_b = "b" * 1000
        # Should complete without hanging
        result = fr.damerau_levenshtein(long_a, long_b)
        assert isinstance(result, int)

    def test_jaro_similarity_long_strings(self):
        """Test Jaro similarity with long strings."""
        import fuzzyrust as fr
        long_a = "a" * 1000
        long_b = "b" * 1000
        result = fr.jaro_similarity(long_a, long_b)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_lcs_length_long_strings(self):
        """Test LCS length with long strings (space-efficient)."""
        import fuzzyrust as fr
        long_a = "a" * 1000
        long_b = "a" * 500 + "b" * 500
        result = fr.lcs_length(long_a, long_b)
        assert result == 500

    def test_lcs_string_very_long_returns_empty(self):
        """Test lcs_string returns empty for very long strings to prevent DoS."""
        import fuzzyrust as fr
        # Strings longer than 10000 should return empty (protection)
        long_a = "a" * 15000
        long_b = "a" * 15000
        result = fr.lcs_string(long_a, long_b)
        # Should return empty due to DoS protection
        assert result == ""

    def test_longest_common_substring_very_long_returns_empty(self):
        """Test longest_common_substring returns empty for very long strings."""
        import fuzzyrust as fr
        long_a = "a" * 15000
        long_b = "a" * 15000
        result = fr.longest_common_substring(long_a, long_b)
        # Should return empty due to DoS protection
        assert result == ""

    def test_ngram_similarity_long_strings(self):
        """Test n-gram similarity with long strings."""
        import fuzzyrust as fr
        long_a = "a" * 1000
        long_b = "a" * 1000
        result = fr.ngram_similarity(long_a, long_b, ngram_size=2)
        assert result == 1.0

    def test_bktree_add_long_strings(self):
        """Test BK-tree can add moderately long strings."""
        import fuzzyrust as fr
        tree = fr.BkTree()
        long_str = "a" * 500
        tree.add(long_str)
        assert len(tree) == 1
        assert tree.contains(long_str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
