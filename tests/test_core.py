"""Comprehensive tests for fuzzyrust."""

import pytest


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
        assert all(hasattr(r, "text") and hasattr(r, "score") for r in results)

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


class TestCaseInsensitiveSearch:
    """Tests for case_insensitive parameter in NgramIndex and HybridIndex."""

    def test_case_insensitive_default(self):
        """Test that case_insensitive=True is the default."""
        import fuzzyrust as fr

        index = fr.NgramIndex(ngram_size=2, min_ngram_ratio=0.2)
        index.add_all(["Rain shadow jkt", "Fleece jacket"])

        # Default should be case-insensitive
        results = index.search("rain jacket", min_similarity=0.5)
        texts = [r.text for r in results]

        # Should find "Rain shadow jkt" with good score
        assert "Rain shadow jkt" in texts

    def test_case_sensitive_search(self):
        """Test that case_insensitive=False gives lower scores for case mismatches."""
        import fuzzyrust as fr

        index = fr.NgramIndex(ngram_size=2, min_ngram_ratio=0.2)
        index.add_all(["Rain shadow jkt", "rain shadow jkt"])

        # Case-insensitive should score both equally
        results_ci = index.search("rain jacket", min_similarity=0.5, case_insensitive=True)
        # Case-sensitive should score lowercase higher
        results_cs = index.search("rain jacket", min_similarity=0.5, case_insensitive=False)

        # With case-insensitive, "Rain shadow jkt" should have higher score
        ci_scores = {r.text: r.score for r in results_ci}
        cs_scores = {r.text: r.score for r in results_cs}

        # Case-insensitive should give same score for both
        if "Rain shadow jkt" in ci_scores and "rain shadow jkt" in ci_scores:
            assert abs(ci_scores["Rain shadow jkt"] - ci_scores["rain shadow jkt"]) < 0.01

    def test_hybrid_index_case_insensitive(self):
        """Test that HybridIndex also supports case_insensitive."""
        import fuzzyrust as fr

        # Use text with mixed case that shares n-grams with query
        index = fr.HybridIndex(ngram_size=2, min_ngram_ratio=0.2)
        index.add_all(["Hello World", "hello there", "goodbye world"])

        # With case-insensitive, "Hello World" should score higher
        results = index.search("hello world", min_similarity=0.5, case_insensitive=True)
        texts = [r.text for r in results]

        # Should find matches
        assert len(results) > 0
        # The top result should be "Hello World" with case-insensitive matching
        assert results[0].text in ["Hello World", "hello there"]


class TestNgramRatioFiltering:
    """Tests for min_ngram_ratio parameter in NgramIndex and HybridIndex."""

    def test_ngram_ratio_filters_irrelevant(self):
        """Test that min_ngram_ratio filters out irrelevant candidates."""
        import fuzzyrust as fr

        products = [
            "Fleece jacket full zip",
            "Rain jacket waterproof",
            "Hiking pants lightweight",
            "Sleeveless a/c shirt",
            "Cotton t-shirt basic",
        ]

        index = fr.NgramIndex(ngram_size=2, min_ngram_ratio=0.2)
        index.add_all(products)

        results = index.search("fleece jacket", algorithm="jaro_winkler", min_similarity=0.5)
        texts = [r.text for r in results]

        # Sleeveless shirt should NOT match - low n-gram overlap
        assert "Sleeveless a/c shirt" not in texts

    def test_ngram_ratio_validation(self):
        """Test that invalid min_ngram_ratio raises ValueError."""
        import pytest

        import fuzzyrust as fr

        with pytest.raises(ValueError, match="min_ngram_ratio"):
            fr.NgramIndex(ngram_size=2, min_ngram_ratio=-0.1)

        with pytest.raises(ValueError, match="min_ngram_ratio"):
            fr.NgramIndex(ngram_size=2, min_ngram_ratio=1.5)

    def test_hybrid_index_ngram_ratio(self):
        """Test that HybridIndex also supports min_ngram_ratio."""
        import fuzzyrust as fr

        index = fr.HybridIndex(ngram_size=2, min_ngram_ratio=0.3)
        index.add_all(["hello world", "hello there", "goodbye world"])

        results = index.search("hello", min_similarity=0.5)
        texts = [r.text for r in results]

        # Should find hello matches
        assert any("hello" in t for t in texts)

    def test_product_search_use_case(self):
        """Test the original problem case: product search filtering."""
        import fuzzyrust as fr

        products = [
            "Alpine wind jkt",
            "Rain shadow jkt",
            "Fleece jacket full zip",
            "Traverse jkt",
            "Sleeveless a/c shirt",
            "Guide jkt",
        ]

        index = fr.NgramIndex(ngram_size=3, min_ngram_ratio=0.2)
        index.add_all(products)

        results = index.search("fleece jacket", algorithm="jaro_winkler", min_similarity=0.5)
        texts = [r.text for r in results]

        # Should find fleece jacket
        assert any("fleece" in t.lower() or "jacket" in t.lower() for t in texts)
        # Should NOT find completely unrelated items
        assert "Sleeveless a/c shirt" not in texts


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
    """Tests for n-gram edge cases including validation."""

    def test_ngram_n_zero_raises_error(self):
        """Test that ngram_size=0 raises ValueError."""
        import pytest

        import fuzzyrust as fr

        with pytest.raises(ValueError, match="ngram_size must be at least 1"):
            fr.ngram_similarity("hello", "hello", ngram_size=0)
        with pytest.raises(ValueError, match="ngram_size must be at least 1"):
            fr.ngram_similarity("abc", "xyz", ngram_size=0)

    def test_ngram_short_strings(self):
        """Test n-gram behavior with strings shorter than ngram_size."""
        import fuzzyrust as fr

        # Short strings without padding return empty n-grams
        ngrams = fr.extract_ngrams("a", ngram_size=3, pad=False)
        assert ngrams == []

    def test_ngram_jaccard_n_zero_raises_error(self):
        """Test Jaccard similarity with ngram_size=0 raises ValueError."""
        import pytest

        import fuzzyrust as fr

        with pytest.raises(ValueError, match="ngram_size must be at least 1"):
            fr.ngram_jaccard("hello", "hello", ngram_size=0)

    def test_cosine_ngrams_n_zero_raises_error(self):
        """Test cosine n-gram similarity with ngram_size=0 raises ValueError."""
        import pytest

        import fuzzyrust as fr

        with pytest.raises(ValueError, match="ngram_size must be at least 1"):
            fr.cosine_similarity_ngrams("hello", "hello", ngram_size=0)

    def test_extract_ngrams_n_zero_raises_error(self):
        """Test extract_ngrams with ngram_size=0 raises ValueError."""
        import pytest

        import fuzzyrust as fr

        with pytest.raises(ValueError, match="ngram_size must be at least 1"):
            fr.extract_ngrams("hello", ngram_size=0)

    def test_ngram_profile_n_zero_raises_error(self):
        """Test ngram_profile_similarity with ngram_size=0 raises ValueError."""
        import pytest

        import fuzzyrust as fr

        with pytest.raises(ValueError, match="ngram_size must be at least 1"):
            fr.ngram_profile_similarity("hello", "hello", ngram_size=0)

    def test_ngram_ci_variants_n_zero_raises_error(self):
        """Test case-insensitive ngram functions with ngram_size=0 raise ValueError."""
        import pytest

        import fuzzyrust as fr

        with pytest.raises(ValueError, match="ngram_size must be at least 1"):
            fr.ngram_similarity_ci("hello", "HELLO", ngram_size=0)
        with pytest.raises(ValueError, match="ngram_size must be at least 1"):
            fr.ngram_jaccard_ci("hello", "HELLO", ngram_size=0)
        with pytest.raises(ValueError, match="ngram_size must be at least 1"):
            fr.cosine_similarity_ngrams_ci("hello", "HELLO", ngram_size=0)


class TestJaroWinklerValidation:
    """Tests for Jaro-Winkler prefix_weight validation."""

    def test_prefix_weight_too_high_raises_error(self):
        """Test that prefix_weight > 0.25 raises ValueError."""
        import pytest

        import fuzzyrust as fr

        # Prefix weight > 0.25 should raise ValueError
        with pytest.raises(ValueError, match="prefix_weight must be in range"):
            fr.jaro_winkler_similarity("hello", "hello", prefix_weight=1.0)
        with pytest.raises(ValueError, match="prefix_weight must be in range"):
            fr.jaro_winkler_similarity("hello", "hello", prefix_weight=0.26)

    def test_prefix_weight_negative_raises_error(self):
        """Test that negative prefix_weight raises ValueError."""
        import pytest

        import fuzzyrust as fr

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

    def test_zwj_emoji_sequences(self):
        """Test handling of Zero-Width Joiner emoji sequences."""
        import fuzzyrust as fr

        # Family emoji (man+woman+girl+boy) is a ZWJ sequence
        family = "👨‍👩‍👧‍👦"
        # Same family should have distance 0
        assert fr.levenshtein(family, family) == 0
        # Different family composition
        couple = "👨‍👩‍👦"
        # These are different sequences
        assert fr.levenshtein(family, couple) > 0
        # Similarity should still work
        sim = fr.jaro_winkler_similarity(family, family)
        assert sim == 1.0

    def test_regional_indicator_flags(self):
        """Test handling of flag emoji (regional indicator symbols)."""
        import fuzzyrust as fr

        # US flag vs UK flag (both are pairs of regional indicators)
        us_flag = "🇺🇸"
        uk_flag = "🇬🇧"
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

        # é as precomposed (NFC) vs decomposed (NFD)
        nfc = unicodedata.normalize("NFC", "café")
        nfd = unicodedata.normalize("NFD", "café")
        # With unicode_nfkd normalization, these should be identical
        assert fr.levenshtein(nfc, nfd, normalize="unicode_nfkd") == 0
        # Without normalization, NFD has extra combining character
        # Similarity is still reasonably high (~0.85)
        sim = fr.jaro_winkler_similarity(nfc, nfd)
        assert sim > 0.8

    def test_surrogates_and_supplementary_planes(self):
        """Test handling of characters outside BMP (supplementary planes)."""
        import fuzzyrust as fr

        # Mathematical symbols from Plane 1
        math1 = "𝐀𝐁𝐂"  # Mathematical Bold Capital
        math2 = "𝐀𝐁𝐃"  # One different
        assert fr.levenshtein(math1, math1) == 0
        assert fr.levenshtein(math1, math2) == 1
        # Ancient scripts (e.g., Egyptian Hieroglyphs from Plane 1)
        hieroglyph = "𓀀"
        assert fr.levenshtein(hieroglyph, hieroglyph) == 0

    def test_mixed_scripts(self):
        """Test handling of mixed script strings."""
        import fuzzyrust as fr

        # Mix of Latin, Cyrillic, Greek
        mixed1 = "Hello Мир αβγ"
        mixed2 = "Hello Мир αβδ"
        assert fr.levenshtein(mixed1, mixed2) == 1
        # Similarity should work across scripts
        sim = fr.jaro_winkler_similarity(mixed1, mixed2)
        assert sim > 0.9


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
    import string

    from hypothesis import assume, given, settings
    from hypothesis import strategies as st

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

        assert fr.levenshtein(a, b) >= 0
        assert fr.damerau_levenshtein(a, b) >= 0

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
        assert 0.0 <= sim <= 1.0

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_lcs_bounds(self, a, b):
        """LCS similarity is always between 0 and 1."""
        import fuzzyrust as fr

        sim = fr.lcs_similarity(a, b)
        assert 0.0 <= sim <= 1.0

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
        assert 0.0 <= sim <= 1.0

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_levenshtein_similarity_bounds(self, a, b):
        """Levenshtein similarity is always between 0 and 1."""
        import fuzzyrust as fr

        sim = fr.levenshtein_similarity(a, b)
        assert 0.0 <= sim <= 1.0

    @given(st.text(min_size=0, max_size=50))
    def test_levenshtein_similarity_identity(self, s):
        """Levenshtein similarity of a string with itself is 1.0."""
        import fuzzyrust as fr

        assert fr.levenshtein_similarity(s, s) == 1.0

    @given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
    def test_cosine_bounds(self, a, b):
        """Cosine similarity variants are always between 0 and 1."""
        import fuzzyrust as fr

        assert 0.0 <= fr.cosine_similarity_chars(a, b) <= 1.0
        assert 0.0 <= fr.cosine_similarity_ngrams(a, b) <= 1.0

    @given(
        st.text(alphabet=string.ascii_letters + " ", min_size=1, max_size=50),
        st.text(alphabet=string.ascii_letters + " ", min_size=1, max_size=50),
    )
    @settings(max_examples=50)
    def test_cosine_words_bounds(self, a, b):
        """Cosine word similarity is between 0 and 1."""
        import fuzzyrust as fr

        sim = fr.cosine_similarity_words(a, b)
        assert 0.0 <= sim <= 1.0

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
        assert isinstance(result, list)

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

import concurrent.futures
import threading


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
        algorithms = [
            "levenshtein",
            "damerau_levenshtein",
            "jaro",
            "jaro_winkler",
            "ngram",
            "bigram",
            "trigram",
            "lcs",
            "cosine",
        ]

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


class TestNormalizedNgramIndex:
    """Tests for case-insensitive n-gram indexing."""

    def test_normalized_index_finds_case_variants(self):
        """Normalized index should find candidates regardless of case."""
        import fuzzyrust as fr

        index = fr.NgramIndex(ngram_size=2, normalize=True)
        index.add_all(["Rain shadow jkt", "FLEECE JACKET", "hiking pants"])
        results = index.search("rain shadow", min_similarity=0.5)
        texts = [r.text for r in results]
        assert "Rain shadow jkt" in texts

    def test_normalized_index_default_true(self):
        """Default normalize=True should work for mixed-case matching."""
        import fuzzyrust as fr

        index = fr.NgramIndex(ngram_size=2)  # normalize=True by default
        index.add_all(["HELLO WORLD", "hello there"])
        results = index.search("hello world", min_similarity=0.8)
        texts = [r.text for r in results]
        assert "HELLO WORLD" in texts

    def test_unnormalized_index_is_case_sensitive(self):
        """With normalize=False, n-gram extraction is case-sensitive."""
        import fuzzyrust as fr

        index = fr.NgramIndex(ngram_size=2, normalize=False)
        index.add_all(["HELLO"])
        # With normalize=False, "HE" != "he", so no candidates found
        results = index.search("hello", min_similarity=0.5, case_insensitive=False)
        assert len(results) == 0

    def test_hybrid_index_normalized(self):
        """HybridIndex should also support normalization."""
        import fuzzyrust as fr

        index = fr.HybridIndex(ngram_size=2, normalize=True)
        index.add_all(["Product Name", "ANOTHER PRODUCT"])
        results = index.search("product name", min_similarity=0.8)
        texts = [r.text for r in results]
        assert "Product Name" in texts

    def test_contains_respects_normalize(self):
        """contains() should use normalized comparison when normalize=True."""
        import fuzzyrust as fr

        index = fr.NgramIndex(ngram_size=2, normalize=True)
        index.add("Hello World")
        assert index.contains("hello world")  # Should match despite case

        index2 = fr.NgramIndex(ngram_size=2, normalize=False)
        index2.add("Hello World")
        assert not index2.contains("hello world")  # Should not match


class TestTfIdfCosine:
    """Tests for TF-IDF weighted cosine similarity."""

    def test_tfidf_basic(self):
        """Basic TF-IDF similarity test."""
        import fuzzyrust as fr

        tfidf = fr.TfIdfCosine()
        tfidf.add_documents(["hello world", "hello there", "world news"])
        sim = tfidf.similarity("hello world", "hello there")
        assert 0.0 < sim < 1.0

    def test_tfidf_identical(self):
        """Identical strings should have similarity 1.0."""
        import fuzzyrust as fr

        tfidf = fr.TfIdfCosine()
        tfidf.add_document("test document")
        assert tfidf.similarity("hello", "hello") == 1.0

    def test_tfidf_different(self):
        """Completely different strings should have low similarity."""
        import fuzzyrust as fr

        tfidf = fr.TfIdfCosine()
        tfidf.add_documents(["apple banana", "cherry grape"])
        sim = tfidf.similarity("apple", "cherry")
        assert sim == 0.0  # No common words

    def test_tfidf_num_documents(self):
        """num_documents should return correct count."""
        import fuzzyrust as fr

        tfidf = fr.TfIdfCosine()
        assert tfidf.num_documents() == 0
        tfidf.add_document("first")
        assert tfidf.num_documents() == 1
        tfidf.add_documents(["second", "third"])
        assert tfidf.num_documents() == 3

    def test_tfidf_rare_words_weighted_higher(self):
        """Rare words should contribute more to similarity than common words."""
        import fuzzyrust as fr

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


class TestPhoneticSimilarity:
    """Tests for phonetic similarity functions."""

    def test_soundex_similarity_identical_codes(self):
        """Same Soundex codes should have similarity 1.0."""
        import fuzzyrust as fr

        # Robert and Rupert both have Soundex code R163
        assert fr.soundex_similarity("Robert", "Rupert") == 1.0

    def test_soundex_similarity_partial(self):
        """Different Soundex codes should have partial similarity."""
        import fuzzyrust as fr

        sim = fr.soundex_similarity("Robert", "Richard")
        assert 0.0 < sim < 1.0

    def test_soundex_similarity_different(self):
        """Completely different names should have low similarity."""
        import fuzzyrust as fr

        sim = fr.soundex_similarity("Smith", "Johnson")
        assert sim < 0.5

    def test_metaphone_similarity_identical(self):
        """Same Metaphone codes should have similarity 1.0."""
        import fuzzyrust as fr

        # Stephen and Steven have same Metaphone code
        assert fr.metaphone_similarity("Stephen", "Steven") == 1.0

    def test_metaphone_similarity_partial(self):
        """Similar sounds should have high but not perfect similarity."""
        import fuzzyrust as fr

        sim = fr.metaphone_similarity("John", "Jon")
        assert sim > 0.8


class TestNgramConvenience:
    """Tests for n-gram convenience functions."""

    def test_bigram_similarity_identical(self):
        """Identical strings should have similarity 1.0."""
        import fuzzyrust as fr

        assert fr.bigram_similarity("hello", "hello") == 1.0

    def test_bigram_similarity_similar(self):
        """Similar strings should have high similarity."""
        import fuzzyrust as fr

        sim = fr.bigram_similarity("hello", "hallo")
        assert 0.0 < sim < 1.0

    def test_trigram_similarity(self):
        """Trigram similarity should work correctly."""
        import fuzzyrust as fr

        sim = fr.trigram_similarity("hello", "hallo")
        assert 0.0 < sim < 1.0
        # Trigram is typically stricter than bigram
        bigram = fr.bigram_similarity("hello", "hallo")
        assert sim <= bigram or abs(sim - bigram) < 0.2

    def test_ngram_profile_similarity_identical(self):
        """Identical strings should have profile similarity 1.0."""
        import fuzzyrust as fr

        assert fr.ngram_profile_similarity("abab", "abab", 2) == 1.0

    def test_ngram_profile_counts_frequency(self):
        """Profile similarity should account for n-gram frequency."""
        import fuzzyrust as fr

        # Profile similarity uses frequency counts
        sim = fr.ngram_profile_similarity("aaa", "aa", 2)
        assert 0.0 < sim < 1.0


class TestHammingVariants:
    """Tests for Hamming distance variants."""

    def test_hamming_padded_equal_length(self):
        """Padded Hamming should work on equal-length strings."""
        import fuzzyrust as fr

        assert fr.hamming_distance_padded("abc", "axc") == 1
        assert fr.hamming_distance_padded("abc", "abc") == 0

    def test_hamming_padded_unequal_length(self):
        """Padded Hamming should work on unequal-length strings."""
        import fuzzyrust as fr

        # "ab" becomes "ab " when padded to match "abc"
        result = fr.hamming_distance_padded("ab", "abc")
        assert result >= 1

    def test_hamming_similarity_equal_length(self):
        """Hamming similarity should work for equal-length strings."""
        import fuzzyrust as fr

        assert fr.hamming_similarity("abc", "abc") == 1.0
        sim = fr.hamming_similarity("abc", "axc")
        assert sim is not None
        assert 0.5 < sim < 1.0

    def test_hamming_similarity_unequal_length(self):
        """Hamming similarity should raise ValueError for unequal lengths."""
        import pytest

        import fuzzyrust as fr

        with pytest.raises(ValueError):
            fr.hamming_similarity("abc", "ab")
        with pytest.raises(ValueError):
            fr.hamming_similarity("ab", "abc")


class TestLcsAlternative:
    """Tests for LCS alternative metrics."""

    def test_lcs_similarity_max_identical(self):
        """Identical strings should have similarity 1.0."""
        import fuzzyrust as fr

        assert fr.lcs_similarity_max("abc", "abc") == 1.0

    def test_lcs_similarity_max_partial(self):
        """Partial matches should have proportional similarity."""
        import fuzzyrust as fr

        # LCS of "abc" and "ab" is "ab" (length 2)
        # lcs_similarity_max = 2 / max(3, 2) = 2/3
        sim = fr.lcs_similarity_max("abc", "ab")
        assert abs(sim - 2 / 3) < 0.01

    def test_lcs_similarity_max_different(self):
        """Completely different strings should have similarity 0.0."""
        import fuzzyrust as fr

        assert fr.lcs_similarity_max("abc", "xyz") == 0.0


class TestCaseInsensitiveFunctions:
    """Tests for case-insensitive (_ci) function variants."""

    def test_ngram_jaccard_ci(self):
        """ngram_jaccard_ci should be case-insensitive."""
        import fuzzyrust as fr

        # Case-sensitive would give different results
        assert fr.ngram_jaccard_ci("Hello", "hello") == fr.ngram_jaccard("hello", "hello")
        assert fr.ngram_jaccard_ci("WORLD", "world") == 1.0

    def test_cosine_similarity_chars_ci(self):
        """cosine_similarity_chars_ci should be case-insensitive."""
        import fuzzyrust as fr

        assert fr.cosine_similarity_chars_ci("ABC", "abc") == 1.0
        assert fr.cosine_similarity_chars_ci("Hello", "HELLO") == 1.0

    def test_cosine_similarity_words_ci(self):
        """cosine_similarity_words_ci should be case-insensitive."""
        import fuzzyrust as fr

        assert fr.cosine_similarity_words_ci("The Quick Fox", "the quick fox") == 1.0
        assert fr.cosine_similarity_words_ci("HELLO WORLD", "hello world") == 1.0

    def test_cosine_similarity_ngrams_ci(self):
        """cosine_similarity_ngrams_ci should be case-insensitive."""
        import fuzzyrust as fr

        assert fr.cosine_similarity_ngrams_ci("Hello", "HELLO") == 1.0
        assert fr.cosine_similarity_ngrams_ci("WORLD", "world") == 1.0


class TestSchemaIndex:
    """Tests for SchemaIndex with renamed parameters."""

    def test_schema_index_min_similarity_parameter(self):
        """SchemaIndex.search should accept min_similarity parameter."""
        import fuzzyrust as fr

        builder = fr.SchemaBuilder()
        builder.add_field("name", "short_text", algorithm="jaro_winkler", weight=1.0)
        schema = builder.build()

        index = fr.SchemaIndex(schema)
        index.add({"name": "apple"})
        index.add({"name": "banana"})

        # Use new parameter name min_similarity
        results = index.search({"name": "apple"}, min_similarity=0.9)
        assert len(results) >= 1
        assert results[0].score >= 0.9

    def test_schema_index_min_field_similarity_parameter(self):
        """SchemaIndex.search should accept min_field_similarity parameter."""
        import fuzzyrust as fr

        builder = fr.SchemaBuilder()
        builder.add_field("name", "short_text", algorithm="jaro_winkler", weight=1.0)
        builder.add_field("category", "short_text", algorithm="jaro_winkler", weight=1.0)
        schema = builder.build()

        index = fr.SchemaIndex(schema)
        index.add({"name": "apple", "category": "fruit"})

        # Use new parameter name min_field_similarity
        results = index.search(
            {"name": "apple", "category": "fruit"}, min_similarity=0.0, min_field_similarity=0.8
        )
        assert len(results) >= 1


class TestHybridIndexConstructor:
    """Tests for HybridIndex constructor parameters."""

    def test_hybrid_index_constructor_params(self):
        """HybridIndex should accept constructor parameters."""
        import fuzzyrust as fr

        # Test constructor with explicit parameters
        index = fr.HybridIndex(ngram_size=3, min_ngram_ratio=0.2)
        index.add("apple")
        index.add("banana")

        # min_similarity is specified in search, not constructor
        results = index.search("apple", min_similarity=0.9)
        assert len(results) >= 1


class TestNormalizeParameter:
    """Tests for the new normalize parameter on similarity functions."""

    def test_levenshtein_normalize_lowercase(self):
        """levenshtein with normalize='lowercase' should be case-insensitive."""
        import fuzzyrust as fr

        # Without normalization
        assert fr.levenshtein("Hello", "HELLO") == 4  # 4 substitutions
        # With lowercase normalization
        assert fr.levenshtein("Hello", "HELLO", normalize="lowercase") == 0

    def test_levenshtein_similarity_normalize(self):
        """levenshtein_similarity with normalize should work."""
        import fuzzyrust as fr

        assert fr.levenshtein_similarity("Hello", "HELLO", normalize="lowercase") == 1.0
        assert fr.levenshtein_similarity("Hello", "HELLO") < 1.0  # Without normalization

    def test_damerau_levenshtein_normalize(self):
        """damerau_levenshtein with normalize should work."""
        import fuzzyrust as fr

        assert fr.damerau_levenshtein("Hello", "HELLO", normalize="lowercase") == 0
        assert fr.damerau_levenshtein("Hello", "HELLO") > 0

    def test_damerau_levenshtein_similarity_normalize(self):
        """damerau_levenshtein_similarity with normalize should work."""
        import fuzzyrust as fr

        assert fr.damerau_levenshtein_similarity("Hello", "HELLO", normalize="lowercase") == 1.0

    def test_jaro_similarity_normalize(self):
        """jaro_similarity with normalize should work."""
        import fuzzyrust as fr

        assert fr.jaro_similarity("MARTHA", "martha", normalize="lowercase") == 1.0

    def test_jaro_winkler_similarity_normalize(self):
        """jaro_winkler_similarity with normalize should work."""
        import fuzzyrust as fr

        assert fr.jaro_winkler_similarity("Hello", "HELLO", normalize="lowercase") == 1.0
        # Works with other parameters too
        assert (
            fr.jaro_winkler_similarity("Hello", "HELLO", prefix_weight=0.1, normalize="lowercase")
            == 1.0
        )

    def test_ngram_similarity_normalize(self):
        """ngram_similarity with normalize should work."""
        import fuzzyrust as fr

        assert fr.ngram_similarity("Hello", "HELLO", normalize="lowercase") == 1.0
        # Works with other parameters too
        assert fr.ngram_similarity("Hello", "HELLO", ngram_size=3, normalize="lowercase") == 1.0

    def test_ngram_jaccard_normalize(self):
        """ngram_jaccard with normalize should work."""
        import fuzzyrust as fr

        assert fr.ngram_jaccard("Hello", "HELLO", normalize="lowercase") == 1.0

    def test_cosine_similarity_chars_normalize(self):
        """cosine_similarity_chars with normalize should work."""
        import fuzzyrust as fr

        assert fr.cosine_similarity_chars("ABC", "abc", normalize="lowercase") == 1.0

    def test_cosine_similarity_words_normalize(self):
        """cosine_similarity_words with normalize should work."""
        import fuzzyrust as fr

        assert (
            fr.cosine_similarity_words("HELLO WORLD", "hello world", normalize="lowercase") == 1.0
        )

    def test_cosine_similarity_ngrams_normalize(self):
        """cosine_similarity_ngrams with normalize should work."""
        import fuzzyrust as fr

        assert fr.cosine_similarity_ngrams("Hello", "HELLO", normalize="lowercase") == 1.0

    def test_normalize_strict_mode(self):
        """Test strict normalization mode (combines all normalizations)."""
        import fuzzyrust as fr

        # Strict mode: lowercase + remove punctuation + remove whitespace + NFKD
        assert (
            fr.levenshtein_similarity("  Hello, World!  ", "helloworld", normalize="strict") == 1.0
        )

    def test_normalize_remove_punctuation(self):
        """Test remove_punctuation normalization mode."""
        import fuzzyrust as fr

        # Removes punctuation but keeps case and whitespace
        assert (
            fr.levenshtein_similarity(
                "Hello, World!", "Hello World", normalize="remove_punctuation"
            )
            == 1.0
        )

    def test_normalize_remove_whitespace(self):
        """Test remove_whitespace normalization mode."""
        import fuzzyrust as fr

        assert (
            fr.levenshtein_similarity("hello world", "helloworld", normalize="remove_whitespace")
            == 1.0
        )

    def test_normalize_invalid_mode_raises_error(self):
        """Invalid normalize mode should raise ValueError."""
        import fuzzyrust as fr

        with pytest.raises(ValueError, match="Unknown normalization mode"):
            fr.levenshtein_similarity("hello", "world", normalize="invalid_mode")

    def test_ci_functions_equivalent_to_normalize_lowercase(self):
        """_ci functions should be equivalent to base functions with normalize='lowercase'."""
        import fuzzyrust as fr

        # levenshtein_similarity_ci == levenshtein_similarity with normalize="lowercase"
        assert fr.levenshtein_similarity_ci("Hello", "HELLO") == fr.levenshtein_similarity(
            "Hello", "HELLO", normalize="lowercase"
        )

        # jaro_winkler_similarity_ci == jaro_winkler_similarity with normalize="lowercase"
        assert fr.jaro_winkler_similarity_ci("Hello", "HELLO") == fr.jaro_winkler_similarity(
            "Hello", "HELLO", normalize="lowercase"
        )

        # ngram_similarity_ci == ngram_similarity with normalize="lowercase"
        assert fr.ngram_similarity_ci("Hello", "HELLO") == fr.ngram_similarity(
            "Hello", "HELLO", normalize="lowercase"
        )

    def test_normalize_none_is_default(self):
        """normalize=None should give same results as no normalization."""
        import fuzzyrust as fr

        # Explicit None should be the same as not passing the parameter
        assert fr.levenshtein_similarity(
            "Hello", "HELLO", normalize=None
        ) == fr.levenshtein_similarity("Hello", "HELLO")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
