"""
Edge case tests for FuzzyRust.

Tests cover:
- Null/None handling
- Empty strings
- Very long strings (100K+ chars)
- Unicode edge cases (ZWJ, combining chars, surrogates)
- Adversarial inputs (repeated patterns)
- Special characters
"""

import pytest

import fuzzyrust as fr


class TestNullNoneHandling:
    """Tests for null/None input handling."""

    def test_levenshtein_none_raises(self):
        """None input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.levenshtein(None, "hello")
        with pytest.raises(TypeError):
            fr.levenshtein("hello", None)
        with pytest.raises(TypeError):
            fr.levenshtein(None, None)

    def test_jaro_winkler_none_raises(self):
        """None input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.jaro_winkler_similarity(None, "hello")
        with pytest.raises(TypeError):
            fr.jaro_winkler_similarity("hello", None)

    def test_ngram_similarity_none_raises(self):
        """None input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.ngram_similarity(None, "hello")
        with pytest.raises(TypeError):
            fr.ngram_similarity("hello", None)

    def test_soundex_none_raises(self):
        """None input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.soundex(None)

    def test_metaphone_none_raises(self):
        """None input should raise TypeError."""
        with pytest.raises(TypeError):
            fr.metaphone(None)

    def test_find_best_matches_none_in_list(self):
        """None in choices list should raise TypeError."""
        with pytest.raises(TypeError):
            fr.find_best_matches(["apple", None, "banana"], "apple")


class TestEmptyStrings:
    """Tests for empty string handling."""

    def test_levenshtein_empty_empty(self):
        """Empty strings should have distance 0."""
        assert fr.levenshtein("", "") == 0

    def test_levenshtein_empty_nonempty(self):
        """Empty vs non-empty should equal length of non-empty."""
        assert fr.levenshtein("", "hello") == 5
        assert fr.levenshtein("hello", "") == 5

    def test_jaro_winkler_empty_empty(self):
        """Empty strings should have similarity 1.0."""
        assert fr.jaro_winkler_similarity("", "") == 1.0

    def test_jaro_winkler_empty_nonempty(self):
        """Empty vs non-empty should have similarity 0.0."""
        assert fr.jaro_winkler_similarity("", "hello") == 0.0
        assert fr.jaro_winkler_similarity("hello", "") == 0.0

    def test_ngram_similarity_empty(self):
        """Empty strings should have defined behavior."""
        # Empty vs empty: no ngrams to compare, but identical strings = similarity 1.0
        sim = fr.ngram_similarity("", "")
        assert sim == 1.0  # Empty strings are identical

    def test_soundex_empty(self):
        """Empty string should return empty soundex."""
        result = fr.soundex("")
        assert result == ""  # Empty input returns empty soundex

    def test_metaphone_empty(self):
        """Empty string should return empty metaphone."""
        result = fr.metaphone("")
        assert result == ""  # Empty input returns empty metaphone

    def test_find_best_matches_empty_query(self):
        """Empty query should still work."""
        matches = fr.find_best_matches(["apple", "banana"], "")
        assert isinstance(matches, list), "find_best_matches should return a list"
        # Empty query has 0 similarity to any non-empty string
        # Results are returned but with score 0.0 (default min_similarity is 0.0)
        assert len(matches) == 2, f"Expected 2 results for empty query, got {len(matches)} matches"
        for match in matches:
            assert match.score == 0.0, (
                f"Expected score 0.0 for empty query match, got {match.score}"
            )
            assert match.text in [
                "apple",
                "banana",
            ], f"Expected 'apple' or 'banana', got {match.text}"

    def test_find_best_matches_empty_choices(self):
        """Empty choices list should return empty results."""
        matches = fr.find_best_matches([], "apple")
        assert matches == []

    def test_bktree_empty_strings(self):
        """BK-tree should handle empty strings."""
        tree = fr.BkTree()
        tree.add("")
        tree.add("hello")
        results = tree.search("", max_distance=0)
        assert len(results) >= 1


class TestVeryLongStrings:
    """Tests for very long string handling."""

    @pytest.mark.slow
    def test_levenshtein_long_strings(self):
        """Levenshtein should handle long strings."""
        # 10K characters
        s1 = "a" * 10000
        s2 = "a" * 10000
        assert fr.levenshtein(s1, s2) == 0

        s3 = "a" * 10000
        s4 = "b" * 10000
        dist = fr.levenshtein(s3, s4)
        assert dist == 10000

    @pytest.mark.slow
    def test_jaro_winkler_long_strings(self):
        """Jaro-Winkler should handle long strings."""
        s1 = "a" * 10000
        s2 = "a" * 10000
        sim = fr.jaro_winkler_similarity(s1, s2)
        assert sim == 1.0

    @pytest.mark.slow
    def test_ngram_similarity_long_strings(self):
        """N-gram similarity should handle long strings."""
        s1 = "hello " * 1000
        s2 = "hello " * 1000
        sim = fr.ngram_similarity(s1, s2)
        assert sim == 1.0

    @pytest.mark.slow
    def test_cosine_similarity_long_strings(self):
        """Cosine similarity should handle long strings."""
        s1 = "word " * 5000
        s2 = "word " * 5000
        sim = fr.cosine_similarity_words(s1, s2)
        assert sim == 1.0

    def test_levenshtein_asymmetric_lengths(self):
        """Handle very different string lengths."""
        short = "hi"
        long = "x" * 1000
        dist = fr.levenshtein(short, long)
        assert dist == 1000  # Replace 2 + insert 998


class TestUnicodeEdgeCases:
    """Tests for Unicode edge cases."""

    def test_combining_characters(self):
        """Combining characters should be handled correctly."""
        # e + combining acute = é
        composed = "é"  # Single codepoint U+00E9
        decomposed = "e\u0301"  # e + combining acute (2 codepoints)

        # These are visually identical but different codepoints
        # Composed "é" (1 codepoint) vs decomposed "e" + combining accent (2 codepoints)
        # Distance is 2: the codepoints are completely different (é ≠ e, é ≠ \u0301)
        dist = fr.levenshtein(composed, decomposed)
        assert dist == 2, (
            f"Expected distance 2 between composed (1 codepoint) and decomposed (2 codepoints), got {dist}"
        )

    def test_emoji(self):
        """Emoji should be handled correctly."""
        s1 = "Hello 👋"
        s2 = "Hello 👋"
        assert fr.levenshtein(s1, s2) == 0
        assert fr.jaro_winkler_similarity(s1, s2) == 1.0

    def test_emoji_zwj_sequences(self):
        """Zero-width joiner sequences should work."""
        # Family emoji (ZWJ sequence)
        # 👨 + ZWJ + 👩 + ZWJ + 👧 + ZWJ + 👦 = 7 codepoints
        family = "👨‍👩‍👧‍👦"  # Man + ZWJ + Woman + ZWJ + Girl + ZWJ + Boy
        single = "👨"  # 1 codepoint

        # Family has 7 codepoints, single has 1, so distance should be 6
        dist = fr.levenshtein(family, single)
        assert dist == 6, (
            f"Expected distance 6 between family emoji (7 codepoints) and single emoji (1 codepoint), got {dist}"
        )

        sim = fr.jaro_winkler_similarity(family, family)
        assert sim == 1.0

    def test_mixed_scripts(self):
        """Mixed Unicode scripts should work."""
        s1 = "Hello世界مرحبا"  # Latin + CJK + Arabic
        s2 = "Hello世界مرحبا"
        assert fr.levenshtein(s1, s2) == 0
        assert fr.jaro_winkler_similarity(s1, s2) == 1.0

    def test_rtl_characters(self):
        """Right-to-left characters should work."""
        arabic = "مرحبا"  # 5 characters
        hebrew = "שלום"  # 4 characters

        # All characters differ, so distance is max(5, 4) = 5
        # (delete all 5 Arabic chars, insert all 4 Hebrew chars = 9 ops,
        # but optimal is replace 4 + delete 1 = 5 ops)
        dist = fr.levenshtein(arabic, hebrew)
        assert dist == 5, (
            f"Expected distance 5 between Arabic (5 chars) and Hebrew (4 chars), got {dist}"
        )

    def test_surrogate_pairs(self):
        """Supplementary plane characters (emoji, etc) should work."""
        # 𝕳𝖊𝖑𝖑𝖔 (mathematical fraktur) - 5 characters
        s1 = "𝕳𝖊𝖑𝖑𝖔"
        s2 = "Hello"  # 5 characters

        # All 5 characters differ (fraktur vs ASCII), so distance should be 5
        dist = fr.levenshtein(s1, s2)
        assert dist == 5, (
            f"Expected distance 5 between fraktur and ASCII (all 5 chars differ), got {dist}"
        )

    def test_null_character_in_string(self):
        """Null characters within strings should work."""
        s1 = "hello\x00world"
        s2 = "hello\x00world"
        assert fr.levenshtein(s1, s2) == 0

    def test_newlines_and_tabs(self):
        """Newlines and tabs should be handled."""
        s1 = "hello\nworld\ttab"
        s2 = "hello\nworld\ttab"
        assert fr.levenshtein(s1, s2) == 0
        assert fr.jaro_winkler_similarity(s1, s2) == 1.0


class TestAdversarialInputs:
    """Tests for adversarial/pathological inputs."""

    def test_repeated_single_char(self):
        """Repeated single characters."""
        s1 = "a" * 100
        s2 = "b" * 100
        dist = fr.levenshtein(s1, s2)
        assert dist == 100

    def test_alternating_pattern(self):
        """Alternating character patterns."""
        s1 = "ab" * 50  # "abab...ab" (100 chars)
        s2 = "ba" * 50  # "baba...ba" (100 chars)
        dist = fr.levenshtein(s1, s2)
        # Pattern offset by one: "abab..." vs "baba..."
        # Optimal: delete first 'a', add 'a' at end = 2 operations
        assert dist == 2, f"Expected distance 2 for offset alternating patterns, got {dist}"

    def test_prefix_suffix_overlap(self):
        """Strings with common prefix and suffix."""
        s1 = "prefix_middle_suffix"  # 20 chars
        s2 = "prefix_other_suffix"  # 19 chars
        sim = fr.jaro_winkler_similarity(s1, s2)
        # Common prefix "prefix_" (7 chars) and suffix "_suffix" (7 chars)
        # Jaro-Winkler should give high score for shared prefix/suffix
        assert 0.75 <= sim <= 0.90, (
            f"Expected Jaro-Winkler 0.75-0.90 for shared prefix/suffix strings, got {sim}"
        )

    def test_single_character_strings(self):
        """Single character strings."""
        assert fr.levenshtein("a", "a") == 0
        assert fr.levenshtein("a", "b") == 1
        assert fr.jaro_winkler_similarity("a", "a") == 1.0
        assert fr.jaro_winkler_similarity("a", "b") == 0.0

    def test_all_whitespace(self):
        """All whitespace strings."""
        s1 = "   "
        s2 = "   "
        assert fr.levenshtein(s1, s2) == 0

        s3 = " " * 10
        s4 = " " * 5
        assert fr.levenshtein(s3, s4) == 5


class TestSpecialCharacters:
    """Tests for special characters and punctuation."""

    def test_punctuation_only(self):
        """Strings with only punctuation."""
        s1 = ".,;:!?"
        s2 = ".,;:!?"
        assert fr.levenshtein(s1, s2) == 0
        assert fr.jaro_winkler_similarity(s1, s2) == 1.0

    def test_mixed_punctuation_and_text(self):
        """Mixed punctuation and text."""
        s1 = "Hello, World!"
        s2 = "Hello World"
        dist = fr.levenshtein(s1, s2)
        assert dist == 2  # Remove comma and exclamation

    def test_quotes_and_apostrophes(self):
        """Various quote characters."""
        # Use explicit Unicode codepoints to ensure different apostrophes
        s1 = "it\u0027s"  # ASCII apostrophe (U+0027)
        s2 = "it\u2019s"  # Curly apostrophe (U+2019 RIGHT SINGLE QUOTATION MARK)
        # The apostrophe characters differ, so distance should be 1
        dist = fr.levenshtein(s1, s2)
        assert dist == 1, f"Expected distance 1 between straight and curly apostrophe, got {dist}"

    def test_mathematical_symbols(self):
        """Mathematical symbols."""
        s1 = "x² + y² = z²"
        s2 = "x² + y² = z²"
        assert fr.levenshtein(s1, s2) == 0


class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_max_similarity(self):
        """Identical strings should have similarity exactly 1.0."""
        test_cases = [
            ("hello", "hello"),
            ("", ""),
            ("a", "a"),
            ("test string", "test string"),
        ]
        for s1, s2 in test_cases:
            jw_score = fr.jaro_winkler_similarity(s1, s2)
            lev_score = fr.levenshtein_similarity(s1, s2)
            ng_score = fr.ngram_similarity(s1, s2)
            assert jw_score == 1.0, (
                f"Jaro-Winkler for identical '{s1}' should be 1.0, got {jw_score}"
            )
            assert lev_score == 1.0, (
                f"Levenshtein similarity for identical '{s1}' should be 1.0, got {lev_score}"
            )
            assert ng_score == 1.0, (
                f"N-gram similarity for identical '{s1}' should be 1.0, got {ng_score}"
            )

    def test_min_similarity(self):
        """Different strings should have specific expected similarity values."""
        # Test case: completely different 3-char strings
        jw_abc_xyz = fr.jaro_winkler_similarity("abc", "xyz")
        lev_abc_xyz = fr.levenshtein_similarity("abc", "xyz")
        ng_abc_xyz = fr.ngram_similarity("abc", "xyz")
        assert jw_abc_xyz == 0.0, (
            f"Jaro-Winkler for 'abc' vs 'xyz' (no common chars) should be 0.0, got {jw_abc_xyz}"
        )
        assert lev_abc_xyz == 0.0, (
            f"Levenshtein similarity for 'abc' vs 'xyz' (all chars differ) should be 0.0, got {lev_abc_xyz}"
        )
        assert ng_abc_xyz == 0.0, (
            f"N-gram similarity for 'abc' vs 'xyz' (no common n-grams) should be 0.0, got {ng_abc_xyz}"
        )

        # Test case: "hello" vs "world" - some shared letters (l, o)
        jw_hello_world = fr.jaro_winkler_similarity("hello", "world")
        lev_hello_world = fr.levenshtein_similarity("hello", "world")
        ng_hello_world = fr.ngram_similarity("hello", "world")
        # Jaro-Winkler: shares 'l' and 'o', but low similarity due to positions
        assert 0.0 <= jw_hello_world <= 0.5, (
            f"Jaro-Winkler for 'hello' vs 'world' should be in [0.0, 0.5], got {jw_hello_world}"
        )
        # Levenshtein: 4 edits out of 5 chars = 0.2 similarity (use approx for float comparison)
        assert lev_hello_world == pytest.approx(0.2, abs=0.001), (
            f"Levenshtein similarity for 'hello' vs 'world' (4 edits) should be ~0.2, got {lev_hello_world}"
        )
        # N-gram: shares 'ld' bigram only (if any), very low
        assert 0.0 <= ng_hello_world <= 0.25, (
            f"N-gram similarity for 'hello' vs 'world' should be in [0.0, 0.25], got {ng_hello_world}"
        )

        # Test case: empty vs non-empty
        jw_empty = fr.jaro_winkler_similarity("", "test")
        lev_empty = fr.levenshtein_similarity("", "test")
        ng_empty = fr.ngram_similarity("", "test")
        assert jw_empty == 0.0, f"Jaro-Winkler for '' vs 'test' should be 0.0, got {jw_empty}"
        assert lev_empty == 0.0, (
            f"Levenshtein similarity for '' vs 'test' should be 0.0, got {lev_empty}"
        )
        assert ng_empty == 0.0, f"N-gram similarity for '' vs 'test' should be 0.0, got {ng_empty}"

    def test_distance_non_negative(self):
        """Distance should match expected values for specific pairs."""
        # Test case: "abc" vs "xyz" - all 3 characters differ
        lev_abc_xyz = fr.levenshtein("abc", "xyz")
        dam_abc_xyz = fr.damerau_levenshtein("abc", "xyz")
        assert lev_abc_xyz == 3, (
            f"Levenshtein distance for 'abc' vs 'xyz' should be 3 (replace all), got {lev_abc_xyz}"
        )
        assert dam_abc_xyz == 3, (
            f"Damerau-Levenshtein distance for 'abc' vs 'xyz' should be 3, got {dam_abc_xyz}"
        )

        # Test case: "hello" vs "world" - 4 edits needed
        lev_hello_world = fr.levenshtein("hello", "world")
        dam_hello_world = fr.damerau_levenshtein("hello", "world")
        assert lev_hello_world == 4, (
            f"Levenshtein distance for 'hello' vs 'world' should be 4, got {lev_hello_world}"
        )
        assert dam_hello_world == 4, (
            f"Damerau-Levenshtein distance for 'hello' vs 'world' should be 4, got {dam_hello_world}"
        )

        # Test case: "" vs "test" - insert 4 characters
        lev_empty = fr.levenshtein("", "test")
        dam_empty = fr.damerau_levenshtein("", "test")
        assert lev_empty == 4, (
            f"Levenshtein distance for '' vs 'test' should be 4 (insert all), got {lev_empty}"
        )
        assert dam_empty == 4, (
            f"Damerau-Levenshtein distance for '' vs 'test' should be 4, got {dam_empty}"
        )

        # Test case: "same" vs "same" - identical strings
        lev_same = fr.levenshtein("same", "same")
        dam_same = fr.damerau_levenshtein("same", "same")
        assert lev_same == 0, (
            f"Levenshtein distance for identical strings should be 0, got {lev_same}"
        )
        assert dam_same == 0, (
            f"Damerau-Levenshtein distance for identical strings should be 0, got {dam_same}"
        )

    def test_symmetric_distance(self):
        """Distance should be symmetric."""
        test_cases = [
            ("hello", "hallo"),
            ("kitten", "sitting"),
            ("", "test"),
        ]
        for s1, s2 in test_cases:
            assert fr.levenshtein(s1, s2) == fr.levenshtein(s2, s1)
            assert fr.damerau_levenshtein(s1, s2) == fr.damerau_levenshtein(s2, s1)

    def test_symmetric_similarity(self):
        """Similarity should be symmetric."""
        test_cases = [
            ("hello", "hallo"),
            ("kitten", "sitting"),
        ]
        for s1, s2 in test_cases:
            assert fr.jaro_winkler_similarity(s1, s2) == pytest.approx(
                fr.jaro_winkler_similarity(s2, s1), abs=0.001
            )
            assert fr.levenshtein_similarity(s1, s2) == pytest.approx(
                fr.levenshtein_similarity(s2, s1), abs=0.001
            )


class TestIndexEdgeCases:
    """Edge cases for index structures."""

    def test_bktree_single_item(self):
        """BK-tree with single item."""
        tree = fr.BkTree()
        tree.add("hello")
        results = tree.search("hello", max_distance=0)
        assert len(results) == 1

    def test_bktree_duplicate_items(self):
        """BK-tree with duplicate items."""
        tree = fr.BkTree()
        tree.add("hello")
        tree.add("hello")
        tree.add("hello")
        results = tree.search("hello", max_distance=0)
        # Should find matches and first result should be exact match
        assert len(results) >= 1, "Expected at least one match for duplicate items"
        assert results[0].text == "hello", f"Expected 'hello', got {results[0].text}"
        assert results[0].distance == 0, (
            f"Expected distance 0 for exact match, got {results[0].distance}"
        )

    def test_ngram_index_single_item(self):
        """N-gram index with single item."""
        index = fr.NgramIndex(ngram_size=2)
        index.add("hello")
        results = index.search("hello", min_similarity=0.5)
        assert len(results) >= 1, "Expected at least one match"
        assert results[0].text == "hello", f"Expected 'hello', got {results[0].text}"
        assert results[0].score == 1.0, (
            f"Expected score 1.0 for exact match, got {results[0].score}"
        )

    def test_hybrid_index_single_item(self):
        """Hybrid index with single item."""
        index = fr.HybridIndex(ngram_size=2)
        index.add("hello")
        results = index.search("hello", min_similarity=0.5)
        assert len(results) >= 1, "Expected at least one match"
        assert results[0].text == "hello", f"Expected 'hello', got {results[0].text}"
        assert results[0].score == 1.0, (
            f"Expected score 1.0 for exact match, got {results[0].score}"
        )

    def test_index_empty_search(self):
        """Searching with empty query."""
        tree = fr.BkTree()
        tree.add("hello")
        tree.add("world")
        # Empty string is distance 5 from "hello" and "world" (5 chars each)
        results = tree.search("", max_distance=5)
        assert len(results) == 2, (
            f"Expected 2 results for empty query with max_distance=5, got {len(results)}"
        )
        texts = {r.text for r in results}
        assert texts == {"hello", "world"}, f"Expected {{'hello', 'world'}}, got {texts}"

    def test_fuzzy_index_unicode(self):
        """FuzzyIndex with Unicode strings."""
        index = fr.FuzzyIndex(["café", "naïve", "日本語"], algorithm="ngram")
        results = index.search("cafe", min_similarity=0.5)
        # "cafe" vs "café" should match with high similarity (only accent difference)
        assert len(results) >= 1, "Expected at least one match for 'cafe' in unicode corpus"
        # The best match should be "café"
        assert results[0].text == "café", f"Expected 'café' as best match, got {results[0].text}"
        assert results[0].score >= 0.7, (
            f"Expected score >= 0.7 for 'cafe' vs 'café', got {results[0].score}"
        )


class TestEmptyStringsCoverage:
    """Empty string tests for functions without coverage."""

    def test_bigram_empty_strings(self):
        """Test bigram_similarity with empty strings."""
        assert fr.bigram_similarity("", "") == 1.0, "Both empty should be 1.0"
        assert fr.bigram_similarity("", "hello") == 0.0, "One empty should be 0.0"
        assert fr.bigram_similarity("hello", "") == 0.0, "One empty should be 0.0"

    def test_trigram_empty_strings(self):
        """Test trigram_similarity with empty strings."""
        assert fr.trigram_similarity("", "") == 1.0
        assert fr.trigram_similarity("hello", "") == 0.0
        assert fr.trigram_similarity("", "hello") == 0.0

    def test_cosine_ngrams_empty(self):
        """Test cosine_similarity_ngrams with empty strings."""
        assert fr.cosine_similarity_ngrams("", "") == 1.0
        assert fr.cosine_similarity_ngrams("", "hello") == 0.0

    def test_lcs_similarity_max_empty(self):
        """Test lcs_similarity_max with empty strings."""
        assert fr.lcs_similarity_max("", "") == 1.0
        assert fr.lcs_similarity_max("", "hello") == 0.0


class TestUnicodeEdgeCasesExtended:
    """Additional Unicode edge cases."""

    def test_control_characters(self):
        """Test handling of control characters."""
        result = fr.levenshtein("hello\x00world", "helloworld")
        assert result == 1, f"Null byte should count as 1 edit, got {result}"

    def test_byte_order_mark(self):
        """Test BOM handling."""
        bom = "\ufeff"
        result = fr.levenshtein(f"{bom}hello", "hello")
        assert result == 1, f"BOM should count as 1 edit, got {result}"

    def test_zero_width_joiner(self):
        """Test zero-width joiner handling."""
        zwj = "\u200d"
        result = fr.levenshtein(f"a{zwj}b", "ab")
        assert result == 1, f"ZWJ should count as 1 edit, got {result}"

    def test_zero_width_non_joiner(self):
        """Test zero-width non-joiner handling."""
        zwnj = "\u200c"
        result = fr.levenshtein(f"a{zwnj}b", "ab")
        assert result == 1, f"ZWNJ should count as 1 edit, got {result}"

    def test_bidirectional_text(self):
        """Test mixed LTR and RTL text."""
        ltr = "hello"
        rtl = "שלום"
        mixed = f"{ltr} {rtl}"
        # Should handle without error
        sim = fr.levenshtein_similarity(mixed, mixed)
        assert sim == 1.0, f"Identical mixed text should be 1.0, got {sim}"


class TestBoundaryConditionsExtended:
    """Test boundary conditions for find_best_matches and related functions."""

    def test_min_similarity_zero(self):
        """min_similarity=0.0 should return all results."""
        results = fr.find_best_matches(["a", "b", "c"], "x", min_similarity=0.0)
        assert len(results) == 3, f"Expected 3 results with min_similarity=0.0, got {len(results)}"

    def test_min_similarity_one(self):
        """min_similarity=1.0 should only return exact matches."""
        results = fr.find_best_matches(["hello", "hallo", "world"], "hello", min_similarity=1.0)
        assert len(results) == 1, f"Expected 1 exact match, got {len(results)}"
        assert results[0].text == "hello"

    def test_limit_zero(self):
        """limit=0 should return empty list."""
        results = fr.find_best_matches(["a", "b"], "a", limit=0)
        assert len(results) == 0, f"Expected 0 results with limit=0, got {len(results)}"

    def test_single_character_strings(self):
        """Test single character handling."""
        assert fr.levenshtein("a", "b") == 1
        assert fr.levenshtein("a", "a") == 0
        assert fr.jaro_winkler_similarity("a", "a") == 1.0
        assert fr.jaro_winkler_similarity("a", "b") == 0.0
