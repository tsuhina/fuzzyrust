"""Tests for RapidFuzz-compatible convenience functions."""

import fuzzyrust as fr


class TestPartialRatio:
    """Tests for partial_ratio function."""

    def test_identical_strings(self):
        """Identical strings should return 1.0."""
        assert fr.partial_ratio("hello", "hello") == 1.0

    def test_substring_match(self):
        """Should return 1.0 when one string is a perfect substring."""
        assert fr.partial_ratio("test", "this is a test") == 1.0
        assert fr.partial_ratio("hello", "say hello world") == 1.0

    def test_reversed_order(self):
        """Order of arguments shouldn't matter much."""
        score1 = fr.partial_ratio("test", "this is a test")
        score2 = fr.partial_ratio("this is a test", "test")
        assert score1 == score2 == 1.0

    def test_partial_match(self):
        """Should find best partial match."""
        score = fr.partial_ratio("hello", "hallo world")
        assert score > 0.7  # "hello" vs "hallo" window

    def test_empty_strings(self):
        """Edge cases with empty strings."""
        assert fr.partial_ratio("", "") == 1.0
        assert fr.partial_ratio("", "hello") == 0.0
        assert fr.partial_ratio("hello", "") == 0.0

    def test_single_char(self):
        """Single character matching."""
        assert fr.partial_ratio("a", "a") == 1.0
        assert fr.partial_ratio("a", "abc") == 1.0
        assert fr.partial_ratio("a", "bcd") == 0.0


class TestTokenSortRatio:
    """Tests for token_sort_ratio function."""

    def test_identical_strings(self):
        """Identical strings should return 1.0."""
        assert fr.token_sort_ratio("hello world", "hello world") == 1.0

    def test_reordered_words(self):
        """Word order should not affect score."""
        assert fr.token_sort_ratio("hello world", "world hello") == 1.0
        assert fr.token_sort_ratio("a b c", "c b a") == 1.0

    def test_different_words(self):
        """Different words should lower score."""
        score = fr.token_sort_ratio("hello world", "hello there")
        assert score > 0.5
        assert score < 1.0

    def test_empty_strings(self):
        """Edge cases with empty strings."""
        assert fr.token_sort_ratio("", "") == 1.0

    def test_extra_whitespace(self):
        """Extra whitespace should be normalized."""
        score = fr.token_sort_ratio("hello  world", "world   hello")
        assert score == 1.0


class TestTokenSetRatio:
    """Tests for token_set_ratio function."""

    def test_identical_strings(self):
        """Identical strings should return 1.0."""
        assert fr.token_set_ratio("hello world", "hello world") == 1.0

    def test_duplicate_words(self):
        """Duplicates should be ignored."""
        score = fr.token_set_ratio("hello hello", "hello")
        assert score > 0.9

    def test_overlapping_sets(self):
        """Overlapping token sets should have high score."""
        score = fr.token_set_ratio("hello world", "hello")
        assert score > 0.6

    def test_disjoint_sets(self):
        """Completely different sets should have low score."""
        score = fr.token_set_ratio("hello world", "foo bar")
        assert score < 0.5

    def test_empty_strings(self):
        """Edge cases with empty strings."""
        assert fr.token_set_ratio("", "") == 1.0
        assert fr.token_set_ratio("", "hello") == 0.0


class TestWRatio:
    """Tests for wratio function."""

    def test_identical_strings(self):
        """Identical strings should return 1.0."""
        assert fr.wratio("hello", "hello") == 1.0

    def test_similar_strings(self):
        """Similar strings should have high score."""
        score = fr.wratio("hello", "hallo")
        assert score > 0.7

    def test_substring_case(self):
        """Should handle substring cases well."""
        score = fr.wratio("test", "this is a test!")
        assert score > 0.8  # Should use partial_ratio

    def test_reordered_words(self):
        """Should handle word reordering."""
        score = fr.wratio("hello world", "world hello")
        assert score > 0.9  # Should use token_sort_ratio

    def test_empty_strings(self):
        """Edge cases with empty strings."""
        assert fr.wratio("", "") == 1.0
        assert fr.wratio("", "hello") == 0.0
        assert fr.wratio("hello", "") == 0.0

    def test_different_strings(self):
        """Completely different strings."""
        score = fr.wratio("hello", "xyz")
        assert score < 0.5


class TestRatio:
    """Tests for ratio function (levenshtein alias)."""

    def test_identical_strings(self):
        """Identical strings should return 1.0."""
        assert fr.ratio("hello", "hello") == 1.0

    def test_similar_strings(self):
        """Similar strings should have high score."""
        score = fr.ratio("hello", "hallo")
        assert score == 0.8  # 1 edit out of 5 chars

    def test_empty_strings(self):
        """Edge cases with empty strings."""
        assert fr.ratio("", "") == 1.0
        assert fr.ratio("", "hello") == 0.0

    def test_completely_different(self):
        """Completely different strings."""
        score = fr.ratio("abc", "xyz")
        assert score == 0.0


class TestExtract:
    """Tests for extract function."""

    def test_basic_extract(self):
        """Basic extraction with limit."""
        choices = ["apple", "apply", "banana", "cherry"]
        results = fr.extract("appel", choices, limit=2)
        assert len(results) == 2
        # Both "apple" and "apply" are close matches to "appel"
        assert results[0].text in ["apple", "apply"]
        assert results[0].score > 0.6

    def test_extract_all(self):
        """Extract with high limit should return all above cutoff."""
        choices = ["apple", "apply", "banana"]
        results = fr.extract("apple", choices, limit=10)
        assert len(results) <= len(choices)

    def test_score_cutoff(self):
        """Score cutoff should filter results."""
        choices = ["apple", "banana", "cherry"]
        results = fr.extract("apple", choices, score_cutoff=0.9)
        assert len(results) == 1
        assert results[0].text == "apple"

    def test_no_matches(self):
        """No matches above cutoff should return empty list."""
        choices = ["apple", "banana", "cherry"]
        results = fr.extract("xyz", choices, score_cutoff=0.9)
        assert len(results) == 0

    def test_empty_choices(self):
        """Empty choices should return empty list."""
        results = fr.extract("query", [], limit=5)
        assert len(results) == 0

    def test_results_sorted(self):
        """Results should be sorted by score descending."""
        choices = ["apple", "apply", "application", "banana"]
        results = fr.extract("appli", choices, limit=10)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score


class TestExtractOne:
    """Tests for extract_one function."""

    def test_basic_extract_one(self):
        """Should return best match."""
        choices = ["apple", "apply", "banana"]
        result = fr.extract_one("appel", choices)
        assert result is not None
        # Both "apple" and "apply" are close matches to "appel"
        assert result.text in ["apple", "apply"]
        assert result.score > 0.6

    def test_score_cutoff(self):
        """Score cutoff should filter result."""
        choices = ["apple", "banana"]
        result = fr.extract_one("xyz", choices, score_cutoff=0.9)
        assert result is None

    def test_no_choices(self):
        """Empty choices should return None."""
        result = fr.extract_one("query", [])
        assert result is None

    def test_exact_match(self):
        """Exact match should return 1.0."""
        choices = ["apple", "banana", "cherry"]
        result = fr.extract_one("apple", choices)
        assert result is not None
        assert result.text == "apple"
        assert result.score == 1.0


class TestFuzzIntegration:
    """Integration tests for fuzz functions."""

    def test_product_matching(self):
        """Simulate product name matching."""
        products = [
            "Apple MacBook Pro 16-inch",
            "Apple MacBook Air M2",
            "Microsoft Surface Pro",
            "Dell XPS 15",
        ]

        # Typo in query
        result = fr.extract_one("macbok pro", products)
        assert result is not None
        assert "MacBook" in result.text

    def test_name_matching(self):
        """Simulate name matching."""
        names = ["John Smith", "Jane Doe", "Bob Johnson", "Alice Williams"]

        # Reversed name
        result = fr.extract_one("Smith John", names)
        assert result is not None
        assert result.text == "John Smith"

    def test_address_matching(self):
        """Simulate address matching with variations."""
        addresses = [
            "123 Main Street",
            "456 Oak Avenue",
            "789 Pine Road",
        ]

        # Abbreviated
        results = fr.extract("123 Main St", addresses, limit=1)
        assert len(results) == 1
        assert "Main" in results[0].text

    def test_consistency_with_find_best_matches(self):
        """Extract should give similar results to find_best_matches."""
        choices = ["apple", "apply", "banana"]
        query = "appel"

        extract_results = fr.extract(query, choices, limit=3)
        best_match_results = fr.find_best_matches(choices, query, limit=3)

        # Both should find apple as best match
        assert extract_results[0].text == "apple"
        assert best_match_results[0].text == "apple"
