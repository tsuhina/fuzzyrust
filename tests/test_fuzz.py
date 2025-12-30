"""Tests for RapidFuzz-compatible convenience functions."""

import pytest

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
        # Best window "hallo" vs "hello": 4/5 chars match = 0.8
        assert (
            0.75 <= score <= 0.85
        ), f"Expected partial_ratio 0.75-0.85 for 'hello' vs 'hallo world', got {score}"

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
        # After sorting: "hello world" vs "hello there" (11 chars each)
        # Levenshtein: 5 edits (world -> there), similarity = 6/11 = ~0.545
        assert (
            0.5 <= score <= 0.65
        ), f"Expected token_sort_ratio 0.5-0.65 for partially matching phrases, got {score}"

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
        # After deduplication: both become just "hello" = perfect match
        assert score == 1.0, f"Expected 1.0 when duplicates normalize to same set, got {score}"

    def test_overlapping_sets(self):
        """Overlapping token sets should have high score."""
        score = fr.token_set_ratio("hello world", "hello")
        # token_set_ratio takes the max of three comparisons:
        # 1. sorted intersection vs sorted intersection = 1.0
        # 2. sorted intersection vs (sorted intersection + sorted rest of s1)
        # 3. sorted intersection vs (sorted intersection + sorted rest of s2)
        # Since intersection "hello" matches perfectly, score is 1.0
        assert (
            score == 1.0
        ), f"Expected token_set_ratio 1.0 for overlapping sets with perfect intersection, got {score}"

    def test_disjoint_sets(self):
        """Completely different sets should have low score."""
        score = fr.token_set_ratio("hello world", "foo bar")
        # No overlapping tokens, similarity based on string comparison of sorted tokens
        # "hello world" (sorted) vs "bar foo" (sorted): completely different character sequences
        # token_set_ratio computes Levenshtein similarity between joined sorted token strings
        # Score should be low (0.25-0.35 range) for completely different token sets
        assert (
            0.2 <= score <= 0.4
        ), f"Expected token_set_ratio in [0.2, 0.4] for disjoint sets (no common tokens), got {score}"

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
        # "hello" vs "hallo": 4/5 chars match = 0.8 base Levenshtein similarity
        # wratio picks the best of multiple algorithms
        assert (
            0.75 <= score <= 0.90
        ), f"Expected wratio 0.75-0.90 for 1-edit-distance strings, got {score}"

    def test_substring_case(self):
        """Should handle substring cases well."""
        score = fr.wratio("test", "this is a test!")
        # "test" is a perfect substring, wratio should use partial_ratio = 1.0
        assert 0.9 <= score <= 1.0, f"Expected wratio >= 0.9 for substring match, got {score}"

    def test_reordered_words(self):
        """Should handle word reordering."""
        score = fr.wratio("hello world", "world hello")
        # Same tokens, different order: token_sort_ratio = 1.0
        # wratio may weight multiple algorithms, so we accept high score
        assert score >= 0.95, f"Expected wratio >= 0.95 for reordered words, got {score}"

    def test_empty_strings(self):
        """Edge cases with empty strings."""
        assert fr.wratio("", "") == 1.0
        assert fr.wratio("", "hello") == 0.0
        assert fr.wratio("hello", "") == 0.0

    def test_different_strings(self):
        """Completely different strings."""
        score = fr.wratio("hello", "xyz")
        # No common characters at all - should be very low
        assert score <= 0.3, f"Expected wratio <= 0.3 for completely different strings, got {score}"


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
        assert len(results) == 2, f"Expected 2 results with limit=2, got {len(results)}"
        # "apple" and "apply" are both close matches to "appel"
        assert results[0].text in [
            "apple",
            "apply",
        ], f"Expected 'apple' or 'apply' as best match, got {results[0].text}"
        # "appel" vs "apple"/"apply": wratio computes weighted combination of algorithms
        # The actual computed score is ~0.633 for both matches
        assert results[0].score == pytest.approx(
            0.633, abs=0.01
        ), f"Expected score ~0.633 for 'appel' vs '{results[0].text}', got {results[0].score}"

    def test_extract_all(self):
        """Extract with high limit should return all above cutoff."""
        choices = ["apple", "apply", "banana"]
        results = fr.extract("apple", choices, limit=10)
        assert len(results) <= len(choices)

    def test_score_cutoff(self):
        """Score cutoff should filter results."""
        choices = ["apple", "banana", "cherry"]
        results = fr.extract("apple", choices, min_similarity=0.9)
        assert len(results) == 1
        assert results[0].text == "apple"

    def test_no_matches(self):
        """No matches above cutoff should return empty list."""
        choices = ["apple", "banana", "cherry"]
        results = fr.extract("xyz", choices, min_similarity=0.9)
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
        # Both "apple" and "apply" are close matches to "appel"
        assert result.text in ["apple", "apply"], f"Expected 'apple' or 'apply', got {result.text}"
        assert 0.6 < result.score <= 1.0, f"Expected score in (0.6, 1.0], got {result.score}"

    def test_score_cutoff(self):
        """Score cutoff should filter result."""
        choices = ["apple", "banana"]
        result = fr.extract_one("xyz", choices, min_similarity=0.9)
        assert result is None

    def test_no_choices(self):
        """Empty choices should return None."""
        result = fr.extract_one("query", [])
        assert result is None

    def test_exact_match(self):
        """Exact match should return 1.0."""
        choices = ["apple", "banana", "cherry"]
        result = fr.extract_one("apple", choices)
        assert result.text == "apple", f"Expected exact match 'apple', got {result.text}"
        assert result.score == 1.0, f"Expected score 1.0 for exact match, got {result.score}"


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

        # Typo in query: "macbok pro" should match "MacBook Pro"
        result = fr.extract_one("macbok pro", products)
        assert (
            "MacBook Pro" in result.text
        ), f"Expected match containing 'MacBook Pro', got {result.text}"
        # "macbok pro" vs "Apple MacBook Pro 16-inch": partial match with typo
        # wratio computes weighted combination; actual score is ~0.54
        assert result.score == pytest.approx(
            0.54, abs=0.05
        ), f"Expected score ~0.54 for 'macbok pro' partial match with typo, got {result.score}"

    def test_name_matching(self):
        """Simulate name matching."""
        names = ["John Smith", "Jane Doe", "Bob Johnson", "Alice Williams"]

        # Reversed name: "Smith John" vs "John Smith" - same tokens, different order
        result = fr.extract_one("Smith John", names)
        assert result.text == "John Smith", f"Expected 'John Smith', got {result.text}"
        # token_sort_ratio should give perfect match for reversed names
        assert (
            0.95 <= result.score <= 1.0
        ), f"Expected score 0.95-1.0 for reversed name, got {result.score}"

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
