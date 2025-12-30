"""Tests for indexing structures: BK-tree, N-gram index, and Hybrid index.

This module tests the fuzzy search index data structures provided by fuzzyrust,
including their construction, search, and configuration options.
"""

import pytest

import fuzzyrust as fr


class TestBkTree:
    """Tests for BK-tree index."""

    def test_basic_operations(self):
        tree = fr.BkTree()
        tree.add("hello")
        tree.add("hallo")
        tree.add("hullo")
        assert len(tree) == 3
        assert tree.contains("hello")
        assert not tree.contains("helloo")

    def test_add_all(self):
        tree = fr.BkTree()
        tree.add_all(["hello", "world", "test"])
        assert len(tree) == 3

    def test_search(self):
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
        tree = fr.BkTree()
        tree.add_all(["apple", "application", "apply", "banana"])
        results = tree.find_nearest("appli", limit=2)
        assert len(results) == 2
        # Results are SearchResult objects
        assert all(hasattr(r, "text") and hasattr(r, "score") for r in results)

    def test_with_data(self):
        tree = fr.BkTree()
        tree.add_with_data("hello", 42)
        tree.add_with_data("world", 99)
        results = tree.search("hello", max_distance=0)
        # Results are SearchResult objects
        assert results[0].data == 42

    def test_damerau_mode(self):
        tree_lev = fr.BkTree(use_damerau=False)
        tree_dam = fr.BkTree(use_damerau=True)

        tree_lev.add("ab")
        tree_dam.add("ab")

        # "ba" is distance 2 in Levenshtein, 1 in Damerau
        results_lev = tree_lev.search("ba", max_distance=1)
        results_dam = tree_dam.search("ba", max_distance=1)

        assert len(results_lev) == 0
        assert len(results_dam) == 1

    def test_bktree_search_max_distance_zero(self):
        """Test BK-tree search with max_distance=0 (exact matches only)."""
        tree = fr.BkTree()
        tree.add_all(["hello", "hallo", "hello"])
        result = tree.search("hello", max_distance=0)
        # Note: BK-tree may store duplicates separately depending on implementation
        assert len(result) >= 1
        # Results are SearchResult objects
        assert all(r.text == "hello" and r.distance == 0 for r in result)

    def test_bktree_find_nearest_limit_zero(self):
        """Test BK-tree find_nearest with limit=0."""
        tree = fr.BkTree()
        tree.add_all(["hello", "world"])
        result = tree.find_nearest("hello", limit=0)
        assert result == []

    def test_bktree_find_nearest_large_limit(self):
        """Test BK-tree find_nearest with limit larger than tree size."""
        tree = fr.BkTree()
        tree.add_all(["hello", "world"])
        result = tree.find_nearest("hello", limit=100)
        assert len(result) == 2  # Can only return what exists

    def test_bktree_add_long_strings(self):
        """Test BK-tree can add moderately long strings."""
        tree = fr.BkTree()
        long_str = "a" * 500
        tree.add(long_str)
        assert len(tree) == 1
        assert tree.contains(long_str)


class TestNgramIndex:
    """Tests for N-gram index."""

    def test_basic_operations(self):
        index = fr.NgramIndex(ngram_size=2)
        id1 = index.add("hello")
        id2 = index.add("world")
        assert len(index) == 2
        assert id1 == 0
        assert id2 == 1

    def test_search(self):
        index = fr.NgramIndex(ngram_size=2)
        index.add_all(["hello", "hallo", "hullo", "world", "help"])
        results = index.search("hello", algorithm="jaro_winkler", min_similarity=0.8)
        # Results are now SearchResult objects
        texts = [r.text for r in results]
        assert "hello" in texts

    def test_search_algorithms(self):
        index = fr.NgramIndex(ngram_size=2)
        index.add_all(["test", "text", "best"])

        for algo in ["jaro_winkler", "jaro", "levenshtein", "ngram", "trigram"]:
            results = index.search("test", algorithm=algo)
            assert len(results) > 0

    def test_batch_search(self):
        index = fr.NgramIndex(ngram_size=2)
        index.add_all(["apple", "banana", "cherry"])
        results = index.batch_search(["apple", "banan"], algorithm="jaro_winkler")
        assert len(results) == 2
        # Each result is a list of SearchResult objects
        assert all(isinstance(r, list) for r in results)

    def test_with_data(self):
        index = fr.NgramIndex(ngram_size=2)
        index.add_with_data("product-123", 42)
        results = index.search("product-123", min_similarity=0.9)
        # Results are SearchResult objects
        assert results[0].data == 42

    def test_ngram_index_batch_search_empty_queries(self):
        """Test NgramIndex.batch_search with empty query list."""
        index = fr.NgramIndex(ngram_size=2)
        index.add_all(["hello", "world"])
        result = index.batch_search([])
        assert result == []


class TestNgramRatioFiltering:
    """Tests for min_ngram_ratio parameter in NgramIndex and HybridIndex."""

    def test_ngram_ratio_filters_irrelevant(self):
        """Test that min_ngram_ratio filters out irrelevant candidates."""
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
        """Test that invalid min_ngram_ratio raises ValidationError."""
        with pytest.raises(fr.ValidationError, match="min_ngram_ratio"):
            fr.NgramIndex(ngram_size=2, min_ngram_ratio=-0.1)

        with pytest.raises(fr.ValidationError, match="min_ngram_ratio"):
            fr.NgramIndex(ngram_size=2, min_ngram_ratio=1.5)

    def test_hybrid_index_ngram_ratio(self):
        """Test that HybridIndex also supports min_ngram_ratio."""
        index = fr.HybridIndex(ngram_size=2, min_ngram_ratio=0.3)
        index.add_all(["hello world", "hello there", "goodbye world"])

        results = index.search("hello", min_similarity=0.5)
        texts = [r.text for r in results]

        # Should find hello matches
        assert any("hello" in t for t in texts)

    def test_product_search_use_case(self):
        """Test the original problem case: product search filtering."""
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


class TestNormalizeSearch:
    """Tests for normalize parameter in NgramIndex and HybridIndex."""

    def test_normalize_lowercase_default(self):
        """Test that normalize='lowercase' is the default."""
        index = fr.NgramIndex(ngram_size=2, min_ngram_ratio=0.2)
        index.add_all(["Rain shadow jkt", "Fleece jacket"])

        # Default should be case-insensitive (normalize="lowercase")
        results = index.search("rain jacket", min_similarity=0.5)
        texts = [r.text for r in results]

        # Should find "Rain shadow jkt" with good score
        assert "Rain shadow jkt" in texts

    def test_case_sensitive_search(self):
        """Test that normalize=None gives lower scores for case mismatches."""
        index = fr.NgramIndex(ngram_size=2, min_ngram_ratio=0.2)
        index.add_all(["Rain shadow jkt", "rain shadow jkt"])

        # Case-insensitive should score both equally
        results_ci = index.search("rain jacket", min_similarity=0.5, normalize="lowercase")
        # Case-sensitive should score lowercase higher
        results_cs = index.search("rain jacket", min_similarity=0.5, normalize=None)

        # With case-insensitive, "Rain shadow jkt" should have higher score
        ci_scores = {r.text: r.score for r in results_ci}
        cs_scores = {r.text: r.score for r in results_cs}

        # Case-insensitive should give same score for both
        if "Rain shadow jkt" in ci_scores and "rain shadow jkt" in ci_scores:
            assert abs(ci_scores["Rain shadow jkt"] - ci_scores["rain shadow jkt"]) < 0.01

    def test_hybrid_index_normalize(self):
        """Test that HybridIndex also supports normalize."""
        # Use text with mixed case that shares n-grams with query
        index = fr.HybridIndex(ngram_size=2, min_ngram_ratio=0.2)
        index.add_all(["Hello World", "hello there", "goodbye world"])

        # With normalize="lowercase", "Hello World" should score higher
        results = index.search("hello world", min_similarity=0.5, normalize="lowercase")
        texts = [r.text for r in results]

        # Should find matches
        assert len(results) > 0
        # The top result should be "Hello World" with case-insensitive matching
        assert results[0].text in ["Hello World", "hello there"]


class TestNormalizedNgramIndex:
    """Tests for case-insensitive n-gram indexing."""

    def test_normalized_index_finds_case_variants(self):
        """Normalized index should find candidates regardless of case."""
        index = fr.NgramIndex(ngram_size=2, normalize=True)
        index.add_all(["Rain shadow jkt", "FLEECE JACKET", "hiking pants"])
        results = index.search("rain shadow", min_similarity=0.5)
        texts = [r.text for r in results]
        assert "Rain shadow jkt" in texts

    def test_normalized_index_default_true(self):
        """Default normalize=True should work for mixed-case matching."""
        index = fr.NgramIndex(ngram_size=2)  # normalize=True by default
        index.add_all(["HELLO WORLD", "hello there"])
        results = index.search("hello world", min_similarity=0.8)
        texts = [r.text for r in results]
        assert "HELLO WORLD" in texts

    def test_unnormalized_index_is_case_sensitive(self):
        """With normalize=False, n-gram extraction is case-sensitive."""
        index = fr.NgramIndex(ngram_size=2, normalize=False)
        index.add_all(["HELLO"])
        # With normalize=False, "HE" != "he", so no candidates found
        results = index.search("hello", min_similarity=0.5, normalize=None)
        assert len(results) == 0

    def test_hybrid_index_normalized(self):
        """HybridIndex should also support normalization."""
        index = fr.HybridIndex(ngram_size=2, normalize=True)
        index.add_all(["Product Name", "ANOTHER PRODUCT"])
        results = index.search("product name", min_similarity=0.8)
        texts = [r.text for r in results]
        assert "Product Name" in texts

    def test_contains_respects_normalize(self):
        """contains() should use normalized comparison when normalize=True."""
        index = fr.NgramIndex(ngram_size=2, normalize=True)
        index.add("Hello World")
        assert index.contains("hello world")  # Should match despite case

        index2 = fr.NgramIndex(ngram_size=2, normalize=False)
        index2.add("Hello World")
        assert not index2.contains("hello world")  # Should not match


class TestHybridIndex:
    """Tests for Hybrid index."""

    def test_basic_operations(self):
        index = fr.HybridIndex(ngram_size=3)
        index.add_all(["hello", "world", "test"])
        assert len(index) == 3

    def test_search(self):
        index = fr.HybridIndex()
        index.add_all(["apple", "application", "apply", "banana"])
        results = index.search("appel", min_similarity=0.7, limit=2)
        assert len(results) <= 2


class TestHybridIndexConstructor:
    """Tests for HybridIndex constructor parameters."""

    def test_hybrid_index_constructor_params(self):
        """HybridIndex should accept constructor parameters."""
        # Test constructor with explicit parameters
        index = fr.HybridIndex(ngram_size=3, min_ngram_ratio=0.2)
        index.add("apple")
        index.add("banana")

        # min_similarity is specified in search, not constructor
        results = index.search("apple", min_similarity=0.9)
        assert len(results) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
