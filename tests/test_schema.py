"""
Comprehensive tests for schema-based multi-field fuzzy matching.

Tests cover:
- Schema builder and validation
- Field types (ShortText, LongText, TokenSet)
- Multi-field indexing and search
- Field weighting and scoring strategies
- Normalization options
- Error handling and edge cases
"""

import pytest

import fuzzyrust as fr


class TestSchemaBuilder:
    """Test schema builder functionality."""

    def test_empty_schema_fails(self):
        """Empty schemas should fail validation."""
        builder = fr.SchemaBuilder()
        with pytest.raises(fr.SchemaError, match="must have at least one field"):
            builder.build()

    def test_single_field_schema(self):
        """Should successfully build schema with one field."""
        builder = fr.SchemaBuilder()
        builder.add_field(
            name="title", field_type="short_text", algorithm="jaro_winkler", weight=1.0
        )
        schema = builder.build()
        assert "title" in schema.field_names()

    def test_multiple_fields_schema(self):
        """Should build schema with multiple fields."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        builder.add_field(name="description", field_type="long_text")
        builder.add_field(name="tags", field_type="token_set", separator=",")

        schema = builder.build()
        field_names = schema.field_names()

        assert "name" in field_names
        assert "description" in field_names
        assert "tags" in field_names
        assert len(field_names) == 3

    def test_duplicate_field_names_fail(self):
        """Duplicate field names should fail validation."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="title", field_type="short_text")
        builder.add_field(name="title", field_type="short_text")

        with pytest.raises(fr.SchemaError, match="Duplicate field"):
            builder.build()

    def test_field_weight_validation(self):
        """Field weights must be in 0-10 range."""
        builder = fr.SchemaBuilder()
        # Valid weight should work
        builder.add_field(name="field1", field_type="short_text", weight=5.0)
        schema = builder.build()
        assert "field1" in schema.field_names(), "Schema should contain field1"

        # Weight > 10 should fail validation at add_field() time
        builder2 = fr.SchemaBuilder()
        with pytest.raises(fr.ValidationError, match="weight"):
            builder2.add_field(name="field2", field_type="short_text", weight=100.0)

        # Weight < 0 should fail validation at add_field() time
        builder3 = fr.SchemaBuilder()
        with pytest.raises(fr.ValidationError, match="weight"):
            builder3.add_field(name="field3", field_type="short_text", weight=-5.0)

    def test_scoring_strategies(self):
        """Should accept different scoring strategies."""
        # Weighted average (default)
        builder1 = fr.SchemaBuilder()
        builder1.add_field(name="name", field_type="short_text")
        schema1 = builder1.build()
        assert "name" in schema1.field_names(), "Schema1 should contain name field"

        # MinMax scaling
        builder2 = fr.SchemaBuilder()
        builder2.add_field(name="name", field_type="short_text")
        builder2.with_scoring("minmax_scaling")
        schema2 = builder2.build()
        assert "name" in schema2.field_names(), "Schema2 should contain name field"


class TestFieldTypes:
    """Test different field types."""

    def test_short_text_field(self):
        """Test ShortText field type."""
        builder = fr.SchemaBuilder()
        builder.add_field(
            name="name", field_type="short_text", algorithm="jaro_winkler", max_length=100
        )
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Add some records
        index.add({"name": "John Doe"})
        index.add({"name": "Jane Doe"})
        index.add({"name": "John Smith"})

        # Search
        results = index.search({"name": "Jon Doe"}, min_similarity=0.7)

        assert len(results) >= 1, "Expected at least one match for 'Jon Doe'"
        # "Jon Doe" is most similar to "John Doe" (1 char difference: missing 'h')
        assert results[0].record["name"] == "John Doe", (
            f"Expected 'John Doe' as best match, got {results[0].record['name']}"
        )
        # Jaro-Winkler similarity for "Jon Doe" vs "John Doe":
        # - 7 chars vs 8 chars, common prefix "Jo", one deletion
        # - Jaro-Winkler should give ~0.95-0.98 for this near-match
        assert 0.93 <= results[0].score <= 1.0, (
            f"Expected score in [0.93, 1.0] for 'Jon Doe' vs 'John Doe' (Jaro-Winkler), got {results[0].score}"
        )

    def test_long_text_field(self):
        """Test LongText field type."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="description", field_type="long_text", algorithm="ngram")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Add documents
        index.add({"description": "The quick brown fox jumps over the lazy dog"})
        index.add({"description": "A fast brown fox leaps over a sleepy dog"})
        index.add({"description": "Python is a high-level programming language"})

        # Search
        results = index.search({"description": "quick brown fox"}, min_similarity=0.3)

        assert len(results) >= 2, "Expected at least 2 matches for 'quick brown fox'"
        # The fox-related documents should rank higher than unrelated Python doc
        top_two_descriptions = [r.record["description"] for r in results[:2]]
        assert all(
            "fox" in desc.lower() or "brown" in desc.lower() for desc in top_two_descriptions
        ), f"Expected fox/brown related docs in top 2, got {top_two_descriptions}"
        # Python doc should not be the best match
        assert results[0].record["description"] != "Python is a high-level programming language"

    def test_token_set_field(self):
        """Test TokenSet field type."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="tags", field_type="token_set", separator=",")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Add items with tags
        index.add({"tags": "python,rust,golang"})
        index.add({"tags": "python,java,c++"})
        index.add({"tags": "rust,c,assembly"})

        # Search for overlap
        results = index.search({"tags": "python,rust"}, min_similarity=0.3)

        assert len(results) >= 1, "Expected at least one match for python,rust tags"
        # Best match should be the item with both python and rust tags
        tags = results[0].record["tags"]
        assert "python" in tags and "rust" in tags and "golang" in tags, (
            f"Expected 'python,rust,golang' as best match for both tags, got {tags}"
        )
        # Score: query has 2 tags (python, rust), target has 3 tags (python, rust, golang)
        # Token set overlap: 2/3 (intersection/union = 2/3) gives Jaccard ~0.67
        # The actual score depends on the token_set algorithm implementation
        assert 0.6 <= results[0].score <= 0.75, (
            f"Expected score in [0.6, 0.75] for 2/3 tag overlap, got {results[0].score}"
        )


class TestMultiFieldSearch:
    """Test multi-field search functionality."""

    def setup_method(self):
        """Create a standard multi-field schema for testing."""
        builder = fr.SchemaBuilder()
        builder.add_field(
            name="name",
            field_type="short_text",
            algorithm="jaro_winkler",
            weight=10.0,
            required=True,
        )
        builder.add_field(name="description", field_type="long_text", algorithm="ngram", weight=5.0)
        builder.add_field(name="tags", field_type="token_set", separator=",", weight=7.0)

        self.schema = builder.build()
        self.index = fr.SchemaIndex(self.schema)

        # Add sample products
        self.index.add(
            {
                "name": "MacBook Pro 14",
                "description": "High-performance laptop with M3 chip",
                "tags": "laptop,apple,computing",
            },
            data=1,
        )

        self.index.add(
            {
                "name": "MacBook Air",
                "description": "Lightweight laptop perfect for everyday use",
                "tags": "laptop,apple,portable",
            },
            data=2,
        )

        self.index.add(
            {
                "name": "ThinkPad X1",
                "description": "Business laptop with excellent keyboard",
                "tags": "laptop,lenovo,business",
            },
            data=3,
        )

        self.index.add(
            {
                "name": "iPad Pro",
                "description": "Powerful tablet with M2 chip",
                "tags": "tablet,apple,portable",
            },
            data=4,
        )

    def test_single_field_query(self):
        """Query with single field."""
        results = self.index.search({"name": "Macbook"}, min_similarity=0.5)

        assert len(results) >= 2, "Expected at least 2 MacBook matches"
        # Should find both MacBook models as top results
        names = [r.record["name"] for r in results[:2]]
        assert all("MacBook" in name for name in names), (
            f"Expected top 2 results to contain 'MacBook', got {names}"
        )

    def test_multi_field_query(self):
        """Query with multiple fields."""
        results = self.index.search({"name": "Macbook", "tags": "laptop,apple"}, min_similarity=0.5)

        assert len(results) >= 2, "Expected at least 2 results for MacBook with laptop,apple tags"
        # MacBook Pro and Air both have laptop,apple tags, should score highest
        assert "MacBook" in results[0].record["name"], (
            f"Expected MacBook as best match, got {results[0].record['name']}"
        )
        # Multi-field scoring: name (weight 10) + tags (weight 7)
        # Name: "Macbook" vs "MacBook Pro 14" (partial match, Jaro-Winkler ~0.7-0.85)
        # Tags: "laptop,apple" matches 2/3 of target tags (laptop,apple,computing)
        # Weighted average should give high score for this strong multi-field match
        assert 0.75 <= results[0].score <= 0.95, (
            f"Expected score in [0.75, 0.95] for multi-field match with strong name and tag overlap, got {results[0].score}"
        )

    def test_field_scores_included(self):
        """Results should include per-field scores."""
        results = self.index.search({"name": "Macbook", "tags": "laptop"}, min_similarity=0.3)

        assert len(results) >= 1, "Expected at least one result"

        # Check field scores are present
        result = results[0]
        assert "name" in result.field_scores, "Expected 'name' field score"
        assert "tags" in result.field_scores, "Expected 'tags' field score"

        # For MacBook, name should have high similarity to "Macbook" (case-insensitive match)
        # Jaro-Winkler for "Macbook" vs "MacBook Pro 14" or "MacBook Air": partial prefix match
        # The matching portion "Macbook" vs "MacBook" should give ~0.85-0.95
        assert 0.8 <= result.field_scores["name"] <= 1.0, (
            f"Expected name score in [0.8, 1.0] for 'Macbook' vs 'MacBook *', got {result.field_scores['name']}"
        )
        # For tags, "laptop" is 1 out of 3 tags in target set
        # Token set: query has 1 tag, target has 3 tags (laptop,apple,computing or laptop,apple,portable)
        # Jaccard similarity = 1/3 = ~0.33
        assert 0.3 <= result.field_scores["tags"] <= 0.4, (
            f"Expected tags score in [0.3, 0.4] for 1/3 tag overlap, got {result.field_scores['tags']}"
        )

    def test_weighted_scoring(self):
        """Higher weight fields should influence overall score more."""
        # Name has weight 10, description has weight 5
        # Query that matches name well but description poorly
        results = self.index.search(
            {
                "name": "MacBook Pro",  # Should match well
                "description": "unrelated text",  # Should match poorly
            },
            min_similarity=0.0,
            limit=5,
        )

        # Should get results despite poor description match (name has 2x weight)
        assert len(results) >= 1, "Expected at least one result due to name match"
        # Best result should be a MacBook (either Pro or Air matches well)
        assert "MacBook" in results[0].record["name"], (
            f"Expected a MacBook as best match, got {results[0].record['name']}"
        )

    def test_result_limit(self):
        """Should respect limit parameter."""
        results = self.index.search({"tags": "laptop"}, limit=2)

        assert len(results) <= 2

    def test_min_score_threshold(self):
        """Should filter results by minimum score."""
        # High threshold should return fewer results
        high_threshold = self.index.search({"name": "laptop"}, min_similarity=0.9)
        low_threshold = self.index.search({"name": "laptop"}, min_similarity=0.1)

        assert len(high_threshold) <= len(low_threshold)

    def test_user_data_preserved(self):
        """User data should be preserved in results."""
        results = self.index.search({"name": "MacBook Pro"}, limit=1)

        assert len(results) > 0, "Expected at least one result for 'MacBook Pro' search"
        # Data values 1-4 correspond to the four products added in setup
        assert results[0].data in [
            1,
            2,
            3,
            4,
        ], f"Expected data to be one of [1, 2, 3, 4], got {results[0].data}"


class TestNormalization:
    """Test string normalization options."""

    def test_lowercase_normalization(self):
        """Lowercase normalization should make search case-insensitive."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="title", field_type="short_text", normalize="lowercase")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"title": "HELLO WORLD"})
        results = index.search({"title": "hello world"}, min_similarity=0.9)

        assert len(results) == 1
        assert results[0].score > 0.9

    def test_strict_normalization(self):
        """Strict normalization should remove punctuation and whitespace."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="text", field_type="short_text", normalize="strict")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"text": "Hello, World!"})
        results = index.search({"text": "helloworld"}, min_similarity=0.9)

        # Should match well despite different formatting
        assert len(results) == 1


class TestValidation:
    """Test schema validation and error handling."""

    def test_required_field_validation(self):
        """Missing required fields should raise error."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text", required=True)
        builder.add_field(name="optional", field_type="short_text", required=False)
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Should succeed with required field
        index.add({"name": "Test", "optional": "Value"})

        # Should succeed with only required field
        index.add({"name": "Test2"})

        # Should fail without required field
        with pytest.raises(fr.FuzzyIndexError, match="required"):
            index.add({"optional": "Value"})

    def test_partial_field_queries(self):
        """Should handle queries with subset of fields."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text", required=True)
        builder.add_field(name="description", field_type="short_text", required=False)
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "Test", "description": "Description"})

        # Query with only one field should work
        results = index.search({"name": "Test"})
        assert len(results) >= 1

        # Query with both fields should work
        results = index.search({"name": "Test", "description": "Description"})
        assert len(results) >= 1


class TestAlgorithms:
    """Test different similarity algorithms."""

    def test_levenshtein_algorithm(self):
        """Test Levenshtein algorithm."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="text", field_type="short_text", algorithm="levenshtein")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"text": "kitten"})
        results = index.search({"text": "sitting"}, min_similarity=0.0)

        assert len(results) == 1

    def test_damerau_levenshtein_algorithm(self):
        """Test Damerau-Levenshtein algorithm."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="text", field_type="short_text", algorithm="damerau_levenshtein")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"text": "abcd"})
        results = index.search({"text": "abdc"}, min_similarity=0.0)

        # Transposition should be handled better than pure Levenshtein
        assert len(results) == 1

    def test_ngram_algorithm(self):
        """Test N-gram algorithm."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="text", field_type="long_text", algorithm="ngram")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"text": "hello world"})
        results = index.search({"text": "helo wrld"}, min_similarity=0.3)

        assert len(results) == 1

    def test_cosine_algorithm(self):
        """Test Cosine similarity algorithm."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="text", field_type="long_text", algorithm="cosine")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"text": "the quick brown fox"})
        results = index.search({"text": "quick brown"}, min_similarity=0.3)

        assert len(results) == 1

    def test_exact_match_algorithm(self):
        """Test exact match algorithm."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="id", field_type="short_text", algorithm="exact_match")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"id": "ABC123"})
        index.add({"id": "ABC124"})

        # Exact match should return only exact match
        results = index.search({"id": "ABC123"}, min_similarity=0.9)

        assert len(results) == 1
        assert results[0].record["id"] == "ABC123"
        assert results[0].score == 1.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_index_search(self):
        """Searching empty index should return empty results."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        results = index.search({"name": "test"})
        assert len(results) == 0

    def test_empty_query(self):
        """Empty query should return no results."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "test"})

        # Query with empty field value
        results = index.search({"name": ""})
        assert len(results) == 0

    def test_index_length(self):
        """Should track number of records correctly."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        assert len(index) == 0

        index.add({"name": "test1"})
        assert len(index) == 1

        index.add({"name": "test2"})
        index.add({"name": "test3"})
        assert len(index) == 3

    def test_unicode_handling(self):
        """Should handle Unicode strings correctly."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="text", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Add Unicode text
        index.add({"text": "café"})
        index.add({"text": "naïve"})
        index.add({"text": "日本語"})

        # Search should work with Unicode
        results = index.search({"text": "café"}, min_similarity=0.8)
        assert len(results) >= 1

    def test_very_long_text(self):
        """Should handle very long text fields."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="content", field_type="long_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Add very long text (1000+ characters)
        long_text = "lorem ipsum " * 100
        index.add({"content": long_text})

        results = index.search({"content": "lorem ipsum"}, min_similarity=0.1)
        assert len(results) == 1


class TestPerformance:
    """Performance and scalability tests."""

    @pytest.mark.slow
    def test_large_index(self):
        """Test with larger dataset."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text", weight=10)
        builder.add_field(name="tags", field_type="token_set", separator=",", weight=5)
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Add 1000 records
        for i in range(1000):
            index.add(
                {"name": f"Product {i}", "tags": f"tag{i % 10},tag{i % 20},category{i % 5}"}, data=i
            )

        assert len(index) == 1000

        # Search should still be fast
        results = index.search({"name": "Product 42", "tags": "tag2"}, limit=10)

        assert len(results) <= 10
        assert len(results) > 0

    def test_repeated_searches(self):
        """Multiple searches should return consistent best match."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        for i in range(100):
            index.add({"name": f"item_{i}"})

        # Search for exact match
        results1 = index.search({"name": "item_42"}, limit=5)
        results2 = index.search({"name": "item_42"}, limit=5)

        # Should return same number of results
        assert len(results1) == len(results2), (
            f"Repeated searches should return same count: {len(results1)} vs {len(results2)}"
        )
        assert len(results1) > 0, "Expected at least one result"

        # The top result should be item_42 (exact match)
        assert results1[0].id == 42, f"Expected item_42 as best match, got id {results1[0].id}"
        assert results2[0].id == 42, (
            f"Expected item_42 as best match on second search, got id {results2[0].id}"
        )

        # Top result should have perfect score for exact match
        # "item_42" vs "item_42" with Jaro-Winkler (default for short_text) should be 1.0
        assert results1[0].score == 1.0, (
            f"Expected score 1.0 for exact match 'item_42', got {results1[0].score}"
        )
        assert results2[0].score == 1.0, (
            f"Expected score 1.0 for exact match on second search, got {results2[0].score}"
        )


class TestMinMaxScalingStrategy:
    """Tests for MinMax scaling scoring strategy in actual searches."""

    def test_minmax_scaling_in_search(self):
        """Test MinMax scaling scoring strategy in actual multi-field search."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text", weight=10.0)
        builder.add_field(name="tags", field_type="token_set", separator=",", weight=5.0)
        builder.with_scoring("minmax_scaling")
        schema = builder.build()

        index = fr.SchemaIndex(schema)
        index.add({"name": "MacBook Pro", "tags": "laptop,apple"}, data=1)
        index.add({"name": "iPad Pro", "tags": "tablet,apple"}, data=2)
        index.add({"name": "iPhone", "tags": "phone,apple"}, data=3)

        results = index.search({"name": "Macbook", "tags": "laptop"}, min_similarity=0.3)

        assert len(results) >= 1, "Expected at least one result for MacBook with laptop tag"
        # Verify the top match is MacBook Pro (best name + tag match)
        assert "MacBook" in results[0].record["name"], (
            f"Expected MacBook as best match, got {results[0].record['name']}"
        )
        # MinMax scaling should produce scores that reflect the relative quality of matches
        # Top match (MacBook Pro): strong name match + partial tag match
        # Score should be in reasonable range for multi-field weighted average
        assert 0.5 <= results[0].score <= 0.9, (
            f"Expected top score in [0.5, 0.9] for MinMax scaled multi-field match, got {results[0].score}"
        )
        for r in results:
            # All scores should be in valid [0, 1] range with MinMax scaling
            assert 0.0 <= r.score <= 1.0, f"Overall score {r.score} out of valid [0, 1] range"
            # Field scores should also be in valid range
            for field_name, field_score in r.field_scores.items():
                assert 0.0 <= field_score <= 1.0, (
                    f"Field '{field_name}' score {field_score} out of valid [0, 1] range"
                )

    def test_minmax_vs_weighted_average(self):
        """Compare MinMax and weighted average strategies."""
        # Create two indexes with different strategies
        builder1 = fr.SchemaBuilder()
        builder1.add_field(name="name", field_type="short_text", weight=10.0)
        builder1.with_scoring("weighted_average")
        schema1 = builder1.build()

        builder2 = fr.SchemaBuilder()
        builder2.add_field(name="name", field_type="short_text", weight=10.0)
        builder2.with_scoring("minmax_scaling")
        schema2 = builder2.build()

        index1 = fr.SchemaIndex(schema1)
        index2 = fr.SchemaIndex(schema2)

        for i, name in enumerate(["Apple", "Banana", "Cherry"]):
            index1.add({"name": name}, data=i)
            index2.add({"name": name}, data=i)

        results1 = index1.search({"name": "Apple"})
        results2 = index2.search({"name": "Apple"})

        # Both should return the exact match with perfect score
        assert len(results1) >= 1, "Weighted average strategy should find Apple"
        assert len(results2) >= 1, "MinMax scaling strategy should find Apple"

        # First result should be "Apple" with score 1.0 for both strategies
        assert results1[0].record["name"] == "Apple", (
            f"Weighted average: expected 'Apple', got {results1[0].record['name']}"
        )
        assert results2[0].record["name"] == "Apple", (
            f"MinMax scaling: expected 'Apple', got {results2[0].record['name']}"
        )
        # Exact match should have score 1.0
        assert results1[0].score == 1.0, (
            f"Weighted average: expected score 1.0 for exact match, got {results1[0].score}"
        )
        assert results2[0].score == 1.0, (
            f"MinMax scaling: expected score 1.0 for exact match, got {results2[0].score}"
        )


class TestSchemaIndexGet:
    """Tests for SchemaIndex.get() method."""

    def test_get_by_id(self):
        """Test retrieving records by ID."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "First Item"}, data=100)
        index.add({"name": "Second Item"}, data=200)
        index.add({"name": "Third Item"}, data=300)

        # Get by ID (IDs are 0-based)
        assert index.get(0)["name"] == "First Item", "Record 0 should be 'First Item'"
        assert index.get(1)["name"] == "Second Item", "Record 1 should be 'Second Item'"
        assert index.get(2)["name"] == "Third Item", "Record 2 should be 'Third Item'"

    def test_get_nonexistent_id(self):
        """Test getting a non-existent ID returns None."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "Only Item"})

        # Non-existent IDs should return None
        assert index.get(999) is None
        assert index.get(100) is None

    def test_get_from_empty_index(self):
        """Test getting from empty index returns None."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        assert index.get(0) is None


class TestTokenSetEdgeCases:
    """Edge case tests for TokenSet field type."""

    def test_token_set_empty_value(self):
        """Test TokenSet with empty value."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="tags", field_type="token_set", separator=",")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"tags": ""})
        results = index.search({"tags": "test"}, min_similarity=0.0)
        # Empty tag set shouldn't match
        assert len(results) == 0

    def test_token_set_whitespace_only(self):
        """Test TokenSet with whitespace-only tokens."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="tags", field_type="token_set", separator=",")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"tags": "  ,  ,  "})
        results = index.search({"tags": "test"}, min_similarity=0.0)
        # Whitespace-only tokens should be filtered out
        assert len(results) == 0

    def test_token_set_single_token(self):
        """Test TokenSet with a single token."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="tags", field_type="token_set", separator=",")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"tags": "solo"})
        results = index.search({"tags": "solo"}, min_similarity=0.5)
        assert len(results) == 1
        assert results[0].field_scores["tags"] == 1.0

    def test_token_set_different_separators(self):
        """Test TokenSet with different separator characters."""
        for sep in [",", ";", "|"]:
            builder = fr.SchemaBuilder()
            builder.add_field(name="tags", field_type="token_set", separator=sep)
            schema = builder.build()
            index = fr.SchemaIndex(schema)

            index.add({"tags": f"a{sep}b{sep}c"})
            results = index.search({"tags": f"a{sep}b"}, min_similarity=0.5)
            assert len(results) >= 1, f"Failed with separator: {sep}"

    def test_token_set_with_spaces(self):
        """Test TokenSet correctly trims whitespace around tokens."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="tags", field_type="token_set", separator=",")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"tags": " apple , banana , cherry "})
        results = index.search({"tags": "apple,banana"}, min_similarity=0.5)
        assert len(results) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
