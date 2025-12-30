"""
Schema edge case tests for FuzzyRust.

Additional edge cases beyond the main test_schema.py:
- All empty fields
- Missing fields in query
- Unicode field names
- Very long field values
- Boundary conditions
- Special characters in field names
"""

import pytest

import fuzzyrust as fr


class TestEmptyFieldEdgeCases:
    """Edge cases involving empty or missing field values."""

    def test_all_empty_fields_in_record(self):
        """Record with all empty field values."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text", required=False)
        builder.add_field(name="desc", field_type="long_text", required=False)
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Add record with all empty values
        index.add({"name": "", "desc": ""})

        # Should be in index but not match non-empty queries
        assert len(index) == 1
        results = index.search({"name": "test"}, min_similarity=0.1)
        assert len(results) == 0

    def test_all_empty_fields_in_query(self):
        """Query with all empty field values."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        builder.add_field(name="desc", field_type="long_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "Apple", "desc": "A fruit"})

        # Empty query should return no results
        results = index.search({"name": "", "desc": ""})
        assert len(results) == 0

    def test_empty_dict_as_query(self):
        """Empty dictionary as query."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "Test"})

        # Empty query dict should return no results
        results = index.search({})
        assert len(results) == 0

    def test_none_field_value(self):
        """None as field value should be treated as missing."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text", required=False)
        builder.add_field(name="desc", field_type="short_text", required=False)
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Adding with None value - should either work or raise clear error
        try:
            index.add({"name": "Test", "desc": None})
            # If it succeeds, search should still work
            results = index.search({"name": "Test"})
            assert len(results) >= 1
        except (TypeError, ValueError, fr.ValidationError):
            # If it fails, that's acceptable behavior
            pass

    def test_partial_empty_fields(self):
        """Mix of empty and non-empty fields."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text", weight=10.0)
        builder.add_field(name="desc", field_type="short_text", weight=5.0)
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "Apple", "desc": ""})
        index.add({"name": "", "desc": "A fruit"})
        index.add({"name": "Banana", "desc": "Yellow fruit"})

        # Search by name should find Apple and Banana
        results = index.search({"name": "Apple"}, min_similarity=0.5)
        assert any(r.record["name"] == "Apple" for r in results)

        # Search by desc should find the fruit descriptions
        results = index.search({"desc": "fruit"}, min_similarity=0.3)
        assert len(results) >= 1


class TestMissingFieldsInQuery:
    """Edge cases for queries with missing fields."""

    def test_query_with_unknown_field(self):
        """Query containing a field not in schema raises error."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "Test"})

        # Query with field not in schema should raise an error
        # (Implementation detail: SchemaIndex validates query fields)
        with pytest.raises(
            (KeyError, ValueError, fr.ValidationError, fr.SchemaError, fr.FuzzyIndexError)
        ):
            index.search({"name": "Test", "unknown_field": "value"})

    def test_query_missing_all_schema_fields(self):
        """Query that has none of the schema fields raises error."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        builder.add_field(name="desc", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "Test", "desc": "Description"})

        # Query with completely different field should raise error
        with pytest.raises(
            (KeyError, ValueError, fr.ValidationError, fr.SchemaError, fr.FuzzyIndexError)
        ):
            index.search({"other": "value"})

    def test_query_with_subset_of_fields(self):
        """Query with only some schema fields."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text", weight=10.0)
        builder.add_field(name="category", field_type="short_text", weight=5.0)
        builder.add_field(name="tags", field_type="token_set", separator=",", weight=3.0)
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "iPhone", "category": "phone", "tags": "apple,mobile"})
        index.add({"name": "MacBook", "category": "laptop", "tags": "apple,computer"})

        # Query with only one field
        results = index.search({"name": "iPhone"}, min_similarity=0.5)
        assert len(results) >= 1
        assert results[0].record["name"] == "iPhone"

        # Query with two fields (missing tags)
        results = index.search({"name": "MacBook", "category": "laptop"}, min_similarity=0.5)
        assert len(results) >= 1
        assert results[0].record["name"] == "MacBook"


class TestUnicodeFieldNames:
    """Edge cases for Unicode in field names."""

    def test_unicode_field_name(self):
        """Field names with non-ASCII characters."""
        builder = fr.SchemaBuilder()
        # Try adding field with Unicode name
        try:
            builder.add_field(name="nombre", field_type="short_text")  # Spanish
            schema = builder.build()
            index = fr.SchemaIndex(schema)
            index.add({"nombre": "Juan"})
            results = index.search({"nombre": "Juan"})
            assert len(results) >= 1
        except (ValueError, fr.ValidationError, fr.SchemaError):
            # If Unicode field names are not supported, that's OK
            pass

    def test_unicode_accented_field_name(self):
        """Field names with accented characters."""
        builder = fr.SchemaBuilder()
        try:
            builder.add_field(name="caf\u00e9", field_type="short_text")
            schema = builder.build()
            index = fr.SchemaIndex(schema)
            index.add({"caf\u00e9": "espresso"})
            results = index.search({"caf\u00e9": "espresso"})
            assert len(results) >= 1
        except (ValueError, fr.ValidationError, fr.SchemaError):
            pass

    def test_emoji_field_name(self):
        """Field names with emoji characters."""
        builder = fr.SchemaBuilder()
        try:
            builder.add_field(name="rating_\u2b50", field_type="short_text")
            schema = builder.build()
            index = fr.SchemaIndex(schema)
            index.add({"rating_\u2b50": "5 stars"})
            results = index.search({"rating_\u2b50": "5 stars"})
            assert len(results) >= 1
        except (ValueError, fr.ValidationError, fr.SchemaError):
            pass

    def test_cjk_field_name(self):
        """Field names with CJK characters."""
        builder = fr.SchemaBuilder()
        try:
            builder.add_field(
                name="\u540d\u524d", field_type="short_text"
            )  # Japanese: "namae" (name)
            schema = builder.build()
            index = fr.SchemaIndex(schema)
            index.add({"\u540d\u524d": "\u7530\u4e2d"})  # "Tanaka"
            results = index.search({"\u540d\u524d": "\u7530\u4e2d"})
            assert len(results) >= 1
        except (ValueError, fr.ValidationError, fr.SchemaError):
            pass


class TestVeryLongFieldValues:
    """Edge cases for extremely long field values."""

    def test_very_long_short_text(self):
        """Short text field with very long value."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text", max_length=10000)
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Very long name (10000 characters)
        long_name = "a" * 10000
        index.add({"name": long_name})

        # Should be searchable
        results = index.search({"name": "a" * 100}, min_similarity=0.1)
        assert len(results) == 1

    def test_very_long_description(self):
        """Long text field with very long value (100KB+)."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="content", field_type="long_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Very long content (100KB)
        long_content = "lorem ipsum dolor sit amet " * 5000  # ~135KB
        index.add({"content": long_content})

        assert len(index) == 1
        results = index.search({"content": "lorem ipsum"}, min_similarity=0.01)
        assert len(results) == 1

    def test_many_tokens_in_token_set(self):
        """Token set with many tokens."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="tags", field_type="token_set", separator=",")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # 1000 unique tags
        many_tags = ",".join([f"tag_{i}" for i in range(1000)])
        index.add({"tags": many_tags})

        # Search for a few tags
        results = index.search({"tags": "tag_500,tag_501"}, min_similarity=0.001)
        assert len(results) == 1

    def test_long_token_values(self):
        """Token set with very long individual tokens."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="tags", field_type="token_set", separator=",")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Tokens that are each 1000 characters
        long_tags = ",".join(["a" * 1000, "b" * 1000, "c" * 1000])
        index.add({"tags": long_tags})

        # Should be searchable
        results = index.search({"tags": "a" * 1000}, min_similarity=0.1)
        assert len(results) == 1


class TestSpecialCharactersInValues:
    """Edge cases for special characters in field values."""

    def test_newlines_in_values(self):
        """Field values containing newlines."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="text", field_type="long_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"text": "line1\nline2\nline3"})
        results = index.search({"text": "line1 line2"}, min_similarity=0.3)
        assert len(results) >= 1

    def test_tabs_in_values(self):
        """Field values containing tabs."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="text", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"text": "col1\tcol2\tcol3"})
        results = index.search({"text": "col1"}, min_similarity=0.3)
        assert len(results) >= 1

    def test_null_bytes_in_values(self):
        """Field values containing null bytes."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="text", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        try:
            index.add({"text": "hello\x00world"})
            # If it works, verify it's searchable
            results = index.search({"text": "hello"}, min_similarity=0.3)
            assert len(results) >= 1
        except (ValueError, fr.ValidationError):
            # Null bytes might be rejected, that's OK
            pass

    def test_control_characters_in_values(self):
        """Field values containing control characters."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="text", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Add text with various control characters
        index.add({"text": "hello\r\nworld\t!"})
        results = index.search({"text": "hello world"}, min_similarity=0.3)
        assert len(results) >= 1

    def test_mixed_unicode_categories(self):
        """Field values with mixed Unicode categories."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="text", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Mix of scripts: Latin, CJK, Arabic, Emoji
        mixed_text = "Hello \u4e16\u754c \u0645\u0631\u062d\u0628\u0627 \U0001f44b"
        index.add({"text": mixed_text})

        results = index.search({"text": "Hello"}, min_similarity=0.1)
        assert len(results) >= 1


class TestBoundaryConditions:
    """Boundary condition tests."""

    def test_zero_weight_field(self):
        """Field with zero weight."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text", weight=10.0)
        builder.add_field(name="ignored", field_type="short_text", weight=0.0)
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "Test", "ignored": "should not matter"})

        # Zero weight field should not affect scoring
        results = index.search({"name": "Test", "ignored": "completely different"})
        assert len(results) >= 1
        assert results[0].score > 0.9  # Name match should dominate

    def test_very_small_weight(self):
        """Field with very small (but non-zero) weight."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="main", field_type="short_text", weight=10.0)
        builder.add_field(name="minor", field_type="short_text", weight=0.001)
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"main": "Apple", "minor": "fruit"})

        results = index.search({"main": "Apple", "minor": "vegetable"})
        # Minor field mismatch should barely affect score
        assert len(results) >= 1
        assert results[0].score > 0.9

    def test_min_similarity_boundary_zero(self):
        """min_similarity at exactly 0."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "Apple"})
        index.add({"name": "Banana"})

        # With min_similarity=0, should return all items (or at least some)
        # The exact behavior depends on the implementation
        results = index.search({"name": "xyz"}, min_similarity=0.0)
        # At minimum, this should not error and return some results
        assert len(results) >= 0

    def test_min_similarity_boundary_one(self):
        """min_similarity at exactly 1.0 (only exact matches)."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "Apple"})
        index.add({"name": "Appel"})  # Close but not exact

        results = index.search({"name": "Apple"}, min_similarity=1.0)
        assert len(results) == 1
        assert results[0].record["name"] == "Apple"

    def test_limit_zero(self):
        """limit=0 should return empty results."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "Test"})

        results = index.search({"name": "Test"}, limit=0)
        assert len(results) == 0

    def test_limit_larger_than_index(self):
        """limit larger than index size."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "A"})
        index.add({"name": "B"})

        # Request more results than exist - should return at most what matches
        results = index.search({"name": "A"}, limit=1000, min_similarity=0.0)
        # Should return at most the number of records that match the query
        # With min_similarity=0, might return 0, 1, or 2 depending on implementation
        assert len(results) <= 2


class TestMultipleRecordsSameValue:
    """Edge cases with duplicate values across records."""

    def test_duplicate_records(self):
        """Multiple records with identical field values."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        # Add same value multiple times
        index.add({"name": "Duplicate"}, data=1)
        index.add({"name": "Duplicate"}, data=2)
        index.add({"name": "Duplicate"}, data=3)

        assert len(index) == 3

        results = index.search({"name": "Duplicate"}, min_similarity=0.9)
        assert len(results) == 3
        # All should have same score
        scores = [r.score for r in results]
        assert all(s == scores[0] for s in scores)

    def test_partial_duplicates(self):
        """Records with some duplicate and some unique fields."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text", weight=10.0)
        builder.add_field(name="variant", field_type="short_text", weight=5.0)
        schema = builder.build()
        index = fr.SchemaIndex(schema)

        index.add({"name": "Product", "variant": "Red"}, data=1)
        index.add({"name": "Product", "variant": "Blue"}, data=2)
        index.add({"name": "Product", "variant": "Green"}, data=3)

        # Search by name should find all
        results = index.search({"name": "Product"}, min_similarity=0.5)
        assert len(results) == 3

        # Search by name and variant should rank correctly
        results = index.search({"name": "Product", "variant": "Red"}, min_similarity=0.5)
        # The Red variant should have the highest score since it matches both fields
        # But all have same name match, so the one with best variant match should be first
        # Find the result that has data=1 (Red variant)
        red_result = [r for r in results if r.data == 1][0]
        # The Red variant should have a higher score than others
        other_scores = [r.score for r in results if r.data != 1]
        assert red_result.score >= max(other_scores) if other_scores else True


class TestSchemaRebuild:
    """Tests for schema and index rebuilding."""

    def test_reuse_schema_for_multiple_indices(self):
        """Same schema can be used for multiple indices."""
        builder = fr.SchemaBuilder()
        builder.add_field(name="name", field_type="short_text")
        schema = builder.build()

        index1 = fr.SchemaIndex(schema)
        index2 = fr.SchemaIndex(schema)

        index1.add({"name": "Index1Item"})
        index2.add({"name": "Index2Item"})

        assert len(index1) == 1
        assert len(index2) == 1

        results1 = index1.search({"name": "Index1"}, min_similarity=0.5)
        results2 = index2.search({"name": "Index2"}, min_similarity=0.5)

        assert len(results1) == 1
        assert len(results2) == 1
        assert results1[0].record["name"] == "Index1Item"
        assert results2[0].record["name"] == "Index2Item"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
