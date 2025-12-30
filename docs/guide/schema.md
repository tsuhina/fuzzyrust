# Multi-Field Matching

FuzzyRust supports matching records across multiple fields with weighted scoring.

## Overview

When matching records with multiple fields (name, address, phone, etc.), you can:

- Use different algorithms per field
- Assign weights to fields based on importance
- Get a combined similarity score

## SchemaBuilder

Define a matching schema:

```python
import fuzzyrust as fr

schema = (
    fr.SchemaBuilder()
    .add_field("name", weight=2.0, algorithm="jaro_winkler")
    .add_field("address", weight=1.0, algorithm="levenshtein")
    .add_field("phone", weight=0.5, algorithm="jaro")
    .build()
)
```

### Field Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `name` | Field name (must match record keys) | Required |
| `weight` | Importance weight (higher = more important) | 1.0 |
| `algorithm` | Similarity algorithm to use | "jaro_winkler" |

### Available Algorithms

- `jaro_winkler` - Best for names
- `jaro` - Similar to Jaro-Winkler without prefix bonus
- `levenshtein` - Edit distance based
- `damerau_levenshtein` - Edit distance with transpositions
- `ngram` - N-gram based similarity
- `cosine` - Cosine similarity

## SchemaIndex

Build and search an index of records:

```python
# Sample records
records = [
    {"name": "John Smith", "address": "123 Main St", "phone": "555-1234"},
    {"name": "Jane Doe", "address": "456 Oak Ave", "phone": "555-5678"},
    {"name": "Bob Wilson", "address": "789 Pine Rd", "phone": "555-9012"},
]

# Build index
index = fr.SchemaIndex(schema, records)

# Search
query = {"name": "Jon Smith", "address": "123 Main"}
results = index.search(query, limit=5, min_similarity=0.7)

for r in results:
    print(f"Score: {r.score:.3f}")
    print(f"  Name: {r.record['name']}")
    print(f"  Address: {r.record['address']}")
```

## Scoring

The final score is a weighted average:

```
score = Σ(field_score × weight) / Σ(weights)
```

### Example

With schema:
- name: weight=2.0
- address: weight=1.0

And field scores:
- name: 0.95
- address: 0.80

Final score = (0.95 × 2.0 + 0.80 × 1.0) / (2.0 + 1.0) = 0.90

### Missing Fields

Fields missing from the query or record are excluded from scoring:

```python
# Query with only name
query = {"name": "John Smith"}
# Only name field contributes to score
```

## With Polars

Use multi-field matching in fuzzy joins:

```python
import polars as pl
from fuzzyrust import fuzzy_join

df1 = pl.DataFrame({
    "first": ["John", "Jane"],
    "last": ["Smith", "Doe"],
    "city": ["NYC", "LA"]
})

df2 = pl.DataFrame({
    "fname": ["Jon", "Janet"],
    "lname": ["Smith", "Doe"],
    "location": ["New York", "Los Angeles"]
})

# Multi-column fuzzy join
result = fuzzy_join(
    df1, df2,
    left_on=["first", "last"],
    right_on=["fname", "lname"],
    threshold=0.8,
    algorithm="jaro_winkler"
)
```

## Best Practices

1. **Weight important fields higher** - Name fields typically matter more than phone numbers

2. **Choose algorithms per field type**:
   - Names: `jaro_winkler`
   - Addresses: `levenshtein` or `ngram`
   - Codes/IDs: `jaro` (exact prefix matching helps)

3. **Normalize data first** - Uppercase, trim whitespace, standardize formats

4. **Start with higher thresholds** - Easier to lower than raise
