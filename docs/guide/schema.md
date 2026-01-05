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
- `ngram` - N-gram based similarity (trigram, n=3)
- `jaccard` - Jaccard similarity (n-gram based)
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
from fuzzyrust import polars as frp

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
result = frp.df_join(
    df1, df2,
    left_on=["first", "last"],
    right_on=["fname", "lname"],
    min_similarity=0.8,
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

## Multi-Field Matching at Scale

### Memory Considerations for SchemaIndex

SchemaIndex stores all records in memory. For large datasets:

| Records | Fields | Avg Field Length | Estimated Memory |
|---------|--------|------------------|------------------|
| 100K | 4 | 30 chars | ~200 MB |
| 1M | 4 | 30 chars | ~2 GB |
| 10M | 4 | 30 chars | ~20 GB |

### Algorithm Recommendations for Industrial Data

| Field Type | Recommended Algorithm | Why |
|------------|----------------------|-----|
| Part Numbers | levenshtein | Exact edit distance for codes |
| Names/Descriptions | jaro_winkler | Handles typos, prefix bonus |
| Manufacturer | jaro_winkler or exact | Often standardized |
| Category | exact or jaccard | Token-based matching |

### Batch Search Pattern

For searching many queries against a large index:

```python
import fuzzyrust as fr

# Define schema
schema = (
    fr.SchemaBuilder()
    .add_field("name", algorithm="jaro_winkler", weight=2.0)
    .add_field("part_number", algorithm="levenshtein", weight=3.0)
    .build()
)

# Build index once
index = fr.SchemaIndex(schema)
for record in records:
    index.add(record)

# Batch search (single Rust call, parallelized)
queries = [{"name": "...", "part_number": "..."} for _ in range(1000)]
all_results = index.batch_search(
    queries,
    min_similarity=0.8,
    limit=5,
)
```

### Large-Scale Recommendations

For datasets larger than 100K records:

1. **Use `df_dedupe_snm()` instead of `df_dedupe()`** - O(N log N) vs O(N^2)
2. **Add blocking keys** - Partition data before comparison
3. **Process in chunks** - Avoid memory pressure
4. **Pre-normalize data** - Reduce comparison overhead

See [Large-Scale Fuzzy Matching Guide](large-scale.md) for detailed strategies
