# Quickstart

Get started with FuzzyRust in 5 minutes.

## Basic Similarity

```python
import fuzzyrust as fr

# Jaro-Winkler similarity (0.0 to 1.0)
fr.jaro_winkler("hello", "hallo")  # 0.88

# Levenshtein distance (edit distance)
fr.levenshtein("kitten", "sitting")  # 3

# Levenshtein similarity (normalized)
fr.levenshtein_similarity("kitten", "sitting")  # 0.57

# Case-insensitive variants
fr.jaro_winkler_ci("Hello", "HELLO")  # 1.0
```

## Finding Best Matches

```python
# Find best matches from a list
choices = ["apple", "application", "apply", "banana", "orange"]
results = fr.find_best_matches("app", choices, limit=3)

for match in results:
    print(f"{match.text}: {match.score:.2f}")
# apple: 0.73
# apply: 0.73
# application: 0.64
```

## Using Index Structures

For large datasets, use index structures for faster searching:

```python
# Build an index
index = fr.NgramIndex(ngram_size=3)
index.add_all(["John Smith", "Jane Doe", "Bob Wilson", ...])

# Search (100-2000x faster than linear scan)
results = index.search("Jon Smith", algorithm="jaro_winkler", min_similarity=0.8)
```

## Polars Integration

### Fuzzy Join

```python
import polars as pl
from fuzzyrust import fuzzy_join

customers = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["John Smith", "Jane Doe", "Bob Wilson"]
})

orders = pl.DataFrame({
    "order_id": [101, 102],
    "customer_name": ["Jon Smith", "Janet Doe"]
})

# Fuzzy join on similar names
result = fuzzy_join(
    customers, orders,
    left_on="name",
    right_on="customer_name",
    threshold=0.8
)
```

### Deduplication

```python
from fuzzyrust import fuzzy_dedupe_rows

df = pl.DataFrame({
    "company": ["Acme Inc", "ACME Inc.", "Acme Incorporated", "Beta Corp"]
})

# Group similar rows
deduped = fuzzy_dedupe_rows(df, on="company", threshold=0.85)
```

### Expression Namespace

```python
# Use fuzzy matching in Polars expressions
df.with_columns(
    score=pl.col("name").fuzzy.similarity("John Smith")
)

df.filter(
    pl.col("name").fuzzy.is_similar("John", threshold=0.8)
)
```

## Next Steps

- [Algorithms Guide](../guide/algorithms.md) - Learn about all available algorithms
- [Index Structures](../guide/indexing.md) - Efficient large-scale matching
- [Polars Integration](../guide/polars.md) - DataFrame operations in detail
