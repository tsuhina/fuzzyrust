# Quickstart

Get started with FuzzyRust in 5 minutes.

## Basic Similarity

```python
import fuzzyrust as fr

# Jaro-Winkler similarity (0.0 to 1.0)
fr.jaro_winkler_similarity("hello", "hallo")  # 0.88

# Levenshtein distance (edit distance)
fr.levenshtein("kitten", "sitting")  # 3

# Levenshtein similarity (normalized)
fr.levenshtein_similarity("kitten", "sitting")  # 0.57

# Case-insensitive comparison (use normalize parameter)
fr.jaro_winkler_similarity("Hello", "HELLO", normalize="lowercase")  # 1.0
```

## Batch Operations

Process lists of strings efficiently:

```python
from fuzzyrust import batch

# Find best matches from a list
choices = ["apple", "application", "apply", "banana", "orange"]
results = batch.best_matches(choices, "app", limit=3)

for match in results:
    print(f"{match.text}: {match.score:.2f}")
# apple: 0.73
# apply: 0.73
# application: 0.64

# Compute similarity scores for multiple strings
results = batch.similarity(["John", "Jon", "Jane"], "John", algorithm="jaro_winkler")
# Returns list of MatchResult: [MatchResult(text='John', score=1.0, id=0), ...]

# Deduplicate a list
result = batch.deduplicate(["John Smith", "Jon Smyth", "Jane Doe"], min_similarity=0.8)
# result.groups contains duplicate groups, result.unique contains unique strings

# Pairwise similarity between two lists
scores = batch.pairwise(["John", "Jane"], ["Jon", "Janet"])

# Full similarity matrix
matrix = batch.similarity_matrix(["John", "Jane"], ["Jon", "Janet", "Bob"])
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
from fuzzyrust import polars as frp

customers = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["John Smith", "Jane Doe", "Bob Wilson"]
})

orders = pl.DataFrame({
    "order_id": [101, 102],
    "customer_name": ["Jon Smith", "Janet Doe"]
})

# Fuzzy join on similar names
result = frp.df_join(
    customers, orders,
    left_on="name",
    right_on="customer_name",
    min_similarity=0.8
)
```

### Deduplication

```python
from fuzzyrust import polars as frp

df = pl.DataFrame({
    "company": ["Acme Inc", "ACME Inc.", "Acme Incorporated", "Beta Corp"]
})

# Group similar rows
deduped = frp.df_dedupe(df, columns=["company"], min_similarity=0.85)
```

### Series Operations

```python
from fuzzyrust import polars as frp

queries = pl.Series(["Jon Smith", "Janet Doe"])
targets = pl.Series(["John Smith", "Jane Doe", "Bob Wilson"])

# Match queries against targets
matches = frp.series_match(queries, targets, min_similarity=0.8)

# Find best match for each query
best = frp.series_best_match(queries, targets, algorithm="jaro_winkler")

# Deduplicate a series
deduped = frp.series_dedupe(targets, min_similarity=0.85)
```

### Expression Namespace

```python
import polars as pl
import fuzzyrust  # Registers the namespace

df = pl.DataFrame({"name": ["John", "Jon", "Jane", "Bob"]})

# Use fuzzy matching in Polars expressions
df.with_columns(
    score=pl.col("name").fuzzy.similarity("John Smith")
)

df.filter(
    pl.col("name").fuzzy.is_similar("John", min_similarity=0.8)
)
```

## Next Steps

- [Algorithms Guide](../guide/algorithms.md) - Learn about all available algorithms
- [Index Structures](../guide/indexing.md) - Efficient large-scale matching
- [Polars Integration](../guide/polars.md) - DataFrame operations in detail
