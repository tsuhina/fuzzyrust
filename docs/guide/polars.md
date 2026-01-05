# Polars Integration

FuzzyRust provides deep integration with Polars DataFrames for scalable fuzzy matching.

## API Overview

FuzzyRust offers three complementary Polars APIs:

| API | Module | Best For |
|-----|--------|----------|
| DataFrame | `fr.polars.df_*` | Joins, deduplication, pair-finding |
| Series | `fr.polars.series_*` | Column operations |
| Expression | `.fuzzy` namespace | Inline Polars expressions |

```python
import polars as pl
from fuzzyrust import polars as frp
```

## DataFrame Operations

### df_join()

Join two DataFrames on similar values:

```python
customers = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["John Smith", "Jane Doe", "Bob Wilson"]
})

orders = pl.DataFrame({
    "order_id": [101, 102, 103],
    "customer": ["Jon Smith", "Janet Doe", "Robert Wilson"]
})

result = frp.df_join(
    customers,
    orders,
    left_on="name",
    right_on="customer",
    min_similarity=0.8,
    algorithm="jaro_winkler"
)
```

### Multi-Column Join

```python
result = frp.df_join(
    df1, df2,
    left_on=["first_name", "last_name"],
    right_on=["fname", "lname"],
    min_similarity=0.8
)
```

### df_dedupe()

Group similar rows using Union-Find clustering:

```python
df = pl.DataFrame({
    "company": [
        "Acme Inc",
        "ACME Inc.",
        "Acme Incorporated",
        "Beta Corp",
        "Beta Corporation"
    ]
})

result = frp.df_dedupe(
    df,
    columns=["company"],
    min_similarity=0.85,
    algorithm="jaro_winkler"
)
# Adds 'group_id' column grouping similar rows
```

### df_match_pairs()

Find all similar pairs within a DataFrame (for manual review):

```python
pairs = frp.df_match_pairs(
    df,
    columns=["name"],
    min_similarity=0.9,
    algorithm="jaro_winkler"
)
# Returns DataFrame with: left_idx, right_idx, left_value, right_value, score
```

### df_dedupe_snm()

Sorted Neighborhood Method for O(N log N) deduplication on large datasets:

```python
result = frp.df_dedupe_snm(
    df,
    columns=["company"],
    min_similarity=0.85,
    window_size=5,
    algorithm="jaro_winkler"
)
```

### df_find_pairs()

Find all similar pairs with configurable method:

```python
# SNM method (O(N log N)) - recommended for large datasets
pairs = frp.df_find_pairs(
    df,
    columns=["name"],
    min_similarity=0.8,
    method="snm",
    window_size=5
)

# Full comparison (O(N^2)) - for smaller datasets
pairs = frp.df_find_pairs(
    df,
    columns=["name"],
    min_similarity=0.8,
    method="full"
)
```

### df_match_records()

Batch match records from two DataFrames:

```python
matches = frp.df_match_records(
    queries_df,
    targets_df,
    columns=["name"],
    min_similarity=0.8,
    algorithm="jaro_winkler"
)
```

## Series Operations

### series_similarity()

Compute element-wise similarity between two series:

```python
df = pl.DataFrame({
    "name1": ["John", "Jane", "Bob"],
    "name2": ["Jon", "Janet", "Robert"]
})

result = df.with_columns(
    score=frp.series_similarity(
        pl.col("name1"),
        pl.col("name2"),
        algorithm="jaro_winkler"
    )
)
```

### series_best_match()

Find best match for each query:

```python
queries = pl.Series(["Jon Smith", "Janet Doe"])
targets = ["John Smith", "Jane Doe", "Bob Wilson"]

matches = frp.series_best_match(
    queries,
    targets,
    algorithm="jaro_winkler",
    min_similarity=0.8
)
# Returns DataFrame with: query, match, score
```

### series_match()

Match each query against all targets:

```python
queries = pl.Series(["Jon", "Janet"])
targets = pl.Series(["John", "Jane", "Bob"])

matches = frp.series_match(
    queries,
    targets,
    min_similarity=0.8,
    algorithm="jaro_winkler"
)
```

### series_dedupe()

Deduplicate a series, returning groups:

```python
series = pl.Series(["John Smith", "Jon Smyth", "Jane Doe"])

result = frp.series_dedupe(
    series,
    min_similarity=0.8,
    algorithm="jaro_winkler"
)
```

## Expression Namespace

Use fuzzy matching directly in Polars expressions:

```python
import polars as pl
import fuzzyrust  # Registers the namespace

df = pl.DataFrame({"name": ["John", "Jane", "Jon"]})

# Similarity score
df.with_columns(
    score=pl.col("name").fuzzy.similarity("John")
)

# Filter by similarity
df.filter(
    pl.col("name").fuzzy.is_similar("John", min_similarity=0.8)
)

# Best match from list
df.with_columns(
    match=pl.col("name").fuzzy.best_match(["John Smith", "Jane Doe"])
)

# Edit distance
df.with_columns(
    dist=pl.col("name").fuzzy.distance("John", algorithm="levenshtein")
)

# Phonetic encoding
df.with_columns(
    soundex=pl.col("name").fuzzy.phonetic("soundex"),
    metaphone=pl.col("name").fuzzy.phonetic("metaphone")
)
```

## Performance Tips

1. **Use df_dedupe_snm() for large datasets** - O(N log N) vs O(N^2) for full comparison

2. **Pre-filter when possible** - Reduce candidates before fuzzy matching:
   ```python
   df.filter(pl.col("name").str.starts_with("J"))
   ```

3. **Choose appropriate thresholds** - Higher thresholds = fewer comparisons

4. **Use blocking for multi-column** - Match on exact columns first, then fuzzy on others

5. **Use the native plugin** - Column-to-column operations use the native Polars plugin automatically for 10-50x speedup

## API Comparison

| Dataset Size | Recommended API |
|--------------|-----------------|
| < 10K rows | Either (df_* is simpler) |
| 10K - 100K | df_* for simplicity, series_* for speed |
| > 100K rows | series_* API + SNM methods (required for performance) |
| > 1M rows | series_* API + df_dedupe_snm/df_find_pairs |

## Old to New API Mapping

| Old Function | New Function |
|--------------|--------------|
| `fuzzy_join()` | `frp.df_join()` |
| `fuzzy_dedupe_rows()` | `frp.df_dedupe()` |
| `match_dataframe()` | `frp.df_match_pairs()` |
| `match_series()` | `frp.series_match()` |
| `dedupe_series()` | `frp.series_dedupe()` |
| `batch_similarity()` | `frp.series_similarity()` |
| `batch_best_match()` | `frp.series_best_match()` |
| `dedupe_snm()` | `frp.df_dedupe_snm()` |
| `find_similar_pairs()` | `frp.df_find_pairs()` |
| `match_records_batch()` | `frp.df_match_records()` |
