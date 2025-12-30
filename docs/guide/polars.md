# Polars Integration

FuzzyRust provides deep integration with Polars DataFrames for scalable fuzzy matching.

## Two APIs

FuzzyRust offers two complementary APIs:

| API | Module | Best For |
|-----|--------|----------|
| High-Level | `polars_ext` | Ease of use, < 100K rows |
| Batch | `polars_api` | Performance, > 100K rows |

## High-Level API

### Fuzzy Join

Join two DataFrames on similar values:

```python
import polars as pl
from fuzzyrust import fuzzy_join

customers = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["John Smith", "Jane Doe", "Bob Wilson"]
})

orders = pl.DataFrame({
    "order_id": [101, 102, 103],
    "customer": ["Jon Smith", "Janet Doe", "Robert Wilson"]
})

result = fuzzy_join(
    customers,
    orders,
    left_on="name",
    right_on="customer",
    threshold=0.8,
    algorithm="jaro_winkler"
)
```

### Multi-Column Join

```python
result = fuzzy_join(
    df1, df2,
    left_on=["first_name", "last_name"],
    right_on=["fname", "lname"],
    threshold=0.8
)
```

### Deduplication

Group similar rows:

```python
from fuzzyrust import fuzzy_dedupe_rows

df = pl.DataFrame({
    "company": [
        "Acme Inc",
        "ACME Inc.",
        "Acme Incorporated",
        "Beta Corp",
        "Beta Corporation"
    ]
})

result = fuzzy_dedupe_rows(
    df,
    on="company",
    threshold=0.85,
    algorithm="jaro_winkler"
)
# Adds 'group_id' column grouping similar rows
```

### Match DataFrame

Find all similar pairs within a DataFrame:

```python
from fuzzyrust import match_dataframe

pairs = match_dataframe(
    df,
    on="name",
    threshold=0.9,
    algorithm="jaro_winkler"
)
# Returns DataFrame with: left_idx, right_idx, left_value, right_value, score
```

## Batch API

For large datasets (100K+ rows), use the batch API:

### Batch Similarity

```python
from fuzzyrust.polars_api import batch_similarity

df = pl.DataFrame({
    "name1": ["John", "Jane", "Bob"],
    "name2": ["Jon", "Janet", "Robert"]
})

result = df.with_columns(
    score=batch_similarity(
        pl.col("name1"),
        pl.col("name2"),
        algorithm="jaro_winkler"
    )
)
```

### Batch Best Match

```python
from fuzzyrust.polars_api import batch_best_match

queries = pl.Series(["Jon Smith", "Janet Doe"])
targets = ["John Smith", "Jane Doe", "Bob Wilson"]

matches = batch_best_match(
    queries,
    targets,
    algorithm="jaro_winkler",
    threshold=0.8
)
```

### SNM Deduplication

Sorted Neighborhood Method for O(N log N) deduplication:

```python
from fuzzyrust.polars_api import dedupe_snm

result = dedupe_snm(
    df,
    on="company",
    threshold=0.85,
    window_size=5,
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
    pl.col("name").fuzzy.is_similar("John", threshold=0.8)
)

# Best match from list
df.with_columns(
    match=pl.col("name").fuzzy.best_match(["John Smith", "Jane Doe"])
)
```

## Performance Tips

1. **Use batch API for large datasets** - Vectorized operations are much faster

2. **Pre-filter when possible** - Reduce candidates before fuzzy matching:
   ```python
   df.filter(pl.col("name").str.starts_with("J"))
   ```

3. **Use SNM for deduplication** - O(N log N) vs O(N²) for full comparison

4. **Choose appropriate thresholds** - Higher thresholds = fewer comparisons

5. **Use blocking for multi-column** - Match on exact columns first, then fuzzy on others

## API Comparison

| Dataset Size | Recommended API |
|--------------|-----------------|
| < 10K rows | Either (high-level is simpler) |
| 10K - 100K | High-level for simplicity, batch for speed |
| > 100K rows | Batch API (required for performance) |
| > 1M rows | Batch API + SNM methods |
