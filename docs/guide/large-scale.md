# Large-Scale Fuzzy Matching Guide

This guide covers best practices for fuzzy matching operations on large datasets (100K to 10M+ records).

## When to Use What

Use this decision tree to choose the right function:

| Dataset Size | Recommended Functions | Notes |
|--------------|----------------------|-------|
| < 10K rows | `df_dedupe()`, `df_join()` | Simple and fast, O(N^2) acceptable |
| 10K - 100K rows | `df_dedupe()` or `df_dedupe_snm()` | Consider SNM for better scaling |
| 100K - 1M rows | `df_dedupe_snm()` with blocking | Use blocking key to partition data |
| > 1M rows | `df_dedupe_snm()` with blocking + larger window | Essential optimizations required |

## Memory Estimation

Approximate memory requirements for multi-field matching:

| Records | Columns | Avg Field Length | Estimated Memory | Time (approx) |
|---------|---------|------------------|------------------|---------------|
| 100K | 4 | 30 chars | ~500 MB | 10-30s |
| 1M | 4 | 30 chars | ~2-4 GB | 2-5 min |
| 10M | 4 | 30 chars | ~20-40 GB | 20-60 min |

**Rule of thumb**: Allocate ~100 bytes per record per indexed field.

## Complexity Comparison

Understanding algorithm complexity is crucial for large-scale operations:

### O(N^2) vs O(N log N)

- `df_dedupe()`: **O(N^2)** - compares every pair, avoid for > 100K records
- `df_dedupe_snm()`: **O(N * W * log N)** where W = window_size

For 1M records:
- O(N^2) = 1,000,000,000,000 comparisons (impractical)
- O(N * W * log N) with W=50 = ~1,000,000,000 comparisons (manageable)

## Window Size Recommendations

The `window_size` parameter in SNM controls accuracy vs speed tradeoff:

| Dataset Size | Recommended Window | Accuracy Trade-off |
|--------------|-------------------|-------------------|
| < 100K | 10-20 | Good |
| 100K - 500K | 20-50 | Good |
| 500K - 1M | 50-100 | Moderate |
| > 1M | 100-200 | Lower (use blocking) |

Larger windows catch more duplicates but take longer. When accuracy drops, use blocking instead of increasing window size.

## Blocking Key Strategies

### What is Blocking?

Blocking partitions records before comparison, reducing complexity from O(N^2) to O(B * (N/B)^2) where B is the number of blocks. With good blocking, this approaches O(N).

```python
from fuzzyrust import polars as frp

# Block by category - only compare within same category
result = frp.df_dedupe_snm(
    df,
    columns=["name", "manufacturer"],
    blocking_key="category",  # Use category column as block key
    min_similarity=0.85,
    window_size=30,
)
```

### Recommended Blocking Keys for Industrial Data

| Data Type | Blocking Key | Example |
|-----------|--------------|---------|
| Part numbers | First 3-4 characters | "PN-001234" -> "PN-0" |
| Names | First character (normalized) | "Bosch" -> "B" |
| Manufacturer | Exact match (pre-normalized) | "BOSCH" |
| Category | Exact match | "Bearings" |
| Composite | category + first_char(name) | "Bearings-B" |

### Custom Blocking Keys

Create blocking keys with any Python callable:

```python
# Block by first letter of name
result = frp.df_dedupe_snm(
    df,
    columns=["name"],
    blocking_key=lambda df: df["name"].str.slice(0, 1).str.to_uppercase(),
    min_similarity=0.85,
)

# Composite blocking key
result = frp.df_dedupe_snm(
    df,
    columns=["name", "manufacturer"],
    blocking_key=lambda df: (
        df["category"] + "-" + df["manufacturer"].str.slice(0, 2)
    ),
    min_similarity=0.85,
)
```

## Best Practices

### 1. Normalize Data First

Always normalize strings before matching:

```python
import polars as pl

df = df.with_columns([
    pl.col("name").str.to_lowercase().str.strip_chars().alias("name_norm"),
    pl.col("manufacturer").str.to_uppercase().alias("mfr_norm"),
])

# Match on normalized columns
result = frp.df_dedupe_snm(df, columns=["name_norm", "mfr_norm"])
```

### 2. Pre-filter Categories

Match within categories when possible:

```python
# Process each category separately
results = []
for category in df["category"].unique().to_list():
    subset = df.filter(pl.col("category") == category)
    deduped = frp.df_dedupe_snm(
        subset,
        columns=["name"],
        min_similarity=0.85,
    )
    results.append(deduped)

final = pl.concat(results)
```

### 3. Use Appropriate Algorithms per Field

Different field types benefit from different algorithms:

| Field Type | Recommended Algorithm | Reason |
|------------|----------------------|--------|
| Part numbers | `levenshtein` | Exact edit distance for codes |
| Names/Descriptions | `jaro_winkler` | Handles typos, prefix bonus |
| Manufacturer | `jaro_winkler` or exact | Often standardized |
| Category | `jaccard` or exact | Token-based matching |

```python
result = frp.df_dedupe_snm(
    df,
    columns=["part_number", "name", "manufacturer"],
    algorithms={
        "part_number": "levenshtein",
        "name": "jaro_winkler",
        "manufacturer": "jaro_winkler",
    },
    weights={
        "part_number": 3.0,  # Most important
        "name": 2.0,
        "manufacturer": 1.0,
    },
    min_similarity=0.85,
)
```

### 4. Use Batch Operations

Never loop over individual searches:

```python
# BAD: Loop with individual searches
for query in queries:
    index.search(query)  # Creates overhead per query

# GOOD: Single batch call
index.batch_search(queries)  # Parallelized internally
```

### 5. Monitor Memory

For very large datasets, process in chunks:

```python
# Process 100K rows at a time
chunk_size = 100_000
results = []

for i in range(0, len(df), chunk_size):
    chunk = df.slice(i, chunk_size)
    deduped = frp.df_dedupe_snm(
        chunk,
        columns=["name"],
        min_similarity=0.85,
    )
    results.append(deduped)
```

## Performance Comparison

### Deduplication Methods

| Method | Complexity | 10K rows | 100K rows | 1M rows |
|--------|-----------|----------|-----------|---------|
| `df_dedupe()` | O(N^2) | 2s | 3 min | 5+ hours |
| `df_dedupe_snm()` | O(N*W*log N) | 0.5s | 5s | 1 min |
| `df_dedupe_snm()` + blocking | O(B*(N/B)*W*log(N/B)) | 0.3s | 2s | 20s |

*Times are approximate and depend on hardware and data characteristics.*

### Index Search Methods

For querying against a reference dataset:

| Method | Build Time | Query Time | Memory |
|--------|-----------|-----------|--------|
| `SchemaIndex` | ~1s/100K | ~1ms/query | ~200MB/100K |
| `NgramIndex` | ~2s/100K | ~0.5ms/query | ~100MB/100K |
| `HybridIndex` | ~3s/100K | ~0.3ms/query | ~300MB/100K |

## Common Pitfalls

### 1. Using O(N^2) Methods on Large Data

```python
# WRONG: Will take hours for 1M rows
result = frp.df_dedupe(df_1m_rows, columns=["name"])

# RIGHT: Use SNM for O(N log N)
result = frp.df_dedupe_snm(df_1m_rows, columns=["name"], window_size=50)
```

### 2. Window Size Too Small

A small window may miss duplicates that are far apart after sorting:

```python
# May miss some duplicates
result = frp.df_dedupe_snm(df, columns=["name"], window_size=5)

# Better coverage
result = frp.df_dedupe_snm(df, columns=["name"], window_size=30)
```

### 3. No Blocking on Very Large Data

For > 1M rows, always use blocking:

```python
# Slow without blocking
result = frp.df_dedupe_snm(df_10m, columns=["name"], window_size=100)

# Much faster with blocking
result = frp.df_dedupe_snm(
    df_10m,
    columns=["name"],
    blocking_key="category",
    window_size=30,
)
```

## See Also

- [Multi-Field Matching](schema.md) - Schema-based matching with weights
- [Polars Integration](polars.md) - DataFrame operations
- [Performance](../performance.md) - Benchmarks and optimization tips
