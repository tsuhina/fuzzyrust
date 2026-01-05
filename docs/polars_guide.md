# Polars Integration Guide

This comprehensive guide covers all aspects of using FuzzyRust with Polars DataFrames for scalable fuzzy string matching operations.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Expression Namespace](#expression-namespace)
3. [API Overview](#api-overview)
4. [Algorithm Options](#algorithm-options)
5. [Performance Tips](#performance-tips)
6. [Advanced Features](#advanced-features)

---

## Quick Start

### Installation

FuzzyRust is installed via pip or uv:

```bash
pip install fuzzyrust
# or
uv add fuzzyrust
```

The native Polars plugin is included by default when built with the `polars-plugin` feature.

### Basic Imports

```python
import polars as pl
import fuzzyrust as fr
from fuzzyrust import polars as frp  # New Polars API

# The .fuzzy namespace is automatically registered when importing fuzzyrust
```

### Simple Examples

```python
import polars as pl
import fuzzyrust as fr

# Create a DataFrame
df = pl.DataFrame({
    "name": ["John Smith", "Jon Smith", "Jane Doe", "Janet Doe"]
})

# Calculate similarity to a reference string
df = df.with_columns(
    score=pl.col("name").fuzzy.similarity("John Smith")
)
print(df)
# shape: (4, 2)
# ┌─────────────┬──────────┐
# │ name        ┆ score    │
# │ ---         ┆ ---      │
# │ str         ┆ f64      │
# ╞═════════════╪══════════╡
# │ John Smith  ┆ 1.0      │
# │ Jon Smith   ┆ 0.961...│
# │ Jane Doe    ┆ 0.633...│
# │ Janet Doe   ┆ 0.572...│
# └─────────────┴──────────┘

# Filter rows by similarity threshold
similar = df.filter(pl.col("name").fuzzy.is_similar("John Smith", min_similarity=0.9))
print(similar)
# shape: (2, 2)
# ┌────────────┬──────────┐
# │ name       ┆ score    │
# │ ---        ┆ ---      │
# │ str        ┆ f64      │
# ╞════════════╪══════════╡
# │ John Smith ┆ 1.0      │
# │ Jon Smith  ┆ 0.961...│
# └────────────┴──────────┘
```

---

## Expression Namespace

The `.fuzzy` namespace is registered on Polars expressions when you import `fuzzyrust`. This provides chainable fuzzy matching operations directly in Polars expression contexts.

### `similarity()`

Calculate similarity score between a column and another value or column.

```python
import polars as pl
import fuzzyrust as fr

df = pl.DataFrame({
    "name": ["John", "Jon", "Jane"],
    "reference": ["John", "John", "Janet"]
})

# Compare column to a literal string
df.with_columns(
    score=pl.col("name").fuzzy.similarity("John")
)

# Compare two columns
df.with_columns(
    score=pl.col("name").fuzzy.similarity(pl.col("reference"))
)

# Use different algorithm
df.with_columns(
    score=pl.col("name").fuzzy.similarity("John", algorithm="levenshtein")
)

# Case-insensitive comparison
df.with_columns(
    score=pl.col("name").fuzzy.similarity("JOHN", case_insensitive=True)
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `other` | `str` or `pl.Expr` | required | String literal or column to compare against |
| `algorithm` | `str` | `"jaro_winkler"` | Similarity algorithm to use |
| `ngram_size` | `int` | `3` | N-gram size for ngram algorithm |
| `case_insensitive` | `bool` | `False` | Perform case-insensitive comparison |

**Returns:** `pl.Expr` producing Float64 scores (0.0 to 1.0)

### `is_similar()`

Check if values exceed a similarity threshold (returns boolean).

```python
# Filter rows with similarity >= 0.8
df.filter(
    pl.col("name").fuzzy.is_similar("John", min_similarity=0.8)
)

# Boolean column
df.with_columns(
    is_match=pl.col("name").fuzzy.is_similar("John", min_similarity=0.85)
)

# Compare two columns
df.with_columns(
    is_match=pl.col("name").fuzzy.is_similar(
        pl.col("reference"),
        min_similarity=0.9,
        algorithm="jaro_winkler"
    )
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `other` | `str` or `pl.Expr` | required | String literal or column to compare against |
| `min_similarity` | `float` | `0.8` | Minimum similarity threshold |
| `algorithm` | `str` | `"jaro_winkler"` | Similarity algorithm to use |
| `ngram_size` | `int` | `3` | N-gram size for ngram algorithm |
| `case_insensitive` | `bool` | `False` | Perform case-insensitive comparison |

**Returns:** `pl.Expr` producing Boolean values

### `best_match()`

Find the best matching string from a list of choices.

```python
categories = ["Electronics", "Clothing", "Food", "Home & Garden"]

df = pl.DataFrame({
    "raw_category": ["electronik", "Cloths", "food items", "gardening"]
})

df.with_columns(
    category=pl.col("raw_category").fuzzy.best_match(
        categories,
        min_similarity=0.5
    )
)
# shape: (4, 2)
# ┌──────────────┬───────────────┐
# │ raw_category ┆ category      │
# │ ---          ┆ ---           │
# │ str          ┆ str           │
# ╞══════════════╪═══════════════╡
# │ electronik   ┆ Electronics   │
# │ Cloths       ┆ Clothing      │
# │ food items   ┆ Food          │
# │ gardening    ┆ Home & Garden │
# └──────────────┴───────────────┘
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `choices` | `list[str]` | required | List of strings to match against |
| `algorithm` | `str` | `"jaro_winkler"` | Similarity algorithm to use |
| `min_similarity` | `float` | `0.0` | Minimum score to return a match (otherwise null) |

**Returns:** `pl.Expr` producing String (best match or null)

### `best_match_score()`

Get both the best match and its score as a struct.

```python
candidates = ["Apple Inc", "Microsoft Corp", "Google LLC"]

df = pl.DataFrame({
    "company": ["apple", "microsft", "gogle"]
})

result = df.with_columns(
    match_result=pl.col("company").fuzzy.best_match_score(
        candidates,
        min_similarity=0.5
    )
)

# Access struct fields
result = result.select(
    pl.col("company"),
    pl.col("match_result").struct.field("match").alias("matched"),
    pl.col("match_result").struct.field("score").alias("score")
)
# shape: (3, 3)
# ┌──────────┬─────────────────┬──────────┐
# │ company  ┆ matched         ┆ score    │
# │ ---      ┆ ---             ┆ ---      │
# │ str      ┆ str             ┆ f64      │
# ╞══════════╪═════════════════╪══════════╡
# │ apple    ┆ Apple Inc       ┆ 0.777...│
# │ microsft ┆ Microsoft Corp  ┆ 0.875...│
# │ gogle    ┆ Google LLC      ┆ 0.866...│
# └──────────┴─────────────────┴──────────┘
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `choices` | `list[str]` | required | List of strings to match against |
| `algorithm` | `str` | `"jaro_winkler"` | Similarity algorithm to use |
| `min_similarity` | `float` | `0.0` | Minimum score to return a match |

**Returns:** `pl.Expr` producing Struct with fields `match` (String) and `score` (Float64)

### `distance()`

Calculate edit distance between strings.

```python
df = pl.DataFrame({
    "word": ["hello", "hallo", "world"]
})

df.with_columns(
    dist=pl.col("word").fuzzy.distance("hello")
)
# shape: (3, 2)
# ┌───────┬──────┐
# │ word  ┆ dist │
# │ ---   ┆ ---  │
# │ str   ┆ i64  │
# ╞═══════╪══════╡
# │ hello ┆ 0    │
# │ hallo ┆ 1    │
# │ world ┆ 4    │
# └───────┴──────┘

# Use different distance algorithm
df.with_columns(
    dist=pl.col("word").fuzzy.distance("hello", algorithm="damerau_levenshtein")
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `other` | `str` or `pl.Expr` | required | String literal or column to compare against |
| `algorithm` | `str` | `"levenshtein"` | Distance algorithm (levenshtein, damerau_levenshtein, hamming, lcs) |

**Returns:** `pl.Expr` producing Int64 distances

### `phonetic()`

Generate phonetic encodings for matching similar-sounding names.

```python
df = pl.DataFrame({
    "name": ["Smith", "Smyth", "Schmidt", "Robert", "Rupert"]
})

df.with_columns(
    soundex=pl.col("name").fuzzy.phonetic("soundex"),
    metaphone=pl.col("name").fuzzy.phonetic("metaphone")
)
# shape: (5, 3)
# ┌─────────┬─────────┬───────────┐
# │ name    ┆ soundex ┆ metaphone │
# │ ---     ┆ ---     ┆ ---       │
# │ str     ┆ str     ┆ str       │
# ╞═════════╪═════════╪═══════════╡
# │ Smith   ┆ S530    ┆ SM0       │
# │ Smyth   ┆ S530    ┆ SM0       │
# │ Schmidt ┆ S530    ┆ SKMTT     │
# │ Robert  ┆ R163    ┆ RBRT      │
# │ Rupert  ┆ R163    ┆ RPRT      │
# └─────────┴─────────┴───────────┘
```

Also available as shorthand methods:

```python
df.with_columns(
    soundex=pl.col("name").fuzzy.soundex(),
    metaphone=pl.col("name").fuzzy.metaphone()
)
```

### `normalize()`

Normalize strings for better matching.

```python
df = pl.DataFrame({
    "text": ["HELLO World", "  foo  bar  ", "cafe"]
})

df.with_columns(
    lower=pl.col("text").fuzzy.normalize("lowercase"),
    strict=pl.col("text").fuzzy.normalize("strict"),
    unicode=pl.col("text").fuzzy.normalize("unicode_nfkd")
)
```

**Normalization Modes:**

| Mode | Description |
|------|-------------|
| `"lowercase"` | Convert to lowercase |
| `"uppercase"` | Convert to uppercase |
| `"unicode_nfkd"` | Unicode NFKD normalization |
| `"remove_punctuation"` | Remove punctuation characters |
| `"remove_whitespace"` | Collapse whitespace |
| `"strict"` | Lowercase + remove punctuation + collapse whitespace |

---

## API Overview

FuzzyRust provides four complementary API layers for Polars integration:

### 1. Native Plugin Expressions (Fastest)

The native Polars plugin provides 10-50x speedup for column operations. It is automatically used when available.

```python
# Both literal and column comparisons use native plugin
df.with_columns(
    score1=pl.col("name").fuzzy.similarity("John"),           # Column vs literal
    score2=pl.col("name1").fuzzy.similarity(pl.col("name2"))  # Column vs column
)
```

The plugin is enabled by default. To check availability or disable:

```python
from fuzzyrust._plugin import is_plugin_available, use_native_plugin

# Check if plugin is available
print(is_plugin_available())  # True

# Disable plugin (for debugging)
use_native_plugin(False)

# Re-enable plugin
use_native_plugin(True)
```

### 2. Expression Namespace (`.fuzzy`)

The `.fuzzy` namespace provides convenient chainable methods on Polars expressions. See [Expression Namespace](#expression-namespace) section above.

### 3. High-Level API (`fuzzyrust.polars`)

User-friendly functions for common fuzzy matching operations. Best for small to medium datasets (< 100K rows).

```python
from fuzzyrust import polars as frp

# Fuzzy join two DataFrames
result = frp.df_join(
    left_df, right_df,
    left_on="name",
    right_on="customer",
    min_similarity=0.8
)

# Deduplicate DataFrame rows
deduped = frp.df_dedupe(
    df,
    columns=["name", "email"],
    min_similarity=0.85
)

# Get unique rows
unique = deduped.filter(pl.col("_is_canonical"))
```

**Available Functions:**

| Function | Description |
|----------|-------------|
| `df_join()` | Fuzzy join two DataFrames (single or multi-column) |
| `df_dedupe()` | Group duplicate rows using Union-Find clustering |
| `df_match_pairs()` | Find similar pairs within a DataFrame |
| `series_match()` | Match each query in a Series against all targets |
| `series_dedupe()` | Deduplicate a Series, grouping similar values |

#### `df_join()`

```python
# Single-column join
result = frp.df_join(
    left_df, right_df,
    left_on="name",
    right_on="customer",
    algorithm="jaro_winkler",
    min_similarity=0.8,
    how="inner"  # or "left"
)

# Multi-column join with per-column configuration
result = frp.df_join(
    left_df, right_df,
    on=[
        ("name", "customer", {"algorithm": "jaro_winkler", "weight": 2.0}),
        ("city", "location", {"algorithm": "levenshtein", "weight": 1.0}),
    ],
    min_similarity=0.7
)
```

#### `df_dedupe()`

```python
result = frp.df_dedupe(
    df,
    columns=["name", "email"],
    algorithm="jaro_winkler",
    min_similarity=0.85,
    weights={"name": 2.0, "email": 1.0},
    algorithms={"name": "jaro_winkler", "email": "levenshtein"},
    keep="first"  # "first", "last", or "most_complete"
)

# Result has added columns:
# - _group_id: Integer group ID for duplicate clusters (null for unique)
# - _is_canonical: True for the row to keep in each group
```

### 4. Batch API (`fuzzyrust.polars`)

High-performance batch operations for large datasets (100K+ rows). Uses vectorized processing and Sorted Neighborhood Method (SNM).

```python
from fuzzyrust import polars as frp

# Batch similarity between two columns
df = df.with_columns(
    score=frp.series_similarity(df["col_a"], df["col_b"])
)

# Batch best match against a list
categories = ["Electronics", "Clothing", "Food"]
matches = frp.series_best_match(df["raw_category"], categories)

# O(N log N) deduplication using SNM
result = frp.df_dedupe_snm(
    df,
    columns=["name"],
    min_similarity=0.85,
    window_size=10
)
```

**Available Functions:**

| Function | Description | Complexity |
|----------|-------------|------------|
| `series_similarity()` | Compute pairwise similarity between aligned Series | O(N) |
| `series_best_match()` | Find best match for each query from a target list | O(N * M) |
| `df_dedupe_snm()` | Deduplicate using Sorted Neighborhood Method | O(N log N) |
| `df_match_records()` | Batch match records from two DataFrames | O(N * log M) |
| `df_find_pairs()` | Find all similar pairs (SNM or full comparison) | O(N log N) / O(N^2) |

### When to Use Which API

| Dataset Size | Recommended API | Notes |
|--------------|-----------------|-------|
| < 10K rows | Either | High-level API is simpler |
| 10K - 100K rows | High-level or Batch | High-level for simplicity, Batch for speed |
| > 100K rows | Batch API | Required for performance |
| > 1M rows | Batch API + SNM | Use `df_dedupe_snm()`, `df_find_pairs()` |

---

## Algorithm Options

FuzzyRust supports multiple similarity algorithms, each optimized for different use cases.

### Available Algorithms

| Algorithm | Function | Best For | Speed |
|-----------|----------|----------|-------|
| `levenshtein` | Edit distance similarity | Typos, OCR errors | Medium |
| `damerau_levenshtein` | Edit distance with transpositions | Typos with swapped characters | Medium |
| `jaro` | Jaro similarity | Short strings, names | Fast |
| `jaro_winkler` | Jaro-Winkler similarity | Names, strings with matching prefixes | Fast |
| `ngram` | N-gram (Dice) similarity | Long text, fuzzy matching | Fast |
| `cosine` | Character-level cosine similarity | Document similarity | Fast |
| `hamming` | Hamming distance similarity | Equal-length strings | Very Fast |
| `lcs` | Longest common subsequence | Substring matching | Medium |

### Algorithm Selection Guide

```python
# Names and short strings - use Jaro-Winkler (default)
df.with_columns(
    score=pl.col("name").fuzzy.similarity("John Smith")
)

# Typos and OCR errors - use Levenshtein
df.with_columns(
    score=pl.col("text").fuzzy.similarity("document", algorithm="levenshtein")
)

# Long text - use N-gram
df.with_columns(
    score=pl.col("description").fuzzy.similarity("product description", algorithm="ngram")
)

# Equal-length codes - use Hamming
df.with_columns(
    score=pl.col("code").fuzzy.similarity("ABC123", algorithm="hamming")
)
```

### N-gram Size Configuration

For the `ngram` algorithm, you can configure the n-gram size:

```python
# Use bigrams (n=2)
df.with_columns(
    score=pl.col("text").fuzzy.similarity("query", algorithm="ngram", ngram_size=2)
)

# Use trigrams (n=3, default)
df.with_columns(
    score=pl.col("text").fuzzy.similarity("query", algorithm="ngram", ngram_size=3)
)

# Larger n-grams for longer strings
df.with_columns(
    score=pl.col("text").fuzzy.similarity("query", algorithm="ngram", ngram_size=4)
)
```

---

## Performance Tips

### 1. Use Native Plugin (Default)

The native Polars plugin provides 10-50x speedup for column-to-column and column-to-literal operations. It is enabled by default.

```python
# Both use native plugin (fast)
df.with_columns(
    score1=pl.col("name").fuzzy.similarity("John"),
    score2=pl.col("name").fuzzy.similarity(pl.col("ref"))
)
```

### 2. Use Batch API for Large Datasets (100K+ rows)

```python
from fuzzyrust import polars as frp

# Batch operations minimize Python/Rust boundary crossings
df = df.with_columns(
    score=frp.series_similarity(df["col_a"], df["col_b"])
)
```

### 3. Use SNM Methods for Million+ Row Deduplication

The Sorted Neighborhood Method (SNM) provides O(N log N) complexity instead of O(N^2):

```python
from fuzzyrust import polars as frp

# O(N log N) deduplication
result = frp.df_dedupe_snm(
    df,
    columns=["name"],
    window_size=10,  # Larger = more accurate, slower
    min_similarity=0.85
)

# O(N log N) pair finding
pairs = frp.df_find_pairs(
    df,
    columns=["name"],
    method="snm",
    window_size=20
)
```

### 4. Pre-filter When Possible

Reduce the dataset before fuzzy matching:

```python
# Filter by prefix
df = df.filter(pl.col("name").str.starts_with("J"))

# Filter by length
df = df.filter(pl.col("name").str.len_chars() > 3)

# Then apply fuzzy matching
result = df.with_columns(
    score=pl.col("name").fuzzy.similarity("John")
)
```

### 5. Use Appropriate Thresholds

Higher thresholds reduce the number of comparisons:

```python
# Strict threshold - fewer matches, faster
result = df.filter(
    pl.col("name").fuzzy.is_similar("John", min_similarity=0.95)
)

# Lenient threshold - more matches, slower
result = df.filter(
    pl.col("name").fuzzy.is_similar("John", min_similarity=0.7)
)
```

### 6. Use Blocking for Multi-Column Matching

Match on exact columns first, then fuzzy on others:

```python
# First, exact match on city
candidates = df.filter(pl.col("city") == "New York")

# Then, fuzzy match on name
result = candidates.filter(
    pl.col("name").fuzzy.is_similar("John Smith", min_similarity=0.8)
)
```

---

## Advanced Features

### Case-Insensitive Comparison

All similarity functions support case-insensitive comparison:

```python
df.with_columns(
    score=pl.col("name").fuzzy.similarity("JOHN SMITH", case_insensitive=True)
)

df.filter(
    pl.col("name").fuzzy.is_similar("JOHN", min_similarity=0.8, case_insensitive=True)
)
```

### LazyFrame Support

All functions accept both DataFrames and LazyFrames. LazyFrames are automatically collected when needed:

```python
from fuzzyrust import polars as frp

# Works with LazyFrame
lazy_df = pl.DataFrame({"name": ["John", "Jane"]}).lazy()

# Expression namespace works with LazyFrame
result = lazy_df.with_columns(
    score=pl.col("name").fuzzy.similarity("John")
).collect()

# High-level functions accept LazyFrame
result = frp.df_join(
    lazy_df1,
    lazy_df2,
    left_on="name",
    right_on="customer"
)
```

Note: Most operations require eager evaluation due to index building. For optimal performance with LazyFrames, collect once and reuse.

### Multi-Column Matching with Schema

For complex multi-field matching, use `SchemaBuilder`:

```python
import fuzzyrust as fr

# Define schema with field weights and algorithms
schema = (
    fr.SchemaBuilder()
    .add_field("name", weight=2.0, algorithm="jaro_winkler")
    .add_field("address", weight=1.0, algorithm="levenshtein")
    .add_field("phone", weight=0.5, algorithm="jaro")
    .build()
)

# Build index from records
records = [
    {"name": "John Smith", "address": "123 Main St", "phone": "555-1234"},
    {"name": "Jane Doe", "address": "456 Oak Ave", "phone": "555-5678"},
]
index = fr.SchemaIndex(schema, records)

# Search returns weighted-average scores
results = index.search({"name": "Jon Smith", "address": "123 Main"}, limit=5)
```

### Struct Output with `best_match_score()`

Get both match and score in a single operation:

```python
result = df.with_columns(
    match_result=pl.col("query").fuzzy.best_match_score(targets)
)

# Unpack struct fields
result = result.select(
    pl.col("query"),
    pl.col("match_result").struct.field("match"),
    pl.col("match_result").struct.field("score")
)
```

### Disabling Native Plugin

For debugging or comparison, you can disable the native plugin:

```python
from fuzzyrust._plugin import use_native_plugin

# Disable plugin (uses fallback map_elements implementation)
use_native_plugin(False)

# Your fuzzy operations here...

# Re-enable plugin
use_native_plugin(True)
```

Or via environment variable:

```bash
export FUZZYRUST_DISABLE_PLUGIN=1
python your_script.py
```

---

## See Also

- [Algorithms Guide](guide/algorithms.md) - Detailed algorithm descriptions
- [Indexing Guide](guide/indexing.md) - Index structures for fast search
- [Schema Guide](guide/schema.md) - Multi-field matching
- [Performance Guide](performance.md) - Benchmarks and optimization
