# Polars API

## High-Level Functions

Import from `fuzzyrust`:

```python
from fuzzyrust import fuzzy_join, fuzzy_dedupe_rows, match_dataframe
```

### fuzzy_join

```python
fuzzy_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_on: str | list[str],
    right_on: str | list[str],
    threshold: float = 0.8,
    algorithm: str = "jaro_winkler",
    how: str = "inner",
    suffix: str = "_right"
) -> pl.DataFrame
```

Fuzzy join two DataFrames.

**Parameters:**

- `left`, `right`: DataFrames to join
- `left_on`, `right_on`: Column name(s) for matching
- `threshold`: Minimum similarity score
- `algorithm`: Similarity algorithm
- `how`: Join type ("inner", "left", "right", "outer")
- `suffix`: Suffix for duplicate column names

---

### fuzzy_dedupe_rows

```python
fuzzy_dedupe_rows(
    df: pl.DataFrame,
    on: str | list[str],
    threshold: float = 0.8,
    algorithm: str = "jaro_winkler",
    group_col: str = "group_id"
) -> pl.DataFrame
```

Group similar rows using Union-Find clustering.

**Parameters:**

- `df`: DataFrame to deduplicate
- `on`: Column(s) to match on
- `threshold`: Minimum similarity for grouping
- `algorithm`: Similarity algorithm
- `group_col`: Name of output group column

---

### match_dataframe

```python
match_dataframe(
    df: pl.DataFrame,
    on: str,
    threshold: float = 0.8,
    algorithm: str = "jaro_winkler",
    limit: int | None = None
) -> pl.DataFrame
```

Find all similar pairs within a DataFrame.

**Returns:** DataFrame with columns: `left_idx`, `right_idx`, `left_value`, `right_value`, `score`

---

### match_series

```python
match_series(
    queries: pl.Series,
    targets: pl.Series | list[str],
    threshold: float = 0.8,
    algorithm: str = "jaro_winkler"
) -> pl.DataFrame
```

Match each query against all targets.

---

### dedupe_series

```python
dedupe_series(
    series: pl.Series,
    threshold: float = 0.8,
    algorithm: str = "jaro_winkler"
) -> pl.DataFrame
```

Deduplicate a series, returning groups.

---

## Batch API

Import from `fuzzyrust.polars_api`:

```python
from fuzzyrust.polars_api import batch_similarity, batch_best_match, dedupe_snm
```

### batch_similarity

```python
batch_similarity(
    series1: pl.Series,
    series2: pl.Series,
    algorithm: str = "jaro_winkler"
) -> pl.Series
```

Compute element-wise similarity between two series.

---

### batch_best_match

```python
batch_best_match(
    queries: pl.Series,
    targets: list[str] | pl.Series,
    algorithm: str = "jaro_winkler",
    threshold: float = 0.0
) -> pl.DataFrame
```

Find best match for each query.

**Returns:** DataFrame with `query`, `match`, `score`

---

### dedupe_snm

```python
dedupe_snm(
    df: pl.DataFrame,
    on: str,
    threshold: float = 0.8,
    window_size: int = 5,
    algorithm: str = "jaro_winkler",
    sort_key: str | None = None
) -> pl.DataFrame
```

Deduplicate using Sorted Neighborhood Method (O(N log N)).

**Parameters:**

- `window_size`: Number of neighbors to compare
- `sort_key`: Column to sort by (default: same as `on`)

---

### find_similar_pairs

```python
find_similar_pairs(
    df: pl.DataFrame,
    on: str,
    threshold: float = 0.8,
    algorithm: str = "jaro_winkler",
    method: str = "snm",
    window_size: int = 5
) -> pl.DataFrame
```

Find all similar pairs.

**Parameters:**

- `method`: "snm" (O(N log N)) or "full" (O(N²))

---

### match_records_batch

```python
match_records_batch(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_on: str,
    right_on: str,
    threshold: float = 0.8,
    algorithm: str = "jaro_winkler"
) -> pl.DataFrame
```

Batch match records from two DataFrames.

---

## Expression Namespace

Register the namespace by importing fuzzyrust:

```python
import polars as pl
import fuzzyrust  # Registers pl.col().fuzzy namespace
```

### similarity

```python
pl.col("name").fuzzy.similarity(
    target: str,
    algorithm: str = "jaro_winkler"
) -> pl.Expr
```

Compute similarity to a target string.

---

### is_similar

```python
pl.col("name").fuzzy.is_similar(
    target: str,
    threshold: float = 0.8,
    algorithm: str = "jaro_winkler"
) -> pl.Expr
```

Filter rows similar to target.

---

### best_match

```python
pl.col("name").fuzzy.best_match(
    choices: list[str],
    algorithm: str = "jaro_winkler"
) -> pl.Expr
```

Find best match from choices for each row.

---

### distance

```python
pl.col("name").fuzzy.distance(
    target: str,
    algorithm: str = "levenshtein"
) -> pl.Expr
```

Compute edit distance to target.

---

## Algorithms

Available algorithms for all functions:

| Algorithm | String |
|-----------|--------|
| Jaro-Winkler | `"jaro_winkler"` |
| Jaro | `"jaro"` |
| Levenshtein | `"levenshtein"` |
| Damerau-Levenshtein | `"damerau_levenshtein"` |
| N-gram | `"ngram"` |
| Cosine | `"cosine"` |
| Soundex | `"soundex"` |
| Metaphone | `"metaphone"` |
