# Polars API

## DataFrame Functions

Import from `fuzzyrust.polars`:

```python
from fuzzyrust import polars as frp
```

### df_join

```python
frp.df_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    on: str | list[tuple] | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.8,
    how: Literal["inner", "left"] = "inner"
) -> pl.DataFrame
```

Fuzzy join two DataFrames.

**Parameters:**

- `left`, `right`: DataFrames to join
- `on`: Column specification. Can be:
    - A string: Same column name in both DataFrames
    - A list of tuples for multi-column join (see example below)
- `left_on`, `right_on`: Column names for single-column join (mutually exclusive with `on`)
- `algorithm`: Default similarity algorithm
- `min_similarity`: Minimum similarity score
- `how`: Join type ("inner" or "left")

**Returns:** DataFrame with all columns from both DataFrames plus `fuzzy_score` column.

**Example (multi-column with weights):**

```python
result = frp.df_join(
    left, right,
    on=[
        ("name", "customer", {"algorithm": "jaro_winkler", "weight": 2.0}),
        ("city", "location", {"algorithm": "levenshtein", "weight": 1.0}),
    ],
    min_similarity=0.7,
)
```

---

### df_dedupe

```python
frp.df_dedupe(
    df: pl.DataFrame,
    columns: list[str],
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.85,
    weights: dict[str, float] | None = None,
    algorithms: dict[str, str] | None = None,
    keep: Literal["first", "last", "most_complete"] = "first"
) -> pl.DataFrame
```

Group similar rows using Union-Find clustering.

**Parameters:**

- `df`: DataFrame to deduplicate
- `columns`: List of column names to match on
- `algorithm`: Default similarity algorithm
- `min_similarity`: Minimum similarity for grouping
- `weights`: Optional dict mapping column names to weights
- `algorithms`: Optional dict mapping column names to algorithms
- `keep`: Strategy for selecting canonical row ("first", "last", "most_complete")

**Returns:** Original DataFrame with added columns:
- `_group_id`: Integer group ID for duplicate clusters (None for unique)
- `_is_canonical`: True for the row to keep in each group

**Example:**
```python
result = frp.df_dedupe(
    df,
    columns=["name", "email"],
    algorithms={"name": "jaro_winkler", "email": "levenshtein"},
    min_similarity=0.7,
)
unique_df = result.filter(pl.col("_is_canonical"))
```

---

### df_match_pairs

```python
frp.df_match_pairs(
    df: pl.DataFrame,
    columns: str | list[str],
    min_similarity: float = 0.8,
    algorithm: str = "jaro_winkler",
    limit: int | None = None
) -> pl.DataFrame
```

Find all similar pairs within a DataFrame.

**Returns:** DataFrame with columns: `left_idx`, `right_idx`, `left_value`, `right_value`, `score`

---

### df_dedupe_snm

```python
frp.df_dedupe_snm(
    df: pl.DataFrame,
    columns: str | list[str],
    min_similarity: float = 0.8,
    window_size: int = 5,
    algorithm: str = "jaro_winkler",
    sort_key: str | None = None
) -> pl.DataFrame
```

Deduplicate using Sorted Neighborhood Method (O(N log N)).

**Parameters:**

- `window_size`: Number of neighbors to compare
- `sort_key`: Column to sort by (default: same as first column)

---

### df_find_pairs

```python
frp.df_find_pairs(
    df: pl.DataFrame,
    columns: str | list[str],
    min_similarity: float = 0.8,
    algorithm: str = "jaro_winkler",
    method: str = "snm",
    window_size: int = 5
) -> pl.DataFrame
```

Find all similar pairs.

**Parameters:**

- `method`: "snm" (O(N log N)) or "full" (O(N^2))

---

### df_match_records

```python
frp.df_match_records(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_on: str,
    right_on: str,
    min_similarity: float = 0.8,
    algorithm: str = "jaro_winkler"
) -> pl.DataFrame
```

Batch match records from two DataFrames.

---

## Series Functions

### series_similarity

```python
frp.series_similarity(
    series1: pl.Series,
    series2: pl.Series,
    algorithm: str = "jaro_winkler"
) -> pl.Series
```

Compute element-wise similarity between two series.

---

### series_best_match

```python
frp.series_best_match(
    queries: pl.Series,
    targets: list[str],
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.0,
    limit: int = 1,
    normalize: str | None = None
) -> pl.Series
```

Find best match for each query from a target list.

**Returns:** Utf8 Series with best matching target for each query (None for no match above threshold).

---

### series_match

```python
frp.series_match(
    queries: pl.Series,
    targets: pl.Series | list[str],
    min_similarity: float = 0.8,
    algorithm: str = "jaro_winkler"
) -> pl.DataFrame
```

Match each query against all targets.

---

### series_dedupe

```python
frp.series_dedupe(
    series: pl.Series,
    min_similarity: float = 0.8,
    algorithm: str = "jaro_winkler"
) -> pl.DataFrame
```

Deduplicate a series, returning groups.

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
    min_similarity: float = 0.8,
    algorithm: str = "jaro_winkler",
    ngram_size: int = 3,
    case_insensitive: bool = False
) -> pl.Expr
```

Check if values are similar to another value/column above a threshold. Returns a boolean expression.

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

### phonetic

```python
pl.col("name").fuzzy.phonetic(
    method: str = "soundex"
) -> pl.Expr
```

Generate phonetic encoding. Methods: "soundex", "metaphone".

---

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
| Hamming | `"hamming"` |
| LCS | `"lcs"` |
| Jaccard | `"jaccard"` |
| Soundex | `"soundex"` |
| Metaphone | `"metaphone"` |
