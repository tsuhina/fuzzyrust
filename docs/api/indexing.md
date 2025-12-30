# Index Classes

## NgramIndex

N-gram based index for fast fuzzy search.

### Constructor

```python
NgramIndex(
    ngram_size: int = 3,
    min_similarity: float = 0.0,
    min_ngram_ratio: float = 0.0,
    normalize: bool = False
)
```

**Parameters:**

- `ngram_size`: Size of n-grams (1-32)
- `min_similarity`: Minimum similarity for results
- `min_ngram_ratio`: Minimum n-gram overlap ratio for candidates
- `normalize`: Lowercase text for case-insensitive matching

### Methods

#### add

```python
add(text: str) -> int
```

Add a string to the index. Returns the assigned ID.

#### add_with_data

```python
add_with_data(text: str, data: int | None = None) -> int
```

Add a string with optional associated data.

#### add_all

```python
add_all(texts: Iterable[str]) -> None
```

Add multiple strings.

#### search

```python
search(
    query: str,
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.0,
    limit: int | None = None
) -> list[SearchResult]
```

Search for similar strings.

**Returns:** List of `SearchResult(id, text, score, distance, data)`

#### batch_search

```python
batch_search(
    queries: list[str],
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.0,
    limit: int | None = None
) -> list[list[SearchResult]]
```

Search for multiple queries in parallel.

#### contains

```python
contains(query: str) -> bool
```

Check if exact match exists in index.

#### compress / decompress

```python
compress() -> None
decompress() -> None
```

Compress/decompress posting lists for memory efficiency.

#### is_compressed

```python
is_compressed() -> bool
```

Check if index is compressed.

#### save / load

```python
save(path: str) -> None
load(path: str) -> NgramIndex  # class method
```

Persist index to disk.

---

## BkTree

BK-tree index for edit distance queries.

### Constructor

```python
BkTree(algorithm: str = "levenshtein")
```

**Parameters:**

- `algorithm`: Distance algorithm ("levenshtein" or "damerau_levenshtein")

### Methods

#### add / add_all

```python
add(text: str) -> None
add_all(texts: Iterable[str]) -> None
```

Add strings to the tree.

#### search

```python
search(query: str, max_distance: int) -> list[SearchResult]
```

Find strings within edit distance threshold.

!!! note
    Consider using `search_similarity()` instead for consistency with NgramIndex and HybridIndex APIs.

#### search_similarity

```python
search_similarity(
    query: str,
    min_similarity: float,
    limit: int | None = None
) -> list[SearchResult]
```

Find strings above similarity threshold. **Recommended over `search()`** for API consistency.

The similarity is computed as: `1 - (distance / max(len(query), len(match)))`

#### save / load

```python
save(path: str) -> None
load(path: str) -> BkTree  # class method
```

---

## HybridIndex

Combined N-gram and similarity index.

### Constructor

```python
HybridIndex(
    ngram_size: int = 3,
    min_ngram_ratio: float = 0.0,
    normalize: bool = False
)
```

### Methods

Same as `NgramIndex`: `add`, `add_all`, `search`, `batch_search`, `contains`.

---

## SchemaBuilder

Build multi-field matching schemas.

### Methods

#### add_field

```python
add_field(
    name: str,
    weight: float = 1.0,
    algorithm: str = "jaro_winkler"
) -> SchemaBuilder
```

Add a field to the schema.

#### build

```python
build() -> Schema
```

Build the schema.

---

## SchemaIndex

Index for multi-field record matching.

### Constructor

```python
SchemaIndex(schema: Schema, records: list[dict])
```

### Methods

#### search

```python
search(
    query: dict,
    limit: int | None = None,
    min_similarity: float = 0.0
) -> list[SchemaSearchResult]
```

Search for matching records.

**Returns:** List of `SchemaSearchResult(id, score, record, field_scores)`

---

## Result Types

### SearchResult

```python
@dataclass
class SearchResult:
    id: int           # Index ID
    text: str         # Matched text
    score: float      # Similarity score
    distance: int | None  # Edit distance (if applicable)
    data: int | None  # Associated data
```

### MatchResult

```python
@dataclass
class MatchResult:
    text: str    # Matched text
    score: float # Similarity score
```

### SchemaSearchResult

```python
@dataclass
class SchemaSearchResult:
    id: int                    # Record ID
    score: float               # Combined score
    record: dict               # Full record
    field_scores: dict[str, float]  # Per-field scores
```
