# Core Functions

## Similarity Functions

All similarity functions return a score between 0.0 (no match) and 1.0 (exact match).

### jaro_winkler_similarity

```python
jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float
```

Jaro-Winkler similarity. Best for short strings like names.

**Parameters:**

- `s1`, `s2`: Strings to compare
- `prefix_weight`: Weight for matching prefix (0.0-0.25, default 0.1)

**Example:**

```python
fr.jaro_winkler_similarity("MARTHA", "MARHTA")  # 0.961
```

---

### jaro_similarity

```python
jaro_similarity(s1: str, s2: str) -> float
```

Jaro similarity (without Winkler prefix bonus).

---

### levenshtein_similarity

```python
levenshtein_similarity(s1: str, s2: str) -> float
```

Normalized Levenshtein similarity: `1 - (distance / max_length)`.

---

### ngram_similarity

```python
ngram_similarity(s1: str, s2: str, ngram_size: int = 2) -> float
```

N-gram similarity using Dice coefficient.

---

### cosine_similarity_words

```python
cosine_similarity_words(s1: str, s2: str) -> float
```

Cosine similarity of word vectors (space-separated tokens).

---

### cosine_similarity_chars

```python
cosine_similarity_chars(s1: str, s2: str) -> float
```

Cosine similarity of character vectors.

---

### cosine_similarity_ngrams

```python
cosine_similarity_ngrams(s1: str, s2: str, ngram_size: int = 2) -> float
```

Cosine similarity of character n-gram vectors.

---

### soundex_similarity

```python
soundex_similarity(s1: str, s2: str) -> float
```

Phonetic similarity using Soundex codes.

---

### metaphone_similarity

```python
metaphone_similarity(s1: str, s2: str) -> float
```

Phonetic similarity using Metaphone codes.

---

## Distance Functions

Distance functions return integers (lower = more similar).

### levenshtein

```python
levenshtein(s1: str, s2: str) -> int
```

Levenshtein edit distance.

---

### levenshtein_bounded

```python
levenshtein_bounded(s1: str, s2: str, max_distance: int) -> int | None
```

Levenshtein distance with early termination. Returns `None` if distance exceeds threshold.

---

### damerau_levenshtein

```python
damerau_levenshtein(s1: str, s2: str) -> int
```

Damerau-Levenshtein distance (includes transpositions).

---

### hamming

```python
hamming(s1: str, s2: str) -> int
```

Hamming distance. Raises error if strings have different lengths.

---

## Phonetic Functions

### soundex

```python
soundex(s: str) -> str
```

Returns Soundex code (e.g., "R163" for "Robert").

---

### metaphone

```python
metaphone(s: str) -> str
```

Returns Metaphone code.

---

## Batch Module

Import from `fuzzyrust.batch`:

```python
from fuzzyrust import batch
```

### similarity

```python
batch.similarity(
    strings: list[str],
    query: str,
    algorithm: str = "jaro_winkler"
) -> list[MatchResult]
```

Compute similarity scores for multiple strings against a query.

**Returns:** List of `MatchResult(text, score, id)` in the same order as input strings.

**Example:**

```python
results = batch.similarity(["John", "Jon", "Jane"], "John", algorithm="jaro_winkler")
# [MatchResult(text='John', score=1.0, id=0), MatchResult(text='Jon', score=0.93, id=1), ...]
```

---

### best_matches

```python
batch.best_matches(
    choices: list[str],
    query: str,
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.0,
    limit: int | None = None
) -> list[MatchResult]
```

Find best matches for a query from a list of choices.

**Returns:** List of `MatchResult(text, score)`

**Example:**

```python
batch.best_matches(["apple", "apply", "banana"], "aple", limit=2)
# [MatchResult(text='apple', score=0.91), MatchResult(text='apply', score=0.80)]
```

---

### deduplicate

```python
batch.deduplicate(
    strings: list[str],
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.8,
    normalize: str | None = None
) -> DeduplicationResult
```

Group similar strings together.

**Returns:** `DeduplicationResult` with:
- `groups`: List of duplicate groups (each group is a list of strings)
- `unique`: List of strings that have no duplicates
- `total_duplicates`: Total count of duplicate strings found

**Example:**

```python
result = batch.deduplicate(["John Smith", "Jon Smyth", "Jane Doe"], min_similarity=0.8)
result.groups  # [["John Smith", "Jon Smyth"]]
result.unique  # ["Jane Doe"]
```

---

### pairwise

```python
batch.pairwise(
    left: list[str],
    right: list[str],
    algorithm: str = "jaro_winkler"
) -> list[float]
```

Compute element-wise similarity between two lists of the same length.

**Example:**

```python
batch.pairwise(["John", "Jane"], ["Jon", "Janet"])
# [0.93, 0.89]
```

---

### similarity_matrix

```python
batch.similarity_matrix(
    queries: list[str],
    choices: list[str],
    algorithm: str = "levenshtein"
) -> list[list[float]]
```

Compute all pairwise similarities between queries and choices.

**Returns:** 2D list where `result[i][j]` is similarity between `queries[i]` and `choices[j]`.

**Example:**

```python
batch.similarity_matrix(["John", "Jane"], ["Jon", "Janet", "Bob"])
# [[0.93, 0.78, 0.0], [0.78, 0.89, 0.0]]
```

---

## Old to New Batch API Mapping

| Old Function | New Function |
|--------------|--------------|
| `fr.batch_jaro_winkler()` | `batch.similarity(algorithm="jaro_winkler")` |
| `fr.batch_levenshtein()` | `batch.similarity(algorithm="levenshtein")` |
| `fr.find_best_matches()` | `batch.best_matches()` |
| `fr.find_duplicates()` | `batch.deduplicate()` |
| `fr.batch_similarity_pairs()` | `batch.pairwise()` |
| `fr.cdist()` | `batch.similarity_matrix()` |
| `batch.distance_matrix()` | `batch.similarity_matrix()` |

---

## Case-Insensitive Comparison

For case-insensitive comparison, use the `normalize` parameter:

```python
# Case-insensitive Levenshtein distance
fr.levenshtein("Hello", "HELLO", normalize="lowercase")  # Returns 0

# Case-insensitive Jaro-Winkler similarity
fr.jaro_winkler_similarity("John", "JOHN", normalize="lowercase")  # Returns 1.0
```

Available normalization modes:
- `"lowercase"`: Convert to lowercase before comparison
- `"unicode_nfkd"`: Apply Unicode NFKD normalization
- `"remove_punctuation"`: Remove ASCII punctuation
- `"remove_whitespace"`: Remove all whitespace
- `"strict"`: Apply all normalizations
