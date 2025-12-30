# Core Functions

## Similarity Functions

All similarity functions return a score between 0.0 (no match) and 1.0 (exact match).

### jaro_winkler

```python
jaro_winkler(s1: str, s2: str, prefix_weight: float = 0.1) -> float
```

Jaro-Winkler similarity. Best for short strings like names.

**Parameters:**

- `s1`, `s2`: Strings to compare
- `prefix_weight`: Weight for matching prefix (0.0-0.25, default 0.1)

**Example:**

```python
fr.jaro_winkler("MARTHA", "MARHTA")  # 0.961
```

---

### jaro

```python
jaro(s1: str, s2: str) -> float
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

### cosine_similarity

```python
cosine_similarity(s1: str, s2: str) -> float
```

Cosine similarity of word vectors.

---

### cosine_ngrams

```python
cosine_ngrams(s1: str, s2: str, ngram_size: int = 2) -> float
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

## Batch Functions

### find_best_matches

```python
find_best_matches(
    query: str,
    choices: list[str],
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.0,
    limit: int | None = None
) -> list[MatchResult]
```

Find best matches for a query from a list of choices.

**Returns:** List of `MatchResult(text, score)`

---

### batch_similarity

```python
batch_similarity(
    queries: list[str],
    targets: list[str],
    algorithm: str = "jaro_winkler"
) -> list[list[float]]
```

Compute all pairwise similarities between queries and targets.

---

### compare_algorithms

```python
compare_algorithms(
    s1: str,
    s2: str,
    algorithms: list[str] | None = None
) -> dict[str, float]
```

Compare multiple algorithms on a string pair.

**Example:**

```python
fr.compare_algorithms("hello", "hallo")
# {'levenshtein': 0.8, 'jaro_winkler': 0.88, ...}
```

---

## Case-Insensitive Variants

All functions have `_ci` suffix variants:

- `jaro_winkler_ci`
- `jaro_ci`
- `levenshtein_ci`
- `levenshtein_similarity_ci`
- `damerau_levenshtein_ci`
- `ngram_similarity_ci`
- `hamming_ci`
- `hamming_similarity_ci`
- `lcs_length_ci`
- `lcs_string_ci`
