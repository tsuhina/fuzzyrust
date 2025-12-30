# Similarity Algorithms

FuzzyRust provides a comprehensive set of string similarity algorithms.

## Algorithm Overview

| Algorithm | Type | Best For | Range |
|-----------|------|----------|-------|
| Levenshtein | Edit distance | Typos, OCR errors | 0 to max(len) |
| Damerau-Levenshtein | Edit distance | Typos with transpositions | 0 to max(len) |
| Jaro-Winkler | Token-based | Names, short strings | 0.0 to 1.0 |
| Jaro | Token-based | Names, short strings | 0.0 to 1.0 |
| N-gram | Set-based | General text | 0.0 to 1.0 |
| Cosine | Vector-based | Documents, longer text | 0.0 to 1.0 |
| Hamming | Position-based | Fixed-length codes | 0 to len |
| LCS | Subsequence | DNA, diff tools | 0 to min(len) |
| Soundex | Phonetic | Name matching | Code comparison |
| Metaphone | Phonetic | Name matching | Code comparison |

## Edit Distance Algorithms

### Levenshtein

Counts minimum edits (insert, delete, substitute) to transform one string to another.

```python
import fuzzyrust as fr

# Distance (lower = more similar)
fr.levenshtein("kitten", "sitting")  # 3

# Similarity (higher = more similar)
fr.levenshtein_similarity("kitten", "sitting")  # 0.57

# With max distance threshold (faster for filtering)
fr.levenshtein_bounded("hello", "world", max_distance=2)  # None (exceeds threshold)
```

### Damerau-Levenshtein

Like Levenshtein but also counts transpositions as single edits.

```python
# "ab" -> "ba" is 1 edit (transposition), not 2
fr.damerau_levenshtein("ab", "ba")  # 1
fr.levenshtein("ab", "ba")  # 2
```

## Token-Based Algorithms

### Jaro-Winkler

Optimized for short strings like names. Gives higher scores to strings with matching prefixes.

```python
fr.jaro_winkler("MARTHA", "MARHTA")  # 0.961
fr.jaro("MARTHA", "MARHTA")  # 0.944

# Adjust prefix weight (default 0.1)
fr.jaro_winkler("MARTHA", "MARHTA", prefix_weight=0.2)
```

## Set-Based Algorithms

### N-gram Similarity

Compares strings by their overlapping n-grams (character sequences).

```python
# Default uses bigrams (n=2)
fr.ngram_similarity("hello", "hallo")  # 0.5

# Trigrams
fr.ngram_similarity("hello", "hallo", ngram_size=3)  # 0.33

# Dice coefficient (default) vs Jaccard
fr.dice_coefficient("night", "nacht")
fr.jaccard_similarity("night", "nacht")
```

### Cosine Similarity

Vector-space model comparing term frequencies.

```python
fr.cosine_similarity("hello world", "world hello")  # 1.0 (same terms)
fr.cosine_ngrams("hello", "hallo", ngram_size=2)  # Character n-gram based
```

## Phonetic Algorithms

Match strings that sound similar.

### Soundex

```python
fr.soundex("Robert")  # "R163"
fr.soundex("Rupert")  # "R163"

# Compare phonetically
fr.soundex_similarity("Robert", "Rupert")  # 1.0
```

### Metaphone

More accurate than Soundex for English.

```python
fr.metaphone("knight")  # "NT"
fr.metaphone("night")   # "NT"

fr.metaphone_similarity("knight", "night")  # 1.0
```

## Case-Insensitive Variants

All algorithms have `_ci` suffix variants for case-insensitive comparison:

```python
fr.jaro_winkler("Hello", "HELLO")     # 0.0
fr.jaro_winkler_ci("Hello", "HELLO")  # 1.0

fr.levenshtein("ABC", "abc")     # 3
fr.levenshtein_ci("ABC", "abc")  # 0
```

## Choosing an Algorithm

| Use Case | Recommended Algorithm |
|----------|----------------------|
| Names | Jaro-Winkler, Soundex |
| Typo correction | Levenshtein, Damerau-Levenshtein |
| Product matching | N-gram, Cosine |
| Address matching | Jaro-Winkler + normalization |
| DNA sequences | LCS, Hamming |
| Document similarity | Cosine |
