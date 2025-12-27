# FuzzyRust 🔍

[![PyPI](https://img.shields.io/pypi/v/fuzzyrust.svg)](https://pypi.org/project/fuzzyrust/)
[![CI](https://github.com/tsuhina/fuzzyrust/actions/workflows/ci.yml/badge.svg)](https://github.com/tsuhina/fuzzyrust/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/tsuhina/d34009f65419835921756ae6457f91fa/raw/coverage.json)](https://github.com/tsuhina/fuzzyrust/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

**High-performance string similarity library for Python, written in Rust.**

FuzzyRust is designed for searching through messy data full of typos, manufacturer part codes, and string variations. It provides multiple similarity algorithms and efficient indexing structures for scalable fuzzy search.

## Features

- ⚡ **Blazing Fast**: Core algorithms written in Rust with parallel processing support
- 🎯 **Multiple Algorithms**: Levenshtein, Damerau-Levenshtein, Jaro-Winkler, Soundex, Metaphone, N-grams, and more
- 📊 **Efficient Indexing**: BK-tree and N-gram indices for fast fuzzy search at scale
- 🔄 **Batch Processing**: Parallel search across millions of records
- 🌍 **Unicode Support**: Full Unicode character handling
- 📝 **Type Hints**: Complete type annotations for IDE support
- 🧩 **Extensible**: Modular design for easy customization

## Installation

```bash
pip install fuzzyrust
```

Or build from source:

```bash
pip install maturin
maturin develop --release
```

## Quick Start

### Basic Similarity

```python
import fuzzyrust as fr

# Jaro-Winkler similarity (great for names)
fr.jaro_winkler_similarity("Robert", "Rupert")  # 0.84

# Levenshtein distance (edit distance)
fr.levenshtein("kitten", "sitting")  # 3

# Damerau-Levenshtein (handles transpositions)
fr.damerau_levenshtein("ca", "ac")  # 1 (just one swap)

# Phonetic matching
fr.soundex_match("Robert", "Rupert")  # True
fr.metaphone_match("phone", "fone")   # True
```

### Finding Best Matches

```python
# Search through a list for best matches
parts = ["ABC-123", "ABC-124", "XYZ-999", "ABC-12", "ABD-123"]
matches = fr.find_best_matches(parts, "ABC-123", algorithm="jaro_winkler", limit=3)

# Returns MatchResult objects with text and score attributes
for match in matches:
    print(f"{match.text}: {match.score:.2f}")
# ABC-123: 1.00
# ABC-124: 0.95
# ABC-12: 0.93
```

### Using Indices for Large Datasets

For searching through large datasets, use the indexing structures:

```python
# BK-tree: Great for edit distance queries
tree = fr.BkTree()
tree.add_all(["hello", "hallo", "hullo", "world", "help"])
results = tree.search("helo", max_distance=2)

# Returns SearchResult objects with id, text, score, distance
for r in results:
    print(f"{r.text}: distance={r.distance}, score={r.score:.2f}")
# hello: distance=1, score=0.80
# hallo: distance=2, score=0.60

# N-gram Index: Fast candidate filtering + similarity scoring
products = ["PRODUCT-ABC", "PRODUCT-XYZ", "ITEM-123"]  # Your product list
index = fr.NgramIndex(ngram_size=2)  # Use bigrams
index.add_all(products)
results = index.search(
    "PRDUCT-XYZ",
    algorithm="jaro_winkler",
    min_similarity=0.7,
    limit=10
)

# Returns SearchResult objects
for r in results:
    print(f"{r.text}: {r.score:.2f}")
```

### Batch Processing

Process millions of comparisons in parallel:

```python
# Compare many strings against a query - returns MatchResult objects
items = ["apple", "application", "apply", "banana"]
matches = fr.batch_jaro_winkler(items, "appel")

for match in matches:
    print(f"{match.text}: {match.score:.2f}")
# apple: 0.87
# application: 0.71
# ...

# Batch search with an index
results = index.batch_search(
    queries=user_queries,
    algorithm="jaro_winkler",
    min_similarity=0.8
)
# Returns List[List[SearchResult]] - one list per query
```

### Case-Insensitive Matching

All similarity functions have case-insensitive variants with the `_ci` suffix:

```python
# Regular functions are case-sensitive
fr.levenshtein("Hello", "hello")  # 1

# Case-insensitive variants ignore case
fr.levenshtein_ci("Hello", "HELLO")  # 0

# Works with all algorithms
fr.jaro_winkler_similarity_ci("Product-ABC", "PRODUCT-ABC")  # 1.0
fr.ngram_similarity_ci("Test", "TEST", ngram_size=2)  # 1.0
fr.damerau_levenshtein_ci("ab", "BA")  # 1 (transposition)
```

### Deduplication

Find duplicate entries in your data with a single function call:

```python
items = ["iPhone 15", "iphone 15", "IPHONE 15", "Samsung Galaxy", "iPhone 14"]

result = fr.find_duplicates(
    items,
    algorithm="jaro_winkler",
    threshold=0.85,
    normalize=True  # Handles case and whitespace
)

print(f"Duplicate groups: {result.groups}")
# [["iPhone 15", "iphone 15", "IPHONE 15"]]

print(f"Unique items: {result.unique}")
# ["Samsung Galaxy", "iPhone 14"]

print(f"Total duplicates found: {result.total_duplicates}")
# 2
```

### Multi-Algorithm Comparison

Compare the same strings using different algorithms to find the best one for your use case:

```python
strings = ["hello", "hallo", "help", "world"]
query = "helo"

comparisons = fr.compare_algorithms(
    strings,
    query,
    algorithms=["levenshtein", "jaro_winkler", "ngram"],  # Optional: specify algorithms
    limit=3  # Top 3 matches per algorithm
)

for comp in comparisons:
    print(f"\n{comp.algorithm}: overall score {comp.score:.3f}")
    for match in comp.matches:
        print(f"  {match.text}: {match.score:.3f}")

# Output:
# jaro_winkler: overall score 0.917
#   hello: 0.917
#   hallo: 0.867
#   help: 0.783
```

## Algorithms

### Edit Distance Family

| Function | Description | Best For |
|----------|-------------|----------|
| `levenshtein` | Classic edit distance | General typos |
| `damerau_levenshtein` | Includes transpositions | Keyboard typos |
| `hamming` | Positional differences | Fixed-length codes |

### Similarity Scores

| Function | Description | Best For |
|----------|-------------|----------|
| `jaro_similarity` | Jaro algorithm | Short strings |
| `jaro_winkler_similarity` | Prefix-weighted Jaro | Names, codes |
| `ngram_similarity` | N-gram overlap | Partial matches |
| `lcs_similarity` | Longest common subsequence | Rearrangements |
| `cosine_similarity_*` | Vector space model | Document similarity |

### Phonetic

| Function | Description | Best For |
|----------|-------------|----------|
| `soundex` / `soundex_match` | Classic phonetic | English names |
| `metaphone` / `metaphone_match` | Improved phonetic | More accurate |

## Indexing Structures

### BkTree

Efficient fuzzy search using metric space properties:

```python
tree = fr.BkTree(use_damerau=False)  # or True for transpositions
tree.add_all(strings)
tree.search(query, max_distance=2)
tree.find_nearest(query, k=5)
```

### NgramIndex

Fast candidate filtering with similarity scoring:

```python
index = fr.NgramIndex(ngram_size=2, min_similarity=0.5)
index.add_with_data("ABC-123", 42)  # Store with user data

# Search returns SearchResult objects
results = index.search(query, algorithm="jaro_winkler")
for r in results:
    print(f"{r.text}: {r.score:.2f} (data: {r.data})")

# Find k-nearest neighbors
nearest = index.find_nearest(query, k=5)

# Check if exact match exists
if index.contains("ABC-123"):
    print("Found exact match!")
```

### HybridIndex

Best of both worlds - combines n-gram filtering with BK-tree precision:

```python
index = fr.HybridIndex(ngram_size=3)
index.add_all(millions_of_records)

# Search with similarity threshold
results = index.search(query, min_similarity=0.8, limit=10)

# Batch search multiple queries
batch_results = index.batch_search(
    ["query1", "query2", "query3"],
    limit=5
)

# Find k-nearest neighbors
nearest = index.find_nearest(query, k=10)

# Check for exact matches
if index.contains("exact-match"):
    print("Found!")
```

## Performance Tips

1. **Choose the right algorithm**:
   - `jaro_winkler`: Best for names and short codes
   - `levenshtein`: Best for general text with typos
   - `ngram`: Best for partial matching

2. **Use indices for large datasets**:
   - < 1,000 items: Direct comparison is fine
   - < 100,000 items: Use `NgramIndex`
   - < 1,000,000+ items: Use `HybridIndex`

3. **Set appropriate thresholds**:
   ```python
   # Pre-filter with min_similarity
   index.search(query, min_similarity=0.7, limit=10)
   ```

4. **Use batch operations**:
   ```python
   # Process many queries in parallel
   index.batch_search(queries, ...)
   ```

## Example: Product Search

```python
import fuzzyrust as fr

# Index your product catalog
products = [
    "iPhone 15 Pro Max 256GB",
    "iPhone 15 Pro 128GB",
    "Samsung Galaxy S24 Ultra",
    "Google Pixel 8 Pro",
    # ... millions more
]

index = fr.HybridIndex(ngram_size=3)
for i, product in enumerate(products):
    index.add_with_data(product, i)

# Search with typos (use case-insensitive search for better results)
results = index.search(
    "iphone 15 pro max",  # User query with different case
    algorithm="jaro_winkler",
    min_similarity=0.7,
    limit=5
)

# Results are SearchResult objects
for r in results:
    print(f"{r.score:.2f}: {r.text} (product_id: {r.data})")
# 0.95: iPhone 15 Pro Max 256GB (product_id: 0)
# 0.89: iPhone 15 Pro 128GB (product_id: 1)
```

## Example: Deduplication

```python
import fuzzyrust as fr

# Customer data with typos and variations
customer_names = [
    "John Smith",
    "Jon Smith",
    "JOHN SMITH",
    "Jane Doe",
    "Jane M. Doe",
    "Bob Johnson",
    "Robert Johnson",
]

# Use the built-in deduplication helper
result = fr.find_duplicates(
    customer_names,
    algorithm="jaro_winkler",
    threshold=0.85,
    normalize=True  # Handles case differences and whitespace
)

# Review duplicate groups
for i, group in enumerate(result.groups):
    print(f"\nDuplicate group {i + 1}:")
    for name in group:
        print(f"  - {name}")

# Output:
# Duplicate group 1:
#   - John Smith
#   - Jon Smith
#   - JOHN SMITH
#
# Duplicate group 2:
#   - Jane Doe
#   - Jane M. Doe

print(f"\nUnique records: {result.unique}")
# ['Bob Johnson', 'Robert Johnson']

print(f"Total duplicates removed: {result.total_duplicates}")
# 3
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Dual-licensed under MIT or Apache-2.0 at your option.
