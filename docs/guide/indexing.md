# Index Structures

For large datasets, index structures provide 100-2000x speedup over linear scanning.

## Overview

| Index | Best For | Build Time | Search Time |
|-------|----------|------------|-------------|
| NgramIndex | General fuzzy search | O(n) | O(candidates) |
| BkTree | Edit distance queries | O(n log n) | O(log n) |
| HybridIndex | Combined approach | O(n) | O(candidates) |

## NgramIndex

Pre-indexes strings by their n-grams for fast candidate filtering.

```python
import fuzzyrust as fr

# Create index with trigrams
index = fr.NgramIndex(ngram_size=3)

# Add strings
index.add("John Smith")
index.add("Jane Doe")
index.add_all(["Bob Wilson", "Alice Brown", "Charlie Davis"])

# Search
results = index.search(
    "Jon Smith",
    algorithm="jaro_winkler",
    min_similarity=0.8,
    limit=5
)

for r in results:
    print(f"{r.text}: {r.score:.3f}")
```

### With Normalization

```python
# Case-insensitive matching
index = fr.NgramIndex(ngram_size=3, normalize=True)
index.add_all(["JOHN SMITH", "jane doe"])

# Finds matches regardless of case
results = index.search("John Smith", min_similarity=0.8)
```

### Compression

For large indices, compress to reduce memory:

```python
index = fr.NgramIndex(ngram_size=3)
index.add_all(large_dataset)  # 1M strings

# Compress posting lists (50-70% memory reduction)
index.compress()

# Searches still work
results = index.search("query", min_similarity=0.8)

# Decompress to add more items
index.decompress()
index.add("new item")
```

## BkTree

Metric space index optimized for edit distance queries.

```python
# Create BK-tree with Levenshtein distance
tree = fr.BkTree(algorithm="levenshtein")

# Add strings
tree.add_all(["apple", "apply", "maple", "orange", "banana"])

# Recommended: Use similarity threshold (consistent with other APIs)
results = tree.search_similarity("aple", min_similarity=0.7)

# Alternative: Use distance threshold (if you need exact edit distance control)
results = tree.search("aple", max_distance=2)
# Returns: apple (1), apply (2), maple (2)
```

### Distance vs Similarity

BkTree supports two search methods:

| Method | Threshold | Use When |
|--------|-----------|----------|
| `search_similarity()` | `min_similarity` (0.0-1.0) | **Recommended** - consistent with other FuzzyRust APIs |
| `search()` | `max_distance` (integer) | When you need exact edit distance control |

**Conversion formula:**
```
similarity ≈ 1 - (distance / max(len(s1), len(s2)))
```

**Example:**
```python
# These are roughly equivalent for strings of length 5:
tree.search("hello", max_distance=1)      # distance <= 1
tree.search_similarity("hello", min_similarity=0.8)  # similarity >= 80%
```

!!! tip "Prefer `search_similarity()`"
    Use `search_similarity()` for consistency with NgramIndex and HybridIndex.
    All three indices then use the same `min_similarity` parameter.

### Parallel Search

For large trees (>10K items), parallel search is used automatically:

```python
tree = fr.BkTree()
tree.add_all(million_strings)

# Automatically uses parallel search
results = tree.search_similarity("query", min_similarity=0.8)
```

## HybridIndex

Combines N-gram candidate filtering with accurate similarity scoring.

```python
index = fr.HybridIndex(ngram_size=3)
index.add_all(dataset)

# Uses n-grams for fast candidate retrieval,
# then scores with specified algorithm
results = index.search(
    "query",
    algorithm="jaro_winkler",
    min_similarity=0.8
)
```

## Batch Operations

All indices support batch search:

```python
queries = ["John Smith", "Jane Doe", "Bob Wilson"]

# Batch search (parallel processing)
all_results = index.batch_search(
    queries,
    algorithm="jaro_winkler",
    min_similarity=0.8,
    limit=5
)

for query, results in zip(queries, all_results):
    print(f"Matches for {query}: {len(results)}")
```

## Persistence

Save and load indices:

```python
# Save
index.save("index.bin")

# Load
index = fr.NgramIndex.load("index.bin")
```

## Choosing an Index

| Dataset Size | Recommendation |
|--------------|----------------|
| < 1,000 | Linear scan (no index needed) |
| 1K - 100K | NgramIndex or HybridIndex |
| 100K - 1M | NgramIndex with compression |
| > 1M | NgramIndex + batch processing |

| Query Type | Recommendation |
|------------|----------------|
| Edit distance threshold | BkTree |
| Similarity threshold | NgramIndex |
| Mixed requirements | HybridIndex |
