# Performance

FuzzyRust is designed for high performance at scale.

## Benchmarks

### Single Pair Comparison

| Library | Jaro-Winkler | Levenshtein |
|---------|--------------|-------------|
| FuzzyRust | 1.0x | 1.0x |
| RapidFuzz | 0.9x | 0.8x |
| jellyfish | 2.5x | 3.0x |
| python-Levenshtein | - | 2.0x |

*Lower is better. RapidFuzz is slightly faster for single pairs.*

### Batch Operations (10K pairs)

| Library | Time |
|---------|------|
| FuzzyRust | 1.0x |
| RapidFuzz | 5-10x |
| Pure Python | 100x+ |

*FuzzyRust excels at batch operations due to Rayon parallelization.*

### Index Search (1M strings, find top 10)

| Method | Time |
|--------|------|
| NgramIndex | ~1ms |
| BkTree | ~5ms |
| Linear scan | ~500ms |

*Index structures provide 100-2000x speedup.*

## Optimization Techniques

FuzzyRust uses several optimization techniques:

### 1. Rust Core

All algorithms are implemented in Rust with zero-copy string handling.

### 2. Parallel Processing

Batch operations use Rayon for automatic parallelization:

```python
# Automatically parallelized across all cores
results = index.batch_search(queries, min_similarity=0.8)
```

### 3. SIMD Acceleration

Myers bit-parallel algorithm for Levenshtein distance processes 64 characters in parallel.

### 4. Early Termination

Bounded distance calculations stop early when threshold is exceeded:

```python
# Returns None immediately if distance > 3
fr.levenshtein_bounded("hello", "world", max_distance=3)
```

### 5. N-gram Candidate Filtering

Index structures filter candidates before expensive similarity calculations.

### 6. Memory-Efficient Indices

Posting list compression reduces memory usage by 50-70%:

```python
index.compress()  # Compress after building
```

## Scaling Guidelines

| Dataset Size | Recommended Approach |
|--------------|---------------------|
| < 1K | Direct comparison |
| 1K - 10K | `find_best_matches()` |
| 10K - 100K | `NgramIndex` |
| 100K - 1M | `NgramIndex` + compression |
| > 1M | Batch API + SNM deduplication |

## Memory Usage

Approximate memory per 1M strings (avg 20 chars):

| Structure | Memory |
|-----------|--------|
| Raw strings | ~20 MB |
| NgramIndex (uncompressed) | ~100 MB |
| NgramIndex (compressed) | ~40 MB |
| BkTree | ~80 MB |

## Tips for Best Performance

### 1. Use Appropriate Algorithms

```python
# Fast for names
fr.jaro_winkler(name1, name2)

# Fast for short codes
fr.hamming(code1, code2)

# Use bounded for filtering
fr.levenshtein_bounded(s1, s2, max_distance=2)
```

### 2. Pre-filter When Possible

```python
# Filter before fuzzy matching
candidates = [s for s in strings if s[0] == query[0]]
matches = fr.find_best_matches(query, candidates)
```

### 3. Use Batch Operations

```python
# Bad: Loop with single calls
for q in queries:
    results.append(fr.find_best_matches(q, targets))

# Good: Single batch call
all_results = index.batch_search(queries, min_similarity=0.8)
```

### 4. Choose Right Index

```python
# For similarity threshold queries
index = fr.NgramIndex(ngram_size=3)

# For edit distance queries
tree = fr.BkTree(algorithm="levenshtein")
```

### 5. Compress Large Indices

```python
index = fr.NgramIndex(ngram_size=3)
index.add_all(large_dataset)
index.compress()  # Reduce memory by 50-70%
```

## Profiling

Enable debug output for performance analysis:

```python
import logging
logging.getLogger("fuzzyrust").setLevel(logging.DEBUG)
```
