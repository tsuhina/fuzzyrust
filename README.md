# FuzzyRust

[![PyPI](https://img.shields.io/pypi/v/fuzzyrust.svg)](https://pypi.org/project/fuzzyrust/)
[![CI](https://github.com/tsuhina/fuzzyrust/actions/workflows/ci.yml/badge.svg)](https://github.com/tsuhina/fuzzyrust/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

**Rust-powered fuzzy matching for Polars DataFrames.**

## Installation

```bash
pip install fuzzyrust
```

## 30-Second Demo

You have messy order data with typos. You need to match it to your clean customer database:

```python
import polars as pl
import fuzzyrust as fr

# Your clean customer database
customers = pl.DataFrame({
    "id": [1, 2, 3, 4],
    "name": ["Apple Inc.", "Microsoft Corporation", "Google LLC", "Amazon.com Inc."]
})

# Incoming orders with typos and variations
orders = pl.DataFrame({
    "order_id": ["A1", "A2", "A3", "A4"],
    "company": ["Appel", "Microsft Corp", "Googel", "Amzon Inc"],
    "amount": [5000, 3000, 7000, 2000]
})

# One line to match them
matched = fr.fuzzy_join(orders, customers, left_on="company", right_on="name", min_similarity=0.5)

print(matched)
# shape: (4, 6)
# +---------+---------------+--------+----+-----------------------+-------------+
# |order_id | company       | amount | id | name                  | fuzzy_score |
# +---------+---------------+--------+----+-----------------------+-------------+
# | A1      | Appel         | 5000   | 1  | Apple Inc.            | 0.84        |
# | A2      | Microsft Corp | 3000   | 2  | Microsoft Corporation | 0.89        |
# | A3      | Googel        | 7000   | 3  | Google LLC            | 0.89        |
# | A4      | Amzon Inc     | 2000   | 4  | Amazon.com Inc.       | 0.83        |
# +---------+---------------+--------+----+-----------------------+-------------+
```

## DataFrame Operations

### fuzzy_join()

Match records across DataFrames despite typos, abbreviations, and variations:

```python
result = fr.fuzzy_join(
    left_df, right_df,
    left_on="company",
    right_on="name",
    algorithm="jaro_winkler",  # or "levenshtein", "ngram"
    min_similarity=0.7
)
```

Multi-column join with per-column algorithms:

```python
result = fr.fuzzy_join(
    left, right,
    on=[
        ("name", "customer", {"algorithm": "jaro_winkler", "weight": 2.0}),
        ("city", "location", {"algorithm": "levenshtein", "weight": 1.0}),
    ],
    min_similarity=0.5
)
```

### fuzzy_dedupe_rows()

Find and remove duplicate records using multi-field matching:

```python
customers = pl.DataFrame({
    "name": ["John Smith", "Jon Smyth", "Jane Doe", "John Smith Jr"],
    "email": ["john@test.com", "john@test.com", "jane@test.com", "john.jr@test.com"],
    "phone": ["555-1234", "555-1234", "555-9999", "555-1234"],
})

result = fr.fuzzy_dedupe_rows(
    customers,
    columns=["name", "email", "phone"],
    algorithms={"name": "jaro_winkler", "email": "levenshtein", "phone": "exact_match"},
    weights={"name": 2.0, "email": 1.5, "phone": 1.0},
    min_similarity=0.5,
    keep="first"  # or "last", "most_complete"
)

# Get only unique rows (canonical = the one to keep from each duplicate group)
unique = result.filter(pl.col("_is_canonical"))
```

For exploratory pair-finding (e.g., manual review before merging), use `match_dataframe()` instead.

### .fuzzy Expression Namespace

Use fuzzy matching directly in Polars expressions:

```python
df = pl.DataFrame({"name": ["John", "Jon", "Jane", "Bob"]})

# Calculate similarity scores
df.with_columns(
    score=pl.col("name").fuzzy.similarity("John", algorithm="jaro_winkler")
)

# Filter by similarity
df.filter(pl.col("name").fuzzy.is_similar("John", threshold=0.8))

# Find best match from a list
categories = ["Electronics", "Clothing", "Food"]
df.with_columns(
    category=pl.col("query").fuzzy.best_match(categories, min_score=0.6)
)

# Edit distance and phonetic encoding
df.with_columns(
    dist=pl.col("name").fuzzy.distance("John"),
    soundex=pl.col("name").fuzzy.phonetic("soundex")
)
```

### FuzzyIndex for Batch Operations

Build reusable indices for repeated searches:

```python
# Build index from a Series
targets = pl.Series(["Apple Inc", "Microsoft Corp", "Google LLC"])
index = fr.FuzzyIndex.from_series(targets, algorithm="ngram")

# Batch search
queries = pl.Series(["Apple", "Microsft"])
results = index.search_series(queries, min_score=0.6)

# Save/load for reuse
index.save("company_index.pkl")
index = fr.FuzzyIndex.load("company_index.pkl")
```

## Search at Scale

For searching millions of records, use the indexing structures:

```python
import fuzzyrust as fr

# BkTree: Metric space indexing for edit distance
tree = fr.BkTree()
tree.add_all(["hello", "hallo", "hullo", "world"])
results = tree.search("helo", max_distance=2)

# NgramIndex: Fast candidate filtering + similarity scoring
index = fr.NgramIndex(ngram_size=2)
index.add_all(products)
results = index.search("PRDUCT-XYZ", algorithm="jaro_winkler", min_similarity=0.7)

# HybridIndex: Best of both for large datasets (1M+ records)
index = fr.HybridIndex(ngram_size=3)
index.add_all(millions_of_records)
results = index.search(query, min_similarity=0.8, limit=10)
results = index.batch_search(queries, limit=5)  # Parallel batch search
```

## Basic Similarity Functions

All algorithms available as standalone functions:

```python
import fuzzyrust as fr

fr.jaro_winkler_similarity("Robert", "Rupert")  # 0.84
fr.levenshtein("kitten", "sitting")             # 3
fr.damerau_levenshtein("ca", "ac")              # 1
fr.soundex_match("Robert", "Rupert")            # True

# Case-insensitive variants (all functions have _ci suffix)
fr.levenshtein_ci("Hello", "HELLO")             # 0
```

## Algorithms Available

**Edit Distance**: levenshtein, damerau_levenshtein, hamming

**Similarity Scores**: jaro, jaro_winkler, ngram, lcs, cosine

**Phonetic**: soundex, metaphone

See [examples/quickstart.py](examples/quickstart.py) for comprehensive examples.

## Performance

FuzzyRust is written in Rust with PyO3 bindings, delivering 10-100x speedups over pure Python implementations. All batch operations use Rayon for automatic parallelization across CPU cores. The indexing structures (BkTree, NgramIndex, HybridIndex) enable sub-linear search times even with millions of records.

## Learn More

- [examples/quickstart.py](examples/quickstart.py) - Comprehensive tutorial with all features
- [API Documentation](https://fuzzyrust.readthedocs.io/) - Full API reference

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

BSD-3-Clause License. See [LICENSE](LICENSE) for details.
