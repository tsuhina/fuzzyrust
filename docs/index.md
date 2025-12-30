# FuzzyRust

High-performance string similarity library for Python, written in Rust.

[![PyPI](https://img.shields.io/pypi/v/fuzzyrust)](https://pypi.org/project/fuzzyrust/)
[![Python](https://img.shields.io/pypi/pyversions/fuzzyrust)](https://pypi.org/project/fuzzyrust/)
[![License](https://img.shields.io/github/license/tsuhina/fuzzyrust)](https://github.com/tsuhina/fuzzyrust/blob/main/LICENSE)

## Features

- **Fast**: Rust-powered algorithms with parallel processing via Rayon
- **Comprehensive**: Levenshtein, Jaro-Winkler, N-gram, Cosine, Phonetic, and more
- **Scalable**: Index structures (BK-tree, N-gram index) for efficient large-scale matching
- **Polars Integration**: Native DataFrame operations for fuzzy joins and deduplication
- **Multi-field Matching**: Schema-based matching with weighted field scoring

## Quick Example

```python
import fuzzyrust as fr

# Simple similarity
fr.jaro_winkler("hello", "hallo")  # 0.88

# Find best matches
fr.find_best_matches("apple", ["apply", "maple", "orange"], limit=2)

# Fuzzy join with Polars
import polars as pl
from fuzzyrust import fuzzy_join

df1 = pl.DataFrame({"name": ["John Smith", "Jane Doe"]})
df2 = pl.DataFrame({"customer": ["Jon Smith", "Janet Doe"]})
result = fuzzy_join(df1, df2, left_on="name", right_on="customer", threshold=0.8)
```

## Installation

```bash
pip install fuzzyrust
```

## Performance

FuzzyRust is designed for performance:

| Operation | vs RapidFuzz |
|-----------|--------------|
| Single pair | Competitive (~1x) |
| Batch (10K) | 5-10x faster |
| Index search | 100-2000x faster |

## Next Steps

- [Installation](getting-started/installation.md) - Detailed installation instructions
- [Quickstart](getting-started/quickstart.md) - Get started in 5 minutes
- [API Reference](api/core.md) - Complete function documentation
