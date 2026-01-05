# FuzzyRust Roadmap: Record Linkage Features

> **Versions:** 0.4.0 - 0.5.0 | **Timeline:** ~24 days total

## Executive Summary

This roadmap positions FuzzyRust as a comprehensive Python library for data cleaning, deduplication, and record linkage. The features are organized into three releases:

| Version | Theme | Key Deliverables | Effort |
|---------|-------|------------------|--------|
| **v0.4.0** | Core Record Linkage | Blocking strategies, evaluation tools, uncertain pair detection | ~8 days |
| **v0.4.1** | ML Integration | Comparison vectors, preprocessing pipelines | ~6 days |
| **v0.5.0** | Probabilistic Models | Naive Bayes linkage classifier | ~10 days |

### Goals

1. Enable large-scale deduplication (1M+ records) via blocking strategies
2. Provide evaluation tools for tuning and validating linkage quality
3. Support human-in-the-loop workflows with uncertain pair detection
4. Generate ML-ready feature vectors for custom classifiers
5. Offer composable preprocessing pipelines

---

## v0.4.0: Core Record Linkage

### Feature 1: Blocking Strategies

**Purpose:** Reduce O(N²) comparisons to O(N × block_size) for large-scale deduplication.

**Effort:** 4-5 days

#### Design

Use an enum-based approach (avoids breaking `Clone, Copy` on `DedupMethod`):

- `FirstN` - First N characters (optionally case-insensitive)
- `Phonetic` - Soundex or Metaphone codes
- `Qgram` - Q-gram signatures with overlap threshold
- `Canopy` - Two-threshold clustering
- `Compound` - Union of multiple blockers

#### API

```python
from fuzzyrust import blocking

# Factory functions
first_n = blocking.first_n(n=3, case_insensitive=True)
soundex = blocking.phonetic("soundex")
compound = blocking.compound([first_n, soundex])

# Extend existing df_dedupe_snm
result = frp.df_dedupe_snm(
    df, columns=["name"],
    blocking=soundex,
    blocking_columns=["name"]
)
```

---

### Feature 2: Record Linkage Evaluation

**Purpose:** Measure linkage quality and tune thresholds.

**Effort:** 2 days

#### Design

Extend existing `ConfusionMatrix` from `metrics.rs`:

- `LinkageEvaluation` - Wraps confusion matrix with pair storage
- `tune_threshold` - Parallel evaluation across threshold range
- Optional pair storage (memory optimization for large datasets)

#### API

```python
import fuzzyrust as fr

# Evaluate linkage quality
result = fr.evaluate_linkage(
    true_pairs=[(0, 1), (2, 3)],
    predicted_pairs=[(0, 1), (4, 5)],
    total_records=100,
    store_pairs=True
)
print(f"Precision: {result.precision}, Recall: {result.recall}")

# Tune threshold
best = fr.tune_threshold(
    items=strings,
    true_pairs=true_pairs,
    algorithm="jaro_winkler",
    thresholds=[0.7, 0.75, 0.8, 0.85, 0.9]
)
```

---

### Feature 3: Uncertain Pair Detection

**Purpose:** Identify matches needing human review.

**Effort:** 1.5 days

**Depends on:** Feature 1 (Blocking) for efficient pair generation

#### Design

- `UncertainPair` - Single-field comparison result
- `UncertainRecordPair` - Multi-field with per-field scores
- Integrates with existing dedup infrastructure (SNM, blocking)

#### API

```python
import fuzzyrust as fr
from fuzzyrust import polars as frp

# Core function
uncertain = fr.find_uncertain_pairs(
    strings,
    algorithm="jaro_winkler",
    low_threshold=0.7,
    high_threshold=0.9,
    limit=1000
)

# DataFrame function
uncertain_df = frp.df_find_uncertain_pairs(
    df, columns=["name", "email"],
    low_threshold=0.7, high_threshold=0.9
)
```

---

## v0.4.1: ML Integration

### Feature 4: Comparison Vectors

**Purpose:** Generate feature vectors for training custom ML classifiers.

**Effort:** 3-4 days

**Depends on:** v0.4.0 features (uses evaluation for validation)

#### Design

Builder pattern with extensible feature list:

- String similarity features (Levenshtein, Jaro-Winkler, N-gram, Cosine)
- Phonetic features (Soundex match, Metaphone match)
- Token-level features (TokenSortRatio, TokenSetRatio, PartialRatio)
- Structural features (LengthDiff, CommonPrefixLen, CommonSuffixLen)

#### API

```python
import fuzzyrust as fr

# Builder pattern (consistent with SchemaBuilder)
builder = (
    fr.ComparisonVectorBuilder()
    .add("levenshtein")
    .add("jaro_winkler")
    .add("soundex_match")
    .add("token_set_ratio")
)

# Single pair
vec = builder.compute("John Smith", "Jon Smyth")

# DataFrame batch (returns Polars DataFrame ready for sklearn)
features_df = builder.compute_dataframe(
    pairs_df, left_col="name1", right_col="name2"
)
```

---

### Feature 5: Preprocessing Pipeline

**Purpose:** Composable string transformations before comparison.

**Effort:** 2-3 days

#### Design

Extend existing `NormalizationMode` enum with new variants:

- `Strip` - Trim whitespace
- `CollapseWhitespace` - Multiple spaces to single
- `RemoveAccents` - Explicit accent removal
- `AsciiOnly` - Drop non-ASCII
- `RegexReplace` - Pattern-based replacement (pre-compiled)

#### API

```python
from fuzzyrust import preprocessing

# Fluent builder
pipeline = (
    preprocessing.Pipeline()
    .lowercase()
    .strip()
    .collapse_whitespace()
    .remove_accents()
)

# Factory presets
pipeline = preprocessing.standard()  # lowercase + strip + collapse
pipeline = preprocessing.name()      # Optimized for name matching

# Apply
clean = pipeline.apply("  Cafe  Resume!  ")
clean_list = pipeline.apply_batch(["Cafe", "Resume"])  # Parallel

# Use with deduplication
result = frp.df_dedupe(
    df, columns=["name", "address"],
    preprocessing={"name": preprocessing.name()}
)
```

---

## v0.5.0: Probabilistic Models

### Feature 6: Probabilistic Record Linkage

**Purpose:** Statistical framework for probabilistic record linkage.

**Effort:** 10 days

**Depends on:** v0.4.1 (uses comparison vectors for feature input)

#### Design Rationale

The full Fellegi-Sunter model has significant complexity (EM algorithm, independence assumptions, threshold derivation). The recommended approach is:

1. Start with **Naive Bayes classifier** - simpler, similar accuracy
2. Enable **sklearn integration** via comparison vectors for advanced users

#### API

```python
import fuzzyrust as fr

# Built-in Naive Bayes model
model = fr.NaiveBayesLinkage()
model.fit(feature_vectors, labels)

decisions = model.predict(new_vectors)
probabilities = model.predict_proba(new_vectors)

# Or use sklearn (via comparison vectors from v0.4.1)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(feature_vectors, labels)
```

---

## Implementation Schedule

### v0.4.0 (~8 days)

| Week | Phase | Features |
|------|-------|----------|
| 1 | Phase 1 | Blocking Strategies (4-5d) |
| 2 | Phase 2 | Evaluation (2d) + Uncertain Pairs (1.5d) |

### v0.4.1 (~6 days)

| Phase | Features |
|-------|----------|
| Phase 3 | Comparison Vectors (3-4d) |
| Phase 4 | Preprocessing Pipeline (2-3d) |

### v0.5.0 (~10 days)

| Phase | Features |
|-------|----------|
| Phase 5 | Naive Bayes / Probabilistic Model (10d) |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Enum for blockers | Avoids breaking Clone, Copy on DedupMethod |
| Extend df_dedupe_snm | Consistent API, no parallel functions |
| Extend NormalizationMode | Reuse existing preprocessing |
| Builder for comparison vectors | More flexible than fixed struct |
| Factory functions in Python | Consistent with batch.py pattern |
| Parallel threshold tuning | Rayon parallelization for speed |
| Optional pair storage | Memory optimization for large datasets |
| Naive Bayes before Fellegi-Sunter | Simpler, covers most use cases |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Performance | 1M record deduplication in <1 minute with blocking |
| API Consistency | All new APIs follow existing patterns |
| Test Coverage | >85% for new code |
| Documentation | Examples in all docstrings |
