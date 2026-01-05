# %% [markdown]
# # FuzzyRust Polars Integration Demo
#
# This demo shows the different levels of API for fuzzy matching with Polars.
#
# ## API Hierarchy
#
# | Level | Module | Input | Use Case |
# |-------|--------|-------|----------|
# | Core | `fr.*` | 2 strings | Single comparison |
# | Batch | `fr.batch.*` | Python lists | List-based matching |
# | Polars Series | `fr.polars.series_*` | pl.Series | Column-level ops |
# | Polars DataFrame | `fr.polars.df_*` | pl.DataFrame | Full table ops |
# | Polars Expression | `.fuzzy.*` | pl.Expr | Expression chains |
#
# ## When to Use What
#
# - **Core (`fr.*`)**: Compare two strings, get a score
# - **Batch (`fr.batch.*`)**: Process Python lists in parallel (no Polars needed)
# - **Series (`fr.polars.series_*`)**: Work with Polars Series directly
# - **DataFrame (`fr.polars.df_*`)**: Join, dedupe, or match DataFrames
# - **Expression (`.fuzzy.*`)**: Use in Polars expression chains (select, filter, with_columns)

# %%
import polars as pl

import fuzzyrust as fr
from fuzzyrust import batch
from fuzzyrust import polars as frp

# %% [markdown]
# ---
# ## Section 1: When to Use What
#
# This section demonstrates the hierarchy of APIs and when each is appropriate.

# %% [markdown]
# ### Core Functions: Compare Two Strings
#
# Use `fr.*` functions when you have exactly two strings to compare.

# %%
print("=== Core Functions (fr.*) ===")
print("Use case: Compare two specific strings")
print()

# Simple similarity comparison
score = fr.jaro_winkler_similarity("John Smith", "Jon Smith")
print(f"  fr.jaro_winkler_similarity('John Smith', 'Jon Smith') = {score:.3f}")

# Edit distance
dist = fr.levenshtein("hello", "hallo")
print(f"  fr.levenshtein('hello', 'hallo') = {dist}")

# Case-insensitive comparison (using normalize parameter)
score_ci = fr.jaro_winkler_similarity("HELLO", "hello", normalize="lowercase")
print(f"  fr.jaro_winkler_similarity('HELLO', 'hello', normalize='lowercase') = {score_ci:.3f}")

# %% [markdown]
# ### Batch Functions: Process Python Lists
#
# Use `fr.batch.*` when you have Python lists and want parallel processing.
# This is independent of Polars - works with pure Python.

# %%
print("\n=== Batch Functions (fr.batch.*) ===")
print("Use case: Process Python lists with parallel execution")
print()

# Sample data as Python lists
names = ["Apple Inc", "Microsoft Corp", "Google LLC", "Amazon.com"]
query = "Microsft"

# Find best matches from a list
matches = batch.best_matches(names, query, algorithm="jaro_winkler", limit=3)
print(f"  batch.best_matches({names}, '{query}'):")
for m in matches:
    print(f"    [{m.score:.2f}] {m.text}")
print()

# Compute similarity for all items against a query
results = batch.similarity(names, query, algorithm="jaro_winkler")
print(f"  batch.similarity({names}, '{query}'):")
for r in results:
    print(f"    [{r.score:.2f}] {r.text}")
print()

# Pairwise similarity (aligned lists)
left = ["hello", "world", "foo"]
right = ["hallo", "word", "bar"]
scores = batch.pairwise(left, right, algorithm="jaro_winkler")
print(f"  batch.pairwise({left}, {right}):")
print(f"    Scores: {[f'{s:.2f}' for s in scores]}")

# %% [markdown]
# ### Polars Series Functions: Column-Level Operations
#
# Use `fr.polars.series_*` when working with Polars Series directly.

# %%
print("\n=== Polars Series Functions (fr.polars.series_*) ===")
print("Use case: Operations on Polars Series")
print()

# Pairwise similarity between two aligned Series
left_series = pl.Series("left", ["hello", "world", "foo"])
right_series = pl.Series("right", ["hallo", "word", "bar"])

scores = frp.series_similarity(left_series, right_series, algorithm="jaro_winkler")
print("  frp.series_similarity(left, right):")
print(f"    {scores}")
print()

# Find best match for each query from a target list
queries = pl.Series("queries", ["Appel", "Microsft", "Googel"])
targets = ["Apple", "Microsoft", "Google", "Amazon"]

matches = frp.series_best_match(queries, targets, algorithm="jaro_winkler", min_similarity=0.6)
print("  frp.series_best_match(queries, targets):")
print(f"    {matches}")

# %% [markdown]
# ### Polars DataFrame Functions: Table-Level Operations
#
# Use `fr.polars.df_*` for joining, deduplicating, or matching DataFrames.

# %%
print("\n=== Polars DataFrame Functions (fr.polars.df_*) ===")
print("Use case: Full DataFrame operations (join, dedupe, match)")
print()

# Fuzzy join two DataFrames
left_df = pl.DataFrame({"name": ["Apple Inc", "Microsft Corp", "Googel LLC"]})
right_df = pl.DataFrame(
    {
        "company": ["Apple", "Microsoft", "Google", "Amazon"],
        "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    }
)

joined = frp.df_join(
    left_df,
    right_df,
    left_on="name",
    right_on="company",
    algorithm="jaro_winkler",
    min_similarity=0.6,
)
print("  frp.df_join(left_df, right_df, left_on='name', right_on='company'):")
print(joined)

# %% [markdown]
# ### Expression Namespace: Chainable Column Operations
#
# Use `.fuzzy.*` in Polars expression chains for maximum flexibility.

# %%
print("\n=== Expression Namespace (.fuzzy.*) ===")
print("Use case: Chainable operations in Polars expressions")
print()

df = pl.DataFrame({"name": ["John", "Jon", "Jane", "Bob", "Robert"]})

# Add similarity score column
result = df.with_columns(score=pl.col("name").fuzzy.similarity("John", algorithm="jaro_winkler"))
print("  df.with_columns(score=pl.col('name').fuzzy.similarity('John')):")
print(result)
print()

# Filter by similarity threshold
similar = df.filter(pl.col("name").fuzzy.is_similar("John", min_similarity=0.8))
print("  df.filter(pl.col('name').fuzzy.is_similar('John', min_similarity=0.8)):")
print(similar)

# %% [markdown]
# ---
# ## Section 2: Expression Namespace (.fuzzy)
#
# The `.fuzzy` namespace provides chainable fuzzy matching operations
# directly in Polars expression contexts.

# %% [markdown]
# ### .fuzzy.similarity() - Compute Similarity Scores

# %%
print("=== .fuzzy.similarity() ===")
print()

df = pl.DataFrame(
    {
        "name": ["John Smith", "Jon Smyth", "Jane Doe", "Johnny Smith"],
        "nickname": ["Johnny", "Jon", "Jane", "John"],
    }
)

# Compare column to literal string
result = df.with_columns(similarity_to_john=pl.col("name").fuzzy.similarity("John Smith"))
print("  Compare to literal 'John Smith':")
print(result)
print()

# Compare two columns
result = df.with_columns(name_vs_nickname=pl.col("name").fuzzy.similarity(pl.col("nickname")))
print("  Compare 'name' column to 'nickname' column:")
print(result)

# %% [markdown]
# ### Algorithm Variations in .fuzzy.similarity()

# %%
print("\n=== Algorithm Variations ===")
print()

df = pl.DataFrame({"text": ["hello world", "hallo welt", "hello there"]})
target = "hello world"

# All 8 algorithms
algorithms = [
    "levenshtein",
    "jaro",
    "jaro_winkler",
    "ngram",
    "cosine",
    "damerau_levenshtein",
    "hamming",
    "lcs",
]

print(f"  Comparing to '{target}':")
for algo in algorithms:
    result = df.with_columns(score=pl.col("text").fuzzy.similarity(target, algorithm=algo))
    scores = result["score"].to_list()
    print(f"    {algo:20}: {[f'{s:.3f}' if s else 'None' for s in scores]}")

# %% [markdown]
# ### Parameters: ngram_size and case_insensitive

# %%
print("\n=== Parameters: ngram_size and case_insensitive ===")
print()

df = pl.DataFrame({"text": ["HELLO", "hello", "HeLLo"]})

# Case-insensitive comparison
result = df.with_columns(
    case_sensitive=pl.col("text").fuzzy.similarity("hello", case_insensitive=False),
    case_insensitive=pl.col("text").fuzzy.similarity("hello", case_insensitive=True),
)
print("  Case sensitivity comparison (target='hello'):")
print(result)
print()

# N-gram size variations
df2 = pl.DataFrame({"text": ["programming", "programmer", "program"]})
result = df2.with_columns(
    bigram=pl.col("text").fuzzy.similarity("program", algorithm="ngram", ngram_size=2),
    trigram=pl.col("text").fuzzy.similarity("program", algorithm="ngram", ngram_size=3),
    fourgram=pl.col("text").fuzzy.similarity("program", algorithm="ngram", ngram_size=4),
)
print("  N-gram size variations (target='program'):")
print(result)

# %% [markdown]
# ### .fuzzy.is_similar() - Boolean Threshold Check

# %%
print("\n=== .fuzzy.is_similar() ===")
print()

df = pl.DataFrame(
    {
        "name": ["John Smith", "Jon Smyth", "Jane Doe", "Bob Wilson", "Johnny Smith"],
    }
)

# Filter rows similar to target
result = df.filter(pl.col("name").fuzzy.is_similar("John Smith", min_similarity=0.8))
print("  Filter names similar to 'John Smith' (min_similarity=0.8):")
print(result)
print()

# Add boolean column
result = df.with_columns(is_john=pl.col("name").fuzzy.is_similar("John Smith", min_similarity=0.7))
print("  Add boolean 'is_john' column (min_similarity=0.7):")
print(result)

# %% [markdown]
# ### .fuzzy.best_match() - Find Best Match from List

# %%
print("\n=== .fuzzy.best_match() ===")
print()

df = pl.DataFrame(
    {
        "raw_category": ["Elecronics", "cloths", "foods", "homewear", "xyz"],
    }
)

categories = ["Electronics", "Clothing", "Food", "Home & Garden"]

# Find best matching category for each row
result = df.with_columns(
    matched_category=pl.col("raw_category").fuzzy.best_match(
        categories, algorithm="jaro_winkler", min_similarity=0.5
    )
)
print("  Match raw categories to standard list:")
print(result)

# %% [markdown]
# ### Getting Match with Score
#
# For getting both match and score, use the batch API or series functions
# which provide more reliable struct handling.

# %%
print("\n=== Getting Match with Score (Alternative) ===")
print()

# Use batch.best_matches for getting match + score together
categories_list = ["Electronics", "Clothing", "Food", "Home & Garden"]

# Create DataFrame with results using batch API
raw_categories = df["raw_category"].to_list()
match_data = []
for cat in raw_categories:
    matches = batch.best_matches(
        categories_list, cat, algorithm="jaro_winkler", limit=1, min_similarity=0.5
    )
    if matches:
        match_data.append({"raw": cat, "matched": matches[0].text, "score": matches[0].score})
    else:
        match_data.append({"raw": cat, "matched": None, "score": None})

result_with_score = pl.DataFrame(match_data)
print("  Match with confidence score:")
print(result_with_score)

# %% [markdown]
# ### .fuzzy.distance() - Edit Distance

# %%
print("\n=== .fuzzy.distance() ===")
print()

df = pl.DataFrame({"text": ["hello", "hallo", "world", "help"]})

result = df.with_columns(
    levenshtein=pl.col("text").fuzzy.distance("hello", algorithm="levenshtein"),
    damerau=pl.col("text").fuzzy.distance("hello", algorithm="damerau_levenshtein"),
)
print("  Edit distance from 'hello':")
print(result)

# %% [markdown]
# ### .fuzzy.normalize() - String Normalization

# %%
print("\n=== .fuzzy.normalize() ===")
print()

df = pl.DataFrame({"text": ["  HELLO World!  ", "John-Smith", "Cafe"]})

result = df.with_columns(
    lowercase=pl.col("text").fuzzy.normalize("lowercase"),
    strict=pl.col("text").fuzzy.normalize("strict"),
    no_punct=pl.col("text").fuzzy.normalize("remove_punctuation"),
)
print("  Normalization modes:")
print(result)

# %% [markdown]
# ### .fuzzy.soundex() and .fuzzy.metaphone() - Phonetic Encoding

# %%
print("\n=== Phonetic Encoding ===")
print()

df = pl.DataFrame(
    {
        "name": ["Smith", "Smyth", "Schmidt", "Robert", "Rupert"],
    }
)

result = df.with_columns(
    soundex=pl.col("name").fuzzy.soundex(),
    metaphone=pl.col("name").fuzzy.metaphone(),
)
print("  Phonetic encodings:")
print(result)

# %% [markdown]
# ---
# ## Section 3: DataFrame Operations (fr.polars.df_*)
#
# High-level operations for complete DataFrame workflows.

# %% [markdown]
# ### df_join() - Fuzzy Join Two DataFrames

# %%
print("=== frp.df_join() ===")
print()

# Source data with typos/variations
orders = pl.DataFrame(
    {
        "order_id": [1, 2, 3, 4],
        "customer": ["Appel Inc", "Microsft", "Googel", "Amazn"],
        "amount": [1000, 2000, 3000, 4000],
    }
)

# Reference data (clean)
companies = pl.DataFrame(
    {
        "company": ["Apple", "Microsoft", "Google", "Amazon"],
        "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    }
)

print("  Orders (messy):")
print(orders)
print()
print("  Companies (clean):")
print(companies)
print()

# Single-column fuzzy join
joined = frp.df_join(
    orders,
    companies,
    left_on="customer",
    right_on="company",
    algorithm="jaro_winkler",
    min_similarity=0.6,
)
print("  Fuzzy joined result:")
print(joined)

# %% [markdown]
# ### Multi-Column Fuzzy Join

# %%
print("\n=== Multi-Column Fuzzy Join ===")
print()

# Data with multiple matching columns
records = pl.DataFrame(
    {
        "name": ["John Smith", "Jane Do", "Bob Willson"],
        "city": ["New York", "LA", "Chicaggo"],
        "value": [100, 200, 300],
    }
)

reference = pl.DataFrame(
    {
        "ref_name": ["John Smith", "Jane Doe", "Bob Wilson"],
        "ref_city": ["New York", "Los Angeles", "Chicago"],
        "ref_id": [1001, 1002, 1003],
    }
)

print("  Records to match:")
print(records)
print()
print("  Reference data:")
print(reference)
print()

# Multi-column join with per-column config
joined = frp.df_join(
    records,
    reference,
    on=[
        ("name", "ref_name", {"algorithm": "jaro_winkler", "weight": 2.0}),
        ("city", "ref_city", {"algorithm": "levenshtein", "weight": 1.0}),
    ],
    min_similarity=0.5,
)
print("  Multi-column fuzzy join:")
print(joined)

# %% [markdown]
# ### df_dedupe() - Deduplicate Rows

# %%
print("\n=== frp.df_dedupe() ===")
print()

# Data with duplicates
customers = pl.DataFrame(
    {
        "name": ["John Smith", "Jon Smith", "John Smyth", "Jane Doe", "Jane Do"],
        "email": [
            "john@test.com",
            "john@test.com",
            "jsmith@mail.com",
            "jane@test.com",
            "jane@test.com",
        ],
        "phone": ["555-1234", "555-1234", "555-1234", "555-5678", "555-5678"],
    }
)

print("  Data with potential duplicates:")
print(customers)
print()

# Deduplicate with multiple columns
deduped = frp.df_dedupe(
    customers,
    columns=["name", "email"],
    algorithm="jaro_winkler",
    min_similarity=0.8,
    keep="first",
)
print("  After deduplication (with _group_id and _is_canonical):")
print(deduped)
print()

# Get only canonical (unique) rows
unique = deduped.filter(pl.col("_is_canonical"))
print("  Canonical rows only:")
print(unique.drop(["_group_id", "_is_canonical"]))

# %% [markdown]
# ### df_dedupe_snm() - Sorted Neighborhood Method
#
# For large datasets (100K+ rows), use SNM for O(N log N) performance.

# %%
print("\n=== frp.df_dedupe_snm() ===")
print("Use case: Large-scale deduplication with O(N log N) complexity")
print()

# Simulate larger dataset
large_df = pl.DataFrame(
    {
        "name": ["John Smith", "Jon Smith", "Jane Doe", "Bob Wilson"] * 25,
        "id": list(range(100)),
    }
)

print(f"  Dataset size: {len(large_df)} rows")

# SNM deduplication
deduped = frp.df_dedupe_snm(
    large_df,
    columns=["name"],
    algorithm="jaro_winkler",
    min_similarity=0.9,
    window_size=10,  # Compare each row with 10 neighbors
)

unique_count = deduped.filter(pl.col("_is_canonical")).height
print(f"  Unique rows found: {unique_count}")

# %% [markdown]
# ### df_match_pairs() - Find Similar Pairs

# %%
print("\n=== frp.df_match_pairs() ===")
print("Use case: Find all similar pairs for review")
print()

data = pl.DataFrame(
    {
        "name": ["John Smith", "Jon Smith", "Jane Doe", "John Smyth"],
        "email": ["john@a.com", "jon@a.com", "jane@b.com", "jsmith@c.com"],
    }
)

pairs = frp.df_match_pairs(
    data,
    columns=["name"],
    algorithm="jaro_winkler",
    min_similarity=0.8,
)
print("  Similar pairs found:")
print(pairs)

# %% [markdown]
# ### df_find_pairs() - Scalable Pair Finding

# %%
print("\n=== frp.df_find_pairs() ===")
print("Use case: Find similar pairs with SNM for large datasets")
print()

# Using SNM method for scalability
pairs = frp.df_find_pairs(
    data,
    columns=["name"],
    algorithm="jaro_winkler",
    min_similarity=0.8,
    method="snm",  # or "full" for complete pairwise
    window_size=5,
)
print("  Pairs found (SNM method):")
print(pairs)

# %% [markdown]
# ### df_match_records() - Batch Record Matching

# %%
print("\n=== frp.df_match_records() ===")
print("Use case: Match records from two DataFrames")
print()

queries_df = pl.DataFrame(
    {
        "name": ["John Smith", "Jane Doe"],
        "city": ["New York", "Los Angeles"],
    }
)

targets_df = pl.DataFrame(
    {
        "name": ["Jon Smith", "Jane Do", "Bob Wilson"],
        "city": ["New York", "LA", "Chicago"],
    }
)

matches = frp.df_match_records(
    queries_df,
    targets_df,
    columns=["name", "city"],
    algorithm="jaro_winkler",
    min_similarity=0.5,
)
print("  Matched records:")
print(matches)

# %% [markdown]
# ---
# ## Section 4: Series Operations (fr.polars.series_*)
#
# Operations that work directly with Polars Series.

# %% [markdown]
# ### series_similarity() - Pairwise Series Comparison

# %%
print("=== frp.series_similarity() ===")
print()

left = pl.Series("left", ["hello", "world", "foo", "bar"])
right = pl.Series("right", ["hallo", "word", "food", "baz"])

# Compute similarity for each aligned pair
scores = frp.series_similarity(left, right, algorithm="jaro_winkler")
print("  Pairwise similarity:")
print(pl.DataFrame({"left": left, "right": right, "score": scores}))

# %% [markdown]
# ### series_best_match() - Find Best Match for Each Query

# %%
print("\n=== frp.series_best_match() ===")
print()

queries = pl.Series("queries", ["Appel", "Microsft", "Gogle", "Amazn", "xyz"])
targets = ["Apple", "Microsoft", "Google", "Amazon"]

matches = frp.series_best_match(
    queries,
    targets,
    algorithm="jaro_winkler",
    min_similarity=0.6,
)
print("  Best matches:")
print(pl.DataFrame({"query": queries, "best_match": matches}))

# %% [markdown]
# ### series_dedupe() - Deduplicate a Series

# %%
print("\n=== frp.series_dedupe() ===")
print()

series = pl.Series("names", ["John Smith", "Jon Smith", "Jane Doe", "John Smyth", "Bob"])

result = frp.series_dedupe(
    series,
    algorithm="jaro_winkler",
    min_similarity=0.8,
)
print("  Deduplicated series:")
print(result)

# %% [markdown]
# ### series_match() - Match Each Query Against All Targets

# %%
print("\n=== frp.series_match() ===")
print()

queries = pl.Series("queries", ["apple", "banana"])
targets = pl.Series("targets", ["appel", "banan", "cherry", "aple"])

matches = frp.series_match(
    queries,
    targets,
    algorithm="jaro_winkler",
    min_similarity=0.7,
)
print("  All matches above threshold:")
print(matches)

# %% [markdown]
# ---
# ## Section 5: Batch Operations (fr.batch.*)
#
# Process Python lists with parallel execution. No Polars required.

# %% [markdown]
# ### batch.similarity() - Compare Query Against All Items

# %%
print("=== batch.similarity() ===")
print()

strings = ["Apple Inc", "Microsoft", "Google", "Amazon"]
query = "Microsft"

results = batch.similarity(strings, query, algorithm="jaro_winkler")
print(f"  Query: '{query}'")
print("  Results:")
for r in results:
    print(f"    [{r.score:.3f}] {r.text}")

# %% [markdown]
# ### batch.best_matches() - Find Top N Matches

# %%
print("\n=== batch.best_matches() ===")
print()

strings = ["Apple Inc", "Microsoft", "Google", "Amazon", "Netflix", "Tesla"]
query = "Googel"

matches = batch.best_matches(
    strings,
    query,
    algorithm="jaro_winkler",
    limit=3,
    min_similarity=0.5,
)
print(f"  Query: '{query}'")
print("  Top 3 matches:")
for m in matches:
    print(f"    [{m.score:.3f}] {m.text}")

# %% [markdown]
# ### batch.pairwise() - Compare Aligned Lists

# %%
print("\n=== batch.pairwise() ===")
print()

left = ["hello", "world", "foo", "bar"]
right = ["hallo", "word", "food", "baz"]

scores = batch.pairwise(left, right, algorithm="jaro_winkler")
print("  Pairwise comparison:")
for lv, rv, s in zip(left, right, scores):
    print(f"    '{lv}' vs '{rv}': {s:.3f}")

# %% [markdown]
# ### batch.deduplicate() - Find Duplicate Groups

# %%
print("\n=== batch.deduplicate() ===")
print()

strings = ["John Smith", "Jon Smith", "Jane Doe", "John Smyth", "Bob Wilson"]

result = batch.deduplicate(strings, algorithm="jaro_winkler", min_similarity=0.8)
print("  Duplicate groups:")
for i, group in enumerate(result.groups):
    print(f"    Group {i}: {group}")
print(f"  Unique items: {result.unique}")

# %% [markdown]
# ### batch.similarity_matrix() - Full Comparison Matrix

# %%
print("\n=== batch.similarity_matrix() ===")
print()

queries = ["hello", "world"]
choices = ["hallo", "word", "help"]

matrix = batch.similarity_matrix(queries, choices, algorithm="levenshtein")
print("  Similarity matrix:")
print(f"  Queries: {queries}")
print(f"  Choices: {choices}")
for i, row in enumerate(matrix):
    print(f"    {queries[i]}: {[f'{v:.2f}' for v in row]}")

# %% [markdown]
# ---
# ## Section 6: Algorithm Variations
#
# FuzzyRust supports 8 core algorithms, each with unique strengths.

# %%
print("=== Algorithm Comparison ===")
print()

# Test strings
s1 = "programming"
s2 = "programmer"

print(f"  Comparing: '{s1}' vs '{s2}'")
print()

# All algorithms with descriptions
algorithms_info = [
    ("levenshtein", "Edit distance: insertions, deletions, substitutions"),
    ("damerau_levenshtein", "Edit distance with transpositions (ab->ba)"),
    ("jaro", "Character matching with transposition penalty"),
    ("jaro_winkler", "Jaro + prefix bonus (best for names)"),
    ("ngram", "N-gram overlap (Dice coefficient)"),
    ("cosine", "Vector-based cosine similarity"),
    ("hamming", "Positional differences (same-length strings)"),
    ("lcs", "Longest common subsequence based"),
]

for algo, description in algorithms_info:
    score = fr.jaro_winkler_similarity(s1, s2) if algo == "jaro_winkler" else None

    # Compute score using the appropriate function
    if algo == "levenshtein":
        score = fr.levenshtein_similarity(s1, s2)
    elif algo == "damerau_levenshtein":
        score = fr.damerau_levenshtein_similarity(s1, s2)
    elif algo == "jaro":
        score = fr.jaro_similarity(s1, s2)
    elif algo == "jaro_winkler":
        score = fr.jaro_winkler_similarity(s1, s2)
    elif algo == "ngram":
        score = fr.ngram_similarity(s1, s2, ngram_size=3)
    elif algo == "cosine":
        score = fr.cosine_similarity_chars(s1, s2)
    elif algo == "hamming":
        # Hamming requires same length, use padded version
        score = fr.hamming_similarity(s1, s2) if len(s1) == len(s2) else None
    elif algo == "lcs":
        score = fr.lcs_similarity(s1, s2)

    score_str = f"{score:.3f}" if score is not None else "N/A"
    print(f"  {algo:20} {score_str}  - {description}")

# %% [markdown]
# ### N-gram Size Parameter
#
# The `ngram` algorithm uses the `ngram_size` parameter (default: 3).

# %%
print("\n=== N-gram Size Impact ===")
print()

s1 = "programming"
s2 = "programmer"

for n in [2, 3, 4, 5]:
    score = fr.ngram_similarity(s1, s2, ngram_size=n)
    print(f"  ngram_size={n}: {score:.3f}")

# %% [markdown]
# ### Case-Insensitive Variants
#
# All algorithms have `_ci` suffix variants for case-insensitive comparison.

# %%
print("\n=== Case-Insensitive Comparison (normalize parameter) ===")
print()

s1 = "HELLO"
s2 = "hello"

print(f"  Comparing: '{s1}' vs '{s2}'")
print()
print(f"  levenshtein:                    {fr.levenshtein(s1, s2)} edits")
print(f"  levenshtein (normalize=lower):  {fr.levenshtein(s1, s2, normalize='lowercase')} edits")
print()
print(f"  jaro_winkler:                   {fr.jaro_winkler_similarity(s1, s2):.3f}")
print(
    f"  jaro_winkler (normalize=lower): {fr.jaro_winkler_similarity(s1, s2, normalize='lowercase'):.3f}"
)
print()
print(f"  ngram_similarity:                   {fr.ngram_similarity(s1, s2):.3f}")
print(
    f"  ngram_similarity (normalize=lower): {fr.ngram_similarity(s1, s2, normalize='lowercase'):.3f}"
)

# %% [markdown]
# ---
# ## Section 7: Performance Tips

# %% [markdown]
# ### Native Polars Plugin
#
# When available, the native plugin provides 10-50x speedup for column operations.

# %%
print("=== Native Plugin Status ===")
print()

from fuzzyrust._plugin import is_plugin_available

if is_plugin_available():
    print("  Native Polars plugin is ENABLED (10-50x faster)")
else:
    print("  Native plugin not available (using fallback)")
print()
print("  The plugin automatically accelerates:")
print("    - .fuzzy.similarity() with column-to-column comparisons")
print("    - .fuzzy.is_similar() threshold checks")
print("    - .fuzzy.soundex() and .fuzzy.metaphone()")

# %% [markdown]
# ### When to Use Which API

# %%
print("\n=== API Selection Guide ===")
print()
print("  Dataset Size        | Recommended API")
print("  --------------------|------------------------")
print("  2 strings           | fr.* (core functions)")
print("  Python lists        | fr.batch.* (parallel)")
print("  < 10K rows          | Any Polars API")
print("  10K - 100K rows     | frp.df_* or .fuzzy")
print("  > 100K rows         | frp.df_dedupe_snm()")
print("  > 1M rows           | SNM with large window_size")

# %% [markdown]
# ### Indexing for Repeated Searches
#
# For repeated searches against the same data, build an index.

# %%
print("\n=== Using Indices for Performance ===")
print()

# Sample data
products = ["iPhone 15 Pro", "MacBook Air", "iPad Pro", "AirPods Pro", "Apple Watch"]

# Build index once
index = fr.HybridIndex(ngram_size=2, normalize=True)
index.add_all(products)

print(f"  Built index with {len(index)} items")

# Fast repeated searches
queries = ["iphone", "macbook", "airpod"]
for q in queries:
    results = index.search(q, algorithm="jaro_winkler", min_similarity=0.5, limit=1)
    match = results[0].text if results else "No match"
    print(f"    '{q}' -> {match}")

# %% [markdown]
# ---
# ## Summary
#
# | Level | Module | Best For |
# |-------|--------|----------|
# | Core | `fr.*` | Single comparisons |
# | Batch | `fr.batch.*` | Python list processing |
# | Series | `frp.series_*` | Polars Series operations |
# | DataFrame | `frp.df_*` | Join/dedupe/match DataFrames |
# | Expression | `.fuzzy.*` | Polars expression chains |
#
# **Key Functions:**
#
# - `frp.df_join()` - Fuzzy join two DataFrames
# - `frp.df_dedupe()` - Deduplicate with clustering
# - `frp.df_dedupe_snm()` - O(N log N) deduplication for large data
# - `frp.series_similarity()` - Pairwise Series comparison
# - `frp.series_best_match()` - Find best match for each query
# - `.fuzzy.similarity()` - In expressions with algorithm/ngram_size/case_insensitive
# - `.fuzzy.is_similar()` - Boolean threshold check
# - `batch.similarity()` / `batch.best_matches()` - Python list processing
