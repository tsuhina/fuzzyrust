# %% [markdown]
# # FuzzyRust: The Complete Guide
#
# **From typos to terabytes** - A hands-on journey through fuzzy string matching
#
# ---
#
# ## The Problem
#
# You have data. It's messy. Names are misspelled, addresses are inconsistent,
# and your users can't spell "definitely" to save their lives.
#
# ```
# "Jhon Smith"      vs  "John Smith"
# "recieve"         vs  "receive"
# "123 Main St."    vs  "123 Main Street"
# "pulp ficton"     vs  "Pulp Fiction"
# ```
#
# **FuzzyRust** fixes this. It's written in Rust for speed, with Python bindings
# for convenience. Let's see what it can do.
#
# ---
#
# ## Table of Contents
#
# | Part | Topic | Description |
# |------|-------|-------------|
# | 1 | The Hook | Quick wins: movie search, dedup, autocomplete |
# | 2 | Polars Power | DataFrame fuzzy join, dedup, expression namespace |
# | 3 | Search at Scale | Index structures, batch operations, performance |
# | 4 | Similarity Fundamentals | Edit distance, Jaro-Winkler, case-insensitive |
# | 5 | Advanced Algorithms | N-grams, cosine, LCS, phonetic matching |
# | 6 | Production Patterns | Preprocessing, algorithm selection, thread safety |
# | 7 | Enterprise Features | Schema matching, deduplication, evaluation metrics |
# | 8 | Edge Cases & Best Practices | Unicode, normalization modes, algorithm guide |
# | Appendix | RapidFuzz Compatibility | Drop-in replacements for migration |

# %%
import time

import fuzzyrust as fr

# %% [markdown]
# ---
# ## Part 1: The Hook
#
# *"Show me what you've got"*
#
# Before we dive into theory, let's solve some real problems.

# %% [markdown]
# ### Problem 1: Find a movie despite typos
#
# Your user typed "pulp ficton" but meant "Pulp Fiction".

# %%
movies = [
    "The Shawshank Redemption",
    "The Godfather",
    "The Dark Knight",
    "Pulp Fiction",
    "Fight Club",
    "Inception",
    "The Matrix",
    "Forrest Gump",
    "Interstellar",
    "Parasite",
]

query = "pulp ficton"  # User's typo

# Find the best match
matches = fr.batch.best_matches(movies, query, algorithm="jaro_winkler", limit=3)

print(f"User searched: '{query}'")
print("Top matches:")
for m in matches:
    print(f"  [{m.score:.0%}] {m.text}")

# %% [markdown]
# ### Problem 2: Clean up duplicate customer names
#
# Your CRM has multiple entries for the same person:

# %%
customers = [
    "John Smith",
    "Jon Smith",
    "John Smyth",
    "Jonathan Smith",
    "Jane Smith",
    "John Smith Jr",
    "Robert Johnson",
    "Bob Johnson",
    "Rob Johnson",
]

result = fr.batch.deduplicate(
    customers,
    algorithm="jaro_winkler",
    min_similarity=0.80,  # Lower threshold to catch name variations like Jonathan/John
)

print("Duplicate groups found:")
for i, group in enumerate(result.groups, 1):
    print(f"  Group {i}: {group}")
print(f"\nUnique records: {result.unique}")

# %% [markdown]
# ### Problem 3: Autocomplete with typo tolerance
#
# Build a search that handles keyboard mistakes:

# %%
products = [
    "MacBook Pro 14-inch",
    "MacBook Pro 16-inch",
    "MacBook Air M3",
    "iPhone 15 Pro",
    "iPhone 15 Pro Max",
    "iPad Pro 12.9",
    "AirPods Pro",
    "Apple Watch Ultra",
]

# Create a fast fuzzy search index
index = fr.HybridIndex(ngram_size=2, normalize=True)
for product in products:
    index.add(product)

# Simulate typos
typos = ["macbok", "iphone pro", "airpod", "ipad"]

print("Fuzzy Autocomplete:")
for typo in typos:
    results = index.search(typo, algorithm="jaro_winkler", min_similarity=0.5, limit=2)
    suggestions = [r.text for r in results]
    print(f"  '{typo}' -> {suggestions}")

# %% [markdown]
# ---
# ## Part 2: Polars Power
#
# *"DataFrame-native fuzzy matching"*
#
# Polars is included as a required dependency, enabling DataFrame-native fuzzy
# matching with per-field algorithm selection.

# %%
import polars as pl

from fuzzyrust import FuzzyIndex
from fuzzyrust import polars as frp

# %% [markdown]
# ### 2A: Fuzzy Join DataFrames
#
# Join two DataFrames on fuzzy matching columns - the core use case.

# %%
print("Fuzzy Join - Match messy data to clean reference:")
print()

# Your clean customer database
customers = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "name": ["Apple Inc.", "Microsoft Corporation", "Google LLC", "Amazon.com Inc."],
    }
)

# Incoming orders with typos and variations
orders = pl.DataFrame(
    {
        "order_id": ["A1", "A2", "A3", "A4"],
        "company": ["Appel", "Microsft Corp", "Googel", "Amzon Inc"],
        "amount": [5000, 3000, 7000, 2000],
    }
)

print("  Clean customer database:")
print(customers)
print()
print("  Messy order data:")
print(orders)
print()

# One line to match them
matched = frp.df_join(
    orders,
    customers,
    left_on="company",
    right_on="name",
    min_similarity=0.5,  # Lower threshold to catch case differences like GOOGLE/Google
)

print("  Fuzzy joined result:")
print(matched)

# %% [markdown]
# ### Multi-Column Fuzzy Join
#
# Join on multiple columns with per-column algorithms and weights:

# %%
print("\nMulti-Column Fuzzy Join:")
print()

# Reference data
reference = pl.DataFrame(
    {
        "ref_name": ["John Smith", "Jane Doe", "Bob Wilson"],
        "ref_city": ["New York", "Los Angeles", "Chicago"],
        "ref_id": [101, 102, 103],
    }
)

# Data to match
to_match = pl.DataFrame(
    {
        "name": ["Jon Smith", "Jane Do", "Robert Wilson"],
        "city": ["New York", "LA", "Chicago"],
        "value": [100, 200, 300],
    }
)

print("  Reference data:")
print(reference)
print()
print("  Data to match:")
print(to_match)
print()

# Multi-column join with per-field algorithms
result = frp.df_join(
    to_match,
    reference,
    on=[
        ("name", "ref_name", {"algorithm": "jaro_winkler", "weight": 2.0}),
        ("city", "ref_city", {"algorithm": "levenshtein", "weight": 1.0}),
    ],
    min_similarity=0.5,
)

print("  Multi-column fuzzy join result:")
print(result)

# %% [markdown]
# ### 2B: DataFrame Deduplication
#
# Two approaches for finding duplicates in a single DataFrame:
# - `frp.df_match_pairs()` - Explore: find similar pairs for review
# - `frp.df_dedupe()` - Act: group duplicates and pick canonical records

# %%
print("\nEntity Deduplication:")
print()

# Customer records with potential duplicates
customers = pl.DataFrame(
    {
        "name": ["John Smith", "Jon Smyth", "Jane Doe", "John Smith Jr"],
        "email": ["john@test.com", "john@test.com", "jane@test.com", "john.jr@test.com"],
        "phone": ["555-1234", "555-1234", "555-9999", "555-1234"],
    }
)

print("  Original data:")
print(customers)

# %% [markdown]
# **Step 1: Explore with `frp.df_match_pairs()`**
#
# Find all similar pairs for review. Great for understanding your data
# or when you need human review before merging.

# %%
print("\n  Step 1: Find similar pairs (df_match_pairs)")
print()

pairs = frp.df_match_pairs(
    customers,
    columns=["name", "email", "phone"],
    algorithms={
        "name": "jaro_winkler",
        "email": "levenshtein",
        "phone": "levenshtein",  # Using levenshtein for phone matching
    },
    weights={"name": 2.0, "email": 1.5, "phone": 1.0},
    min_similarity=0.5,
)

print("  Similar pairs found:")
print(pairs)
print()
print("  Use case: Review these pairs manually, export for data steward review,")
print("  or feed into a workflow that needs human approval before merging.")

# %% [markdown]
# **Step 2: Act with `frp.df_dedupe()`**
#
# Automatically group duplicates and mark one as "canonical" (the keeper).
# Uses Union-Find clustering to handle transitive duplicates (A~B, B~C -> A,B,C grouped).

# %%
print("\n  Step 2: Group and pick winners (df_dedupe)")
print()

result = frp.df_dedupe(
    customers,
    columns=["name", "email", "phone"],
    algorithms={
        "name": "jaro_winkler",
        "email": "levenshtein",
        "phone": "levenshtein",  # Using levenshtein for phone matching
    },
    weights={"name": 2.0, "email": 1.5, "phone": 1.0},
    min_similarity=0.5,
    keep="first",  # or "last", "most_complete"
)

print("  With deduplication columns:")
print(result)
print()
print("  _group_id: Which duplicate group this row belongs to (null = unique)")
print("  _is_canonical: True = keep this row, False = it's a duplicate")

# Get only unique (canonical) rows
unique = result.filter(pl.col("_is_canonical"))
print("\n  Canonical (deduplicated) records:")
print(unique)

# %% [markdown]
# ### 2C: Expression Namespace
#
# Use the `.fuzzy` namespace for column-level operations:

# %%
print("\nExpression Namespace (.fuzzy):")
print()

df = pl.DataFrame({"name": ["John", "Jon", "Jane", "Bob"]})

# Calculate similarity to a target
result = df.with_columns(score=pl.col("name").fuzzy.similarity("John", algorithm="jaro_winkler"))
print("  Similarity scores:")
print(result)
print()

# Filter by similarity
similar = df.filter(pl.col("name").fuzzy.is_similar("John", min_similarity=0.8))
print("  Names similar to 'John' (min_similarity=0.8):")
print(similar)

# %% [markdown]
# ### 2D: FuzzyIndex for Batch Operations
#
# Build reusable indices for efficient repeated searches:

# %%
print("\nFuzzyIndex for Batch Operations:")
print()

# Build index from Series
targets = pl.Series(["Apple Inc", "Microsoft Corp", "Google LLC"])
index = FuzzyIndex.from_series(targets, algorithm="ngram")

print(f"  Built index: {index}")

# Batch search
queries = pl.Series(["Apple", "Microsft"])
results = index.search_series(queries, min_similarity=0.5)
print("\n  Batch search results:")
print(results)

# %% [markdown]
# ---
# ## Part 3: Search at Scale
#
# *"From 100 to 1,000,000 records"*

# %% [markdown]
# ### 3A: Index Structures
#
# For large datasets, build an index once, then search many times.

# %%
print("Index Structures Comparison:")
print()

# Sample data
words = ["apple", "application", "apply", "banana", "bandana", "orange", "grape", "grapefruit"]

# %% [markdown]
# **BkTree**: Best for exact edit distance queries ("find all within 2 edits")

# %%
print("BkTree (edit distance queries):")

bktree = fr.BkTree()  # Uses Levenshtein by default
bktree.add_all(words)

results = bktree.search("aple", max_distance=2)
print("  Search: 'aple' (max_distance=2)")
for r in results:
    print(f"    [dist={r.distance}] {r.text}")

# With Damerau-Levenshtein (handles transpositions)
bktree_damerau = fr.BkTree(use_damerau=True)
bktree_damerau.add_all(words)
print("\n  With Damerau (transpositions):")
results = bktree_damerau.search("aplpe", max_distance=2)  # transposition
for r in results:
    print(f"    [dist={r.distance}] {r.text}")

# %% [markdown]
# **NgramIndex**: Fast candidate filtering with flexible similarity

# %%
print("\nNgramIndex (flexible similarity):")

ngram_idx = fr.NgramIndex(ngram_size=2, normalize=True)
for i, word in enumerate(words):
    ngram_idx.add_with_data(word, i)  # Store user data (e.g., database ID)

results = ngram_idx.search("banan", algorithm="jaro_winkler", min_similarity=0.5, limit=3)
print("  Search: 'banan'")
for r in results:
    print(f"    [{r.score:.2f}] {r.text} (id={r.id}, data={r.data})")

# Find k nearest neighbors
print("\n  find_nearest (k-NN):")
nearest = ngram_idx.find_nearest("orangee", limit=2)
for r in nearest:
    print(f"    [{r.score:.2f}] {r.text}")

# %% [markdown]
# ### User Data: Connecting Results to Your Database
#
# The `data` field lets you store database IDs, row numbers, or any integer identifier
# with each indexed item. This is **critical** for production use.

# %%
print("\nUser Data Association:")
print()

# Simulate a product database
product_db = {
    101: {"name": "iPhone 15 Pro", "price": 999},
    102: {"name": "iPhone 15 Pro Max", "price": 1199},
    103: {"name": "Samsung Galaxy S24", "price": 899},
    104: {"name": "Google Pixel 8", "price": 699},
}

# Build index with database IDs
product_index = fr.NgramIndex(ngram_size=2, normalize=True)
for db_id, product in product_db.items():
    product_index.add_with_data(product["name"], db_id)  # Store the DB ID!

# Search returns the database ID in r.data
results = product_index.search("iphone pro", algorithm="jaro_winkler", min_similarity=0.6, limit=3)
print("  Search: 'iphone pro'")
for r in results:
    # Use r.data to look up the full record
    full_record = product_db[r.data]
    print(f"    [{r.score:.0%}] {r.text} -> DB ID: {r.data}, Price: ${full_record['price']}")

# %% [markdown]
# ### Exact Match Checking with contains()
#
# Check if a string exists in the index without running a full search.

# %%
print("\nExact Match with contains():")
print()

# Check for exact matches
print(f"  contains('apple'): {ngram_idx.contains('apple')}")
print(f"  contains('Apple'): {ngram_idx.contains('Apple')}")  # Case-sensitive!
print(f"  contains('pineapple'): {ngram_idx.contains('pineapple')}")

# Useful for deduplication: skip if already indexed
new_item = "banana"
if not ngram_idx.contains(new_item):
    ngram_idx.add(new_item)
    print(f"\n  Added '{new_item}' (wasn't in index)")
else:
    print(f"\n  Skipped '{new_item}' (already indexed)")

# %% [markdown]
# ### Two-Stage Search with get_candidates()
#
# For custom scoring logic, get candidates first, then apply your own scoring.

# %%
print("\nTwo-Stage Search Pipeline:")
print()

# Build a simple index
custom_index = fr.NgramIndex(ngram_size=2)
custom_index.add_all(
    ["machine learning", "deep learning", "reinforcement learning", "transfer learning"]
)

# Stage 1: Fast n-gram candidate filtering
query = "machin lerning"
candidates = custom_index.get_candidates(query)  # Returns list of (id, text) tuples
print(f"  Query: '{query}'")
print(f"  Stage 1 - Candidates (n-gram match): {len(candidates)} items")

# Stage 2: Custom scoring on candidates only
print("  Stage 2 - Custom scoring:")
for cid, text in candidates:  # Unpack (id, text) tuple
    # Apply any custom scoring logic you want
    jw_score = fr.jaro_winkler_similarity(query, text)
    lev_score = fr.levenshtein_similarity(query, text)
    combined = (jw_score + lev_score) / 2
    print(f"    '{text}': JW={jw_score:.2f}, Lev={lev_score:.2f}, Combined={combined:.2f}")

# %% [markdown]
# **HybridIndex**: Best balance of speed and accuracy for large datasets

# %%
print("\nHybridIndex (production-ready):")

hybrid = fr.HybridIndex(ngram_size=2, normalize=True)
hybrid.add_all(words)

# Single search
results = hybrid.search("grape", algorithm="jaro_winkler", min_similarity=0.4, limit=3)
print("  Search: 'grape'")
for r in results:
    print(f"    [{r.score:.2f}] {r.text}")

# Batch search (parallel)
print("\n  batch_search (parallel):")
queries = ["apple", "banana"]
batch_results = hybrid.batch_search(queries, algorithm="jaro_winkler", min_similarity=0.5)
for query, results in zip(queries, batch_results):
    print(f"    '{query}': {[r.text for r in results]}")

# %% [markdown]
# ### 3B: Batch Operations
#
# Process many strings at once with automatic parallelization.

# %%
print("Batch Operations (parallel processing):")
print()

# Generate test data
names = [f"Customer {i:05d}" for i in range(10_000)]

# Time the batch operation
start = time.time()
results = fr.batch.similarity(names, "Customer 05000", algorithm="jaro_winkler")
elapsed = time.time() - start

print(f"  Processed {len(names):,} strings in {elapsed:.3f}s")
print(f"  Throughput: {len(names) / elapsed:,.0f} comparisons/sec")
# Find the best match (results are not sorted, so we need to find the max)
best = max(results, key=lambda r: r.score)
print(f"  Best match: [{best.score:.2f}] {best.text}")

# %%
print("\nbatch.best_matches (with algorithm selection):")
print()

artists = [
    "The Beatles",
    "Led Zeppelin",
    "Pink Floyd",
    "Queen",
    "Nirvana",
    "Radiohead",
    "Coldplay",
    "Oasis",
    "Arctic Monkeys",
    "The Rolling Stones",
]

query = "beatles"
matches = fr.batch.best_matches(
    artists, query, algorithm="jaro_winkler", limit=3, min_similarity=0.3
)

print(f"  Query: '{query}'")
for m in matches:
    print(f"    [{m.score:.0%}] {m.text}")

# %%
print("\ncompare_algorithms (find the best algorithm for your data):")
print()

sample = ["MacBook Pro", "MacBook Air", "iPad Pro", "iPhone 15"]
query = "mac book"

comparisons = fr.compare_algorithms(sample, query, limit=2)

for c in comparisons:
    print(f"  {c.algorithm}: avg score {c.score:.3f}")
    for m in c.matches:
        print(f"    [{m.score:.2f}] {m.text}")
    print()

# %% [markdown]
# ### 3C: Performance at Scale

# %%
print("\nPerformance at Scale:")
print()

# Generate larger dataset
large_data = [f"product_{i:06d}" for i in range(100_000)]

# Build index
start = time.time()
large_index = fr.HybridIndex(ngram_size=2)
large_index.add_all(large_data)
build_time = time.time() - start
print(f"  Built index with {len(large_index):,} items in {build_time:.2f}s")

# Search
start = time.time()
for _ in range(100):
    large_index.search("product_050000", algorithm="jaro_winkler", min_similarity=0.8, limit=5)
search_time = (time.time() - start) / 100
print(f"  Average search time: {search_time * 1000:.2f}ms")

# %% [markdown]
# ---
# ## Part 4: Similarity Fundamentals
#
# *"How do we measure 'close enough'?"*
#
# There are many ways to compare strings. Each has strengths and weaknesses.

# %% [markdown]
# ### 4A: Edit Distance
#
# **Levenshtein Distance**: Count insertions, deletions, and substitutions.

# %%
print("Levenshtein Distance (edit operations):")
print()

examples = [
    ("kitten", "sitting", "3 edits: k->s, e->i, +g"),
    ("hello", "hallo", "1 edit: e->a"),
    ("flaw", "lawn", "2 edits: f->l, +n, -w"),
    ("receive", "recieve", "1 edit: swap i/e"),
    ("", "hello", "5 edits: insert all"),
]

for a, b, explanation in examples:
    dist = fr.levenshtein(a, b)
    sim = fr.levenshtein_similarity(a, b)
    print(f"  '{a}' -> '{b}'")
    print(f"    Distance: {dist}, Similarity: {sim:.0%} ({explanation})")
    print()

# %% [markdown]
# **Damerau-Levenshtein**: Also counts transpositions (ab -> ba) as 1 edit.
#
# Perfect for keyboard typos where adjacent keys get swapped.

# %%
print("Damerau-Levenshtein (handles transpositions):")
print()

typos = [
    ("the", "teh", "Common keyboard typo"),
    ("from", "form", "Letter swap"),
    ("receive", "recieve", "Classic misspelling"),
]

for correct, typo, desc in typos:
    lev = fr.levenshtein(correct, typo)
    dam = fr.damerau_levenshtein(correct, typo)
    print(f"  '{correct}' -> '{typo}' ({desc})")
    print(f"    Levenshtein: {lev}, Damerau: {dam}")
    print()

# %% [markdown]
# **Hamming Distance**: Only works on equal-length strings. Counts positions that differ.
#
# Great for codes, DNA sequences, and fixed-format data.

# %%
print("Hamming Distance (positional differences):")
print()

# DNA sequences
dna1 = "GATTACA"
dna2 = "GACTACA"
print(f"  DNA: '{dna1}' vs '{dna2}'")
print(f"    Hamming distance: {fr.hamming(dna1, dna2)}")
print(f"    Hamming similarity: {fr.hamming_similarity(dna1, dna2):.0%}")
print()

# Binary codes
code1 = "10110101"
code2 = "10100111"
print(f"  Binary: '{code1}' vs '{code2}'")
print(f"    Hamming distance: {fr.hamming(code1, code2)} bit differences")
print()

# For unequal lengths, use padded version
print("  Padded: 'abc' vs 'abcd'")
print(f"    Hamming padded: {fr.hamming_distance_padded('abc', 'abcd')}")

# %% [markdown]
# ### 4B: Similarity Scores (0.0 to 1.0)
#
# **Jaro Similarity**: Character matching with transposition penalty.

# %%
print("Jaro Similarity:")
print()

# Classic example from the original Jaro paper
print("  'MARTHA' vs 'MARHTA'")
jaro = fr.jaro_similarity("MARTHA", "MARHTA")
print(f"    Jaro: {jaro:.3f}")
print()

# %% [markdown]
# **Jaro-Winkler**: Adds a bonus for matching prefixes.
#
# This is the **best algorithm for names** because:
# - First letters matter more (people get first letters right)
# - Common prefixes boost the score

# %%
print("Jaro-Winkler (with prefix bonus):")
print()

name_pairs = [
    ("JOHNSON", "JONSON"),
    ("WILLIAMS", "WILLIAMSON"),
    ("Catherine", "Kathryn"),
    ("Michael", "Michel"),
]

for a, b in name_pairs:
    jaro = fr.jaro_similarity(a, b)
    jaro_winkler = fr.jaro_winkler_similarity(a, b)
    boost = jaro_winkler - jaro
    print(f"  '{a}' vs '{b}'")
    print(f"    Jaro: {jaro:.3f}, Jaro-Winkler: {jaro_winkler:.3f} (+{boost:.3f} prefix bonus)")

# %% [markdown]
# ### 4C: Case-Insensitive Matching
#
# **Case-Insensitive Variants** (`_ci` suffix):
#
# Every algorithm has a case-insensitive version. Use it when case shouldn't matter.

# %%
print("Case-Insensitive Variants (_ci suffix):")
print()

# Compare the same algorithm with and without _ci
pairs = [
    ("Hello", "HELLO"),
    ("Product-ABC", "product-abc"),
    ("iPhone", "iphone"),
]

for a, b in pairs:
    regular = fr.levenshtein(a, b)
    ci = fr.levenshtein(a, b, normalize="lowercase")
    print(f"  '{a}' vs '{b}'")
    print(f"    levenshtein:                    {regular} edits")
    print(f"    levenshtein (normalize=lower):  {ci} edits")
    print()

print("  All algorithms support the normalize parameter:")
print("    jaro_winkler_similarity, ngram_similarity, damerau_levenshtein, etc.")

# %% [markdown]
# ---
# ## Part 5: Advanced Algorithms
#
# *"N-grams, cosine, sequences, and sound"*

# %% [markdown]
# ### 5A: N-grams
#
# N-grams capture local patterns. Two strings with similar n-grams are similar.

# %%
print("N-gram Extraction:")
print()

text = "hello"
print(f"  Text: '{text}'")
print(f"  Bigrams (n=2):  {fr.extract_ngrams(text, ngram_size=2)}")
print(f"  Trigrams (n=3): {fr.extract_ngrams(text, ngram_size=3)}")

# %% [markdown]
# **Dice Coefficient** (ngram_similarity): 2 * |intersection| / (|A| + |B|)
#
# **Jaccard Index** (ngram_jaccard): |intersection| / |union| - stricter

# %%
print("\nN-gram Similarity (Dice vs Jaccard):")
print()

pairs = [
    ("night", "nacht"),
    ("hello world", "hello there"),
    ("iPhone 15 Pro Max", "Apple iPhone 15 Pro"),
]

for a, b in pairs:
    dice = fr.ngram_similarity(a, b, ngram_size=2)
    jaccard = fr.ngram_jaccard(a, b, ngram_size=2)
    print(f"  '{a}' vs '{b}'")
    print(f"    Dice: {dice:.3f}, Jaccard: {jaccard:.3f}")
    print()

# Convenience functions
print("Convenience functions:")
print(f"  bigram_similarity('hello', 'hallo'): {fr.bigram_similarity('hello', 'hallo'):.3f}")
print(f"  trigram_similarity('hello', 'hallo'): {fr.trigram_similarity('hello', 'hallo'):.3f}")

# %% [markdown]
# **Profile Similarity**: Considers n-gram frequency, not just presence.

# %%
print("\nN-gram Profile (frequency-aware):")
print()

# 'aaa' has repeated bigram 'aa'
print(f"  'aaa' vs 'aaa': {fr.ngram_profile_similarity('aaa', 'aaa', ngram_size=2):.3f}")
print(f"  'aaa' vs 'abc': {fr.ngram_profile_similarity('aaa', 'abc', ngram_size=2):.3f}")
print(
    f"  'banana' vs 'bandana': {fr.ngram_profile_similarity('banana', 'bandana', ngram_size=2):.3f}"
)

# %% [markdown]
# ### 5B: Cosine & LCS Similarity
#
# **Longest Common Subsequence (LCS)**: Find the longest sequence of characters
# that appear in both strings (in order, but not necessarily contiguous).

# %%
print("Longest Common Subsequence:")
print()

# Classic example
a, b = "AGGTAB", "GXTXAYB"
print(f"  Strings: '{a}' and '{b}'")
print(f"  LCS string: '{fr.lcs_string(a, b)}'")
print(f"  LCS length: {fr.lcs_length(a, b)}")
print(f"  LCS similarity: {fr.lcs_similarity(a, b):.3f}")
print(f"  LCS similarity (max): {fr.lcs_similarity_max(a, b):.3f}")
print()

# Code similarity (plagiarism detection)
code1 = "for i in range(10): print(i)"
code2 = "for x in range(10): print(x)"
print(f"  Code similarity: {fr.lcs_similarity(code1, code2):.3f}")

# %% [markdown]
# **Longest Common Substring**: Contiguous match (more strict).

# %%
print("\nLongest Common Substring (contiguous):")
print()

words = [
    ("photograph", "tomography"),
    ("programming", "programmer"),
    ("interstellar", "stellar"),
]

for a, b in words:
    substr = fr.longest_common_substring(a, b)
    length = fr.longest_common_substring_length(a, b)
    print(f"  '{a}' & '{b}'")
    print(f"    Substring: '{substr}' (length {length})")

# %% [markdown]
# **Cosine Similarity**: Treat strings as vectors. Compare their direction, not magnitude.

# %%
print("\nCosine Similarity (vector-based):")
print()

a = "the quick brown fox jumps"
b = "the quick red fox leaps"

print(f"  '{a}'")
print(f"  '{b}'")
print()
print(f"  Character-level: {fr.cosine_similarity_chars(a, b):.3f}")
print(f"  Word-level:      {fr.cosine_similarity_words(a, b):.3f}")
print(f"  N-gram level:    {fr.cosine_similarity_ngrams(a, b, ngram_size=2):.3f}")

# %% [markdown]
# **TF-IDF Cosine**: Weight rare words higher than common ones.
#
# "the" appears everywhere, so it's less important. "quantum" is rare, so it matters more.

# %%
print("\nTF-IDF Cosine (corpus-aware):")
print()

# Build a corpus
tfidf = fr.TfIdfCosine()
docs = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps all day",
    "quick brown rabbits hop in the field",
    "quantum physics explains the universe",
    "the theory of quantum mechanics",
]
tfidf.add_documents(docs)

print(f"  Corpus size: {tfidf.num_documents()} documents")
print()

# Compare documents
comparisons = [
    ("quick fox", "lazy dog"),
    ("quantum physics", "quantum mechanics"),
    ("the the the", "the dog"),  # Common words matter less
]

for a, b in comparisons:
    sim = tfidf.similarity(a, b)
    print(f"  '{a}' vs '{b}': {sim:.3f}")

# %% [markdown]
# ### 5C: Phonetic Matching
#
# *"When spelling doesn't matter, but pronunciation does"*
#
# Phonetic algorithms encode words by their sound, not spelling.

# %%
print("Soundex (classic 4-character code):")
print()

names = ["Robert", "Rupert", "Roberta", "Smith", "Smyth", "Schmidt"]

for name in names:
    code = fr.soundex(name)
    print(f"  {name:12} -> {code}")

print()
print("  Matching:")
print(f"    Robert == Rupert? {fr.soundex_match('Robert', 'Rupert')}")
print(f"    Smith == Smyth?   {fr.soundex_match('Smith', 'Smyth')}")
print(f"    Similarity:       {fr.soundex_similarity('Robert', 'Rupert'):.2f}")

# %%
print("\nMetaphone (more accurate, variable length):")
print()

words = ["Smith", "Schmidt", "Smythe", "Wright", "Right", "Rite"]

for word in words:
    code = fr.metaphone(word)
    print(f"  {word:12} -> {code}")

print()
print("  Wright == Right?", fr.metaphone_match("Wright", "Right"))
print("  Metaphone similarity:", fr.metaphone_similarity("Wright", "Right"))

# %% [markdown]
# ### Use Case: Voice Transcription Errors
#
# When names come from speech-to-text, phonetic matching saves the day:

# %%
print("\nVoice Transcription Matching:")
print()

# Database of names
database = ["Catherine", "Katherine", "Kathryn", "Michael", "Mikhail", "Sean", "Shawn"]

# Transcription attempt
heard = "Katrin"

print(f"  Heard: '{heard}'")
print("  Phonetic matches:")

for name in database:
    phonetic_sim = fr.metaphone_similarity(heard, name)
    if phonetic_sim > 0.5:  # Phonetically similar
        text_sim = fr.jaro_winkler_similarity(heard, name)
        print(f"    {name} (phonetic: {phonetic_sim:.2f}, text: {text_sim:.2f})")

# %% [markdown]
# ---
# ## Part 6: Production Patterns
#
# *"Real-world strategies for real-world data"*
#
# Now that you know the algorithms and indices, let's talk about how to use them
# effectively in production.

# %% [markdown]
# ### 6A: Preprocessing with normalize_pair()
#
# Before comparing strings, normalize them consistently. `normalize_pair()` applies
# the same transformation to both strings.

# %%
print("normalize_pair() - Consistent Preprocessing:")
print()

# Raw input from different sources
raw_a = "  CAFE  "
raw_b = "cafe"

print(f"  Raw inputs: '{raw_a}' vs '{raw_b}'")
print()

# Without normalization: different
print("  Without normalization:")
print(f"    jaro_winkler: {fr.jaro_winkler_similarity(raw_a, raw_b):.3f}")

# With normalize_pair: identical
a_norm, b_norm = fr.normalize_pair(raw_a, raw_b, mode="strict")
print("  With normalize_pair (mode='strict'):")
print(f"    Normalized: '{a_norm}' vs '{b_norm}'")
print(f"    jaro_winkler: {fr.jaro_winkler_similarity(a_norm, b_norm):.3f}")

# %% [markdown]
# ### 6B: Choosing the Right Algorithm
#
# | Use Case | Recommended Algorithm |
# |----------|----------------------|
# | **Names** | Jaro-Winkler (prefix bonus helps) |
# | **Typos** | Damerau-Levenshtein (handles swaps) |
# | **Long text** | N-gram or Cosine (token-based) |
# | **Sound-alike** | Soundex or Metaphone |
# | **Exact edit count** | Levenshtein |
# | **Large-scale search** | HybridIndex + Jaro-Winkler |
# | **Multi-field records** | SchemaIndex |
#
# ### Normalization Strategy Guide
#
# | Scenario | Where to Normalize | Mode | Why |
# |----------|-------------------|------|-----|
# | User search | Both index & query | `strict` | Users type inconsistently |
# | Data deduplication | Records only | `strict` | Consistent comparison basis |
# | Exact lookup | Neither | N/A | Preserve original case |
# | Mixed data sources | Both | `nfkc_casefold` | Handle encoding differences |
# | Code/IDs | Whitespace only | `remove_whitespace` | Preserve case sensitivity |

# %%
print("Normalization in Index Construction:")
print()

# Example: Index with normalization vs without
records = ["iPhone 15 Pro", "IPHONE 15 PRO", "iphone-15-pro", "IPhone 15 Pro"]

# With normalize=True, searches are case-insensitive
idx_normalized = fr.NgramIndex(ngram_size=2, normalize=True)
idx_normalized.add_all(records)

# Without normalize, case matters
idx_raw = fr.NgramIndex(ngram_size=2, normalize=False)
idx_raw.add_all(records)

query = "iphone 15"
print(f"  Query: '{query}'")
print(
    f"  Normalized index: {len(idx_normalized.search(query, algorithm='jaro_winkler', min_similarity=0.7))} matches"
)
print(
    f"  Raw index:        {len(idx_raw.search(query, algorithm='jaro_winkler', min_similarity=0.7))} matches"
)

# %% [markdown]
# ### Batch Processing Patterns
#
# For multiple queries, use batch operations instead of loops.

# %%
print("\nBatch Processing - Good vs Bad:")
print()

# Sample data
catalog = [f"Product-{i:04d}" for i in range(1000)]
queries = ["Product-0100", "Product-0500", "Product-0999"]

batch_index = fr.HybridIndex(ngram_size=2)
batch_index.add_all(catalog)

# BAD: Individual searches in a loop (shows the pattern to avoid)
print("  Pattern to AVOID (loop with individual searches):")
print("    for q in queries:")
print("        results = index.search(q, ...)  # Creates overhead per query")
print()

# GOOD: Batch search
print("  BETTER (batch_search):")
start = time.time()
all_results = batch_index.batch_search(queries, algorithm="jaro_winkler", min_similarity=0.9)
elapsed = time.time() - start

print(f"    Processed {len(queries)} queries in {elapsed * 1000:.1f}ms")
for query, results in zip(queries, all_results):
    match_text = results[0].text if results else "No match"
    print(f"    '{query}' -> {match_text}")

# %% [markdown]
# ### Choosing the Right Index
#
# | Dataset Size | Recommended | Why |
# |-------------|-------------|-----|
# | < 1,000 | `find_best_matches()` | Index overhead not worth it |
# | 1K - 100K | `NgramIndex` | Fast candidate filtering |
# | 100K - 1M | `HybridIndex` | Best speed/accuracy balance |
# | > 1M | `HybridIndex` + SNM dedup | Memory-efficient for sorted data |
#
# **Rule of thumb**: If you're calling `search()` in a loop, use an index.

# %%
print("\nIndex Selection Examples:")
print()

# Small dataset: direct comparison is fine
small_data = ["apple", "banana", "cherry", "date", "elderberry"]
print(f"  Small dataset ({len(small_data)} items):")
matches = fr.batch.best_matches(small_data, "aple", algorithm="jaro_winkler", limit=2)
print(f"    batch.best_matches: {[m.text for m in matches]}")

# Large dataset: use index
print(f"\n  Large dataset ({len(catalog):,} items):")
print("    Use HybridIndex for repeated searches")
print(f"    Index size: {len(batch_index):,} items")

# %% [markdown]
# ### 6C: Thread Safety
#
# Index classes (BkTree, NgramIndex, HybridIndex, SchemaIndex) are **NOT thread-safe**.
#
# For multi-threaded applications:
# - Create one index per thread, OR
# - Use a lock when accessing shared index, OR
# - Build index once, then use read-only (search is safe if no concurrent writes)
#
# ```python
# # Example with threading.Lock
# import threading
#
# index = fr.HybridIndex(ngram_size=2)
# index.add_all(data)  # Build once
# lock = threading.Lock()
#
# def search_thread(query):
#     with lock:
#         return index.search(query, ...)
# ```

# %% [markdown]
# ---
# ## Part 7: Enterprise Features
#
# *"Production-ready matching"*

# %% [markdown]
# ### 7A: Schema-Based Multi-Field Matching
#
# Match records with multiple fields, each using the optimal algorithm.

# %%
print("Schema-Based Matching:")
print()

# Define a customer matching schema
schema_builder = fr.SchemaBuilder()

# Name field: short text, use Jaro-Winkler, highest weight
schema_builder.add_field(
    name="name",
    field_type="short_text",
    algorithm="jaro_winkler",
    weight=10.0,
    required=True,
)

# Email field: exact-ish matching
schema_builder.add_field(
    name="email",
    field_type="short_text",
    algorithm="levenshtein",
    weight=8.0,
)

# Tags field: set-based matching
schema_builder.add_field(
    name="tags",
    field_type="token_set",
    separator=",",
    weight=5.0,
)

schema = schema_builder.build()
print(f"  Schema fields: {schema.field_names()}")

# %%
# Create index and add records
customer_index = fr.SchemaIndex(schema)

customers = [
    {"name": "John Smith", "email": "jsmith@email.com", "tags": "premium,active"},
    {"name": "Jane Doe", "email": "jdoe@email.com", "tags": "basic,new"},
    {"name": "Jon Smith", "email": "jonsmith@mail.com", "tags": "premium,loyal"},
    {"name": "John Smyth", "email": "john.smyth@email.com", "tags": "active"},
]

for i, customer in enumerate(customers):
    customer_index.add(customer, data=i)

print(f"  Indexed {len(customer_index)} customers")

# %%
# Search with multiple fields
print("\nMulti-field search:")

query = {"name": "John Smith", "email": "jsmith@email.com"}
results = customer_index.search(query, min_similarity=0.3, limit=3)

print(f"  Query: {query}")
print("  Results:")
for r in results:
    print(f"    [{r.score:.0%}] {r.record['name']} ({r.record['email']})")
    print(
        f"         name: {r.field_scores.get('name', 0):.0%}, email: {r.field_scores.get('email', 0):.0%}"
    )

# %% [markdown]
# ### Field-Level Filtering with min_field_similarity
#
# Filter out results where any field scores too low.

# %%
print("\nField-Level Filtering:")
print()

# Without min_field_similarity: might match on name but terrible email
print("  Without min_field_similarity:")
query_partial = {"name": "John Smith", "email": "totally_different@xyz.com"}
results_all = customer_index.search(query_partial, min_similarity=0.3, limit=2)
for r in results_all:
    print(f"    [{r.score:.0%}] {r.record['name']} (email: {r.field_scores.get('email', 0):.0%})")

# With min_field_similarity: require EACH field to meet threshold
print("\n  With min_field_similarity=0.4 (each field must score >= 40%):")
results_filtered = customer_index.search(
    query_partial,
    min_similarity=0.3,
    min_field_similarity=0.4,  # Each field must score at least 40%
    limit=2,
)
if results_filtered:
    for r in results_filtered:
        print(f"    [{r.score:.0%}] {r.record['name']}")
else:
    print("    No results (email field too different)")

# %% [markdown]
# ### Scoring Strategies
#
# Control how field scores combine into the final score:
#
# - `sum` (default): Simple sum of weighted field scores
# - `weighted`: Normalized weighted average
# - `minmax_scaling`: Normalize scores to 0-1 range first
#
# ```python
# schema = (fr.SchemaBuilder()
#     .add_field("name", "short_text", weight=2.0)
#     .add_field("email", "short_text", weight=1.0)
#     .with_scoring("weighted")  # Use weighted average
#     .build())
# ```

# %% [markdown]
# ### 7B: Deduplication with Clustering
#
# Find duplicate groups using Union-Find graph clustering.

# %%
print("\nDeduplication with Clustering:")
print()

# Messy company data
companies = [
    "Apple Inc",
    "Apple Inc.",
    "APPLE INC",
    "Apple, Inc.",
    "Microsoft Corporation",
    "Microsoft Corp",
    "Microsoft Corp.",
    "MSFT Corporation",
    "Google LLC",
    "Google Inc",
    "Alphabet Inc",  # Different company, should NOT merge with Google
]

# Find duplicates with normalization
result = fr.batch.deduplicate(
    companies,
    algorithm="jaro_winkler",
    min_similarity=0.85,
    normalize="strict",  # lowercase + punctuation removal
)

print(f"  Input: {len(companies)} records")
print(f"  Duplicate groups: {len(result.groups)}")
for i, group in enumerate(result.groups, 1):
    print(f"    Group {i}: {group}")
print(f"  Unique records: {result.unique}")
print(f"  Total duplicates: {result.total_duplicates}")

# %%
print("\nDeduplication Methods:")
print()

# For large sorted datasets, use SNM (Sorted Neighborhood Method)
# Note: The batch.deduplicate function automatically uses SNM for large datasets (>2000 items).
# For manual control of SNM with Polars DataFrames, use frp.df_dedupe_snm().
result_snm = fr.batch.deduplicate(
    sorted(companies),  # Must be sorted for best SNM results
    min_similarity=0.9,
)
print(f"  Deduplication result: {len(result_snm.groups)} groups")

# %% [markdown]
# ### 7C: Evaluation Metrics
#
# Measure how well your matching works with precision, recall, and F-score.

# %%
print("\nEvaluation Metrics:")
print()

# Ground truth: which record pairs are actual duplicates (as tuples)
true_matches = [
    (0, 1),
    (0, 2),
    (0, 3),  # Records 0,1,2,3 are all duplicates of each other
    (1, 2),
    (1, 3),
    (2, 3),
    (4, 5),
    (4, 6),
    (5, 6),  # Records 4,5,6 are duplicates
]

# What our algorithm predicted as matches
predicted_matches = [
    (0, 1),
    (0, 2),
    (0, 3),  # Correct: TP
    (1, 2),
    (1, 3),
    (2, 3),  # Correct: TP
    (4, 5),
    (4, 6),  # Correct: TP (but missed 5,6: FN)
    (7, 8),  # Wrong: FP (not actually duplicates)
]

# Calculate metrics using pair lists
prec = fr.precision(true_matches, predicted_matches)
rec = fr.recall(true_matches, predicted_matches)
f1 = fr.f_score(true_matches, predicted_matches, beta=1.0)
f2 = fr.f_score(true_matches, predicted_matches, beta=2.0)

print(f"  True matches:      {len(true_matches)} pairs")
print(f"  Predicted matches: {len(predicted_matches)} pairs")
print()
print(f"  Precision: {prec:.0%} (of predicted matches, how many are correct)")
print(f"  Recall:    {rec:.0%} (of actual matches, how many did we find)")
print(f"  F1 Score:  {f1:.0%} (harmonic mean of precision & recall)")
print(f"  F2 Score:  {f2:.0%} (recall-weighted, favors finding more matches)")

# %%
print("\nConfusion Matrix (from match pair lists):")
print()

# Total possible pairs for 10 records: 10 * 9 / 2 = 45
total_pairs = 45
cm = fr.confusion_matrix(true_matches, predicted_matches, total_pairs)

print(f"  Total possible pairs: {total_pairs}")
print(f"  TP: {cm.tp}, FP: {cm.fp}, FN: {cm.fn_count}, TN: {cm.tn}")
print()
print(f"  Precision: {cm.precision():.0%}")
print(f"  Recall:    {cm.recall():.0%}")
print(f"  F1:        {cm.f_score():.0%}")

# %% [markdown]
# ---
# ## Part 8: Edge Cases & Best Practices
#
# *"The gotchas and pro tips"*

# %% [markdown]
# ### Normalization Modes
#
# Preprocess strings for consistent comparison.

# %%
print("String Normalization:")
print()

text = "Hello, World! 123   Cafe"
print(f"  Original: '{text}'")
print()

modes = ["lowercase", "remove_punctuation", "remove_whitespace", "unicode_nfkd", "strict"]

for mode in modes:
    normalized = fr.normalize_string(text, mode)
    print(f"  {mode:20}: '{normalized}'")

# %% [markdown]
# ### Case-Insensitive Variants
#
# Every similarity function has a `_ci` suffix variant.

# %%
print("\nCase-Insensitive Comparison (normalize parameter):")
print()

a, b = "HELLO", "hello"
print(f"  Comparing '{a}' and '{b}':")
print()
print(f"  levenshtein:                    {fr.levenshtein(a, b)} edits")
print(f"  levenshtein (normalize=lower):  {fr.levenshtein(a, b, normalize='lowercase')} edits")
print()
print(f"  jaro_winkler:                   {fr.jaro_winkler_similarity(a, b):.3f}")
print(
    f"  jaro_winkler (normalize=lower): {fr.jaro_winkler_similarity(a, b, normalize='lowercase'):.3f}"
)
print()
print(f"  ngram_similarity:                   {fr.ngram_similarity(a, b):.3f}")
print(
    f"  ngram_similarity (normalize=lower): {fr.ngram_similarity(a, b, normalize='lowercase'):.3f}"
)

# %% [markdown]
# ### Unicode Handling
#
# FuzzyRust handles Unicode correctly: emoji, accents, CJK, etc.

# %%
print("\nUnicode Handling:")
print()

# Emoji
emoji1, emoji2 = "hello 👋", "hello 🖐️"
print(f"  Emoji: levenshtein('{emoji1}', '{emoji2}') = {fr.levenshtein(emoji1, emoji2)}")

# Accents
accent1, accent2 = "café", "cafe"
print(f"  Accents: levenshtein('{accent1}', '{accent2}') = {fr.levenshtein(accent1, accent2)}")

# CJK (Chinese)
cjk1, cjk2 = "東京", "东京"
print(f"  CJK: levenshtein('{cjk1}', '{cjk2}') = {fr.levenshtein(cjk1, cjk2)}")

# RTL (Hebrew)
hebrew1, hebrew2 = "שלום", "שלומ"
print(f"  Hebrew: levenshtein('{hebrew1}', '{hebrew2}') = {fr.levenshtein(hebrew1, hebrew2)}")

# %% [markdown]
# ### Algorithm Selection Guide
#
# | Use Case | Recommended Algorithm |
# |----------|----------------------|
# | **Names** | Jaro-Winkler (prefix bonus helps) |
# | **Typos** | Damerau-Levenshtein (handles swaps) |
# | **Long text** | N-gram or Cosine (token-based) |
# | **Sound-alike** | Soundex or Metaphone |
# | **Exact edit count** | Levenshtein |
# | **Large-scale search** | HybridIndex + Jaro-Winkler |
# | **Multi-field records** | SchemaIndex |

# %% [markdown]
# ---
# ## Appendix: RapidFuzz Compatibility
#
# *"Easy migration from other libraries"*
#
# FuzzyRust provides drop-in replacements for common RapidFuzz functions.

# %%
print("RapidFuzz-Compatible Functions:")
print()

# partial_ratio: Best match within a substring
print("partial_ratio (substring matching):")
print("  'The Dark Knight' in 'Batman: The Dark Knight Rises'")
print(f"  Score: {fr.partial_ratio('The Dark Knight', 'Batman: The Dark Knight Rises'):.0%}")
print()

# token_sort_ratio: Word order doesn't matter
print("token_sort_ratio (word order independent):")
print("  'New York City' vs 'City of New York'")
print(f"  Regular:    {fr.ratio('New York City', 'City of New York'):.0%}")
print(f"  Token sort: {fr.token_sort_ratio('New York City', 'City of New York'):.0%}")
print()

# token_set_ratio: Duplicate words don't matter
print("token_set_ratio (ignores duplicates):")
print("  'the the quick' vs 'quick'")
print(f"  Token set: {fr.token_set_ratio('the the quick', 'quick'):.0%}")
print()

# wratio: Automatic best method selection
print("wratio (automatic method selection):")
print("  'fuzzy wuzzy was a bear' vs 'wuzzy fuzzy was he'")
print(f"  Score: {fr.wratio('fuzzy wuzzy was a bear', 'wuzzy fuzzy was he'):.0%}")

# %%
print("\nextract / extract_one (find matches from list):")
print()

choices = ["Atlanta Falcons", "New York Jets", "New York Giants", "Dallas Cowboys"]
query = "new york"

# Find best single match (query first, then choices)
best = fr.extract_one(query, choices)
if best:
    print(f"  Best match for '{query}': '{best.text}' ({best.score:.0%})")

# Find top matches
print("  Top 3 matches:")
matches = fr.extract(query, choices, limit=3)
for m in matches:
    print(f"    [{m.score:.0%}] {m.text}")

# %% [markdown]
# ---
# ## Summary
#
# **This Guide Covered (8 Parts + Appendix):**
#
# | Part | Topic | Key Features |
# |------|-------|--------------|
# | 1 | The Hook | Movie search, dedup, autocomplete |
# | 2 | Polars Power | DataFrame fuzzy join, dedup, expression namespace, FuzzyIndex |
# | 3 | Search at Scale | BkTree, NgramIndex, HybridIndex, batch operations, user data |
# | 4 | Similarity Fundamentals | Levenshtein, Damerau, Hamming, Jaro-Winkler, `_ci` variants |
# | 5 | Advanced Algorithms | N-grams, LCS, cosine, TF-IDF, Soundex, Metaphone |
# | 6 | Production Patterns | `normalize_pair()`, batch patterns, index selection, thread safety |
# | 7 | Enterprise Features | Schema matching, `min_field_similarity`, deduplication, metrics |
# | 8 | Edge Cases | Unicode, normalization modes, algorithm guide |
# | Appendix | RapidFuzz | `partial_ratio`, `token_sort_ratio`, `extract` |
#
# **Functions by Category:**
#
# | Category | Functions |
# |----------|-----------|
# | Edit Distance | `levenshtein`, `damerau_levenshtein`, `hamming` (+ `_ci` variants) |
# | Similarity | `jaro_similarity`, `jaro_winkler_similarity` (+ `_ci` variants) |
# | N-gram | `ngram_similarity`, `ngram_jaccard`, `ngram_profile_similarity` |
# | Sequence | `lcs_string`, `lcs_similarity`, `longest_common_substring` |
# | Cosine | `cosine_similarity_chars/words/ngrams`, `TfIdfCosine` |
# | Phonetic | `soundex`, `metaphone` + `_match`, `_similarity` |
# | RapidFuzz | `partial_ratio`, `token_sort_ratio`, `wratio`, `extract` |
# | Batch | `batch.similarity`, `batch.best_matches`, `batch.deduplicate` |
# | Indices | `BkTree`, `NgramIndex`, `HybridIndex` + `add_with_data`, `get_candidates` |
# | Schema | `SchemaBuilder`, `SchemaIndex` + `min_field_similarity` |
# | Metrics | `precision`, `recall`, `f_score`, `confusion_matrix` |
# | Utilities | `normalize_string`, `normalize_pair`, `extract_ngrams` |
# | Polars | `frp.df_join`, `frp.df_dedupe`, `frp.df_match_pairs`, `.fuzzy` namespace |
#
# **Why FuzzyRust?**
#
# - **Fast**: Rust core with Rayon parallelism
# - **Complete**: 95+ functions covering all use cases
# - **Production-ready**: User data association, batch operations, thread-safe patterns
# - **Compatible**: RapidFuzz API for easy migration
# - **DataFrame-native**: First-class Polars integration
#
# **Get Started:**
# ```bash
# pip install fuzzyrust
# ```
