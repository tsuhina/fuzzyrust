"""
Industrial Spare Parts Matching Example

This example demonstrates how to use FuzzyRust for large-scale spare parts
matching with multiple fields (part_number, name, manufacturer, category).

Use cases:
- Search: Find parts matching a query across multiple fields
- Deduplication: Identify duplicate part entries in a catalog
- Record Linkage: Match parts between two catalogs (e.g., supplier vs internal)

Run with: uv run python examples/industrial_parts.py
"""

from __future__ import annotations

import random
import time

import polars as pl

import fuzzyrust as fr
from fuzzyrust import polars as frp

# =============================================================================
# Sample Data Generation
# =============================================================================


def generate_sample_parts(n: int = 10000) -> pl.DataFrame:
    """Generate sample spare parts data with realistic variations.

    Creates a dataset with intentional duplicates and variations to
    demonstrate fuzzy matching capabilities.

    Args:
        n: Base number of unique parts to generate.

    Returns:
        DataFrame with part_number, name, manufacturer, and category columns.
    """
    manufacturers = ["Bosch", "Siemens", "ABB", "Schneider", "Honeywell"]
    categories = ["Bearings", "Motors", "Sensors", "Valves", "Pumps"]

    parts = []
    for i in range(n):
        base_name = f"Industrial {categories[i % len(categories)]} Unit"
        # Add variations: typos, abbreviations, extra spaces
        name_variations = [
            base_name,
            base_name.replace("Industrial", "Ind."),
            base_name.replace("Unit", "Unt"),  # typo
            f"  {base_name}  ",  # extra spaces
        ]

        parts.append(
            {
                "part_number": f"PN-{i:06d}",
                "name": random.choice(name_variations),
                "manufacturer": manufacturers[i % len(manufacturers)],
                "category": categories[i % len(categories)],
            }
        )

    # Add intentional duplicates (10% of dataset)
    for i in range(n // 10):
        orig = parts[i].copy()
        orig["name"] = orig["name"].replace("a", "e")  # Simulate typo
        parts.append(orig)

    return pl.DataFrame(parts)


# =============================================================================
# Use Case 1: Multi-Field Search
# =============================================================================


def demo_search():
    """Demonstrate multi-field search with SchemaIndex."""
    print("\n" + "=" * 60)
    print("USE CASE 1: Multi-Field Search")
    print("=" * 60 + "\n")

    # Define schema with field-specific algorithms and weights
    builder = fr.SchemaBuilder()
    builder.add_field("part_number", field_type="short_text", algorithm="levenshtein", weight=3.0)
    builder.add_field("name", field_type="short_text", algorithm="jaro_winkler", weight=2.0)
    builder.add_field("manufacturer", field_type="short_text", algorithm="jaro_winkler", weight=1.5)
    builder.add_field("category", field_type="short_text", algorithm="jaccard", weight=1.0)
    schema = builder.build()

    # Generate sample data
    df = generate_sample_parts(10000)
    print(f"Dataset: {len(df):,} parts")

    # Build index
    start = time.time()
    index = fr.SchemaIndex(schema)
    for row in df.iter_rows(named=True):
        index.add(row)
    build_time = time.time() - start
    print(f"Index build time: {build_time:.2f}s")

    # Search example
    query = {
        "part_number": "PN-000100",
        "name": "Industrial Bearings Unt",  # typo
        "manufacturer": "Bosch",
    }

    start = time.time()
    results = index.search(query, min_similarity=0.7, limit=5)
    search_time = time.time() - start

    print(f"\nQuery: {query}")
    print(f"Search time: {search_time * 1000:.2f}ms")
    print("\nTop matches:")
    for r in results:
        print(f"  [{r.score:.0%}] {r.record}")

    # Batch search demonstration
    print("\n--- Batch Search (1000 queries) ---")
    queries = [
        {"part_number": f"PN-{i:06d}", "name": "Industrial", "manufacturer": "Bosch"}
        for i in range(0, 10000, 10)
    ]

    start = time.time()
    batch_results = index.batch_search(queries, min_similarity=0.6, limit=3)
    batch_time = time.time() - start

    print(f"Batch search: {len(queries)} queries in {batch_time:.2f}s")
    print(f"Throughput: {len(queries) / batch_time:.0f} queries/sec")


# =============================================================================
# Use Case 2: Deduplication
# =============================================================================


def demo_deduplication():
    """Demonstrate large-scale deduplication with SNM."""
    print("\n" + "=" * 60)
    print("USE CASE 2: Deduplication")
    print("=" * 60 + "\n")

    # Generate data with duplicates
    df = generate_sample_parts(50000)
    print(f"Dataset: {len(df):,} parts")

    # Method comparison: df_dedupe vs df_dedupe_snm
    print("\n--- Comparing Deduplication Methods ---\n")

    # For small subset, show df_dedupe (O(N^2))
    df_small = df.head(5000)

    start = time.time()
    result_dedupe = frp.df_dedupe(
        df_small,
        columns=["name", "manufacturer"],
        algorithms={"name": "jaro_winkler", "manufacturer": "jaro_winkler"},
        weights={"name": 2.0, "manufacturer": 1.0},
        min_similarity=0.85,
    )
    dedupe_time = time.time() - start

    groups = result_dedupe.filter(pl.col("_group_id").is_not_null())
    print(f"df_dedupe (5K rows): {dedupe_time:.2f}s, {len(groups)} duplicates found")

    # For full dataset, use df_dedupe_snm (O(N log N))
    start = time.time()
    result_snm = frp.df_dedupe_snm(
        df,
        columns=["name", "manufacturer"],
        algorithm="jaro_winkler",
        min_similarity=0.85,
        window_size=30,
    )
    snm_time = time.time() - start

    groups_snm = result_snm.filter(pl.col("_group_id").is_not_null())
    print(f"df_dedupe_snm (50K rows): {snm_time:.2f}s, {len(groups_snm)} duplicates found")

    # Get unique records
    unique = result_snm.filter(pl.col("_is_canonical"))
    print(f"\nUnique records after dedup: {len(unique):,}")

    # Show sample duplicate groups
    print("\nSample duplicate groups:")
    sample_groups = (
        result_snm.filter(pl.col("_group_id").is_not_null()).group_by("_group_id").head(2).head(10)
    )
    print(sample_groups)


# =============================================================================
# Use Case 3: Record Linkage
# =============================================================================


def demo_record_linkage():
    """Demonstrate matching between two catalogs."""
    print("\n" + "=" * 60)
    print("USE CASE 3: Record Linkage")
    print("=" * 60 + "\n")

    # Simulate two catalogs with overlapping parts
    internal_catalog = pl.DataFrame(
        {
            "internal_id": [f"INT-{i:04d}" for i in range(1000)],
            "name": [f"Part Type {i}" for i in range(1000)],
            "manufacturer": ["Bosch"] * 500 + ["Siemens"] * 500,
        }
    )

    # Supplier catalog with variations (same column names for df_match_records)
    supplier_catalog = pl.DataFrame(
        {
            "supplier_id": [f"SUP-{i:04d}" for i in range(800)],
            "name": [f"Part Typ {i}" for i in range(800)],  # typo in values
            "manufacturer": ["BOSCH"] * 400 + ["SIEMENS"] * 400,  # uppercase
        }
    )

    print(f"Internal catalog: {len(internal_catalog):,} parts")
    print(f"Supplier catalog: {len(supplier_catalog):,} parts")

    # Match using df_match_records with multi-column matching
    start = time.time()
    matched = frp.df_match_records(
        supplier_catalog,
        internal_catalog,
        columns=["name"],
        algorithms={"name": "jaro_winkler"},
        min_similarity=0.7,
    )
    match_time = time.time() - start

    print(f"\nMatching time: {match_time:.2f}s")
    print(f"Matched: {len(matched)} / {len(supplier_catalog)} supplier parts")

    # Show sample matches
    if len(matched) > 0:
        print("\nSample matches:")
        print(
            matched.select(
                ["supplier_id_query", "name_query", "internal_id_target", "name_target", "score"]
            ).head(5)
        )


# =============================================================================
# Performance Tips
# =============================================================================


def demo_performance_tips():
    """Demonstrate key performance optimization patterns."""
    print("\n" + "=" * 60)
    print("PERFORMANCE TIPS")
    print("=" * 60 + "\n")

    df = generate_sample_parts(10000)

    print("1. Pre-normalize data for consistent matching:")
    print("-" * 40)

    # Normalize before matching
    df_normalized = df.with_columns(
        [
            pl.col("name").str.to_lowercase().str.strip_chars().alias("name_norm"),
            pl.col("manufacturer").str.to_lowercase().alias("mfr_norm"),
        ]
    )
    print("  df.with_columns([pl.col('name').str.to_lowercase().str.strip_chars()])")
    print(f"  Normalized {len(df_normalized)} rows")

    print("\n2. Use blocking for very large datasets:")
    print("-" * 40)
    print("  # Process by category to reduce comparison space")
    print("  for category in df['category'].unique():")
    print("      subset = df.filter(pl.col('category') == category)")
    print("      result = frp.df_dedupe_snm(subset, ...)")

    print("\n3. Choose window_size based on dataset size:")
    print("-" * 40)
    print("  # 100K records: window_size=20")
    print("  # 1M records: window_size=50-100")
    print("  # 10M records: window_size=100-200 (with blocking)")

    print("\n4. Use batch operations instead of loops:")
    print("-" * 40)
    print("  # BAD: Loop with individual searches")
    print("  for query in queries:")
    print("      index.search(query)  # Creates overhead per query")
    print()
    print("  # GOOD: Single batch call")
    print("  index.batch_search(queries)  # Parallelized internally")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("FuzzyRust: Industrial Spare Parts Matching")
    print("==========================================")

    demo_search()
    demo_deduplication()
    demo_record_linkage()
    demo_performance_tips()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
