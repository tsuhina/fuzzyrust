#!/usr/bin/env python3
"""
FuzzyRust Benchmark Suite
=========================

Comprehensive benchmarks comparing FuzzyRust with other Python fuzzy matching libraries.

REQUIRES benchmark dependencies:
    pip install fuzzyrust[benchmarks]
    # or
    uv sync --extra benchmarks

Run benchmarks:
    python examples/benchmarks.py
    # or with uv
    uv run python examples/benchmarks.py
"""

import random
import string
import sys
import time
from dataclasses import dataclass
from typing import Callable

import fuzzyrust as fr


def check_dependencies():
    """Verify all benchmark dependencies are installed."""
    missing = []

    try:
        import rapidfuzz  # noqa: F401
    except ImportError:
        missing.append("rapidfuzz")

    try:
        import jellyfish  # noqa: F401
    except ImportError:
        missing.append("jellyfish")

    try:
        from thefuzz import fuzz  # noqa: F401
    except ImportError:
        missing.append("thefuzz")

    if missing:
        print("ERROR: Missing benchmark dependencies:", ", ".join(missing))
        print()
        print("Install with:")
        print("    pip install fuzzyrust[benchmarks]")
        print("    # or")
        print("    uv sync --extra benchmarks")
        sys.exit(1)


check_dependencies()

# Import benchmark dependencies (guaranteed available after check)
import jellyfish
from rapidfuzz import distance as rf_distance
from rapidfuzz import fuzz as rf_fuzz
from rapidfuzz import process as rf_process
from thefuzz import fuzz as tf_fuzz

# =============================================================================
# Benchmark Infrastructure
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""

    name: str
    time_seconds: float
    iterations: int
    items_processed: int = 0

    @property
    def ops_per_second(self) -> float:
        if self.items_processed > 0:
            return self.items_processed / self.time_seconds
        return self.iterations / self.time_seconds


def benchmark(
    name: str,
    func: Callable,
    iterations: int = 1,
    warmup: int = 1,
    items_per_iteration: int = 0,
) -> BenchmarkResult:
    """Run a benchmark with warmup and timing."""
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name=name,
        time_seconds=elapsed,
        iterations=iterations,
        items_processed=items_per_iteration * iterations if items_per_iteration else 0,
    )


def format_time(seconds: float) -> str:
    """Format time with appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}us"
    elif seconds < 1:
        return f"{seconds * 1_000:.2f}ms"
    else:
        return f"{seconds:.3f}s"


def format_throughput(ops: float) -> str:
    """Format operations per second."""
    if ops >= 1_000_000:
        return f"{ops / 1_000_000:.2f}M/s"
    elif ops >= 1_000:
        return f"{ops / 1_000:.1f}K/s"
    else:
        return f"{ops:.0f}/s"


def print_results(results: list[BenchmarkResult], baseline_name: str = "fuzzyrust"):
    """Print benchmark results with speedup ratios."""
    # Find baseline
    baseline = next((r for r in results if r.name == baseline_name), results[0])

    # Sort by time (fastest first)
    sorted_results = sorted(results, key=lambda r: r.time_seconds)

    for result in sorted_results:
        speedup = (
            baseline.time_seconds / result.time_seconds if result.time_seconds > 0 else float("inf")
        )
        time_str = format_time(result.time_seconds)

        if result.items_processed > 0:
            throughput = format_throughput(result.ops_per_second)
            print(f"  {result.name:20s} {time_str:>10s}  ({throughput:>10s})  {speedup:.2f}x")
        else:
            print(f"  {result.name:20s} {time_str:>10s}  {speedup:.2f}x")


# =============================================================================
# Test Data Generation
# =============================================================================


def generate_strings(count: int, min_len: int = 5, max_len: int = 20, seed: int = 42) -> list[str]:
    """Generate random strings for benchmarking."""
    random.seed(seed)
    return [
        "".join(random.choices(string.ascii_lowercase, k=random.randint(min_len, max_len)))
        for _ in range(count)
    ]


def generate_names(count: int, seed: int = 42) -> list[str]:
    """Generate realistic-looking names for benchmarking."""
    random.seed(seed)
    first_names = [
        "John",
        "Jane",
        "Michael",
        "Sarah",
        "David",
        "Emily",
        "Robert",
        "Lisa",
        "William",
        "Jennifer",
        "James",
        "Maria",
        "Thomas",
        "Patricia",
        "Charles",
    ]
    last_names = [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Garcia",
        "Miller",
        "Davis",
        "Rodriguez",
        "Martinez",
        "Anderson",
        "Taylor",
        "Thomas",
        "Moore",
    ]

    return [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(count)]


# =============================================================================
# Benchmark: Single String Comparison
# =============================================================================


def benchmark_single_string():
    """Compare single string pair performance."""
    print("\n" + "=" * 70)
    print("SINGLE STRING COMPARISON")
    print("=" * 70)

    s1 = "kitten sitting on the comfortable couch"
    s2 = "sitting kittens on a comfortable sofa"
    iterations = 100_000

    # --- Levenshtein Distance ---
    print(f"\nLevenshtein Distance ({iterations:,} iterations)")
    print(f"  Strings: '{s1[:30]}...' vs '{s2[:30]}...'")
    print()

    results = []

    # FuzzyRust
    results.append(
        benchmark(
            "fuzzyrust",
            lambda: fr.levenshtein(s1, s2),
            iterations=iterations,
        )
    )

    # RapidFuzz
    results.append(
        benchmark(
            "rapidfuzz",
            lambda: rf_distance.Levenshtein.distance(s1, s2),
            iterations=iterations,
        )
    )

    # thefuzz (uses python-Levenshtein internally)
    results.append(
        benchmark(
            "thefuzz",
            lambda: tf_fuzz.ratio(s1, s2),
            iterations=iterations,
        )
    )

    # Jellyfish
    results.append(
        benchmark(
            "jellyfish",
            lambda: jellyfish.levenshtein_distance(s1, s2),
            iterations=iterations,
        )
    )

    print_results(results)

    # --- Jaro-Winkler Similarity ---
    print(f"\nJaro-Winkler Similarity ({iterations:,} iterations)")
    print()

    results = []

    # FuzzyRust
    results.append(
        benchmark(
            "fuzzyrust",
            lambda: fr.jaro_winkler_similarity(s1, s2),
            iterations=iterations,
        )
    )

    # RapidFuzz
    results.append(
        benchmark(
            "rapidfuzz",
            lambda: rf_distance.JaroWinkler.similarity(s1, s2),
            iterations=iterations,
        )
    )

    # thefuzz doesn't have direct Jaro-Winkler

    # Jellyfish
    results.append(
        benchmark(
            "jellyfish",
            lambda: jellyfish.jaro_winkler_similarity(s1, s2),
            iterations=iterations,
        )
    )

    print_results(results)


# =============================================================================
# Benchmark: Batch Processing (Large Scale)
# =============================================================================


def benchmark_batch_processing():
    """Compare batch/bulk processing performance."""
    print("\n" + "=" * 70)
    print("BATCH PROCESSING (FuzzyRust's Rayon Parallelism)")
    print("=" * 70)

    query = "John Smith"

    for size in [10_000, 100_000, 1_000_000]:
        print(f"\n--- {size:,} strings ---")
        strings = generate_names(size)

        results = []

        # FuzzyRust batch (parallel)
        results.append(
            benchmark(
                "fuzzyrust (parallel)",
                lambda: fr.batch_jaro_winkler(strings, query),
                iterations=1,
                warmup=1,
                items_per_iteration=size,
            )
        )

        # FuzzyRust find_best_matches
        results.append(
            benchmark(
                "fuzzyrust best_match",
                lambda: fr.find_best_matches(strings, query, algorithm="jaro_winkler", limit=10),
                iterations=1,
                warmup=1,
                items_per_iteration=size,
            )
        )

        # RapidFuzz
        def rapidfuzz_batch():
            return [rf_distance.JaroWinkler.similarity(s, query) for s in strings]

        results.append(
            benchmark(
                "rapidfuzz (loop)",
                rapidfuzz_batch,
                iterations=1,
                warmup=1,
                items_per_iteration=size,
            )
        )

        # RapidFuzz extract (their batch API)
        results.append(
            benchmark(
                "rapidfuzz extract",
                lambda: rf_process.extract(query, strings, scorer=rf_fuzz.WRatio, limit=10),
                iterations=1,
                warmup=1,
                items_per_iteration=size,
            )
        )

        # Jellyfish (no batch API, pure loop) - skip 1M for slow libs
        if size <= 100_000:

            def jellyfish_batch():
                return [jellyfish.jaro_winkler_similarity(s, query) for s in strings]

            results.append(
                benchmark(
                    "jellyfish (loop)",
                    jellyfish_batch,
                    iterations=1,
                    warmup=1,
                    items_per_iteration=size,
                )
            )

        print_results(results)


# =============================================================================
# Benchmark: Phonetic Algorithms
# =============================================================================


def benchmark_phonetic():
    """Compare phonetic algorithm performance."""
    print("\n" + "=" * 70)
    print("PHONETIC ALGORITHMS")
    print("=" * 70)

    names = generate_names(10_000)
    iterations = 10

    # --- Soundex ---
    print(f"\nSoundex Encoding ({len(names):,} names x {iterations} iterations)")
    print()

    results = []

    # FuzzyRust
    def fuzzyrust_soundex():
        return [fr.soundex(name) for name in names]

    results.append(
        benchmark(
            "fuzzyrust",
            fuzzyrust_soundex,
            iterations=iterations,
            items_per_iteration=len(names),
        )
    )

    # Jellyfish
    def jellyfish_soundex():
        return [jellyfish.soundex(name) for name in names]

    results.append(
        benchmark(
            "jellyfish",
            jellyfish_soundex,
            iterations=iterations,
            items_per_iteration=len(names),
        )
    )

    print_results(results)

    # --- Metaphone ---
    print(f"\nMetaphone Encoding ({len(names):,} names x {iterations} iterations)")
    print()

    results = []

    # FuzzyRust
    def fuzzyrust_metaphone():
        return [fr.metaphone(name) for name in names]

    results.append(
        benchmark(
            "fuzzyrust",
            fuzzyrust_metaphone,
            iterations=iterations,
            items_per_iteration=len(names),
        )
    )

    # Jellyfish
    def jellyfish_metaphone():
        return [jellyfish.metaphone(name) for name in names]

    results.append(
        benchmark(
            "jellyfish",
            jellyfish_metaphone,
            iterations=iterations,
            items_per_iteration=len(names),
        )
    )

    print_results(results)


# =============================================================================
# Benchmark: Index Structures (FuzzyRust Exclusive)
# =============================================================================


def benchmark_index_structures():
    """Demonstrate FuzzyRust's unique index capabilities."""
    print("\n" + "=" * 70)
    print("INDEX STRUCTURES (FuzzyRust Exclusive Features)")
    print("=" * 70)
    print("\nThese features are NOT available in rapidfuzz, thefuzz, or jellyfish.")

    # --- BK-Tree ---
    print("\n--- BK-Tree: Edit Distance Index ---")

    for size in [10_000, 100_000, 1_000_000]:
        print(f"\n{size:,} entries:")
        strings = generate_strings(size, min_len=8, max_len=15)

        # Build
        bktree = fr.BkTree()
        start = time.perf_counter()
        for s in strings:
            bktree.add(s)
        build_time = time.perf_counter() - start
        print(f"  Build time:  {format_time(build_time)}")

        # Search
        query = strings[size // 2][:5] + "xxx"  # Partial match
        search_times = []
        for _ in range(100):
            start = time.perf_counter()
            results = bktree.search(query, max_distance=2)
            search_times.append(time.perf_counter() - start)

        avg_search = sum(search_times) / len(search_times)
        print(f"  Search time: {format_time(avg_search)} (avg of 100 queries, max_dist=2)")
        print(f"  Results:     {len(results)} matches")

        # Compare with brute force (only for smaller sizes)
        if size <= 100_000:
            start = time.perf_counter()
            brute_results = [s for s in strings if fr.levenshtein(query, s) <= 2]
            brute_time = time.perf_counter() - start
            speedup = brute_time / avg_search
            print(f"  Brute force: {format_time(brute_time)}")
            print(f"  Speedup:     {speedup:.0f}x faster than brute force")

    # --- N-gram Index ---
    print("\n--- N-gram Index: Fuzzy Search ---")

    for size in [10_000, 100_000, 1_000_000]:
        print(f"\n{size:,} entries:")
        strings = generate_names(size)

        # Build
        ngram_idx = fr.NgramIndex(ngram_size=2, min_ngram_ratio=0.2)
        start = time.perf_counter()
        for s in strings:
            ngram_idx.add(s)
        build_time = time.perf_counter() - start
        print(f"  Build time:  {format_time(build_time)}")

        # Search
        query = "Jon Smyth"
        search_times = []
        for _ in range(100):
            start = time.perf_counter()
            results = ngram_idx.search(
                query, algorithm="jaro_winkler", min_similarity=0.7, limit=10
            )
            search_times.append(time.perf_counter() - start)

        avg_search = sum(search_times) / len(search_times)
        print(f"  Search time: {format_time(avg_search)} (avg of 100 queries)")
        print(f"  Results:     {len(results)} matches")

    # --- Hybrid Index with Batch Search ---
    print("\n--- Hybrid Index: Parallel Batch Search ---")

    size = 100_000
    strings = generate_names(size)
    queries = generate_names(1_000, seed=99)

    hybrid = fr.HybridIndex(ngram_size=2, normalize=True)
    for s in strings:
        hybrid.add(s)

    start = time.perf_counter()
    batch_results = hybrid.batch_search(
        queries, algorithm="jaro_winkler", min_similarity=0.7, limit=5
    )
    batch_time = time.perf_counter() - start

    print(f"\n{size:,} entries, {len(queries):,} queries:")
    print(f"  Batch search: {format_time(batch_time)}")
    print(f"  Throughput:   {format_throughput(len(queries) / batch_time)}")


# =============================================================================
# Benchmark: Deduplication
# =============================================================================


def benchmark_deduplication():
    """Compare deduplication performance."""
    print("\n" + "=" * 70)
    print("DEDUPLICATION (Graph-Based Clustering)")
    print("=" * 70)
    print("\nFuzzyRust uses Union-Find with path compression for efficient clustering.")

    for size in [1_000, 10_000, 50_000]:
        print(f"\n--- {size:,} items ---")

        # Generate data with duplicates
        base_names = generate_names(size // 5)
        items = []
        for name in base_names:
            items.append(name)
            # Add variations
            items.append(name.upper())
            items.append(name.lower())
            items.append(name.replace(" ", "  "))
            items.append(name + ".")

        items = items[:size]

        # FuzzyRust deduplication
        start = time.perf_counter()
        result = fr.find_duplicates(
            items,
            algorithm="jaro_winkler",
            min_similarity=0.85,
            normalize="strict",
        )
        elapsed = time.perf_counter() - start

        print(f"  Time:       {format_time(elapsed)}")
        print(f"  Groups:     {len(result.groups)}")
        print(f"  Unique:     {len(result.unique)}")
        print(f"  Duplicates: {result.total_duplicates}")
        print(f"  Throughput: {format_throughput(size / elapsed)}")


# =============================================================================
# Main
# =============================================================================


def print_library_status():
    """Print which libraries are being compared."""
    print("=" * 70)
    print("FuzzyRust Benchmark Suite")
    print("=" * 70)
    print("\nComparing libraries:")
    print("  fuzzyrust   (this library)")
    print("  rapidfuzz")
    print("  thefuzz")
    print("  jellyfish")


def main():
    print_library_status()

    benchmark_single_string()
    benchmark_batch_processing()
    benchmark_phonetic()
    benchmark_index_structures()
    benchmark_deduplication()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
FuzzyRust differentiators:

1. PERFORMANCE: Rust core with Rayon parallelism
   - GIL released during computation
   - Automatic multi-core utilization
   - Competitive with C++ rapidfuzz, faster than pure Python

2. INDEX STRUCTURES (exclusive to FuzzyRust):
   - BkTree: O(log n) edit distance queries
   - NgramIndex/HybridIndex: O(1) candidate filtering
   - Massive speedup over brute force for large datasets

3. ENTERPRISE FEATURES:
   - Multi-field schema matching with weighted fields
   - Graph-based deduplication with Union-Find clustering
   - TF-IDF corpus-aware similarity

4. DEVELOPER EXPERIENCE:
   - Full type hints for IDE support
   - Consistent API across all algorithms
   - Production-ready error handling
""")


if __name__ == "__main__":
    main()
