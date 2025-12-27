#!/usr/bin/env python3
"""
Accuracy Comparison: FuzzyRust vs Other Libraries

Compares algorithm outputs between FuzzyRust, RapidFuzz, and Jellyfish
to verify correctness and document any differences.

REQUIRES benchmark dependencies:
    pip install fuzzyrust[benchmarks]
    # or
    uv sync --extra benchmarks
"""

import sys
from typing import Any, Callable, Optional

import fuzzyrust as fr


def check_dependencies():
    """Verify all benchmark dependencies are installed."""
    missing = []

    try:
        from rapidfuzz import distance  # noqa: F401
    except ImportError:
        missing.append("rapidfuzz")

    try:
        import jellyfish  # noqa: F401
    except ImportError:
        missing.append("jellyfish")

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

# =============================================================================
# Test Data
# =============================================================================

TEST_PAIRS = [
    # Classic examples
    ("kitten", "sitting"),
    ("saturday", "sunday"),
    ("MARTHA", "MARHTA"),
    ("DWAYNE", "DUANE"),
    # Transpositions (important for Damerau vs Levenshtein)
    ("ca", "ac"),
    ("ab", "ba"),
    ("abc", "acb"),
    # Identical strings
    ("test", "test"),
    ("", ""),
    # Empty string edge cases
    ("", "abc"),
    ("abc", ""),
    # Single characters
    ("a", "b"),
    ("a", "a"),
    # Unicode
    ("cafe", "cafe"),
    # Case variations
    ("Hello", "hello"),
    ("ABC", "abc"),
    # Longer strings
    ("algorithm", "altruistic"),
    ("intention", "execution"),
    ("the quick brown fox", "the quick brown dog"),
]

PHONETIC_WORDS = [
    "Robert",
    "Rupert",
    "Smith",
    "Smyth",
    "Stephen",
    "Steven",
    "Johnson",
    "Jonson",
    "Thompson",
    "Thomson",
    "Michael",
    "Micheal",
    "Catherine",
    "Katherine",
    "Phillip",
    "Philip",
    "Geoffrey",
    "Jeffrey",
]

HAMMING_PAIRS = [
    # Equal length strings only
    ("karolin", "kathrin"),
    ("karolin", "kerstin"),
    ("1011101", "1001001"),
    ("abc", "axc"),
    ("test", "test"),
]


# =============================================================================
# Comparison Utilities
# =============================================================================


def format_value(val: Any) -> str:
    """Format a value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def values_match(a: Any, b: Any, tolerance: float = 0.0001) -> bool:
    """Check if two values match (with tolerance for floats)."""
    if a is None or b is None:
        return a == b
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) < tolerance
    return a == b


def compare_algorithm(
    name: str,
    test_pairs: list,
    fuzzyrust_fn: Callable,
    rapidfuzz_fn: Optional[Callable] = None,
    jellyfish_fn: Optional[Callable] = None,
    is_float: bool = False,
) -> dict:
    """Compare an algorithm across libraries."""

    results = {
        "name": name,
        "tests": [],
        "passed": 0,
        "failed": 0,
        "deviations": [],
    }

    print(f"\n{'=' * 70}")
    print(f"{name}")
    print("=" * 70)

    # Header
    cols = ["Inputs", "FuzzyRust"]
    if rapidfuzz_fn:
        cols.append("RapidFuzz")
    if jellyfish_fn:
        cols.append("Jellyfish")
    cols.append("Match")

    header = " | ".join(f"{c:^15}" for c in cols)
    print(header)
    print("-" * len(header))

    for a, b in test_pairs:
        row = {}
        display_input = f"{a[:8]}../{b[:8]}.." if len(a) > 8 or len(b) > 8 else f"{a}/{b}"

        # FuzzyRust result
        try:
            fr_result = fuzzyrust_fn(a, b)
        except Exception as e:
            fr_result = f"ERR:{e}"
        row["fuzzyrust"] = fr_result

        # RapidFuzz result
        rf_result = None
        if rapidfuzz_fn:
            try:
                rf_result = rapidfuzz_fn(a, b)
            except Exception as e:
                rf_result = f"ERR:{e}"
        row["rapidfuzz"] = rf_result

        # Jellyfish result
        jf_result = None
        if jellyfish_fn:
            try:
                jf_result = jellyfish_fn(a, b)
            except Exception as e:
                jf_result = f"ERR:{e}"
        row["jellyfish"] = jf_result

        # Check match
        all_match = True
        deviation_info = []

        if rf_result is not None and not isinstance(rf_result, str):
            if not values_match(fr_result, rf_result):
                all_match = False
                deviation_info.append(f"RF diff: {abs(fr_result - rf_result):.6f}")

        if jf_result is not None and not isinstance(jf_result, str):
            if not values_match(fr_result, jf_result):
                all_match = False
                deviation_info.append(f"JF diff: {abs(fr_result - jf_result):.6f}")

        match_symbol = "PASS" if all_match else "DIFF"

        # Build output row
        values = [f"{display_input:^15}", f"{format_value(fr_result):^15}"]
        if rapidfuzz_fn:
            values.append(f"{format_value(rf_result):^15}")
        if jellyfish_fn:
            values.append(f"{format_value(jf_result):^15}")
        values.append(f"{match_symbol:^15}")

        print(" | ".join(values))

        if all_match:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["deviations"].append(
                {
                    "inputs": (a, b),
                    "fuzzyrust": fr_result,
                    "rapidfuzz": rf_result,
                    "jellyfish": jf_result,
                    "info": deviation_info,
                }
            )

        results["tests"].append(row)

    return results


def compare_phonetic(
    name: str,
    words: list,
    fuzzyrust_fn: Callable,
    jellyfish_fn: Optional[Callable] = None,
) -> dict:
    """Compare phonetic encoding algorithms."""

    results = {
        "name": name,
        "tests": [],
        "passed": 0,
        "failed": 0,
        "deviations": [],
    }

    print(f"\n{'=' * 70}")
    print(f"{name}")
    print("=" * 70)

    cols = ["Word", "FuzzyRust"]
    if jellyfish_fn:
        cols.append("Jellyfish")
    cols.append("Match")

    header = " | ".join(f"{c:^15}" for c in cols)
    print(header)
    print("-" * len(header))

    for word in words:
        # FuzzyRust result
        try:
            fr_result = fuzzyrust_fn(word)
        except Exception as e:
            fr_result = f"ERR:{e}"

        # Jellyfish result
        jf_result = None
        if jellyfish_fn:
            try:
                jf_result = jellyfish_fn(word)
            except Exception as e:
                jf_result = f"ERR:{e}"

        # Check match
        all_match = True
        if jf_result is not None and fr_result != jf_result:
            all_match = False

        match_symbol = "PASS" if all_match else "DIFF"

        values = [f"{word:^15}", f"{format_value(fr_result):^15}"]
        if jellyfish_fn:
            values.append(f"{format_value(jf_result):^15}")
        values.append(f"{match_symbol:^15}")

        print(" | ".join(values))

        if all_match:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["deviations"].append(
                {
                    "word": word,
                    "fuzzyrust": fr_result,
                    "jellyfish": jf_result,
                }
            )

    return results


# =============================================================================
# Main Comparison
# =============================================================================


def main():
    print("=" * 70)
    print("FuzzyRust Accuracy Comparison")
    print("=" * 70)

    print("\nComparing libraries:")
    print("  fuzzyrust")
    print("  rapidfuzz")
    print("  jellyfish")

    all_results = []

    # -------------------------------------------------------------------------
    # Levenshtein Distance
    # -------------------------------------------------------------------------
    results = compare_algorithm(
        "Levenshtein Distance",
        TEST_PAIRS,
        fuzzyrust_fn=fr.levenshtein,
        rapidfuzz_fn=rf_distance.Levenshtein.distance,
        jellyfish_fn=jellyfish.levenshtein_distance,
    )
    all_results.append(results)

    # -------------------------------------------------------------------------
    # Damerau-Levenshtein Distance
    # -------------------------------------------------------------------------
    results = compare_algorithm(
        "Damerau-Levenshtein Distance",
        TEST_PAIRS,
        fuzzyrust_fn=fr.damerau_levenshtein,
        rapidfuzz_fn=rf_distance.DamerauLevenshtein.distance,
        jellyfish_fn=jellyfish.damerau_levenshtein_distance,
    )
    all_results.append(results)

    # -------------------------------------------------------------------------
    # Jaro Similarity
    # -------------------------------------------------------------------------
    results = compare_algorithm(
        "Jaro Similarity",
        TEST_PAIRS,
        fuzzyrust_fn=fr.jaro_similarity,
        rapidfuzz_fn=rf_distance.Jaro.similarity,
        jellyfish_fn=jellyfish.jaro_similarity,
        is_float=True,
    )
    all_results.append(results)

    # -------------------------------------------------------------------------
    # Jaro-Winkler Similarity
    # -------------------------------------------------------------------------
    results = compare_algorithm(
        "Jaro-Winkler Similarity",
        TEST_PAIRS,
        fuzzyrust_fn=fr.jaro_winkler_similarity,
        rapidfuzz_fn=rf_distance.JaroWinkler.similarity,
        jellyfish_fn=jellyfish.jaro_winkler_similarity,
        is_float=True,
    )
    all_results.append(results)

    # -------------------------------------------------------------------------
    # Hamming Distance
    # -------------------------------------------------------------------------
    results = compare_algorithm(
        "Hamming Distance (equal length only)",
        HAMMING_PAIRS,
        fuzzyrust_fn=fr.hamming,
        rapidfuzz_fn=rf_distance.Hamming.distance,
        jellyfish_fn=jellyfish.hamming_distance,
    )
    all_results.append(results)

    # -------------------------------------------------------------------------
    # Soundex
    # -------------------------------------------------------------------------
    results = compare_phonetic(
        "Soundex Encoding",
        PHONETIC_WORDS,
        fuzzyrust_fn=fr.soundex,
        jellyfish_fn=jellyfish.soundex,
    )
    all_results.append(results)

    # -------------------------------------------------------------------------
    # Metaphone
    # -------------------------------------------------------------------------
    results = compare_phonetic(
        "Metaphone Encoding",
        PHONETIC_WORDS,
        fuzzyrust_fn=fr.metaphone,
        jellyfish_fn=jellyfish.metaphone,
    )
    all_results.append(results)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_passed = 0
    total_failed = 0

    for r in all_results:
        status = "PASS" if r["failed"] == 0 else "DIFF"
        print(f"  {r['name']:40} {r['passed']:3}/{r['passed'] + r['failed']:3} {status}")
        total_passed += r["passed"]
        total_failed += r["failed"]

        if r["deviations"]:
            print("    Deviations:")
            for d in r["deviations"][:3]:  # Show first 3
                if "word" in d:
                    print(f"      - {d['word']}: FR={d['fuzzyrust']} JF={d['jellyfish']}")
                else:
                    a, b = d["inputs"]
                    print(
                        f"      - '{a}'/'{b}': FR={d['fuzzyrust']} RF={d.get('rapidfuzz')} JF={d.get('jellyfish')}"
                    )

    print("-" * 70)
    print(f"  {'TOTAL':40} {total_passed:3}/{total_passed + total_failed:3}")

    if total_failed > 0:
        print(f"\n  {total_failed} differences found - see details above")
        return 1
    else:
        print("\n  All results match!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
