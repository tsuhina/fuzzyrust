"""Type stubs for fuzzyrust."""

from collections.abc import Sequence
from enum import Enum
from typing import Any, TypeAlias

__version__: str

# Type alias for user data - must fit in u64 range (0 to 2^64-1)
# Values outside this range will cause a runtime error
UserData: TypeAlias = int

# =============================================================================
# Result Types
# =============================================================================

class SearchResult:
    """
    Unified search result across all index types.

    Attributes:
        id: Unique identifier assigned when the item was added
        text: The matched text
        score: Similarity score (0.0-1.0)
        distance: Edit distance (only for BkTree searches)
        data: User-provided data (must be in u64 range: 0 to 2^64-1)

    Supports equality comparison and hashing for use in sets and as dict keys.
    """

    id: int
    text: str
    score: float
    distance: int | None
    data: UserData | None

    def __init__(
        self,
        id: int,
        text: str,
        score: float,
        distance: int | None = None,
        data: UserData | None = None,
    ) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class MatchResult:
    """
    Result from batch operations.

    Supports equality comparison and hashing for use in sets and as dict keys.
    """

    text: str
    score: float
    id: int | None

    def __init__(self, text: str, score: float, id: int | None = None) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class DeduplicationResult:
    """Result from deduplication operation."""

    groups: list[list[str]]
    unique: list[str]
    total_duplicates: int

    def __init__(
        self,
        groups: list[list[str]],
        unique: list[str],
        total_duplicates: int,
    ) -> None: ...

class AlgorithmComparison:
    """Result from multi-algorithm comparison."""

    algorithm: str
    score: float
    matches: list[MatchResult]

    def __init__(
        self,
        algorithm: str,
        score: float,
        matches: list[MatchResult],
    ) -> None: ...

# =============================================================================
# Exceptions
# =============================================================================

class FuzzyRustError(Exception):
    """Base exception for all FuzzyRust errors."""

    ...

class ValidationError(FuzzyRustError):
    """Raised when input validation fails (invalid parameters, out of range values)."""

    ...

class AlgorithmError(FuzzyRustError):
    """Raised when an unknown or unsupported algorithm is specified."""

    ...

class FuzzyIndexError(FuzzyRustError):
    """Raised when index operations fail (add, search, serialization).

    Note: Aliased from IndexError in Rust to avoid shadowing Python's built-in IndexError.
    """

    ...

class SchemaError(FuzzyRustError):
    """Raised when schema validation or operations fail."""

    ...

# =============================================================================
# Enums
# =============================================================================

class Algorithm(str, Enum):
    """
    Available similarity algorithms for fuzzy string matching.

    Algorithm Selection Guide:

    For NAME MATCHING (person names, company names):
        - JARO_WINKLER: Best overall for names. Weighs prefix matches highly.
        - SOUNDEX/METAPHONE: Use for phonetic matching (sounds-alike).

    For TYPO DETECTION (user input, search queries):
        - DAMERAU_LEVENSHTEIN: Handles transpositions ("teh" -> "the").
        - LEVENSHTEIN: Classic edit distance, good baseline.

    For SHORT STRINGS (< 20 chars):
        - JARO_WINKLER: Fast and accurate for short strings.
        - JARO: Slightly faster, no prefix boost.

    For LONG TEXT (sentences, paragraphs):
        - COSINE: Best for documents, ignores word order.
        - NGRAM/TRIGRAM: Good for partial matches in text.

    For TOKEN-BASED MATCHING (tags, keywords):
        - NGRAM: Works well with short tokens.
        - COSINE: For word-based similarity.

    Performance Comparison (fastest to slowest):
        1. JARO, JARO_WINKLER: O(m*n) but typically O(m+n) for similar strings
        2. NGRAM, COSINE: O(m+n) for n-gram/word extraction
        3. LEVENSHTEIN: O(m*n) with early termination
        4. DAMERAU_LEVENSHTEIN: O(m*n), higher memory than Levenshtein
        5. LCS: O(m*n), good for alignment tasks

    Index Selection:
        - BkTree: Best for exact distance threshold searches
        - NgramIndex: Best for similarity scoring with fast candidate filtering
        - HybridIndex: Best balance for large datasets (100K+ items)
    """

    LEVENSHTEIN = "levenshtein"
    DAMERAU_LEVENSHTEIN = "damerau_levenshtein"
    DAMERAU = "damerau"
    """Alias for DAMERAU_LEVENSHTEIN"""
    JARO = "jaro"
    JARO_WINKLER = "jaro_winkler"
    NGRAM = "ngram"
    BIGRAM = "bigram"
    TRIGRAM = "trigram"
    JACCARD = "jaccard"
    LCS = "lcs"
    COSINE = "cosine"
    HAMMING = "hamming"

class NormalizationMode(str, Enum):
    """String normalization modes."""

    LOWERCASE = "lowercase"
    UNICODE_NFKD = "unicode_nfkd"
    REMOVE_PUNCTUATION = "remove_punctuation"
    REMOVE_WHITESPACE = "remove_whitespace"
    STRICT = "strict"

# =============================================================================
# Distance / Similarity Functions
# =============================================================================

def levenshtein(
    a: str,
    b: str,
    max_distance: int | None = None,
    normalize: (str | NormalizationMode) | None = None,
) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.

    Args:
        a: First string
        b: Second string
        max_distance: Optional maximum distance for early termination.
                     Returns max_distance + 1 if exceeded (RapidFuzz compatible).
                     Use levenshtein_bounded() if you prefer None semantics.
        normalize: Optional normalization mode to apply before comparison:
                  "lowercase", "unicode_nfkd", "remove_punctuation",
                  "remove_whitespace", "strict"

    Returns:
        The minimum number of single-character edits needed to transform a into b.
        Returns max_distance + 1 if threshold exceeded.

    Complexity:
        Time: O(m*n) where m, n are string lengths. O(m*k) with max_distance=k.
        Space: O(min(m, n)) using two-row optimization.

    Example:
        >>> levenshtein("kitten", "sitting")
        3
        >>> levenshtein("Hello", "HELLO", normalize="lowercase")
        0
        >>> levenshtein("abc", "xyz", max_distance=2)  # Distance is 3, exceeds threshold
        3  # Returns max_distance + 1
    """
    ...

def levenshtein_bounded(
    a: str,
    b: str,
    max_distance: int,
    normalize: (str | NormalizationMode) | None = None,
) -> int | None:
    """
    Compute Levenshtein distance with explicit None return when threshold exceeded.

    Args:
        a: First string
        b: Second string
        max_distance: Maximum distance threshold. Returns None if exceeded.
        normalize: Optional normalization mode to apply before comparison

    Returns:
        The edit distance if within threshold, None if exceeded.

    Example:
        >>> levenshtein_bounded("abc", "abd", max_distance=2)
        1
        >>> levenshtein_bounded("abc", "xyz", max_distance=2)
        None
    """
    ...

def levenshtein_similarity(
    a: str,
    b: str,
    normalize: (str | NormalizationMode) | None = None,
) -> float:
    """
    Compute normalized Levenshtein similarity (0.0 to 1.0).

    Args:
        a: First string
        b: Second string
        normalize: Optional normalization mode to apply before comparison

    Returns:
        Similarity score where 1.0 means identical strings.

    Complexity:
        Time: O(m*n) where m, n are string lengths.
        Space: O(min(m, n)) using two-row optimization.

    Example:
        >>> levenshtein_similarity("Hello", "HELLO", normalize="lowercase")
        1.0
    """
    ...

def damerau_levenshtein(
    a: str,
    b: str,
    max_distance: int | None = None,
    normalize: (str | NormalizationMode) | None = None,
) -> int:
    """
    Compute Damerau-Levenshtein distance (includes transpositions).

    Like Levenshtein but also counts character swaps as a single edit.
    Useful for typo detection where letter swaps are common.

    This uses the "true" Damerau-Levenshtein algorithm which allows multiple
    edits on the same substring. This matches jellyfish's behavior.
    For the restricted (OSA) variant, use optimal_string_alignment().

    Args:
        a: First string
        b: Second string
        max_distance: Optional maximum distance for early termination.
                     Returns max_distance + 1 if exceeded (RapidFuzz compatible).
                     Use damerau_levenshtein_bounded() if you prefer None semantics.
        normalize: Optional normalization mode to apply before comparison

    Returns:
        The edit distance. Returns max_distance + 1 if threshold exceeded.

    Complexity:
        Time: O(m*n) where m, n are string lengths.
        Space: O(m*n) for the full DP matrix (transposition tracking requires it).

    Example:
        >>> damerau_levenshtein("ca", "ac")  # One transposition
        1
        >>> levenshtein("ca", "ac")  # Two edits without transposition
        2
    """
    ...

def damerau_levenshtein_bounded(
    a: str,
    b: str,
    max_distance: int,
    normalize: (str | NormalizationMode) | None = None,
) -> int | None:
    """
    Compute Damerau-Levenshtein distance with explicit None return when threshold exceeded.

    Args:
        a: First string
        b: Second string
        max_distance: Maximum distance threshold. Returns None if exceeded.
        normalize: Optional normalization mode to apply before comparison

    Returns:
        The edit distance if within threshold, None if exceeded.

    Example:
        >>> damerau_levenshtein_bounded("ca", "ac", max_distance=2)
        1
        >>> damerau_levenshtein_bounded("abc", "xyz", max_distance=2)
        None
    """
    ...

def damerau_levenshtein_similarity(
    a: str,
    b: str,
    normalize: (str | NormalizationMode) | None = None,
) -> float:
    """
    Compute normalized Damerau-Levenshtein similarity (0.0 to 1.0).

    Uses the "true" Damerau-Levenshtein algorithm which allows multiple edits
    on the same substring. This matches jellyfish's behavior.

    Args:
        normalize: Optional normalization mode to apply before comparison

    Complexity:
        Time: O(m*n) where m, n are string lengths.
        Space: O(m*n) for the full DP matrix.
    """
    ...

def optimal_string_alignment(
    a: str,
    b: str,
    max_distance: int | None = None,
    normalize: (str | NormalizationMode) | None = None,
) -> int:
    """
    Compute Optimal String Alignment (OSA) distance - restricted Damerau-Levenshtein.

    This is the restricted variant that doesn't allow multiple edits on the
    same substring. It's slightly faster than the true algorithm but may
    produce different results for certain edge cases.

    Args:
        max_distance: Optional threshold. Returns max_distance + 1 if exceeded.
        normalize: Optional normalization mode to apply before comparison.

    Example:
        >>> optimal_string_alignment("ab", "ba")  # Single transposition
        1
        >>> optimal_string_alignment("00210000", "001020000")  # OSA = 3, true DL = 2
        3
    """
    ...

def optimal_string_alignment_similarity(
    a: str,
    b: str,
    normalize: (str | NormalizationMode) | None = None,
) -> float:
    """
    Compute normalized Optimal String Alignment similarity (0.0 to 1.0).

    Uses the restricted (OSA) Damerau-Levenshtein algorithm.

    Args:
        normalize: Optional normalization mode to apply before comparison
    """
    ...

def jaro_similarity(
    a: str,
    b: str,
    normalize: (str | NormalizationMode) | None = None,
) -> float:
    """
    Compute Jaro similarity (0.0 to 1.0).

    Good for short strings and name matching.

    Args:
        normalize: Optional normalization mode to apply before comparison

    Complexity:
        Time: O(m*n) worst case, typically O(m+n) for similar strings.
        Space: O(m+n) for matching character tracking.

    Example:
        >>> jaro_similarity("MARTHA", "MARHTA")
        0.944...
    """
    ...

def jaro_winkler_similarity(
    a: str,
    b: str,
    prefix_weight: float = 0.1,
    max_prefix_length: int = 4,
    normalize: (str | NormalizationMode) | None = None,
) -> float:
    """
    Compute Jaro-Winkler similarity (0.0 to 1.0).

    Extends Jaro similarity by giving extra weight to common prefixes.
    Excellent for name matching.

    Args:
        a: First string
        b: Second string
        prefix_weight: Weight given to common prefix (default 0.1)
        max_prefix_length: Maximum prefix length to consider (default 4)
        normalize: Optional normalization mode to apply before comparison

    Complexity:
        Time: O(m*n) worst case, typically O(m+n) for similar strings.
        Space: O(m+n) for matching character tracking.
    """
    ...

def hamming(a: str, b: str) -> int:
    """
    Compute Hamming distance between two equal-length strings.

    Raises:
        ValueError: If strings have different lengths.

    Complexity:
        Time: O(n) where n is the string length.
        Space: O(1) constant.

    Example:
        >>> hamming("karolin", "kathrin")
        3
    """
    ...

def hamming_distance_padded(a: str, b: str) -> int:
    """
    Compute Hamming distance with padding for unequal-length strings.

    Unlike regular Hamming distance which requires equal-length strings,
    this pads the shorter string to enable comparison.

    Complexity:
        Time: O(max(m, n)) where m, n are string lengths.
        Space: O(1) constant.

    Example:
        >>> hamming_distance_padded("abc", "ab")
        1  # Compares "abc" with "ab " (padded)
    """
    ...

def hamming_similarity(a: str, b: str) -> float:
    """
    Compute normalized Hamming similarity (0.0 to 1.0).

    Note:
        Hamming similarity is only defined for strings of equal length.
        For unequal-length strings, use hamming_distance_padded() which
        pads the shorter string with spaces.

    Raises:
        ValidationError: If strings have different lengths.

    Complexity:
        Time: O(n) where n is the string length.
        Space: O(1) constant.

    Example:
        >>> hamming_similarity("abc", "axc")
        0.666...  # 2 out of 3 match
    """
    ...

def ngram_similarity(
    a: str,
    b: str,
    ngram_size: int = 3,
    pad: bool = True,
    normalize: (str | NormalizationMode) | None = None,
) -> float:
    """
    Compute n-gram similarity (Sørensen-Dice coefficient).

    Args:
        a: First string
        b: Second string
        ngram_size: Size of n-grams (default 3 for trigrams)
        pad: Whether to pad strings for edge matching
        normalize: Optional normalization mode to apply before comparison

    Complexity:
        Time: O(m + n) for n-gram extraction and set operations.
        Space: O(m + n) for storing n-gram sets.

    Example:
        >>> ngram_similarity("night", "nacht")
        0.5
    """
    ...

def ngram_jaccard(
    a: str,
    b: str,
    ngram_size: int = 3,
    pad: bool = True,
    normalize: (str | NormalizationMode) | None = None,
) -> float:
    """
    Compute n-gram Jaccard similarity.

    Args:
        normalize: Optional normalization mode to apply before comparison

    Complexity:
        Time: O(m + n) for n-gram extraction and set operations.
        Space: O(m + n) for storing n-gram sets.
    """
    ...

def bigram_similarity(a: str, b: str) -> float:
    """
    Compute bigram similarity (n-gram with n=2).

    Convenience function equivalent to ngram_similarity(a, b, 2).

    Example:
        >>> bigram_similarity("hello", "hallo")
        0.4
    """
    ...

def trigram_similarity(a: str, b: str) -> float:
    """
    Compute trigram similarity (n-gram with n=3).

    Convenience function equivalent to ngram_similarity(a, b, 3).

    Example:
        >>> trigram_similarity("hello", "hallo")
        0.285...
    """
    ...

def ngram_profile_similarity(a: str, b: str, ngram_size: int = 3) -> float:
    """
    Compute n-gram profile similarity.

    Unlike regular n-gram similarity which only checks presence,
    this counts n-gram frequencies for a more accurate comparison
    when strings have repeated patterns.

    Example:
        >>> ngram_profile_similarity("abab", "abab")
        1.0
        >>> ngram_profile_similarity("aaa", "aa")
        0.666...  # Frequency matters
    """
    ...

def extract_ngrams(s: str, ngram_size: int = 3, pad: bool = True) -> list[str]:
    """
    Extract n-grams from a string.

    Example:
        >>> extract_ngrams("hello", ngram_size=2, pad=False)
        ['he', 'el', 'll', 'lo']
    """
    ...

def soundex(s: str) -> str:
    """
    Encode a string using Soundex algorithm.

    Returns a 4-character phonetic code.

    Complexity:
        Time: O(n) where n is the string length.
        Space: O(1) constant (always returns 4 characters).

    Example:
        >>> soundex("Robert")
        'R163'
        >>> soundex("Rupert")
        'R163'
    """
    ...

def soundex_match(a: str, b: str) -> bool:
    """Check if two strings have the same Soundex code."""
    ...

def soundex_similarity(a: str, b: str) -> float:
    """
    Compute Soundex phonetic similarity (0.0 to 1.0).

    Returns 1.0 for identical Soundex codes, otherwise a partial match
    score based on matching positions in the 4-character codes.

    Example:
        >>> soundex_similarity("Robert", "Rupert")
        1.0  # Same Soundex code R163
        >>> soundex_similarity("Robert", "Rubin")
        0.5  # Partial match
    """
    ...

def metaphone(s: str, max_length: int = 4) -> str:
    """
    Encode a string using Metaphone algorithm.

    More accurate than Soundex for many cases.

    Complexity:
        Time: O(n) where n is the string length.
        Space: O(max_length) for the result.
    """
    ...

def metaphone_match(a: str, b: str) -> bool:
    """Check if two strings have the same Metaphone code."""
    ...

def metaphone_similarity(a: str, b: str, max_length: int = 4) -> float:
    """
    Compute Metaphone phonetic similarity (0.0 to 1.0).

    Uses Jaro-Winkler similarity on the Metaphone codes for partial matching.
    More granular than metaphone_match() which only returns a boolean.

    Example:
        >>> metaphone_similarity("Stephen", "Steven")
        1.0  # Same Metaphone code
        >>> metaphone_similarity("John", "Jon")
        0.933...  # Very similar phonetically
    """
    ...

def double_metaphone(text: str, max_length: int = 4) -> tuple[str, str]:
    """
    Encode a string using Double Metaphone algorithm.

    Double Metaphone is an improvement over Metaphone that handles European
    names (Germanic, Slavic, Italian, etc.) better by returning two possible
    phonetic codes - a primary and an alternate pronunciation.

    Args:
        text: String to encode
        max_length: Maximum length of each code (default: 4)

    Returns:
        Tuple of (primary_code, alternate_code). The alternate may be empty
        if there's only one pronunciation.

    Example:
        >>> double_metaphone("Schmidt")
        ('XMT', 'SMT')  # German and Anglicized pronunciations
        >>> double_metaphone("Smith")
        ('SM0', 'XMT')  # 0 represents the TH sound
    """
    ...

def double_metaphone_match(a: str, b: str) -> bool:
    """
    Check if two strings match phonetically using Double Metaphone.

    Returns True if any of the codes match (primary-primary, primary-alternate,
    alternate-primary, or alternate-alternate).

    More lenient than metaphone_match because it considers alternate
    pronunciations, making it better for matching names with varied spellings.

    Example:
        >>> double_metaphone_match("Schmidt", "Smith")
        True  # Matches via alternate codes
        >>> double_metaphone_match("Katherine", "Catherine")
        True
    """
    ...

def double_metaphone_similarity(a: str, b: str, max_length: int = 4) -> float:
    """
    Compute Double Metaphone similarity between two strings.

    Returns 1.0 for exact phonetic match (any code combination matches),
    or a partial score based on Jaro-Winkler similarity of the best
    matching code pair.

    Args:
        a: First string
        b: Second string
        max_length: Maximum length of metaphone codes (default: 4)

    Returns:
        Similarity score from 0.0 to 1.0

    Example:
        >>> double_metaphone_similarity("Stephen", "Steven")
        1.0
        >>> double_metaphone_similarity("Schmidt", "Smith")
        1.0  # Matches via alternate codes
    """
    ...

# =============================================================================
# Grapheme Cluster Mode Functions
# =============================================================================
# These functions treat grapheme clusters as single units, which is important
# for correctly handling emoji sequences, combining characters, and other
# multi-codepoint characters.

def levenshtein_grapheme(a: str, b: str) -> int:
    """
    Compute Levenshtein distance treating grapheme clusters as single units.

    A grapheme cluster is a user-perceived character, which may consist of
    multiple Unicode code points (e.g., family emoji 👨‍👩‍👧‍👦 is 7 code points
    but 1 grapheme cluster).

    Args:
        a: First string
        b: Second string

    Returns:
        Edit distance counting each grapheme cluster as one character.

    Example:
        >>> # Family emoji: 7 code points but 1 grapheme
        >>> levenshtein("👨‍👩‍👧‍👦", "👨‍👩‍👧")  # Regular: counts code points
        4
        >>> levenshtein_grapheme("👨‍👩‍👧‍👦", "👨‍👩‍👧")  # Grapheme: counts visual chars
        1
    """
    ...

def levenshtein_similarity_grapheme(a: str, b: str) -> float:
    """
    Compute normalized Levenshtein similarity using grapheme clusters.

    Like levenshtein_similarity but treats grapheme clusters as single units.
    This gives more intuitive results for text containing emoji or combining characters.

    Args:
        a: First string
        b: Second string

    Returns:
        Similarity score from 0.0 to 1.0 based on grapheme distance.

    Example:
        >>> levenshtein_similarity_grapheme("café", "cafe")  # é is 1 grapheme
        0.8  # 1 difference out of 5 graphemes
    """
    ...

def jaro_similarity_grapheme(a: str, b: str) -> float:
    """
    Compute Jaro similarity treating grapheme clusters as single units.

    Like jaro_similarity but correctly handles multi-codepoint characters
    such as emoji sequences, combining characters, and regional indicators.

    Args:
        a: First string
        b: Second string

    Returns:
        Jaro similarity score from 0.0 to 1.0.

    Example:
        >>> jaro_similarity_grapheme("👨‍👩‍👧", "👨‍👩‍👦")
        0.833...  # 2 matching graphemes out of 3
    """
    ...

def jaro_winkler_similarity_grapheme(
    a: str,
    b: str,
    prefix_weight: float = 0.1,
    max_prefix_length: int = 4,
) -> float:
    """
    Compute Jaro-Winkler similarity treating grapheme clusters as single units.

    Like jaro_winkler_similarity but counts grapheme clusters instead of
    code points. This gives correct results for strings with emoji or
    combining characters.

    Args:
        a: First string
        b: Second string
        prefix_weight: Weight for common prefix bonus (default: 0.1)
        max_prefix_length: Maximum prefix length to consider (default: 4)

    Returns:
        Jaro-Winkler similarity score from 0.0 to 1.0.

    Example:
        >>> jaro_winkler_similarity_grapheme("🇺🇸🇬🇧", "🇺🇸🇫🇷")
        0.9  # Common prefix "🇺🇸" boosts the score
    """
    ...

# =============================================================================
# SIMD-Accelerated Functions
# =============================================================================
# These functions use triple_accel's SIMD-optimized implementation for
# 20-30x faster edit distance calculations on x86/x86-64 CPUs.

def levenshtein_simd(a: str, b: str) -> int:
    """
    SIMD-accelerated Levenshtein distance.

    Uses triple_accel's SIMD-optimized implementation which can be 20-30x faster
    than the scalar implementation for longer strings. Automatically falls back
    to scalar on CPUs without SIMD support (e.g., ARM).

    Best for:
    - Long strings (>100 characters)
    - Batch processing many comparisons
    - x86/x86-64 CPUs with AVX2 or SSE4.1

    Note: Works on byte-level, so for Unicode strings the result may differ
    slightly from character-level distance (use regular levenshtein for
    character-accurate Unicode distance).

    Args:
        a: First string
        b: Second string

    Returns:
        Edit distance (number of insertions, deletions, substitutions).

    Example:
        >>> levenshtein_simd("kitten", "sitting")
        3
    """
    ...

def levenshtein_simd_bounded(a: str, b: str, max_distance: int) -> int | None:
    """
    SIMD-accelerated Levenshtein distance with max threshold.

    Returns None if the distance exceeds max_distance, enabling early termination.
    Uses SIMD acceleration when available for optimal performance.

    Args:
        a: First string
        b: Second string
        max_distance: Maximum distance threshold

    Returns:
        Edit distance if within threshold, None if exceeded.

    Example:
        >>> levenshtein_simd_bounded("hello", "hallo", 2)
        1
        >>> levenshtein_simd_bounded("abc", "xyz", 2)
        None
    """
    ...

def levenshtein_similarity_simd(a: str, b: str) -> float:
    """
    SIMD-accelerated Levenshtein similarity (0.0 to 1.0).

    Uses SIMD acceleration for the distance calculation, providing
    significant speedups on x86/x86-64 CPUs.

    Args:
        a: First string
        b: Second string

    Returns:
        Similarity score from 0.0 (completely different) to 1.0 (identical).

    Example:
        >>> levenshtein_similarity_simd("hello", "hallo")
        0.8
    """
    ...

def lcs_length(a: str, b: str) -> int:
    """
    Compute the length of the Longest Common Subsequence.

    Complexity:
        Time: O(m*n) where m, n are string lengths.
        Space: O(min(m, n)) using space-optimized DP.

    Example:
        >>> lcs_length("ABCDGH", "AEDFHR")
        3  # ADH
    """
    ...

def lcs_string(a: str, b: str) -> str:
    """
    Get the actual Longest Common Subsequence string.

    Complexity:
        Time: O(m*n) where m, n are string lengths.
        Space: O(m*n) for backtracking matrix.

    Raises:
        ValidationError: If either string exceeds 10,000 characters.
            Use lcs_length for a space-efficient alternative.
    """
    ...

def lcs_similarity(a: str, b: str) -> float:
    """
    Compute LCS-based similarity (0.0 to 1.0).

    Complexity:
        Time: O(m*n) where m, n are string lengths.
        Space: O(min(m, n)) using space-optimized DP.
    """
    ...

def lcs_similarity_max(a: str, b: str) -> float:
    """
    Compute LCS similarity using max length as denominator.

    Alternative to lcs_similarity which uses sum of lengths.
    Formula: LCS_length / max(len_a, len_b)

    This may be more intuitive for some use cases as it represents
    "what fraction of the longer string is covered by the LCS".

    Complexity:
        Time: O(m*n) where m, n are string lengths.
        Space: O(min(m, n)) using space-optimized DP.

    Example:
        >>> lcs_similarity_max("abc", "abc")
        1.0
        >>> lcs_similarity_max("abc", "ab")
        0.666...  # LCS "ab" covers 2/3 of "abc"
    """
    ...

def longest_common_substring_length(a: str, b: str) -> int:
    """
    Compute the length of the longest common contiguous substring.

    Complexity:
        Time: O(m*n) where m, n are string lengths.
        Space: O(min(m, n)) using space-optimized DP.
    """
    ...

def longest_common_substring(a: str, b: str) -> str:
    """
    Get the longest common contiguous substring.

    Complexity:
        Time: O(m*n) where m, n are string lengths.
        Space: O(m*n) for backtracking.

    Raises:
        ValidationError: If either string exceeds 10,000 characters.
            Use longest_common_substring_length for a space-efficient alternative.
    """
    ...

def cosine_similarity_chars(
    a: str,
    b: str,
    normalize: (str | NormalizationMode) | None = None,
) -> float:
    """
    Compute character-level cosine similarity.

    Args:
        normalize: Optional normalization mode to apply before comparison

    Complexity:
        Time: O(m + n) for character frequency counting.
        Space: O(|alphabet|) for frequency vectors.
    """
    ...

def cosine_similarity_words(
    a: str,
    b: str,
    normalize: (str | NormalizationMode) | None = None,
) -> float:
    """
    Compute word-level cosine similarity.

    Args:
        normalize: Optional normalization mode to apply before comparison

    Complexity:
        Time: O(m + n) for word tokenization and counting.
        Space: O(w) where w is the number of unique words.
    """
    ...

def cosine_similarity_ngrams(
    a: str,
    b: str,
    ngram_size: int = 3,
    normalize: (str | NormalizationMode) | None = None,
) -> float:
    """
    Compute n-gram cosine similarity.

    Args:
        normalize: Optional normalization mode to apply before comparison

    Complexity:
        Time: O(m + n) for n-gram extraction and counting.
        Space: O(m + n) for n-gram frequency vectors.
    """
    ...

# =============================================================================
# String Normalization
# =============================================================================

def normalize_string(s: str, mode: str | NormalizationMode) -> str:
    """
    Normalize a string according to the specified mode.

    Args:
        s: String to normalize
        mode: Normalization mode - one of:
              "lowercase", "unicode_nfkd", "remove_punctuation",
              "remove_whitespace", "strict"
              Can also use NormalizationMode enum values.

    Returns:
        Normalized string

    Example:
        >>> normalize_string("  Hello, World!  ", "strict")
        'helloworld'
        >>> normalize_string("Hello", NormalizationMode.LOWERCASE)
        'hello'
    """
    ...

def normalize_pair(a: str, b: str, mode: str | NormalizationMode) -> tuple[str, str]:
    """
    Normalize both strings according to the specified mode.

    Args:
        a: First string to normalize
        b: Second string to normalize
        mode: Normalization mode

    Returns:
        Tuple of (normalized_a, normalized_b)

    Example:
        >>> normalize_pair("Hello", "WORLD", "lowercase")
        ('hello', 'world')
    """
    ...

# =============================================================================
# RapidFuzz-Compatible Convenience Functions
# =============================================================================

def partial_ratio(s1: str, s2: str) -> float:
    """
    Compute best partial match ratio between two strings.

    Slides the shorter string across the longer string and returns the
    maximum similarity found. Useful for matching when one string is
    a substring of the other.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score (0.0 to 1.0). Returns 1.0 if one string is a
        perfect substring of the other.

    Example:
        >>> partial_ratio("test", "this is a test!")
        1.0
        >>> partial_ratio("hello", "hello world")
        1.0
    """
    ...

def token_sort_ratio(s1: str, s2: str) -> float:
    """
    Compute similarity after tokenizing and sorting both strings.

    Useful for comparing strings where word order doesn't matter.
    "fuzzy wuzzy was a bear" matches "was a bear fuzzy wuzzy" perfectly.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score (0.0 to 1.0).

    Example:
        >>> token_sort_ratio("fuzzy wuzzy", "wuzzy fuzzy")
        1.0
        >>> token_sort_ratio("hello world", "world hello")
        1.0
    """
    ...

def token_set_ratio(s1: str, s2: str) -> float:
    """
    Compute set-based token similarity.

    Useful for comparing strings where duplicates and order don't matter.
    "fuzzy fuzzy was a bear" matches "fuzzy was a bear" highly.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score (0.0 to 1.0).

    Example:
        >>> token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
        0.95
        >>> token_set_ratio("hello world", "hello")
        0.8
    """
    ...

def wratio(
    s1: str,
    s2: str,
    partial_weight: float = 0.9,
    token_weight: float = 0.95,
) -> float:
    """
    Compute weighted ratio using the best method for the input.

    Automatically selects the best comparison method based on string
    characteristics:
    - For similar-length strings: uses basic ratio
    - For different-length strings: uses partial_ratio (scaled)
    - Also considers token_sort_ratio and token_set_ratio (scaled)

    Returns the maximum of all methods (with appropriate weights).
    This is the recommended function for general-purpose fuzzy matching.

    Args:
        s1: First string
        s2: Second string
        partial_weight: Weight for partial_ratio (0.0 to 1.0, default: 0.9).
            Higher values favor substring matching. The effective weight is
            further adjusted based on length ratio between strings.
        token_weight: Weight for token-based ratios (0.0 to 1.0, default: 0.95).
            Higher values favor word-order-independent matching (token_sort_ratio
            and token_set_ratio).

    Returns:
        Similarity score (0.0 to 1.0).

    Example:
        >>> wratio("hello world", "hello there world")
        0.82
        >>> wratio("test", "this is a test!")
        0.9
        >>> # Custom weights to favor token matching
        >>> wratio("hello world", "world hello", partial_weight=0.9, token_weight=1.0)
        1.0
    """
    ...

def ratio(s1: str, s2: str) -> float:
    """
    Compute basic similarity ratio (Levenshtein-based).

    This is an alias for levenshtein_similarity, providing API compatibility
    with RapidFuzz's `fuzz.ratio`.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score (0.0 to 1.0).

    Example:
        >>> ratio("hello", "hallo")
        0.8
    """
    ...

def qratio(s1: str, s2: str) -> float:
    """
    Compute similarity ratio with automatic Unicode NFC normalization.

    This is equivalent to ratio() but first applies Unicode NFC (Canonical
    Decomposition, followed by Canonical Composition) normalization to both
    strings. This ensures that equivalent Unicode representations (e.g., "é"
    as a single character vs "e" + combining accent) are treated as identical.

    Use this function when comparing strings that may have different Unicode
    representations of the same visual characters.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score (0.0 to 1.0).

    Example:
        >>> # These are the same character in different Unicode forms
        >>> nfc = "café"  # é as single codepoint U+00E9
        >>> nfd = "cafe\\u0301"  # e + combining acute accent
        >>> qratio(nfc, nfd)
        1.0
    """
    ...

def qwratio(
    s1: str,
    s2: str,
    partial_weight: float = 0.9,
    token_weight: float = 0.95,
) -> float:
    """
    Compute weighted ratio with automatic Unicode NFC normalization.

    This is equivalent to wratio() but first applies Unicode NFC normalization
    to both strings before comparison. Combines the intelligent method selection
    of wratio with proper Unicode handling.

    Args:
        s1: First string
        s2: Second string
        partial_weight: Weight for partial_ratio (0.0 to 1.0, default: 0.9)
        token_weight: Weight for token-based ratios (0.0 to 1.0, default: 0.95)

    Returns:
        Similarity score (0.0 to 1.0).

    Example:
        >>> qwratio("naïve approach", "naive approach")
        1.0
        >>> qwratio("Müller", "Mueller")
        0.85
    """
    ...

def extract(
    query: str,
    choices: list[str],
    limit: int = 10,
    min_similarity: float = 0.0,
) -> list[MatchResult]:
    """
    Find top N matches from a list (RapidFuzz-compatible).

    Uses wratio by default for best automatic matching. This function
    provides API compatibility with RapidFuzz's `process.extract`.

    Args:
        query: Query string to match
        choices: List of strings to search
        limit: Maximum number of results (default 10)
        min_similarity: Minimum similarity threshold (default 0.0)

    Returns:
        List of MatchResult objects sorted by score descending.

    Example:
        >>> results = extract("appel", ["apple", "apply", "banana"], limit=2)
        >>> [(r.text, r.score) for r in results]
        [('apple', 0.93), ('apply', 0.80)]
    """
    ...

def extract_one(
    query: str,
    choices: list[str],
    min_similarity: float = 0.0,
) -> MatchResult | None:
    """
    Find the single best match from a list (RapidFuzz-compatible).

    Returns the best match if one exists above the minimum similarity, otherwise None.
    This function provides API compatibility with RapidFuzz's `process.extractOne`.

    Args:
        query: Query string to match
        choices: List of strings to search
        min_similarity: Minimum similarity threshold (default 0.0)

    Returns:
        Best MatchResult if found above threshold, otherwise None.

    Example:
        >>> result = extract_one("appel", ["apple", "banana"])
        >>> result.text, result.score
        ('apple', 0.93)
        >>> extract_one("xyz", ["apple", "banana"], min_similarity=0.9)
        None
    """
    ...

# =============================================================================
# Deduplication
# =============================================================================

def find_duplicate_pairs(
    items: list[str],
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.85,
    window_size: int = 50,
    normalize: str = "lowercase",
) -> list[tuple[int, int, float]]:
    """
    Find duplicate pairs using Sorted Neighborhood Method (SNM).

    Returns pairs of similar items with their indices and similarity scores.
    This is more efficient than brute force for large datasets (O(N log N) vs O(N^2))
    and returns results in a format suitable for DataFrame integration.

    Args:
        items: List of strings to check for duplicates
        algorithm: Similarity algorithm (default: "jaro_winkler")
        min_similarity: Minimum similarity threshold (0.0 to 1.0, default: 0.85)
        window_size: Window size for SNM comparison (default: 50)
        normalize: Normalization mode to apply before comparison:
            - "none": No normalization
            - "lowercase": Convert to lowercase (default)
            - "unicode_nfkd" or "nfkd": Apply Unicode NFKD normalization
            - "remove_punctuation": Remove ASCII punctuation
            - "remove_whitespace": Remove all whitespace
            - "strict": Apply all normalizations

    Returns:
        List of (idx_a, idx_b, score) tuples for each pair of similar items,
        where idx_a < idx_b (smaller index always first for consistency).

    Example:
        >>> items = ["hello", "hallo", "world", "word"]
        >>> pairs = find_duplicate_pairs(items, min_similarity=0.8)
        >>> for a, b, score in pairs:
        ...     print(f"{items[a]} ~ {items[b]}: {score:.2f}")
        hello ~ hallo: 0.88
        world ~ word: 0.75
    """
    ...

def find_duplicates(
    items: list[str],
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.85,
    normalize: str = "lowercase",
    method: str = "auto",
    window_size: int = 50,
) -> DeduplicationResult:
    """
    Find duplicate items in a list using the specified similarity algorithm.

    Groups similar strings together using Union-Find clustering. For large datasets
    (> 2000 items), automatically switches to Sorted Neighborhood Method (SNM)
    for better performance.

    Args:
        items: List of strings to deduplicate
        algorithm: Similarity algorithm (default: "jaro_winkler"). Options:
            - "levenshtein", "damerau_levenshtein", "jaro", "jaro_winkler"
            - "ngram", "bigram", "trigram", "cosine", "lcs"
        min_similarity: Minimum similarity score to consider items as duplicates
            (0.0 to 1.0, default: 0.85)
        normalize: Normalization mode to apply before comparison:
            - "none": No normalization
            - "lowercase": Convert to lowercase (default)
            - "unicode_nfkd": Apply Unicode NFKD normalization
            - "remove_punctuation": Remove ASCII punctuation
            - "remove_whitespace": Remove all whitespace
            - "strict": Apply all normalizations
        method: Deduplication method:
            - "auto": Automatically choose based on dataset size (default)
            - "brute_force": O(N^2) pairwise comparison (accurate but slow)
            - "snm": Sorted Neighborhood Method O(N log N) (fast for large datasets)
        window_size: Window size for SNM method (default: 50)

    Returns:
        DeduplicationResult with:
            - groups: List of duplicate groups (each group is a list of strings)
            - unique: List of strings that have no duplicates
            - total_duplicates: Total count of duplicate strings found

    Warning:
        For datasets with many similar items, memory usage can be high as all
        similar pairs are collected. Consider using smaller batches for
        datasets > 100K items.

    Example:
        >>> result = find_duplicates(["hello", "Hello", "HELLO", "world"])
        >>> result.groups
        [['hello', 'Hello', 'HELLO']]
        >>> result.unique
        ['world']
        >>> result.total_duplicates
        3
    """
    ...

# =============================================================================
# Evaluation Metrics
# =============================================================================

class ConfusionMatrixResult:
    """
    Result from confusion matrix calculation.

    Attributes:
        tp: True positives - correctly predicted matches
        fp: False positives - incorrectly predicted matches
        fn_count: False negatives - missed matches
        tn: True negatives - correctly rejected non-matches
    """

    tp: int
    fp: int
    fn_count: int
    tn: int

    def __init__(self, tp: int, fp: int, fn_count: int, tn: int) -> None: ...
    def precision(self) -> float:
        """Calculate precision from confusion matrix: TP / (TP + FP)."""
        ...

    def recall(self) -> float:
        """Calculate recall from confusion matrix: TP / (TP + FN)."""
        ...

    def f_score(self, beta: float = 1.0) -> float:
        """Calculate F-beta score from confusion matrix."""
        ...

def precision(
    true_matches: list[tuple[int, int]],
    predicted_matches: list[tuple[int, int]],
) -> float:
    """
    Compute precision: TP / (TP + FP).

    Precision measures the accuracy of positive predictions.
    A precision of 1.0 means no false positives.

    Args:
        true_matches: List of actual match pairs (ground truth) as (id1, id2) tuples
        predicted_matches: List of predicted match pairs as (id1, id2) tuples

    Returns:
        Precision score between 0.0 and 1.0

    Example:
        >>> true_matches = [(0, 1), (1, 2)]
        >>> predicted = [(0, 1), (2, 3)]  # One correct, one wrong
        >>> precision(true_matches, predicted)
        0.5
    """
    ...

def recall(
    true_matches: list[tuple[int, int]],
    predicted_matches: list[tuple[int, int]],
) -> float:
    """
    Compute recall: TP / (TP + FN).

    Recall measures the completeness of positive predictions.
    A recall of 1.0 means no false negatives (all true matches found).

    Args:
        true_matches: List of actual match pairs (ground truth) as (id1, id2) tuples
        predicted_matches: List of predicted match pairs as (id1, id2) tuples

    Returns:
        Recall score between 0.0 and 1.0

    Example:
        >>> true_matches = [(0, 1), (1, 2)]
        >>> predicted = [(0, 1)]  # Found one, missed one
        >>> recall(true_matches, predicted)
        0.5
    """
    ...

def f_score(
    true_matches: list[tuple[int, int]],
    predicted_matches: list[tuple[int, int]],
    beta: float = 1.0,
) -> float:
    """
    Compute F-beta score: weighted harmonic mean of precision and recall.

    F1 score (beta=1.0) gives equal weight to precision and recall.
    F0.5 (beta=0.5) weighs precision higher than recall.
    F2 (beta=2.0) weighs recall higher than precision.

    Args:
        true_matches: List of actual match pairs (ground truth) as (id1, id2) tuples
        predicted_matches: List of predicted match pairs as (id1, id2) tuples
        beta: Weight parameter (default 1.0 for F1 score)

    Returns:
        F-beta score between 0.0 and 1.0

    Example:
        >>> true_matches = [(0, 1)]
        >>> predicted = [(0, 1)]  # Perfect match
        >>> f_score(true_matches, predicted)
        1.0
        >>> f_score(true_matches, predicted, beta=0.5)  # Precision-weighted
        1.0
    """
    ...

def confusion_matrix(
    true_matches: list[tuple[int, int]],
    predicted_matches: list[tuple[int, int]],
    total_pairs: int,
) -> ConfusionMatrixResult:
    """
    Compute confusion matrix from match sets.

    Args:
        true_matches: List of actual match pairs (ground truth) as (id1, id2) tuples
        predicted_matches: List of predicted match pairs as (id1, id2) tuples
        total_pairs: Total number of possible pairs (for computing TN).
            For n items, total_pairs = n * (n - 1) / 2

    Returns:
        ConfusionMatrixResult with tp, fp, fn_count, tn counts.
        Also provides precision(), recall(), and f_score(beta) methods.

    Example:
        >>> true_matches = [(0, 1), (1, 2)]
        >>> predicted = [(0, 1), (2, 3)]
        >>> cm = confusion_matrix(true_matches, predicted, total_pairs=10)
        >>> cm.tp, cm.fp, cm.fn_count, cm.tn
        (1, 1, 1, 7)
        >>> cm.precision()
        0.5
        >>> cm.recall()
        0.5
        >>> cm.f_score()
        0.5
    """
    ...

# =============================================================================
# Multi-Algorithm Comparison
# =============================================================================

def compare_algorithms(
    strings: list[str],
    query: str,
    algorithms: list[str] | None = None,
    limit: int = 3,
) -> list[AlgorithmComparison]:
    """
    Compare query against strings using multiple similarity algorithms.

    Args:
        strings: List of strings to compare against
        query: Query string to find matches for
        algorithms: List of algorithm names to use (if None, uses all common algorithms)
        limit: Maximum number of top matches to return per algorithm

    Returns:
        List of AlgorithmComparison objects, one for each algorithm, sorted by average score.

    Example:
        >>> results = compare_algorithms(["hello", "hallo"], "helo")
        >>> for r in results:
        ...     print(f"{r.algorithm}: {r.score:.3f}")
        jaro_winkler: 0.881
        levenshtein: 0.575
    """
    ...

# =============================================================================
# TF-IDF Cosine Similarity
# =============================================================================

class TfIdfCosine:
    """
    TF-IDF weighted cosine similarity for corpus-based matching.

    Builds a corpus of documents and uses TF-IDF weighting for similarity.
    Words that appear in fewer documents get higher weight, improving matching
    for domain-specific or rare terms.

    Example:
        >>> tfidf = TfIdfCosine()
        >>> tfidf.add_documents([
        ...     "the quick brown fox",
        ...     "the lazy dog",
        ...     "quantum physics theory"
        ... ])
        >>> # Common word "the" gets lower weight, rare word "quantum" gets higher
        >>> tfidf.similarity("quick fox", "brown fox")
        0.707...
    """

    def __init__(self) -> None:
        """Create a new TF-IDF corpus."""
        ...

    def add_document(self, doc: str) -> None:
        """
        Add a document to build IDF scores.

        Args:
            doc: Document text (will be tokenized on whitespace)
        """
        ...

    def add_documents(self, docs: list[str]) -> None:
        """
        Add multiple documents to build IDF scores.

        Args:
            docs: List of document texts
        """
        ...

    def similarity(self, a: str, b: str) -> float:
        """
        Calculate TF-IDF weighted cosine similarity between two strings.

        Args:
            a: First string
            b: Second string

        Returns:
            Similarity score (0.0 to 1.0). Rare words contribute more to the score.
        """
        ...

    def num_documents(self) -> int:
        """Get the number of documents in the corpus."""
        ...

# =============================================================================
# Index Classes
# =============================================================================

class BkTree:
    """
    BK-tree (Burkhard-Keller tree) for efficient fuzzy string search.

    Uses metric space properties to prune search space, making fuzzy
    search much faster than linear comparison for large datasets.

    Complexity:
        Build: O(n * m * log(n)) average, where n=items, m=avg string length.
        Search: O(n^(d/D)) average with max_distance=d, tree depth=D.
                Pruning makes this much better than O(n) linear scan.
        Space: O(n * m) for storing strings.

    Note:
        This class is NOT thread-safe. Do not share instances between
        Python threads. Create separate instances for each thread or
        use appropriate synchronization in your code.

    Example:
        >>> tree = BkTree()
        >>> tree.add_all(["hello", "hallo", "hullo", "world"])
        >>> results = tree.search("helo", max_distance=2)
        >>> [r.text for r in results]
        ['hello', 'hallo', 'hullo']
    """

    def __init__(self, use_damerau: bool = False) -> None:
        """
        Create a new BK-tree.

        Args:
            use_damerau: If True, use Damerau-Levenshtein distance
                        (includes transpositions). Default is Levenshtein.
        """
        ...

    def add(self, text: str) -> bool:
        """
        Add a string to the tree.

        Returns:
            True if added successfully, False if duplicate or depth limit exceeded.
        """
        ...

    def add_with_data(self, text: str, data: UserData) -> bool:
        """
        Add a string with associated numeric data.

        Args:
            text: The string to add
            data: User data (must be in u64 range: 0 to 2^64-1)

        Returns:
            True if added successfully, False if duplicate or depth limit exceeded.
        """
        ...

    def add_all(self, texts: Sequence[str]) -> None:
        """Add multiple strings to the tree."""
        ...

    def search(
        self,
        query: str,
        max_distance: int | None = None,
        min_similarity: float | None = None,
        limit: int | None = None,
        normalize: str | None = None,
    ) -> list[SearchResult]:
        """
        Search for strings within a given edit distance or above a similarity threshold.

        Args:
            query: The string to search for
            max_distance: Maximum edit distance (mutually exclusive with min_similarity)
            min_similarity: Minimum similarity threshold 0.0-1.0 (converted to max_distance)
            limit: Maximum number of results to return
            normalize: Optional normalization mode. Use "lowercase" for case-insensitive comparison.

        Returns:
            List of SearchResult objects sorted by distance/similarity.

        Raises:
            ValueError: If neither max_distance nor min_similarity is provided,
                       or if both are provided.
        """
        ...

    def batch_search(
        self,
        queries: Sequence[str],
        max_distance: int,
        limit: int | None = None,
        normalize: str | None = None,
    ) -> list[list[SearchResult]]:
        """
        Search for multiple queries in parallel.

        Args:
            queries: List of query strings
            max_distance: Maximum edit distance
            limit: Maximum number of results per query
            normalize: Optional normalization mode. Use "lowercase" for case-insensitive comparison.

        Returns:
            List of result lists, one per query.
        """
        ...

    def find_nearest(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        """
        Find the nearest neighbors to the query.

        Args:
            query: The string to search for
            limit: Number of nearest neighbors to return

        Returns:
            List of SearchResult objects sorted by distance.
        """
        ...

    def contains(self, query: str) -> bool:
        """Check if the tree contains an exact match."""
        ...

    def remove(self, id: int) -> bool:
        """
        Remove an item by its ID (soft delete using tombstone).

        Returns:
            True if the item was found and deleted, False otherwise.
        """
        ...

    def remove_text(self, text: str) -> bool:
        """
        Remove an item by its text (soft delete using tombstone).

        Returns:
            True if the item was found and deleted, False otherwise.
        """
        ...

    def deleted_count(self) -> int:
        """Get the number of deleted (tombstoned) items."""
        ...

    def active_count(self) -> int:
        """Get the number of active (non-deleted) items."""
        ...

    def compact(self) -> None:
        """Rebuild the tree without deleted items."""
        ...

    def to_bytes(self) -> bytes:
        """
        Serialize the tree to bytes.

        Note: The distance function is not serialized. When deserializing,
        the tree will use the default Levenshtein distance.
        """
        ...

    @staticmethod
    def from_bytes(data: bytes) -> BkTree:
        """
        Deserialize a tree from bytes.

        Note: The deserialized tree will use Levenshtein distance by default.
        """
        ...

    @staticmethod
    def from_bytes_damerau(data: bytes) -> BkTree:
        """Deserialize a tree from bytes using Damerau-Levenshtein distance."""
        ...

    def __len__(self) -> int:
        """Get the number of items in the tree."""
        ...

class NgramIndex:
    """
    N-gram based index for fast fuzzy search.

    Pre-indexes strings by their n-grams for O(1) candidate lookup.
    Faster than BK-tree for large datasets when combined with
    similarity scoring.

    The min_ngram_ratio parameter controls candidate filtering:
    - Higher values (0.3-0.5) reduce false positives but may miss some matches
    - Lower values (0.1-0.2) allow more candidates through for scoring

    Complexity:
        Build: O(n * m) where n=items, m=avg string length.
        Search: O(g + c * m) where g=query n-grams, c=candidates, m=string length.
                Typically much faster than O(n) for selective queries.
        Space: O(n * m) for strings + O(n * g) for inverted index.

    Note:
        This class is NOT thread-safe. Do not share instances between
        Python threads. Create separate instances for each thread or
        use appropriate synchronization in your code.

    Example:
        >>> index = NgramIndex(ngram_size=2, min_ngram_ratio=0.2)
        >>> index.add_all(["apple", "application", "apply", "banana"])
        >>> results = index.search("appel", algorithm="jaro_winkler", min_similarity=0.7)
        >>> [(r.text, r.score) for r in results]
        [('apple', 0.93), ('apply', 0.84)]
    """

    def __init__(
        self,
        ngram_size: int = 3,
        min_ngram_ratio: float = 0.2,
        normalize: bool = True,
    ) -> None:
        """
        Create a new n-gram index.

        Args:
            ngram_size: Size of n-grams (2 for bigrams, 3 for trigrams)
            min_ngram_ratio: Minimum ratio of query n-grams that must match
                for an entry to be considered a candidate (0.0 to 1.0).
                Default 0.2 means at least 20% of query n-grams must match.
            normalize: Whether to normalize text during indexing. When True,
                text is lowercased for n-gram generation, improving candidate
                retrieval for mixed-case text. This affects which candidates
                are found, not the final similarity score computation.
                See also: `case_insensitive` parameter in search methods.
        """
        ...

    def add(self, text: str) -> int:
        """Add a string and return its ID."""
        ...

    def add_with_data(self, text: str, data: UserData) -> int:
        """
        Add a string with associated data and return its ID.

        Args:
            text: The string to add
            data: User data (must be in u64 range: 0 to 2^64-1)
        """
        ...

    def add_all(self, texts: Sequence[str]) -> None:
        """Add multiple strings."""
        ...

    def search(
        self,
        query: str,
        algorithm: str = "jaro_winkler",
        min_similarity: float = 0.0,
        limit: int | None = None,
        normalize: str | None = "lowercase",
    ) -> list[SearchResult]:
        """
        Search with similarity scoring.

        Args:
            query: Query string
            algorithm: "jaro_winkler", "jaro", "levenshtein", "ngram", "trigram"
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            limit: Maximum results to return
            normalize: Optional normalization mode. Use "lowercase" for case-insensitive
                comparison (default). Use None for case-sensitive comparison.

        Returns:
            List of SearchResult objects sorted by similarity.
        """
        ...

    def batch_search(
        self,
        queries: Sequence[str],
        algorithm: str = "jaro_winkler",
        min_similarity: float = 0.0,
        limit: int | None = None,
        normalize: str | None = "lowercase",
    ) -> list[list[SearchResult]]:
        """Search for multiple queries in parallel."""
        ...

    def find_nearest(
        self,
        query: str,
        limit: int,
        algorithm: str = "jaro_winkler",
        normalize: str | None = "lowercase",
    ) -> list[SearchResult]:
        """
        Find the nearest neighbors by similarity.

        Args:
            query: The string to search for
            limit: Number of nearest neighbors to return
            algorithm: Similarity algorithm to use
            normalize: Optional normalization mode. Use "lowercase" for case-insensitive
                comparison (default). Use None for case-sensitive comparison.

        Returns:
            Up to limit SearchResult objects sorted by similarity.
        """
        ...

    def contains(self, query: str) -> bool:
        """Check if the index contains an exact match."""
        ...

    def get_candidates(self, query: str) -> list[tuple[int, str]]:
        """Get all candidates that share n-grams with query."""
        ...

    def to_bytes(self) -> bytes:
        """Serialize the index to bytes."""
        ...

    @staticmethod
    def from_bytes(data: bytes) -> NgramIndex:
        """Deserialize an index from bytes."""
        ...

    def __len__(self) -> int:
        """Get the number of indexed items."""
        ...

    def compress(self) -> None:
        """
        Compress the posting lists to reduce memory usage.

        This converts the uncompressed posting lists to delta + varint encoded
        format, typically achieving 50-70% memory reduction for large indices.

        After compression:
        - The original uncompressed index is cleared to free memory
        - Searches will use the compressed index (slightly slower decode)
        - New items cannot be added (call decompress() first)

        Example:
            >>> index = NgramIndex()
            >>> index.add_all(large_dataset)
            >>> index.compress()  # Reduce memory usage
            >>> # Searches still work, using compressed index
        """
        ...

    def decompress(self) -> None:
        """
        Decompress the posting lists to allow adding new items.

        This restores the uncompressed index from the compressed format.
        Call this before adding new items to a compressed index.
        """
        ...

    def is_compressed(self) -> bool:
        """Check if the index is currently compressed."""
        ...

    def compression_stats(self) -> tuple[int, int, float] | None:
        """
        Get compression statistics.

        Returns:
            A tuple of (compressed_bytes, uncompressed_bytes, compression_ratio),
            or None if the index is not compressed.
        """
        ...

class HybridIndex:
    """
    Hybrid index combining n-gram indexing with similarity search.

    Optimized for the best balance of speed and accuracy. Suitable for
    datasets with 100K+ items where both speed and accuracy matter.

    The min_ngram_ratio parameter controls candidate filtering:
    - Higher values (0.3-0.5) reduce false positives but may miss some matches
    - Lower values (0.1-0.2) allow more candidates through for scoring

    Complexity:
        Build: O(n * m) where n=items, m=avg string length.
        Search: O(g + c * m) where g=query n-grams, c=candidates, m=string length.
                Combines n-gram filtering with Jaro-Winkler scoring.
        Space: O(n * m) for strings + O(n * g) for inverted index.

    Note:
        This class is NOT thread-safe. Do not share instances between
        Python threads. Create separate instances for each thread or
        use appropriate synchronization in your code.

    Example:
        >>> index = HybridIndex(ngram_size=3, min_ngram_ratio=0.2)
        >>> index.add_all(["apple", "application", "apply", "banana"])
        >>> results = index.search("appel", min_similarity=0.7, limit=3)
        >>> [r.text for r in results]
        ['apple', 'apply', 'application']
    """

    def __init__(
        self,
        ngram_size: int = 3,
        min_ngram_ratio: float = 0.2,
        normalize: bool = True,
    ) -> None:
        """
        Create a new hybrid index.

        Args:
            ngram_size: Size of n-grams for candidate filtering (default 3)
            min_ngram_ratio: Minimum ratio of query n-grams that must match
                for an entry to be considered a candidate (0.0 to 1.0).
                Default 0.2 means at least 20% of query n-grams must match.
            normalize: Whether to normalize text during indexing. When True,
                text is lowercased for n-gram generation, improving candidate
                retrieval for mixed-case text. This affects which candidates
                are found, not the final similarity score computation.
                See also: `case_insensitive` parameter in search methods.
        """
        ...

    def add(self, text: str) -> int:
        """Add a string and return its ID."""
        ...

    def add_with_data(self, text: str, data: UserData) -> int:
        """
        Add a string with associated data and return its ID.

        Args:
            text: The string to add
            data: User data (must be in u64 range: 0 to 2^64-1)
        """
        ...

    def add_all(self, texts: Sequence[str]) -> None:
        """Add multiple strings to the index."""
        ...

    def search(
        self,
        query: str,
        algorithm: str = "jaro_winkler",
        min_similarity: float = 0.0,
        limit: int | None = None,
        normalize: str | None = "lowercase",
    ) -> list[SearchResult]:
        """
        Search for similar strings.

        Args:
            query: Query string to search for
            algorithm: Similarity algorithm ("jaro_winkler", "jaro", "levenshtein")
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            limit: Maximum number of results to return
            normalize: Optional normalization mode. Use "lowercase" for case-insensitive
                comparison (default). Use None for case-sensitive comparison.

        Returns:
            List of SearchResult objects sorted by similarity (highest first).
        """
        ...

    def batch_search(
        self,
        queries: Sequence[str],
        algorithm: str = "jaro_winkler",
        min_similarity: float = 0.0,
        limit: int | None = None,
        normalize: str | None = "lowercase",
    ) -> list[list[SearchResult]]:
        """
        Search for multiple queries in parallel.

        Args:
            queries: List of query strings
            algorithm: Similarity algorithm to use
            min_similarity: Minimum similarity score
            limit: Maximum results per query
            normalize: Optional normalization mode. Use "lowercase" for case-insensitive
                comparison (default). Use None for case-sensitive comparison.

        Returns:
            List of result lists, one per query.
        """
        ...

    def find_nearest(
        self,
        query: str,
        limit: int | None = None,
        algorithm: str = "jaro_winkler",
        normalize: str | None = "lowercase",
        *,
        k: int | None = None,
    ) -> list[SearchResult]:
        """
        Find the nearest neighbors by similarity.

        Args:
            query: Query string
            limit: Number of nearest neighbors to return
            algorithm: Similarity algorithm to use
            normalize: Optional normalization mode. Use "lowercase" for case-insensitive
                comparison (default). Use None for case-sensitive comparison.
            k: Deprecated alias for limit

        Returns:
            Up to limit SearchResult objects sorted by similarity.
        """
        ...

    def contains(self, query: str) -> bool:
        """Check if the index contains an exact match for the query."""
        ...

    def to_bytes(self) -> bytes:
        """Serialize the index to bytes."""
        ...

    @staticmethod
    def from_bytes(data: bytes) -> HybridIndex:
        """Deserialize an index from bytes."""
        ...

    def __len__(self) -> int:
        """Get the number of items in the index."""
        ...

# =============================================================================
# Thread-Safe Index Classes
# =============================================================================

class ThreadSafeBkTree:
    """
    Thread-safe BK-tree for concurrent fuzzy string search.

    Uses a read-write lock internally to allow safe concurrent access from
    multiple Python threads. Multiple readers can search simultaneously,
    while writers get exclusive access.

    Note:
        Thread safety has some overhead. For single-threaded use, prefer BkTree.
        For parallel processing with separate indices per thread, prefer BkTree.

    Example:
        >>> tree = ThreadSafeBkTree()
        >>> tree.add_all(["hello", "hallo", "world"])
        >>> # Safe to search from multiple threads
        >>> results = tree.search("helo", max_distance=2)
    """

    def __init__(self, use_damerau: bool = False) -> None:
        """
        Create a new thread-safe BK-tree.

        Args:
            use_damerau: If True, use Damerau-Levenshtein distance
        """
        ...

    def add(self, text: str) -> bool:
        """Add a string to the tree. Thread-safe."""
        ...

    def add_all(self, texts: Sequence[str]) -> None:
        """Add multiple strings to the tree. Thread-safe."""
        ...

    def search(
        self,
        query: str,
        max_distance: int | None = None,
        min_similarity: float | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search for strings within a given edit distance. Thread-safe."""
        ...

    def contains(self, query: str) -> bool:
        """Check if the tree contains an exact match. Thread-safe."""
        ...

    def __len__(self) -> int:
        """Get the number of items in the tree."""
        ...

class ThreadSafeNgramIndex:
    """
    Thread-safe N-gram index for concurrent fuzzy string search.

    Uses a read-write lock internally to allow safe concurrent access from
    multiple Python threads.

    Example:
        >>> index = ThreadSafeNgramIndex(ngram_size=3)
        >>> index.add_all(["apple", "application", "banana"])
        >>> # Safe to search from multiple threads
        >>> results = index.search("appel", min_similarity=0.7)
    """

    def __init__(
        self,
        ngram_size: int = 3,
        min_similarity: float = 0.0,
        min_ngram_ratio: float = 0.0,
        case_insensitive: bool = False,
    ) -> None:
        """Create a new thread-safe N-gram index."""
        ...

    def add(self, text: str) -> int:
        """Add a string and return its ID. Thread-safe."""
        ...

    def add_all(self, texts: Sequence[str]) -> None:
        """Add multiple strings. Thread-safe."""
        ...

    def search(
        self,
        query: str,
        algorithm: str = "jaro_winkler",
        min_similarity: float = 0.0,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search with similarity scoring. Thread-safe."""
        ...

    def contains(self, query: str) -> bool:
        """Check if the index contains an exact match. Thread-safe."""
        ...

    def __len__(self) -> int:
        """Get the number of indexed items."""
        ...

# =============================================================================
# Sharded Index Classes
# =============================================================================

class ShardedBkTree:
    """
    Sharded BK-tree for parallel fuzzy string search on large datasets.

    Distributes items across multiple BK-tree shards. Searches are performed
    in parallel across all shards, providing better performance for very
    large datasets (millions of items).

    Example:
        >>> tree = ShardedBkTree(num_shards=8)
        >>> tree.add_all(large_dataset)  # Items distributed across 8 shards
        >>> results = tree.search("query", max_distance=2)  # Parallel search
    """

    def __init__(self, num_shards: int = 8, use_damerau: bool = False) -> None:
        """
        Create a new sharded BK-tree.

        Args:
            num_shards: Number of shards to use (default: 8)
            use_damerau: If True, use Damerau-Levenshtein distance
        """
        ...

    def add(self, text: str) -> bool:
        """Add a string to the tree (assigned to a shard based on hash)."""
        ...

    def add_all(self, texts: Sequence[str]) -> None:
        """Add multiple strings, distributing across shards."""
        ...

    def search(
        self,
        query: str,
        max_distance: int | None = None,
        min_similarity: float | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search across all shards in parallel."""
        ...

    def contains(self, query: str) -> bool:
        """Check if any shard contains an exact match."""
        ...

    def __len__(self) -> int:
        """Get the total number of items across all shards."""
        ...

class ShardedNgramIndex:
    """
    Sharded N-gram index for parallel fuzzy string search on large datasets.

    Distributes items across multiple N-gram index shards. Searches are
    performed in parallel across all shards.

    Example:
        >>> index = ShardedNgramIndex(num_shards=8, ngram_size=3)
        >>> index.add_all(large_dataset)
        >>> results = index.search("query", min_similarity=0.7)
    """

    def __init__(
        self,
        num_shards: int = 8,
        ngram_size: int = 3,
        min_similarity: float = 0.0,
        min_ngram_ratio: float = 0.2,
        normalize: bool = True,
    ) -> None:
        """
        Create a new sharded N-gram index.

        Args:
            num_shards: Number of shards to use (default: 8)
            ngram_size: Size of n-grams (default: 3)
            min_similarity: Minimum similarity for candidate filtering
            min_ngram_ratio: Minimum ratio of matching n-grams
            normalize: Whether to normalize text during indexing
        """
        ...

    def add(self, text: str) -> int:
        """Add a string (assigned to a shard based on hash)."""
        ...

    def add_all(self, texts: Sequence[str]) -> None:
        """Add multiple strings, distributing across shards."""
        ...

    def search(
        self,
        query: str,
        algorithm: str = "jaro_winkler",
        min_similarity: float = 0.0,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search across all shards in parallel."""
        ...

    def contains(self, query: str) -> bool:
        """Check if any shard contains an exact match."""
        ...

    def __len__(self) -> int:
        """Get the total number of items across all shards."""
        ...

# =============================================================================
# High-Level FuzzyIndex Wrapper
# =============================================================================

class FuzzyIndex:
    """
    A reusable fuzzy matching index for efficient batch operations.

    FuzzyIndex wraps fuzzyrust's underlying index structures (NgramIndex,
    BKTree, or HybridIndex) and provides a convenient API for building
    indices from Polars Series and performing batch searches.

    The index can be persisted to disk and reloaded for later use.

    Warning:
        This class is NOT thread-safe. Create separate instances for each
        thread when using in concurrent applications.

    Example:
        >>> import polars as pl
        >>> from fuzzyrust import FuzzyIndex
        >>>
        >>> # Build index from a Series
        >>> names = pl.Series(["Apple Inc", "Microsoft Corp", "Google LLC"])
        >>> index = FuzzyIndex.from_series(names, algorithm="ngram")
        >>>
        >>> # Search for similar strings
        >>> queries = pl.Series(["Apple", "Microsft"])
        >>> results = index.search_series(queries, min_similarity=0.6)
        >>>
        >>> # Save for later reuse
        >>> index.save("names_index.pkl")
        >>> index = FuzzyIndex.load("names_index.pkl")
    """

    def __init__(
        self,
        items: list[str],
        algorithm: str = "ngram",
        ngram_size: int = 3,
    ) -> None:
        """
        Create a FuzzyIndex from a list of strings.

        Args:
            items: List of strings to index
            algorithm: Index algorithm to use:
                - "ngram": N-gram based inverted index (fast for fuzzy matching)
                - "bktree": BK-tree for metric space (good for edit distance)
                - "hybrid": Combines ngram and bktree for best of both
            ngram_size: Size of n-grams (only for "ngram" and "hybrid")
        """
        ...

    @classmethod
    def from_series(
        cls,
        series: Any,  # pl.Series
        algorithm: str = "ngram",
        ngram_size: int = 3,
    ) -> FuzzyIndex:
        """
        Create a FuzzyIndex from a Polars Series.

        Args:
            series: Polars Series of strings to index
            algorithm: Index algorithm ("ngram", "bktree", or "hybrid")
            ngram_size: Size of n-grams (only for "ngram" and "hybrid")

        Returns:
            FuzzyIndex instance
        """
        ...

    @classmethod
    def from_dataframe(
        cls,
        df: Any,  # pl.DataFrame
        column: str,
        algorithm: str = "ngram",
        ngram_size: int = 3,
    ) -> FuzzyIndex:
        """
        Create a FuzzyIndex from a DataFrame column.

        Args:
            df: Polars DataFrame
            column: Column name to index
            algorithm: Index algorithm ("ngram", "bktree", or "hybrid")
            ngram_size: Size of n-grams (only for "ngram" and "hybrid")

        Returns:
            FuzzyIndex instance
        """
        ...

    def search(
        self,
        query: str,
        min_similarity: float = 0.0,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Search the index for strings similar to the query.

        Args:
            query: Query string to search for
            min_similarity: Minimum similarity score (0.0 to 1.0)
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects with text, score, and index
        """
        ...

    def search_series(
        self,
        queries: Any,  # pl.Series
        min_similarity: float = 0.0,
        limit: int = 1,
        include_query: bool = True,
    ) -> Any:  # pl.DataFrame
        """
        Search for each query in a Series, returning a DataFrame of results.

        Args:
            queries: Series of query strings
            min_similarity: Minimum similarity score (0.0 to 1.0)
            limit: Maximum matches per query (default: 1 for best match only)
            include_query: Include query column in results

        Returns:
            DataFrame with columns: query_idx, query (optional), match, match_idx, score
        """
        ...

    def batch_search(
        self,
        queries: list[str],
        min_similarity: float = 0.0,
        limit: int = 1,
    ) -> list[list[SearchResult]]:
        """
        Search for multiple queries, returning results for each.

        Args:
            queries: List of query strings
            min_similarity: Minimum similarity score (0.0 to 1.0)
            limit: Maximum matches per query

        Returns:
            List of lists, where each inner list contains SearchResult
            objects for the corresponding query
        """
        ...

    def get_items(self) -> list[str]:
        """Return the list of indexed items."""
        ...

    def save(self, path: str | Any) -> None:  # path: str | Path
        """
        Save the index to a file.

        Args:
            path: File path to save to (typically .pkl extension)
        """
        ...

    @classmethod
    def load(cls, path: str | Any) -> FuzzyIndex:  # path: str | Path
        """
        Load an index from a file.

        Args:
            path: File path to load from

        Returns:
            FuzzyIndex instance
        """
        ...

    def __len__(self) -> int:
        """Return the number of indexed items."""
        ...

    def __repr__(self) -> str: ...

# =============================================================================
# Schema-Based Multi-Field Matching
# =============================================================================

class SchemaSearchResult:
    """
    Result from multi-field schema search.

    Attributes:
        id: Record ID
        score: Overall combined similarity score (0.0 to 1.0)
        field_scores: Dictionary mapping field names to their individual scores
        record: Dictionary containing all field values for this record
        data: Optional user-defined data (must be in u64 range: 0 to 2^64-1)
    """

    id: int
    score: float
    field_scores: dict[str, float]
    record: dict[str, str]
    data: UserData | None

    def __init__(
        self,
        id: int,
        score: float,
        field_scores: dict[str, float],
        record: dict[str, str],
        data: UserData | None = None,
    ) -> None: ...

class SchemaBuilder:
    """
    Builder for creating multi-field search schemas.

    Use the fluent API to define fields with types, algorithms, weights, and constraints.

    Example:
        >>> builder = SchemaBuilder()
        >>> builder.add_field(
        ...     name="name",
        ...     field_type="short_text",
        ...     algorithm="jaro_winkler",
        ...     weight=10.0,
        ...     required=True
        ... )
        >>> builder.with_scoring("weighted_average")
        >>> schema = builder.build()
    """

    def __init__(self) -> None:
        """Create a new schema builder."""
        ...

    def add_field(
        self,
        name: str,
        field_type: str,
        algorithm: str | None = None,
        weight: float = 1.0,
        required: bool = False,
        normalize: str | None = None,
        max_length: int | None = None,
        separator: str | None = None,
        chunk_size: int | None = None,
    ) -> None:
        """
        Add a field to the schema.

        Args:
            name: Field name (must be unique)
            field_type: One of "short_text", "long_text", "token_set"
            algorithm: Similarity algorithm for this field. If None, defaults based on field_type:
                      - short_text: "jaro_winkler"
                      - long_text: "ngram"
                      - token_set: "jaccard"
                      Valid algorithms:
                      - For short_text: "levenshtein", "damerau_levenshtein", "jaro_winkler", "ngram", "exact_match"
                      - For long_text: "ngram", "cosine", "levenshtein", "exact_match"
                      - For token_set: "jaccard" (always uses Jaccard)
            weight: Relative importance of this field (0.0 to 10.0, default 1.0)
            required: If True, records without this field will be rejected
            normalize: Normalization mode: "lowercase", "unicode_nfkd", "remove_punctuation",
                      "remove_whitespace", "strict" (all normalizations)
            max_length: For short_text only - maximum expected length (default 100)
            separator: For token_set only - separator character (default ',')
            chunk_size: For long_text only - optional chunk size for processing

        Raises:
            ValueError: If field configuration is invalid
        """
        ...

    def with_scoring(self, strategy: str = "weighted_average") -> None:
        """
        Set the scoring strategy for combining field scores.

        Args:
            strategy: One of "weighted_average" (default) or "minmax_scaling"

        Returns:
            None (modifies builder in place)
        """
        ...

    def build(self) -> Schema:
        """
        Build and validate the schema.

        Returns:
            A frozen Schema object ready for use with SchemaIndex.

        Raises:
            ValueError: If schema validation fails (e.g., no fields, duplicate names)
        """
        ...

class Schema:
    """
    Immutable schema definition for multi-field fuzzy matching.

    Schemas are created using SchemaBuilder and define the structure,
    algorithms, and scoring for multi-field searches.

    Note:
        Schemas are frozen after creation and cannot be modified.
        Use SchemaBuilder to create new schemas.
    """

    def field_names(self) -> list[str]:
        """Get list of all field names in the schema."""
        ...

    def __repr__(self) -> str: ...

class SchemaIndex:
    """
    Multi-field fuzzy matching index with schema-based type safety.

    Provides efficient searching across multiple fields with different types,
    algorithms, and weights. Each field uses an optimized index structure
    based on its type:
    - ShortText: N-gram index with Jaro-Winkler, Levenshtein, etc.
    - LongText: Hybrid index with N-gram, Cosine similarity
    - TokenSet: Inverted index with Jaccard similarity

    Note:
        This class is NOT thread-safe. Do not share instances between
        Python threads. Create separate instances for each thread.

    Example:
        >>> # Define schema
        >>> builder = SchemaBuilder()
        >>> builder.add_field("name", "short_text", algorithm="jaro_winkler", weight=10, required=True)
        >>> builder.add_field("description", "long_text", algorithm="ngram", weight=5)
        >>> builder.add_field("tags", "token_set", algorithm="jaccard", weight=7, separator=",")
        >>> schema = builder.build()
        >>>
        >>> # Create index and add records
        >>> index = SchemaIndex(schema)
        >>> index.add({
        ...     "name": "MacBook Pro",
        ...     "description": "High-performance laptop",
        ...     "tags": "laptop,apple,computing"
        ... }, data=12345)
        >>>
        >>> # Search across multiple fields
        >>> results = index.search({
        ...     "name": "Macbook",
        ...     "tags": "laptop"
        ... }, min_similarity=0.5, limit=10)
        >>>
        >>> for r in results:
        ...     print(f"Score: {r.score:.3f}, Name: {r.record['name']}")
    """

    def __init__(self, schema: Schema) -> None:
        """
        Create a new schema-based index.

        Args:
            schema: The schema defining field types and algorithms
        """
        ...

    def add(
        self,
        record: dict[str, str],
        data: UserData | None = None,
    ) -> int:
        """
        Add a record to the index.

        Args:
            record: Dictionary mapping field names to values
            data: Optional user-defined data (e.g., database ID)

        Returns:
            The record ID assigned by the index

        Raises:
            ValueError: If record fails schema validation (missing required fields, etc.)

        Example:
            >>> index.add({
            ...     "name": "Product Name",
            ...     "description": "Product description",
            ...     "tags": "tag1,tag2,tag3"
            ... }, data=42)
            0
        """
        ...

    def get(self, id: int) -> dict[str, str] | None:
        """
        Retrieve a record by its ID.

        Args:
            id: The record ID returned from add()

        Returns:
            Dictionary with field values, or None if ID doesn't exist

        Example:
            >>> record = index.get(0)
            >>> if record:
            ...     print(record['name'])
        """
        ...

    def search(
        self,
        query: dict[str, str],
        min_similarity: float = 0.0,
        limit: int | None = None,
        min_field_similarity: float = 0.0,
        field_boosts: dict[str, float] | None = None,
    ) -> list[SchemaSearchResult]:
        """
        Search for records matching the query across multiple fields.

        Each field in the query is matched against the corresponding field in indexed records
        using the algorithm defined in the schema. Field scores are combined using the
        schema's scoring strategy (e.g., weighted average).

        Args:
            query: Dictionary mapping field names to query values
                  (only fields present in query are searched)
            min_similarity: Minimum combined score to include in results (0.0 to 1.0)
            limit: Maximum number of results to return (None for unlimited)
            min_field_similarity: Minimum score for individual fields (0.0 to 1.0).
                           Fields with scores below this threshold are excluded
            field_boosts: Optional dictionary of field name to boost multiplier.
                Boosts are multiplied with schema weights at query time.
                Example: {"name": 2.0, "city": 0.5} doubles name's weight, halves city's.
                This allows dynamic re-weighting without rebuilding the index.

        Returns:
            List of SchemaSearchResult objects sorted by descending score.
            Each result includes the overall score, per-field scores, and record data.

        Raises:
            ValueError: If query contains unknown field names

        Example:
            >>> # Basic search
            >>> results = index.search({
            ...     "name": "laptop",
            ...     "tags": "computing,apple"
            ... }, min_similarity=0.7, limit=5)
            >>>
            >>> # With query-time field boosting
            >>> results = index.search(
            ...     {"name": "John", "city": "NYC"},
            ...     field_boosts={"name": 2.0, "city": 0.5}  # Emphasize name matches
            ... )
            >>>
            >>> for r in results:
            ...     print(f"Overall: {r.score:.3f}")
            ...     print(f"  Name score: {r.field_scores['name']:.3f}")
            ...     print(f"  Record: {r.record}")
        """
        ...

    def batch_search(
        self,
        queries: list[dict[str, str]],
        min_similarity: float = 0.0,
        limit: int | None = None,
        min_field_similarity: float = 0.0,
        field_boosts: dict[str, float] | None = None,
    ) -> list[list[SchemaSearchResult]]:
        """
        Search for multiple queries in parallel.

        Executes multiple searches concurrently using parallel processing,
        maintaining the order of results corresponding to input queries.

        Args:
            queries: List of query records (each a dict mapping field names to values)
            min_similarity: Minimum overall similarity score (0.0 to 1.0)
            limit: Maximum number of results per query (None for unlimited)
            min_field_similarity: Minimum per-field similarity threshold (0.0 to 1.0)
            field_boosts: Optional dictionary of field name to boost multiplier.
                Applied to all queries in the batch.

        Returns:
            List of lists, where each inner list contains SchemaSearchResult objects
            for the corresponding query, sorted by descending score.

        Raises:
            ValueError: If any query contains unknown field names

        Example:
            >>> queries = [
            ...     {"name": "laptop", "tags": "computing"},
            ...     {"name": "phone", "tags": "mobile"}
            ... ]
            >>> results = index.batch_search(
            ...     queries,
            ...     min_similarity=0.7,
            ...     limit=5,
            ...     field_boosts={"name": 2.0}  # Boost name field for all queries
            ... )
            >>> for i, query_results in enumerate(results):
            ...     print(f"Query {i}: {len(query_results)} matches")
        """
        ...

    def __len__(self) -> int:
        """Get the number of records in the index."""
        ...

# Convenience aliases - these are function references, not type aliases
def edit_distance(a: str, b: str, max_distance: int | None = None) -> int:
    """Alias for levenshtein(). Compute edit distance between two strings."""
    ...

def similarity(
    a: str,
    b: str,
    prefix_weight: float = 0.1,
    max_prefix_length: int = 4,
) -> float:
    """Alias for jaro_winkler_similarity(). Compute Jaro-Winkler similarity."""
    ...

# =============================================================================
# Batch Operations Module
# =============================================================================

class batch:
    """Batch operations API for list-based string similarity operations."""

    @staticmethod
    def similarity(
        strings: list[str],
        query: str,
        algorithm: str | Algorithm = "jaro_winkler",
    ) -> list[MatchResult]:
        """
        Compute similarity of a query against all strings.

        Args:
            strings: List of strings to compare against the query.
            query: The query string to match.
            algorithm: Similarity algorithm to use (string or Algorithm enum).

        Returns:
            List of MatchResult objects in the same order as input strings.
        """
        ...

    @staticmethod
    def best_matches(
        strings: list[str],
        query: str,
        algorithm: str | Algorithm = "jaro_winkler",
        limit: int = 5,
        min_similarity: float = 0.0,
    ) -> list[MatchResult]:
        """
        Find top N best matches for a query from a list of strings.

        Args:
            strings: List of strings to search.
            query: The query string to match.
            algorithm: Similarity algorithm to use (string or Algorithm enum).
            limit: Maximum number of results to return (default: 5).
            min_similarity: Minimum similarity score to include (default: 0.0).

        Returns:
            List of MatchResult objects sorted by score descending.
        """
        ...

    @staticmethod
    def deduplicate(
        strings: list[str],
        algorithm: str | Algorithm = "jaro_winkler",
        min_similarity: float = 0.8,
        normalize: str | None = None,
    ) -> DeduplicationResult:
        """
        Find duplicate groups in a list of strings.

        Args:
            strings: List of strings to deduplicate.
            algorithm: Similarity algorithm to use (string or Algorithm enum).
            min_similarity: Minimum similarity to consider as duplicates (default: 0.8).
            normalize: Optional normalization mode (default: None).

        Returns:
            DeduplicationResult with groups, unique items, and total_duplicates.
        """
        ...

    @staticmethod
    def pairwise(
        left: list[str],
        right: list[str],
        algorithm: str | Algorithm = "jaro_winkler",
    ) -> list[float]:
        """
        Compute pairwise similarity between two equal-length lists.

        Args:
            left: First list of strings.
            right: Second list of strings (must be same length as left).
            algorithm: Similarity algorithm to use (string or Algorithm enum).

        Returns:
            List of similarity scores (0.0 to 1.0), one for each pair.

        Raises:
            ValueError: If left and right have different lengths.
        """
        ...

    @staticmethod
    def similarity_matrix(
        queries: list[str],
        choices: list[str],
        algorithm: str | Algorithm = "levenshtein",
    ) -> list[list[float]]:
        """
        Compute similarity matrix between all queries and all choices.

        Args:
            queries: First list of strings (rows of output matrix).
            choices: Second list of strings (columns of output matrix).
            algorithm: Similarity algorithm to use (string or Algorithm enum).

        Returns:
            2D list where result[i][j] is the similarity between
            queries[i] and choices[j].
        """
        ...

    @staticmethod
    def distance_matrix(
        queries: list[str],
        choices: list[str],
        algorithm: str | Algorithm = "levenshtein",
    ) -> list[list[float]]:
        """Deprecated: Use similarity_matrix() instead.

        .. deprecated:: 0.3.0
            Use :func:`similarity_matrix` instead.
        """
        ...

# =============================================================================
# Polars Integration Module
# =============================================================================
# Note: The polars module is a subpackage. For detailed type stubs,
# see fuzzyrust/polars/__init__.pyi (if it exists) or the module docstrings.
# This provides basic type information for IDE support.

class polars:
    """Polars integration for fuzzy string matching on DataFrames and Series.

    This module provides fuzzy matching capabilities for Polars DataFrames
    and Series, including joins, deduplication, and similarity scoring.

    See `fuzzyrust.polars` module documentation for complete API details.
    """

    # Import polars types for annotations
    from typing import Literal, Union

    import polars as pl

    DataFrameType = Union[pl.DataFrame, pl.LazyFrame]

    @staticmethod
    def series_similarity(
        left: pl.Series,
        right: pl.Series,
        algorithm: str | Algorithm = "jaro_winkler",
        ngram_size: int = 3,
    ) -> pl.Series:
        """Compute pairwise similarity between two aligned Series."""
        ...

    @staticmethod
    def series_best_match(
        queries: pl.Series,
        targets: list[str],
        algorithm: str | Algorithm = "jaro_winkler",
        min_similarity: float = 0.0,
        limit: int = 1,
        normalize: str | None = None,
    ) -> pl.Series:
        """Find best match for each query from a target list."""
        ...

    @staticmethod
    def series_dedupe(
        series: pl.Series,
        algorithm: str | Algorithm = "jaro_winkler",
        min_similarity: float = 0.85,
        normalize: str = "lowercase",
    ) -> pl.DataFrame:
        """Deduplicate a Series, grouping similar values together."""
        ...

    @staticmethod
    def series_match(
        queries: pl.Series,
        targets: pl.Series,
        algorithm: str | Algorithm = "jaro_winkler",
        min_similarity: float = 0.0,
    ) -> pl.DataFrame:
        """Match each query against all targets."""
        ...

    @staticmethod
    def df_join(
        left: DataFrameType,
        right: DataFrameType,
        on: str | list | None = None,
        left_on: str | None = None,
        right_on: str | None = None,
        algorithm: str | Algorithm = "jaro_winkler",
        min_similarity: float = 0.8,
        how: Literal["inner", "left"] = "inner",
        combine_scores: Literal[
            "weighted_average", "min", "max", "harmonic_mean"
        ] = "weighted_average",
    ) -> pl.DataFrame:
        """Fuzzy join two DataFrames based on string similarity."""
        ...

    @staticmethod
    def df_dedupe(
        df: DataFrameType,
        columns: list[str],
        algorithm: str | Algorithm = "jaro_winkler",
        min_similarity: float = 0.85,
        weights: dict[str, float] | None = None,
        algorithms: dict[str, str | Algorithm] | None = None,
        keep: Literal["first", "last", "most_complete"] = "first",
    ) -> pl.DataFrame:
        """Deduplicate DataFrame rows using fuzzy matching."""
        ...

    @staticmethod
    def df_dedupe_snm(
        df: DataFrameType,
        columns: list[str],
        algorithm: str | Algorithm = "jaro_winkler",
        min_similarity: float = 0.85,
        window_size: int = 10,
        keep: Literal["first", "last", "most_complete"] = "first",
    ) -> pl.DataFrame:
        """Deduplicate DataFrame using Sorted Neighborhood Method (SNM)."""
        ...

    @staticmethod
    def df_match_pairs(
        df: DataFrameType,
        columns: list[str],
        algorithm: str | Algorithm = "jaro_winkler",
        min_similarity: float = 0.8,
        weights: dict[str, float] | None = None,
        algorithms: dict[str, str | Algorithm] | None = None,
    ) -> pl.DataFrame:
        """Find similar pairs within a DataFrame."""
        ...

    @staticmethod
    def df_find_pairs(
        df: DataFrameType,
        columns: list[str],
        algorithm: str | Algorithm = "jaro_winkler",
        min_similarity: float = 0.8,
        weights: dict[str, float] | None = None,
        algorithms: dict[str, str | Algorithm] | None = None,
        method: Literal["snm", "full"] = "snm",
        window_size: int = 10,
        min_field_similarity: float = 0.0,
    ) -> pl.DataFrame:
        """Find all similar pairs in a DataFrame."""
        ...

    @staticmethod
    def df_match_records(
        queries: DataFrameType,
        targets: DataFrameType,
        columns: list[str],
        algorithm: str | Algorithm = "jaro_winkler",
        min_similarity: float = 0.0,
        weights: dict[str, float] | None = None,
        algorithms: dict[str, str | Algorithm] | None = None,
        limit: int = 1,
        min_field_similarity: float = 0.0,
    ) -> pl.DataFrame:
        """Batch match records from two DataFrames."""
        ...
