"""Type stubs for fuzzyrust v0.1.0."""

from typing import List, Optional, Sequence, Union, TypeAlias
from enum import Enum

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
    """
    id: int
    text: str
    score: float
    distance: Optional[int]
    data: Optional[UserData]

    def __init__(
        self,
        id: int,
        text: str,
        score: float,
        distance: Optional[int] = None,
        data: Optional[UserData] = None,
    ) -> None: ...

class MatchResult:
    """Result from find_best_matches and batch operations."""
    text: str
    score: float

    def __init__(self, text: str, score: float) -> None: ...

class DeduplicationResult:
    """Result from deduplication operation."""
    groups: List[List[str]]
    unique: List[str]
    total_duplicates: int

    def __init__(
        self,
        groups: List[List[str]],
        unique: List[str],
        total_duplicates: int,
    ) -> None: ...

class AlgorithmComparison:
    """Result from multi-algorithm comparison."""
    algorithm: str
    score: float
    matches: List[MatchResult]

    def __init__(
        self,
        algorithm: str,
        score: float,
        matches: List[MatchResult],
    ) -> None: ...

# =============================================================================
# Enums
# =============================================================================

class Algorithm(str, Enum):
    """Available similarity algorithms."""
    LEVENSHTEIN = "levenshtein"
    DAMERAU_LEVENSHTEIN = "damerau_levenshtein"
    JARO = "jaro"
    JARO_WINKLER = "jaro_winkler"
    NGRAM = "ngram"
    BIGRAM = "bigram"
    TRIGRAM = "trigram"
    LCS = "lcs"
    COSINE = "cosine"

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

def levenshtein(a: str, b: str, max_distance: Optional[int] = None) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.

    Args:
        a: First string
        b: Second string
        max_distance: Optional maximum distance for early termination.
                     Returns very large value if exceeded.

    Returns:
        The minimum number of single-character edits needed to transform a into b.

    Example:
        >>> levenshtein("kitten", "sitting")
        3
    """
    ...

def levenshtein_similarity(a: str, b: str) -> float:
    """
    Compute normalized Levenshtein similarity (0.0 to 1.0).

    Args:
        a: First string
        b: Second string

    Returns:
        Similarity score where 1.0 means identical strings.
    """
    ...

def damerau_levenshtein(a: str, b: str, max_distance: Optional[int] = None) -> int:
    """
    Compute Damerau-Levenshtein distance (includes transpositions).

    Like Levenshtein but also counts character swaps as a single edit.
    Useful for typo detection where letter swaps are common.

    Example:
        >>> damerau_levenshtein("ca", "ac")  # One transposition
        1
        >>> levenshtein("ca", "ac")  # Two edits without transposition
        2
    """
    ...

def damerau_levenshtein_similarity(a: str, b: str) -> float:
    """Compute normalized Damerau-Levenshtein similarity (0.0 to 1.0)."""
    ...

def jaro_similarity(a: str, b: str) -> float:
    """
    Compute Jaro similarity (0.0 to 1.0).

    Good for short strings and name matching.

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
    """
    ...

def hamming(a: str, b: str) -> int:
    """
    Compute Hamming distance between two equal-length strings.

    Raises:
        ValueError: If strings have different lengths.

    Example:
        >>> hamming("karolin", "kathrin")
        3
    """
    ...

def ngram_similarity(a: str, b: str, ngram_size: int = 2, pad: bool = True) -> float:
    """
    Compute n-gram similarity (Sørensen-Dice coefficient).

    Args:
        a: First string
        b: Second string
        ngram_size: Size of n-grams (default 2 for bigrams)
        pad: Whether to pad strings for edge matching

    Example:
        >>> ngram_similarity("night", "nacht")
        0.5
    """
    ...

def ngram_jaccard(a: str, b: str, ngram_size: int = 2, pad: bool = True) -> float:
    """Compute n-gram Jaccard similarity."""
    ...

def extract_ngrams(s: str, ngram_size: int = 2, pad: bool = True) -> List[str]:
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

def metaphone(s: str, max_length: int = 4) -> str:
    """
    Encode a string using Metaphone algorithm.

    More accurate than Soundex for many cases.
    """
    ...

def metaphone_match(a: str, b: str) -> bool:
    """Check if two strings have the same Metaphone code."""
    ...

def lcs_length(a: str, b: str) -> int:
    """
    Compute the length of the Longest Common Subsequence.

    Example:
        >>> lcs_length("ABCDGH", "AEDFHR")
        3  # ADH
    """
    ...

def lcs_string(a: str, b: str) -> str:
    """Get the actual Longest Common Subsequence string."""
    ...

def lcs_similarity(a: str, b: str) -> float:
    """Compute LCS-based similarity (0.0 to 1.0)."""
    ...

def longest_common_substring_length(a: str, b: str) -> int:
    """Compute the length of the longest common contiguous substring."""
    ...

def longest_common_substring(a: str, b: str) -> str:
    """Get the longest common contiguous substring."""
    ...

def cosine_similarity_chars(a: str, b: str) -> float:
    """Compute character-level cosine similarity."""
    ...

def cosine_similarity_words(a: str, b: str) -> float:
    """Compute word-level cosine similarity."""
    ...

def cosine_similarity_ngrams(a: str, b: str, ngram_size: int = 2) -> float:
    """Compute n-gram cosine similarity."""
    ...

# =============================================================================
# Case-Insensitive Variants
# =============================================================================

def levenshtein_ci(a: str, b: str, max_distance: Optional[int] = None) -> int:
    """Case-insensitive Levenshtein distance."""
    ...

def levenshtein_similarity_ci(a: str, b: str) -> float:
    """Case-insensitive Levenshtein similarity."""
    ...

def damerau_levenshtein_ci(a: str, b: str, max_distance: Optional[int] = None) -> int:
    """Case-insensitive Damerau-Levenshtein distance."""
    ...

def damerau_levenshtein_similarity_ci(a: str, b: str) -> float:
    """Case-insensitive Damerau-Levenshtein similarity."""
    ...

def jaro_similarity_ci(a: str, b: str) -> float:
    """Case-insensitive Jaro similarity."""
    ...

def jaro_winkler_similarity_ci(
    a: str,
    b: str,
    prefix_weight: float = 0.1,
    max_prefix_length: int = 4,
) -> float:
    """Case-insensitive Jaro-Winkler similarity."""
    ...

def ngram_similarity_ci(a: str, b: str, ngram_size: int = 2, pad: bool = True) -> float:
    """Case-insensitive n-gram similarity."""
    ...

# =============================================================================
# String Normalization
# =============================================================================

def normalize_string(s: str, mode: Union[str, NormalizationMode]) -> str:
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

def normalize_pair(a: str, b: str, mode: Union[str, NormalizationMode]) -> tuple[str, str]:
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
# Batch Processing Functions
# =============================================================================

def batch_levenshtein(strings: List[str], query: str) -> List[MatchResult]:
    """
    Compute Levenshtein distances for all strings against a query in parallel.

    Args:
        strings: List of strings to compare
        query: Query string to compare against

    Returns:
        List of MatchResult objects in the same order as input strings.
    """
    ...

def batch_jaro_winkler(strings: List[str], query: str) -> List[MatchResult]:
    """Compute Jaro-Winkler similarities for all strings in parallel."""
    ...

def find_best_matches(
    strings: List[str],
    query: str,
    algorithm: str = "jaro_winkler",
    limit: int = 10,
    min_similarity: float = 0.0,
) -> List[MatchResult]:
    """
    Find best matching strings from a list.

    Args:
        strings: List of strings to search
        query: Query string to match against
        algorithm: One of "levenshtein", "damerau_levenshtein", "jaro",
                  "jaro_winkler", "ngram", "bigram", "trigram", "lcs", "cosine"
        limit: Maximum number of results to return
        min_similarity: Minimum similarity score to include

    Returns:
        List of MatchResult objects sorted by score descending.

    Example:
        >>> results = find_best_matches(["apple", "apply", "banana"], "appel", limit=2)
        >>> [(r.text, r.score) for r in results]
        [('apple', 0.93), ('apply', 0.84)]
    """
    ...

# =============================================================================
# Deduplication
# =============================================================================

def find_duplicates(
    items: List[str],
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.85,
    normalize: str = "lowercase",
    method: str = "auto",
    window_size: int = 50,
) -> DeduplicationResult:
    """
    Find duplicate items in a list using the specified similarity algorithm.

    Args:
        items: List of strings to deduplicate
        algorithm: Similarity algorithm to use (default: "jaro_winkler")
        min_similarity: Minimum similarity score to consider items as duplicates (0.0 to 1.0)
        normalize: Normalization mode to apply before comparison:
            - "none": No normalization
            - "lowercase": Convert to lowercase (default)
            - "unicode_nfkd" or "nfkd": Apply Unicode NFKD normalization
            - "remove_punctuation": Remove ASCII punctuation
            - "remove_whitespace": Remove all whitespace
            - "strict": Apply all normalizations
        method: Deduplication method: "auto", "brute_force", or "snm" (default: "auto")
        window_size: Window size for "snm" method (default: 50)

    Returns:
        DeduplicationResult containing groups of duplicates and unique items.

    Example:
        >>> result = find_duplicates(["hello", "Hello", "HELLO", "world"])
        >>> result.groups
        [['hello', 'Hello', 'HELLO']]
        >>> result.unique
        ['world']
    """
    ...

# =============================================================================
# Multi-Algorithm Comparison
# =============================================================================

def compare_algorithms(
    strings: List[str],
    query: str,
    algorithms: Optional[List[str]] = None,
    limit: int = 3,
) -> List[AlgorithmComparison]:
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
# Index Classes
# =============================================================================

class BkTree:
    """
    BK-tree (Burkhard-Keller tree) for efficient fuzzy string search.

    Uses metric space properties to prune search space, making fuzzy
    search much faster than linear comparison for large datasets.

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
        self, query: str, max_distance: int
    ) -> List[SearchResult]:
        """
        Search for strings within a given edit distance.

        Returns:
            List of SearchResult objects sorted by distance.
        """
        ...

    def find_nearest(
        self, query: str, k: int
    ) -> List[SearchResult]:
        """Find the k nearest neighbors to the query."""
        ...

    def contains(self, query: str) -> bool:
        """Check if the tree contains an exact match."""
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

    Note:
        This class is NOT thread-safe. Do not share instances between
        Python threads. Create separate instances for each thread or
        use appropriate synchronization in your code.

    Example:
        >>> index = NgramIndex(ngram_size=2)
        >>> index.add_all(["apple", "application", "apply", "banana"])
        >>> results = index.search("appel", algorithm="jaro_winkler", min_similarity=0.7)
        >>> [(r.text, r.score) for r in results]
        [('apple', 0.93), ('apply', 0.84)]
    """

    def __init__(self, ngram_size: int = 2, min_similarity: float = 0.0) -> None:
        """
        Create a new n-gram index.

        Args:
            ngram_size: Size of n-grams (2 for bigrams, 3 for trigrams)
            min_similarity: Default minimum similarity threshold
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
        limit: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search with similarity scoring.

        Args:
            query: Query string
            algorithm: "jaro_winkler", "jaro", "levenshtein", "ngram", "trigram"
            min_similarity: Minimum similarity to include
            limit: Maximum results to return

        Returns:
            List of SearchResult objects sorted by similarity.
        """
        ...

    def batch_search(
        self,
        queries: Sequence[str],
        algorithm: str = "jaro_winkler",
        min_similarity: float = 0.0,
        limit: Optional[int] = None,
    ) -> List[List[SearchResult]]:
        """Search for multiple queries in parallel."""
        ...

    def find_nearest(
        self,
        query: str,
        k: int,
        algorithm: str = "jaro_winkler",
    ) -> List[SearchResult]:
        """Find the k nearest neighbors by similarity."""
        ...

    def contains(self, query: str) -> bool:
        """Check if the index contains an exact match."""
        ...

    def get_candidates(self, query: str) -> List[tuple[int, str]]:
        """Get all candidates that share n-grams with query."""
        ...

    def __len__(self) -> int:
        """Get the number of indexed items."""
        ...


class HybridIndex:
    """
    Hybrid index combining n-gram indexing with similarity search.

    Optimized for the best balance of speed and accuracy. Suitable for
    datasets with 100K+ items where both speed and accuracy matter.

    Note:
        This class is NOT thread-safe. Do not share instances between
        Python threads. Create separate instances for each thread or
        use appropriate synchronization in your code.

    Example:
        >>> index = HybridIndex(ngram_size=3)
        >>> index.add_all(["apple", "application", "apply", "banana"])
        >>> results = index.search("appel", min_similarity=0.7, limit=3)
        >>> [r.text for r in results]
        ['apple', 'apply', 'application']
    """

    def __init__(self, ngram_size: int = 3) -> None:
        """
        Create a new hybrid index.

        Args:
            ngram_size: Size of n-grams for candidate filtering (default 3)
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
        limit: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search for similar strings.

        Args:
            query: Query string to search for
            algorithm: Similarity algorithm ("jaro_winkler", "jaro", "levenshtein")
            min_similarity: Minimum similarity score to include (0.0-1.0)
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects sorted by similarity (highest first).
        """
        ...

    def batch_search(
        self,
        queries: Sequence[str],
        algorithm: str = "jaro_winkler",
        min_similarity: float = 0.0,
        limit: Optional[int] = None,
    ) -> List[List[SearchResult]]:
        """
        Search for multiple queries in parallel.

        Args:
            queries: List of query strings
            algorithm: Similarity algorithm to use
            min_similarity: Minimum similarity score
            limit: Maximum results per query

        Returns:
            List of result lists, one per query.
        """
        ...

    def find_nearest(
        self,
        query: str,
        k: int,
        algorithm: str = "jaro_winkler",
    ) -> List[SearchResult]:
        """
        Find the k nearest neighbors by similarity.

        Args:
            query: Query string
            k: Number of nearest neighbors to return
            algorithm: Similarity algorithm to use

        Returns:
            Up to k SearchResult objects sorted by similarity.
        """
        ...

    def contains(self, query: str) -> bool:
        """Check if the index contains an exact match for the query."""
        ...

    def __len__(self) -> int:
        """Get the number of items in the index."""
        ...


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
    data: Optional[UserData]

    def __init__(
        self,
        id: int,
        score: float,
        field_scores: dict[str, float],
        record: dict[str, str],
        data: Optional[UserData] = None,
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
        algorithm: str = "jaro_winkler",
        weight: float = 1.0,
        required: bool = False,
        normalize: Optional[str] = None,
        max_length: Optional[int] = None,
        separator: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        """
        Add a field to the schema.

        Args:
            name: Field name (must be unique)
            field_type: One of "short_text", "long_text", "token_set"
            algorithm: Similarity algorithm for this field
                      - For short_text: "levenshtein", "damerau_levenshtein", "jaro_winkler", "ngram", "exact_match"
                      - For long_text: "ngram", "cosine", "levenshtein", "exact_match"
                      - For token_set: "jaccard" (ignored, always uses Jaccard)
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

    def build(self) -> "Schema":
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

    def field_names(self) -> List[str]:
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
        ... }, min_score=0.5, limit=10)
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
        data: Optional[UserData] = None,
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

    def get(self, id: int) -> Optional[dict[str, str]]:
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
        min_score: float = 0.0,
        limit: Optional[int] = None,
        min_field_score: float = 0.0,
    ) -> List[SchemaSearchResult]:
        """
        Search for records matching the query across multiple fields.

        Each field in the query is matched against the corresponding field in indexed records
        using the algorithm defined in the schema. Field scores are combined using the
        schema's scoring strategy (e.g., weighted average).

        Args:
            query: Dictionary mapping field names to query values
                  (only fields present in query are searched)
            min_score: Minimum combined score to include in results (0.0 to 1.0)
            limit: Maximum number of results to return (None for unlimited)
            min_field_score: Minimum score for individual fields (0.0 to 1.0).
                           Fields with scores below this threshold are excluded

        Returns:
            List of SchemaSearchResult objects sorted by descending score.
            Each result includes the overall score, per-field scores, and record data.

        Raises:
            ValueError: If query contains unknown field names

        Example:
            >>> results = index.search({
            ...     "name": "laptop",
            ...     "tags": "computing,apple"
            ... }, min_score=0.7, limit=5)
            >>>
            >>> for r in results:
            ...     print(f"Overall: {r.score:.3f}")
            ...     print(f"  Name score: {r.field_scores['name']:.3f}")
            ...     print(f"  Tags score: {r.field_scores['tags']:.3f}")
            ...     print(f"  Record: {r.record}")
        """
        ...

    def __len__(self) -> int:
        """Get the number of records in the index."""
        ...


# Convenience aliases - these are function references, not type aliases
def edit_distance(a: str, b: str, max_distance: Optional[int] = None) -> int:
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
