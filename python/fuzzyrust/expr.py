"""Polars expression namespace for fuzzy string matching.

This module registers a `.fuzzy` namespace on Polars expressions,
enabling chainable fuzzy matching operations directly in Polars
expression contexts.

When the native Polars plugin is available (fuzzyrust built with
`polars-plugin` feature), both column-to-column and column-to-literal
comparisons use the high-performance native plugin (10-50x speedup).
Otherwise, falls back to map_elements.

Warning:
    For large datasets (10K+ rows), consider using the batch API:
    - fr.batch_similarity() for similarity computation
    - fr.batch_best_match() for finding best matches

Example:
    >>> import polars as pl
    >>> import fuzzyrust  # Registers the namespace
    >>>
    >>> df = pl.DataFrame({"name": ["John", "Jon", "Jane"]})
    >>> df.with_columns(
    ...     is_similar=pl.col("name").fuzzy.is_similar("John", min_similarity=0.8)
    ... )
"""

from typing import Literal, Union

import polars as pl

import fuzzyrust as fr
from fuzzyrust._utils import normalize_algorithm

# Comprehensive type for all supported similarity algorithms
AlgorithmType = Literal[
    "levenshtein",
    "damerau_levenshtein",
    "jaro",
    "jaro_winkler",
    "ngram",
    "bigram",
    "trigram",
    "jaccard",
    "cosine",
    "lcs",
    "hamming",
]

# Type for distance-based algorithms
DistanceAlgorithmType = Literal[
    "levenshtein",
    "damerau_levenshtein",
    "hamming",
    "lcs",
]

# Try to import native plugin support
try:
    from fuzzyrust._plugin import (
        fuzzy_best_match as _plugin_best_match,
    )
    from fuzzyrust._plugin import (
        fuzzy_best_match_score as _plugin_best_match_score,
    )
    from fuzzyrust._plugin import (
        fuzzy_distance as _plugin_distance,
    )
    from fuzzyrust._plugin import (
        fuzzy_is_match as _plugin_is_match,
    )
    from fuzzyrust._plugin import (
        fuzzy_is_match_literal as _plugin_is_match_literal,
    )
    from fuzzyrust._plugin import (
        fuzzy_metaphone as _plugin_metaphone,
    )
    from fuzzyrust._plugin import (
        fuzzy_similarity as _plugin_similarity,
    )
    from fuzzyrust._plugin import (
        fuzzy_similarity_literal as _plugin_similarity_literal,
    )
    from fuzzyrust._plugin import (
        fuzzy_soundex as _plugin_soundex,
    )
    from fuzzyrust._plugin import (
        is_plugin_available,
    )

    _PLUGIN_AVAILABLE = is_plugin_available()
except ImportError:
    _PLUGIN_AVAILABLE = False
    _plugin_best_match = None
    _plugin_best_match_score = None
    _plugin_distance = None


@pl.api.register_expr_namespace("fuzzy")
class FuzzyExprNamespace:
    """
    Fuzzy string matching namespace for Polars expressions.

    Provides chainable methods for fuzzy matching directly on columns.
    Access via `.fuzzy` on any string expression.
    """

    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def similarity(
        self,
        other: Union[str, pl.Expr],
        algorithm: Union[str, AlgorithmType] = "jaro_winkler",
        ngram_size: int = 3,
        case_insensitive: bool = False,
    ) -> pl.Expr:
        """
        Calculate similarity score between this column and another value/column.

        Args:
            other: String literal or column expression to compare against
            algorithm: Similarity algorithm to use. Options:
                - "levenshtein": Edit distance based similarity
                - "damerau_levenshtein": Edit distance with transpositions
                - "jaro": Jaro similarity (good for short strings)
                - "jaro_winkler": Jaro-Winkler (favors matching prefixes)
                - "ngram": N-gram based similarity (uses ngram_size parameter)
                - "cosine": Character-level cosine similarity
                - "lcs": Longest common subsequence based similarity
                - "hamming": Hamming distance based similarity
            ngram_size: N-gram size for ngram algorithm (default: 3)
            case_insensitive: Perform case-insensitive comparison (default: False)

        Returns:
            Expression producing similarity scores (0.0 to 1.0)

        Example:
            >>> df.with_columns(
            ...     score=pl.col("name").fuzzy.similarity("John")
            ... )
            >>> df.with_columns(
            ...     score=pl.col("name1").fuzzy.similarity(pl.col("name2"))
            ... )
            >>> df.with_columns(
            ...     score=pl.col("name").fuzzy.similarity("John", algorithm="ngram", ngram_size=2)
            ... )
            >>> df.with_columns(
            ...     score=pl.col("name").fuzzy.similarity("john", case_insensitive=True)
            ... )
        """
        algo = normalize_algorithm(algorithm)

        # Build algo_map with ngram_size parameter
        algo_map = {
            "levenshtein": fr.levenshtein_similarity,
            "damerau_levenshtein": fr.damerau_levenshtein_similarity,
            "jaro": fr.jaro_similarity,
            "jaro_winkler": fr.jaro_winkler_similarity,
            "ngram": lambda a, b: fr.ngram_similarity(a, b, ngram_size=ngram_size),
            "bigram": lambda a, b: fr.ngram_similarity(a, b, ngram_size=2),
            "trigram": lambda a, b: fr.ngram_similarity(a, b, ngram_size=3),
            "jaccard": lambda a, b: fr.ngram_jaccard(a, b, ngram_size=ngram_size),
            "cosine": fr.cosine_similarity_chars,
            "lcs": fr.lcs_similarity,
            "hamming": fr.hamming_similarity,
        }

        if algo not in algo_map:
            raise ValueError(f"Unknown algorithm: {algo}. Valid: {list(algo_map.keys())}")

        sim_func = algo_map[algo]

        if isinstance(other, str):
            # Compare against a literal string
            # Use native plugin if available for better performance (10-50x speedup)
            if _PLUGIN_AVAILABLE:
                return _plugin_similarity_literal(
                    self._expr,
                    target=other,
                    algorithm=algo,
                    ngram_size=ngram_size,
                    case_insensitive=case_insensitive,
                )

            # Fallback: use map_elements
            other_cmp = other.lower() if case_insensitive else other

            def compute_similarity(s):
                if s is None:
                    return None
                s_cmp = str(s).lower() if case_insensitive else str(s)
                return sim_func(s_cmp, other_cmp)

            return self._expr.map_elements(compute_similarity, return_dtype=pl.Float64)
        else:
            # Compare against another column
            # Use native plugin if available for better performance
            if _PLUGIN_AVAILABLE:
                return _plugin_similarity(
                    self._expr,
                    other,
                    algorithm=algo,
                    ngram_size=ngram_size,
                    case_insensitive=case_insensitive,
                )

            # Fallback: use struct + map_elements
            def compute_similarity_row(row):
                left = row["_left"]
                right = row["_right"]
                if left is None or right is None:
                    return None
                left_str = str(left).lower() if case_insensitive else str(left)
                right_str = str(right).lower() if case_insensitive else str(right)
                return sim_func(left_str, right_str)

            return pl.struct([self._expr.alias("_left"), other.alias("_right")]).map_elements(
                compute_similarity_row,
                return_dtype=pl.Float64,
            )

    def is_similar(
        self,
        other: Union[str, pl.Expr],
        min_similarity: float = 0.8,
        algorithm: Union[str, AlgorithmType] = "jaro_winkler",
        ngram_size: int = 3,
        case_insensitive: bool = False,
    ) -> pl.Expr:
        """
        Check if values are similar to another value/column above a threshold.

        Args:
            other: String literal or column expression to compare against
            min_similarity: Minimum similarity score to return True (0.0 to 1.0)
            algorithm: Similarity algorithm to use (string or Algorithm enum)
            ngram_size: N-gram size for ngram algorithm (default: 3)
            case_insensitive: Perform case-insensitive comparison (default: False)

        Returns:
            Boolean expression

        Example:
            >>> df.filter(pl.col("name").fuzzy.is_similar("John", min_similarity=0.85))
            >>> df.filter(pl.col("name").fuzzy.is_similar("John", algorithm="ngram", ngram_size=2))
            >>> df.filter(pl.col("name").fuzzy.is_similar("john", case_insensitive=True))
        """
        algo = normalize_algorithm(algorithm)

        if _PLUGIN_AVAILABLE:
            if isinstance(other, str):
                # Use native plugin for column-to-literal comparison (10-50x speedup)
                return _plugin_is_match_literal(
                    self._expr,
                    target=other,
                    algorithm=algo,
                    threshold=min_similarity,
                    ngram_size=ngram_size,
                    case_insensitive=case_insensitive,
                )
            else:
                # Use native plugin for column-to-column comparison
                return _plugin_is_match(
                    self._expr,
                    other,
                    algorithm=algo,
                    threshold=min_similarity,
                    ngram_size=ngram_size,
                    case_insensitive=case_insensitive,
                )

        # Fallback: compute similarity and compare
        return (
            self.similarity(
                other, algorithm=algorithm, ngram_size=ngram_size, case_insensitive=case_insensitive
            )
            >= min_similarity
        )

    def best_match(
        self,
        choices: list[str],
        algorithm: Union[str, AlgorithmType] = "jaro_winkler",
        min_similarity: float = 0.0,
        ngram_size: int = 3,
        case_insensitive: bool = False,
    ) -> pl.Expr:
        """
        Find the best matching string from a list of choices.

        Args:
            choices: List of strings to match against
            algorithm: Similarity algorithm to use (string or Algorithm enum)
            min_similarity: Minimum score to return a match (otherwise null)
            ngram_size: N-gram size for ngram algorithm (default: 3)
            case_insensitive: Perform case-insensitive comparison (default: False)

        Returns:
            Expression with the best matching string (or null)

        Example:
            >>> categories = ["Electronics", "Clothing", "Food"]
            >>> df.with_columns(
            ...     category=pl.col("raw_category").fuzzy.best_match(categories)
            ... )
            >>> df.with_columns(
            ...     category=pl.col("raw_category").fuzzy.best_match(
            ...         categories, algorithm="ngram", ngram_size=2
            ...     )
            ... )
            >>> df.with_columns(
            ...     category=pl.col("raw_category").fuzzy.best_match(
            ...         categories, case_insensitive=True
            ...     )
            ... )
        """
        algo = normalize_algorithm(algorithm)

        # Use native plugin if available for better performance
        if _PLUGIN_AVAILABLE and _plugin_best_match is not None:
            return _plugin_best_match(
                self._expr,
                targets=choices,
                algorithm=algo,
                min_similarity=min_similarity,
                ngram_size=ngram_size,
                case_insensitive=case_insensitive,
            )

        # Fallback: use map_elements
        # Pre-process choices for case-insensitive matching
        choices_cmp = [c.lower() for c in choices] if case_insensitive else choices

        def find_best(value):
            if value is None:
                return None
            value_str = str(value).lower() if case_insensitive else str(value)
            results = fr.batch.best_matches(
                choices_cmp, value_str, algorithm=algo, limit=1, min_similarity=min_similarity
            )
            if results:
                # Return original choice (not lowercased version)
                idx = choices_cmp.index(results[0].text)
                return choices[idx]
            return None

        return self._expr.map_elements(find_best, return_dtype=pl.Utf8)

    def best_match_score(
        self,
        choices: list[str],
        algorithm: Union[str, AlgorithmType] = "jaro_winkler",
        min_similarity: float = 0.0,
        ngram_size: int = 3,
        case_insensitive: bool = False,
    ) -> pl.Expr:
        """
        Get both the best match and its score as a struct.

        Args:
            choices: List of strings to match against
            algorithm: Similarity algorithm to use (string or Algorithm enum)
            min_similarity: Minimum score to return a match
            ngram_size: N-gram size for ngram algorithm (default: 3)
            case_insensitive: Perform case-insensitive comparison (default: False)

        Returns:
            Struct expression with fields 'match' and 'score'

        Example:
            >>> df.with_columns(
            ...     result=pl.col("name").fuzzy.best_match_score(candidates)
            ... ).select(
            ...     pl.col("result").struct.field("match"),
            ...     pl.col("result").struct.field("score"),
            ... )
            >>> df.with_columns(
            ...     result=pl.col("name").fuzzy.best_match_score(
            ...         candidates, algorithm="ngram", ngram_size=2
            ...     )
            ... )
            >>> df.with_columns(
            ...     result=pl.col("name").fuzzy.best_match_score(
            ...         candidates, case_insensitive=True
            ...     )
            ... )
        """
        algo = normalize_algorithm(algorithm)

        # Use native plugin if available for better performance
        if _PLUGIN_AVAILABLE and _plugin_best_match_score is not None:
            return _plugin_best_match_score(
                self._expr,
                targets=choices,
                algorithm=algo,
                min_similarity=min_similarity,
                ngram_size=ngram_size,
                case_insensitive=case_insensitive,
            )

        # Fallback: use map_elements
        # Pre-process choices for case-insensitive matching
        choices_cmp = [c.lower() for c in choices] if case_insensitive else choices

        def find_best_with_score(value):
            if value is None:
                return {"match": None, "score": None}
            value_str = str(value).lower() if case_insensitive else str(value)
            results = fr.batch.best_matches(
                choices_cmp, value_str, algorithm=algo, limit=1, min_similarity=min_similarity
            )
            if results:
                # Return original choice (not lowercased version)
                idx = choices_cmp.index(results[0].text)
                return {"match": choices[idx], "score": results[0].score}
            return {"match": None, "score": None}

        return self._expr.map_elements(
            find_best_with_score,
            return_dtype=pl.Struct({"match": pl.Utf8, "score": pl.Float64}),
        )

    def normalize(
        self,
        mode: Literal[
            "lowercase",
            "uppercase",
            "unicode_nfkd",
            "remove_punctuation",
            "remove_whitespace",
            "strict",
        ] = "lowercase",
    ) -> pl.Expr:
        """
        Normalize strings for fuzzy matching.

        Args:
            mode: Normalization mode:
                - "lowercase": Convert to lowercase
                - "uppercase": Convert to uppercase
                - "unicode_nfkd": Unicode NFKD normalization
                - "remove_punctuation": Remove punctuation
                - "remove_whitespace": Collapse whitespace
                - "strict": Lowercase + remove punctuation + collapse whitespace

        Returns:
            Normalized string expression

        Example:
            >>> df.with_columns(
            ...     normalized=pl.col("name").fuzzy.normalize("strict")
            ... )
        """

        def normalize_value(value):
            if value is None:
                return None
            return fr.normalize_string(str(value), mode)

        return self._expr.map_elements(normalize_value, return_dtype=pl.Utf8)

    def phonetic(
        self,
        algorithm: Literal["soundex", "metaphone"] = "soundex",
    ) -> pl.Expr:
        """
        Generate phonetic encoding of strings.

        Args:
            algorithm: Phonetic algorithm to use:
                - "soundex": Classic Soundex encoding
                - "metaphone": Metaphone encoding

        Returns:
            Phonetic code expression

        Example:
            >>> df.with_columns(
            ...     soundex=pl.col("name").fuzzy.phonetic("soundex")
            ... )
        """
        algo_map = {
            "soundex": fr.soundex,
            "metaphone": fr.metaphone,
        }

        if algorithm not in algo_map:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        func = algo_map[algorithm]

        def encode(value):
            if value is None:
                return None
            return func(str(value))

        return self._expr.map_elements(encode, return_dtype=pl.Utf8)

    def soundex(self) -> pl.Expr:
        """
        Generate Soundex phonetic encoding of strings.

        Returns a 4-character code that represents the phonetic sound of the string.
        Useful for matching names that sound similar but are spelled differently.

        Returns:
            Expression producing Soundex codes (e.g., "S530" for "Smith")

        Example:
            >>> df.with_columns(
            ...     code=pl.col("name").fuzzy.soundex()
            ... )
        """
        if _PLUGIN_AVAILABLE:
            return _plugin_soundex(self._expr)

        def encode(value):
            if value is None:
                return None
            return fr.soundex(str(value))

        return self._expr.map_elements(encode, return_dtype=pl.Utf8)

    def metaphone(self) -> pl.Expr:
        """
        Generate Metaphone phonetic encoding of strings.

        Returns a variable-length code that represents the phonetic sound of the string.
        Generally more accurate than Soundex for English words.

        Returns:
            Expression producing Metaphone codes

        Example:
            >>> df.with_columns(
            ...     code=pl.col("name").fuzzy.metaphone()
            ... )
        """
        if _PLUGIN_AVAILABLE:
            return _plugin_metaphone(self._expr)

        def encode(value):
            if value is None:
                return None
            return fr.metaphone(str(value))

        return self._expr.map_elements(encode, return_dtype=pl.Utf8)

    def distance(
        self,
        other: Union[str, pl.Expr],
        algorithm: Union[str, DistanceAlgorithmType] = "levenshtein",
    ) -> pl.Expr:
        """
        Calculate edit distance between this column and another value/column.

        Args:
            other: String literal or column expression to compare against
            algorithm: Distance algorithm to use. Options:
                - "levenshtein": Classic edit distance (uses native plugin for col-to-col)
                - "damerau_levenshtein": Edit distance with transpositions
                - "hamming": Hamming distance (same-length strings)
                - "lcs": Inverse LCS length (max length - LCS length)

        Returns:
            Expression producing integer distances

        Example:
            >>> df.with_columns(
            ...     dist=pl.col("name").fuzzy.distance("John")
            ... )
            >>> df.with_columns(
            ...     dist=pl.col("name1").fuzzy.distance(pl.col("name2"))
            ... )

        Note:
            The native plugin is only used for Levenshtein column-to-column
            comparisons. Other algorithms and literal comparisons use the
            fallback implementation.
        """
        algo = normalize_algorithm(algorithm)

        algo_map = {
            "levenshtein": fr.levenshtein,
            "damerau_levenshtein": fr.damerau_levenshtein,
            "hamming": fr.hamming,
            "lcs": lambda a, b: max(len(a), len(b)) - fr.lcs_length(a, b),
        }

        if algo not in algo_map:
            raise ValueError(f"Unknown algorithm: {algo}. Valid: {list(algo_map.keys())}")

        dist_func = algo_map[algo]

        if isinstance(other, str):
            # Literal comparison - always use fallback
            return self._expr.map_elements(
                lambda s: dist_func(str(s) if s is not None else "", other),
                return_dtype=pl.Int64,
            )
        else:
            # Column-to-column comparison
            # Use native plugin for Levenshtein if available
            if algo == "levenshtein" and _PLUGIN_AVAILABLE and _plugin_distance is not None:
                return _plugin_distance(self._expr, other)

            # Fallback: use struct + map_elements
            return pl.struct([self._expr.alias("_left"), other.alias("_right")]).map_elements(
                lambda row: dist_func(
                    str(row["_left"]) if row["_left"] is not None else "",
                    str(row["_right"]) if row["_right"] is not None else "",
                ),
                return_dtype=pl.Int64,
            )
