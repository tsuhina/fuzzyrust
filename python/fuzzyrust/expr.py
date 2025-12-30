"""Polars expression namespace for fuzzy string matching.

This module registers a `.fuzzy` namespace on Polars expressions,
enabling chainable fuzzy matching operations directly in Polars
expression contexts.

When the native Polars plugin is available (fuzzyrust built with
`polars-plugin` feature), column-to-column comparisons use the
high-performance native plugin. Otherwise, falls back to map_elements.

Note:
    For literal comparisons (column vs string), map_elements is always used
    since the plugin only accelerates column-to-column operations.

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

# Try to import native plugin support
try:
    from fuzzyrust._plugin import (
        fuzzy_is_match as _plugin_is_match,
    )
    from fuzzyrust._plugin import (
        fuzzy_similarity as _plugin_similarity,
    )
    from fuzzyrust._plugin import (
        is_plugin_available,
    )

    _PLUGIN_AVAILABLE = is_plugin_available()
except ImportError:
    _PLUGIN_AVAILABLE = False


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
        algorithm: Union[
            str,
            Literal[
                "levenshtein", "damerau_levenshtein", "jaro", "jaro_winkler", "ngram", "cosine"
            ],
        ] = "jaro_winkler",
    ) -> pl.Expr:
        """
        Calculate similarity score between this column and another value/column.

        Args:
            other: String literal or column expression to compare against
            algorithm: Similarity algorithm to use (string or Algorithm enum)

        Returns:
            Expression producing similarity scores (0.0 to 1.0)

        Example:
            >>> df.with_columns(
            ...     score=pl.col("name").fuzzy.similarity("John")
            ... )
            >>> df.with_columns(
            ...     score=pl.col("name1").fuzzy.similarity(pl.col("name2"))
            ... )
        """
        algo = normalize_algorithm(algorithm)

        algo_map = {
            "levenshtein": fr.levenshtein_similarity,
            "damerau_levenshtein": fr.damerau_levenshtein_similarity,
            "jaro": fr.jaro_similarity,
            "jaro_winkler": fr.jaro_winkler_similarity,
            "ngram": lambda a, b: fr.ngram_similarity(a, b, ngram_size=3),
            "cosine": fr.cosine_similarity_chars,
        }

        if algo not in algo_map:
            raise ValueError(f"Unknown algorithm: {algo}. Valid: {list(algo_map.keys())}")

        sim_func = algo_map[algo]

        if isinstance(other, str):
            # Compare against a literal string - always use map_elements
            return self._expr.map_elements(
                lambda s: sim_func(str(s) if s is not None else "", other),
                return_dtype=pl.Float64,
            )
        else:
            # Compare against another column
            # Use native plugin if available for better performance
            if _PLUGIN_AVAILABLE:
                return _plugin_similarity(self._expr, other, algorithm=algo)

            # Fallback: use struct + map_elements
            return pl.struct([self._expr.alias("_left"), other.alias("_right")]).map_elements(
                lambda row: sim_func(
                    str(row["_left"]) if row["_left"] is not None else "",
                    str(row["_right"]) if row["_right"] is not None else "",
                ),
                return_dtype=pl.Float64,
            )

    def is_similar(
        self,
        other: Union[str, pl.Expr],
        min_similarity: float = 0.8,
        algorithm: Union[
            str,
            Literal[
                "levenshtein", "damerau_levenshtein", "jaro", "jaro_winkler", "ngram", "cosine"
            ],
        ] = "jaro_winkler",
    ) -> pl.Expr:
        """
        Check if values are similar to another value/column above a threshold.

        Args:
            other: String literal or column expression to compare against
            min_similarity: Minimum similarity score to return True (0.0 to 1.0)
            algorithm: Similarity algorithm to use (string or Algorithm enum)

        Returns:
            Boolean expression

        Example:
            >>> df.filter(pl.col("name").fuzzy.is_similar("John", min_similarity=0.85))
        """
        algo = normalize_algorithm(algorithm)

        # Use native plugin for column-to-column comparison if available
        if _PLUGIN_AVAILABLE and isinstance(other, pl.Expr):
            return _plugin_is_match(self._expr, other, algorithm=algo, threshold=min_similarity)

        # Fallback: compute similarity and compare
        return self.similarity(other, algorithm=algorithm) >= min_similarity

    def best_match(
        self,
        choices: list[str],
        algorithm: Union[
            str,
            Literal["levenshtein", "damerau_levenshtein", "jaro", "jaro_winkler", "ngram"],
        ] = "jaro_winkler",
        min_similarity: float = 0.0,
    ) -> pl.Expr:
        """
        Find the best matching string from a list of choices.

        Args:
            choices: List of strings to match against
            algorithm: Similarity algorithm to use (string or Algorithm enum)
            min_similarity: Minimum score to return a match (otherwise null)

        Returns:
            Expression with the best matching string (or null)

        Example:
            >>> categories = ["Electronics", "Clothing", "Food"]
            >>> df.with_columns(
            ...     category=pl.col("raw_category").fuzzy.best_match(categories)
            ... )
        """
        algo = normalize_algorithm(algorithm)

        def find_best(value):
            if value is None:
                return None
            results = fr.find_best_matches(
                choices, str(value), algorithm=algo, limit=1, min_similarity=min_similarity
            )
            return results[0].text if results else None

        return self._expr.map_elements(find_best, return_dtype=pl.Utf8)

    def best_match_score(
        self,
        choices: list[str],
        algorithm: Union[
            str,
            Literal["levenshtein", "damerau_levenshtein", "jaro", "jaro_winkler", "ngram"],
        ] = "jaro_winkler",
        min_similarity: float = 0.0,
    ) -> pl.Expr:
        """
        Get both the best match and its score as a struct.

        Args:
            choices: List of strings to match against
            algorithm: Similarity algorithm to use (string or Algorithm enum)
            min_similarity: Minimum score to return a match

        Returns:
            Struct expression with fields 'match' and 'score'

        Example:
            >>> df.with_columns(
            ...     result=pl.col("name").fuzzy.best_match_score(candidates)
            ... ).select(
            ...     pl.col("result").struct.field("match"),
            ...     pl.col("result").struct.field("score"),
            ... )
        """
        algo = normalize_algorithm(algorithm)

        def find_best_with_score(value):
            if value is None:
                return {"match": None, "score": None}
            results = fr.find_best_matches(
                choices, str(value), algorithm=algo, limit=1, min_similarity=min_similarity
            )
            if results:
                return {"match": results[0].text, "score": results[0].score}
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

    def distance(
        self,
        other: Union[str, pl.Expr],
        algorithm: Union[
            str,
            Literal["levenshtein", "damerau_levenshtein", "hamming"],
        ] = "levenshtein",
    ) -> pl.Expr:
        """
        Calculate edit distance between this column and another value/column.

        Args:
            other: String literal or column expression to compare against
            algorithm: Distance algorithm to use (string or Algorithm enum)

        Returns:
            Expression producing integer distances

        Example:
            >>> df.with_columns(
            ...     dist=pl.col("name").fuzzy.distance("John")
            ... )
        """
        algo = normalize_algorithm(algorithm)

        algo_map = {
            "levenshtein": fr.levenshtein,
            "damerau_levenshtein": fr.damerau_levenshtein,
            "hamming": fr.hamming,
        }

        if algo not in algo_map:
            raise ValueError(f"Unknown algorithm: {algo}. Valid: {list(algo_map.keys())}")

        dist_func = algo_map[algo]

        if isinstance(other, str):
            return self._expr.map_elements(
                lambda s: dist_func(str(s) if s is not None else "", other),
                return_dtype=pl.Int64,
            )
        else:
            return pl.struct([self._expr.alias("_left"), other.alias("_right")]).map_elements(
                lambda row: dist_func(
                    str(row["_left"]) if row["_left"] is not None else "",
                    str(row["_right"]) if row["_right"] is not None else "",
                ),
                return_dtype=pl.Int64,
            )
