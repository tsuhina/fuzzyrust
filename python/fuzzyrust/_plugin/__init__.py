"""Native Polars plugin detection and registration.

This module provides functions to detect whether the native Polars plugin
is available and to register plugin functions for use in Polars expressions.

The native plugin provides significant performance improvements (10-50x) for
column-to-column similarity comparisons by avoiding Python/Rust boundary
crossings for each row.

Logging:
    Debug messages are logged to help diagnose plugin detection issues.
    Enable with: logging.getLogger("fuzzyrust._plugin").setLevel(logging.DEBUG)
"""

import importlib.util
import logging
import os
from pathlib import Path
from typing import Optional

_logger = logging.getLogger(__name__)


class _PluginState:
    """Encapsulates plugin state to avoid global variables."""

    path: Optional[Path] = None
    available: Optional[bool] = None


_state = _PluginState()


def _find_plugin_lib() -> Optional[Path]:
    """Locate the plugin shared library.

    The plugin is compiled into the main fuzzyrust library when built with
    the `polars-plugin` feature.

    Returns:
        Path to the shared library, or None if not found.
    """
    if _state.path is not None:
        _logger.debug("Using cached plugin path: %s", _state.path)
        return _state.path

    try:
        import fuzzyrust._core

        # The plugin is in the same library as _core
        core_path = Path(fuzzyrust._core.__file__)
        if core_path.exists():
            _state.path = core_path
            _logger.debug("Plugin library found at: %s", _state.path)
            return _state.path
    except (ImportError, AttributeError) as e:
        _logger.debug("Failed to locate _core module: %s", e)

    _logger.debug("Plugin library not found")
    return None


def is_plugin_available() -> bool:
    """Check if the native Polars plugin is available.

    The plugin is available when:
    1. fuzzyrust was built with the `polars-plugin` feature
    2. The polars package is installed with plugin support

    Returns:
        True if the native plugin can be used, False otherwise.

    Example:
        >>> from fuzzyrust._plugin import is_plugin_available
        >>> if is_plugin_available():
        ...     print("Native Polars plugin is available!")
    """
    # Check if disabled by environment variable
    if os.environ.get("FUZZYRUST_DISABLE_PLUGIN", "").lower() in ("1", "true", "yes"):
        _logger.debug("Plugin disabled via FUZZYRUST_DISABLE_PLUGIN environment variable")
        _state.available = False
        return False

    if _state.available is not None:
        _logger.debug("Using cached plugin availability: %s", _state.available)
        return _state.available

    # Check if plugin library exists
    lib_path = _find_plugin_lib()
    if lib_path is None:
        _logger.debug("Plugin unavailable: library not found")
        _state.available = False
        return False

    # Check if polars has plugin support
    if importlib.util.find_spec("polars.plugins") is not None:
        _logger.debug("Plugin available: library found and polars.plugins supported")
        _state.available = True
    else:
        _logger.debug("Plugin unavailable: polars.plugins module not found")
        _state.available = False

    return _state.available


def use_native_plugin(enabled: bool = True) -> None:
    """Enable or disable the native Polars plugin at runtime.

    This can be useful for debugging or comparing performance between
    the native plugin and the fallback implementation.

    Args:
        enabled: Whether to enable the native plugin.

    Example:
        >>> import fuzzyrust
        >>> fuzzyrust.use_native_plugin(False)  # Disable plugin
        >>> # Now all operations use the fallback implementation
    """
    if enabled:
        _logger.debug("Re-enabling native plugin, clearing cached state")
        # Re-detect availability
        _state.available = None
        is_plugin_available()
        _logger.info("Native plugin %s", "enabled" if _state.available else "unavailable")
    else:
        _logger.info("Native plugin disabled by user")
        _state.available = False


# Plugin function wrappers
# These are only called when is_plugin_available() returns True


def fuzzy_similarity(
    left,
    right,
    algorithm: str = "jaro_winkler",
    ngram_size: int = 3,
    case_insensitive: bool = False,
):
    """Compute fuzzy similarity using native plugin.

    Args:
        left: Left Polars expression
        right: Right Polars expression
        algorithm: Similarity algorithm to use
        ngram_size: N-gram size for ngram algorithm (default: 3)
        case_insensitive: Perform case-insensitive comparison (default: False)

    Returns:
        Polars expression returning Float64 similarity scores
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_find_plugin_lib(),
        function_name="pl_fuzzy_similarity",
        args=[left, right],
        is_elementwise=True,
        kwargs={
            "algorithm": algorithm,
            "ngram_size": ngram_size,
            "case_insensitive": case_insensitive,
        },
    )


def fuzzy_is_match(
    left,
    right,
    algorithm: str = "jaro_winkler",
    threshold: float = 0.8,
    ngram_size: int = 3,
    case_insensitive: bool = False,
):
    """Check if similarity exceeds threshold using native plugin.

    Args:
        left: Left Polars expression
        right: Right Polars expression
        algorithm: Similarity algorithm to use
        threshold: Minimum similarity threshold
        ngram_size: N-gram size for ngram algorithm (default: 3)
        case_insensitive: Perform case-insensitive comparison (default: False)

    Returns:
        Polars expression returning Boolean
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_find_plugin_lib(),
        function_name="pl_fuzzy_is_match",
        args=[left, right],
        is_elementwise=True,
        kwargs={
            "algorithm": algorithm,
            "threshold": threshold,
            "ngram_size": ngram_size,
            "case_insensitive": case_insensitive,
        },
    )


def fuzzy_best_match(
    query,
    targets: list,
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.0,
    ngram_size: int = 3,
    case_insensitive: bool = False,
):
    """Find best match from targets using native plugin.

    Args:
        query: Query Polars expression
        targets: List of target strings to match against
        algorithm: Similarity algorithm to use
        min_similarity: Minimum similarity threshold (returns null if below)
        ngram_size: N-gram size for ngram algorithm (default: 3)
        case_insensitive: Perform case-insensitive comparison (default: False)

    Returns:
        Polars expression returning String (best match or null)
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_find_plugin_lib(),
        function_name="pl_fuzzy_best_match",
        args=[query],
        is_elementwise=True,
        kwargs={
            "targets": targets,
            "algorithm": algorithm,
            "min_score": min_similarity,  # Rust kwarg expects min_score
            "ngram_size": ngram_size,
            "case_insensitive": case_insensitive,
        },
    )


def fuzzy_best_match_score(
    query,
    targets: list,
    algorithm: str = "jaro_winkler",
    min_similarity: float = 0.0,
    ngram_size: int = 3,
    case_insensitive: bool = False,
):
    """Find best match with score from targets using native plugin.

    Args:
        query: Query Polars expression
        targets: List of target strings to match against
        algorithm: Similarity algorithm to use
        min_similarity: Minimum similarity threshold (returns null if below)
        ngram_size: N-gram size for ngram algorithm (default: 3)
        case_insensitive: Perform case-insensitive comparison (default: False)

    Returns:
        Polars expression returning Struct with fields:
        - match: String (best matching target or null)
        - score: Float64 (similarity score or null)
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_find_plugin_lib(),
        function_name="pl_fuzzy_best_match_score",
        args=[query],
        is_elementwise=True,
        kwargs={
            "targets": targets,
            "algorithm": algorithm,
            "min_score": min_similarity,  # Rust kwarg expects min_score
            "ngram_size": ngram_size,
            "case_insensitive": case_insensitive,
        },
    )


def fuzzy_distance(left, right):
    """Compute edit distance using native plugin.

    Args:
        left: Left Polars expression
        right: Right Polars expression

    Returns:
        Polars expression returning UInt32 distances
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_find_plugin_lib(),
        function_name="pl_fuzzy_distance",
        args=[left, right],
        is_elementwise=True,
    )


def fuzzy_soundex(expr):
    """Generate Soundex encoding using native plugin.

    Args:
        expr: Polars expression

    Returns:
        Polars expression returning String Soundex codes
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_find_plugin_lib(),
        function_name="pl_fuzzy_soundex",
        args=[expr],
        is_elementwise=True,
    )


def fuzzy_metaphone(expr):
    """Generate Metaphone encoding using native plugin.

    Args:
        expr: Polars expression

    Returns:
        Polars expression returning String Metaphone codes
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_find_plugin_lib(),
        function_name="pl_fuzzy_metaphone",
        args=[expr],
        is_elementwise=True,
    )


def fuzzy_similarity_literal(
    expr,
    target: str,
    algorithm: str = "jaro_winkler",
    ngram_size: int = 3,
    case_insensitive: bool = False,
):
    """Compute fuzzy similarity between a column and a literal string using native plugin.

    This is optimized for comparing a column against a single target string,
    providing 10-50x speedup over map_elements.

    Args:
        expr: Polars expression (string column)
        target: Target string to compare against
        algorithm: Similarity algorithm to use
        ngram_size: N-gram size for ngram algorithm (default: 3)
        case_insensitive: Perform case-insensitive comparison (default: False)

    Returns:
        Polars expression returning Float64 similarity scores (0.0 to 1.0)
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_find_plugin_lib(),
        function_name="pl_fuzzy_similarity_literal",
        args=[expr],
        is_elementwise=True,
        kwargs={
            "target": target,
            "algorithm": algorithm,
            "ngram_size": ngram_size,
            "case_insensitive": case_insensitive,
        },
    )


def fuzzy_is_match_literal(
    expr,
    target: str,
    algorithm: str = "jaro_winkler",
    threshold: float = 0.8,
    ngram_size: int = 3,
    case_insensitive: bool = False,
):
    """Check if similarity between a column and a literal string exceeds threshold.

    This is optimized for comparing a column against a single target string,
    providing 10-50x speedup over map_elements.

    Args:
        expr: Polars expression (string column)
        target: Target string to compare against
        algorithm: Similarity algorithm to use
        threshold: Minimum similarity threshold
        ngram_size: N-gram size for ngram algorithm (default: 3)
        case_insensitive: Perform case-insensitive comparison (default: False)

    Returns:
        Polars expression returning Boolean
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_find_plugin_lib(),
        function_name="pl_fuzzy_is_match_literal",
        args=[expr],
        is_elementwise=True,
        kwargs={
            "target": target,
            "algorithm": algorithm,
            "threshold": threshold,
            "ngram_size": ngram_size,
            "case_insensitive": case_insensitive,
        },
    )
