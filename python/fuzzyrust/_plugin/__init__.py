"""Native Polars plugin detection and registration.

This module provides functions to detect whether the native Polars plugin
is available and to register plugin functions for use in Polars expressions.

The native plugin provides significant performance improvements (10-50x) for
column-to-column similarity comparisons by avoiding Python/Rust boundary
crossings for each row.
"""

import importlib.util
import os
from pathlib import Path
from typing import Optional


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
        return _state.path

    try:
        import fuzzyrust._core

        # The plugin is in the same library as _core
        core_path = Path(fuzzyrust._core.__file__)
        if core_path.exists():
            _state.path = core_path
            return _state.path
    except (ImportError, AttributeError):
        pass

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
        _state.available = False
        return False

    if _state.available is not None:
        return _state.available

    # Check if plugin library exists
    lib_path = _find_plugin_lib()
    if lib_path is None:
        _state.available = False
        return False

    # Check if polars has plugin support
    if importlib.util.find_spec("polars.plugins") is not None:
        _state.available = True
    else:
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
        # Re-detect availability
        _state.available = None
        is_plugin_available()
    else:
        _state.available = False


# Plugin function wrappers
# These are only called when is_plugin_available() returns True


def fuzzy_similarity(left, right, algorithm: str = "jaro_winkler"):
    """Compute fuzzy similarity using native plugin.

    Args:
        left: Left Polars expression
        right: Right Polars expression
        algorithm: Similarity algorithm to use

    Returns:
        Polars expression returning Float64 similarity scores
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_find_plugin_lib(),
        function_name="pl_fuzzy_similarity",
        args=[left, right],
        is_elementwise=True,
        kwargs={"algorithm": algorithm},
    )


def fuzzy_is_match(left, right, algorithm: str = "jaro_winkler", threshold: float = 0.8):
    """Check if similarity exceeds threshold using native plugin.

    Args:
        left: Left Polars expression
        right: Right Polars expression
        algorithm: Similarity algorithm to use
        threshold: Minimum similarity threshold

    Returns:
        Polars expression returning Boolean
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_find_plugin_lib(),
        function_name="pl_fuzzy_is_match",
        args=[left, right],
        is_elementwise=True,
        kwargs={"algorithm": algorithm, "threshold": threshold},
    )


def fuzzy_best_match(query, targets: list, algorithm: str = "jaro_winkler", min_score: float = 0.0):
    """Find best match from targets using native plugin.

    Args:
        query: Query Polars expression
        targets: List of target strings to match against
        algorithm: Similarity algorithm to use
        min_score: Minimum score threshold (returns null if below)

    Returns:
        Polars expression returning String (best match or null)
    """
    from polars.plugins import register_plugin_function

    return register_plugin_function(
        plugin_path=_find_plugin_lib(),
        function_name="pl_fuzzy_best_match",
        args=[query],
        is_elementwise=True,
        kwargs={"targets": targets, "algorithm": algorithm, "min_score": min_score},
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
