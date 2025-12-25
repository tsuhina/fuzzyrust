//! Levenshtein (edit) distance implementation
//!
//! Optimized with:
//! - Single-row DP for O(min(m,n)) space
//! - Early termination with max distance threshold
//! - Unicode-aware character handling

use super::EditDistance;
use smallvec::SmallVec;

/// Levenshtein distance calculator with optional early termination
///
/// # Complexity
/// - Time: O(m*n) where m and n are string lengths
/// - Space: O(min(m,n)) using single-row DP optimization
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Levenshtein {
    /// Maximum distance to compute (for early termination)
    pub max_distance: Option<usize>,
}

impl Levenshtein {
    pub fn new() -> Self {
        Self { max_distance: None }
    }

    pub fn with_max_distance(max_distance: usize) -> Self {
        Self { max_distance: Some(max_distance) }
    }

    /// Compute distance with proper Option semantics.
    /// Returns `None` if distance exceeds max_distance threshold.
    /// Returns `Some(distance)` otherwise.
    pub fn compute(&self, a: &str, b: &str) -> Option<usize> {
        levenshtein_distance_bounded(a, b, self.max_distance)
    }
}

impl EditDistance for Levenshtein {
    fn distance(&self, a: &str, b: &str) -> usize {
        // When max_distance is set and exceeded, return max_distance + 1
        // to indicate "greater than threshold" without using sentinel values
        match self.max_distance {
            Some(max_d) => levenshtein_distance_bounded(a, b, Some(max_d))
                .unwrap_or(max_d.saturating_add(1)),
            None => levenshtein_distance_bounded(a, b, None).unwrap_or(0),
        }
    }

    fn name(&self) -> &'static str {
        "levenshtein"
    }
}

/// Compute Levenshtein distance with optional max threshold.
///
/// Returns `None` if distance exceeds `max_distance` (early termination).
/// Returns `Some(distance)` if distance is within threshold or no threshold set.
///
/// # Example
/// ```
/// use fuzzyrust::algorithms::levenshtein::levenshtein_distance_bounded;
///
/// // No threshold - always returns Some
/// assert_eq!(levenshtein_distance_bounded("kitten", "sitting", None), Some(3));
///
/// // Within threshold - returns actual distance
/// assert_eq!(levenshtein_distance_bounded("abc", "abd", Some(2)), Some(1));
///
/// // Exceeds threshold - returns None
/// assert_eq!(levenshtein_distance_bounded("abcdef", "ghijkl", Some(3)), None);
/// ```
#[inline]
pub fn levenshtein_distance_bounded(a: &str, b: &str, max_distance: Option<usize>) -> Option<usize> {
    if a == b { return Some(0); }

    let m_count = a.chars().count();
    let n_count = b.chars().count();

    if m_count == 0 { return Some(n_count); }
    if n_count == 0 { return Some(m_count); }

    // Early termination check based on length difference
    if let Some(max_d) = max_distance {
        if m_count.abs_diff(n_count) > max_d {
            return None;
        }
    }

    // Ensure we iterate over the shorter string in the inner loop (columns)
    // to minimize space usage and cache misses.
    // target = columns (inner loop), source = rows (outer loop)
    let (target_str, source_str, n_target) = if m_count < n_count {
        (a, b, m_count)
    } else {
        (b, a, n_count) // target is b (shorter)
    };

    let target_chars: Vec<char> = target_str.chars().collect();
    
    // Single-row DP with O(min(m,n)) space.
    // Use SmallVec to avoid heap allocation for common string lengths.
    // Size 64 covers most names, words, and short phrases.
    let mut row: SmallVec<[usize; 64]> = (0..=n_target).collect();

    for (i, sc) in source_str.chars().enumerate() {
        let mut prev_substitution_cost = row[0];
        
        // Update first cell of new row (distance for empty target prefix)
        row[0] = i + 1;
        
        let mut row_min = row[0];

        for j in 0..n_target {
            let tc = target_chars[j];
            let cost = if sc == tc { 0 } else { 1 };
            
            // deletion: row[j+1] (value from previous row, same col) + 1
            // insertion: row[j] (value from current row, prev col) + 1
            // substitution: prev_substitution_cost (value from prev row, prev col) + cost
            
            let deletion = row[j + 1] + 1;
            let insertion = row[j] + 1;
            let substitution = prev_substitution_cost + cost;
            
            // Save current cell value before overwriting it (it becomes prev_sub_cost for next col)
            prev_substitution_cost = row[j + 1];
            
            let cell_cost = substitution.min(deletion).min(insertion);
            row[j + 1] = cell_cost;
            
            if cell_cost < row_min {
                row_min = cell_cost;
            }
        }
        
        // Early termination if optimal path exceeds max_distance
        if let Some(max_d) = max_distance {
            if row_min > max_d {
                return None;
            }
        }
    }

    let result = row[n_target];
    if let Some(max_d) = max_distance {
        if result > max_d {
            return None;
        }
    }
    Some(result)
}

/// Compute Levenshtein distance with optional max threshold.
///
/// **Deprecated**: Use `levenshtein_distance_bounded` for proper Option semantics.
/// This function returns `usize::MAX` when threshold is exceeded, which can be
/// confused with legitimate distance values.
///
/// Returns `usize::MAX` if distance exceeds max_distance (when provided).
#[inline]
#[deprecated(since = "0.2.0", note = "Use levenshtein_distance_bounded for proper Option semantics")]
pub fn levenshtein_distance(a: &str, b: &str, max_distance: Option<usize>) -> usize {
    levenshtein_distance_bounded(a, b, max_distance).unwrap_or(usize::MAX)
}

/// Convenience function for simple distance calculation
#[inline]
pub fn levenshtein(a: &str, b: &str) -> usize {
    // This never fails because there's no threshold
    levenshtein_distance_bounded(a, b, None).unwrap_or(0)
}

/// Convenience function for normalized similarity (0.0 to 1.0)
#[inline]
pub fn levenshtein_similarity(a: &str, b: &str) -> f64 {
    let dist = levenshtein(a, b);
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        1.0
    } else {
        1.0 - (dist as f64 / max_len as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_basic() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("saturday", "sunday"), 3);
    }

    #[test]
    fn test_levenshtein_unicode() {
        assert_eq!(levenshtein("café", "cafe"), 1);
        assert_eq!(levenshtein("日本語", "日本"), 1);
    }

    #[test]
    fn test_levenshtein_bounded_returns_none_when_exceeded() {
        // Exceeds threshold - returns None
        assert_eq!(levenshtein_distance_bounded("abcdef", "ghijkl", Some(3)), None);
        // Within threshold - returns actual distance
        assert_eq!(levenshtein_distance_bounded("abc", "abd", Some(2)), Some(1));
        // Equal strings always return Some(0)
        assert_eq!(levenshtein_distance_bounded("abc", "abc", Some(0)), Some(0));
    }

    #[test]
    fn test_levenshtein_bounded_no_threshold() {
        // Without threshold, always returns Some
        assert_eq!(levenshtein_distance_bounded("kitten", "sitting", None), Some(3));
        assert_eq!(levenshtein_distance_bounded("", "", None), Some(0));
        assert_eq!(levenshtein_distance_bounded("abc", "", None), Some(3));
    }

    #[test]
    fn test_levenshtein_struct_compute() {
        let lev = Levenshtein::with_max_distance(2);
        assert_eq!(lev.compute("abc", "abd"), Some(1));
        assert_eq!(lev.compute("abc", "xyz"), None);

        let lev_unbounded = Levenshtein::new();
        assert_eq!(lev_unbounded.compute("abc", "xyz"), Some(3));
    }

    #[test]
    #[allow(deprecated)]
    fn test_deprecated_levenshtein_distance() {
        // Backward compatibility: deprecated function still works
        assert_eq!(levenshtein_distance("abcdef", "ghijkl", Some(3)), usize::MAX);
        assert_eq!(levenshtein_distance("abc", "abd", Some(2)), 1);
    }
}
