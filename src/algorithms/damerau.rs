//! Damerau-Levenshtein distance implementation
//!
//! Extends Levenshtein with transposition operations.
//! Particularly useful for typo detection where letter swaps are common.
//!
//! # String Length Limits
//!
//! **Important**: The `true_damerau_levenshtein` function (which uses O(m*n) space)
//! automatically falls back to the space-efficient restricted Damerau-Levenshtein
//! algorithm for strings longer than 10,000 characters. This prevents excessive
//! memory allocation (potential DoS vector) but may produce slightly different
//! results for edge cases involving multiple transpositions.
//!
//! The restricted variant (`damerau_levenshtein_distance_bounded`) uses O(n) space
//! and works efficiently for strings of any length.
//!
//! For explicit error handling when strings exceed the limit, use
//! `true_damerau_levenshtein_checked` which returns a `Result`.

use super::EditDistance;
use ahash::AHashMap;
use smallvec::SmallVec;
use std::fmt;

/// Maximum string length for O(m*n) space algorithms.
/// Strings longer than this will use space-efficient fallback algorithms.
/// This prevents DoS attacks via excessive memory allocation.
const MAX_QUADRATIC_STRING_LENGTH: usize = 10_000;

/// Error returned when string length exceeds the safe limit for O(m*n) algorithms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringTooLongError {
    /// Length of the first string
    pub len_a: usize,
    /// Length of the second string
    pub len_b: usize,
    /// Maximum allowed length
    pub max_length: usize,
}

impl fmt::Display for StringTooLongError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "String length exceeds safe limit for O(m*n) algorithm: \
             len_a={}, len_b={}, max={}",
            self.len_a, self.len_b, self.max_length
        )
    }
}

impl std::error::Error for StringTooLongError {}

/// Damerau-Levenshtein distance calculator
///
/// Uses the "restricted" (optimal string alignment) variant which handles
/// transpositions without allowing multiple operations on the same character.
///
/// # Complexity
/// - Time: O(m*n) where m and n are string lengths
/// - Space: O(m*n) for full matrix, or O(n) fallback for very long strings
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DamerauLevenshtein {
    /// Maximum distance for early termination
    pub max_distance: Option<usize>,
}

impl DamerauLevenshtein {
    #[must_use]
    pub fn new() -> Self {
        Self { max_distance: None }
    }

    #[must_use]
    pub fn with_max_distance(max_distance: usize) -> Self {
        Self {
            max_distance: Some(max_distance),
        }
    }

    /// Compute distance with proper Option semantics.
    /// Returns `None` if distance exceeds max_distance threshold.
    /// Returns `Some(distance)` otherwise.
    #[must_use]
    pub fn compute(&self, a: &str, b: &str) -> Option<usize> {
        damerau_levenshtein_distance_bounded(a, b, self.max_distance)
    }
}

impl EditDistance for DamerauLevenshtein {
    fn distance(&self, a: &str, b: &str) -> usize {
        // When max_distance is set and exceeded, return max_distance + 1
        // to indicate "greater than threshold" without using sentinel values
        match self.max_distance {
            Some(max_d) => damerau_levenshtein_distance_bounded(a, b, Some(max_d))
                .unwrap_or(max_d.saturating_add(1)),
            None => damerau_levenshtein_distance_bounded(a, b, None).unwrap_or(0),
        }
    }

    fn name(&self) -> &'static str {
        "damerau_levenshtein"
    }
}

/// Compute optimal string alignment distance (restricted Damerau-Levenshtein).
///
/// Returns `None` if distance exceeds `max_distance` (early termination).
/// Returns `Some(distance)` if distance is within threshold or no threshold set.
///
/// This version doesn't allow multiple edits on the same substring.
#[inline]
#[must_use]
pub fn damerau_levenshtein_distance_bounded(
    a: &str,
    b: &str,
    max_distance: Option<usize>,
) -> Option<usize> {
    if a == b {
        return Some(0);
    }

    // Use SmallVec to avoid heap allocation for typical string lengths
    let a_chars: SmallVec<[char; 64]> = a.chars().collect();
    let b_chars: SmallVec<[char; 64]> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return Some(n);
    }
    if n == 0 {
        return Some(m);
    }

    if let Some(max_d) = max_distance {
        if m.abs_diff(n) > max_d {
            return None;
        }
    }

    // Need 3 rows for transposition detection - use SmallVec
    let mut prev2_row: SmallVec<[usize; 64]> = smallvec::smallvec![0; n + 1];
    let mut prev_row: SmallVec<[usize; 64]> = (0..=n).collect();
    let mut curr_row: SmallVec<[usize; 64]> = smallvec::smallvec![0; n + 1];

    for i in 1..=m {
        curr_row[0] = i;
        let mut row_min = i;

        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };

            curr_row[j] = (prev_row[j] + 1) // deletion
                .min(curr_row[j - 1] + 1) // insertion
                .min(prev_row[j - 1] + cost); // substitution

            // Transposition check
            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                curr_row[j] = curr_row[j].min(prev2_row[j - 2] + 1);
            }

            row_min = row_min.min(curr_row[j]);
        }

        if let Some(max_d) = max_distance {
            if row_min > max_d {
                return None;
            }
        }

        // Rotate rows
        std::mem::swap(&mut prev2_row, &mut prev_row);
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    Some(prev_row[n])
}

/// Compute optimal string alignment distance (restricted Damerau-Levenshtein).
///
/// **Deprecated**: Use `damerau_levenshtein_distance_bounded` for proper Option semantics.
/// This function returns `usize::MAX` when threshold is exceeded.
#[inline]
#[must_use]
#[deprecated(
    since = "0.2.0",
    note = "Use damerau_levenshtein_distance_bounded for proper Option semantics"
)]
pub fn damerau_levenshtein_distance(a: &str, b: &str, max_distance: Option<usize>) -> usize {
    damerau_levenshtein_distance_bounded(a, b, max_distance).unwrap_or(usize::MAX)
}

/// True Damerau-Levenshtein using the algorithm with full transposition support.
/// More computationally expensive but handles all edit sequences.
///
/// Note: For strings longer than 10,000 characters, this falls back to the
/// space-efficient restricted Damerau-Levenshtein algorithm to prevent
/// excessive memory allocation.
#[must_use]
pub fn true_damerau_levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // For very long strings, fall back to space-efficient algorithm
    // to prevent DoS via excessive memory allocation (O(m*n) space)
    if m > MAX_QUADRATIC_STRING_LENGTH || n > MAX_QUADRATIC_STRING_LENGTH {
        // Use bounded version - this never fails when max_distance is None
        return damerau_levenshtein_distance_bounded(a, b, None).unwrap_or(0);
    }

    let max_dist = m + n;

    // Character position map
    let mut char_map: AHashMap<char, usize> = AHashMap::new();

    // DP matrix with extra row and column
    let mut d: Vec<Vec<usize>> = vec![vec![0; n + 2]; m + 2];

    d[0][0] = max_dist;
    for i in 0..=m {
        d[i + 1][0] = max_dist;
        d[i + 1][1] = i;
    }
    for j in 0..=n {
        d[0][j + 1] = max_dist;
        d[1][j + 1] = j;
    }

    for i in 1..=m {
        let mut db = 0usize;

        for j in 1..=n {
            let i1 = *char_map.get(&b_chars[j - 1]).unwrap_or(&0);
            let j1 = db;

            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                db = j;
                0
            } else {
                1
            };

            d[i + 1][j + 1] = (d[i][j] + cost) // substitution
                .min(d[i + 1][j] + 1) // insertion
                .min(d[i][j + 1] + 1) // deletion
                .min(d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1)); // transposition
        }

        char_map.insert(a_chars[i - 1], i);
    }

    d[m + 1][n + 1]
}

/// True Damerau-Levenshtein with explicit error handling for long strings.
///
/// Unlike `true_damerau_levenshtein` which silently falls back to the OSA variant
/// for strings longer than 10,000 characters, this function returns an error,
/// allowing callers to handle the case explicitly.
///
/// # Errors
///
/// Returns `StringTooLongError` if either string exceeds 10,000 characters.
///
/// # Example
///
/// ```
/// use fuzzyrust::algorithms::damerau::{true_damerau_levenshtein_checked, StringTooLongError};
///
/// // Normal usage
/// assert_eq!(true_damerau_levenshtein_checked("hello", "hallo"), Ok(1));
///
/// // Handle long strings explicitly
/// let long_string = "a".repeat(15000);
/// match true_damerau_levenshtein_checked(&long_string, "test") {
///     Ok(dist) => println!("Distance: {}", dist),
///     Err(e) => println!("String too long: {}", e),
/// }
/// ```
pub fn true_damerau_levenshtein_checked(a: &str, b: &str) -> Result<usize, StringTooLongError> {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return Ok(n);
    }
    if n == 0 {
        return Ok(m);
    }

    // Return error for very long strings instead of silent fallback
    if m > MAX_QUADRATIC_STRING_LENGTH || n > MAX_QUADRATIC_STRING_LENGTH {
        return Err(StringTooLongError {
            len_a: m,
            len_b: n,
            max_length: MAX_QUADRATIC_STRING_LENGTH,
        });
    }

    let max_dist = m + n;

    // Character position map
    let mut char_map: AHashMap<char, usize> = AHashMap::new();

    // DP matrix with extra row and column
    let mut d: Vec<Vec<usize>> = vec![vec![0; n + 2]; m + 2];

    d[0][0] = max_dist;
    for i in 0..=m {
        d[i + 1][0] = max_dist;
        d[i + 1][1] = i;
    }
    for j in 0..=n {
        d[0][j + 1] = max_dist;
        d[1][j + 1] = j;
    }

    for i in 1..=m {
        let mut db = 0usize;

        for j in 1..=n {
            let i1 = *char_map.get(&b_chars[j - 1]).unwrap_or(&0);
            let j1 = db;

            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                db = j;
                0
            } else {
                1
            };

            d[i + 1][j + 1] = (d[i][j] + cost) // substitution
                .min(d[i + 1][j] + 1) // insertion
                .min(d[i][j + 1] + 1) // deletion
                .min(d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1)); // transposition
        }

        char_map.insert(a_chars[i - 1], i);
    }

    Ok(d[m + 1][n + 1])
}

/// Convenience function - uses the "true" Damerau-Levenshtein algorithm
/// which allows multiple edits on the same substring.
///
/// This matches the behavior of jellyfish's `damerau_levenshtein_distance`.
/// For the restricted (OSA) variant, use `optimal_string_alignment`.
#[inline]
#[must_use]
pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
    true_damerau_levenshtein(a, b)
}

/// Optimal String Alignment (OSA) distance - restricted Damerau-Levenshtein.
///
/// This is the restricted variant that doesn't allow multiple edits on the
/// same substring. It's slightly faster than the true algorithm but may
/// produce different results for certain edge cases.
///
/// Example where OSA differs from true Damerau-Levenshtein:
/// - "CA" -> "ABC": OSA = 3 (sub C->A, sub A->B, ins C), True DL = 2 (transpose CA->AC, ins B)
#[inline]
#[must_use]
pub fn optimal_string_alignment(a: &str, b: &str) -> usize {
    // This never fails because there's no threshold
    damerau_levenshtein_distance_bounded(a, b, None).unwrap_or(0)
}

/// Normalized similarity (0.0 to 1.0) using true Damerau-Levenshtein.
#[inline]
#[must_use]
pub fn damerau_levenshtein_similarity(a: &str, b: &str) -> f64 {
    let dist = true_damerau_levenshtein(a, b);
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        1.0
    } else {
        1.0 - (dist as f64 / max_len as f64)
    }
}

/// Normalized similarity (0.0 to 1.0) using OSA (restricted Damerau-Levenshtein).
#[inline]
#[must_use]
pub fn optimal_string_alignment_similarity(a: &str, b: &str) -> f64 {
    let dist = optimal_string_alignment(a, b);
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
    fn test_damerau_basic() {
        assert_eq!(damerau_levenshtein("", ""), 0);
        assert_eq!(damerau_levenshtein("abc", "abc"), 0);
        assert_eq!(damerau_levenshtein("ab", "ba"), 1); // transposition
        assert_eq!(damerau_levenshtein("abc", "acb"), 1); // transposition
    }

    #[test]
    fn test_transposition_vs_levenshtein() {
        // Levenshtein would give 2, Damerau gives 1
        assert_eq!(damerau_levenshtein("ca", "ac"), 1);
    }

    #[test]
    fn test_true_vs_osa() {
        // This case shows the difference between true DL and OSA
        // '00210000' vs '001020000' - true DL = 2, OSA = 3
        assert_eq!(true_damerau_levenshtein("00210000", "001020000"), 2);
        assert_eq!(optimal_string_alignment("00210000", "001020000"), 3);
    }

    #[test]
    fn test_osa_basic() {
        assert_eq!(optimal_string_alignment("", ""), 0);
        assert_eq!(optimal_string_alignment("abc", "abc"), 0);
        assert_eq!(optimal_string_alignment("ab", "ba"), 1); // transposition
        assert_eq!(optimal_string_alignment("abc", "acb"), 1); // transposition
    }

    #[test]
    fn test_damerau_bounded_returns_none_when_exceeded() {
        // Exceeds threshold - returns None
        assert_eq!(
            damerau_levenshtein_distance_bounded("abcdef", "ghijkl", Some(3)),
            None
        );
        // Within threshold - returns actual distance
        assert_eq!(
            damerau_levenshtein_distance_bounded("abc", "acb", Some(2)),
            Some(1)
        );
        // Equal strings always return Some(0)
        assert_eq!(
            damerau_levenshtein_distance_bounded("abc", "abc", Some(0)),
            Some(0)
        );
    }

    #[test]
    fn test_damerau_struct_compute() {
        let dl = DamerauLevenshtein::with_max_distance(2);
        assert_eq!(dl.compute("abc", "acb"), Some(1)); // transposition within limit
        assert_eq!(dl.compute("abc", "xyz"), None); // exceeds limit

        let dl_unbounded = DamerauLevenshtein::new();
        assert_eq!(dl_unbounded.compute("abc", "xyz"), Some(3));
    }

    #[test]
    fn test_true_damerau_levenshtein_checked() {
        // Normal cases should return Ok
        assert_eq!(true_damerau_levenshtein_checked("hello", "hallo"), Ok(1));
        assert_eq!(true_damerau_levenshtein_checked("", ""), Ok(0));
        assert_eq!(true_damerau_levenshtein_checked("abc", ""), Ok(3));
        assert_eq!(true_damerau_levenshtein_checked("", "abc"), Ok(3));
        assert_eq!(true_damerau_levenshtein_checked("ab", "ba"), Ok(1)); // transposition

        // Results should match true_damerau_levenshtein for short strings
        assert_eq!(
            true_damerau_levenshtein_checked("00210000", "001020000"),
            Ok(2)
        );
    }

    #[test]
    fn test_string_too_long_error() {
        let err = StringTooLongError {
            len_a: 15000,
            len_b: 100,
            max_length: 10000,
        };
        assert!(err.to_string().contains("15000"));
        assert!(err.to_string().contains("10000"));
    }
}
