//! Longest Common Subsequence (LCS) implementation
//!
//! Finds the longest subsequence present in both strings.
//! Useful for detecting partial matches and rearrangements.
//!
//! # Complexity
//! - Time: O(m*n) for length calculation, O(m*n) for string extraction
//! - Space: O(n) for length (space-optimized), O(m*n) for string extraction
//!
//! # String Length Limits
//!
//! **Important**: Functions that require O(m*n) space (`lcs_string`, `longest_common_substring`)
//! will return an empty string for inputs longer than 10,000 characters to prevent
//! excessive memory allocation (potential DoS vector).
//!
//! Space-efficient alternatives:
//! - Use `lcs_length` instead of `lcs_string` (O(n) space)
//! - Use `longest_common_substring_length` instead of `longest_common_substring` (O(n) space)
//! - Use `lcs_similarity` for similarity scoring (uses `lcs_length` internally)

use super::Similarity;

/// Maximum string length for O(m*n) space algorithms.
/// Strings longer than this will return empty string to prevent
/// DoS attacks via excessive memory allocation.
const MAX_QUADRATIC_STRING_LENGTH: usize = 10_000;

/// LCS-based similarity calculator
///
/// Stateless calculator - all instances are equivalent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Lcs;

impl Lcs {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Similarity for Lcs {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        lcs_similarity(a, b)
    }

    fn name(&self) -> &'static str {
        "lcs"
    }
}

/// Calculate the length of the Longest Common Subsequence.
#[must_use]
pub fn lcs_length(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 || n == 0 {
        return 0;
    }

    // Space-optimized: only keep current and previous row
    let mut prev: Vec<usize> = vec![0; n + 1];
    let mut curr: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = 0; // Reset first element for this row
        for j in 1..=n {
            if a_chars[i - 1] == b_chars[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Get the actual LCS string (not just length).
///
/// Note: For strings longer than 10,000 characters, this returns an empty
/// string to prevent excessive memory allocation (O(m*n) space required).
/// Use `lcs_length` for a space-efficient length-only computation.
#[must_use]
pub fn lcs_string(a: &str, b: &str) -> String {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 || n == 0 {
        return String::new();
    }

    // For very long strings, return empty to prevent DoS via excessive memory allocation
    if m > MAX_QUADRATIC_STRING_LENGTH || n > MAX_QUADRATIC_STRING_LENGTH {
        return String::new();
    }

    // Full DP table needed for backtracking
    let mut dp: Vec<Vec<usize>> = vec![vec![0; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a_chars[i - 1] == b_chars[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Backtrack to find the LCS
    let mut lcs = Vec::with_capacity(dp[m][n]);
    let mut i = m;
    let mut j = n;

    while i > 0 && j > 0 {
        if a_chars[i - 1] == b_chars[j - 1] {
            lcs.push(a_chars[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    lcs.reverse();
    lcs.into_iter().collect()
}

/// Calculate LCS-based similarity (0.0 to 1.0).
/// Uses the formula: 2 * LCS_length / (len(a) + len(b))
#[must_use]
pub fn lcs_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }

    let len_a = a.chars().count();
    let len_b = b.chars().count();

    if len_a == 0 && len_b == 0 {
        return 1.0;
    }

    if len_a == 0 || len_b == 0 {
        return 0.0;
    }

    let lcs_len = lcs_length(a, b);

    // Dice coefficient formula
    (2.0 * lcs_len as f64) / (len_a + len_b) as f64
}

/// Alternative similarity using max length as denominator
#[must_use]
pub fn lcs_similarity_max(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }

    let len_a = a.chars().count();
    let len_b = b.chars().count();
    let max_len = len_a.max(len_b);

    if max_len == 0 {
        return 1.0;
    }

    lcs_length(a, b) as f64 / max_len as f64
}

/// Longest Common Substring (contiguous, not subsequence)
#[must_use]
pub fn longest_common_substring_length(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 || n == 0 {
        return 0;
    }

    let mut prev: Vec<usize> = vec![0; n + 1];
    let mut curr: Vec<usize> = vec![0; n + 1];
    let mut max_len = 0;

    for i in 1..=m {
        curr[0] = 0; // Reset first element for this row
        for j in 1..=n {
            if a_chars[i - 1] == b_chars[j - 1] {
                curr[j] = prev[j - 1] + 1;
                max_len = max_len.max(curr[j]);
            } else {
                curr[j] = 0;
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    max_len
}

/// Get the actual longest common substring
///
/// Note: For strings longer than 10,000 characters, this returns an empty
/// string to prevent excessive memory allocation (O(m*n) space required).
/// Use `longest_common_substring_length` for a space-efficient length-only computation.
#[must_use]
pub fn longest_common_substring(a: &str, b: &str) -> String {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 || n == 0 {
        return String::new();
    }

    // For very long strings, return empty to prevent DoS via excessive memory allocation
    if m > MAX_QUADRATIC_STRING_LENGTH || n > MAX_QUADRATIC_STRING_LENGTH {
        return String::new();
    }

    let mut dp: Vec<Vec<usize>> = vec![vec![0; n + 1]; m + 1];
    let mut max_len = 0;
    let mut end_idx = 0;

    for i in 1..=m {
        for j in 1..=n {
            if a_chars[i - 1] == b_chars[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                if dp[i][j] > max_len {
                    max_len = dp[i][j];
                    end_idx = i;
                }
            }
        }
    }

    a_chars[end_idx - max_len..end_idx].iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcs_length() {
        assert_eq!(lcs_length("", ""), 0);
        assert_eq!(lcs_length("abc", "abc"), 3);
        assert_eq!(lcs_length("abc", "def"), 0);
        assert_eq!(lcs_length("ABCDGH", "AEDFHR"), 3); // ADH
        assert_eq!(lcs_length("AGGTAB", "GXTXAYB"), 4); // GTAB
    }

    #[test]
    fn test_lcs_string() {
        assert_eq!(lcs_string("ABCDGH", "AEDFHR"), "ADH");
        assert_eq!(lcs_string("AGGTAB", "GXTXAYB"), "GTAB");
    }

    #[test]
    fn test_longest_common_substring() {
        assert_eq!(longest_common_substring_length("abcdef", "zbcdf"), 3); // bcd
        assert_eq!(longest_common_substring("abcdef", "zbcdf"), "bcd");
    }
}
