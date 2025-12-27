//! RapidFuzz-compatible convenience functions for fuzzy string matching.
//!
//! This module provides high-level functions familiar to FuzzyWuzzy/RapidFuzz users:
//! - `partial_ratio`: Best substring match ratio
//! - `token_sort_ratio`: Order-insensitive comparison
//! - `token_set_ratio`: Set-based comparison
//! - `wratio`: Weighted auto-selection of best method

use super::levenshtein::levenshtein_similarity;

/// Compute the best partial match ratio between two strings.
///
/// Slides the shorter string across the longer string and returns the
/// maximum similarity found. Useful for matching when one string is
/// a substring of the other.
///
/// # Examples
/// ```
/// use fuzzyrust::algorithms::fuzz::partial_ratio;
/// assert_eq!(partial_ratio("test", "this is a test!"), 1.0);
/// ```
#[must_use]
pub fn partial_ratio(s1: &str, s2: &str) -> f64 {
    if s1.is_empty() && s2.is_empty() {
        return 1.0;
    }
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }

    // Ensure shorter string is the "needle"
    let (shorter, longer) = if s1.chars().count() <= s2.chars().count() {
        (s1, s2)
    } else {
        (s2, s1)
    };

    let shorter_chars: Vec<char> = shorter.chars().collect();
    let longer_chars: Vec<char> = longer.chars().collect();
    let shorter_len = shorter_chars.len();
    let longer_len = longer_chars.len();

    if shorter_len == longer_len {
        return levenshtein_similarity(shorter, longer);
    }

    let mut max_score = 0.0f64;

    // Slide the shorter string across the longer string
    for start in 0..=(longer_len - shorter_len) {
        let window: String = longer_chars[start..start + shorter_len].iter().collect();
        let score = levenshtein_similarity(shorter, &window);
        max_score = max_score.max(score);
        if max_score == 1.0 {
            break; // Can't do better than perfect match
        }
    }

    max_score
}

/// Tokenize a string into words, sort them, and rejoin.
fn sorted_tokens(s: &str) -> String {
    let mut tokens: Vec<&str> = s.split_whitespace().collect();
    tokens.sort_unstable();
    tokens.join(" ")
}

/// Compute similarity after tokenizing and sorting both strings.
///
/// Useful for comparing strings where word order doesn't matter.
/// "fuzzy wuzzy was a bear" matches "was a bear fuzzy wuzzy" perfectly.
///
/// # Examples
/// ```
/// use fuzzyrust::algorithms::fuzz::token_sort_ratio;
/// assert_eq!(token_sort_ratio("fuzzy wuzzy", "wuzzy fuzzy"), 1.0);
/// ```
#[must_use]
pub fn token_sort_ratio(s1: &str, s2: &str) -> f64 {
    let sorted1 = sorted_tokens(s1);
    let sorted2 = sorted_tokens(s2);
    levenshtein_similarity(&sorted1, &sorted2)
}

/// Extract unique tokens from a string as a sorted set.
fn token_set(s: &str) -> Vec<String> {
    let mut tokens: Vec<String> = s
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .collect();
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

/// Compute set-based token similarity.
///
/// Useful for comparing strings where duplicates and order don't matter.
/// "fuzzy fuzzy was a bear" matches "fuzzy was a bear" highly.
///
/// Uses the intersection, s1-only, and s2-only token sets to compute
/// multiple comparisons and returns the maximum.
///
/// # Examples
/// ```
/// use fuzzyrust::algorithms::fuzz::token_set_ratio;
/// let score = token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear");
/// assert!(score > 0.8);
/// ```
#[must_use]
pub fn token_set_ratio(s1: &str, s2: &str) -> f64 {
    let tokens1 = token_set(s1);
    let tokens2 = token_set(s2);

    if tokens1.is_empty() && tokens2.is_empty() {
        return 1.0;
    }
    if tokens1.is_empty() || tokens2.is_empty() {
        return 0.0;
    }

    // Find intersection and differences
    let set1: std::collections::HashSet<_> = tokens1.iter().collect();
    let set2: std::collections::HashSet<_> = tokens2.iter().collect();

    let intersection: Vec<_> = set1.intersection(&set2).collect();
    let diff1: Vec<_> = set1.difference(&set2).collect();
    let diff2: Vec<_> = set2.difference(&set1).collect();

    // Build comparison strings
    let intersection_str: String = intersection.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ");
    let combined1: String = if diff1.is_empty() {
        intersection_str.clone()
    } else {
        format!("{} {}", intersection_str, diff1.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" "))
    };
    let combined2: String = if diff2.is_empty() {
        intersection_str.clone()
    } else {
        format!("{} {}", intersection_str, diff2.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" "))
    };

    // Compare intersection with each combined version
    let score1 = if intersection_str.is_empty() {
        0.0
    } else {
        levenshtein_similarity(&intersection_str, &combined1)
    };
    let score2 = if intersection_str.is_empty() {
        0.0
    } else {
        levenshtein_similarity(&intersection_str, &combined2)
    };
    let score3 = levenshtein_similarity(&combined1, &combined2);

    // Also compare sorted token strings directly
    let sorted1 = tokens1.join(" ");
    let sorted2 = tokens2.join(" ");
    let score4 = levenshtein_similarity(&sorted1, &sorted2);

    score1.max(score2).max(score3).max(score4)
}

/// Compute weighted ratio using the best method for the input.
///
/// Automatically selects the best comparison method based on string
/// characteristics:
/// - For similar-length strings: uses basic ratio
/// - For different-length strings: uses partial_ratio (scaled)
/// - Also considers token_sort_ratio and token_set_ratio (scaled)
///
/// Returns the maximum of all methods (with appropriate weights).
///
/// # Examples
/// ```
/// use fuzzyrust::algorithms::fuzz::wratio;
/// let score = wratio("hello world", "hello there world");
/// assert!(score > 0.6);
/// ```
#[must_use]
pub fn wratio(s1: &str, s2: &str) -> f64 {
    if s1.is_empty() && s2.is_empty() {
        return 1.0;
    }
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }

    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let len_ratio = len1.max(len2) as f64 / len1.min(len2).max(1) as f64;

    // Basic ratio
    let base_ratio = levenshtein_similarity(s1, s2);

    // Partial ratio (weighted based on length difference)
    let partial = partial_ratio(s1, s2);
    let partial_weight = if len_ratio > 8.0 {
        0.6
    } else if len_ratio > 1.5 {
        0.9
    } else {
        0.95
    };
    let partial_score = partial * partial_weight;

    // Token-based ratios (slightly lower weight)
    let token_sort = token_sort_ratio(s1, s2) * 0.95;
    let token_set = token_set_ratio(s1, s2) * 0.95;

    base_ratio
        .max(partial_score)
        .max(token_sort)
        .max(token_set)
}

/// Compute basic similarity ratio (alias for levenshtein_similarity).
///
/// This provides API compatibility with RapidFuzz's `fuzz.ratio`.
#[must_use]
pub fn ratio(s1: &str, s2: &str) -> f64 {
    levenshtein_similarity(s1, s2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_ratio_identical() {
        assert_eq!(partial_ratio("hello", "hello"), 1.0);
    }

    #[test]
    fn test_partial_ratio_substring() {
        // "test" is a perfect substring of "this is a test"
        assert_eq!(partial_ratio("test", "this is a test"), 1.0);
    }

    #[test]
    fn test_partial_ratio_empty() {
        assert_eq!(partial_ratio("", ""), 1.0);
        assert_eq!(partial_ratio("", "hello"), 0.0);
        assert_eq!(partial_ratio("hello", ""), 0.0);
    }

    #[test]
    fn test_token_sort_ratio_reordered() {
        assert_eq!(token_sort_ratio("hello world", "world hello"), 1.0);
    }

    #[test]
    fn test_token_sort_ratio_different() {
        let score = token_sort_ratio("hello world", "hello there");
        assert!(score > 0.5);
        assert!(score < 1.0);
    }

    #[test]
    fn test_token_set_ratio_duplicates() {
        let score = token_set_ratio("fuzzy fuzzy", "fuzzy");
        assert!(score > 0.9); // Nearly identical after dedup
    }

    #[test]
    fn test_token_set_ratio_overlap() {
        let score = token_set_ratio("hello world", "hello");
        assert!(score > 0.6);
    }

    #[test]
    fn test_wratio_auto_select() {
        // Should use best method automatically
        let score = wratio("hello", "hello world");
        assert!(score > 0.8);
    }

    #[test]
    fn test_wratio_identical() {
        assert_eq!(wratio("hello", "hello"), 1.0);
    }

    #[test]
    fn test_ratio_basic() {
        let score = ratio("hello", "hallo");
        assert!(score > 0.7);
    }
}
