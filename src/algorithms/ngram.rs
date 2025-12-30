//! N-gram similarity implementation
//!
//! Compares strings based on shared n-character substrings.
//! Excellent for partial matching and manufacturer codes.
//!
//! # N-gram Size Limits
//!
//! Valid n-gram sizes are in the range 1-32 (inclusive).
//! - `n = 0` returns empty results / 0.0 similarity (invalid)
//! - `n > 32` is clamped to 32 to prevent excessive memory usage

use super::Similarity;
use ahash::AHashSet;

/// Maximum valid n-gram size. Values above this are clamped.
pub const MAX_NGRAM_SIZE: usize = 32;

/// Validate and clamp n-gram size to valid range.
/// Returns 0 for n=0 (caller should handle as error), otherwise clamps to 1-32.
#[inline]
fn validate_ngram_size(n: usize) -> usize {
    if n == 0 {
        0 // Invalid - callers should return early
    } else {
        n.min(MAX_NGRAM_SIZE)
    }
}

/// Configuration for N-gram similarity
#[derive(Debug, Clone, Copy)]
pub struct NgramConfig {
    /// Size of each n-gram (typically 2-3)
    pub n: usize,
    /// Whether to pad strings for edge matching
    pub pad: bool,
    /// Padding character
    pub pad_char: char,
}

impl Default for NgramConfig {
    fn default() -> Self {
        Self {
            n: 2,
            pad: true,
            pad_char: ' ',
        }
    }
}

/// N-gram similarity calculator
///
/// # Parameters
/// - `n`: Size of n-grams (2 for bigram, 3 for trigram)
/// - `pad`: Whether to add padding for edge matching
///
/// # Complexity
/// - Time: O(m+n) for n-gram extraction and comparison
/// - Space: O(m+n) for n-gram sets
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ngram {
    /// Size of each n-gram (typically 2-3)
    pub n: usize,
    /// Whether to pad strings for edge matching
    pub pad: bool,
    /// Padding character
    pub pad_char: char,
}

impl Default for Ngram {
    fn default() -> Self {
        Self::from_config(NgramConfig::default())
    }
}

impl Ngram {
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self { n, ..Default::default() }
    }

    /// Create from configuration
    #[must_use]
    pub fn from_config(config: NgramConfig) -> Self {
        Self {
            n: config.n,
            pad: config.pad,
            pad_char: config.pad_char,
        }
    }

    /// Get current configuration
    #[must_use]
    pub fn config(&self) -> NgramConfig {
        NgramConfig {
            n: self.n,
            pad: self.pad,
            pad_char: self.pad_char,
        }
    }

    #[must_use]
    pub fn bigram() -> Self {
        Self::new(2)
    }

    #[must_use]
    pub fn trigram() -> Self {
        Self::new(3)
    }

    #[must_use]
    pub fn with_padding(mut self, pad: bool) -> Self {
        self.pad = pad;
        self
    }

    /// Extract n-grams from a string
    #[must_use]
    pub fn extract(&self, s: &str) -> Vec<String> {
        extract_ngrams(s, self.n, self.pad, self.pad_char)
    }
}

impl Similarity for Ngram {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        ngram_similarity(a, b, self.n, self.pad, self.pad_char)
    }
    
    fn name(&self) -> &'static str {
        "ngram"
    }
}

/// Extract n-grams from a string.
///
/// # Arguments
/// * `n` - N-gram size (1-32). Values of 0 return empty vec, values >32 are clamped.
#[must_use]
pub fn extract_ngrams(s: &str, n: usize, pad: bool, pad_char: char) -> Vec<String> {
    let n = validate_ngram_size(n);
    if n == 0 {
        return vec![];
    }

    let chars: Vec<char> = if pad {
        let char_count = s.chars().count();
        let mut result = Vec::with_capacity(char_count + 2 * (n - 1));
        result.extend(std::iter::repeat(pad_char).take(n - 1));
        result.extend(s.chars());
        result.extend(std::iter::repeat(pad_char).take(n - 1));
        result
    } else {
        s.chars().collect()
    };

    if chars.len() < n {
        return vec![];
    }

    chars.windows(n)
        .map(|w| w.iter().collect())
        .collect()
}

/// Extract n-grams as a set for fast comparison
#[must_use]
pub fn extract_ngram_set(s: &str, n: usize, pad: bool, pad_char: char) -> AHashSet<String> {
    extract_ngrams(s, n, pad, pad_char).into_iter().collect()
}

/// Calculate n-gram similarity (Sørensen-Dice coefficient).
/// Returns 0.0 if n is 0 (no valid n-grams can be extracted).
#[must_use]
pub fn ngram_similarity(a: &str, b: &str, n: usize, pad: bool, pad_char: char) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }

    let a_ngrams = extract_ngram_set(a, n, pad, pad_char);
    let b_ngrams = extract_ngram_set(b, n, pad, pad_char);
    
    if a_ngrams.is_empty() && b_ngrams.is_empty() {
        return 1.0;
    }
    
    if a_ngrams.is_empty() || b_ngrams.is_empty() {
        return 0.0;
    }
    
    let intersection = a_ngrams.intersection(&b_ngrams).count();
    
    // Sørensen-Dice coefficient
    (2.0 * intersection as f64) / (a_ngrams.len() + b_ngrams.len()) as f64
}

/// Jaccard similarity based on n-grams.
/// Returns 0.0 if n is 0 (no valid n-grams can be extracted).
#[must_use]
pub fn ngram_jaccard_similarity(a: &str, b: &str, n: usize, pad: bool, pad_char: char) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }

    let a_ngrams = extract_ngram_set(a, n, pad, pad_char);
    let b_ngrams = extract_ngram_set(b, n, pad, pad_char);
    
    if a_ngrams.is_empty() && b_ngrams.is_empty() {
        return 1.0;
    }
    
    let intersection = a_ngrams.intersection(&b_ngrams).count();
    let union = a_ngrams.union(&b_ngrams).count();
    
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Convenience functions for common n-gram sizes
#[inline]
#[must_use]
pub fn bigram_similarity(a: &str, b: &str) -> f64 {
    ngram_similarity(a, b, 2, true, ' ')
}

#[inline]
#[must_use]
pub fn trigram_similarity(a: &str, b: &str) -> f64 {
    ngram_similarity(a, b, 3, true, ' ')
}

/// Profile-based n-gram similarity for multiset comparison.
/// Counts n-gram frequencies instead of just presence.
/// Returns 0.0 if n is 0 (no valid n-grams can be extracted).
#[must_use]
pub fn ngram_profile_similarity(a: &str, b: &str, n: usize) -> f64 {
    use ahash::AHashMap;

    if n == 0 {
        return 0.0;
    }

    fn build_profile(s: &str, n: usize) -> AHashMap<String, usize> {
        let mut profile = AHashMap::new();
        for ngram in extract_ngrams(s, n, true, ' ') {
            *profile.entry(ngram).or_insert(0) += 1;
        }
        profile
    }

    if a == b {
        return 1.0;
    }
    
    let a_profile = build_profile(a, n);
    let b_profile = build_profile(b, n);
    
    if a_profile.is_empty() && b_profile.is_empty() {
        return 1.0;
    }
    
    let mut intersection = 0usize;
    let mut union = 0usize;
    
    let all_keys: AHashSet<_> = a_profile.keys().chain(b_profile.keys()).collect();
    
    for key in all_keys {
        let a_count = *a_profile.get(key).unwrap_or(&0);
        let b_count = *b_profile.get(key).unwrap_or(&0);
        intersection += a_count.min(b_count);
        union += a_count.max(b_count);
    }
    
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_ngrams() {
        let ngrams = extract_ngrams("abc", 2, false, ' ');
        assert_eq!(ngrams, vec!["ab", "bc"]);
        
        let ngrams_padded = extract_ngrams("abc", 2, true, ' ');
        assert_eq!(ngrams_padded, vec![" a", "ab", "bc", "c "]);
    }
    
    #[test]
    fn test_bigram_similarity() {
        // "night" with padding: " n", "ni", "ig", "gh", "ht", "t " (6 bigrams)
        // "nacht" with padding: " n", "na", "ac", "ch", "ht", "t " (6 bigrams)
        // Intersection: {" n", "ht", "t "} = 3
        // Sørensen-Dice = 2*3/(6+6) = 0.5
        assert!((bigram_similarity("night", "nacht") - 0.5).abs() < 0.01);
        assert!((bigram_similarity("abc", "abc") - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_trigram_similarity() {
        let sim = trigram_similarity("hello", "hallo");
        assert!(sim > 0.0 && sim < 1.0);
    }
}
