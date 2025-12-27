//! Jaro and Jaro-Winkler similarity implementations
//!
//! Excellent for name matching and short strings.
//! Jaro-Winkler gives extra weight to common prefixes.

use super::Similarity;
use smallvec::SmallVec;

// ============================================================================
// Public API
// ============================================================================

/// Jaro similarity calculator
///
/// # Complexity
/// - Time: O(m*n) for matching characters
/// - Space: O(m+n) for match flags
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Jaro;

impl Jaro {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Similarity for Jaro {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        jaro_similarity(a, b)
    }
    
    fn name(&self) -> &'static str {
        "jaro"
    }
}

/// Configuration for Jaro-Winkler similarity
#[derive(Debug, Clone, Copy)]
pub struct JaroWinklerConfig {
    /// Prefix weight (typically 0.1, max 0.25)
    pub prefix_weight: f64,
    /// Maximum prefix length to consider (typically 4)
    pub max_prefix_length: usize,
}

impl Default for JaroWinklerConfig {
    fn default() -> Self {
        Self {
            prefix_weight: 0.1,
            max_prefix_length: 4,
        }
    }
}

/// Jaro-Winkler similarity calculator
///
/// Extends Jaro similarity by giving extra weight to common prefixes.
/// Best for names and short identifiers.
///
/// # Parameters
/// - `prefix_weight`: How much to boost prefix matches (0.0-0.25, typically 0.1)
/// - `max_prefix_length`: Maximum prefix length to consider (typically 4)
#[derive(Debug, Clone, PartialEq)]
pub struct JaroWinkler {
    /// Prefix weight (typically 0.1)
    pub prefix_weight: f64,
    /// Maximum prefix length to consider (typically 4)
    pub max_prefix_length: usize,
}

impl Default for JaroWinkler {
    fn default() -> Self {
        Self::from_config(JaroWinklerConfig::default())
    }
}

impl JaroWinkler {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from configuration
    #[must_use]
    pub fn from_config(config: JaroWinklerConfig) -> Self {
        Self {
            prefix_weight: config.prefix_weight.clamp(0.0, 0.25),
            max_prefix_length: config.max_prefix_length,
        }
    }

    /// Get current configuration
    #[must_use]
    pub fn config(&self) -> JaroWinklerConfig {
        JaroWinklerConfig {
            prefix_weight: self.prefix_weight,
            max_prefix_length: self.max_prefix_length,
        }
    }

    #[must_use]
    pub fn with_prefix_weight(mut self, weight: f64) -> Self {
        self.prefix_weight = weight.clamp(0.0, 0.25); // Max 0.25 to keep score <= 1.0
        self
    }

    #[must_use]
    pub fn with_max_prefix_length(mut self, length: usize) -> Self {
        self.max_prefix_length = length;
        self
    }
}

impl Similarity for JaroWinkler {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        jaro_winkler_similarity_params(a, b, self.prefix_weight, self.max_prefix_length)
    }
    
    fn name(&self) -> &'static str {
        "jaro_winkler"
    }
}

/// Calculate Jaro similarity between two strings.
/// Returns a value between 0.0 and 1.0.
#[inline]
#[must_use]
pub fn jaro_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }

    // Use SmallVec to avoid heap allocation for typical string lengths
    let a_chars: SmallVec<[char; 64]> = a.chars().collect();
    let b_chars: SmallVec<[char; 64]> = b.chars().collect();

    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 && b_len == 0 {
        return 1.0;
    }
    if a_len == 0 || b_len == 0 {
        return 0.0;
    }

    // Use standard algorithm (bitwise optimization disabled pending fixes)
    jaro_standard(&a_chars, &b_chars)
}

/// Standard Jaro algorithm (fallback for strings > 64 chars)
#[inline]
fn jaro_standard(a_chars: &[char], b_chars: &[char]) -> f64 {
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    // Match window
    let match_distance = (a_len.max(b_len) / 2).saturating_sub(1);

    // Use SmallVec for match arrays to avoid heap allocation
    let mut a_matches: SmallVec<[bool; 64]> = smallvec::smallvec![false; a_len];
    let mut b_matches: SmallVec<[bool; 64]> = smallvec::smallvec![false; b_len];

    let mut matches = 0usize;
    let mut transpositions = 0usize;

    // Find matches
    for i in 0..a_len {
        let start = i.saturating_sub(match_distance);
        let end = (i + match_distance + 1).min(b_len);

        for j in start..end {
            if b_matches[j] || a_chars[i] != b_chars[j] {
                continue;
            }
            a_matches[i] = true;
            b_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions
    let mut k = 0usize;
    for i in 0..a_len {
        if !a_matches[i] {
            continue;
        }
        // Find next matched position in b (with bounds check for safety)
        while k < b_len && !b_matches[k] {
            k += 1;
        }
        // Safety guard (should never happen mathematically, but be defensive)
        if k >= b_len {
            break;
        }
        if a_chars[i] != b_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let matches = matches as f64;
    let transpositions = (transpositions / 2) as f64;

    (matches / a_len as f64 + matches / b_len as f64 + (matches - transpositions) / matches) / 3.0
}

/// Calculate Jaro-Winkler similarity with custom parameters.
/// Note: prefix_weight is clamped to [0.0, 0.25] to ensure the result stays in [0.0, 1.0].
#[inline]
#[must_use]
pub fn jaro_winkler_similarity_params(a: &str, b: &str, prefix_weight: f64, max_prefix_len: usize) -> f64 {
    let jaro_sim = jaro_similarity(a, b);

    if jaro_sim == 0.0 {
        return 0.0;
    }

    // Clamp prefix_weight to ensure score stays in valid range [0.0, 1.0]
    let prefix_weight = prefix_weight.clamp(0.0, 0.25);

    // Find common prefix length
    let prefix_len = a.chars()
        .zip(b.chars())
        .take(max_prefix_len)
        .take_while(|(ac, bc)| ac == bc)
        .count();

    // Jaro-Winkler formula
    jaro_sim + (prefix_len as f64 * prefix_weight * (1.0 - jaro_sim))
}

/// Calculate Jaro-Winkler similarity with default parameters.
#[inline]
#[must_use]
pub fn jaro_winkler_similarity(a: &str, b: &str) -> f64 {
    jaro_winkler_similarity_params(a, b, 0.1, 4)
}

/// Distance version (1.0 - similarity)
#[inline]
#[must_use]
pub fn jaro_distance(a: &str, b: &str) -> f64 {
    1.0 - jaro_similarity(a, b)
}

#[inline]
#[must_use]
pub fn jaro_winkler_distance(a: &str, b: &str) -> f64 {
    1.0 - jaro_winkler_similarity(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 0.001
    }
    
    #[test]
    fn test_jaro_basic() {
        assert!(approx_eq(jaro_similarity("", ""), 1.0));
        assert!(approx_eq(jaro_similarity("abc", "abc"), 1.0));
        assert!(approx_eq(jaro_similarity("abc", "xyz"), 0.0));
    }
    
    #[test]
    fn test_jaro_examples() {
        // Classic example
        assert!(approx_eq(jaro_similarity("MARTHA", "MARHTA"), 0.944));
        assert!(approx_eq(jaro_similarity("DWAYNE", "DUANE"), 0.822));
    }
    
    #[test]
    fn test_jaro_winkler_boost() {
        // Jaro-Winkler should boost strings with common prefix
        let jaro = jaro_similarity("MARTHA", "MARHTA");
        let jaro_winkler = jaro_winkler_similarity("MARTHA", "MARHTA");
        assert!(jaro_winkler > jaro);
    }
}
