//! Hamming distance implementation
//!
//! Counts positions where characters differ.
//! Only works on strings of equal length.
//!
//! # Important: Length Mismatch Behavior
//!
//! Hamming distance is mathematically undefined for strings of different lengths.
//! This module handles length mismatches differently depending on the trait used:
//!
//! - **`FallibleEditDistance`**: Returns `None` for unequal lengths (explicit failure)
//! - **`Similarity`**: Returns `0.0` for unequal lengths (treats as "no similarity")
//!
//! If you need Hamming-like behavior on unequal strings, use `hamming_distance_padded()`
//! which pads the shorter string with spaces.
//!
//! # Complexity
//! - Time: O(n) where n is the string length
//! - Space: O(n) for character vectors

use super::{FallibleEditDistance, Similarity};

/// Hamming distance calculator
///
/// Implements `FallibleEditDistance` because Hamming distance is only
/// defined for strings of equal length. Returns `None` for unequal lengths.
///
/// **Important**: When used via the `Similarity` trait, unequal-length strings
/// return `0.0` (no similarity) rather than failing. Use `FallibleEditDistance`
/// if you need to distinguish between "0 similarity" and "undefined comparison".
///
/// Stateless calculator - all instances are equivalent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Hamming;

impl Hamming {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl FallibleEditDistance for Hamming {
    fn distance(&self, a: &str, b: &str) -> Option<usize> {
        hamming_distance(a, b)
    }

    fn name(&self) -> &'static str {
        "hamming"
    }
}

/// Implement Similarity trait for Hamming
///
/// **Note**: Unequal-length strings return `0.0` (no similarity) since Hamming
/// distance is undefined for them. This differs from `FallibleEditDistance`
/// which returns `None` to explicitly indicate undefined comparison.
///
/// If you need to distinguish "0.0 similarity" from "undefined", use the
/// `FallibleEditDistance` trait instead:
/// ```
/// use fuzzyrust::algorithms::{Hamming, FallibleEditDistance};
/// let h = Hamming::new();
/// assert_eq!(h.distance("abc", "xyz"), Some(3)); // defined
/// assert_eq!(h.distance("abc", "xy"), None);     // undefined (different lengths)
/// ```
impl Similarity for Hamming {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        FallibleEditDistance::similarity(self, a, b).unwrap_or(0.0)
    }

    fn name(&self) -> &'static str {
        "hamming"
    }
}

/// Calculate Hamming distance between two strings.
/// Returns None if strings have different lengths.
#[must_use]
pub fn hamming_distance(a: &str, b: &str) -> Option<usize> {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    
    if a_chars.len() != b_chars.len() {
        return None;
    }
    
    Some(a_chars.iter()
        .zip(b_chars.iter())
        .filter(|(ac, bc)| ac != bc)
        .count())
}

/// Hamming distance that pads shorter string with spaces.
/// Useful when you want Hamming-like behavior on unequal strings.
#[must_use]
pub fn hamming_distance_padded(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    
    let max_len = a_chars.len().max(b_chars.len());
    
    let mut distance = 0;
    for i in 0..max_len {
        let a_char = a_chars.get(i);
        let b_char = b_chars.get(i);
        
        if a_char != b_char {
            distance += 1;
        }
    }
    
    distance
}

/// Normalized Hamming similarity (0.0 to 1.0)
/// Returns None for strings of different lengths.
#[must_use]
pub fn hamming_similarity(a: &str, b: &str) -> Option<f64> {
    let dist = hamming_distance(a, b)?;
    let len = a.chars().count();
    
    if len == 0 {
        Some(1.0)
    } else {
        Some(1.0 - (dist as f64 / len as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hamming_basic() {
        assert_eq!(hamming_distance("", ""), Some(0));
        assert_eq!(hamming_distance("abc", "abc"), Some(0));
        assert_eq!(hamming_distance("abc", "axc"), Some(1));
        assert_eq!(hamming_distance("karolin", "kathrin"), Some(3));
    }
    
    #[test]
    fn test_hamming_different_lengths() {
        assert_eq!(hamming_distance("abc", "ab"), None);
        assert_eq!(hamming_distance_padded("abc", "ab"), 1);
    }
}
