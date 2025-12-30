//! String normalization utilities
//!
//! Provides various normalization modes for preprocessing strings
//! before comparison.

use serde::{Deserialize, Serialize};

/// Normalization mode for string preprocessing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationMode {
    /// Convert to lowercase only
    Lowercase,
    /// Apply Unicode NFKD normalization
    UnicodeNFKD,
    /// Remove punctuation characters
    RemovePunctuation,
    /// Remove all whitespace
    RemoveWhitespace,
    /// Apply all normalizations
    Strict,
}

/// Normalize a string according to the specified mode
#[must_use]
pub fn normalize_string(s: &str, mode: NormalizationMode) -> String {
    match mode {
        NormalizationMode::Lowercase => s.to_lowercase(),
        NormalizationMode::UnicodeNFKD => {
            use unicode_normalization::UnicodeNormalization;
            s.nfkd().collect::<String>()
        }
        NormalizationMode::RemovePunctuation => {
            s.chars().filter(|c| !c.is_ascii_punctuation()).collect()
        }
        NormalizationMode::RemoveWhitespace => s.chars().filter(|c| !c.is_whitespace()).collect(),
        NormalizationMode::Strict => {
            use unicode_normalization::UnicodeNormalization;
            s.nfkd()
                .collect::<String>()
                .to_lowercase()
                .chars()
                .filter(|c| !c.is_ascii_punctuation() && !c.is_whitespace())
                .collect()
        }
    }
}

/// Normalize both strings according to the specified mode
#[must_use]
pub fn normalize_pair(a: &str, b: &str, mode: NormalizationMode) -> (String, String) {
    (normalize_string(a, mode), normalize_string(b, mode))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowercase() {
        assert_eq!(
            normalize_string("Hello World", NormalizationMode::Lowercase),
            "hello world"
        );
    }

    #[test]
    fn test_remove_punctuation() {
        assert_eq!(
            normalize_string("Hello, World!", NormalizationMode::RemovePunctuation),
            "Hello World"
        );
    }

    #[test]
    fn test_remove_whitespace() {
        assert_eq!(
            normalize_string("Hello World", NormalizationMode::RemoveWhitespace),
            "HelloWorld"
        );
    }

    #[test]
    fn test_strict() {
        assert_eq!(
            normalize_string("  Hello, World!  ", NormalizationMode::Strict),
            "helloworld"
        );
    }

    #[test]
    fn test_normalize_pair() {
        let (a, b) = normalize_pair("Hello", "WORLD", NormalizationMode::Lowercase);
        assert_eq!(a, "hello");
        assert_eq!(b, "world");
    }
}
