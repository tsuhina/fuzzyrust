///! Field type definitions and algorithm selection for schema-based multi-field matching
///!
///! This module defines the type system for schema fields, including:
///! - FieldType: Different kinds of data (text, tokens, numeric, dates)
///! - Algorithm: Fuzzy matching algorithms for each field type
///!
///! The type system enables type-safe schema definition and optimal index selection.

use serde::{Deserialize, Serialize};

/// Field type defines the data type and indexing strategy for a schema field
///
/// Each field type has specific characteristics and optimal algorithms:
/// - `ShortText`: Names, titles, short strings (≤100 chars). Best with Jaro-Winkler.
/// - `LongText`: Descriptions, documents, paragraphs. Best with N-gram similarity.
/// - `TokenSet`: Tags, categories, comma-separated values. Best with Jaccard similarity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldType {
    /// Short text field (names, titles, IDs)
    ///
    /// Optimized for strings typically ≤100 characters.
    /// Uses character-based algorithms like Jaro-Winkler or Levenshtein.
    ///
    /// # Example
    /// ```rust
    /// use fuzzyrust::indexing::schema::types::*;
    ///
    /// let field_type = FieldType::ShortText {
    ///     max_length: 100,
    ///     default_algorithm: Algorithm::JaroWinkler(JaroWinklerConfig::default()),
    /// };
    /// ```
    ShortText {
        /// Maximum expected length (for validation)
        max_length: usize,
        /// Default algorithm if not specified per-field
        default_algorithm: Algorithm,
    },

    /// Long text field (descriptions, documents, articles)
    ///
    /// Optimized for longer strings (>100 characters).
    /// Uses token-based or n-gram algorithms for efficiency.
    ///
    /// # Example
    /// ```rust
    /// use fuzzyrust::indexing::schema::types::*;
    ///
    /// let field_type = FieldType::LongText {
    ///     default_algorithm: Algorithm::Ngram { ngram_size: 2 },
    ///     chunk_size: None,
    /// };
    /// ```
    LongText {
        /// Default algorithm if not specified per-field
        default_algorithm: Algorithm,
        /// Optional chunk size for very long text (None = no chunking)
        chunk_size: Option<usize>,
    },

    /// Token set field (tags, categories, multi-value)
    ///
    /// Optimized for comma-separated or delimited values.
    /// Uses set-based algorithms like Jaccard similarity.
    ///
    /// # Example
    /// ```rust
    /// use fuzzyrust::indexing::schema::types::*;
    ///
    /// // Single-character separator
    /// let field_type = FieldType::TokenSet {
    ///     separator: ",".to_string(),
    ///     default_algorithm: Algorithm::Jaccard,
    /// };
    ///
    /// // Multi-character separator (e.g., ", " for comma-space)
    /// let field_type = FieldType::TokenSet {
    ///     separator: ", ".to_string(),
    ///     default_algorithm: Algorithm::Jaccard,
    /// };
    /// ```
    TokenSet {
        /// Token separator string (e.g., "," for CSV, ", " for comma-space)
        /// Supports multi-character separators like " | " or " - "
        separator: String,
        /// Default algorithm if not specified per-field
        default_algorithm: Algorithm,
    },
}

/// Fuzzy matching algorithm selection
///
/// Defines which similarity/distance algorithm to use for a field.
/// Each algorithm has different characteristics:
///
/// - **Levenshtein**: Edit distance (insertions, deletions, substitutions). Good for typos.
/// - **DamerauLevenshtein**: Includes transpositions. Better for keyboard errors.
/// - **JaroWinkler**: Optimized for short strings, prefix-weighted. Best for names.
/// - **Ngram**: Token overlap. Fast for long text, handles word reordering.
/// - **Jaccard**: Set similarity. Ideal for tag/category matching.
/// - **Cosine**: Vector similarity. Good for document comparison.
/// - **ExactMatch**: No fuzzy matching, exact string comparison only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Algorithm {
    /// Levenshtein edit distance
    ///
    /// Counts minimum insertions, deletions, and substitutions to transform one string into another.
    /// Normalized to 0-1 similarity score.
    ///
    /// **Best for**: Typos, spelling errors, short strings
    /// **Performance**: O(m*n) where m, n are string lengths
    Levenshtein,

    /// Damerau-Levenshtein edit distance
    ///
    /// Like Levenshtein but also includes transpositions (swap adjacent characters).
    /// Better models keyboard typing errors.
    ///
    /// **Best for**: Keyboard errors, adjacent character swaps
    /// **Performance**: O(m*n)
    DamerauLevenshtein,

    /// Jaro-Winkler similarity
    ///
    /// Optimized for comparing short strings, with prefix weighting.
    /// Gives higher scores to strings with common prefixes.
    ///
    /// **Best for**: Person names, titles, short identifiers
    /// **Performance**: O(m*n) but typically faster than Levenshtein
    JaroWinkler(JaroWinklerConfig),

    /// N-gram similarity
    ///
    /// Compares strings based on overlapping character n-grams.
    /// More robust to word reordering and long strings.
    ///
    /// **Best for**: Long text, descriptions, documents
    /// **Performance**: O(m + n) for n-gram generation, then set comparison
    Ngram {
        /// Size of n-grams (typically 2 or 3)
        ngram_size: usize,
    },

    /// Jaccard similarity
    ///
    /// Set-based similarity: |A ∩ B| / |A ∪ B|
    /// Compares tokenized sets (e.g., tags, categories).
    ///
    /// **Best for**: Tags, categories, multi-value fields
    /// **Performance**: O(m + n) for tokenization, then set operations
    Jaccard,

    /// Cosine similarity
    ///
    /// Vector space similarity using term frequency.
    /// Treats strings as bags of words.
    ///
    /// **Best for**: Document similarity, TF-IDF style matching
    /// **Performance**: O(m + n) for tokenization, then vector operations
    Cosine,

    /// Exact match only (no fuzzy matching)
    ///
    /// String equality comparison with no tolerance for differences.
    ///
    /// **Best for**: IDs, exact codes, case-sensitive fields
    /// **Performance**: O(min(m, n))
    ExactMatch,
}

/// Configuration for Jaro-Winkler algorithm
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JaroWinklerConfig {
    /// Prefix weight (0.0 - 0.25, default 0.1)
    ///
    /// How much to boost scores for strings with common prefixes.
    /// Higher values give more weight to matching prefixes.
    pub prefix_weight: f64,

    /// Maximum prefix length to consider (default 4)
    ///
    /// Only the first N matching characters are considered for prefix boost.
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

impl Algorithm {
    /// Get the algorithm name as a string
    pub fn name(&self) -> &'static str {
        match self {
            Algorithm::Levenshtein => "levenshtein",
            Algorithm::DamerauLevenshtein => "damerau_levenshtein",
            Algorithm::JaroWinkler(_) => "jaro_winkler",
            Algorithm::Ngram { .. } => "ngram",
            Algorithm::Jaccard => "jaccard",
            Algorithm::Cosine => "cosine",
            Algorithm::ExactMatch => "exact",
        }
    }

    /// Check if algorithm is suitable for the given field type
    ///
    /// Some algorithms are better suited for certain field types.
    /// This method validates algorithm/field type compatibility.
    pub fn is_suitable_for(&self, field_type: &FieldType) -> bool {
        match (self, field_type) {
            // Short text: character-based algorithms work well
            (Algorithm::Levenshtein, FieldType::ShortText { .. }) => true,
            (Algorithm::DamerauLevenshtein, FieldType::ShortText { .. }) => true,
            (Algorithm::JaroWinkler(_), FieldType::ShortText { .. }) => true,
            (Algorithm::ExactMatch, FieldType::ShortText { .. }) => true,

            // Long text: n-gram and token-based algorithms are better
            (Algorithm::Ngram { .. }, FieldType::LongText { .. }) => true,
            (Algorithm::Cosine, FieldType::LongText { .. }) => true,

            // Token sets: set-based algorithms
            (Algorithm::Jaccard, FieldType::TokenSet { .. }) => true,

            // Everything else is suboptimal but allowed
            _ => true,
        }
    }
}

impl FieldType {
    /// Get the default algorithm for this field type
    pub fn default_algorithm(&self) -> &Algorithm {
        match self {
            FieldType::ShortText { default_algorithm, .. } => default_algorithm,
            FieldType::LongText { default_algorithm, .. } => default_algorithm,
            FieldType::TokenSet { default_algorithm, .. } => default_algorithm,
        }
    }

    /// Get a recommended algorithm for this field type if none is specified
    pub fn recommended_algorithm() -> Self {
        FieldType::ShortText {
            max_length: 100,
            default_algorithm: Algorithm::JaroWinkler(JaroWinklerConfig::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_names() {
        assert_eq!(Algorithm::Levenshtein.name(), "levenshtein");
        assert_eq!(Algorithm::DamerauLevenshtein.name(), "damerau_levenshtein");
        assert_eq!(
            Algorithm::JaroWinkler(JaroWinklerConfig::default()).name(),
            "jaro_winkler"
        );
        assert_eq!(Algorithm::Ngram { ngram_size: 2 }.name(), "ngram");
        assert_eq!(Algorithm::Jaccard.name(), "jaccard");
        assert_eq!(Algorithm::Cosine.name(), "cosine");
        assert_eq!(Algorithm::ExactMatch.name(), "exact");
    }

    #[test]
    fn test_algorithm_suitability() {
        let short_text = FieldType::ShortText {
            max_length: 100,
            default_algorithm: Algorithm::JaroWinkler(JaroWinklerConfig::default()),
        };
        let long_text = FieldType::LongText {
            default_algorithm: Algorithm::Ngram { ngram_size: 2 },
            chunk_size: None,
        };
        let token_set = FieldType::TokenSet {
            separator: ",".to_string(),
            default_algorithm: Algorithm::Jaccard,
        };

        // Short text - character-based algorithms
        assert!(Algorithm::Levenshtein.is_suitable_for(&short_text));
        assert!(Algorithm::JaroWinkler(JaroWinklerConfig::default()).is_suitable_for(&short_text));

        // Long text - n-gram and cosine
        assert!(Algorithm::Ngram { ngram_size: 2 }.is_suitable_for(&long_text));
        assert!(Algorithm::Cosine.is_suitable_for(&long_text));

        // Token sets - Jaccard
        assert!(Algorithm::Jaccard.is_suitable_for(&token_set));
    }

    #[test]
    fn test_jaro_winkler_config_default() {
        let config = JaroWinklerConfig::default();
        assert_eq!(config.prefix_weight, 0.1);
        assert_eq!(config.max_prefix_length, 4);
    }

    #[test]
    fn test_field_type_default_algorithm() {
        let short_text = FieldType::ShortText {
            max_length: 100,
            default_algorithm: Algorithm::Levenshtein,
        };
        assert_eq!(short_text.default_algorithm(), &Algorithm::Levenshtein);
    }
}
