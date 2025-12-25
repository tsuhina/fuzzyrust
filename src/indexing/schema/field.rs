///! Field definition for schema-based multi-field matching
///!
///! A Field represents a single named column in a schema with configuration for:
///! - Data type (FieldType)
///! - Fuzzy matching algorithm
///! - Weight for scoring (0-10 scale, higher = more important)
///! - Required/optional flag
///! - Normalization mode

use super::types::{Algorithm, FieldType};
use crate::algorithms::normalize::NormalizationMode;
use serde::{Deserialize, Serialize};

/// A field in a schema
///
/// Each field defines how a particular attribute should be indexed and matched.
/// Fields have a name, type, algorithm, weight, and optional normalization.
///
/// # Example
///
/// ```rust
/// use fuzzyrust::indexing::schema::*;
/// use fuzzyrust::algorithms::normalize::NormalizationMode;
///
/// let field = Field {
///     name: "name".to_string(),
///     field_type: FieldType::ShortText {
///         max_length: 100,
///         default_algorithm: Algorithm::JaroWinkler(Default::default()),
///     },
///     algorithm: Algorithm::JaroWinkler(Default::default()),
///     weight: 10.0,
///     required: true,
///     normalization: Some(NormalizationMode::Lowercase),
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Field {
    /// Field name (must be unique within schema)
    pub name: String,

    /// Field type determines indexing strategy
    pub field_type: FieldType,

    /// Algorithm to use for fuzzy matching this field
    ///
    /// If not specified, uses the field_type's default algorithm.
    /// Should be chosen based on the data characteristics:
    /// - Short text (names): JaroWinkler
    /// - Long text (descriptions): Ngram
    /// - Tags/categories: Jaccard
    pub algorithm: Algorithm,

    /// Field weight for scoring (0-10 scale)
    ///
    /// Higher values mean this field is more important in overall similarity.
    /// Typical values:
    /// - 10: Critical field (e.g., name, title)
    /// - 7-9: Important field (e.g., primary description, key identifier)
    /// - 4-6: Moderate importance (e.g., category, tags)
    /// - 1-3: Low importance (e.g., metadata, auxiliary info)
    /// - 0: Not used for matching (but indexed)
    pub weight: f64,

    /// Whether this field must be present in records
    ///
    /// If true, records missing this field will fail validation.
    /// If false, field is optional and missing values are treated as non-matches.
    pub required: bool,

    /// Optional normalization to apply before matching
    ///
    /// Normalization is applied to both indexed values and queries.
    /// Common modes:
    /// - `None`: No normalization
    /// - `Lowercase`: Case-insensitive matching
    /// - `Strict`: Lowercase + remove punctuation + whitespace
    pub normalization: Option<NormalizationMode>,
}

impl Field {
    /// Create a new field with the given name and type
    ///
    /// Uses default settings:
    /// - Algorithm from field_type default
    /// - Weight: 5.0 (moderate)
    /// - Required: false
    /// - Normalization: None
    pub fn new(name: impl Into<String>, field_type: FieldType) -> Self {
        let algorithm = field_type.default_algorithm().clone();
        Self {
            name: name.into(),
            field_type,
            algorithm,
            weight: 5.0,
            required: false,
            normalization: None,
        }
    }

    /// Builder: Set the fuzzy matching algorithm
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Builder: Set the field weight (0-10 scale)
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight.clamp(0.0, 10.0);
        self
    }

    /// Builder: Mark field as required
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Builder: Mark field as optional
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// Builder: Set normalization mode
    pub fn with_normalization(mut self, normalization: NormalizationMode) -> Self {
        self.normalization = Some(normalization);
        self
    }

    /// Validate field configuration
    ///
    /// Checks:
    /// - Name is not empty
    /// - Weight is in valid range (0-10)
    /// - Algorithm is suitable for field type
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Field name cannot be empty".to_string());
        }

        if self.weight < 0.0 || self.weight > 10.0 {
            return Err(format!(
                "Field weight must be in range 0-10, got {}",
                self.weight
            ));
        }

        if !self.algorithm.is_suitable_for(&self.field_type) {
            // Note: This is a warning, not an error. Suboptimal algorithms are allowed.
            // In production, you might want to log a warning here.
        }

        Ok(())
    }

    /// Get the effective weight for scoring.
    ///
    /// Weight of 0 means the field is indexed but not used for scoring.
    #[inline]
    pub fn effective_weight(&self) -> f64 {
        self.weight
    }

    /// Check if this field contributes to scoring
    pub fn contributes_to_score(&self) -> bool {
        self.weight > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_new() {
        let field = Field::new(
            "name",
            FieldType::ShortText {
                max_length: 100,
                default_algorithm: Algorithm::JaroWinkler(Default::default()),
            },
        );

        assert_eq!(field.name, "name");
        assert_eq!(field.weight, 5.0);
        assert!(!field.required);
        assert!(field.normalization.is_none());
    }

    #[test]
    fn test_field_builders() {
        let field = Field::new(
            "email",
            FieldType::ShortText {
                max_length: 100,
                default_algorithm: Algorithm::Levenshtein,
            },
        )
        .with_weight(8.0)
        .required()
        .with_normalization(NormalizationMode::Lowercase);

        assert_eq!(field.name, "email");
        assert_eq!(field.weight, 8.0);
        assert!(field.required);
        assert_eq!(field.normalization, Some(NormalizationMode::Lowercase));
    }

    #[test]
    fn test_field_validate() {
        let valid_field = Field::new(
            "name",
            FieldType::ShortText {
                max_length: 100,
                default_algorithm: Algorithm::JaroWinkler(Default::default()),
            },
        );
        assert!(valid_field.validate().is_ok());

        // Empty name
        let mut invalid_field = valid_field.clone();
        invalid_field.name = "".to_string();
        assert!(invalid_field.validate().is_err());

        // Invalid weight (negative)
        let mut invalid_field = valid_field.clone();
        invalid_field.weight = -1.0;
        assert!(invalid_field.validate().is_err());

        // Invalid weight (too high)
        let mut invalid_field = valid_field.clone();
        invalid_field.weight = 11.0;
        assert!(invalid_field.validate().is_err());
    }

    #[test]
    fn test_weight_clamping() {
        let field = Field::new(
            "test",
            FieldType::ShortText {
                max_length: 100,
                default_algorithm: Algorithm::Levenshtein,
            },
        )
        .with_weight(15.0); // Should be clamped to 10.0

        assert_eq!(field.weight, 10.0);

        let field = Field::new(
            "test",
            FieldType::ShortText {
                max_length: 100,
                default_algorithm: Algorithm::Levenshtein,
            },
        )
        .with_weight(-5.0); // Should be clamped to 0.0

        assert_eq!(field.weight, 0.0);
    }

    #[test]
    fn test_effective_weight() {
        let field = Field::new(
            "test",
            FieldType::ShortText {
                max_length: 100,
                default_algorithm: Algorithm::Levenshtein,
            },
        )
        .with_weight(7.0);
        assert_eq!(field.effective_weight(), 7.0);

        let field = field.with_weight(0.0);
        assert_eq!(field.effective_weight(), 0.0);
    }

    #[test]
    fn test_contributes_to_score() {
        let field = Field::new(
            "test",
            FieldType::ShortText {
                max_length: 100,
                default_algorithm: Algorithm::Levenshtein,
            },
        )
        .with_weight(7.0);
        assert!(field.contributes_to_score());

        let field = field.with_weight(0.0);
        assert!(!field.contributes_to_score());
    }
}
