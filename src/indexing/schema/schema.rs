//! Schema definition and builder for multi-field fuzzy matching
//!
//! A Schema defines the structure of records in a multi-field index.
//! It specifies:
//! - Which fields exist and their types
//! - How each field should be matched (algorithm, weight)
//! - Validation rules (required fields, constraints)
//! - Scoring strategy for combining field matches

use super::field::Field;
use super::scoring::ScoringStrategy;
use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during schema operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SchemaError {
    /// Schema has no fields defined
    #[error("Schema must have at least one field")]
    EmptySchema,

    /// Field name appears multiple times in schema
    #[error("Duplicate field name: {0}")]
    DuplicateField(String),

    /// Field validation failed
    #[error("Field '{0}' validation failed: {1}")]
    InvalidField(String, String),

    /// Record is missing a required field
    #[error("Missing required field: {0}")]
    MissingRequiredField(String),

    /// Storage operation failed
    #[error("Storage error: {0}")]
    StorageError(String),

    /// Record has a field not in schema
    #[error("Unknown field: {0}")]
    UnknownField(String),

    /// Field value doesn't match expected type
    #[error("Field '{0}' type mismatch: {1}")]
    TypeMismatch(String, String),
}

/// A complete schema defining the structure for multi-field matching
///
/// Schema is immutable once built. Use `SchemaBuilder` to construct.
///
/// # Example
///
/// ```rust
/// use fuzzyrust::indexing::schema::*;
/// use fuzzyrust::algorithms::normalize::NormalizationMode;
///
/// let schema = Schema::builder()
///     .add_field(Field::new(
///         "name",
///         FieldType::ShortText {
///             max_length: 100,
///             default_algorithm: Algorithm::JaroWinkler(Default::default()),
///         },
///     )
///     .with_weight(10.0)
///     .required()
///     .with_normalization(NormalizationMode::Lowercase))
///     .add_field(Field::new(
///         "tags",
///         FieldType::TokenSet {
///             separator: ",".to_string(),
///             default_algorithm: Algorithm::Jaccard,
///         },
///     )
///     .with_weight(7.0))
///     .with_scoring_strategy(ScoringStrategy::WeightedAverage)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    /// Fields in the schema (ordered)
    fields: Vec<Field>,

    /// Field name -> index mapping for O(1) lookup
    field_map: AHashMap<String, usize>,

    /// Scoring strategy for combining field scores
    scoring_strategy: ScoringStrategy,
}

impl Schema {
    /// Create a new schema builder
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder::new()
    }

    /// Get all fields in the schema
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    /// Get a field by name
    pub fn get_field(&self, name: &str) -> Option<&Field> {
        self.field_map.get(name).map(|&idx| &self.fields[idx])
    }

    /// Check if the schema has a field with the given name
    pub fn has_field(&self, name: &str) -> bool {
        self.field_map.contains_key(name)
    }

    /// Get a field by index
    pub fn get_field_by_index(&self, index: usize) -> Option<&Field> {
        self.fields.get(index)
    }

    /// Get the number of fields
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Get the scoring strategy
    pub fn scoring_strategy(&self) -> &ScoringStrategy {
        &self.scoring_strategy
    }

    /// Get all required field names
    pub fn required_fields(&self) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|f| f.required)
            .map(|f| f.name.as_str())
            .collect()
    }

    /// Get all optional field names
    pub fn optional_fields(&self) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|f| !f.required)
            .map(|f| f.name.as_str())
            .collect()
    }

    /// Get fields that contribute to scoring (weight > 0)
    pub fn scoring_fields(&self) -> Vec<&Field> {
        self.fields
            .iter()
            .filter(|f| f.contributes_to_score())
            .collect()
    }

    /// Validate a record against this schema
    ///
    /// Checks:
    /// - All required fields are present
    /// - No unknown fields (optional, can be relaxed)
    /// - Field values match expected types (basic validation)
    pub fn validate_record(&self, record_fields: &AHashMap<String, String>) -> Result<(), SchemaError> {
        // Check required fields
        for field in &self.fields {
            if field.required && !record_fields.contains_key(&field.name) {
                return Err(SchemaError::MissingRequiredField(field.name.clone()));
            }
        }

        // Check for unknown fields (optional - could be a warning instead)
        for field_name in record_fields.keys() {
            if !self.field_map.contains_key(field_name) {
                return Err(SchemaError::UnknownField(field_name.clone()));
            }
        }

        // Type validation would go here (checking if values match field types)
        // For now, we accept all strings since that's what we store

        Ok(())
    }

    /// Validate the schema structure itself
    fn validate(&self) -> Result<(), SchemaError> {
        if self.fields.is_empty() {
            return Err(SchemaError::EmptySchema);
        }

        // Validate each field
        for field in &self.fields {
            field
                .validate()
                .map_err(|e| SchemaError::InvalidField(field.name.clone(), e))?;
        }

        // Check for duplicate field names (should be caught by builder, but double-check)
        let mut seen = AHashMap::new();
        for (idx, field) in self.fields.iter().enumerate() {
            if seen.contains_key(&field.name) {
                return Err(SchemaError::DuplicateField(field.name.clone()));
            }
            seen.insert(&field.name, idx);
        }

        Ok(())
    }
}

/// Builder for constructing a Schema
///
/// Provides a fluent API for defining fields and configuration.
///
/// # Example
///
/// ```rust
/// use fuzzyrust::indexing::schema::*;
///
/// let schema = Schema::builder()
///     .add_field(Field::new("name", FieldType::ShortText {
///         max_length: 100,
///         default_algorithm: Algorithm::JaroWinkler(Default::default()),
///     }).with_weight(10.0).required())
///     .add_field(Field::new("email", FieldType::ShortText {
///         max_length: 100,
///         default_algorithm: Algorithm::Levenshtein,
///     }).with_weight(8.0))
///     .with_scoring_strategy(ScoringStrategy::WeightedAverage)
///     .build()
///     .unwrap();
/// ```
pub struct SchemaBuilder {
    fields: Vec<Field>,
    scoring_strategy: Option<ScoringStrategy>,
}

impl SchemaBuilder {
    /// Create a new schema builder
    pub fn new() -> Self {
        Self {
            fields: Vec::new(),
            scoring_strategy: None,
        }
    }

    /// Add a field to the schema
    pub fn add_field(mut self, field: Field) -> Self {
        self.fields.push(field);
        self
    }

    /// Set the scoring strategy
    pub fn with_scoring_strategy(mut self, strategy: ScoringStrategy) -> Self {
        self.scoring_strategy = Some(strategy);
        self
    }

    /// Build the schema
    ///
    /// Validates the schema and makes it immutable.
    pub fn build(self) -> Result<Schema, SchemaError> {
        // Build field_map for O(1) lookups
        let mut field_map = AHashMap::new();
        for (idx, field) in self.fields.iter().enumerate() {
            if field_map.contains_key(&field.name) {
                return Err(SchemaError::DuplicateField(field.name.clone()));
            }
            field_map.insert(field.name.clone(), idx);
        }

        let schema = Schema {
            fields: self.fields,
            field_map,
            scoring_strategy: self.scoring_strategy.unwrap_or(ScoringStrategy::WeightedAverage),
        };

        // Validate the complete schema
        schema.validate()?;

        Ok(schema)
    }
}

impl Default for SchemaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexing::schema::types::{Algorithm, FieldType};

    #[test]
    fn test_schema_builder_basic() {
        let schema = Schema::builder()
            .add_field(
                Field::new(
                    "name",
                    FieldType::ShortText {
                        max_length: 100,
                        default_algorithm: Algorithm::JaroWinkler(Default::default()),
                    },
                )
                .with_weight(10.0)
                .required(),
            )
            .add_field(
                Field::new(
                    "email",
                    FieldType::ShortText {
                        max_length: 100,
                        default_algorithm: Algorithm::Levenshtein,
                    },
                )
                .with_weight(7.0),
            )
            .build()
            .unwrap();

        assert_eq!(schema.field_count(), 2);
        assert!(schema.get_field("name").is_some());
        assert!(schema.get_field("email").is_some());
        assert!(schema.get_field("unknown").is_none());
    }

    #[test]
    fn test_empty_schema() {
        let result = Schema::builder().build();
        assert!(matches!(result, Err(SchemaError::EmptySchema)));
    }

    #[test]
    fn test_duplicate_field() {
        let result = Schema::builder()
            .add_field(Field::new(
                "name",
                FieldType::ShortText {
                    max_length: 100,
                    default_algorithm: Algorithm::Levenshtein,
                },
            ))
            .add_field(Field::new(
                "name", // Duplicate!
                FieldType::ShortText {
                    max_length: 100,
                    default_algorithm: Algorithm::Levenshtein,
                },
            ))
            .build();

        assert!(matches!(result, Err(SchemaError::DuplicateField(_))));
    }

    #[test]
    fn test_required_fields() {
        let schema = Schema::builder()
            .add_field(
                Field::new(
                    "name",
                    FieldType::ShortText {
                        max_length: 100,
                        default_algorithm: Algorithm::Levenshtein,
                    },
                )
                .required(),
            )
            .add_field(
                Field::new(
                    "email",
                    FieldType::ShortText {
                        max_length: 100,
                        default_algorithm: Algorithm::Levenshtein,
                    },
                )
                .optional(),
            )
            .build()
            .unwrap();

        let required = schema.required_fields();
        assert_eq!(required.len(), 1);
        assert!(required.contains(&"name"));

        let optional = schema.optional_fields();
        assert_eq!(optional.len(), 1);
        assert!(optional.contains(&"email"));
    }

    #[test]
    fn test_validate_record() {
        let schema = Schema::builder()
            .add_field(
                Field::new(
                    "name",
                    FieldType::ShortText {
                        max_length: 100,
                        default_algorithm: Algorithm::Levenshtein,
                    },
                )
                .required(),
            )
            .add_field(
                Field::new(
                    "email",
                    FieldType::ShortText {
                        max_length: 100,
                        default_algorithm: Algorithm::Levenshtein,
                    },
                )
                .optional(),
            )
            .build()
            .unwrap();

        // Valid record with all required fields
        let mut record = AHashMap::new();
        record.insert("name".to_string(), "John Doe".to_string());
        assert!(schema.validate_record(&record).is_ok());

        // Valid record with optional field
        record.insert("email".to_string(), "john@example.com".to_string());
        assert!(schema.validate_record(&record).is_ok());

        // Invalid: missing required field
        let mut record = AHashMap::new();
        record.insert("email".to_string(), "john@example.com".to_string());
        assert!(matches!(
            schema.validate_record(&record),
            Err(SchemaError::MissingRequiredField(_))
        ));

        // Invalid: unknown field
        let mut record = AHashMap::new();
        record.insert("name".to_string(), "John Doe".to_string());
        record.insert("unknown".to_string(), "value".to_string());
        assert!(matches!(
            schema.validate_record(&record),
            Err(SchemaError::UnknownField(_))
        ));
    }

    #[test]
    fn test_scoring_fields() {
        let schema = Schema::builder()
            .add_field(
                Field::new(
                    "name",
                    FieldType::ShortText {
                        max_length: 100,
                        default_algorithm: Algorithm::Levenshtein,
                    },
                )
                .with_weight(10.0),
            )
            .add_field(
                Field::new(
                    "meta",
                    FieldType::ShortText {
                        max_length: 100,
                        default_algorithm: Algorithm::Levenshtein,
                    },
                )
                .with_weight(0.0), // Not used for scoring
            )
            .build()
            .unwrap();

        let scoring_fields = schema.scoring_fields();
        assert_eq!(scoring_fields.len(), 1);
        assert_eq!(scoring_fields[0].name, "name");
    }

    #[test]
    fn test_get_field_by_index() {
        let schema = Schema::builder()
            .add_field(Field::new(
                "field1",
                FieldType::ShortText {
                    max_length: 100,
                    default_algorithm: Algorithm::Levenshtein,
                },
            ))
            .add_field(Field::new(
                "field2",
                FieldType::ShortText {
                    max_length: 100,
                    default_algorithm: Algorithm::Levenshtein,
                },
            ))
            .build()
            .unwrap();

        assert_eq!(schema.get_field_by_index(0).unwrap().name, "field1");
        assert_eq!(schema.get_field_by_index(1).unwrap().name, "field2");
        assert!(schema.get_field_by_index(2).is_none());
    }
}
