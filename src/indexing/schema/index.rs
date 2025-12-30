//! Main SchemaIndex implementation for multi-field fuzzy matching
//!
//! This is the primary entry point for using schema-based multi-field matching.
//! It coordinates field indices, storage, and scoring strategies.

use super::field_indices::{create_field_index, FieldIndex};
use super::schema::{Schema, SchemaError};
use super::scoring::FieldScore;
use super::storage::{OptimizedStorage, Record};
use crate::algorithms::normalize;
use ahash::AHashMap;
use rayon::prelude::*;
use std::sync::Arc;

/// Search options for SchemaIndex queries
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Minimum overall similarity score (0.0 - 1.0)
    pub min_score: f64,

    /// Maximum number of results to return
    pub limit: Option<usize>,

    /// Minimum per-field similarity (applied before scoring)
    /// Fields with scores below this threshold are ignored
    pub min_field_score: f64,

    /// Query-time field weight boosts (multipliers applied to schema weights)
    /// Keys are field names, values are multipliers (e.g., 2.0 doubles the weight)
    pub field_boosts: Option<AHashMap<String, f64>>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            min_score: 0.0,
            limit: None,
            min_field_score: 0.0,
            field_boosts: None,
        }
    }
}

impl SearchOptions {
    /// Create options with a minimum score threshold
    pub fn with_min_score(min_score: f64) -> Self {
        Self {
            min_score,
            ..Default::default()
        }
    }

    /// Set the result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set minimum per-field score
    pub fn with_min_field_score(mut self, min_field_score: f64) -> Self {
        self.min_field_score = min_field_score;
        self
    }

    /// Set query-time field boosts
    ///
    /// Field boosts are multipliers applied to the schema weights at query time.
    /// For example, a boost of 2.0 doubles the field's weight.
    pub fn with_field_boosts(mut self, boosts: AHashMap<String, f64>) -> Self {
        self.field_boosts = Some(boosts);
        self
    }
}

/// Search result from SchemaIndex
#[derive(Debug, Clone)]
pub struct SchemaSearchResult {
    /// Record ID
    pub id: usize,

    /// Overall similarity score (0.0 - 1.0)
    pub score: f64,

    /// Per-field scores
    pub field_scores: AHashMap<String, f64>,

    /// The matched record
    pub record: Record,
}

/// Multi-field fuzzy matching index with schema
///
/// SchemaIndex provides type-safe, schema-driven multi-field fuzzy matching.
/// It validates records against a schema, uses optimized per-field indices,
/// and combines field scores using configurable strategies.
///
/// # Example
///
/// ```rust
/// use fuzzyrust::indexing::schema::*;
/// use fuzzyrust::algorithms::normalize::NormalizationMode;
///
/// // Define schema
/// let schema = Schema::builder()
///     .add_field(
///         Field::new("name", FieldType::ShortText {
///             max_length: 100,
///             default_algorithm: Algorithm::JaroWinkler(Default::default()),
///         })
///         .with_weight(10.0)
///         .required()
///         .with_normalization(NormalizationMode::Lowercase)
///     )
///     .add_field(
///         Field::new("tags", FieldType::TokenSet {
///             separator: ",".to_string(),
///             default_algorithm: Algorithm::Jaccard,
///         })
///         .with_weight(7.0)
///     )
///     .build()
///     .unwrap();
///
/// // Create index
/// let mut index = SchemaIndex::new(schema);
///
/// // Add records
/// let mut record = Record::new();
/// record.set_field("name", "MacBook Pro");
/// record.set_field("tags", "laptop,apple,computing");
/// index.add_record(record, None).unwrap();
///
/// // Search
/// let mut query = Record::new();
/// query.set_field("name", "Macbook");
/// let results = index.search(&query, SearchOptions::default()).unwrap();
/// ```
pub struct SchemaIndex {
    /// Schema defining the structure
    schema: Arc<Schema>,

    /// Column-oriented storage
    storage: OptimizedStorage,

    /// Per-field indices
    field_indices: Vec<Box<dyn FieldIndex>>,

    /// Number of records indexed
    record_count: usize,
}

impl SchemaIndex {
    /// Create a new schema index
    pub fn new(schema: Schema) -> Self {
        let schema = Arc::new(schema);

        // Create storage
        let mut storage = OptimizedStorage::new();
        for field in schema.fields() {
            storage.add_column(&field.name, field.field_type.clone());
        }

        // Create field indices
        let field_indices: Vec<Box<dyn FieldIndex>> = schema
            .fields()
            .iter()
            .map(|field| create_field_index(&field.field_type, &field.algorithm))
            .collect();

        Self {
            schema,
            storage,
            field_indices,
            record_count: 0,
        }
    }

    /// Add a record to the index
    ///
    /// The record is validated against the schema before being added.
    /// Returns the record ID on success.
    pub fn add_record(&mut self, record: Record, data: Option<u64>) -> Result<usize, SchemaError> {
        // Validate record
        self.schema.validate_record(&record.fields)?;

        // Prepare record with data
        let mut record_with_data = record;
        record_with_data.data = data;

        // Add to storage
        let record_id = self.storage.add_record(&record_with_data).map_err(|e| {
            SchemaError::StorageError(format!("Failed to add record to storage: {}", e))
        })?;

        // Add to each field index
        for (field_idx, field) in self.schema.fields().iter().enumerate() {
            if let Some(value) = record_with_data.get_field(&field.name) {
                // Apply normalization if specified
                let normalized_value = if let Some(norm_mode) = field.normalization {
                    normalize::normalize_string(value, norm_mode)
                } else {
                    value.clone()
                };

                // Add to field index
                self.field_indices[field_idx].add(record_id, &normalized_value, data);
            } else if !field.required {
                // Optional field not present - add empty entry to keep indices aligned
                // (Some indices may handle this differently)
            }
        }

        self.record_count += 1;
        Ok(record_id)
    }

    /// Search for matching records
    ///
    /// Queries each field index, combines scores using the schema's scoring strategy,
    /// and returns results sorted by descending overall score.
    ///
    /// # Query Field Behavior
    ///
    /// - All query fields must be defined in the schema (unknown fields return `SchemaError::UnknownField`)
    /// - Empty queries (no fields) return an empty result set
    /// - Empty string values in query fields are processed normally:
    ///   - TokenSet fields: return no matches (empty token set)
    ///   - Text fields: may return matches based on the similarity algorithm
    /// - Query fields not present in a record are ignored for that record's score
    pub fn search(
        &self,
        query: &Record,
        options: SearchOptions,
    ) -> Result<Vec<SchemaSearchResult>, SchemaError> {
        // Validate query fields against schema (Bug 10 fix)
        for field_name in query.field_names() {
            if !self.schema.has_field(field_name) {
                return Err(SchemaError::UnknownField(field_name.clone()));
            }
        }

        // Collect field scores for all candidates
        let mut candidate_scores: AHashMap<usize, Vec<FieldScore>> = AHashMap::new();

        // Search each field that's present in the query
        for (field_idx, field) in self.schema.fields().iter().enumerate() {
            if let Some(query_value) = query.get_field(&field.name) {
                // Apply normalization if specified
                let normalized_query = if let Some(norm_mode) = field.normalization {
                    normalize::normalize_string(query_value, norm_mode)
                } else {
                    query_value.clone()
                };

                // Search field index
                let field_matches = self.field_indices[field_idx].search(
                    &normalized_query,
                    options.min_field_score,
                    None, // Don't limit per-field results yet
                );

                // Collect scores for each candidate
                for field_match in field_matches {
                    // Apply query-time field boost if specified
                    let effective_weight = if let Some(ref boosts) = options.field_boosts {
                        if let Some(&boost) = boosts.get(&field.name) {
                            field.weight * boost
                        } else {
                            field.weight
                        }
                    } else {
                        field.weight
                    };

                    candidate_scores
                        .entry(field_match.id)
                        .or_default()
                        .push(FieldScore::new(
                            field.name.clone(),
                            field_match.score,
                            effective_weight,
                        ));
                }
            }
        }

        // Combine field scores using schema's scoring strategy
        // Fail fast on first storage error to avoid returning partial results
        let mut results: Vec<SchemaSearchResult> = Vec::new();

        for (record_id, field_scores) in candidate_scores {
            let overall_score = self.schema.scoring_strategy().combine(&field_scores);

            if overall_score >= options.min_score {
                // Build field scores map
                let field_scores_map: AHashMap<String, f64> = field_scores
                    .into_iter()
                    .map(|fs| (fs.field_name, fs.score))
                    .collect();

                // Retrieve record from storage - fail fast on error
                match self.storage.get_record(record_id) {
                    Ok(Some(record)) => {
                        results.push(SchemaSearchResult {
                            id: record_id,
                            score: overall_score,
                            field_scores: field_scores_map,
                            record,
                        });
                    }
                    Ok(None) => {
                        // Record doesn't exist (shouldn't happen normally - indicates index corruption)
                        // Continue silently as this isn't a fatal error
                    }
                    Err(e) => {
                        // Fail fast on storage error - don't return partial results
                        return Err(SchemaError::StorageError(format!(
                            "Failed to get record {}: {}",
                            record_id, e
                        )));
                    }
                }
            }
        }

        // Filter out NaN scores and warn in release mode
        // NaN scores indicate a bug in the scoring algorithm and should never occur
        #[cfg(debug_assertions)]
        let original_count = results.len();

        results.retain(|r| {
            if r.score.is_nan() {
                eprintln!(
                    "[fuzzyrust warning] NaN score detected for record id: {}. \
                     This indicates a bug in the scoring algorithm. The result was filtered out.",
                    r.id
                );
                false
            } else {
                true
            }
        });

        // Debug assertion for development - panics if NaN was detected
        #[cfg(debug_assertions)]
        if results.len() < original_count {
            debug_assert!(
                false,
                "NaN scores were detected and filtered out. {} records had NaN scores.",
                original_count - results.len()
            );
        }

        // Sort by descending score
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply limit
        if let Some(limit) = options.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Get the number of records in the index
    pub fn len(&self) -> usize {
        self.record_count
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.record_count == 0
    }

    /// Get the schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get a record by ID
    ///
    /// Returns `Ok(Some(record))` if found, `Ok(None)` if the ID doesn't exist,
    /// or `Err` if there was a storage error (e.g., data corruption).
    pub fn get_record(&self, id: usize) -> Result<Option<Record>, SchemaError> {
        self.storage
            .get_record(id)
            .map_err(|e| SchemaError::StorageError(format!("Failed to get record {}: {}", id, e)))
    }

    /// Batch search for multiple queries in parallel
    ///
    /// Executes searches for all queries concurrently using Rayon,
    /// returning results in the same order as the input queries.
    ///
    /// # Arguments
    /// * `queries` - Slice of query records to search for
    /// * `options` - Search options (min_score, limit, etc.)
    ///
    /// # Returns
    /// A vector of search results for each query, maintaining input order.
    /// Each inner vector contains matches sorted by descending score.
    ///
    /// # Example
    /// ```rust,ignore
    /// let queries = vec![query1, query2, query3];
    /// let results = index.batch_search(&queries, SearchOptions::with_min_score(0.7))?;
    /// // results[0] contains matches for query1, etc.
    /// ```
    pub fn batch_search(
        &self,
        queries: &[Record],
        options: SearchOptions,
    ) -> Result<Vec<Vec<SchemaSearchResult>>, SchemaError> {
        queries
            .par_iter()
            .map(|query| self.search(query, options.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::normalize::NormalizationMode;
    use crate::indexing::schema::types::{Algorithm, FieldType};
    use crate::indexing::schema::{Field, ScoringStrategy};

    fn create_test_schema() -> Schema {
        Schema::builder()
            .add_field(
                Field::new(
                    "name",
                    FieldType::ShortText {
                        max_length: 100,
                        default_algorithm: Algorithm::JaroWinkler(Default::default()),
                    },
                )
                .with_weight(10.0)
                .required()
                .with_normalization(NormalizationMode::Lowercase),
            )
            .add_field(
                Field::new(
                    "description",
                    FieldType::LongText {
                        default_algorithm: Algorithm::Ngram { ngram_size: 2 },
                        chunk_size: None,
                    },
                )
                .with_weight(5.0)
                .optional(),
            )
            .add_field(
                Field::new(
                    "tags",
                    FieldType::TokenSet {
                        separator: ",".to_string(),
                        default_algorithm: Algorithm::Jaccard,
                    },
                )
                .with_weight(7.0)
                .optional(),
            )
            .with_scoring_strategy(ScoringStrategy::WeightedAverage)
            .build()
            .unwrap()
    }

    #[test]
    fn test_schema_index_create() {
        let schema = create_test_schema();
        let index = SchemaIndex::new(schema);

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_add_record() {
        let schema = create_test_schema();
        let mut index = SchemaIndex::new(schema);

        let mut record = Record::new();
        record.set_field("name", "MacBook Pro");
        record.set_field("description", "High-performance laptop");
        record.set_field("tags", "laptop,apple,computing");

        let id = index.add_record(record, Some(12345)).unwrap();
        assert_eq!(id, 0);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_add_record_validation() {
        let schema = create_test_schema();
        let mut index = SchemaIndex::new(schema);

        // Missing required field "name"
        let mut record = Record::new();
        record.set_field("description", "Some description");

        let result = index.add_record(record, None);
        assert!(matches!(result, Err(SchemaError::MissingRequiredField(_))));
    }

    #[test]
    fn test_search_single_field() {
        let schema = create_test_schema();
        let mut index = SchemaIndex::new(schema);

        // Add records
        let mut record1 = Record::new();
        record1.set_field("name", "MacBook Pro");
        record1.set_field("tags", "laptop,apple");
        index.add_record(record1, Some(100)).unwrap();

        let mut record2 = Record::new();
        record2.set_field("name", "MacBook Air");
        record2.set_field("tags", "laptop,apple");
        index.add_record(record2, Some(101)).unwrap();

        let mut record3 = Record::new();
        record3.set_field("name", "Dell XPS");
        record3.set_field("tags", "laptop,dell");
        index.add_record(record3, Some(102)).unwrap();

        // Search for "Macbook"
        let mut query = Record::new();
        query.set_field("name", "Macbook");

        let results = index
            .search(&query, SearchOptions::with_min_score(0.5))
            .unwrap();

        // Should find MacBook Pro and MacBook Air
        assert!(results.len() >= 2);
        assert!(results
            .iter()
            .any(|r| r.record.get_field("name").unwrap().contains("MacBook")));
    }

    #[test]
    fn test_search_multi_field() {
        let schema = create_test_schema();
        let mut index = SchemaIndex::new(schema);

        // Add records
        let mut record1 = Record::new();
        record1.set_field("name", "Python Programming");
        record1.set_field("tags", "python,programming,tutorial");
        index.add_record(record1, None).unwrap();

        let mut record2 = Record::new();
        record2.set_field("name", "Rust Programming");
        record2.set_field("tags", "rust,programming,systems");
        index.add_record(record2, None).unwrap();

        let mut record3 = Record::new();
        record3.set_field("name", "Python Tutorial");
        record3.set_field("tags", "python,beginner,tutorial");
        index.add_record(record3, None).unwrap();

        // Search for both name and tags
        let mut query = Record::new();
        query.set_field("name", "Python");
        query.set_field("tags", "tutorial");

        let results = index
            .search(&query, SearchOptions::with_min_score(0.3))
            .unwrap();

        // Should find results, with Python Tutorial likely scoring highest
        assert!(!results.is_empty());
        // First result should have both fields matching well
        assert!(results[0].field_scores.contains_key("name"));
        assert!(results[0].field_scores.contains_key("tags"));
    }

    #[test]
    fn test_search_with_limit() {
        let schema = create_test_schema();
        let mut index = SchemaIndex::new(schema);

        // Add 5 records
        for i in 0..5 {
            let mut record = Record::new();
            record.set_field("name", format!("Item {}", i));
            index.add_record(record, None).unwrap();
        }

        // Search with limit=3
        let mut query = Record::new();
        query.set_field("name", "Item");

        let results = index
            .search(&query, SearchOptions::default().with_limit(3))
            .unwrap();

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_get_record() {
        let schema = create_test_schema();
        let mut index = SchemaIndex::new(schema);

        let mut record = Record::new();
        record.set_field("name", "Test Item");
        let id = index.add_record(record, None).unwrap();

        let retrieved = index.get_record(id).unwrap().unwrap();
        assert_eq!(retrieved.get_field("name").unwrap(), "Test Item");
    }

    #[test]
    fn test_normalization_in_search() {
        let schema = create_test_schema();
        let mut index = SchemaIndex::new(schema);

        // Add record with uppercase
        let mut record = Record::new();
        record.set_field("name", "UPPERCASE NAME");
        index.add_record(record, None).unwrap();

        // Search with lowercase (normalization should handle it)
        let mut query = Record::new();
        query.set_field("name", "uppercase name");

        let results = index
            .search(&query, SearchOptions::with_min_score(0.9))
            .unwrap();

        // Should find exact match due to lowercase normalization
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 0.99);
    }
}
