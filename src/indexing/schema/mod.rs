// Multi-field fuzzy matching with schema-first design
//
// This module provides a production-grade schema-based multi-field fuzzy matching system.
// Key features:
// - Type-safe field definitions with validation
// - Per-field algorithm and weight configuration
// - Column-oriented storage for optimal cache locality
// - Specialized indices for different field types
// - Flexible scoring strategies (weighted average, min-max, BM25, RRF)
//
// # Architecture
//
// ```text
// User defines Schema → SchemaIndex validates & optimizes → Type-specific indices → Weighted scoring
// ```
//
// # Example
//
// ```rust
// use fuzzyrust::indexing::schema::*;
//
// // Define schema
// let schema = Schema::builder()
//     .add_field(Field {
//         name: "name".into(),
//         field_type: FieldType::ShortText {
//             max_length: 100,
//             default_algorithm: Algorithm::JaroWinkler(Default::default()),
//         },
//         algorithm: Algorithm::JaroWinkler(Default::default()),
//         weight: 10.0,
//         required: true,
//         normalization: Some(NormalizationMode::Lowercase),
//     })
//     .with_scoring_strategy(ScoringStrategy::WeightedAverage)
//     .build()?;
//
// // Create index
// let mut index = SchemaIndex::new(schema);
//
// // Add records (validated against schema)
// let mut record = Record::new();
// record.set_field("name", "MacBook Pro");
// index.add_record(record, Some(12345))?;
//
// // Search
// let mut query = Record::new();
// query.set_field("name", "Macbook");
// let results = index.search(&query, SearchOptions::default())?;
// ```

// Module declarations
pub mod field;
pub mod field_indices;
pub mod index;
pub mod schema;
pub mod scoring;
pub mod storage;
pub mod types;

// Re-export commonly used types for convenience
pub use field::Field;
pub use field_indices::{FieldIndex, LongTextIndex, ShortTextIndex, TokenSetIndex};
pub use index::{SchemaIndex, SchemaSearchResult, SearchOptions};
pub use schema::{Schema, SchemaBuilder, SchemaError};
pub use scoring::{MinMaxScaling, ScoringStrategy, WeightedAverage};
pub use storage::{ColumnError, FieldColumn, OptimizedStorage, Record};
pub use types::{Algorithm, FieldType};
