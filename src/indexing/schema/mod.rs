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
pub mod types;
pub mod field;
pub mod schema;
pub mod storage;
pub mod field_indices;
pub mod scoring;
pub mod index;

// Re-export commonly used types for convenience
pub use types::{FieldType, Algorithm};
pub use field::Field;
pub use schema::{Schema, SchemaBuilder, SchemaError};
pub use storage::{Record, FieldColumn, OptimizedStorage, ColumnError};
pub use field_indices::{FieldIndex, ShortTextIndex, LongTextIndex, TokenSetIndex};
pub use scoring::{ScoringStrategy, WeightedAverage, MinMaxScaling};
pub use index::{SchemaIndex, SchemaSearchResult, SearchOptions};
