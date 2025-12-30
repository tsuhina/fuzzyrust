//! Indexing structures for efficient fuzzy search
//!
//! - BK-tree: Fast fuzzy search using metric space properties
//! - N-gram index: Quick candidate filtering
//! - Schema: Multi-field fuzzy matching with type-safe schema
//! - Thread-safe wrappers: Concurrent access to indices
//! - Sharded indices: Scalability for very large datasets (>10M items)

pub mod bktree;
pub mod ngram_index;
pub mod schema;
pub mod threadsafe;
pub mod sharded;

pub use bktree::*;
pub use ngram_index::*;
pub use threadsafe::*;
pub use sharded::*;
