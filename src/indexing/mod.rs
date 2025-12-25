//! Indexing structures for efficient fuzzy search
//!
//! - BK-tree: Fast fuzzy search using metric space properties
//! - N-gram index: Quick candidate filtering
//! - Schema: Multi-field fuzzy matching with type-safe schema

pub mod bktree;
pub mod ngram_index;
pub mod schema;

pub use bktree::*;
pub use ngram_index::*;
