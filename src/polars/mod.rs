//! Polars expression plugin for zero-copy fuzzy matching
//!
//! This module provides native Polars expressions that operate directly
//! on Arrow arrays without crossing the Python/Rust boundary per row.
//!
//! # Features
//!
//! - `fuzzy_similarity`: Compute similarity between two string columns
//! - `fuzzy_is_match`: Check if similarity exceeds threshold
//! - `fuzzy_best_match`: Find best match from a list of candidates
//!
//! # Example
//!
//! ```python
//! import polars as pl
//! import fuzzyrust as fr
//!
//! df = df.with_columns(
//!     score=fr.fuzzy.similarity(pl.col("a"), pl.col("b"))
//! )
//! ```

pub mod expressions;

pub use expressions::*;
