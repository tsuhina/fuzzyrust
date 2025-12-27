//! Core string similarity algorithms
//! 
//! Each algorithm is implemented as a standalone function for composability,
//! plus a trait-based interface for extensibility.

pub mod levenshtein;
pub mod damerau;
pub mod jaro;
pub mod hamming;
pub mod ngram;
pub mod phonetic;
pub mod lcs;
pub mod cosine;
pub mod normalize;
pub mod fuzz;

pub use levenshtein::*;
pub use damerau::*;
pub use jaro::*;
pub use hamming::*;
pub use ngram::*;
pub use phonetic::*;
pub use lcs::*;
pub use cosine::*;

/// Trait for all similarity metrics.
/// Returns a value between 0.0 (completely different) and 1.0 (identical).
pub trait Similarity: Send + Sync {
    #[must_use]
    fn similarity(&self, a: &str, b: &str) -> f64;

    /// Convenience method for distance (1.0 - similarity)
    #[must_use]
    fn distance(&self, a: &str, b: &str) -> f64 {
        1.0 - self.similarity(a, b)
    }

    /// Name of the algorithm for debugging/logging
    #[must_use]
    fn name(&self) -> &'static str;
}

/// Trait for edit distance algorithms that return integer distances
pub trait EditDistance: Send + Sync {
    #[must_use]
    fn distance(&self, a: &str, b: &str) -> usize;

    /// Convert to normalized similarity score (0.0 to 1.0)
    #[must_use]
    fn similarity(&self, a: &str, b: &str) -> f64 {
        let dist = self.distance(a, b);
        let max_len = a.chars().count().max(b.chars().count());
        if max_len == 0 {
            1.0
        } else {
            1.0 - (dist as f64 / max_len as f64)
        }
    }

    #[must_use]
    fn name(&self) -> &'static str;
}

/// Blanket implementation: any EditDistance is also a Similarity
impl<T: EditDistance> Similarity for T {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        EditDistance::similarity(self, a, b)
    }

    fn name(&self) -> &'static str {
        EditDistance::name(self)
    }
}

/// Trait for edit distance algorithms that may fail.
///
/// Some algorithms have constraints (e.g., Hamming requires equal-length strings).
/// This trait returns `Option<usize>` to properly handle failure cases.
///
/// Note: Types implementing this trait should also manually implement `Similarity`
/// to integrate with the rest of the ecosystem.
pub trait FallibleEditDistance: Send + Sync {
    /// Compute distance, returning None if operation is invalid
    #[must_use]
    fn distance(&self, a: &str, b: &str) -> Option<usize>;

    /// Convert to normalized similarity score (0.0 to 1.0)
    ///
    /// Returns None if the distance computation fails.
    #[must_use]
    fn similarity(&self, a: &str, b: &str) -> Option<f64> {
        self.distance(a, b).map(|dist| {
            let max_len = a.chars().count().max(b.chars().count());
            if max_len == 0 {
                1.0
            } else {
                1.0 - (dist as f64 / max_len as f64)
            }
        })
    }

    #[must_use]
    fn name(&self) -> &'static str;
}

/// Wrapper that makes any Similarity implementation case-insensitive.
///
/// Lowercases both strings before comparison.
pub struct CaseInsensitive<T>(pub T);

impl<T: Similarity> Similarity for CaseInsensitive<T> {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        self.0.similarity(&a.to_lowercase(), &b.to_lowercase())
    }

    fn name(&self) -> &'static str {
        self.0.name()
    }
}
