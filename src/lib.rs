//! FuzzyRust - High-performance string similarity library
//!
//! A Rust library with Python bindings for fast fuzzy string matching.
//!
//! # Features
//! - Multiple similarity algorithms (Levenshtein, Jaro-Winkler, etc.)
//! - Efficient indexing structures (BK-tree, N-gram index)
//! - Parallel batch processing
//! - Unicode support

pub mod algorithms;
pub mod dedup;
pub mod indexing;
pub mod metrics;

// Polars plugin module (enabled with polars-plugin feature)
#[cfg(feature = "polars-plugin")]
pub mod polars;

use algorithms::normalize::NormalizationMode;
use pyo3::create_exception;
use pyo3::prelude::*;
use std::borrow::Cow;

// ============================================================================
// Custom Python Exceptions
// ============================================================================
//
// FuzzyRust defines a hierarchy of custom exceptions for better error handling:
//
// FuzzyRustError (base)
//   ├── ValidationError - Invalid parameter values (similarity out of range, etc.)
//   ├── AlgorithmError - Unknown or unsupported algorithm
//   ├── IndexError - Index operations (add, search, etc.)
//   └── SchemaError - Schema definition and validation errors
//
// These exceptions provide more specific error messages and allow users to
// catch and handle different error types appropriately.

create_exception!(fuzzyrust, FuzzyRustError, pyo3::exceptions::PyException);
create_exception!(fuzzyrust, ValidationError, FuzzyRustError);
create_exception!(fuzzyrust, AlgorithmError, FuzzyRustError);
create_exception!(fuzzyrust, IndexError, FuzzyRustError);
create_exception!(fuzzyrust, SchemaError, FuzzyRustError);

// Re-exports for Rust users (explicit to avoid conflicts with pyfunction wrappers)
pub use algorithms::{
    cosine as cosine_mod, damerau as damerau_mod, hamming as hamming_mod, jaro as jaro_mod,
    lcs as lcs_mod, levenshtein as levenshtein_mod, ngram as ngram_mod, phonetic as phonetic_mod,
    EditDistance, Similarity,
};
pub use indexing::{bktree, ngram_index};

// ============================================================================
// Constants
// ============================================================================

/// Minimum input size for parallel processing.
///
/// For inputs smaller than this threshold, sequential processing is faster
/// due to the overhead of thread pool coordination. This value was chosen
/// based on typical fuzzy matching workloads where the comparison cost
/// is relatively low per item.
const PARALLEL_THRESHOLD: usize = 100;

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate that a similarity score or min_similarity value is in the valid range [0.0, 1.0]
fn validate_similarity(value: f64, param_name: &str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(ValidationError::new_err(format!(
            "{} must be a finite number, got {}",
            param_name, value
        )));
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(ValidationError::new_err(format!(
            "{} must be in range [0.0, 1.0], got {}",
            param_name, value
        )));
    }
    Ok(())
}

/// Validate that a Jaro-Winkler prefix_weight is in the valid range [0.0, 0.25]
fn validate_prefix_weight(value: f64) -> PyResult<()> {
    if !value.is_finite() {
        return Err(ValidationError::new_err(format!(
            "prefix_weight must be a finite number, got {}",
            value
        )));
    }
    if !(0.0..=0.25).contains(&value) {
        return Err(ValidationError::new_err(format!(
            "prefix_weight must be in range [0.0, 0.25], got {} (values > 0.25 can produce scores > 1.0)",
            value
        )));
    }
    Ok(())
}

/// Validate that an ngram_size is at least 1
fn validate_ngram_size(ngram_size: usize, param_name: &str) -> PyResult<()> {
    if ngram_size < 1 {
        return Err(ValidationError::new_err(format!(
            "{} must be at least 1, got {}",
            param_name, ngram_size
        )));
    }
    Ok(())
}

/// Validate that min_ngram_ratio is in the valid range [0.0, 1.0]
fn validate_ngram_ratio(value: f64) -> PyResult<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(ValidationError::new_err(format!(
            "min_ngram_ratio must be in range [0.0, 1.0], got {}",
            value
        )));
    }
    Ok(())
}

/// Apply optional normalization to a pair of strings.
///
/// Uses Cow for zero-cost passthrough when no normalization is needed.
fn apply_normalization<'a>(
    a: &'a str,
    b: &'a str,
    normalize: Option<&str>,
) -> PyResult<(Cow<'a, str>, Cow<'a, str>)> {
    match normalize {
        None => Ok((Cow::Borrowed(a), Cow::Borrowed(b))),
        Some(mode) => {
            let norm_mode = parse_normalization_mode(mode)?;
            Ok((
                Cow::Owned(algorithms::normalize::normalize_string(a, norm_mode)),
                Cow::Owned(algorithms::normalize::normalize_string(b, norm_mode)),
            ))
        }
    }
}

/// Parse normalization mode from string (used by multiple functions)
fn parse_normalization_mode(norm: &str) -> PyResult<NormalizationMode> {
    match norm.to_lowercase().as_str() {
        "lowercase" => Ok(NormalizationMode::Lowercase),
        "unicode_nfkd" | "nfkd" => Ok(NormalizationMode::UnicodeNFKD),
        "remove_punctuation" => Ok(NormalizationMode::RemovePunctuation),
        "remove_whitespace" => Ok(NormalizationMode::RemoveWhitespace),
        "strict" => Ok(NormalizationMode::Strict),
        _ => Err(ValidationError::new_err(format!(
            "Unknown normalization mode: '{}'. Valid: lowercase, unicode_nfkd, remove_punctuation, remove_whitespace, strict",
            norm
        ))),
    }
}

// ============================================================================
// Algorithm Dispatch Helpers
// ============================================================================

/// Get a boxed similarity metric for the given algorithm name.
///
/// Centralizes algorithm dispatch to avoid duplicating match statements.
/// Returns a trait object that can be used with indices and deduplication.
fn get_similarity_metric(
    algorithm: &str,
) -> PyResult<Box<dyn algorithms::Similarity + Send + Sync>> {
    get_similarity_metric_with_normalize(algorithm, None)
}

/// Get a boxed similarity metric with optional normalization.
///
/// When `normalize` is Some("lowercase"), the metric will compare strings case-insensitively.
fn get_similarity_metric_with_normalize(
    algorithm: &str,
    normalize: Option<&str>,
) -> PyResult<Box<dyn algorithms::Similarity + Send + Sync>> {
    let use_case_insensitive = matches!(normalize, Some("lowercase"));

    macro_rules! wrap_metric {
        ($metric:expr, $case_insensitive:expr) => {
            if $case_insensitive {
                Box::new(algorithms::CaseInsensitive($metric))
                    as Box<dyn algorithms::Similarity + Send + Sync>
            } else {
                Box::new($metric) as Box<dyn algorithms::Similarity + Send + Sync>
            }
        };
    }

    match algorithm {
        "levenshtein" => Ok(wrap_metric!(algorithms::levenshtein::Levenshtein::new(), use_case_insensitive)),
        "damerau_levenshtein" | "damerau" => Ok(wrap_metric!(algorithms::damerau::DamerauLevenshtein::new(), use_case_insensitive)),
        "jaro" => Ok(wrap_metric!(algorithms::jaro::Jaro::new(), use_case_insensitive)),
        "jaro_winkler" => Ok(wrap_metric!(algorithms::jaro::JaroWinkler::new(), use_case_insensitive)),
        "ngram" | "bigram" => Ok(wrap_metric!(algorithms::ngram::Ngram::bigram(), use_case_insensitive)),
        "trigram" => Ok(wrap_metric!(algorithms::ngram::Ngram::trigram(), use_case_insensitive)),
        "lcs" => Ok(wrap_metric!(algorithms::lcs::Lcs::new(), use_case_insensitive)),
        "cosine" | "cosine_chars" => Ok(wrap_metric!(algorithms::cosine::CosineSimilarity::character_based(), use_case_insensitive)),
        _ => Err(AlgorithmError::new_err(format!(
            "Unknown algorithm: '{}'. Valid: levenshtein, damerau_levenshtein, jaro, jaro_winkler, ngram, trigram, lcs, cosine",
            algorithm
        ))),
    }
}

/// Type alias for boxed similarity functions.
type SimilarityFn = Box<dyn Fn(&str, &str) -> f64 + Send + Sync>;

/// Get a boxed similarity function for the given algorithm name.
///
/// Used by find_best_matches where a closure is more appropriate than a trait object.
fn get_similarity_fn(algorithm: &str) -> PyResult<SimilarityFn> {
    match algorithm {
        "levenshtein" => Ok(Box::new(algorithms::levenshtein::levenshtein_similarity)),
        "damerau_levenshtein" | "damerau" => Ok(Box::new(algorithms::damerau::damerau_levenshtein_similarity)),
        "jaro" => Ok(Box::new(algorithms::jaro::jaro_similarity)),
        "jaro_winkler" => Ok(Box::new(algorithms::jaro::jaro_winkler_similarity)),
        "ngram" | "bigram" => Ok(Box::new(algorithms::ngram::bigram_similarity)),
        "trigram" => Ok(Box::new(algorithms::ngram::trigram_similarity)),
        "lcs" => Ok(Box::new(algorithms::lcs::lcs_similarity)),
        "cosine" | "cosine_chars" => Ok(Box::new(algorithms::cosine::cosine_similarity_chars)),
        _ => Err(AlgorithmError::new_err(format!(
            "Unknown algorithm: '{}'. Valid: levenshtein, damerau_levenshtein, jaro, jaro_winkler, ngram, trigram, lcs, cosine",
            algorithm
        ))),
    }
}

// ============================================================================
// SearchResult Conversion Helpers
// ============================================================================

/// Convert a BK-tree SearchResult to our Python SearchResult.
///
/// Computes similarity score from edit distance using normalized formula.
/// Clamps score to [0.0, 1.0] to handle edge cases where distance might exceed max_len
/// (shouldn't happen with proper edit distance functions, but defensive programming).
fn convert_bktree_result(r: indexing::bktree::SearchResult, query: &str) -> SearchResult {
    let max_len = query.chars().count().max(r.text.chars().count());
    let score = if max_len == 0 {
        1.0
    } else {
        // Clamp to [0.0, 1.0] to handle edge cases where distance > max_len
        // (e.g., with custom distance functions or pathological inputs)
        (1.0 - (r.distance as f64 / max_len as f64)).clamp(0.0, 1.0)
    };
    SearchResult {
        id: r.id,
        text: r.text,
        score,
        distance: Some(r.distance),
        data: r.data,
    }
}

/// Convert a SearchMatch (from NgramIndex/HybridIndex) to SearchResult.
fn convert_search_match(m: indexing::SearchMatch) -> SearchResult {
    SearchResult {
        id: m.id,
        text: m.text,
        score: m.similarity,
        distance: None, // N-gram indices don't track edit distance
        data: m.data,
    }
}

// ============================================================================
// Python Result Types
// ============================================================================

/// Unified search result across all index types.
#[pyclass]
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Unique ID of the matched item
    #[pyo3(get)]
    pub id: usize,

    /// The matched text
    #[pyo3(get)]
    pub text: String,

    /// Similarity score (0.0-1.0)
    #[pyo3(get)]
    pub score: f64,

    /// Optional edit distance
    #[pyo3(get)]
    pub distance: Option<usize>,

    /// Optional user-provided data
    #[pyo3(get)]
    pub data: Option<u64>,
}

#[pymethods]
impl SearchResult {
    #[new]
    #[pyo3(signature = (id, text, score, distance=None, data=None))]
    fn new(
        id: usize,
        text: String,
        score: f64,
        distance: Option<usize>,
        data: Option<u64>,
    ) -> Self {
        Self {
            id,
            text,
            score,
            distance,
            data,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(id={}, text='{}', score={:.3}, distance={:?}, data={:?})",
            self.id, self.text, self.score, self.distance, self.data
        )
    }

    fn __str__(&self) -> String {
        format!("{}: {:.3}", self.text, self.score)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.id == other.id
            && self.text == other.text
            && (self.score - other.score).abs() < 1e-9
            && self.distance == other.distance
            && self.data == other.data
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.id.hash(&mut hasher);
        self.text.hash(&mut hasher);
        // Convert score to bits for stable hashing (round to 9 decimal places)
        ((self.score * 1e9).round() as i64).hash(&mut hasher);
        self.distance.hash(&mut hasher);
        self.data.hash(&mut hasher);
        hasher.finish()
    }
}

/// Result from find_best_matches and batch operations.
#[pyclass]
#[derive(Clone, Debug)]
pub struct MatchResult {
    /// The matched text
    #[pyo3(get)]
    pub text: String,

    /// Similarity score (0.0-1.0)
    #[pyo3(get)]
    pub score: f64,

    /// Optional index/ID of the original item
    #[pyo3(get)]
    pub id: Option<usize>,
}

#[pymethods]
impl MatchResult {
    #[new]
    #[pyo3(signature = (text, score, id=None))]
    fn new(text: String, score: f64, id: Option<usize>) -> Self {
        Self { text, score, id }
    }

    fn __repr__(&self) -> String {
        match self.id {
            Some(id) => format!(
                "MatchResult(text='{}', score={:.3}, id={})",
                self.text, self.score, id
            ),
            None => format!("MatchResult(text='{}', score={:.3})", self.text, self.score),
        }
    }

    fn __str__(&self) -> String {
        format!("{}: {:.3}", self.text, self.score)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.text == other.text && (self.score - other.score).abs() < 1e-9 && self.id == other.id
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.text.hash(&mut hasher);
        // Convert score to bits for stable hashing (round to 9 decimal places)
        ((self.score * 1e9).round() as i64).hash(&mut hasher);
        self.id.hash(&mut hasher);
        hasher.finish()
    }
}

/// Result from deduplication operation.
#[pyclass]
#[derive(Clone, Debug)]
pub struct DeduplicationResult {
    /// Groups of duplicate items (each group contains similar strings)
    #[pyo3(get)]
    pub groups: Vec<Vec<String>>,

    /// Items that are unique (no duplicates found)
    #[pyo3(get)]
    pub unique: Vec<String>,

    /// Total number of duplicate items found
    #[pyo3(get)]
    pub total_duplicates: usize,
}

#[pymethods]
impl DeduplicationResult {
    #[new]
    fn new(groups: Vec<Vec<String>>, unique: Vec<String>, total_duplicates: usize) -> Self {
        Self {
            groups,
            unique,
            total_duplicates,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DeduplicationResult(groups={}, unique={}, total_duplicates={})",
            self.groups.len(),
            self.unique.len(),
            self.total_duplicates
        )
    }

    fn __str__(&self) -> String {
        format!(
            "{} duplicate groups, {} unique items, {} total duplicates",
            self.groups.len(),
            self.unique.len(),
            self.total_duplicates
        )
    }
}

/// Result from multi-algorithm comparison.
#[pyclass]
#[derive(Clone, Debug)]
pub struct AlgorithmComparison {
    /// Algorithm name
    #[pyo3(get)]
    pub algorithm: String,

    /// Average similarity score across all comparisons (0.0-1.0)
    #[pyo3(get)]
    pub score: f64,

    /// Top matches for this algorithm
    #[pyo3(get)]
    pub matches: Vec<MatchResult>,
}

#[pymethods]
impl AlgorithmComparison {
    #[new]
    fn new(algorithm: String, score: f64, matches: Vec<MatchResult>) -> Self {
        Self {
            algorithm,
            score,
            matches,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AlgorithmComparison(algorithm='{}', score={:.3}, matches={})",
            self.algorithm,
            self.score,
            self.matches.len()
        )
    }

    fn __str__(&self) -> String {
        format!(
            "{}: {:.3} ({} matches)",
            self.algorithm,
            self.score,
            self.matches.len()
        )
    }
}

/// Result from confusion matrix calculation.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ConfusionMatrixResult {
    /// True positives: correctly predicted matches
    #[pyo3(get)]
    pub tp: usize,

    /// False positives: incorrectly predicted matches
    #[pyo3(get)]
    pub fp: usize,

    /// False negatives: missed matches
    #[pyo3(get)]
    pub fn_count: usize,

    /// True negatives: correctly rejected non-matches
    #[pyo3(get)]
    pub tn: usize,
}

#[pymethods]
impl ConfusionMatrixResult {
    #[new]
    fn new(tp: usize, fp: usize, fn_count: usize, tn: usize) -> Self {
        Self {
            tp,
            fp,
            fn_count,
            tn,
        }
    }

    /// Calculate precision from confusion matrix values.
    fn precision(&self) -> f64 {
        let denominator = self.tp + self.fp;
        if denominator == 0 {
            if self.fn_count == 0 {
                1.0
            } else {
                0.0
            }
        } else {
            self.tp as f64 / denominator as f64
        }
    }

    /// Calculate recall from confusion matrix values.
    fn recall(&self) -> f64 {
        let denominator = self.tp + self.fn_count;
        if denominator == 0 {
            if self.fp == 0 {
                1.0
            } else {
                0.0
            }
        } else {
            self.tp as f64 / denominator as f64
        }
    }

    /// Calculate F-beta score from confusion matrix values.
    #[pyo3(signature = (beta=1.0))]
    fn f_score(&self, beta: f64) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            let beta_sq = beta * beta;
            (1.0 + beta_sq) * p * r / (beta_sq * p + r)
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ConfusionMatrixResult(tp={}, fp={}, fn={}, tn={})",
            self.tp, self.fp, self.fn_count, self.tn
        )
    }

    fn __str__(&self) -> String {
        format!(
            "TP={}, FP={}, FN={}, TN={} (precision={:.3}, recall={:.3})",
            self.tp,
            self.fp,
            self.fn_count,
            self.tn,
            self.precision(),
            self.recall()
        )
    }
}

// ============================================================================
// Python Bindings
// ============================================================================

/// Compute Levenshtein (edit) distance between two strings.
///
/// # Arguments
/// * `max_distance` - Optional threshold. If distance exceeds this, returns `max_distance + 1`
/// * `normalize` - Optional normalization mode: "lowercase", "unicode_nfkd", "remove_punctuation", "remove_whitespace", "strict"
///
/// Returns `max_distance + 1` if distance exceeds threshold (for RapidFuzz compatibility).
/// Use `levenshtein_bounded()` if you prefer `None` semantics.
#[pyfunction]
#[pyo3(signature = (a, b, max_distance=None, normalize=None))]
fn levenshtein(
    a: &str,
    b: &str,
    max_distance: Option<usize>,
    normalize: Option<&str>,
) -> PyResult<usize> {
    let (a, b) = apply_normalization(a, b, normalize)?;
    match max_distance {
        Some(max_d) => {
            Ok(
                algorithms::levenshtein::levenshtein_distance_bounded(&a, &b, Some(max_d))
                    .unwrap_or_else(|| max_d.saturating_add(1)),
            )
        }
        None => Ok(
            algorithms::levenshtein::levenshtein_distance_bounded(&a, &b, None)
                .expect("unbounded distance should always return Some"),
        ),
    }
}

/// Compute Levenshtein distance with explicit `None` return when threshold exceeded.
///
/// # Arguments
/// * `max_distance` - Threshold. Returns `None` if distance exceeds this value.
/// * `normalize` - Optional normalization mode
///
/// Returns `Some(distance)` if within threshold, `None` if exceeded.
#[pyfunction]
#[pyo3(signature = (a, b, max_distance, normalize=None))]
fn levenshtein_bounded(
    a: &str,
    b: &str,
    max_distance: usize,
    normalize: Option<&str>,
) -> PyResult<Option<usize>> {
    let (a, b) = apply_normalization(a, b, normalize)?;
    Ok(algorithms::levenshtein::levenshtein_distance_bounded(
        &a,
        &b,
        Some(max_distance),
    ))
}

/// Compute normalized Levenshtein similarity (0.0 to 1.0).
///
/// # Arguments
/// * `normalize` - Optional normalization mode: "lowercase", "unicode_nfkd", "remove_punctuation", "remove_whitespace", "strict"
#[pyfunction]
#[pyo3(signature = (a, b, normalize=None))]
fn levenshtein_similarity(a: &str, b: &str, normalize: Option<&str>) -> PyResult<f64> {
    let (a, b) = apply_normalization(a, b, normalize)?;
    Ok(algorithms::levenshtein::levenshtein_similarity(&a, &b))
}

/// Levenshtein distance treating grapheme clusters as single units.
///
/// Useful for text with emoji sequences or combining characters where a single
/// visual character may be multiple Unicode code points.
///
/// # Example
/// ```python
/// >>> # Family emoji is 7 code points but 1 grapheme
/// >>> levenshtein_grapheme("👨‍👩‍👧‍👦", "👨")
/// 1
/// >>> levenshtein("👨‍👩‍👧‍👦", "👨")  # Regular mode counts code points
/// 6
/// ```
#[pyfunction]
fn levenshtein_grapheme(a: &str, b: &str) -> usize {
    algorithms::levenshtein::levenshtein_grapheme(a, b)
}

/// Levenshtein similarity (0.0 to 1.0) treating grapheme clusters as single units.
#[pyfunction]
fn levenshtein_similarity_grapheme(a: &str, b: &str) -> f64 {
    algorithms::levenshtein::levenshtein_similarity_grapheme(a, b)
}

/// Jaro similarity treating grapheme clusters as single units.
#[pyfunction]
fn jaro_similarity_grapheme(a: &str, b: &str) -> f64 {
    algorithms::jaro::jaro_similarity_grapheme(a, b)
}

/// Jaro-Winkler similarity treating grapheme clusters as single units.
#[pyfunction]
#[pyo3(signature = (a, b, prefix_weight=0.1, max_prefix_length=4))]
fn jaro_winkler_similarity_grapheme(
    a: &str,
    b: &str,
    prefix_weight: f64,
    max_prefix_length: usize,
) -> f64 {
    algorithms::jaro::jaro_winkler_similarity_grapheme_params(
        a,
        b,
        prefix_weight,
        max_prefix_length,
    )
}

// ============================================================================
// SIMD-Accelerated Functions
// ============================================================================

/// SIMD-accelerated Levenshtein distance.
///
/// Uses triple_accel's SIMD-optimized implementation which can be 20-30x faster
/// than the scalar implementation for longer strings. Automatically falls back
/// to scalar on CPUs without SIMD support (e.g., ARM).
///
/// Best for:
/// - Long strings (>100 characters)
/// - Batch processing many comparisons
/// - x86/x86-64 CPUs with AVX2 or SSE4.1
///
/// Note: Works on byte-level, so for Unicode strings the result may differ
/// slightly from character-level distance (use regular levenshtein for
/// character-accurate Unicode distance).
#[pyfunction]
fn levenshtein_simd(a: &str, b: &str) -> usize {
    algorithms::levenshtein::levenshtein_simd(a, b)
}

/// SIMD-accelerated Levenshtein distance with max threshold.
///
/// Returns None if the distance exceeds max_distance, enabling early termination.
/// Uses SIMD acceleration when available for optimal performance.
#[pyfunction]
fn levenshtein_simd_bounded(a: &str, b: &str, max_distance: usize) -> Option<usize> {
    algorithms::levenshtein::levenshtein_simd_bounded(a, b, max_distance)
}

/// SIMD-accelerated Levenshtein similarity (0.0 to 1.0).
///
/// Uses SIMD acceleration for the distance calculation, providing
/// significant speedups on x86/x86-64 CPUs.
#[pyfunction]
fn levenshtein_similarity_simd(a: &str, b: &str) -> f64 {
    algorithms::levenshtein::levenshtein_similarity_simd(a, b)
}

/// Compute Damerau-Levenshtein distance (includes transpositions).
///
/// # Arguments
/// * `max_distance` - Optional threshold. If distance exceeds this, returns `max_distance + 1`
/// * `normalize` - Optional normalization mode: "lowercase", "unicode_nfkd", "remove_punctuation", "remove_whitespace", "strict"
///
/// Returns `max_distance + 1` if distance exceeds threshold (for RapidFuzz compatibility).
/// Use `damerau_levenshtein_bounded()` if you prefer `None` semantics.
#[pyfunction]
#[pyo3(signature = (a, b, max_distance=None, normalize=None))]
fn damerau_levenshtein(
    a: &str,
    b: &str,
    max_distance: Option<usize>,
    normalize: Option<&str>,
) -> PyResult<usize> {
    let (a, b) = apply_normalization(a, b, normalize)?;
    match max_distance {
        Some(max_d) => {
            Ok(
                algorithms::damerau::damerau_levenshtein_distance_bounded(&a, &b, Some(max_d))
                    .unwrap_or_else(|| max_d.saturating_add(1)),
            )
        }
        None => Ok(
            algorithms::damerau::damerau_levenshtein_distance_bounded(&a, &b, None)
                .expect("unbounded distance should always return Some"),
        ),
    }
}

/// Compute Damerau-Levenshtein distance with explicit `None` return when threshold exceeded.
///
/// # Arguments
/// * `max_distance` - Threshold. Returns `None` if distance exceeds this value.
/// * `normalize` - Optional normalization mode
///
/// Returns `Some(distance)` if within threshold, `None` if exceeded.
#[pyfunction]
#[pyo3(signature = (a, b, max_distance, normalize=None))]
fn damerau_levenshtein_bounded(
    a: &str,
    b: &str,
    max_distance: usize,
    normalize: Option<&str>,
) -> PyResult<Option<usize>> {
    let (a, b) = apply_normalization(a, b, normalize)?;
    Ok(algorithms::damerau::damerau_levenshtein_distance_bounded(
        &a,
        &b,
        Some(max_distance),
    ))
}

/// Compute normalized Damerau-Levenshtein similarity.
///
/// # Arguments
/// * `normalize` - Optional normalization mode: "lowercase", "unicode_nfkd", "remove_punctuation", "remove_whitespace", "strict"
#[pyfunction]
#[pyo3(signature = (a, b, normalize=None))]
fn damerau_levenshtein_similarity(a: &str, b: &str, normalize: Option<&str>) -> PyResult<f64> {
    let (a, b) = apply_normalization(a, b, normalize)?;
    Ok(algorithms::damerau::damerau_levenshtein_similarity(&a, &b))
}

/// Compute Jaro similarity (0.0 to 1.0).
///
/// # Arguments
/// * `normalize` - Optional normalization mode: "lowercase", "unicode_nfkd", "remove_punctuation", "remove_whitespace", "strict"
#[pyfunction]
#[pyo3(signature = (a, b, normalize=None))]
fn jaro_similarity(a: &str, b: &str, normalize: Option<&str>) -> PyResult<f64> {
    let (a, b) = apply_normalization(a, b, normalize)?;
    Ok(algorithms::jaro::jaro_similarity(&a, &b))
}

/// Compute Jaro-Winkler similarity (0.0 to 1.0).
///
/// # Arguments
/// * `prefix_weight` - Weight for common prefix bonus (must be in [0.0, 0.25])
/// * `max_prefix_length` - Maximum prefix length to consider
/// * `normalize` - Optional normalization mode: "lowercase", "unicode_nfkd", "remove_punctuation", "remove_whitespace", "strict"
#[pyfunction]
#[pyo3(signature = (a, b, prefix_weight=0.1, max_prefix_length=4, normalize=None))]
fn jaro_winkler_similarity(
    a: &str,
    b: &str,
    prefix_weight: f64,
    max_prefix_length: usize,
    normalize: Option<&str>,
) -> PyResult<f64> {
    validate_prefix_weight(prefix_weight)?;
    let (a, b) = apply_normalization(a, b, normalize)?;
    Ok(algorithms::jaro::jaro_winkler_similarity_params(
        &a,
        &b,
        prefix_weight,
        max_prefix_length,
    ))
}

/// Compute Hamming distance (strings must have equal length).
#[pyfunction]
fn hamming(a: &str, b: &str) -> PyResult<usize> {
    algorithms::hamming::hamming_distance(a, b).ok_or_else(|| {
        ValidationError::new_err("Strings must have equal length for Hamming distance")
    })
}

/// Compute n-gram similarity (Sørensen-Dice coefficient).
///
/// # Arguments
/// * `normalize` - Optional normalization mode: "lowercase", "unicode_nfkd", "remove_punctuation", "remove_whitespace", "strict"
#[pyfunction]
#[pyo3(signature = (a, b, ngram_size=3, pad=true, normalize=None))]
fn ngram_similarity(
    a: &str,
    b: &str,
    ngram_size: usize,
    pad: bool,
    normalize: Option<&str>,
) -> PyResult<f64> {
    validate_ngram_size(ngram_size, "ngram_size")?;
    let (a, b) = apply_normalization(a, b, normalize)?;
    Ok(algorithms::ngram::ngram_similarity(
        &a, &b, ngram_size, pad, ' ',
    ))
}

/// Compute n-gram Jaccard similarity.
///
/// # Arguments
/// * `normalize` - Optional normalization mode: "lowercase", "unicode_nfkd", "remove_punctuation", "remove_whitespace", "strict"
#[pyfunction]
#[pyo3(signature = (a, b, ngram_size=3, pad=true, normalize=None))]
fn ngram_jaccard(
    a: &str,
    b: &str,
    ngram_size: usize,
    pad: bool,
    normalize: Option<&str>,
) -> PyResult<f64> {
    validate_ngram_size(ngram_size, "ngram_size")?;
    let (a, b) = apply_normalization(a, b, normalize)?;
    Ok(algorithms::ngram::ngram_jaccard_similarity(
        &a, &b, ngram_size, pad, ' ',
    ))
}

/// Extract n-grams from a string.
#[pyfunction]
#[pyo3(signature = (s, ngram_size=3, pad=true))]
fn extract_ngrams(s: &str, ngram_size: usize, pad: bool) -> PyResult<Vec<String>> {
    validate_ngram_size(ngram_size, "ngram_size")?;
    Ok(algorithms::ngram::extract_ngrams(s, ngram_size, pad, ' '))
}

/// Encode a string using Soundex algorithm.
#[pyfunction]
fn soundex(s: &str) -> String {
    algorithms::phonetic::soundex(s)
}

/// Check if two strings match phonetically using Soundex.
#[pyfunction]
fn soundex_match(a: &str, b: &str) -> bool {
    algorithms::phonetic::soundex_match(a, b)
}

/// Encode a string using Metaphone algorithm.
#[pyfunction]
#[pyo3(signature = (s, max_length=4))]
fn metaphone(s: &str, max_length: usize) -> String {
    algorithms::phonetic::metaphone(s, max_length)
}

/// Check if two strings match phonetically using Metaphone.
#[pyfunction]
fn metaphone_match(a: &str, b: &str) -> bool {
    algorithms::phonetic::metaphone_match(a, b)
}

/// Encode a string using Double Metaphone algorithm.
///
/// Returns a tuple of (primary, alternate) codes. Double Metaphone is an improvement
/// over Metaphone that handles European names (Germanic, Slavic, Italian, etc.) better
/// by returning two possible pronunciations.
///
/// Args:
///     text: String to encode
///     max_length: Maximum length of each code (default: 4)
///
/// Returns:
///     Tuple of (primary_code, alternate_code). The alternate may be empty if there's
///     only one pronunciation.
///
/// # Example
/// ```python
/// >>> double_metaphone("Schmidt")
/// ('XMT', 'SMT')  # German and Anglicized pronunciations
/// >>> double_metaphone("Smith")
/// ('SM0', 'XMT')  # 0 represents the TH sound
/// ```
#[pyfunction]
#[pyo3(signature = (text, max_length=4))]
fn double_metaphone(text: &str, max_length: usize) -> (String, String) {
    algorithms::phonetic::double_metaphone(text, max_length)
}

/// Check if two strings match phonetically using Double Metaphone.
///
/// Returns true if any of the codes match (primary-primary, primary-alternate,
/// alternate-primary, or alternate-alternate).
///
/// This is more lenient than regular metaphone_match because it considers
/// alternate pronunciations, making it better for matching names with
/// varied spellings (e.g., Schmidt/Smith, Catherine/Katherine).
#[pyfunction]
fn double_metaphone_match(a: &str, b: &str) -> bool {
    algorithms::phonetic::double_metaphone_match(a, b)
}

/// Compute Double Metaphone similarity between two strings.
///
/// Returns 1.0 for exact phonetic match (any code combination matches),
/// or a partial score based on Jaro-Winkler similarity of the best
/// matching code pair.
///
/// Args:
///     a: First string
///     b: Second string
///     max_length: Maximum length of metaphone codes (default: 4)
///
/// Returns:
///     Similarity score from 0.0 to 1.0
#[pyfunction]
#[pyo3(signature = (a, b, max_length=4))]
fn double_metaphone_similarity(a: &str, b: &str, max_length: usize) -> f64 {
    algorithms::phonetic::double_metaphone_similarity(a, b, max_length)
}

/// Compute LCS (Longest Common Subsequence) length.
#[pyfunction]
fn lcs_length(a: &str, b: &str) -> usize {
    algorithms::lcs::lcs_length(a, b)
}

/// Get the actual LCS string.
///
/// Note: For strings longer than 10,000 characters, this raises a ValidationError
/// to prevent excessive memory allocation (O(m*n) space required).
/// Use `lcs_length` for a space-efficient length-only computation.
#[pyfunction]
fn lcs_string(a: &str, b: &str) -> PyResult<String> {
    const MAX_LEN: usize = 10_000;
    if a.chars().count() > MAX_LEN || b.chars().count() > MAX_LEN {
        return Err(ValidationError::new_err(format!(
            "Input strings exceed maximum length of {} characters for lcs_string. \
             Use lcs_length for a space-efficient alternative.",
            MAX_LEN
        )));
    }
    Ok(algorithms::lcs::lcs_string(a, b))
}

/// Compute LCS-based similarity.
#[pyfunction]
fn lcs_similarity(a: &str, b: &str) -> f64 {
    algorithms::lcs::lcs_similarity(a, b)
}

/// Compute longest common substring length.
#[pyfunction]
fn longest_common_substring_length(a: &str, b: &str) -> usize {
    algorithms::lcs::longest_common_substring_length(a, b)
}

/// Get the longest common substring.
///
/// Note: For strings longer than 10,000 characters, this raises a ValidationError
/// to prevent excessive memory allocation (O(m*n) space required).
/// Use `longest_common_substring_length` for a space-efficient alternative.
#[pyfunction]
fn longest_common_substring(a: &str, b: &str) -> PyResult<String> {
    const MAX_LEN: usize = 10_000;
    if a.chars().count() > MAX_LEN || b.chars().count() > MAX_LEN {
        return Err(ValidationError::new_err(format!(
            "Input strings exceed maximum length of {} characters for longest_common_substring. \
             Use longest_common_substring_length for a space-efficient alternative.",
            MAX_LEN
        )));
    }
    Ok(algorithms::lcs::longest_common_substring(a, b))
}

/// Compute character-level cosine similarity.
///
/// # Arguments
/// * `normalize` - Optional normalization mode: "lowercase", "unicode_nfkd", "remove_punctuation", "remove_whitespace", "strict"
#[pyfunction]
#[pyo3(signature = (a, b, normalize=None))]
fn cosine_similarity_chars(a: &str, b: &str, normalize: Option<&str>) -> PyResult<f64> {
    let (a, b) = apply_normalization(a, b, normalize)?;
    Ok(algorithms::cosine::cosine_similarity_chars(&a, &b))
}

/// Compute word-level cosine similarity.
///
/// # Arguments
/// * `normalize` - Optional normalization mode: "lowercase", "unicode_nfkd", "remove_punctuation", "remove_whitespace", "strict"
#[pyfunction]
#[pyo3(signature = (a, b, normalize=None))]
fn cosine_similarity_words(a: &str, b: &str, normalize: Option<&str>) -> PyResult<f64> {
    let (a, b) = apply_normalization(a, b, normalize)?;
    Ok(algorithms::cosine::cosine_similarity_words(&a, &b))
}

/// Compute n-gram cosine similarity.
///
/// # Arguments
/// * `normalize` - Optional normalization mode: "lowercase", "unicode_nfkd", "remove_punctuation", "remove_whitespace", "strict"
#[pyfunction]
#[pyo3(signature = (a, b, ngram_size=3, normalize=None))]
fn cosine_similarity_ngrams(
    a: &str,
    b: &str,
    ngram_size: usize,
    normalize: Option<&str>,
) -> PyResult<f64> {
    validate_ngram_size(ngram_size, "ngram_size")?;
    let (a, b) = apply_normalization(a, b, normalize)?;
    Ok(algorithms::cosine::cosine_similarity_ngrams(
        &a, &b, ngram_size,
    ))
}

/// Compute Soundex phonetic similarity (0.0 to 1.0).
///
/// Returns 1.0 for identical codes, otherwise a partial match score
/// based on matching positions in the 4-character Soundex codes.
#[pyfunction]
fn soundex_similarity(a: &str, b: &str) -> f64 {
    algorithms::phonetic::soundex_similarity(a, b)
}

/// Compute Metaphone phonetic similarity (0.0 to 1.0).
///
/// Uses Jaro-Winkler similarity on the Metaphone codes for partial matching.
#[pyfunction]
#[pyo3(signature = (a, b, max_length=4))]
fn metaphone_similarity(a: &str, b: &str, max_length: usize) -> f64 {
    algorithms::phonetic::metaphone_similarity(a, b, max_length)
}

/// Compute bigram similarity (n-gram with n=2).
///
/// Convenience function equivalent to ngram_similarity(a, b, 2).
#[pyfunction]
fn bigram_similarity(a: &str, b: &str) -> f64 {
    algorithms::ngram::bigram_similarity(a, b)
}

/// Compute trigram similarity (n-gram with n=3).
///
/// Convenience function equivalent to ngram_similarity(a, b, 3).
#[pyfunction]
fn trigram_similarity(a: &str, b: &str) -> f64 {
    algorithms::ngram::trigram_similarity(a, b)
}

/// Compute n-gram profile similarity.
///
/// Unlike regular n-gram similarity which only checks presence,
/// this counts n-gram frequencies for a more accurate comparison
/// when strings have repeated patterns.
#[pyfunction]
#[pyo3(signature = (a, b, ngram_size=3))]
fn ngram_profile_similarity(a: &str, b: &str, ngram_size: usize) -> PyResult<f64> {
    validate_ngram_size(ngram_size, "ngram_size")?;
    Ok(algorithms::ngram::ngram_profile_similarity(
        a, b, ngram_size,
    ))
}

/// Compute Hamming distance with padding.
///
/// Unlike regular Hamming distance which requires equal-length strings,
/// this pads the shorter string to enable comparison.
#[pyfunction]
fn hamming_distance_padded(a: &str, b: &str) -> usize {
    algorithms::hamming::hamming_distance_padded(a, b)
}

/// Compute normalized Hamming similarity (0.0 to 1.0).
///
/// Raises ValidationError if strings have different lengths.
#[pyfunction]
fn hamming_similarity(a: &str, b: &str) -> PyResult<f64> {
    algorithms::hamming::hamming_similarity(a, b).ok_or_else(|| {
        ValidationError::new_err("Strings must have equal length for Hamming similarity")
    })
}

/// Compute LCS similarity using max length as denominator.
///
/// Alternative to lcs_similarity which uses sum of lengths.
/// Formula: LCS_length / max(len_a, len_b)
#[pyfunction]
fn lcs_similarity_max(a: &str, b: &str) -> f64 {
    algorithms::lcs::lcs_similarity_max(a, b)
}

// ============================================================================
// RapidFuzz-Compatible Convenience Functions
// ============================================================================

/// Compute best partial match ratio between two strings.
///
/// Slides the shorter string across the longer string and returns the
/// maximum similarity found. Useful for matching when one string is
/// a substring of the other.
///
/// # Arguments
/// * `s1` - First string
/// * `s2` - Second string
///
/// # Returns
/// Similarity score (0.0 to 1.0). Returns 1.0 if one string is a perfect substring of the other.
///
/// # Example
/// ```python
/// >>> partial_ratio("test", "this is a test!")
/// 1.0
/// ```
#[pyfunction]
fn partial_ratio(s1: &str, s2: &str) -> f64 {
    algorithms::fuzz::partial_ratio(s1, s2)
}

/// Compute similarity after tokenizing and sorting both strings.
///
/// Useful for comparing strings where word order doesn't matter.
/// "fuzzy wuzzy was a bear" matches "was a bear fuzzy wuzzy" perfectly.
///
/// # Arguments
/// * `s1` - First string
/// * `s2` - Second string
///
/// # Returns
/// Similarity score (0.0 to 1.0).
///
/// # Example
/// ```python
/// >>> token_sort_ratio("fuzzy wuzzy", "wuzzy fuzzy")
/// 1.0
/// ```
#[pyfunction]
fn token_sort_ratio(s1: &str, s2: &str) -> f64 {
    algorithms::fuzz::token_sort_ratio(s1, s2)
}

/// Compute set-based token similarity.
///
/// Useful for comparing strings where duplicates and order don't matter.
/// "fuzzy fuzzy was a bear" matches "fuzzy was a bear" highly.
///
/// # Arguments
/// * `s1` - First string
/// * `s2` - Second string
///
/// # Returns
/// Similarity score (0.0 to 1.0).
///
/// # Example
/// ```python
/// >>> token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear")
/// 0.95
/// ```
#[pyfunction]
fn token_set_ratio(s1: &str, s2: &str) -> f64 {
    algorithms::fuzz::token_set_ratio(s1, s2)
}

/// Compute weighted ratio using the best method for the input.
///
/// Automatically selects the best comparison method based on string
/// characteristics:
/// - For similar-length strings: uses basic ratio
/// - For different-length strings: uses partial_ratio (scaled)
/// - Also considers token_sort_ratio and token_set_ratio (scaled)
///
/// Returns the maximum of all methods (with appropriate weights).
///
/// # Arguments
/// * `s1` - First string
/// * `s2` - Second string
/// * `partial_weight` - Weight for partial_ratio (0.0 to 1.0, default: 0.9).
///   Higher values favor substring matching.
/// * `token_weight` - Weight for token-based ratios (0.0 to 1.0, default: 0.95).
///   Higher values favor word-order-independent matching.
///
/// # Returns
/// Similarity score (0.0 to 1.0).
///
/// # Example
/// ```python
/// >>> wratio("hello world", "hello there world")
/// 0.82
/// >>> # Custom weights to favor token matching
/// >>> wratio("hello world", "world hello", partial_weight=0.9, token_weight=1.0)
/// 1.0
/// ```
#[pyfunction]
#[pyo3(signature = (s1, s2, partial_weight=0.9, token_weight=0.95))]
fn wratio(s1: &str, s2: &str, partial_weight: f64, token_weight: f64) -> f64 {
    algorithms::fuzz::wratio_with_weights(s1, s2, partial_weight, token_weight)
}

/// Compute basic similarity ratio (Levenshtein-based).
///
/// This is an alias for levenshtein_similarity, providing API compatibility
/// with RapidFuzz's `fuzz.ratio`.
///
/// # Arguments
/// * `s1` - First string
/// * `s2` - Second string
///
/// # Returns
/// Similarity score (0.0 to 1.0).
#[pyfunction]
fn ratio(s1: &str, s2: &str) -> f64 {
    algorithms::fuzz::ratio(s1, s2)
}

/// Compute similarity ratio with automatic Unicode NFC normalization.
///
/// This is equivalent to ratio() but first applies Unicode NFC normalization
/// to both strings. This ensures that equivalent Unicode representations
/// (e.g., "é" as a single character vs "e" + combining accent) are treated
/// as identical.
///
/// Args:
///     s1: First string
///     s2: Second string
///
/// Returns:
///     Similarity score (0.0 to 1.0).
#[pyfunction]
fn qratio(s1: &str, s2: &str) -> f64 {
    algorithms::fuzz::qratio(s1, s2)
}

/// Compute weighted ratio with automatic Unicode NFC normalization.
///
/// This is equivalent to wratio() but first applies Unicode NFC normalization
/// to both strings before comparison.
///
/// Args:
///     s1: First string
///     s2: Second string
///     partial_weight: Weight for partial_ratio (0.0 to 1.0, default: 0.9)
///     token_weight: Weight for token-based ratios (0.0 to 1.0, default: 0.95)
///
/// Returns:
///     Similarity score (0.0 to 1.0).
#[pyfunction]
#[pyo3(signature = (s1, s2, partial_weight=0.9, token_weight=0.95))]
fn qwratio(s1: &str, s2: &str, partial_weight: f64, token_weight: f64) -> f64 {
    algorithms::fuzz::qwratio_with_weights(s1, s2, partial_weight, token_weight)
}

/// Find top N matches from a list (RapidFuzz-compatible).
///
/// This is an alias for find_best_matches with RapidFuzz-compatible naming.
/// Uses wratio by default for best automatic matching.
///
/// # Arguments
/// * `query` - Query string to match
/// * `choices` - List of strings to search
/// * `limit` - Maximum number of results (default 10)
/// * `min_similarity` - Minimum similarity threshold (default 0.0)
///
/// # Returns
/// List of MatchResult objects sorted by score descending.
///
/// # Example
/// ```python
/// >>> extract("appel", ["apple", "apply", "banana"], limit=2)
/// [MatchResult(text='apple', score=0.93), MatchResult(text='apply', score=0.80)]
/// ```
#[pyfunction]
#[pyo3(signature = (query, choices, limit=10, min_similarity=0.0))]
fn extract(
    py: Python<'_>,
    query: &str,
    choices: Vec<String>,
    limit: usize,
    min_similarity: f64,
) -> Vec<MatchResult> {
    let query = query.to_string();
    let mut results: Vec<MatchResult> = py.allow_threads(|| {
        if choices.len() >= PARALLEL_THRESHOLD {
            use rayon::prelude::*;
            choices
                .par_iter()
                .enumerate()
                .map(|(i, s)| {
                    let score = algorithms::fuzz::wratio(&query, s);
                    MatchResult {
                        text: s.clone(),
                        score,
                        id: Some(i),
                    }
                })
                .filter(|r| r.score >= min_similarity)
                .collect()
        } else {
            choices
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let score = algorithms::fuzz::wratio(&query, s);
                    MatchResult {
                        text: s.clone(),
                        score,
                        id: Some(i),
                    }
                })
                .filter(|r| r.score >= min_similarity)
                .collect()
        }
    });

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit);
    results
}

/// Find the single best match from a list (RapidFuzz-compatible).
///
/// Returns the best match if one exists above the minimum similarity, otherwise None.
///
/// # Arguments
/// * `query` - Query string to match
/// * `choices` - List of strings to search
/// * `min_similarity` - Minimum similarity threshold (default 0.0)
///
/// # Returns
/// Best MatchResult if found above threshold, otherwise None.
///
/// # Example
/// ```python
/// >>> extract_one("appel", ["apple", "banana"])
/// MatchResult(text='apple', score=0.93)
/// ```
#[pyfunction]
#[pyo3(signature = (query, choices, min_similarity=0.0))]
fn extract_one(
    py: Python<'_>,
    query: &str,
    choices: Vec<String>,
    min_similarity: f64,
) -> Option<MatchResult> {
    let query = query.to_string();
    let results: Vec<MatchResult> = py.allow_threads(|| {
        if choices.len() >= PARALLEL_THRESHOLD {
            use rayon::prelude::*;
            choices
                .par_iter()
                .enumerate()
                .map(|(i, s)| {
                    let score = algorithms::fuzz::wratio(&query, s);
                    MatchResult {
                        text: s.clone(),
                        score,
                        id: Some(i),
                    }
                })
                .collect()
        } else {
            choices
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let score = algorithms::fuzz::wratio(&query, s);
                    MatchResult {
                        text: s.clone(),
                        score,
                        id: Some(i),
                    }
                })
                .collect()
        }
    });

    results
        .into_iter()
        .filter(|r| r.score >= min_similarity)
        .max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

// ============================================================================
// Batch Processing Functions
// ============================================================================

/// Internal batch similarity computation.
///
/// Uses parallel processing for large inputs (>100 items) and sequential
/// processing for smaller inputs to avoid thread pool overhead.
fn batch_similarity_internal<F>(strings: &[String], query: &str, scorer: F) -> Vec<MatchResult>
where
    F: Fn(&str, &str) -> f64 + Send + Sync,
{
    if strings.len() >= PARALLEL_THRESHOLD {
        use rayon::prelude::*;
        strings
            .par_iter()
            .enumerate()
            .map(|(i, s)| {
                let score = scorer(s, query);
                MatchResult {
                    text: s.clone(),
                    score,
                    id: Some(i),
                }
            })
            .collect()
    } else {
        strings
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let score = scorer(s, query);
                MatchResult {
                    text: s.clone(),
                    score,
                    id: Some(i),
                }
            })
            .collect()
    }
}

/// Compute Levenshtein similarities in batch.
///
/// Uses parallel processing for large inputs (>100 items) and sequential
/// processing for smaller inputs to avoid thread pool overhead.
/// Releases the Python GIL during computation to allow other threads to run.
///
/// # Arguments
/// * `strings` - List of strings to compare
/// * `query` - Query string to compare against
/// * `normalize` - Optional normalization mode. Use "lowercase" for case-insensitive comparison.
#[pyfunction]
#[pyo3(signature = (strings, query, normalize=None))]
fn batch_levenshtein(
    py: Python<'_>,
    strings: Vec<String>,
    query: &str,
    normalize: Option<&str>,
) -> Vec<MatchResult> {
    let query = query.to_string();
    py.allow_threads(|| {
        if matches!(normalize, Some("lowercase")) {
            let query_lower = query.to_lowercase();
            batch_similarity_internal(&strings, &query_lower, |s, q| {
                algorithms::levenshtein::levenshtein_similarity(&s.to_lowercase(), q)
            })
        } else {
            batch_similarity_internal(
                &strings,
                &query,
                algorithms::levenshtein::levenshtein_similarity,
            )
        }
    })
}

/// Compute Jaro-Winkler similarities in batch.
///
/// Uses parallel processing for large inputs (>100 items) and sequential
/// processing for smaller inputs to avoid thread pool overhead.
/// Releases the Python GIL during computation to allow other threads to run.
///
/// # Arguments
/// * `strings` - List of strings to compare
/// * `query` - Query string to compare against
/// * `normalize` - Optional normalization mode. Use "lowercase" for case-insensitive comparison.
#[pyfunction]
#[pyo3(signature = (strings, query, normalize=None))]
fn batch_jaro_winkler(
    py: Python<'_>,
    strings: Vec<String>,
    query: &str,
    normalize: Option<&str>,
) -> Vec<MatchResult> {
    let query = query.to_string();
    py.allow_threads(|| {
        if matches!(normalize, Some("lowercase")) {
            let query_lower = query.to_lowercase();
            batch_similarity_internal(&strings, &query_lower, |s, q| {
                algorithms::jaro::jaro_winkler_similarity(&s.to_lowercase(), q)
            })
        } else {
            batch_similarity_internal(&strings, &query, algorithms::jaro::jaro_winkler_similarity)
        }
    })
}

/// Find best matches from a list using specified algorithm.
///
/// Uses parallel processing for large inputs (>100 items) and sequential
/// processing for smaller inputs to avoid thread pool overhead.
/// Releases the Python GIL during computation to allow other threads to run.
///
/// # Arguments
/// * `strings` - List of strings to search
/// * `query` - Query string to match against
/// * `algorithm` - Similarity algorithm to use (default: "jaro_winkler")
/// * `limit` - Maximum number of results to return
/// * `min_similarity` - Minimum similarity score to include
/// * `normalize` - Optional normalization mode. Use "lowercase" for case-insensitive comparison.
#[pyfunction]
#[pyo3(signature = (strings, query, algorithm="jaro_winkler", limit=10, min_similarity=0.0, normalize=None))]
fn find_best_matches(
    py: Python<'_>,
    strings: Vec<String>,
    query: &str,
    algorithm: &str,
    limit: usize,
    min_similarity: f64,
    normalize: Option<&str>,
) -> PyResult<Vec<MatchResult>> {
    validate_similarity(min_similarity, "min_similarity")?;

    let similarity_fn = get_similarity_fn(algorithm)?;
    let use_lowercase = matches!(normalize, Some("lowercase"));

    // Release GIL during computation
    let query = query.to_string();
    let mut results: Vec<MatchResult> = py.allow_threads(|| {
        if strings.len() >= PARALLEL_THRESHOLD {
            use rayon::prelude::*;
            strings
                .par_iter()
                .enumerate()
                .map(|(i, s)| {
                    let score = if use_lowercase {
                        similarity_fn(&s.to_lowercase(), &query.to_lowercase())
                    } else {
                        similarity_fn(s, &query)
                    };
                    MatchResult {
                        text: s.clone(),
                        score,
                        id: Some(i),
                    }
                })
                .filter(|r| r.score >= min_similarity)
                .collect()
        } else {
            strings
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let score = if use_lowercase {
                        similarity_fn(&s.to_lowercase(), &query.to_lowercase())
                    } else {
                        similarity_fn(s, &query)
                    };
                    MatchResult {
                        text: s.clone(),
                        score,
                        id: Some(i),
                    }
                })
                .filter(|r| r.score >= min_similarity)
                .collect()
        }
    });

    // Validate scores in all builds (not just debug)
    for r in &results {
        if r.score.is_nan() {
            return Err(FuzzyRustError::new_err(format!(
                "NaN score detected for text: '{}'. This indicates a bug in the similarity algorithm.",
                r.text
            )));
        }
        if !r.score.is_finite() {
            return Err(FuzzyRustError::new_err(format!(
                "Non-finite score detected for text: '{}' (score: {}). This indicates a bug in the similarity algorithm.",
                r.text, r.score
            )));
        }
        // Ensure score is in valid range [0.0, 1.0]
        if !(0.0..=1.0).contains(&r.score) {
            return Err(FuzzyRustError::new_err(format!(
                "Score out of bounds for text: '{}' (score: {}). Scores must be in range [0.0, 1.0].",
                r.text, r.score
            )));
        }
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit);

    Ok(results)
}

/// Compute similarity scores for pairs of strings in parallel.
///
/// Takes two equal-length vectors of strings and computes the similarity
/// for each corresponding pair using Rayon for parallel processing.
///
/// # Arguments
/// * `left` - First vector of strings
/// * `right` - Second vector of strings (must be same length as left)
/// * `algorithm` - Similarity algorithm to use (default: "jaro_winkler"). Supported algorithms:
///   - "levenshtein": Normalized Levenshtein similarity
///   - "jaro": Jaro similarity
///   - "jaro_winkler": Jaro-Winkler similarity (default)
///   - "ngram": Trigram (n=3) similarity
///   - "cosine": Character-level cosine similarity
/// * `normalize` - Optional normalization mode. Use "lowercase" for case-insensitive comparison.
///
/// # Returns
/// Vector of Option<f64> scores. Returns None for pairs where an unknown algorithm is specified.
///
/// # Example
/// ```python
/// >>> left = ["hello", "world"]
/// >>> right = ["hallo", "word"]
/// >>> batch_similarity_pairs(left, right)
/// [0.88, 0.75]
/// ```
#[pyfunction]
#[pyo3(signature = (left, right, algorithm="jaro_winkler", normalize=None))]
fn batch_similarity_pairs(
    py: Python<'_>,
    left: Vec<String>,
    right: Vec<String>,
    algorithm: &str,
    normalize: Option<&str>,
) -> PyResult<Vec<Option<f64>>> {
    use rayon::prelude::*;

    if left.len() != right.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Series must have equal length",
        ));
    }

    let use_lowercase = matches!(normalize, Some("lowercase"));

    let results: Vec<Option<f64>> = py.allow_threads(|| {
        left.par_iter()
            .zip(right.par_iter())
            .map(|(a, b)| {
                let (a_cmp, b_cmp) = if use_lowercase {
                    (a.to_lowercase(), b.to_lowercase())
                } else {
                    (a.clone(), b.clone())
                };
                let score = match algorithm {
                    "levenshtein" => {
                        crate::algorithms::levenshtein::levenshtein_similarity(&a_cmp, &b_cmp)
                    }
                    "jaro" => crate::algorithms::jaro::jaro_similarity(&a_cmp, &b_cmp),
                    "jaro_winkler" => {
                        crate::algorithms::jaro::jaro_winkler_similarity(&a_cmp, &b_cmp)
                    }
                    "ngram" => {
                        crate::algorithms::ngram::ngram_similarity(&a_cmp, &b_cmp, 3, true, ' ')
                    }
                    "cosine" => crate::algorithms::cosine::cosine_similarity_chars(&a_cmp, &b_cmp),
                    _ => return None,
                };
                Some(score)
            })
            .collect()
    });

    Ok(results)
}

/// Compute pairwise distance/similarity matrix between two lists of strings.
///
/// Similar to scipy.spatial.distance.cdist, this function computes the similarity
/// between every pair of strings from queries and choices, returning a 2D matrix.
///
/// Args:
///     queries: First list of strings (rows of output matrix)
///     choices: Second list of strings (columns of output matrix)
///     scorer: Algorithm to use. Options: "levenshtein", "jaro", "jaro_winkler",
///             "ngram", "cosine", "damerau" (default: "levenshtein")
///     workers: Number of parallel workers. -1 means use all available cores (default: -1)
///     normalize: Optional normalization mode. Use "lowercase" for case-insensitive comparison.
///
/// Returns:
///     2D list where result[i][j] is the similarity between queries[i] and choices[j]
///
/// # Example
/// ```python
/// >>> queries = ["hello", "world"]
/// >>> choices = ["hallo", "word", "help"]
/// >>> matrix = cdist(queries, choices, scorer="jaro_winkler")
/// >>> # matrix[0] = similarities of "hello" with each choice
/// >>> # matrix[1] = similarities of "world" with each choice
/// ```
#[pyfunction]
#[pyo3(signature = (queries, choices, scorer="levenshtein", workers=-1, normalize=None))]
fn cdist(
    py: Python<'_>,
    queries: Vec<String>,
    choices: Vec<String>,
    scorer: &str,
    workers: i32,
    normalize: Option<&str>,
) -> PyResult<Vec<Vec<f64>>> {
    use rayon::prelude::*;

    // Validate scorer before doing any work
    let scorer_fn: fn(&str, &str) -> f64 = match scorer {
        "levenshtein" => |a, b| crate::algorithms::levenshtein::levenshtein_similarity(a, b),
        "jaro" => |a, b| crate::algorithms::jaro::jaro_similarity(a, b),
        "jaro_winkler" => |a, b| crate::algorithms::jaro::jaro_winkler_similarity(a, b),
        "ngram" => |a, b| crate::algorithms::ngram::ngram_similarity(a, b, 3, true, ' '),
        "cosine" => |a, b| crate::algorithms::cosine::cosine_similarity_chars(a, b),
        "damerau" => |a, b| crate::algorithms::damerau::damerau_levenshtein_similarity(a, b),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown scorer: '{}'. Valid options: levenshtein, jaro, jaro_winkler, ngram, cosine, damerau",
                scorer
            )));
        }
    };

    let use_lowercase = matches!(normalize, Some("lowercase"));

    // Pre-process strings for case-insensitive comparison
    let queries_cmp: Vec<String> = if use_lowercase {
        queries.iter().map(|s| s.to_lowercase()).collect()
    } else {
        queries.clone()
    };
    let choices_cmp: Vec<String> = if use_lowercase {
        choices.iter().map(|s| s.to_lowercase()).collect()
    } else {
        choices.clone()
    };

    // Configure thread pool if workers specified
    let result = py.allow_threads(|| {
        if workers > 0 {
            // Use custom thread pool with specified number of workers
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers as usize)
                .build()
                .map_err(|e| format!("Failed to create thread pool: {}", e))?;

            pool.install(|| {
                Ok(queries_cmp
                    .par_iter()
                    .map(|query| {
                        choices_cmp
                            .iter()
                            .map(|choice| scorer_fn(query, choice))
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<Vec<f64>>>())
            })
        } else {
            // Use default thread pool (all cores)
            Ok(queries_cmp
                .par_iter()
                .map(|query| {
                    choices_cmp
                        .iter()
                        .map(|choice| scorer_fn(query, choice))
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>())
        }
    });

    result.map_err(|e: String| pyo3::exceptions::PyRuntimeError::new_err(e))
}

// ============================================================================
// Unified Batch Similarity
// ============================================================================

/// Compute similarity scores for all strings against a query using any algorithm.
///
/// This is a unified batch similarity function that supports all algorithms.
/// Results include an id field for tracking the original position.
///
/// # Arguments
/// * `strings` - List of strings to compare
/// * `query` - Query string to compare against
/// * `algorithm` - Similarity algorithm to use (default: "jaro_winkler")
/// * `normalize` - Optional normalization mode. Use "lowercase" for case-insensitive comparison.
///
/// # Returns
/// List of MatchResult objects in the same order as input strings.
/// Each result has text, score, and id fields.
#[pyfunction]
#[pyo3(signature = (strings, query, algorithm="jaro_winkler", normalize=None))]
fn batch_similarity(
    py: Python<'_>,
    strings: Vec<String>,
    query: &str,
    algorithm: &str,
    normalize: Option<&str>,
) -> PyResult<Vec<MatchResult>> {
    let similarity_fn = get_similarity_fn(algorithm)?;
    let query = query.to_string();
    let use_lowercase = matches!(normalize, Some("lowercase"));

    let results = py.allow_threads(|| {
        if strings.len() >= PARALLEL_THRESHOLD {
            use rayon::prelude::*;
            strings
                .par_iter()
                .enumerate()
                .map(|(i, s)| {
                    let score = if use_lowercase {
                        similarity_fn(&s.to_lowercase(), &query.to_lowercase())
                    } else {
                        similarity_fn(s, &query)
                    };
                    MatchResult {
                        text: s.clone(),
                        score,
                        id: Some(i),
                    }
                })
                .collect()
        } else {
            strings
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let score = if use_lowercase {
                        similarity_fn(&s.to_lowercase(), &query.to_lowercase())
                    } else {
                        similarity_fn(s, &query)
                    };
                    MatchResult {
                        text: s.clone(),
                        score,
                        id: Some(i),
                    }
                })
                .collect()
        }
    });

    Ok(results)
}

// ============================================================================
// Case-Insensitive Variants (Aliases for base functions with normalize="lowercase")
// ============================================================================

/// Case-insensitive Levenshtein distance.
/// Equivalent to `levenshtein(a, b, normalize="lowercase")`.
#[pyfunction]
#[pyo3(signature = (a, b, max_distance=None))]
fn levenshtein_ci(a: &str, b: &str, max_distance: Option<usize>) -> PyResult<usize> {
    levenshtein(a, b, max_distance, Some("lowercase"))
}

/// Case-insensitive Levenshtein similarity.
/// Equivalent to `levenshtein_similarity(a, b, normalize="lowercase")`.
#[pyfunction]
fn levenshtein_similarity_ci(a: &str, b: &str) -> PyResult<f64> {
    levenshtein_similarity(a, b, Some("lowercase"))
}

/// Case-insensitive Damerau-Levenshtein distance.
/// Equivalent to `damerau_levenshtein(a, b, normalize="lowercase")`.
#[pyfunction]
#[pyo3(signature = (a, b, max_distance=None))]
fn damerau_levenshtein_ci(a: &str, b: &str, max_distance: Option<usize>) -> PyResult<usize> {
    damerau_levenshtein(a, b, max_distance, Some("lowercase"))
}

/// Case-insensitive Damerau-Levenshtein similarity.
/// Equivalent to `damerau_levenshtein_similarity(a, b, normalize="lowercase")`.
#[pyfunction]
fn damerau_levenshtein_similarity_ci(a: &str, b: &str) -> PyResult<f64> {
    damerau_levenshtein_similarity(a, b, Some("lowercase"))
}

/// Case-insensitive Jaro similarity.
/// Equivalent to `jaro_similarity(a, b, normalize="lowercase")`.
#[pyfunction]
fn jaro_similarity_ci(a: &str, b: &str) -> PyResult<f64> {
    jaro_similarity(a, b, Some("lowercase"))
}

/// Case-insensitive Jaro-Winkler similarity.
/// Equivalent to `jaro_winkler_similarity(a, b, normalize="lowercase")`.
#[pyfunction]
#[pyo3(signature = (a, b, prefix_weight=0.1, max_prefix_length=4))]
fn jaro_winkler_similarity_ci(
    a: &str,
    b: &str,
    prefix_weight: f64,
    max_prefix_length: usize,
) -> PyResult<f64> {
    jaro_winkler_similarity(a, b, prefix_weight, max_prefix_length, Some("lowercase"))
}

/// Case-insensitive n-gram similarity.
/// Equivalent to `ngram_similarity(a, b, normalize="lowercase")`.
#[pyfunction]
#[pyo3(signature = (a, b, ngram_size=3, pad=true))]
fn ngram_similarity_ci(a: &str, b: &str, ngram_size: usize, pad: bool) -> PyResult<f64> {
    ngram_similarity(a, b, ngram_size, pad, Some("lowercase"))
}

/// Case-insensitive n-gram Jaccard similarity.
/// Equivalent to `ngram_jaccard(a, b, normalize="lowercase")`.
#[pyfunction]
#[pyo3(signature = (a, b, ngram_size=3, pad=true))]
fn ngram_jaccard_ci(a: &str, b: &str, ngram_size: usize, pad: bool) -> PyResult<f64> {
    ngram_jaccard(a, b, ngram_size, pad, Some("lowercase"))
}

/// Case-insensitive character-level cosine similarity.
/// Equivalent to `cosine_similarity_chars(a, b, normalize="lowercase")`.
#[pyfunction]
fn cosine_similarity_chars_ci(a: &str, b: &str) -> PyResult<f64> {
    cosine_similarity_chars(a, b, Some("lowercase"))
}

/// Case-insensitive word-level cosine similarity.
/// Equivalent to `cosine_similarity_words(a, b, normalize="lowercase")`.
#[pyfunction]
fn cosine_similarity_words_ci(a: &str, b: &str) -> PyResult<f64> {
    cosine_similarity_words(a, b, Some("lowercase"))
}

/// Case-insensitive n-gram cosine similarity.
/// Equivalent to `cosine_similarity_ngrams(a, b, normalize="lowercase")`.
#[pyfunction]
#[pyo3(signature = (a, b, ngram_size=3))]
fn cosine_similarity_ngrams_ci(a: &str, b: &str, ngram_size: usize) -> PyResult<f64> {
    cosine_similarity_ngrams(a, b, ngram_size, Some("lowercase"))
}

/// Case-insensitive bigram similarity.
/// Equivalent to `ngram_similarity(a, b, ngram_size=2, normalize="lowercase")`.
#[pyfunction]
fn bigram_similarity_ci(a: &str, b: &str) -> PyResult<f64> {
    ngram_similarity(a, b, 2, true, Some("lowercase"))
}

/// Case-insensitive trigram similarity.
/// Equivalent to `ngram_similarity(a, b, ngram_size=3, normalize="lowercase")`.
#[pyfunction]
fn trigram_similarity_ci(a: &str, b: &str) -> PyResult<f64> {
    ngram_similarity(a, b, 3, true, Some("lowercase"))
}

/// Case-insensitive LCS similarity.
/// Equivalent to `lcs_similarity(a.lower(), b.lower())`.
#[pyfunction]
fn lcs_similarity_ci(a: &str, b: &str) -> f64 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    algorithms::lcs::lcs_similarity(&a_lower, &b_lower)
}

/// Case-insensitive Soundex similarity.
/// Equivalent to `soundex_similarity(a.lower(), b.lower())`.
#[pyfunction]
fn soundex_similarity_ci(a: &str, b: &str) -> f64 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    algorithms::phonetic::soundex_similarity(&a_lower, &b_lower)
}

/// Case-insensitive Metaphone similarity.
/// Equivalent to `metaphone_similarity(a.lower(), b.lower())`.
#[pyfunction]
#[pyo3(signature = (a, b, max_length=4))]
fn metaphone_similarity_ci(a: &str, b: &str, max_length: usize) -> f64 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    algorithms::phonetic::metaphone_similarity(&a_lower, &b_lower, max_length)
}

/// Case-insensitive Hamming distance.
///
/// Converts both strings to lowercase before computing Hamming distance.
/// Strings must have equal length.
#[pyfunction]
fn hamming_ci(a: &str, b: &str) -> PyResult<usize> {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    algorithms::hamming::hamming_distance(&a_lower, &b_lower).ok_or_else(|| {
        ValidationError::new_err("Strings must have equal length for Hamming distance")
    })
}

/// Case-insensitive Hamming similarity (0.0 to 1.0).
///
/// Converts both strings to lowercase before computing Hamming similarity.
/// Strings must have equal length.
#[pyfunction]
fn hamming_similarity_ci(a: &str, b: &str) -> PyResult<f64> {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    algorithms::hamming::hamming_similarity(&a_lower, &b_lower).ok_or_else(|| {
        ValidationError::new_err("Strings must have equal length for Hamming similarity")
    })
}

/// Case-insensitive LCS length.
/// Equivalent to `lcs_length(a.lower(), b.lower())`.
#[pyfunction]
fn lcs_length_ci(a: &str, b: &str) -> usize {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    algorithms::lcs::lcs_length(&a_lower, &b_lower)
}

/// Case-insensitive LCS string.
///
/// Returns the actual Longest Common Subsequence string after converting
/// both inputs to lowercase.
///
/// Note: For strings longer than 10,000 characters, this raises a ValidationError
/// to prevent excessive memory allocation (O(m*n) space required).
#[pyfunction]
fn lcs_string_ci(a: &str, b: &str) -> PyResult<String> {
    const MAX_LEN: usize = 10_000;
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    if a_lower.chars().count() > MAX_LEN || b_lower.chars().count() > MAX_LEN {
        return Err(ValidationError::new_err(format!(
            "Input strings exceed maximum length of {} characters for lcs_string_ci. \
             Use lcs_length_ci for a space-efficient alternative.",
            MAX_LEN
        )));
    }
    Ok(algorithms::lcs::lcs_string(&a_lower, &b_lower))
}

/// Case-insensitive longest common substring.
///
/// Returns the longest common contiguous substring after converting
/// both inputs to lowercase.
///
/// Note: For strings longer than 10,000 characters, this raises a ValidationError
/// to prevent excessive memory allocation (O(m*n) space required).
#[pyfunction]
fn longest_common_substring_ci(a: &str, b: &str) -> PyResult<String> {
    const MAX_LEN: usize = 10_000;
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    if a_lower.chars().count() > MAX_LEN || b_lower.chars().count() > MAX_LEN {
        return Err(ValidationError::new_err(format!(
            "Input strings exceed maximum length of {} characters for longest_common_substring_ci. \
             Use longest_common_substring_length for a space-efficient alternative.",
            MAX_LEN
        )));
    }
    Ok(algorithms::lcs::longest_common_substring(
        &a_lower, &b_lower,
    ))
}

// ============================================================================
// String Normalization
// ============================================================================

#[pyfunction]
fn normalize_string(s: &str, mode: &str) -> PyResult<String> {
    let norm_mode = parse_normalization_mode(mode)?;
    Ok(algorithms::normalize::normalize_string(s, norm_mode))
}

#[pyfunction]
fn normalize_pair(a: &str, b: &str, mode: &str) -> PyResult<(String, String)> {
    let norm_mode = parse_normalization_mode(mode)?;
    Ok(algorithms::normalize::normalize_pair(a, b, norm_mode))
}

// ============================================================================
// Deduplication
// ============================================================================

/// Find duplicate items in a list using the specified similarity algorithm.
///
/// # Arguments
/// * `items` - List of strings to deduplicate
/// * `algorithm` - Similarity algorithm to use (default: "jaro_winkler")
/// * `min_similarity` - Minimum similarity score to consider items as duplicates (0.0 to 1.0, default: 0.85)
/// * `normalize` - Normalization mode to apply before comparison:
///   - `None` or `"none"`: No normalization
///   - `true` or `"lowercase"`: Convert to lowercase (default)
///   - `"unicode_nfkd"` or `"nfkd"`: Apply Unicode NFKD normalization
///   - `"remove_punctuation"`: Remove ASCII punctuation
///   - `"remove_whitespace"`: Remove all whitespace
///   - `"strict"`: Apply all normalizations (NFKD + lowercase + remove punctuation + remove whitespace)
/// * `method` - Deduplication method: "auto", "brute_force", or "snm" (default: "auto")
/// * `window_size` - Window size for "snm" method (default: 50)
///
/// # Returns
/// DeduplicationResult containing groups of duplicates and unique items
///
/// # Warning
/// For datasets with many similar items, memory usage can be high as all
/// similar pairs are collected. For N items with high similarity, up to O(N^2) pairs
/// may be generated. Consider using smaller batches for datasets > 100K items.
#[pyfunction]
#[pyo3(signature = (items, algorithm="jaro_winkler", min_similarity=0.85, normalize="lowercase", method="auto", window_size=50))]
fn find_duplicates(
    items: Vec<String>,
    algorithm: &str,
    min_similarity: f64,
    normalize: &str,
    method: &str,
    window_size: usize,
) -> PyResult<DeduplicationResult> {
    validate_similarity(min_similarity, "min_similarity")?;

    // Determine deduplication method
    let dedup_method = match method.to_lowercase().as_str() {
        "brute_force" => dedup::DedupMethod::BruteForce,
        "snm" | "sorted_neighborhood" => dedup::DedupMethod::SortedNeighborhood { window_size },
        "auto" => {
            if items.len() < 2000 {
                dedup::DedupMethod::BruteForce
            } else {
                dedup::DedupMethod::SortedNeighborhood { window_size }
            }
        }
        _ => {
            return Err(ValidationError::new_err(format!(
                "Unknown deduplication method: '{}'. Valid: auto, brute_force, snm",
                method
            )))
        }
    };

    // Parse normalization mode
    let norm_mode = match normalize.to_lowercase().as_str() {
        "none" | "false" => None,
        "true" | "lowercase" => Some(NormalizationMode::Lowercase),
        "unicode_nfkd" | "nfkd" => Some(NormalizationMode::UnicodeNFKD),
        "remove_punctuation" => Some(NormalizationMode::RemovePunctuation),
        "remove_whitespace" => Some(NormalizationMode::RemoveWhitespace),
        "strict" => Some(NormalizationMode::Strict),
        _ => return Err(ValidationError::new_err(format!(
            "Unknown normalization mode: '{}'. Valid: none, lowercase, unicode_nfkd, remove_punctuation, remove_whitespace, strict",
            normalize
        ))),
    };

    // Normalize items if requested
    let (processed_items, original_items): (Vec<String>, Option<Vec<String>>) = match norm_mode {
        Some(mode) => {
            let processed = items
                .iter()
                .map(|s| algorithms::normalize::normalize_string(s, mode))
                .collect();
            (processed, Some(items))
        }
        None => (items, None),
    };

    // Choose similarity function based on algorithm
    let metric = get_similarity_metric(algorithm)?;
    let result = dedup::find_duplicates_with_metric(
        &processed_items,
        metric.as_ref(),
        min_similarity,
        dedup_method,
    );

    // If we normalized, map back to original strings
    let final_result = if let Some(items) = original_items {
        // Build index map: normalized string -> list of original indices
        // This gives O(n) preprocessing instead of O(n²) per-group lookup
        use std::collections::HashMap;
        let mut index_map: HashMap<&String, Vec<usize>> = HashMap::new();
        for (i, proc_item) in processed_items.iter().enumerate() {
            index_map.entry(proc_item).or_default().push(i);
        }

        // Map groups back to original strings using the index map
        let original_groups: Vec<Vec<String>> = result
            .groups
            .iter()
            .map(|group| {
                // Collect all original indices for items in this group
                let mut indices = Vec::new();
                for group_item in group {
                    if let Some(item_indices) = index_map.get(group_item) {
                        indices.extend(item_indices.iter().copied());
                    }
                }
                // Map indices to original strings
                indices.iter().map(|&i| items[i].clone()).collect()
            })
            .collect();

        // Map unique items back to original using the same index map
        let original_unique: Vec<String> = result
            .unique
            .iter()
            .flat_map(|unique_item| {
                index_map
                    .get(unique_item)
                    .map(|indices| {
                        indices
                            .iter()
                            .map(|&i| items[i].clone())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
            })
            .collect();

        DeduplicationResult {
            groups: original_groups,
            unique: original_unique,
            total_duplicates: result.total_duplicates,
        }
    } else {
        DeduplicationResult {
            groups: result.groups,
            unique: result.unique,
            total_duplicates: result.total_duplicates,
        }
    };

    Ok(final_result)
}

/// Find duplicate pairs using Sorted Neighborhood Method (SNM).
///
/// Returns pairs of similar items with their indices and similarity scores.
/// This is more efficient than brute force for large datasets (O(N log N) vs O(N^2))
/// and returns results in a format suitable for DataFrame integration.
///
/// # Arguments
/// * `items` - List of strings to check for duplicates
/// * `algorithm` - Similarity algorithm (default: "jaro_winkler")
/// * `min_similarity` - Minimum similarity threshold (0.0 to 1.0)
/// * `window_size` - Window size for SNM comparison (default: 50)
/// * `normalize` - Normalization mode to apply before comparison:
///   - `"none"`: No normalization
///   - `"lowercase"`: Convert to lowercase (default)
///   - `"unicode_nfkd"` or `"nfkd"`: Apply Unicode NFKD normalization
///   - `"remove_punctuation"`: Remove ASCII punctuation
///   - `"remove_whitespace"`: Remove all whitespace
///   - `"strict"`: Apply all normalizations
///
/// # Returns
/// List of (idx_a, idx_b, score) tuples for each pair of similar items
///
/// # Warning
/// For datasets with many similar items, memory usage can be high as all
/// similar pairs are collected. For N items with high similarity, up to O(N^2) pairs
/// may be generated. Consider using smaller batches for datasets > 100K items.
#[pyfunction]
#[pyo3(signature = (items, algorithm="jaro_winkler", min_similarity=0.85, window_size=50, normalize="lowercase"))]
fn find_duplicate_pairs(
    items: Vec<String>,
    algorithm: &str,
    min_similarity: f64,
    window_size: usize,
    normalize: &str,
) -> PyResult<Vec<(usize, usize, f64)>> {
    validate_similarity(min_similarity, "min_similarity")?;

    if items.is_empty() {
        return Ok(vec![]);
    }

    // Parse normalization mode
    let norm_mode = match normalize.to_lowercase().as_str() {
        "none" | "false" => None,
        "true" | "lowercase" => Some(NormalizationMode::Lowercase),
        "unicode_nfkd" | "nfkd" => Some(NormalizationMode::UnicodeNFKD),
        "remove_punctuation" => Some(NormalizationMode::RemovePunctuation),
        "remove_whitespace" => Some(NormalizationMode::RemoveWhitespace),
        "strict" => Some(NormalizationMode::Strict),
        _ => return Err(ValidationError::new_err(format!(
            "Unknown normalization mode: '{}'. Valid: none, lowercase, unicode_nfkd, remove_punctuation, remove_whitespace, strict",
            normalize
        ))),
    };

    // Normalize items if requested
    let processed_items: Vec<String> = match norm_mode {
        Some(mode) => items
            .iter()
            .map(|s| algorithms::normalize::normalize_string(s, mode))
            .collect(),
        None => items.clone(),
    };

    let metric = get_similarity_metric(algorithm)?;

    // Create sorted indices based on processed (normalized) items
    let mut indices: Vec<usize> = (0..processed_items.len()).collect();
    indices.sort_by(|&a, &b| processed_items[a].cmp(&processed_items[b]));

    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();

    // Compare within window using sorted order
    for i in 0..indices.len() {
        let idx_i = indices[i];
        let end = (i + window_size).min(indices.len());

        for &idx_j in &indices[(i + 1)..end] {
            let score = metric.similarity(&processed_items[idx_i], &processed_items[idx_j]);

            if score >= min_similarity {
                // Always store with smaller index first for consistency
                let (a, b) = if idx_i < idx_j {
                    (idx_i, idx_j)
                } else {
                    (idx_j, idx_i)
                };
                pairs.push((a, b, score));
            }
        }
    }

    Ok(pairs)
}

// ============================================================================
// Evaluation Metrics
// ============================================================================

/// Compute precision: TP / (TP + FP).
///
/// Precision measures the accuracy of positive predictions.
/// A precision of 1.0 means no false positives.
///
/// # Arguments
/// * `true_matches` - List of actual match pairs (ground truth) as (id1, id2) tuples
/// * `predicted_matches` - List of predicted match pairs as (id1, id2) tuples
///
/// # Returns
/// Precision score between 0.0 and 1.0
#[pyfunction]
fn precision(true_matches: Vec<(usize, usize)>, predicted_matches: Vec<(usize, usize)>) -> f64 {
    let true_set = metrics::normalize_pairs(&true_matches);
    let pred_set = metrics::normalize_pairs(&predicted_matches);
    metrics::precision(&true_set, &pred_set)
}

/// Compute recall: TP / (TP + FN).
///
/// Recall measures the completeness of positive predictions.
/// A recall of 1.0 means no false negatives (all true matches found).
///
/// # Arguments
/// * `true_matches` - List of actual match pairs (ground truth) as (id1, id2) tuples
/// * `predicted_matches` - List of predicted match pairs as (id1, id2) tuples
///
/// # Returns
/// Recall score between 0.0 and 1.0
#[pyfunction]
fn recall(true_matches: Vec<(usize, usize)>, predicted_matches: Vec<(usize, usize)>) -> f64 {
    let true_set = metrics::normalize_pairs(&true_matches);
    let pred_set = metrics::normalize_pairs(&predicted_matches);
    metrics::recall(&true_set, &pred_set)
}

/// Compute F-beta score: weighted harmonic mean of precision and recall.
///
/// F1 score (beta=1.0) gives equal weight to precision and recall.
/// F0.5 (beta=0.5) weighs precision higher than recall.
/// F2 (beta=2.0) weighs recall higher than precision.
///
/// # Arguments
/// * `true_matches` - List of actual match pairs (ground truth) as (id1, id2) tuples
/// * `predicted_matches` - List of predicted match pairs as (id1, id2) tuples
/// * `beta` - Weight parameter (default 1.0 for F1 score)
///
/// # Returns
/// F-beta score between 0.0 and 1.0
#[pyfunction]
#[pyo3(signature = (true_matches, predicted_matches, beta=1.0))]
fn f_score(
    true_matches: Vec<(usize, usize)>,
    predicted_matches: Vec<(usize, usize)>,
    beta: f64,
) -> PyResult<f64> {
    if beta < 0.0 {
        return Err(ValidationError::new_err(format!(
            "beta must be non-negative, got {}",
            beta
        )));
    }
    let true_set = metrics::normalize_pairs(&true_matches);
    let pred_set = metrics::normalize_pairs(&predicted_matches);
    Ok(metrics::f_score(&true_set, &pred_set, beta))
}

/// Compute confusion matrix from match sets.
///
/// # Arguments
/// * `true_matches` - List of actual match pairs (ground truth) as (id1, id2) tuples
/// * `predicted_matches` - List of predicted match pairs as (id1, id2) tuples
/// * `total_pairs` - Total number of possible pairs (for computing TN)
///
/// # Returns
/// ConfusionMatrixResult with tp, fp, fn_count, tn counts and methods for
/// precision(), recall(), and f_score(beta).
#[pyfunction]
fn confusion_matrix(
    true_matches: Vec<(usize, usize)>,
    predicted_matches: Vec<(usize, usize)>,
    total_pairs: usize,
) -> ConfusionMatrixResult {
    let true_set = metrics::normalize_pairs(&true_matches);
    let pred_set = metrics::normalize_pairs(&predicted_matches);
    let cm = metrics::confusion_matrix(&true_set, &pred_set, total_pairs);
    ConfusionMatrixResult {
        tp: cm.true_positives,
        fp: cm.false_positives,
        fn_count: cm.false_negatives,
        tn: cm.true_negatives,
    }
}

// ============================================================================
// Multi-Algorithm Comparison
// ============================================================================

/// Compare query against strings using multiple similarity algorithms.
///
/// # Arguments
/// * `strings` - List of strings to compare against
/// * `query` - Query string to find matches for
/// * `algorithms` - List of algorithm names to use (if None, uses all common algorithms)
/// * `limit` - Maximum number of top matches to return per algorithm (default: 3)
///
/// # Returns
/// List of AlgorithmComparison objects, one for each algorithm, sorted by average score (highest first).
///
/// Releases the Python GIL during computation to allow other threads to run.
#[pyfunction]
#[pyo3(signature = (strings, query, algorithms=None, limit=3))]
fn compare_algorithms(
    py: Python<'_>,
    strings: Vec<String>,
    query: &str,
    algorithms: Option<Vec<String>>,
    limit: usize,
) -> Vec<AlgorithmComparison> {
    use rayon::prelude::*;

    // Default algorithms if none specified
    let algos = algorithms.unwrap_or_else(|| {
        vec![
            "levenshtein".to_string(),
            "damerau_levenshtein".to_string(),
            "jaro".to_string(),
            "jaro_winkler".to_string(),
            "ngram".to_string(),
            "lcs".to_string(),
        ]
    });

    // Build metrics outside py.allow_threads (requires PyResult handling)
    let metrics: Vec<(String, Box<dyn algorithms::Similarity + Send + Sync>)> = algos
        .iter()
        .filter_map(|algo| get_similarity_metric(algo).ok().map(|m| (algo.clone(), m)))
        .collect();

    // Release GIL during parallel computation
    let query = query.to_string();
    let mut results: Vec<AlgorithmComparison> = py.allow_threads(|| {
        metrics
            .par_iter()
            .map(|(algo, metric)| {
                // Compute similarity for all strings
                let mut scored: Vec<(String, f64)> = strings
                    .iter()
                    .map(|s| {
                        let score = metric.similarity(&query, s);
                        (s.clone(), score)
                    })
                    .collect();

                // Sort by score descending
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Take top N matches
                let top_matches: Vec<MatchResult> = scored
                    .iter()
                    .enumerate()
                    .take(limit)
                    .map(|(i, (text, score))| MatchResult {
                        text: text.clone(),
                        score: *score,
                        id: Some(i),
                    })
                    .collect();

                // Compute average score across all comparisons
                let avg_score = if !scored.is_empty() {
                    scored.iter().map(|(_, s)| s).sum::<f64>() / scored.len() as f64
                } else {
                    0.0
                };

                AlgorithmComparison {
                    algorithm: algo.clone(),
                    score: avg_score,
                    matches: top_matches,
                }
            })
            .collect()
    });

    // Sort by average score (highest first)
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

// ============================================================================
// Python Index Classes
// ============================================================================

// ============================================================================
// TF-IDF Cosine Similarity
// ============================================================================

/// TF-IDF weighted cosine similarity for corpus-based matching.
///
/// Builds a corpus of documents and uses TF-IDF weighting for similarity.
/// Words that appear in fewer documents get higher weight, improving matching
/// for domain-specific or rare terms.
///
/// Example:
///     tfidf = fr.TfIdfCosine()
///     tfidf.add_documents(["hello world", "hello there", "world news"])
///     score = tfidf.similarity("hello world", "hello there")
#[pyclass(name = "TfIdfCosine")]
struct PyTfIdfCosine {
    inner: algorithms::cosine::TfIdfCosine,
}

#[pymethods]
impl PyTfIdfCosine {
    #[new]
    fn new() -> Self {
        Self {
            inner: algorithms::cosine::TfIdfCosine::new(),
        }
    }

    /// Add a document to build IDF scores.
    fn add_document(&mut self, doc: &str) {
        self.inner.add_document(doc);
    }

    /// Add multiple documents to build IDF scores.
    fn add_documents(&mut self, docs: Vec<String>) {
        for doc in docs {
            self.inner.add_document(&doc);
        }
    }

    /// Calculate TF-IDF weighted cosine similarity between two strings.
    fn similarity(&self, a: &str, b: &str) -> f64 {
        self.inner.similarity(a, b)
    }

    /// Get the number of documents in the corpus.
    fn num_documents(&self) -> usize {
        self.inner.num_documents()
    }
}

/// Python wrapper for BK-tree.
///
/// Note: This class is NOT thread-safe and cannot be shared between Python
/// threads. Create separate instances for each thread or use appropriate
/// synchronization in your Python code.
///
/// Note: Duplicate strings are silently ignored. Adding the same string twice
/// will not increase the tree size or update any associated data.
#[pyclass]
struct PyBkTree {
    inner: indexing::bktree::BkTree,
}

#[pymethods]
impl PyBkTree {
    #[new]
    #[pyo3(signature = (use_damerau=false))]
    fn new(use_damerau: bool) -> Self {
        let inner = if use_damerau {
            indexing::bktree::BkTree::with_damerau()
        } else {
            indexing::bktree::BkTree::new()
        };
        Self { inner }
    }

    /// Add a string to the tree.
    /// Returns true if added successfully, false if duplicate or tree depth exceeded.
    fn add(&mut self, text: String) -> bool {
        self.inner.add(text)
    }

    /// Add a string with associated data.
    /// Returns true if added successfully, false if duplicate or tree depth exceeded.
    fn add_with_data(&mut self, text: String, data: u64) -> bool {
        self.inner.add_with_data(text, Some(data))
    }

    /// Add multiple strings.
    fn add_all(&mut self, texts: Vec<String>) {
        self.inner.add_all(texts);
    }

    /// Search for strings within a given edit distance or above a similarity threshold.
    ///
    /// # Arguments
    /// * `query` - The string to search for
    /// * `max_distance` - Maximum edit distance (mutually exclusive with min_similarity)
    /// * `min_similarity` - Minimum similarity threshold 0.0-1.0 (converted to max_distance)
    /// * `limit` - Maximum number of results to return
    /// * `normalize` - Optional normalization mode. Use "lowercase" for case-insensitive comparison.
    ///
    /// # Returns
    /// List of SearchResult objects sorted by distance/similarity.
    #[pyo3(signature = (query, max_distance=None, min_similarity=None, limit=None, normalize=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        max_distance: Option<usize>,
        min_similarity: Option<f64>,
        limit: Option<usize>,
        normalize: Option<&str>,
    ) -> PyResult<Vec<SearchResult>> {
        // Determine max_distance from either parameter
        let effective_max_distance = match (max_distance, min_similarity) {
            (Some(d), None) => d,
            (None, Some(sim)) => {
                if !(0.0..=1.0).contains(&sim) {
                    return Err(ValidationError::new_err(
                        "min_similarity must be between 0.0 and 1.0",
                    ));
                }
                // Convert similarity to max_distance based on query length
                // similarity = 1 - (distance / max_len), so distance = (1 - similarity) * max_len
                let query_len = query.chars().count();
                ((1.0 - sim) * query_len as f64).ceil() as usize
            }
            (Some(_), Some(_)) => {
                return Err(ValidationError::new_err(
                    "Cannot specify both max_distance and min_similarity",
                ));
            }
            (None, None) => {
                return Err(ValidationError::new_err(
                    "Must specify either max_distance or min_similarity",
                ));
            }
        };

        let use_lowercase = matches!(normalize, Some("lowercase"));

        // Release GIL during search
        let query_owned = query.to_string();
        let results = py.allow_threads(|| {
            if use_lowercase {
                let query_lower = query_owned.to_lowercase();
                // For case-insensitive search, we need to search and compare lowercase
                // This is a workaround since the underlying BkTree doesn't support case-insensitive natively
                self.inner
                    .search(&query_lower, effective_max_distance)
                    .into_iter()
                    .filter(|r| {
                        let dist = algorithms::levenshtein::levenshtein(
                            &r.text.to_lowercase(),
                            &query_lower,
                        );
                        dist <= effective_max_distance
                    })
                    .collect::<Vec<_>>()
            } else {
                self.inner.search(&query_owned, effective_max_distance)
            }
        });

        let mut converted: Vec<SearchResult> = results
            .into_iter()
            .map(|r| convert_bktree_result(r, &query_owned))
            .collect();

        // Sort by distance (ascending) for consistent results
        converted.sort_by(|a, b| a.distance.cmp(&b.distance));

        // Apply limit if specified
        if let Some(lim) = limit {
            converted.truncate(lim);
        }

        Ok(converted)
    }

    /// Search for multiple queries in parallel.
    ///
    /// # Arguments
    /// * `queries` - List of query strings
    /// * `max_distance` - Maximum edit distance
    /// * `limit` - Maximum number of results per query
    /// * `normalize` - Optional normalization mode. Use "lowercase" for case-insensitive comparison.
    ///
    /// # Returns
    /// List of result lists, one per query.
    #[pyo3(signature = (queries, max_distance, limit=None, normalize=None))]
    fn batch_search(
        &self,
        py: Python<'_>,
        queries: Vec<String>,
        max_distance: usize,
        limit: Option<usize>,
        normalize: Option<&str>,
    ) -> Vec<Vec<SearchResult>> {
        use rayon::prelude::*;

        let use_lowercase = matches!(normalize, Some("lowercase"));

        py.allow_threads(|| {
            queries
                .par_iter()
                .map(|query| {
                    let search_query = if use_lowercase {
                        query.to_lowercase()
                    } else {
                        query.clone()
                    };

                    let results = self.inner.search(&search_query, max_distance);

                    let mut converted: Vec<SearchResult> = if use_lowercase {
                        results
                            .into_iter()
                            .filter(|r| {
                                let dist = algorithms::levenshtein::levenshtein(
                                    &r.text.to_lowercase(),
                                    &search_query,
                                );
                                dist <= max_distance
                            })
                            .map(|r| convert_bktree_result(r, query))
                            .collect()
                    } else {
                        results
                            .into_iter()
                            .map(|r| convert_bktree_result(r, query))
                            .collect()
                    };

                    // Sort by distance
                    converted.sort_by(|a, b| a.distance.cmp(&b.distance));

                    // Apply limit if specified
                    if let Some(lim) = limit {
                        converted.truncate(lim);
                    }

                    converted
                })
                .collect()
        })
    }

    /// Find the nearest neighbors to the query.
    ///
    /// # Arguments
    /// * `query` - The string to search for
    /// * `limit` - Number of nearest neighbors to return
    #[pyo3(signature = (query, limit))]
    fn find_nearest(
        &self,
        py: Python<'_>,
        query: &str,
        limit: usize,
    ) -> PyResult<Vec<SearchResult>> {
        let effective_limit = limit;

        let query = query.to_string();
        let results = py.allow_threads(|| self.inner.find_nearest(&query, effective_limit));

        Ok(results
            .into_iter()
            .map(|r| convert_bktree_result(r, &query))
            .collect())
    }

    /// Parallel search for large trees (>10K nodes).
    ///
    /// For trees with more than 10,000 nodes, this method partitions the search
    /// across multiple CPU cores for improved performance (typically 2-4x speedup).
    /// For smaller trees, automatically falls back to sequential search.
    ///
    /// # Arguments
    /// * `query` - The string to search for
    /// * `max_distance` - Maximum edit distance (mutually exclusive with min_similarity)
    /// * `min_similarity` - Minimum similarity threshold 0.0-1.0 (converted to max_distance)
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// List of SearchResult objects sorted by distance/similarity.
    #[pyo3(signature = (query, max_distance=None, min_similarity=None, limit=None))]
    fn search_parallel(
        &self,
        py: Python<'_>,
        query: &str,
        max_distance: Option<usize>,
        min_similarity: Option<f64>,
        limit: Option<usize>,
    ) -> PyResult<Vec<SearchResult>> {
        // Determine max_distance from either parameter
        let effective_max_distance = match (max_distance, min_similarity) {
            (Some(d), None) => d,
            (None, Some(sim)) => {
                if !(0.0..=1.0).contains(&sim) {
                    return Err(ValidationError::new_err(
                        "min_similarity must be between 0.0 and 1.0",
                    ));
                }
                // Convert similarity to max_distance based on query length
                let query_len = query.chars().count();
                ((1.0 - sim) * query_len as f64).ceil() as usize
            }
            (Some(_), Some(_)) => {
                return Err(ValidationError::new_err(
                    "Cannot specify both max_distance and min_similarity",
                ));
            }
            (None, None) => {
                return Err(ValidationError::new_err(
                    "Must specify either max_distance or min_similarity",
                ));
            }
        };

        let query_owned = query.to_string();
        let results = py.allow_threads(|| {
            self.inner
                .search_parallel(&query_owned, effective_max_distance, limit)
        });

        Ok(results
            .into_iter()
            .map(|r| convert_bktree_result(r, &query_owned))
            .collect())
    }

    /// Find the nearest neighbors using parallel search for large trees.
    ///
    /// Similar to `find_nearest`, but uses parallel processing for trees with
    /// more than 10,000 nodes. Falls back to sequential for smaller trees.
    ///
    /// # Arguments
    /// * `query` - The string to search for
    /// * `limit` - Number of nearest neighbors to return
    #[pyo3(signature = (query, limit))]
    fn find_nearest_parallel(
        &self,
        py: Python<'_>,
        query: &str,
        limit: usize,
    ) -> PyResult<Vec<SearchResult>> {
        let query = query.to_string();
        let results = py.allow_threads(|| self.inner.find_nearest_parallel(&query, limit));

        Ok(results
            .into_iter()
            .map(|r| convert_bktree_result(r, &query))
            .collect())
    }

    /// Check if the tree contains an exact match.
    fn contains(&self, query: &str) -> bool {
        self.inner.contains(query)
    }

    /// Get the number of items in the tree.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    // =========================================================================
    // Serialization
    // =========================================================================

    /// Serialize the tree to bytes.
    ///
    /// Note: The distance function is not serialized. When deserializing,
    /// the tree will use the default Levenshtein distance.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self.inner.to_bytes().map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Serialization failed: {}", e))
        })?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Deserialize a tree from bytes.
    ///
    /// Note: The deserialized tree will use Levenshtein distance by default.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        indexing::bktree::BkTree::from_bytes(data)
            .map(|inner| Self { inner })
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Deserialization failed: {}", e))
            })
    }

    /// Deserialize a tree from bytes using Damerau-Levenshtein distance.
    #[staticmethod]
    fn from_bytes_damerau(data: &[u8]) -> PyResult<Self> {
        use std::sync::Arc;
        indexing::bktree::BkTree::from_bytes_with_distance(
            data,
            Arc::new(|a: &str, b: &str| algorithms::damerau::damerau_levenshtein(a, b)),
        )
        .map(|inner| Self { inner })
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Deserialization failed: {}", e)))
    }

    // =========================================================================
    // Deletion
    // =========================================================================

    /// Remove an item by its ID (soft delete using tombstone).
    ///
    /// The item is marked as deleted but remains in the tree structure.
    /// Use `compact()` to rebuild the tree without deleted items.
    ///
    /// Returns True if the item was found and deleted, False otherwise.
    fn remove(&mut self, id: usize) -> bool {
        self.inner.remove(id)
    }

    /// Remove an item by its text (soft delete using tombstone).
    ///
    /// Returns True if the item was found and deleted, False otherwise.
    fn remove_text(&mut self, text: &str) -> bool {
        self.inner.remove_text(text)
    }

    /// Get the number of deleted (tombstoned) items.
    fn deleted_count(&self) -> usize {
        self.inner.deleted_count()
    }

    /// Get the number of active (non-deleted) items.
    fn active_count(&self) -> usize {
        self.inner.active_count()
    }

    /// Rebuild the tree without deleted items.
    ///
    /// This reclaims memory and improves search performance.
    /// Note: Item IDs will be reassigned during compaction.
    fn compact(&mut self) {
        self.inner.compact()
    }
}

/// Python wrapper for N-gram index.
///
/// Note: This class is NOT thread-safe and cannot be shared between Python
/// threads. Create separate instances for each thread or use appropriate
/// synchronization in your Python code.
///
/// Note: Unlike BkTree, this index allows duplicate strings. Adding the same
/// string twice will create two separate entries with different IDs.
#[pyclass]
struct PyNgramIndex {
    inner: indexing::ngram_index::NgramIndex,
}

#[pymethods]
impl PyNgramIndex {
    #[new]
    #[pyo3(signature = (ngram_size=3, min_ngram_ratio=0.2, normalize=true))]
    fn new(ngram_size: usize, min_ngram_ratio: f64, normalize: bool) -> PyResult<Self> {
        if ngram_size < 2 {
            return Err(ValidationError::new_err(format!(
                "ngram_size must be >= 2, got {}",
                ngram_size
            )));
        }
        validate_ngram_ratio(min_ngram_ratio)?;
        Ok(Self {
            inner: indexing::ngram_index::NgramIndex::with_params(
                ngram_size,
                0.0,
                min_ngram_ratio,
                normalize,
            ),
        })
    }

    /// Add a string to the index.
    fn add(&mut self, text: String) -> usize {
        self.inner.add(text)
    }

    /// Add a string with associated data.
    fn add_with_data(&mut self, text: String, data: u64) -> usize {
        self.inner.add_with_data(text, Some(data))
    }

    /// Add multiple strings.
    fn add_all(&mut self, texts: Vec<String>) {
        self.inner.add_all(texts);
    }

    /// Add multiple strings using parallel processing.
    ///
    /// This method extracts n-grams in parallel, which is significantly faster
    /// for large batches (1000+ items).
    fn add_all_parallel(&mut self, texts: Vec<String>) {
        self.inner.add_all_parallel(texts);
    }

    /// Search with similarity scoring.
    ///
    /// # Arguments
    /// * `query` - Query string
    /// * `algorithm` - Similarity algorithm ("jaro_winkler", "jaro", "levenshtein", "ngram", "trigram")
    /// * `min_similarity` - Minimum similarity threshold (0.0 to 1.0)
    /// * `limit` - Maximum results to return
    /// * `normalize` - Optional normalization mode. Use "lowercase" for case-insensitive comparison.
    ///                Default is "lowercase" for backwards compatibility.
    #[pyo3(signature = (query, algorithm="jaro_winkler", min_similarity=0.0, limit=None, normalize=Some("lowercase")))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        algorithm: &str,
        min_similarity: f64,
        limit: Option<usize>,
        normalize: Option<&str>,
    ) -> PyResult<Vec<SearchResult>> {
        validate_similarity(min_similarity, "min_similarity")?;

        let metric = get_similarity_metric_with_normalize(algorithm, normalize)?;

        let query = query.to_string();
        let results = py.allow_threads(|| {
            self.inner
                .search_parallel(&query, metric.as_ref(), min_similarity, limit)
        });

        Ok(results.into_iter().map(convert_search_match).collect())
    }

    /// Batch search for multiple queries.
    ///
    /// # Arguments
    /// * `queries` - List of query strings
    /// * `algorithm` - Similarity algorithm to use
    /// * `min_similarity` - Minimum similarity threshold
    /// * `limit` - Maximum results per query
    /// * `normalize` - Optional normalization mode. Use "lowercase" for case-insensitive comparison.
    ///                Default is "lowercase" for backwards compatibility.
    #[pyo3(signature = (queries, algorithm="jaro_winkler", min_similarity=0.0, limit=None, normalize=Some("lowercase")))]
    fn batch_search(
        &self,
        py: Python<'_>,
        queries: Vec<String>,
        algorithm: &str,
        min_similarity: f64,
        limit: Option<usize>,
        normalize: Option<&str>,
    ) -> PyResult<Vec<Vec<SearchResult>>> {
        let metric = get_similarity_metric_with_normalize(algorithm, normalize)?;

        let results = py.allow_threads(|| {
            self.inner
                .batch_search(&queries, metric.as_ref(), min_similarity, limit)
        });

        Ok(results
            .into_iter()
            .map(|matches| matches.into_iter().map(convert_search_match).collect())
            .collect())
    }

    /// Get candidates that share n-grams with the query.
    fn get_candidates(&self, query: &str) -> Vec<(usize, String)> {
        self.inner
            .get_candidates(query)
            .into_iter()
            .map(|id| {
                // Retrieve original text from index entries
                let text = self.inner.get_text(id).unwrap_or_default();
                (id, text)
            })
            .collect()
    }

    /// Find the nearest neighbors by similarity.
    ///
    /// # Arguments
    /// * `query` - The string to search for
    /// * `limit` - Number of nearest neighbors to return
    /// * `algorithm` - Similarity algorithm to use
    /// * `normalize` - Optional normalization mode. Use "lowercase" for case-insensitive comparison.
    ///                Default is "lowercase" for backwards compatibility.
    #[pyo3(signature = (query, limit, algorithm="jaro_winkler", normalize=Some("lowercase")))]
    fn find_nearest(
        &self,
        py: Python<'_>,
        query: &str,
        limit: usize,
        algorithm: &str,
        normalize: Option<&str>,
    ) -> PyResult<Vec<SearchResult>> {
        // Use search with no minimum similarity and limit results
        self.search(py, query, algorithm, 0.0, Some(limit), normalize)
    }

    /// Check if the index contains an exact match.
    fn contains(&self, query: &str) -> bool {
        self.inner.contains(query)
    }

    /// Get number of indexed items.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    // =========================================================================
    // Serialization
    // =========================================================================

    /// Serialize the index to bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self.inner.to_bytes().map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Serialization failed: {}", e))
        })?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Deserialize an index from bytes.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        indexing::ngram_index::NgramIndex::from_bytes(data)
            .map(|inner| Self { inner })
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Deserialization failed: {}", e))
            })
    }

    // =========================================================================
    // Compression
    // =========================================================================

    /// Compress the posting lists to reduce memory usage.
    ///
    /// This converts the uncompressed posting lists to delta + varint encoded
    /// format, typically achieving 50-70% memory reduction for large indices.
    ///
    /// After compression:
    /// - The original uncompressed index is cleared to free memory
    /// - Searches will use the compressed index (slightly slower decode)
    /// - New items cannot be added (call `decompress()` first)
    ///
    /// Example:
    ///     index = NgramIndex()
    ///     index.add_all(large_dataset)
    ///     index.compress()  # Reduce memory usage
    ///     # Searches still work, using compressed index
    fn compress(&mut self) {
        self.inner.compress();
    }

    /// Decompress the posting lists to allow adding new items.
    ///
    /// This restores the uncompressed index from the compressed format.
    /// Call this before adding new items to a compressed index.
    fn decompress(&mut self) {
        self.inner.decompress();
    }

    /// Check if the index is currently compressed.
    fn is_compressed(&self) -> bool {
        self.inner.is_compressed()
    }

    /// Get compression statistics.
    ///
    /// Returns a tuple of (compressed_bytes, uncompressed_bytes, compression_ratio).
    /// Returns None if the index is not compressed.
    fn compression_stats(&self) -> Option<(usize, usize, f64)> {
        self.inner.compression_stats()
    }
}

/// Python wrapper for Hybrid Index.
///
/// Note: This class is NOT thread-safe and cannot be shared between Python
/// threads. Create separate instances for each thread or use appropriate
/// synchronization in your Python code.
///
/// Note: Unlike BkTree, this index allows duplicate strings. Adding the same
/// string twice will create two separate entries with different IDs.
#[pyclass]
struct PyHybridIndex {
    inner: indexing::ngram_index::HybridIndex,
}

#[pymethods]
impl PyHybridIndex {
    #[new]
    #[pyo3(signature = (ngram_size=3, min_ngram_ratio=0.2, normalize=true))]
    fn new(ngram_size: usize, min_ngram_ratio: f64, normalize: bool) -> PyResult<Self> {
        if ngram_size < 2 {
            return Err(ValidationError::new_err(format!(
                "ngram_size must be >= 2, got {}",
                ngram_size
            )));
        }
        validate_ngram_ratio(min_ngram_ratio)?;
        Ok(Self {
            inner: indexing::ngram_index::HybridIndex::with_params(
                ngram_size,
                min_ngram_ratio,
                normalize,
            ),
        })
    }

    fn add(&mut self, text: String) -> usize {
        self.inner.add(text)
    }

    fn add_with_data(&mut self, text: String, data: u64) -> usize {
        self.inner.add_with_data(text, Some(data))
    }

    fn add_all(&mut self, texts: Vec<String>) {
        for text in texts {
            self.inner.add(text);
        }
    }

    /// Search for similar strings.
    ///
    /// # Arguments
    /// * `query` - Query string to search for
    /// * `algorithm` - Similarity algorithm ("jaro_winkler", "jaro", "levenshtein")
    /// * `min_similarity` - Minimum similarity threshold (0.0 to 1.0)
    /// * `limit` - Maximum number of results to return
    /// * `normalize` - Optional normalization mode. Use "lowercase" for case-insensitive comparison.
    ///                Default is "lowercase" for backwards compatibility.
    #[pyo3(signature = (query, algorithm="jaro_winkler", min_similarity=0.0, limit=None, normalize=Some("lowercase")))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        algorithm: &str,
        min_similarity: f64,
        limit: Option<usize>,
        normalize: Option<&str>,
    ) -> PyResult<Vec<SearchResult>> {
        validate_similarity(min_similarity, "min_similarity")?;

        let metric = get_similarity_metric_with_normalize(algorithm, normalize)?;

        let query = query.to_string();
        let results = py.allow_threads(|| {
            self.inner
                .search(&query, metric.as_ref(), min_similarity, limit)
        });

        Ok(results.into_iter().map(convert_search_match).collect())
    }

    /// Batch search for multiple queries (parallel processing).
    ///
    /// # Arguments
    /// * `queries` - List of query strings
    /// * `algorithm` - Similarity algorithm to use
    /// * `min_similarity` - Minimum similarity score
    /// * `limit` - Maximum results per query
    /// * `normalize` - Optional normalization mode. Use "lowercase" for case-insensitive comparison.
    ///                Default is "lowercase" for backwards compatibility.
    #[pyo3(signature = (queries, algorithm="jaro_winkler", min_similarity=0.0, limit=None, normalize=Some("lowercase")))]
    fn batch_search(
        &self,
        py: Python<'_>,
        queries: Vec<String>,
        algorithm: &str,
        min_similarity: f64,
        limit: Option<usize>,
        normalize: Option<&str>,
    ) -> PyResult<Vec<Vec<SearchResult>>> {
        let metric = get_similarity_metric_with_normalize(algorithm, normalize)?;

        let results = py.allow_threads(|| {
            self.inner
                .batch_search(&queries, metric.as_ref(), min_similarity, limit)
        });

        Ok(results
            .into_iter()
            .map(|matches| matches.into_iter().map(convert_search_match).collect())
            .collect())
    }

    /// Find the nearest neighbors by similarity.
    ///
    /// # Arguments
    /// * `query` - The string to search for
    /// * `limit` - Number of nearest neighbors to return
    /// * `algorithm` - Similarity algorithm to use
    /// * `normalize` - Optional normalization mode. Use "lowercase" for case-insensitive comparison.
    ///                Default is "lowercase" for backwards compatibility.
    #[pyo3(signature = (query, limit, algorithm="jaro_winkler", normalize=Some("lowercase")))]
    fn find_nearest(
        &self,
        py: Python<'_>,
        query: &str,
        limit: usize,
        algorithm: &str,
        normalize: Option<&str>,
    ) -> PyResult<Vec<SearchResult>> {
        // Use search with no minimum similarity and limit results
        self.search(py, query, algorithm, 0.0, Some(limit), normalize)
    }

    /// Check if the index contains an exact match.
    fn contains(&self, query: &str) -> bool {
        self.inner.contains(query)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

// ============================================================================
// Thread-Safe Index Wrappers
// ============================================================================

/// Thread-safe BK-tree for concurrent fuzzy string search.
///
/// This wrapper uses a read-write lock to provide safe concurrent access
/// from multiple Python threads. Read operations (search, contains) can
/// run concurrently, while write operations (add, remove) have exclusive access.
///
/// # Example
/// ```python
/// from concurrent.futures import ThreadPoolExecutor
/// import fuzzyrust as fr
///
/// tree = fr.ThreadSafeBkTree()
/// tree.add_all(["hello", "world", "help"])
///
/// with ThreadPoolExecutor(max_workers=4) as executor:
///     # Concurrent searches are safe
///     futures = [executor.submit(tree.search, "hello", 2) for _ in range(10)]
///     results = [f.result() for f in futures]
/// ```
#[pyclass]
struct PyThreadSafeBkTree {
    inner: indexing::threadsafe::ThreadSafeBkTree,
}

#[pymethods]
impl PyThreadSafeBkTree {
    #[new]
    #[pyo3(signature = (use_damerau=false))]
    fn new(use_damerau: bool) -> Self {
        let inner = if use_damerau {
            indexing::threadsafe::ThreadSafeBkTree::new_damerau()
        } else {
            indexing::threadsafe::ThreadSafeBkTree::new()
        };
        Self { inner }
    }

    /// Add a string to the tree. Thread-safe.
    fn add(&self, text: String) -> bool {
        self.inner.add(text)
    }

    /// Add a string with associated data. Thread-safe.
    fn add_with_data(&self, text: String, data: u64) -> bool {
        self.inner.add_with_data(text, data)
    }

    /// Add multiple strings. Thread-safe.
    fn add_all(&self, texts: Vec<String>) {
        self.inner.add_all(texts);
    }

    /// Search for strings within a given edit distance. Thread-safe.
    fn search(&self, py: Python<'_>, query: &str, max_distance: usize) -> Vec<SearchResult> {
        let query = query.to_string();
        let results = py.allow_threads(|| self.inner.search(&query, max_distance));
        results
            .into_iter()
            .map(|r| {
                let text_len = r.text.len();
                let score = 1.0 - (r.distance as f64 / (query.len().max(text_len) as f64).max(1.0));
                SearchResult {
                    id: r.id,
                    text: r.text,
                    score,
                    distance: Some(r.distance),
                    data: r.data,
                }
            })
            .collect()
    }

    /// Find the k nearest neighbors. Thread-safe.
    fn find_nearest(&self, py: Python<'_>, query: &str, k: usize) -> Vec<SearchResult> {
        let query = query.to_string();
        let results = py.allow_threads(|| self.inner.find_nearest(&query, k));
        results
            .into_iter()
            .map(|r| {
                let text_len = r.text.len();
                let score = 1.0 - (r.distance as f64 / (query.len().max(text_len) as f64).max(1.0));
                SearchResult {
                    id: r.id,
                    text: r.text,
                    score,
                    distance: Some(r.distance),
                    data: r.data,
                }
            })
            .collect()
    }

    /// Check if the tree contains an exact match. Thread-safe.
    fn contains(&self, query: &str) -> bool {
        self.inner.contains(query)
    }

    /// Remove an entry by ID. Thread-safe.
    fn remove(&self, id: usize) -> bool {
        self.inner.remove(id)
    }

    /// Remove an entry by text value. Thread-safe.
    fn remove_text(&self, text: &str) -> bool {
        self.inner.remove_text(text)
    }

    /// Rebuild the tree without tombstones. Thread-safe.
    fn compact(&self) {
        self.inner.compact();
    }

    /// Get the number of deleted entries.
    fn deleted_count(&self) -> usize {
        self.inner.deleted_count()
    }

    /// Get the number of active entries.
    fn active_count(&self) -> usize {
        self.inner.active_count()
    }

    /// Serialize the tree to bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self.inner.to_bytes().map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Serialization failed: {}", e))
        })?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Deserialize a tree from bytes.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        indexing::threadsafe::ThreadSafeBkTree::from_bytes(data)
            .map(|inner| Self { inner })
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Deserialization failed: {}", e))
            })
    }

    /// Deserialize a tree from bytes using Damerau-Levenshtein distance.
    #[staticmethod]
    fn from_bytes_damerau(data: &[u8]) -> PyResult<Self> {
        indexing::threadsafe::ThreadSafeBkTree::from_bytes_damerau(data)
            .map(|inner| Self { inner })
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Deserialization failed: {}", e))
            })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Thread-safe N-gram index for concurrent fuzzy string search.
///
/// This wrapper uses a read-write lock to provide safe concurrent access
/// from multiple Python threads.
#[pyclass]
struct PyThreadSafeNgramIndex {
    inner: indexing::threadsafe::ThreadSafeNgramIndex,
}

#[pymethods]
impl PyThreadSafeNgramIndex {
    #[new]
    #[pyo3(signature = (ngram_size=3, min_similarity=0.0, min_ngram_ratio=0.0, case_insensitive=false))]
    fn new(
        ngram_size: usize,
        min_similarity: f64,
        min_ngram_ratio: f64,
        case_insensitive: bool,
    ) -> Self {
        Self {
            inner: indexing::threadsafe::ThreadSafeNgramIndex::with_params(
                ngram_size,
                min_similarity,
                min_ngram_ratio,
                case_insensitive,
            ),
        }
    }

    /// Add a string to the index. Thread-safe.
    fn add(&self, text: String) -> usize {
        self.inner.add(text)
    }

    /// Add a string with associated data. Thread-safe.
    fn add_with_data(&self, text: String, data: u64) -> usize {
        self.inner.add_with_data(text, Some(data))
    }

    /// Add multiple strings. Thread-safe.
    fn add_all(&self, texts: Vec<String>) {
        self.inner.add_all(texts);
    }

    /// Add multiple strings using parallel processing. Thread-safe.
    fn add_all_parallel(&self, texts: Vec<String>) {
        self.inner.add_all_parallel(texts);
    }

    /// Search for similar strings. Thread-safe.
    #[pyo3(signature = (query, min_similarity=0.0, limit=None, algorithm="jaro_winkler"))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        min_similarity: f64,
        limit: Option<usize>,
        algorithm: &str,
    ) -> PyResult<Vec<SearchResult>> {
        let query = query.to_string();
        let algorithm = algorithm.to_string();

        let results: Result<Vec<indexing::ngram_index::SearchMatch>, String> =
            py.allow_threads(|| match algorithm.as_str() {
                "jaro_winkler" | "jaro-winkler" => {
                    let sim = algorithms::JaroWinkler::new();
                    Ok(self
                        .inner
                        .search_parallel(&query, &sim, min_similarity, limit))
                }
                "levenshtein" => {
                    let sim = algorithms::Levenshtein::new();
                    Ok(self
                        .inner
                        .search_parallel(&query, &sim, min_similarity, limit))
                }
                "damerau" | "damerau_levenshtein" | "damerau-levenshtein" => {
                    let sim = algorithms::DamerauLevenshtein::new();
                    Ok(self
                        .inner
                        .search_parallel(&query, &sim, min_similarity, limit))
                }
                "ngram" => {
                    let sim = algorithms::Ngram::new(3);
                    Ok(self
                        .inner
                        .search_parallel(&query, &sim, min_similarity, limit))
                }
                _ => Err(format!("Unknown algorithm: {}", algorithm)),
            });

        let results = results.map_err(AlgorithmError::new_err)?;

        Ok(results
            .into_iter()
            .map(|m| SearchResult {
                id: m.id,
                text: m.text,
                score: m.similarity,
                distance: None,
                data: m.data,
            })
            .collect())
    }

    /// Check if the index contains an exact match. Thread-safe.
    fn contains(&self, query: &str) -> bool {
        self.inner.contains(query)
    }

    /// Get the text for a specific ID. Thread-safe.
    fn get_text(&self, id: usize) -> Option<String> {
        self.inner.get_text(id)
    }

    /// Clear the index. Thread-safe.
    fn clear(&self) {
        self.inner.clear();
    }

    /// Serialize the index to bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = self.inner.to_bytes().map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Serialization failed: {}", e))
        })?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Deserialize an index from bytes.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        indexing::threadsafe::ThreadSafeNgramIndex::from_bytes(data)
            .map(|inner| Self { inner })
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Deserialization failed: {}", e))
            })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

// ============================================================================
// Sharded Indices for Large Datasets
// ============================================================================

/// Sharded BK-tree for very large datasets (>1M items).
///
/// Distributes items across multiple BK-tree shards for better performance.
/// Searches are performed in parallel across all shards.
///
/// # Example
/// ```python
/// import fuzzyrust as fr
///
/// # Create with 8 shards (good default for modern CPUs)
/// tree = fr.ShardedBkTree(num_shards=8)
/// tree.add_all([f"item_{i}" for i in range(1000000)])
///
/// # Search across all shards in parallel
/// results = tree.search("item_500000", max_distance=2)
/// ```
#[pyclass]
struct PyShardedBkTree {
    inner: indexing::sharded::ShardedBkTree,
}

#[pymethods]
impl PyShardedBkTree {
    #[new]
    #[pyo3(signature = (num_shards=8, use_damerau=false))]
    fn new(num_shards: usize, use_damerau: bool) -> Self {
        let inner = if use_damerau {
            indexing::sharded::ShardedBkTree::new_damerau(num_shards)
        } else {
            indexing::sharded::ShardedBkTree::new(num_shards)
        };
        Self { inner }
    }

    /// Add a string to the tree.
    fn add(&mut self, text: String) -> bool {
        self.inner.add(text)
    }

    /// Add a string with associated data.
    fn add_with_data(&mut self, text: String, data: u64) -> bool {
        self.inner.add_with_data(text, Some(data))
    }

    /// Add multiple strings.
    fn add_all(&mut self, texts: Vec<String>) {
        self.inner.add_all(texts);
    }

    /// Add multiple strings with parallel distribution.
    fn add_all_parallel(&mut self, texts: Vec<String>) {
        self.inner.add_all_parallel(texts);
    }

    /// Search all shards in parallel.
    fn search(&self, py: Python<'_>, query: &str, max_distance: usize) -> Vec<SearchResult> {
        let query = query.to_string();
        let results = py.allow_threads(|| self.inner.search(&query, max_distance));
        results
            .into_iter()
            .map(|r| {
                let text_len = r.text.len();
                let score = 1.0 - (r.distance as f64 / (query.len().max(text_len) as f64).max(1.0));
                SearchResult {
                    id: r.id,
                    text: r.text,
                    score,
                    distance: Some(r.distance),
                    data: r.data,
                }
            })
            .collect()
    }

    /// Find k nearest neighbors across all shards.
    fn find_nearest(&self, py: Python<'_>, query: &str, k: usize) -> Vec<SearchResult> {
        let query = query.to_string();
        let results = py.allow_threads(|| self.inner.find_nearest(&query, k));
        results
            .into_iter()
            .map(|r| {
                let text_len = r.text.len();
                let score = 1.0 - (r.distance as f64 / (query.len().max(text_len) as f64).max(1.0));
                SearchResult {
                    id: r.id,
                    text: r.text,
                    score,
                    distance: Some(r.distance),
                    data: r.data,
                }
            })
            .collect()
    }

    /// Check if the tree contains an exact match.
    fn contains(&self, query: &str) -> bool {
        self.inner.contains(query)
    }

    /// Remove an entry by text.
    fn remove_text(&mut self, text: &str) -> bool {
        self.inner.remove_text(text)
    }

    /// Compact all shards to remove tombstones.
    fn compact(&mut self) {
        self.inner.compact();
    }

    /// Get the number of shards.
    fn num_shards(&self) -> usize {
        self.inner.num_shards()
    }

    /// Get the distribution of items across shards.
    fn shard_distribution(&self) -> Vec<usize> {
        self.inner.shard_distribution()
    }

    /// Get the number of active entries.
    fn active_count(&self) -> usize {
        self.inner.active_count()
    }

    /// Get the number of deleted entries.
    fn deleted_count(&self) -> usize {
        self.inner.deleted_count()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Sharded N-gram index for very large datasets.
///
/// Distributes items across multiple N-gram index shards.
/// Searches are performed in parallel across all shards.
#[pyclass]
struct PyShardedNgramIndex {
    inner: indexing::sharded::ShardedNgramIndex,
}

#[pymethods]
impl PyShardedNgramIndex {
    #[new]
    #[pyo3(signature = (num_shards=8, ngram_size=3, min_similarity=0.0, min_ngram_ratio=0.2, normalize=true))]
    fn new(
        num_shards: usize,
        ngram_size: usize,
        min_similarity: f64,
        min_ngram_ratio: f64,
        normalize: bool,
    ) -> PyResult<Self> {
        if ngram_size < 2 {
            return Err(ValidationError::new_err(format!(
                "ngram_size must be >= 2, got {}",
                ngram_size
            )));
        }
        Ok(Self {
            inner: indexing::sharded::ShardedNgramIndex::with_params(
                num_shards,
                ngram_size,
                min_similarity,
                min_ngram_ratio,
                normalize,
            ),
        })
    }

    /// Add a string to the index.
    fn add(&mut self, text: String) -> usize {
        self.inner.add(text)
    }

    /// Add a string with associated data.
    fn add_with_data(&mut self, text: String, data: u64) -> usize {
        self.inner.add_with_data(text, Some(data))
    }

    /// Add multiple strings.
    fn add_all(&mut self, texts: Vec<String>) {
        self.inner.add_all(texts);
    }

    /// Add multiple strings with parallel distribution and insertion.
    fn add_all_parallel(&mut self, texts: Vec<String>) {
        self.inner.add_all_parallel(texts);
    }

    /// Search all shards in parallel.
    #[pyo3(signature = (query, algorithm="jaro_winkler", min_similarity=0.0, limit=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        algorithm: &str,
        min_similarity: f64,
        limit: Option<usize>,
    ) -> PyResult<Vec<SearchResult>> {
        let query = query.to_string();
        let algorithm = algorithm.to_string();

        let results: Result<Vec<indexing::ngram_index::SearchMatch>, String> =
            py.allow_threads(|| match algorithm.as_str() {
                "jaro_winkler" | "jaro-winkler" => {
                    let sim = algorithms::JaroWinkler::new();
                    Ok(self.inner.search(&query, &sim, min_similarity, limit))
                }
                "levenshtein" => {
                    let sim = algorithms::Levenshtein::new();
                    Ok(self.inner.search(&query, &sim, min_similarity, limit))
                }
                "damerau" | "damerau_levenshtein" | "damerau-levenshtein" => {
                    let sim = algorithms::DamerauLevenshtein::new();
                    Ok(self.inner.search(&query, &sim, min_similarity, limit))
                }
                "ngram" => {
                    let sim = algorithms::Ngram::new(3);
                    Ok(self.inner.search(&query, &sim, min_similarity, limit))
                }
                _ => Err(format!("Unknown algorithm: {}", algorithm)),
            });

        let results = results.map_err(AlgorithmError::new_err)?;

        Ok(results
            .into_iter()
            .map(|m| SearchResult {
                id: m.id,
                text: m.text,
                score: m.similarity,
                distance: None,
                data: m.data,
            })
            .collect())
    }

    /// Check if the index contains an exact match.
    fn contains(&self, query: &str) -> bool {
        self.inner.contains(query)
    }

    /// Get text by ID.
    fn get_text(&self, id: usize) -> Option<String> {
        self.inner.get_text(id)
    }

    /// Clear all shards.
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get the number of shards.
    fn num_shards(&self) -> usize {
        self.inner.num_shards()
    }

    /// Get the N-gram size.
    fn ngram_size(&self) -> usize {
        self.inner.ngram_size()
    }

    /// Get the distribution of items across shards.
    fn shard_distribution(&self) -> Vec<usize> {
        self.inner.shard_distribution()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

// ============================================================================
// Schema-Based Multi-Field Matching
// ============================================================================

use indexing::schema;

/// Python wrapper for SchemaSearchResult
#[pyclass]
#[derive(Clone, Debug)]
pub struct SchemaSearchResult {
    /// Record ID
    #[pyo3(get)]
    pub id: usize,

    /// Overall similarity score (0.0-1.0)
    #[pyo3(get)]
    pub score: f64,

    /// Per-field scores as a dictionary
    #[pyo3(get)]
    pub field_scores: std::collections::HashMap<String, f64>,

    /// The matched record as a dictionary
    #[pyo3(get)]
    pub record: std::collections::HashMap<String, String>,

    /// Optional user data
    #[pyo3(get)]
    pub data: Option<u64>,
}

#[pymethods]
impl SchemaSearchResult {
    fn __repr__(&self) -> String {
        format!(
            "SchemaSearchResult(id={}, score={:.3}, fields={})",
            self.id,
            self.score,
            self.record.len()
        )
    }

    fn __str__(&self) -> String {
        format!("Record {}: {:.3}", self.id, self.score)
    }
}

/// Python wrapper for Schema
#[pyclass]
pub struct Schema {
    inner: schema::Schema,
}

#[pymethods]
impl Schema {
    /// Create a new schema builder
    #[staticmethod]
    fn builder() -> SchemaBuilder {
        SchemaBuilder {
            fields: Vec::new(),
            scoring: None,
        }
    }

    /// Get all field names
    fn field_names(&self) -> Vec<String> {
        self.inner.fields().iter().map(|f| f.name.clone()).collect()
    }

    /// Get number of fields
    fn field_count(&self) -> usize {
        self.inner.field_count()
    }

    /// Get required field names
    fn required_fields(&self) -> Vec<String> {
        self.inner
            .required_fields()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("Schema(fields={})", self.inner.field_count())
    }
}

/// Python wrapper for SchemaBuilder
#[pyclass]
pub struct SchemaBuilder {
    fields: Vec<schema::Field>,
    scoring: Option<schema::ScoringStrategy>,
}

#[pymethods]
impl SchemaBuilder {
    /// Create a new schema builder
    #[new]
    fn new() -> Self {
        SchemaBuilder {
            fields: Vec::new(),
            scoring: None,
        }
    }

    /// Add a field to the schema
    #[pyo3(signature = (name, field_type, algorithm=None, weight=1.0, required=false, normalize=None, max_length=None, separator=None, chunk_size=None))]
    fn add_field(
        &mut self,
        name: String,
        field_type: &str,
        algorithm: Option<&str>,
        weight: f64,
        required: bool,
        normalize: Option<&str>,
        max_length: Option<usize>,
        separator: Option<&str>,
        chunk_size: Option<usize>,
    ) -> PyResult<()> {
        // Validate weight parameter
        if !weight.is_finite() {
            return Err(ValidationError::new_err(format!(
                "weight must be a finite number, got {}",
                weight
            )));
        }
        if !(0.0..=10.0).contains(&weight) {
            return Err(ValidationError::new_err(format!(
                "weight must be in range [0.0, 10.0], got {}",
                weight
            )));
        }

        // Determine default algorithm based on field type if not provided
        let default_algo = match field_type.to_lowercase().as_str() {
            "short_text" => "jaro_winkler",
            "long_text" => "ngram",
            "token_set" => "jaccard",
            _ => {
                return Err(SchemaError::new_err(format!(
                    "Unknown field type: '{}'. Valid: short_text, long_text, token_set",
                    field_type
                )))
            }
        };

        let algo_str = algorithm.unwrap_or(default_algo);
        let parsed_algorithm = parse_algorithm(algo_str)?;

        // Parse field type with parameters
        let ft = match field_type.to_lowercase().as_str() {
            "short_text" => schema::types::FieldType::ShortText {
                max_length: max_length.unwrap_or(100),
                default_algorithm: parsed_algorithm.clone(),
            },
            "long_text" => schema::types::FieldType::LongText {
                default_algorithm: parsed_algorithm.clone(),
                chunk_size,
            },
            "token_set" => {
                // Separator can be any string (single or multi-character)
                // e.g., "," or ", " or " | "
                let sep_str = separator.unwrap_or(",");
                if sep_str.is_empty() {
                    return Err(ValidationError::new_err("separator cannot be empty"));
                }
                schema::types::FieldType::TokenSet {
                    separator: sep_str.to_string(),
                    default_algorithm: parsed_algorithm.clone(),
                }
            }
            _ => {
                return Err(SchemaError::new_err(format!(
                    "Unknown field type: '{}'. Valid: short_text, long_text, token_set",
                    field_type
                )))
            }
        };

        // Parse normalization
        let norm = if let Some(n) = normalize {
            Some(parse_normalization_mode(n)?)
        } else {
            None
        };

        // Create field
        let mut field = schema::Field::new(name, ft);
        field.algorithm = parsed_algorithm;
        field.weight = weight;
        field.required = required;
        field.normalization = norm;

        // Store field for later (we'll collect them when building)
        self.fields.push(field);
        Ok(())
    }

    /// Set the scoring strategy
    #[pyo3(signature = (strategy="weighted_average"))]
    fn with_scoring(&mut self, strategy: &str) -> PyResult<()> {
        let scoring = match strategy.to_lowercase().as_str() {
            "weighted_average" => schema::ScoringStrategy::WeightedAverage,
            "minmax_scaling" | "minmax" => schema::ScoringStrategy::MinMaxScaling,
            _ => {
                return Err(SchemaError::new_err(format!(
                    "Unknown scoring strategy: '{}'. Valid: weighted_average, minmax_scaling",
                    strategy
                )))
            }
        };

        self.scoring = Some(scoring);
        Ok(())
    }

    /// Build the schema
    fn build(&self) -> PyResult<Schema> {
        // Build the schema manually from collected fields
        let mut builder = schema::Schema::builder();
        for field in self.fields.iter().cloned() {
            builder = builder.add_field(field);
        }
        if let Some(scoring) = &self.scoring {
            builder = builder.with_scoring_strategy(scoring.clone());
        }

        match builder.build() {
            Ok(schema) => Ok(Schema { inner: schema }),
            Err(e) => Err(SchemaError::new_err(format!(
                "Schema validation failed: {}",
                e
            ))),
        }
    }
}

/// Python wrapper for SchemaIndex
#[pyclass]
pub struct SchemaIndex {
    inner: schema::SchemaIndex,
}

#[pymethods]
impl SchemaIndex {
    /// Create a new schema index
    #[new]
    fn new(schema: &Schema) -> Self {
        Self {
            inner: schema::SchemaIndex::new(schema.inner.clone()),
        }
    }

    /// Add a record to the index
    #[pyo3(signature = (record, data=None))]
    fn add(
        &mut self,
        record: std::collections::HashMap<String, String>,
        data: Option<u64>,
    ) -> PyResult<usize> {
        let mut rec = schema::Record::new();
        for (key, value) in record {
            rec.set_field(key, value);
        }

        match self.inner.add_record(rec, data) {
            Ok(id) => Ok(id),
            Err(e) => Err(IndexError::new_err(format!("Failed to add record: {}", e))),
        }
    }

    /// Search for matching records
    ///
    /// Args:
    ///     query: Dictionary mapping field names to query values
    ///     min_similarity: Minimum overall similarity score (0.0 to 1.0)
    ///     limit: Maximum number of results to return (None for unlimited)
    ///     min_field_similarity: Minimum per-field similarity threshold
    ///     field_boosts: Optional dict of field name to boost multiplier.
    ///         Boosts are multiplied with schema weights at query time.
    ///         Example: {"name": 2.0, "city": 0.5} doubles name's weight, halves city's.
    #[pyo3(signature = (query, min_similarity=0.0, limit=None, min_field_similarity=0.0, field_boosts=None))]
    fn search(
        &self,
        query: std::collections::HashMap<String, String>,
        min_similarity: f64,
        limit: Option<usize>,
        min_field_similarity: f64,
        field_boosts: Option<std::collections::HashMap<String, f64>>,
    ) -> PyResult<Vec<SchemaSearchResult>> {
        validate_similarity(min_similarity, "min_similarity")?;
        validate_similarity(min_field_similarity, "min_field_similarity")?;

        let mut q = schema::Record::new();
        for (key, value) in query {
            q.set_field(key, value);
        }

        // Convert Python dict to AHashMap for field_boosts
        let boosts = field_boosts.map(|b| b.into_iter().collect());

        let options = schema::SearchOptions {
            min_score: min_similarity,
            limit,
            min_field_score: min_field_similarity,
            field_boosts: boosts,
        };

        match self.inner.search(&q, options) {
            Ok(results) => {
                let py_results = results
                    .into_iter()
                    .map(|r| SchemaSearchResult {
                        id: r.id,
                        score: r.score,
                        field_scores: r.field_scores.into_iter().collect(),
                        record: r.record.fields.into_iter().collect(),
                        data: r.record.data,
                    })
                    .collect();
                Ok(py_results)
            }
            Err(e) => Err(IndexError::new_err(format!("Search failed: {}", e))),
        }
    }

    /// Get a record by ID
    ///
    /// Returns None if the record doesn't exist, or raises an error if there's a storage problem.
    fn get(&self, id: usize) -> PyResult<Option<std::collections::HashMap<String, String>>> {
        match self.inner.get_record(id) {
            Ok(Some(r)) => Ok(Some(r.fields.into_iter().collect())),
            Ok(None) => Ok(None),
            Err(e) => Err(IndexError::new_err(format!("Failed to get record: {}", e))),
        }
    }

    /// Batch search for multiple queries in parallel
    ///
    /// Executes searches for all queries concurrently using Rayon parallelization.
    /// Returns results in the same order as input queries.
    ///
    /// Args:
    ///     queries: List of query records (each a dict mapping field names to values)
    ///     min_similarity: Minimum overall similarity score (0.0 to 1.0)
    ///     limit: Maximum number of results per query (None for unlimited)
    ///     min_field_similarity: Minimum per-field similarity threshold
    ///     field_boosts: Optional dict of field name to boost multiplier (applied to all queries)
    ///
    /// Returns:
    ///     List of lists, where each inner list contains SchemaSearchResult objects
    ///     for the corresponding query, sorted by descending score.
    #[pyo3(signature = (queries, min_similarity=0.0, limit=None, min_field_similarity=0.0, field_boosts=None))]
    fn batch_search(
        &self,
        queries: Vec<std::collections::HashMap<String, String>>,
        min_similarity: f64,
        limit: Option<usize>,
        min_field_similarity: f64,
        field_boosts: Option<std::collections::HashMap<String, f64>>,
    ) -> PyResult<Vec<Vec<SchemaSearchResult>>> {
        validate_similarity(min_similarity, "min_similarity")?;
        validate_similarity(min_field_similarity, "min_field_similarity")?;

        // Convert Python dicts to Records
        let records: Vec<schema::Record> = queries
            .into_iter()
            .map(|q| {
                let mut rec = schema::Record::new();
                for (key, value) in q {
                    rec.set_field(key, value);
                }
                rec
            })
            .collect();

        // Convert Python dict to AHashMap for field_boosts
        let boosts = field_boosts.map(|b| b.into_iter().collect());

        let options = schema::SearchOptions {
            min_score: min_similarity,
            limit,
            min_field_score: min_field_similarity,
            field_boosts: boosts,
        };

        match self.inner.batch_search(&records, options) {
            Ok(results) => {
                let py_results: Vec<Vec<SchemaSearchResult>> = results
                    .into_iter()
                    .map(|query_results| {
                        query_results
                            .into_iter()
                            .map(|r| SchemaSearchResult {
                                id: r.id,
                                score: r.score,
                                field_scores: r.field_scores.into_iter().collect(),
                                record: r.record.fields.into_iter().collect(),
                                data: r.record.data,
                            })
                            .collect()
                    })
                    .collect();
                Ok(py_results)
            }
            Err(e) => Err(IndexError::new_err(format!("Batch search failed: {}", e))),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("SchemaIndex(records={})", self.inner.len())
    }
}

// Helper functions for parsing
fn parse_algorithm(algo: &str) -> PyResult<schema::types::Algorithm> {
    match algo.to_lowercase().as_str() {
        "levenshtein" => Ok(schema::types::Algorithm::Levenshtein),
        "damerau_levenshtein" | "damerau" => Ok(schema::types::Algorithm::DamerauLevenshtein),
        "jaro_winkler" => Ok(schema::types::Algorithm::JaroWinkler(Default::default())),
        "ngram" => Ok(schema::types::Algorithm::Ngram { ngram_size: 2 }),
        "jaccard" => Ok(schema::types::Algorithm::Jaccard),
        "cosine" => Ok(schema::types::Algorithm::Cosine),
        "exact" | "exact_match" => Ok(schema::types::Algorithm::ExactMatch),
        _ => Err(AlgorithmError::new_err(format!(
            "Unknown algorithm: '{}'. Valid algorithms: levenshtein, damerau_levenshtein, jaro_winkler, ngram, jaccard, cosine, exact_match",
            algo
        ))),
    }
}

// ============================================================================
// Python Module
// ============================================================================

#[pymodule]
fn _core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Custom exceptions
    m.add("FuzzyRustError", py.get_type::<FuzzyRustError>())?;
    m.add("ValidationError", py.get_type::<ValidationError>())?;
    m.add("AlgorithmError", py.get_type::<AlgorithmError>())?;
    m.add("IndexError", py.get_type::<IndexError>())?;
    m.add("SchemaError", py.get_type::<SchemaError>())?;

    // Result types
    m.add_class::<SearchResult>()?;
    m.add_class::<MatchResult>()?;
    m.add_class::<DeduplicationResult>()?;
    m.add_class::<AlgorithmComparison>()?;

    // Distance/similarity functions
    m.add_function(wrap_pyfunction!(levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_bounded, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_bounded, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_similarity, m)?)?;

    // Grapheme cluster mode functions
    m.add_function(wrap_pyfunction!(levenshtein_grapheme, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_similarity_grapheme, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_similarity_grapheme, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_similarity_grapheme, m)?)?;

    // SIMD-accelerated functions
    m.add_function(wrap_pyfunction!(levenshtein_simd, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_simd_bounded, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_similarity_simd, m)?)?;

    m.add_function(wrap_pyfunction!(hamming, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(extract_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(soundex, m)?)?;
    m.add_function(wrap_pyfunction!(soundex_match, m)?)?;
    m.add_function(wrap_pyfunction!(metaphone, m)?)?;
    m.add_function(wrap_pyfunction!(metaphone_match, m)?)?;
    m.add_function(wrap_pyfunction!(double_metaphone, m)?)?;
    m.add_function(wrap_pyfunction!(double_metaphone_match, m)?)?;
    m.add_function(wrap_pyfunction!(double_metaphone_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_length, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_string, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(longest_common_substring_length, m)?)?;
    m.add_function(wrap_pyfunction!(longest_common_substring, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_chars, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_words, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(soundex_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(metaphone_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(bigram_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(trigram_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_profile_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distance_padded, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_similarity_max, m)?)?;

    // RapidFuzz-compatible convenience functions
    m.add_function(wrap_pyfunction!(partial_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(token_sort_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(token_set_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(wratio, m)?)?;
    m.add_function(wrap_pyfunction!(ratio, m)?)?;
    m.add_function(wrap_pyfunction!(qratio, m)?)?;
    m.add_function(wrap_pyfunction!(qwratio, m)?)?;
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    m.add_function(wrap_pyfunction!(extract_one, m)?)?;

    // Batch processing
    m.add_function(wrap_pyfunction!(batch_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(batch_jaro_winkler, m)?)?;
    m.add_function(wrap_pyfunction!(find_best_matches, m)?)?;
    m.add_function(wrap_pyfunction!(batch_similarity_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(cdist, m)?)?;
    m.add_function(wrap_pyfunction!(batch_similarity, m)?)?;

    // Case-insensitive variants
    m.add_function(wrap_pyfunction!(levenshtein_ci, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_similarity_ci, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_ci, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_similarity_ci, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_similarity_ci, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_similarity_ci, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_similarity_ci, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_jaccard_ci, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_chars_ci, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_words_ci, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_ngrams_ci, m)?)?;
    m.add_function(wrap_pyfunction!(bigram_similarity_ci, m)?)?;
    m.add_function(wrap_pyfunction!(trigram_similarity_ci, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_similarity_ci, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_length_ci, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_string_ci, m)?)?;
    m.add_function(wrap_pyfunction!(longest_common_substring_ci, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_ci, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_similarity_ci, m)?)?;
    m.add_function(wrap_pyfunction!(soundex_similarity_ci, m)?)?;
    m.add_function(wrap_pyfunction!(metaphone_similarity_ci, m)?)?;

    // Normalization
    m.add_function(wrap_pyfunction!(normalize_string, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_pair, m)?)?;

    // Deduplication
    m.add_function(wrap_pyfunction!(find_duplicates, m)?)?;
    m.add_function(wrap_pyfunction!(find_duplicate_pairs, m)?)?;

    // Evaluation metrics
    m.add_class::<ConfusionMatrixResult>()?;
    m.add_function(wrap_pyfunction!(precision, m)?)?;
    m.add_function(wrap_pyfunction!(recall, m)?)?;
    m.add_function(wrap_pyfunction!(f_score, m)?)?;
    m.add_function(wrap_pyfunction!(confusion_matrix, m)?)?;

    // Multi-algorithm comparison
    m.add_function(wrap_pyfunction!(compare_algorithms, m)?)?;

    // Similarity classes
    m.add_class::<PyTfIdfCosine>()?;

    // Index classes
    m.add_class::<PyBkTree>()?;
    m.add_class::<PyNgramIndex>()?;
    m.add_class::<PyHybridIndex>()?;

    // Thread-safe index classes
    m.add_class::<PyThreadSafeBkTree>()?;
    m.add_class::<PyThreadSafeNgramIndex>()?;

    // Sharded index classes
    m.add_class::<PyShardedBkTree>()?;
    m.add_class::<PyShardedNgramIndex>()?;

    // Schema-based multi-field matching
    m.add_class::<Schema>()?;
    m.add_class::<SchemaBuilder>()?;
    m.add_class::<SchemaIndex>()?;
    m.add_class::<SchemaSearchResult>()?;

    Ok(())
}
