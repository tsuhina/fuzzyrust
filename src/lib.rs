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
pub mod indexing;
pub mod dedup;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

// Re-exports for Rust users (explicit to avoid conflicts with pyfunction wrappers)
pub use algorithms::{
    Similarity, EditDistance,
    levenshtein as levenshtein_mod,
    damerau as damerau_mod,
    jaro as jaro_mod,
    hamming as hamming_mod,
    ngram as ngram_mod,
    phonetic as phonetic_mod,
    lcs as lcs_mod,
    cosine as cosine_mod,
};
pub use indexing::{bktree, ngram_index};

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate that a similarity score or min_similarity value is in the valid range [0.0, 1.0]
fn validate_similarity(value: f64, param_name: &str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{} must be a finite number, got {}",
            param_name, value
        )));
    }
    if value < 0.0 || value > 1.0 {
        return Err(PyValueError::new_err(format!(
            "{} must be in range [0.0, 1.0], got {}",
            param_name, value
        )));
    }
    Ok(())
}

/// Validate that a Jaro-Winkler prefix_weight is in the valid range [0.0, 0.25]
fn validate_prefix_weight(value: f64) -> PyResult<()> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "prefix_weight must be a finite number, got {}",
            value
        )));
    }
    if value < 0.0 || value > 0.25 {
        return Err(PyValueError::new_err(format!(
            "prefix_weight must be in range [0.0, 0.25], got {} (values > 0.25 can produce scores > 1.0)",
            value
        )));
    }
    Ok(())
}

// ============================================================================
// Algorithm Dispatch Helpers
// ============================================================================

/// Get a boxed similarity metric for the given algorithm name.
///
/// Centralizes algorithm dispatch to avoid duplicating match statements.
/// Returns a trait object that can be used with indices and deduplication.
fn get_similarity_metric(algorithm: &str) -> PyResult<Box<dyn algorithms::Similarity + Send + Sync>> {
    match algorithm {
        "levenshtein" => Ok(Box::new(algorithms::levenshtein::Levenshtein::new())),
        "damerau_levenshtein" | "damerau" => Ok(Box::new(algorithms::damerau::DamerauLevenshtein::new())),
        "jaro" => Ok(Box::new(algorithms::jaro::Jaro::new())),
        "jaro_winkler" => Ok(Box::new(algorithms::jaro::JaroWinkler::new())),
        "ngram" | "bigram" => Ok(Box::new(algorithms::ngram::Ngram::bigram())),
        "trigram" => Ok(Box::new(algorithms::ngram::Ngram::trigram())),
        "lcs" => Ok(Box::new(algorithms::lcs::Lcs::new())),
        "cosine" | "cosine_chars" => Ok(Box::new(algorithms::cosine::CosineSimilarity::character_based())),
        _ => Err(PyValueError::new_err(format!(
            "Unknown algorithm: '{}'. Valid: levenshtein, damerau_levenshtein, jaro, jaro_winkler, ngram, trigram, lcs, cosine",
            algorithm
        ))),
    }
}

/// Get a boxed similarity function for the given algorithm name.
///
/// Used by find_best_matches where a closure is more appropriate than a trait object.
fn get_similarity_fn(algorithm: &str) -> PyResult<Box<dyn Fn(&str, &str) -> f64 + Send + Sync>> {
    match algorithm {
        "levenshtein" => Ok(Box::new(algorithms::levenshtein::levenshtein_similarity)),
        "damerau_levenshtein" | "damerau" => Ok(Box::new(algorithms::damerau::damerau_levenshtein_similarity)),
        "jaro" => Ok(Box::new(algorithms::jaro::jaro_similarity)),
        "jaro_winkler" => Ok(Box::new(algorithms::jaro::jaro_winkler_similarity)),
        "ngram" | "bigram" => Ok(Box::new(algorithms::ngram::bigram_similarity)),
        "trigram" => Ok(Box::new(algorithms::ngram::trigram_similarity)),
        "lcs" => Ok(Box::new(algorithms::lcs::lcs_similarity)),
        "cosine" | "cosine_chars" => Ok(Box::new(algorithms::cosine::cosine_similarity_chars)),
        _ => Err(PyValueError::new_err(format!(
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
fn convert_bktree_result(r: indexing::bktree::SearchResult, query: &str) -> SearchResult {
    let max_len = query.chars().count().max(r.text.chars().count());
    let score = if max_len == 0 {
        1.0
    } else {
        1.0 - (r.distance as f64 / max_len as f64)
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
    fn new(id: usize, text: String, score: f64, distance: Option<usize>, data: Option<u64>) -> Self {
        Self { id, text, score, distance, data }
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
}

#[pymethods]
impl MatchResult {
    #[new]
    fn new(text: String, score: f64) -> Self {
        Self { text, score }
    }

    fn __repr__(&self) -> String {
        format!("MatchResult(text='{}', score={:.3})", self.text, self.score)
    }

    fn __str__(&self) -> String {
        format!("{}: {:.3}", self.text, self.score)
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
        Self { groups, unique, total_duplicates }
    }

    fn __repr__(&self) -> String {
        format!(
            "DeduplicationResult(groups={}, unique={}, total_duplicates={})",
            self.groups.len(), self.unique.len(), self.total_duplicates
        )
    }

    fn __str__(&self) -> String {
        format!(
            "{} duplicate groups, {} unique items, {} total duplicates",
            self.groups.len(), self.unique.len(), self.total_duplicates
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
        Self { algorithm, score, matches }
    }

    fn __repr__(&self) -> String {
        format!(
            "AlgorithmComparison(algorithm='{}', score={:.3}, matches={})",
            self.algorithm, self.score, self.matches.len()
        )
    }

    fn __str__(&self) -> String {
        format!("{}: {:.3} ({} matches)", self.algorithm, self.score, self.matches.len())
    }
}

// ============================================================================
// Python Bindings
// ============================================================================

/// Compute Levenshtein (edit) distance between two strings.
///
/// Returns `usize::MAX` if distance exceeds `max_distance` threshold.
#[pyfunction]
#[pyo3(signature = (a, b, max_distance=None))]
fn levenshtein(a: &str, b: &str, max_distance: Option<usize>) -> usize {
    algorithms::levenshtein::levenshtein_distance_bounded(a, b, max_distance)
        .unwrap_or(usize::MAX)
}

/// Compute normalized Levenshtein similarity (0.0 to 1.0).
#[pyfunction]
fn levenshtein_similarity(a: &str, b: &str) -> f64 {
    algorithms::levenshtein::levenshtein_similarity(a, b)
}

/// Compute Damerau-Levenshtein distance (includes transpositions).
///
/// Returns `usize::MAX` if distance exceeds `max_distance` threshold.
#[pyfunction]
#[pyo3(signature = (a, b, max_distance=None))]
fn damerau_levenshtein(a: &str, b: &str, max_distance: Option<usize>) -> usize {
    algorithms::damerau::damerau_levenshtein_distance_bounded(a, b, max_distance)
        .unwrap_or(usize::MAX)
}

/// Compute normalized Damerau-Levenshtein similarity.
#[pyfunction]
fn damerau_levenshtein_similarity(a: &str, b: &str) -> f64 {
    algorithms::damerau::damerau_levenshtein_similarity(a, b)
}

/// Compute Jaro similarity (0.0 to 1.0).
#[pyfunction]
fn jaro_similarity(a: &str, b: &str) -> f64 {
    algorithms::jaro::jaro_similarity(a, b)
}

/// Compute Jaro-Winkler similarity (0.0 to 1.0).
///
/// # Arguments
/// * `prefix_weight` - Weight for common prefix bonus (must be in [0.0, 0.25])
/// * `max_prefix_length` - Maximum prefix length to consider
#[pyfunction]
#[pyo3(signature = (a, b, prefix_weight=0.1, max_prefix_length=4))]
fn jaro_winkler_similarity(a: &str, b: &str, prefix_weight: f64, max_prefix_length: usize) -> PyResult<f64> {
    validate_prefix_weight(prefix_weight)?;
    Ok(algorithms::jaro::jaro_winkler_similarity_params(a, b, prefix_weight, max_prefix_length))
}

/// Compute Hamming distance (strings must have equal length).
#[pyfunction]
fn hamming(a: &str, b: &str) -> PyResult<usize> {
    algorithms::hamming::hamming_distance(a, b)
        .ok_or_else(|| PyValueError::new_err("Strings must have equal length for Hamming distance"))
}

/// Compute n-gram similarity (Sørensen-Dice coefficient).
#[pyfunction]
#[pyo3(signature = (a, b, ngram_size=2, pad=true))]
fn ngram_similarity(a: &str, b: &str, ngram_size: usize, pad: bool) -> f64 {
    algorithms::ngram::ngram_similarity(a, b, ngram_size, pad, ' ')
}

/// Compute n-gram Jaccard similarity.
#[pyfunction]
#[pyo3(signature = (a, b, ngram_size=2, pad=true))]
fn ngram_jaccard(a: &str, b: &str, ngram_size: usize, pad: bool) -> f64 {
    algorithms::ngram::ngram_jaccard_similarity(a, b, ngram_size, pad, ' ')
}

/// Extract n-grams from a string.
#[pyfunction]
#[pyo3(signature = (s, ngram_size=2, pad=true))]
fn extract_ngrams(s: &str, ngram_size: usize, pad: bool) -> Vec<String> {
    algorithms::ngram::extract_ngrams(s, ngram_size, pad, ' ')
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

/// Compute LCS (Longest Common Subsequence) length.
#[pyfunction]
fn lcs_length(a: &str, b: &str) -> usize {
    algorithms::lcs::lcs_length(a, b)
}

/// Get the actual LCS string.
#[pyfunction]
fn lcs_string(a: &str, b: &str) -> String {
    algorithms::lcs::lcs_string(a, b)
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
#[pyfunction]
fn longest_common_substring(a: &str, b: &str) -> String {
    algorithms::lcs::longest_common_substring(a, b)
}

/// Compute character-level cosine similarity.
#[pyfunction]
fn cosine_similarity_chars(a: &str, b: &str) -> f64 {
    algorithms::cosine::cosine_similarity_chars(a, b)
}

/// Compute word-level cosine similarity.
#[pyfunction]
fn cosine_similarity_words(a: &str, b: &str) -> f64 {
    algorithms::cosine::cosine_similarity_words(a, b)
}

/// Compute n-gram cosine similarity.
#[pyfunction]
#[pyo3(signature = (a, b, ngram_size=2))]
fn cosine_similarity_ngrams(a: &str, b: &str, ngram_size: usize) -> f64 {
    algorithms::cosine::cosine_similarity_ngrams(a, b, ngram_size)
}

// ============================================================================
// Batch Processing Functions
// ============================================================================

/// Compute Levenshtein distances for all pairs in parallel.
///
/// Releases the Python GIL during computation to allow other threads to run.
#[pyfunction]
fn batch_levenshtein(py: Python<'_>, strings: Vec<String>, query: &str) -> Vec<MatchResult> {
    use rayon::prelude::*;

    // Release GIL during parallel computation
    let query = query.to_string();
    py.allow_threads(|| {
        strings
            .par_iter()
            .map(|s| {
                let score = algorithms::levenshtein::levenshtein_similarity(s, &query);
                MatchResult {
                    text: s.clone(),
                    score,
                }
            })
            .collect()
    })
}

/// Compute Jaro-Winkler similarities for all pairs in parallel.
///
/// Releases the Python GIL during computation to allow other threads to run.
#[pyfunction]
fn batch_jaro_winkler(py: Python<'_>, strings: Vec<String>, query: &str) -> Vec<MatchResult> {
    use rayon::prelude::*;

    // Release GIL during parallel computation
    let query = query.to_string();
    py.allow_threads(|| {
        strings
            .par_iter()
            .map(|s| {
                let score = algorithms::jaro::jaro_winkler_similarity(s, &query);
                MatchResult {
                    text: s.clone(),
                    score,
                }
            })
            .collect()
    })
}

/// Find best matches from a list using specified algorithm.
///
/// Releases the Python GIL during computation to allow other threads to run.
#[pyfunction]
#[pyo3(signature = (strings, query, algorithm="jaro_winkler", limit=10, min_similarity=0.0))]
fn find_best_matches(
    py: Python<'_>,
    strings: Vec<String>,
    query: &str,
    algorithm: &str,
    limit: usize,
    min_similarity: f64,
) -> PyResult<Vec<MatchResult>> {
    use rayon::prelude::*;

    validate_similarity(min_similarity, "min_similarity")?;

    let similarity_fn = get_similarity_fn(algorithm)?;

    // Release GIL during parallel computation
    let query = query.to_string();
    let mut results: Vec<MatchResult> = py.allow_threads(|| {
        strings
            .par_iter()
            .map(|s| {
                let score = similarity_fn(s, &query);
                MatchResult {
                    text: s.clone(),
                    score,
                }
            })
            .filter(|r| r.score >= min_similarity)
            .collect()
    });

    // Validate scores in all builds (not just debug)
    for r in &results {
        if r.score.is_nan() {
            return Err(PyValueError::new_err(format!(
                "NaN score detected for text: '{}'. This indicates a bug in the similarity algorithm.",
                r.text
            )));
        }
    }

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);

    Ok(results)
}

use std::borrow::Cow;

/// Helper to convert string to lowercase only if necessary (Copy-on-Write)
fn to_lowercase_cow(s: &str) -> Cow<'_, str> {
    if s.chars().any(char::is_uppercase) {
        Cow::Owned(s.to_lowercase())
    } else {
        Cow::Borrowed(s)
    }
}

// ============================================================================
// Case-Insensitive Variants
// ============================================================================

/// Case-insensitive Levenshtein distance.
///
/// Returns `usize::MAX` if distance exceeds `max_distance` threshold.
#[pyfunction]
#[pyo3(signature = (a, b, max_distance=None))]
fn levenshtein_ci(a: &str, b: &str, max_distance: Option<usize>) -> usize {
    let a_lower = to_lowercase_cow(a);
    let b_lower = to_lowercase_cow(b);
    algorithms::levenshtein::levenshtein_distance_bounded(&a_lower, &b_lower, max_distance)
        .unwrap_or(usize::MAX)
}


/// Case-insensitive Levenshtein similarity
#[pyfunction]
fn levenshtein_similarity_ci(a: &str, b: &str) -> f64 {
    let a_lower = to_lowercase_cow(a);
    let b_lower = to_lowercase_cow(b);
    algorithms::levenshtein::levenshtein_similarity(&a_lower, &b_lower)
}


/// Case-insensitive Damerau-Levenshtein distance.
///
/// Returns `usize::MAX` if distance exceeds `max_distance` threshold.
#[pyfunction]
#[pyo3(signature = (a, b, max_distance=None))]
fn damerau_levenshtein_ci(a: &str, b: &str, max_distance: Option<usize>) -> usize {
    let a_lower = to_lowercase_cow(a);
    let b_lower = to_lowercase_cow(b);
    algorithms::damerau::damerau_levenshtein_distance_bounded(&a_lower, &b_lower, max_distance)
        .unwrap_or(usize::MAX)
}


/// Case-insensitive Damerau-Levenshtein similarity
#[pyfunction]
fn damerau_levenshtein_similarity_ci(a: &str, b: &str) -> f64 {
    let a_lower = to_lowercase_cow(a);
    let b_lower = to_lowercase_cow(b);
    algorithms::damerau::damerau_levenshtein_similarity(&a_lower, &b_lower)
}


/// Case-insensitive Jaro similarity
#[pyfunction]
fn jaro_similarity_ci(a: &str, b: &str) -> f64 {
    let a_lower = to_lowercase_cow(a);
    let b_lower = to_lowercase_cow(b);
    algorithms::jaro::jaro_similarity(&a_lower, &b_lower)
}


/// Case-insensitive Jaro-Winkler similarity
#[pyfunction]
#[pyo3(signature = (a, b, prefix_weight=0.1, max_prefix_length=4))]
fn jaro_winkler_similarity_ci(a: &str, b: &str, prefix_weight: f64, max_prefix_length: usize) -> PyResult<f64> {
    validate_prefix_weight(prefix_weight)?;
    let a_lower = to_lowercase_cow(a);
    let b_lower = to_lowercase_cow(b);
    Ok(algorithms::jaro::jaro_winkler_similarity_params(&a_lower, &b_lower, prefix_weight, max_prefix_length))
}


/// Case-insensitive n-gram similarity
#[pyfunction]
#[pyo3(signature = (a, b, ngram_size=2, pad=true))]
fn ngram_similarity_ci(a: &str, b: &str, ngram_size: usize, pad: bool) -> f64 {
    let a_lower = to_lowercase_cow(a);
    let b_lower = to_lowercase_cow(b);
    algorithms::ngram::ngram_similarity(&a_lower, &b_lower, ngram_size, pad, ' ')
}


// ============================================================================
// String Normalization
// ============================================================================

#[pyfunction]
fn normalize_string(s: &str, mode: &str) -> PyResult<String> {
    let norm_mode = parse_normalization(mode)?;
    Ok(algorithms::normalize::normalize_string(s, norm_mode))
}

#[pyfunction]
fn normalize_pair(a: &str, b: &str, mode: &str) -> PyResult<(String, String)> {
    let norm_mode = parse_normalization(mode)?;
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
        },
        _ => return Err(PyValueError::new_err(format!(
            "Unknown deduplication method: '{}'. Valid: auto, brute_force, snm",
            method
        ))),
    };

    // Parse normalization mode
    let norm_mode = match normalize.to_lowercase().as_str() {
        "none" | "false" => None,
        "true" | "lowercase" => Some(NormalizationMode::Lowercase),
        "unicode_nfkd" | "nfkd" => Some(NormalizationMode::UnicodeNFKD),
        "remove_punctuation" => Some(NormalizationMode::RemovePunctuation),
        "remove_whitespace" => Some(NormalizationMode::RemoveWhitespace),
        "strict" => Some(NormalizationMode::Strict),
        _ => return Err(PyValueError::new_err(format!(
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
    let result = dedup::find_duplicates_with_metric(&processed_items, metric.as_ref(), min_similarity, dedup_method);

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
                    .map(|indices| indices.iter().map(|&i| items[i].clone()).collect::<Vec<_>>())
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
        .filter_map(|algo| {
            get_similarity_metric(algo).ok().map(|m| (algo.clone(), m))
        })
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
                    .take(limit)
                    .map(|(text, score)| MatchResult {
                        text: text.clone(),
                        score: *score,
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
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    results
}

// ============================================================================
// Python Index Classes
// ============================================================================

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
    
    /// Search for strings within a given edit distance.
    fn search(&self, py: Python<'_>, query: &str, max_distance: usize) -> Vec<SearchResult> {
        // Release GIL during search
        let query = query.to_string();
        let results = py.allow_threads(|| {
            self.inner.search(&query, max_distance)
        });
        
        results
            .into_iter()
            .map(|r| convert_bktree_result(r, &query))
            .collect()
    }

    /// Find the k nearest neighbors.
    fn find_nearest(&self, py: Python<'_>, query: &str, k: usize) -> Vec<SearchResult> {
        let query = query.to_string();
        let results = py.allow_threads(|| {
            self.inner.find_nearest(&query, k)
        });

        results
            .into_iter()
            .map(|r| convert_bktree_result(r, &query))
            .collect()
    }
    
    /// Check if the tree contains an exact match.
    fn contains(&self, query: &str) -> bool {
        self.inner.contains(query)
    }
    
    /// Get the number of items in the tree.
    fn __len__(&self) -> usize {
        self.inner.len()
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
    #[pyo3(signature = (ngram_size=2, min_similarity=0.0))]
    fn new(ngram_size: usize, min_similarity: f64) -> PyResult<Self> {
        if ngram_size < 2 {
            return Err(PyValueError::new_err(format!(
                "ngram_size must be >= 2, got {}",
                ngram_size
            )));
        }
        validate_similarity(min_similarity, "min_similarity")?;
        Ok(Self {
            inner: indexing::ngram_index::NgramIndex::with_min_similarity(ngram_size, min_similarity),
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
    
    /// Search with similarity scoring.
    #[pyo3(signature = (query, algorithm="jaro_winkler", min_similarity=0.0, limit=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        algorithm: &str,
        min_similarity: f64,
        limit: Option<usize>,
    ) -> PyResult<Vec<SearchResult>> {
        validate_similarity(min_similarity, "min_similarity")?;

        let metric = get_similarity_metric(algorithm)?;
        
        let query = query.to_string();
        let results = py.allow_threads(|| {
            self.inner.search_parallel(&query, metric.as_ref(), min_similarity, limit)
        });

        Ok(results
            .into_iter()
            .map(convert_search_match)
            .collect())
    }

    /// Batch search for multiple queries.
    #[pyo3(signature = (queries, algorithm="jaro_winkler", min_similarity=0.0, limit=None))]
    fn batch_search(
        &self,
        py: Python<'_>,
        queries: Vec<String>,
        algorithm: &str,
        min_similarity: f64,
        limit: Option<usize>,
    ) -> PyResult<Vec<Vec<SearchResult>>> {
        let metric = get_similarity_metric(algorithm)?;
        
        let results = py.allow_threads(|| {
            self.inner.batch_search(&queries, metric.as_ref(), min_similarity, limit)
        });

        Ok(results
            .into_iter()
            .map(|matches| {
                matches
                    .into_iter()
                    .map(convert_search_match)
                    .collect()
            })
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

    /// Find the k nearest neighbors by similarity.
    #[pyo3(signature = (query, k, algorithm="jaro_winkler"))]
    fn find_nearest(&self, py: Python<'_>, query: &str, k: usize, algorithm: &str) -> PyResult<Vec<SearchResult>> {
        // Use search with no minimum similarity and limit to k results
        self.search(py, query, algorithm, 0.0, Some(k))
    }

    /// Check if the index contains an exact match.
    fn contains(&self, query: &str) -> bool {
        self.inner.contains(query)
    }

    /// Get number of indexed items.
    fn __len__(&self) -> usize {
        self.inner.len()
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
    #[pyo3(signature = (ngram_size=3))]
    fn new(ngram_size: usize) -> PyResult<Self> {
        if ngram_size < 2 {
            return Err(PyValueError::new_err(format!(
                "ngram_size must be >= 2, got {}",
                ngram_size
            )));
        }
        Ok(Self {
            inner: indexing::ngram_index::HybridIndex::new(ngram_size),
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
    
    #[pyo3(signature = (query, algorithm="jaro_winkler", min_similarity=0.0, limit=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        algorithm: &str,
        min_similarity: f64,
        limit: Option<usize>,
    ) -> PyResult<Vec<SearchResult>> {
        validate_similarity(min_similarity, "min_similarity")?;

        let metric = get_similarity_metric(algorithm)?;
        
        let query = query.to_string();
        let results = py.allow_threads(|| {
            self.inner.search(&query, metric.as_ref(), min_similarity, limit)
        });

        Ok(results
            .into_iter()
            .map(convert_search_match)
            .collect())
    }

    /// Batch search for multiple queries (parallel processing).
    #[pyo3(signature = (queries, algorithm="jaro_winkler", min_similarity=0.0, limit=None))]
    fn batch_search(
        &self,
        py: Python<'_>,
        queries: Vec<String>,
        algorithm: &str,
        min_similarity: f64,
        limit: Option<usize>,
    ) -> PyResult<Vec<Vec<SearchResult>>> {
        let metric = get_similarity_metric(algorithm)?;
        
        let results = py.allow_threads(|| {
            self.inner.batch_search(&queries, metric.as_ref(), min_similarity, limit)
        });

        Ok(results
            .into_iter()
            .map(|matches| {
                matches
                    .into_iter()
                    .map(convert_search_match)
                    .collect()
            })
            .collect())
    }

    /// Find the k nearest neighbors by similarity.
    #[pyo3(signature = (query, k, algorithm="jaro_winkler"))]
    fn find_nearest(&self, py: Python<'_>, query: &str, k: usize, algorithm: &str) -> PyResult<Vec<SearchResult>> {
        // Use search with no minimum similarity and limit to k results
        self.search(py, query, algorithm, 0.0, Some(k))
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
// Schema-Based Multi-Field Matching
// ============================================================================

use indexing::schema;
use algorithms::normalize::NormalizationMode;

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
            self.id, self.score, self.record.len()
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
        self.inner.required_fields().into_iter().map(|s| s.to_string()).collect()
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
            return Err(PyValueError::new_err(format!(
                "weight must be a finite number, got {}",
                weight
            )));
        }
        if weight < 0.0 || weight > 10.0 {
            return Err(PyValueError::new_err(format!(
                "weight must be in range [0.0, 10.0], got {}",
                weight
            )));
        }

        // Determine default algorithm based on field type if not provided
        let default_algo = match field_type.to_lowercase().as_str() {
            "short_text" => "jaro_winkler",
            "long_text" => "ngram",
            "token_set" => "jaccard",
            _ => return Err(PyValueError::new_err(format!("Unknown field type: {}", field_type))),
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
                    return Err(PyValueError::new_err("separator cannot be empty"));
                }
                schema::types::FieldType::TokenSet {
                    separator: sep_str.to_string(),
                    default_algorithm: parsed_algorithm.clone(),
                }
            },
            _ => return Err(PyValueError::new_err(format!("Unknown field type: {}", field_type))),
        };

        // Parse normalization
        let norm = if let Some(n) = normalize {
            Some(parse_normalization(n)?)
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
            _ => return Err(PyValueError::new_err(format!("Unknown scoring strategy: {}", strategy))),
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
            Err(e) => Err(PyValueError::new_err(format!("Schema validation failed: {}", e))),
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
    fn add(&mut self, record: std::collections::HashMap<String, String>, data: Option<u64>) -> PyResult<usize> {
        let mut rec = schema::Record::new();
        for (key, value) in record {
            rec.set_field(key, value);
        }

        match self.inner.add_record(rec, data) {
            Ok(id) => Ok(id),
            Err(e) => Err(PyValueError::new_err(format!("Failed to add record: {}", e))),
        }
    }

    /// Search for matching records
    #[pyo3(signature = (query, min_score=0.0, limit=None, min_field_score=0.0))]
    fn search(
        &self,
        query: std::collections::HashMap<String, String>,
        min_score: f64,
        limit: Option<usize>,
        min_field_score: f64,
    ) -> PyResult<Vec<SchemaSearchResult>> {
        validate_similarity(min_score, "min_score")?;
        validate_similarity(min_field_score, "min_field_score")?;

        let mut q = schema::Record::new();
        for (key, value) in query {
            q.set_field(key, value);
        }

        let options = schema::SearchOptions {
            min_score,
            limit,
            min_field_score,
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
            Err(e) => Err(PyValueError::new_err(format!("Search failed: {}", e))),
        }
    }

    /// Get a record by ID
    ///
    /// Returns None if the record doesn't exist, or raises an error if there's a storage problem.
    fn get(&self, id: usize) -> PyResult<Option<std::collections::HashMap<String, String>>> {
        match self.inner.get_record(id) {
            Ok(Some(r)) => Ok(Some(r.fields.into_iter().collect())),
            Ok(None) => Ok(None),
            Err(e) => Err(PyValueError::new_err(format!("Failed to get record: {}", e))),
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
        _ => Err(PyValueError::new_err(format!(
            "Unknown algorithm: '{}'. Valid algorithms: levenshtein, damerau_levenshtein, jaro_winkler, ngram, jaccard, cosine, exact_match",
            algo
        ))),
    }
}

fn parse_normalization(norm: &str) -> PyResult<NormalizationMode> {
    match norm.to_lowercase().as_str() {
        "lowercase" => Ok(NormalizationMode::Lowercase),
        "unicode_nfkd" | "nfkd" => Ok(NormalizationMode::UnicodeNFKD),
        "remove_punctuation" => Ok(NormalizationMode::RemovePunctuation),
        "remove_whitespace" => Ok(NormalizationMode::RemoveWhitespace),
        "strict" => Ok(NormalizationMode::Strict),
        _ => Err(PyValueError::new_err(format!("Unknown normalization mode: {}", norm))),
    }
}

// ============================================================================
// Python Module
// ============================================================================

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Result types
    m.add_class::<SearchResult>()?;
    m.add_class::<MatchResult>()?;
    m.add_class::<DeduplicationResult>()?;
    m.add_class::<AlgorithmComparison>()?;

    // Distance/similarity functions
    m.add_function(wrap_pyfunction!(levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(hamming, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(extract_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(soundex, m)?)?;
    m.add_function(wrap_pyfunction!(soundex_match, m)?)?;
    m.add_function(wrap_pyfunction!(metaphone, m)?)?;
    m.add_function(wrap_pyfunction!(metaphone_match, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_length, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_string, m)?)?;
    m.add_function(wrap_pyfunction!(lcs_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(longest_common_substring_length, m)?)?;
    m.add_function(wrap_pyfunction!(longest_common_substring, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_chars, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_words, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity_ngrams, m)?)?;
    
    // Batch processing
    m.add_function(wrap_pyfunction!(batch_levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(batch_jaro_winkler, m)?)?;
    m.add_function(wrap_pyfunction!(find_best_matches, m)?)?;

    // Case-insensitive variants
    m.add_function(wrap_pyfunction!(levenshtein_ci, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_similarity_ci, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_ci, m)?)?;
    m.add_function(wrap_pyfunction!(damerau_levenshtein_similarity_ci, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_similarity_ci, m)?)?;
    m.add_function(wrap_pyfunction!(jaro_winkler_similarity_ci, m)?)?;
    m.add_function(wrap_pyfunction!(ngram_similarity_ci, m)?)?;

    // Normalization
    m.add_function(wrap_pyfunction!(normalize_string, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_pair, m)?)?;

    // Deduplication
    m.add_function(wrap_pyfunction!(find_duplicates, m)?)?;

    // Multi-algorithm comparison
    m.add_function(wrap_pyfunction!(compare_algorithms, m)?)?;

    // Index classes
    m.add_class::<PyBkTree>()?;
    m.add_class::<PyNgramIndex>()?;
    m.add_class::<PyHybridIndex>()?;

    // Schema-based multi-field matching
    m.add_class::<Schema>()?;
    m.add_class::<SchemaBuilder>()?;
    m.add_class::<SchemaIndex>()?;
    m.add_class::<SchemaSearchResult>()?;

    Ok(())
}
