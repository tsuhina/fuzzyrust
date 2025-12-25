//! Cosine similarity implementation
//!
//! Treats strings as vectors and computes the cosine of the angle between them.
//! Works at both character and word levels.
//!
//! # Complexity
//! - Time: O(m+n) for building frequency maps and computing similarity
//! - Space: O(unique_tokens) for frequency maps

use super::Similarity;
use ahash::AHashMap;

/// Character-level cosine similarity calculator
///
/// # Configuration
/// - `ngram_size`: Use n-grams instead of single characters (None = single chars)
/// - `word_level`: Use words instead of characters (overrides ngram_size)
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CosineSimilarity {
    /// Whether to use character n-grams instead of single characters
    pub ngram_size: Option<usize>,
    /// Whether to use word-level instead of character-level
    pub word_level: bool,
}

impl CosineSimilarity {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn character_based() -> Self {
        Self {
            ngram_size: None,
            word_level: false,
        }
    }
    
    pub fn word_based() -> Self {
        Self {
            ngram_size: None,
            word_level: true,
        }
    }
    
    pub fn ngram_based(n: usize) -> Self {
        Self {
            ngram_size: Some(n),
            word_level: false,
        }
    }
}

impl Similarity for CosineSimilarity {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        if self.word_level {
            cosine_similarity_words(a, b)
        } else if let Some(n) = self.ngram_size {
            cosine_similarity_ngrams(a, b, n)
        } else {
            cosine_similarity_chars(a, b)
        }
    }
    
    fn name(&self) -> &'static str {
        "cosine"
    }
}

/// Builds a frequency map from an iterator of items.
///
/// Counts occurrences of each unique item for constructing
/// term frequency vectors used in cosine similarity calculation.
fn build_frequency_map<T, I>(iter: I) -> AHashMap<T, usize>
where
    T: std::hash::Hash + Eq,
    I: Iterator<Item = T>,
{
    let mut map = AHashMap::new();
    for item in iter {
        *map.entry(item).or_insert(0) += 1;
    }
    map
}

/// Calculate cosine similarity between two frequency maps.
fn cosine_from_maps<T: std::hash::Hash + Eq>(
    map_a: &AHashMap<T, usize>,
    map_b: &AHashMap<T, usize>,
) -> f64 {
    if map_a.is_empty() && map_b.is_empty() {
        return 1.0;
    }
    
    if map_a.is_empty() || map_b.is_empty() {
        return 0.0;
    }
    
    let mut dot_product = 0.0f64;
    let mut magnitude_a = 0.0f64;
    let mut magnitude_b = 0.0f64;
    
    // Calculate magnitude of A and dot product for common elements
    for (key, &count_a) in map_a {
        let count_a = count_a as f64;
        magnitude_a += count_a * count_a;
        
        if let Some(&count_b) = map_b.get(key) {
            dot_product += count_a * count_b as f64;
        }
    }
    
    // Calculate magnitude of B
    for &count_b in map_b.values() {
        let count_b = count_b as f64;
        magnitude_b += count_b * count_b;
    }
    
    let magnitude = (magnitude_a * magnitude_b).sqrt();
    
    if magnitude == 0.0 {
        0.0
    } else {
        dot_product / magnitude
    }
}

/// Character-level cosine similarity.
pub fn cosine_similarity_chars(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    
    let map_a = build_frequency_map(a.chars());
    let map_b = build_frequency_map(b.chars());
    
    cosine_from_maps(&map_a, &map_b)
}

/// Word-level cosine similarity.
pub fn cosine_similarity_words(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    
    let map_a = build_frequency_map(
        a.split_whitespace().map(|s| s.to_lowercase())
    );
    let map_b = build_frequency_map(
        b.split_whitespace().map(|s| s.to_lowercase())
    );
    
    cosine_from_maps(&map_a, &map_b)
}

/// Maximum valid n-gram size (consistent with ngram module)
const MAX_NGRAM_SIZE: usize = 32;

/// Build frequency map directly from n-gram windows (avoids intermediate Vec allocation)
///
/// # Arguments
/// * `n` - N-gram size (1-32). Values of 0 return empty map, values >32 are clamped.
fn build_ngram_frequency_map_direct(s: &str, n: usize, pad: bool, pad_char: char) -> AHashMap<String, usize> {
    let mut map = AHashMap::new();
    if n == 0 {
        return map;
    }
    // Clamp n-gram size to valid range
    let n = n.min(MAX_NGRAM_SIZE);

    let chars: Vec<char> = if pad {
        let padding: String = std::iter::repeat(pad_char).take(n - 1).collect();
        format!("{}{}{}", padding, s, padding).chars().collect()
    } else {
        s.chars().collect()
    };

    if chars.len() < n {
        return map;
    }

    for window in chars.windows(n) {
        let ngram: String = window.iter().collect();
        *map.entry(ngram).or_insert(0) += 1;
    }

    map
}

/// N-gram-level cosine similarity.
/// Returns 0.0 if n is 0 (no valid n-grams can be extracted).
pub fn cosine_similarity_ngrams(a: &str, b: &str, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }

    // Build frequency maps directly without intermediate Vec<String>
    let map_a = build_ngram_frequency_map_direct(a, n, true, ' ');
    let map_b = build_ngram_frequency_map_direct(b, n, true, ' ');

    cosine_from_maps(&map_a, &map_b)
}

/// TF-IDF weighted cosine similarity for a corpus.
///
/// This is an internal implementation not currently exposed via Python bindings.
/// It provides corpus-based TF-IDF weighting for more sophisticated document
/// similarity. May be exposed in a future version.
///
/// # Usage (Rust only)
/// ```ignore
/// let mut tfidf = TfIdfCosine::new();
/// tfidf.add_document("first document");
/// tfidf.add_document("second document");
/// let sim = tfidf.similarity("first doc", "second doc");
/// ```
#[allow(dead_code)]
pub struct TfIdfCosine {
    /// Document frequency for each term
    df: AHashMap<String, usize>,
    /// Total number of documents
    num_docs: usize,
}

impl TfIdfCosine {
    pub fn new() -> Self {
        Self {
            df: AHashMap::new(),
            num_docs: 0,
        }
    }
    
    /// Add a document to build the IDF scores.
    pub fn add_document(&mut self, doc: &str) {
        let terms: std::collections::HashSet<String> = doc
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        
        for term in terms {
            *self.df.entry(term).or_insert(0) += 1;
        }
        self.num_docs += 1;
    }
    
    /// Calculate TF-IDF weighted cosine similarity.
    pub fn similarity(&self, a: &str, b: &str) -> f64 {
        if a == b {
            return 1.0;
        }
        
        let tfidf_a = self.compute_tfidf(a);
        let tfidf_b = self.compute_tfidf(b);
        
        if tfidf_a.is_empty() && tfidf_b.is_empty() {
            return 1.0;
        }
        
        if tfidf_a.is_empty() || tfidf_b.is_empty() {
            return 0.0;
        }
        
        let mut dot_product = 0.0f64;
        let mut magnitude_a = 0.0f64;
        let mut magnitude_b = 0.0f64;
        
        for (term, &weight_a) in &tfidf_a {
            magnitude_a += weight_a * weight_a;
            if let Some(&weight_b) = tfidf_b.get(term) {
                dot_product += weight_a * weight_b;
            }
        }
        
        for &weight_b in tfidf_b.values() {
            magnitude_b += weight_b * weight_b;
        }
        
        let magnitude = (magnitude_a * magnitude_b).sqrt();
        
        if magnitude == 0.0 {
            0.0
        } else {
            dot_product / magnitude
        }
    }
    
    fn compute_tfidf(&self, doc: &str) -> AHashMap<String, f64> {
        let terms: Vec<String> = doc
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        
        let tf = build_frequency_map(terms.iter().cloned());
        let doc_len = terms.len() as f64;
        
        let mut tfidf = AHashMap::new();
        
        for (term, count) in tf {
            let tf_score = count as f64 / doc_len;
            let idf_score = if let Some(&df) = self.df.get(&term) {
                ((self.num_docs as f64 + 1.0) / (df as f64 + 1.0)).ln() + 1.0
            } else {
                1.0
            };
            tfidf.insert(term, tf_score * idf_score);
        }
        
        tfidf
    }
}

impl Default for TfIdfCosine {
    fn default() -> Self {
        Self::new()
    }
}

/// Implement Similarity trait for TfIdfCosine
///
/// This allows TfIdfCosine to be used with generic code that accepts
/// any Similarity metric, such as NgramIndex.
impl super::Similarity for TfIdfCosine {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        TfIdfCosine::similarity(self, a, b)
    }

    fn name(&self) -> &'static str {
        "tfidf_cosine"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 0.01
    }
    
    #[test]
    fn test_cosine_chars() {
        assert!(approx_eq(cosine_similarity_chars("abc", "abc"), 1.0));
        assert!(approx_eq(cosine_similarity_chars("abc", "def"), 0.0));
    }
    
    #[test]
    fn test_cosine_words() {
        let a = "the quick brown fox";
        let b = "the quick brown dog";
        let sim = cosine_similarity_words(a, b);
        assert!(sim > 0.5 && sim < 1.0);
    }
    
    #[test]
    fn test_tfidf() {
        let mut tfidf = TfIdfCosine::new();
        tfidf.add_document("hello world");
        tfidf.add_document("hello there");
        tfidf.add_document("world news");
        
        let sim = tfidf.similarity("hello world", "hello there");
        assert!(sim > 0.0 && sim < 1.0);
    }
}
