//! N-gram index for fast fuzzy search candidate filtering
//! 
//! Pre-indexes strings by their n-grams for O(1) candidate lookup.
//! Much faster than BK-tree for large datasets when combined with
//! a secondary similarity check.

use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;
use crate::algorithms::Similarity;

/// Entry in the n-gram index
#[derive(Debug, Clone)]
pub struct IndexEntry {
    pub id: usize,
    pub text: String,
    pub data: Option<u64>,
}

/// N-gram based index for fast fuzzy search
#[derive(Debug, Clone)]
pub struct NgramIndex {
    /// Size of n-grams
    n: usize,
    /// Map from n-gram hash to list of entry IDs containing it
    index: AHashMap<u64, Vec<usize>>,
    /// All indexed entries
    entries: Vec<IndexEntry>,
    /// Minimum similarity threshold for candidates
    min_similarity: f64,
    /// Map from string hash to entry ID for O(1) exact match lookup.
    /// Uses hash of text to avoid duplicating string storage.
    exact_lookup: AHashMap<u64, usize>,
}

/// Maximum valid n-gram size for index construction
const MAX_NGRAM_SIZE: usize = 32;

impl NgramIndex {
    /// Create a new n-gram index
    ///
    /// # Arguments
    /// * `n` - N-gram size (1-32). Values of 0 are treated as 1, values >32 are clamped to 32.
    pub fn new(n: usize) -> Self {
        // Validate and clamp n-gram size
        let n = if n == 0 { 1 } else { n.min(MAX_NGRAM_SIZE) };
        Self {
            n,
            index: AHashMap::new(),
            entries: Vec::new(),
            min_similarity: 0.0,
            exact_lookup: AHashMap::new(),
        }
    }

    /// Create with minimum similarity threshold
    ///
    /// # Arguments
    /// * `n` - N-gram size (1-32). Values of 0 are treated as 1, values >32 are clamped to 32.
    /// * `min_similarity` - Minimum similarity score for search results
    pub fn with_min_similarity(n: usize, min_similarity: f64) -> Self {
        // Validate and clamp n-gram size
        let n = if n == 0 { 1 } else { n.min(MAX_NGRAM_SIZE) };
        Self {
            n,
            index: AHashMap::new(),
            entries: Vec::new(),
            min_similarity,
            exact_lookup: AHashMap::new(),
        }
    }

    /// Compute a hash for a string (for exact lookup)
    #[inline]
    fn hash_string(s: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = ahash::AHasher::default();
        s.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Add a string to the index
    pub fn add(&mut self, text: impl Into<String>) -> usize {
        self.add_with_data(text, None)
    }
    
    /// Add a string with associated data
    pub fn add_with_data(&mut self, text: impl Into<String>, data: Option<u64>) -> usize {
        let text = text.into();
        let id = self.entries.len();

        // Deduplicate n-grams to avoid adding same ID multiple times
        // (e.g., "aaa" with n=2 produces ["aa", "aa"])
        let ngrams: AHashSet<String> = extract_ngrams(&text, self.n).into_iter().collect();

        for ngram in ngrams {
            let hash = Self::hash_string(&ngram);
            self.index
                .entry(hash)
                .or_insert_with(Vec::new)
                .push(id);
        }

        // Store hash -> id mapping for O(1) contains lookup (avoids duplicate storage)
        let hash = Self::hash_string(&text);
        self.exact_lookup.insert(hash, id);

        self.entries.push(IndexEntry { id, text, data });
        id
    }
    
    /// Add multiple strings
    pub fn add_all<I, S>(&mut self, iter: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for text in iter {
            self.add(text);
        }
    }
    
    /// Get candidates that share at least one n-gram with the query
    pub fn get_candidates(&self, query: &str) -> Vec<usize> {
        let query_ngrams = extract_ngrams(query, self.n);
        let mut candidate_ids = AHashSet::new();

        for ngram in &query_ngrams {
            let hash = Self::hash_string(ngram);
            if let Some(ids) = self.index.get(&hash) {
                candidate_ids.extend(ids.iter().copied());
            }
        }

        candidate_ids.into_iter().collect()
    }
    
    /// Get candidates with a minimum n-gram overlap ratio
    pub fn get_candidates_with_min_ratio(&self, query: &str, min_ratio: f64) -> Vec<usize> {
        let query_ngrams: AHashSet<String> = extract_ngrams(query, self.n).into_iter().collect();
        let query_ngram_count = query_ngrams.len();

        if query_ngram_count == 0 {
            return Vec::new();
        }

        // Count how many query n-grams each candidate has
        let mut candidate_counts: AHashMap<usize, usize> = AHashMap::new();

        for ngram in &query_ngrams {
            let hash = Self::hash_string(ngram);
            if let Some(ids) = self.index.get(&hash) {
                for &id in ids {
                    *candidate_counts.entry(id).or_insert(0) += 1;
                }
            }
        }

        // Filter by ratio
        let min_matches = (query_ngram_count as f64 * min_ratio).ceil() as usize;

        candidate_counts
            .into_iter()
            .filter(|&(_, count)| count >= min_matches)
            .map(|(id, _)| id)
            .collect()
    }
    
    /// Get text for a specific ID
    pub fn get_text(&self, id: usize) -> Option<String> {
        self.entries.get(id).map(|e| e.text.clone())
    }
    
    /// Search with a similarity function
    pub fn search<S: Similarity + ?Sized>(
        &self,
        query: &str,
        similarity: &S,
        min_similarity: f64,
        limit: Option<usize>,
    ) -> Vec<SearchMatch> {
        let candidates = self.get_candidates(query);
        
        let mut matches: Vec<SearchMatch> = candidates
            .into_iter()
            .filter_map(|id| {
                if let Some(entry) = self.entries.get(id) {
                     let sim = similarity.similarity(query, &entry.text);
                     if sim >= min_similarity.max(self.min_similarity) {
                        Some(SearchMatch {
                            id: entry.id,
                            text: entry.text.clone(),
                            similarity: sim,
                            data: entry.data,
                        })
                     } else {
                         None
                     }
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by similarity descending
        #[cfg(debug_assertions)]
        for m in &matches {
            debug_assert!(!m.similarity.is_nan(), "NaN similarity detected for text: {}", m.text);
        }
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(limit) = limit {
            matches.truncate(limit);
        }

        matches
    }

    /// Parallel search for large datasets
    pub fn search_parallel<S: Similarity + Send + Sync + ?Sized>(
        &self,
        query: &str,
        similarity: &S,
        min_similarity: f64,
        limit: Option<usize>,
    ) -> Vec<SearchMatch> {
        let candidates = self.get_candidates(query);
        
        let mut matches: Vec<SearchMatch> = candidates
            .into_par_iter()
            .filter_map(|id| {
                if let Some(entry) = self.entries.get(id) {
                    let sim = similarity.similarity(query, &entry.text);
                     if sim >= min_similarity.max(self.min_similarity) {
                        Some(SearchMatch {
                            id: entry.id,
                            text: entry.text.clone(),
                            similarity: sim,
                            data: entry.data,
                        })
                     } else {
                         None
                     }
                } else {
                    None
                }
            })
            .collect();

        #[cfg(debug_assertions)]
        for m in &matches {
            debug_assert!(!m.similarity.is_nan(), "NaN similarity detected for text: {}", m.text);
        }
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(limit) = limit {
            matches.truncate(limit);
        }

        matches
    }

    /// Batch search for multiple queries
    pub fn batch_search<S: Similarity + Send + Sync + ?Sized>(
        &self,
        queries: &[String],
        similarity: &S,
        min_similarity: f64,
        limit: Option<usize>,
    ) -> Vec<Vec<SearchMatch>> {
        queries
            .par_iter()
            .map(|query| self.search_parallel(query, similarity, min_similarity, limit))
            .collect()
    }
    
    /// Get index size
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check if the index contains an exact match for the query
    pub fn contains(&self, query: &str) -> bool {
        let hash = Self::hash_string(query);
        // Check if hash exists and verify the actual text matches (handle hash collisions)
        self.exact_lookup.get(&hash)
            .map(|&id| self.entries.get(id).map(|e| e.text == query).unwrap_or(false))
            .unwrap_or(false)
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.index.clear();
        self.entries.clear();
        self.exact_lookup.clear();
    }
    
    /// Get entry by ID
    pub fn get(&self, id: usize) -> Option<&IndexEntry> {
        self.entries.get(id)
    }
}

/// Result from a similarity search
#[derive(Debug, Clone)]
pub struct SearchMatch {
    pub id: usize,
    pub text: String,
    pub similarity: f64,
    pub data: Option<u64>,
}

/// Extract n-grams from a string.
/// Returns empty vec if string is shorter than n or if n is 0.
/// Note: This is consistent with algorithms/ngram.rs behavior.
fn extract_ngrams(s: &str, n: usize) -> Vec<String> {
    if n == 0 {
        return vec![];
    }

    let chars: Vec<char> = s.chars().collect();

    if chars.len() < n {
        return vec![];
    }

    chars
        .windows(n)
        .map(|w| w.iter().collect())
        .collect()
}

/// Combined index using both n-grams and BK-tree for optimal performance
pub struct HybridIndex {
    ngram_index: NgramIndex,
}

impl HybridIndex {
    pub fn new(ngram_size: usize) -> Self {
        Self {
            ngram_index: NgramIndex::new(ngram_size),
        }
    }
    
    pub fn add(&mut self, text: impl Into<String>) -> usize {
        self.ngram_index.add(text)
    }
    
    pub fn add_with_data(&mut self, text: impl Into<String>, data: Option<u64>) -> usize {
        self.ngram_index.add_with_data(text, data)
    }
    
    pub fn search<S: Similarity + Send + Sync + ?Sized>(
        &self,
        query: &str,
        similarity: &S,
        min_similarity: f64,
        limit: Option<usize>,
    ) -> Vec<SearchMatch> {
        // Use parallel search for better performance
        self.ngram_index.search_parallel(query, similarity, min_similarity, limit)
    }
    
    pub fn len(&self) -> usize {
        self.ngram_index.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.ngram_index.is_empty()
    }

    /// Check if the index contains an exact match for the query
    pub fn contains(&self, query: &str) -> bool {
        self.ngram_index.contains(query)
    }

    /// Batch search for multiple queries (parallel processing)
    pub fn batch_search<S: Similarity + Send + Sync + ?Sized>(
        &self,
        queries: &[String],
        similarity: &S,
        min_similarity: f64,
        limit: Option<usize>,
    ) -> Vec<Vec<SearchMatch>> {
        self.ngram_index.batch_search(queries, similarity, min_similarity, limit)
    }
}

impl Default for HybridIndex {
    fn default() -> Self {
        Self::new(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::JaroWinkler;
    
    #[test]
    fn test_ngram_index_basic() {
        let mut index = NgramIndex::new(2);
        index.add_all(["hello", "hallo", "hullo", "world"]);
        
        let candidates = index.get_candidates("hella");
        assert!(!candidates.is_empty());
    }
    
    #[test]
    fn test_ngram_index_search() {
        let mut index = NgramIndex::new(2);
        index.add_all(["hello", "hallo", "hullo", "world", "help"]);
        
        let jw = JaroWinkler::new();
        let results = index.search("hello", &jw, 0.8, Some(3));
        
        assert!(!results.is_empty());
        assert!(results[0].similarity >= results.last().unwrap().similarity);
    }
}
