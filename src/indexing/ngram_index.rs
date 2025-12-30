//! N-gram index for fast fuzzy search candidate filtering
//!
//! Pre-indexes strings by their n-grams for O(1) candidate lookup.
//! Much faster than BK-tree for large datasets when combined with
//! a secondary similarity check.
//!
//! ## Posting List Compression
//!
//! For large indices (100k+ entries), posting lists can consume significant memory.
//! The `compress()` method converts posting lists to a delta + varint encoded format
//! that typically reduces memory usage by 50-70%.

use std::borrow::Cow;

use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use crate::algorithms::Similarity;

// ============================================================================
// Posting List Compression
// ============================================================================

/// A compressed posting list using delta + varint encoding.
///
/// For sorted ID sequences, this typically achieves 50-70% compression:
/// - Delta encoding: Store differences between consecutive IDs
/// - Varint encoding: Use 1-5 bytes per delta (most deltas are small)
///
/// Example: IDs [100, 105, 108, 200]
/// - Deltas: [100, 5, 3, 92]
/// - Varint encoded: ~6 bytes instead of 32 bytes (4 × 8 bytes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedPostingList {
    /// Varint-encoded deltas
    data: Vec<u8>,
    /// Number of IDs in the list
    len: usize,
}

impl CompressedPostingList {
    /// Create a compressed posting list from sorted IDs.
    ///
    /// # Panics
    /// Panics if IDs are not sorted in ascending order.
    pub fn from_sorted_ids(ids: &[usize]) -> Self {
        if ids.is_empty() {
            return Self { data: Vec::new(), len: 0 };
        }

        // Pre-allocate with estimated size (average ~2 bytes per ID for typical data)
        let mut data = Vec::with_capacity(ids.len() * 2);
        let mut prev = 0usize;

        for &id in ids {
            debug_assert!(id >= prev, "IDs must be sorted");
            let delta = id - prev;
            encode_varint(delta, &mut data);
            prev = id;
        }

        Self { data, len: ids.len() }
    }

    /// Decode all IDs from the compressed format.
    pub fn decode(&self) -> Vec<usize> {
        if self.len == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.len);
        let mut pos = 0;
        let mut prev = 0usize;

        while pos < self.data.len() {
            let (delta, bytes_read) = decode_varint(&self.data[pos..]);
            pos += bytes_read;
            prev += delta;
            result.push(prev);
        }

        result
    }

    /// Get the number of IDs in the list.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the list is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the compressed size in bytes.
    #[inline]
    pub fn compressed_size(&self) -> usize {
        self.data.len()
    }

    /// Get the uncompressed size in bytes (if stored as Vec<usize>).
    #[inline]
    pub fn uncompressed_size(&self) -> usize {
        self.len * std::mem::size_of::<usize>()
    }

    /// Get compression ratio (compressed / uncompressed).
    pub fn compression_ratio(&self) -> f64 {
        if self.len == 0 {
            return 1.0;
        }
        self.compressed_size() as f64 / self.uncompressed_size() as f64
    }
}

/// Encode a usize as varint (1-10 bytes, typically 1-3 for posting list deltas).
#[inline]
fn encode_varint(mut value: usize, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Decode a varint from bytes. Returns (value, bytes_consumed).
#[inline]
fn decode_varint(data: &[u8]) -> (usize, usize) {
    let mut result = 0usize;
    let mut shift = 0;
    let mut pos = 0;

    loop {
        let byte = data[pos];
        result |= ((byte & 0x7F) as usize) << shift;
        pos += 1;

        if byte < 0x80 {
            break;
        }
        shift += 7;
    }

    (result, pos)
}

/// Entry in the n-gram index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    pub id: usize,
    pub text: String,
    pub data: Option<u64>,
}

/// N-gram based index for fast fuzzy search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramIndex {
    /// Size of n-grams
    n: usize,
    /// Map from n-gram hash to list of entry IDs containing it (uncompressed)
    index: AHashMap<u64, Vec<usize>>,
    /// Compressed posting lists (populated after calling `compress()`)
    compressed_index: Option<AHashMap<u64, CompressedPostingList>>,
    /// All indexed entries
    entries: Vec<IndexEntry>,
    /// Minimum similarity threshold for candidates
    min_similarity: f64,
    /// Minimum ratio of query n-grams that must match for candidate filtering
    min_ngram_ratio: f64,
    /// Map from string hash to entry ID for O(1) exact match lookup.
    /// Uses hash of text to avoid duplicating string storage.
    exact_lookup: AHashMap<u64, usize>,
    /// Whether to normalize (lowercase) text before indexing and querying.
    /// When true, n-gram extraction uses lowercased text for case-insensitive matching.
    normalize: bool,
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
            compressed_index: None,
            entries: Vec::new(),
            min_similarity: 0.0,
            min_ngram_ratio: 0.0,
            exact_lookup: AHashMap::new(),
            normalize: false,
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
            compressed_index: None,
            entries: Vec::new(),
            min_similarity,
            min_ngram_ratio: 0.0,
            exact_lookup: AHashMap::new(),
            normalize: false,
        }
    }

    /// Create with all parameters
    ///
    /// # Arguments
    /// * `n` - N-gram size (1-32). Values of 0 are treated as 1, values >32 are clamped to 32.
    /// * `min_similarity` - Minimum similarity score for search results
    /// * `min_ngram_ratio` - Minimum ratio of query n-grams that must match (0.0 to 1.0)
    /// * `normalize` - Whether to lowercase text for case-insensitive n-gram matching
    pub fn with_params(n: usize, min_similarity: f64, min_ngram_ratio: f64, normalize: bool) -> Self {
        // Validate and clamp n-gram size
        let n = if n == 0 { 1 } else { n.min(MAX_NGRAM_SIZE) };
        Self {
            n,
            index: AHashMap::new(),
            compressed_index: None,
            entries: Vec::new(),
            min_similarity,
            min_ngram_ratio: min_ngram_ratio.clamp(0.0, 1.0),
            exact_lookup: AHashMap::new(),
            normalize,
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

        // Normalize text for n-gram extraction if enabled (for case-insensitive indexing)
        // Use Cow<str> to avoid unnecessary clones when normalization is disabled
        let normalized: Cow<str> = if self.normalize {
            Cow::Owned(text.to_lowercase())
        } else {
            Cow::Borrowed(&text)
        };

        // Deduplicate n-grams to avoid adding same ID multiple times
        // (e.g., "aaa" with n=2 produces ["aa", "aa"])
        let ngrams: AHashSet<String> = extract_ngrams(&normalized, self.n).into_iter().collect();

        for ngram in ngrams {
            let hash = Self::hash_string(&ngram);
            self.index
                .entry(hash)
                .or_default()
                .push(id);
        }

        // Store hash -> id mapping for O(1) contains lookup
        // Use normalized text for consistent lookup when normalize is enabled
        let hash = Self::hash_string(&normalized);
        self.exact_lookup.insert(hash, id);

        // Store original text in entry (preserves case for display)
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

    /// Add multiple strings in parallel (faster for large batches).
    ///
    /// This method extracts n-grams in parallel using Rayon, then merges
    /// them into the index. For datasets with 1000+ items, this can be
    /// significantly faster than sequential `add_all`.
    ///
    /// For small batches (< 100 items), falls back to sequential processing
    /// to avoid parallel overhead.
    ///
    /// # Example
    /// ```ignore
    /// let mut index = NgramIndex::new(3);
    /// let items: Vec<String> = (0..10000).map(|i| format!("item_{}", i)).collect();
    /// index.add_all_parallel(items);
    /// ```
    pub fn add_all_parallel<I, S>(&mut self, iter: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String> + Send,
    {
        let items: Vec<String> = iter.into_iter().map(|s| s.into()).collect();

        if items.is_empty() {
            return;
        }

        // Only use parallel processing if worth the overhead
        if items.len() < 100 {
            self.add_all(items);
            return;
        }

        let normalize = self.normalize;
        let n = self.n;

        // Extract ngrams in parallel
        // Each item produces: (original_text, normalized_text, ngrams_set)
        let processed: Vec<(String, String, AHashSet<String>)> = items
            .into_par_iter()
            .map(|text| {
                let normalized = if normalize {
                    text.to_lowercase()
                } else {
                    text.clone()
                };
                let ngrams: AHashSet<String> = extract_ngrams(&normalized, n).into_iter().collect();
                (text, normalized, ngrams)
            })
            .collect();

        // Merge into index sequentially (to maintain correct IDs)
        let start_id = self.entries.len();

        for (idx, (text, normalized, ngrams)) in processed.into_iter().enumerate() {
            let id = start_id + idx;

            // Add ngrams to index
            for ngram in ngrams {
                let hash = Self::hash_string(&ngram);
                self.index
                    .entry(hash)
                    .or_default()
                    .push(id);
            }

            // Add to exact lookup
            let hash = Self::hash_string(&normalized);
            self.exact_lookup.insert(hash, id);

            // Add entry
            self.entries.push(IndexEntry { id, text, data: None });
        }
    }

    /// Add multiple strings with data in parallel.
    ///
    /// Similar to `add_all_parallel` but allows associating u64 data with each item.
    /// For small batches (< 100 items), falls back to sequential processing
    /// to avoid parallel overhead.
    pub fn add_all_with_data_parallel<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = (String, Option<u64>)> + Send,
        I::IntoIter: Send,
    {
        let items: Vec<(String, Option<u64>)> = iter.into_iter().collect();

        if items.is_empty() {
            return;
        }

        // Only use parallel processing if worth the overhead
        if items.len() < 100 {
            for (text, data) in items {
                self.add_with_data(text, data);
            }
            return;
        }

        let normalize = self.normalize;
        let n = self.n;

        // Extract ngrams in parallel
        let processed: Vec<(String, String, AHashSet<String>, Option<u64>)> = items
            .into_par_iter()
            .map(|(text, data)| {
                let normalized = if normalize {
                    text.to_lowercase()
                } else {
                    text.clone()
                };
                let ngrams: AHashSet<String> = extract_ngrams(&normalized, n).into_iter().collect();
                (text, normalized, ngrams, data)
            })
            .collect();

        // Merge into index sequentially
        let start_id = self.entries.len();

        for (idx, (text, normalized, ngrams, data)) in processed.into_iter().enumerate() {
            let id = start_id + idx;

            for ngram in ngrams {
                let hash = Self::hash_string(&ngram);
                self.index
                    .entry(hash)
                    .or_default()
                    .push(id);
            }

            let hash = Self::hash_string(&normalized);
            self.exact_lookup.insert(hash, id);

            self.entries.push(IndexEntry { id, text, data });
        }
    }
    
    /// Get candidates that share at least one n-gram with the query
    pub fn get_candidates(&self, query: &str) -> Vec<usize> {
        // Normalize query to match indexed n-grams when normalize is enabled
        let normalized_query = if self.normalize {
            query.to_lowercase()
        } else {
            query.to_string()
        };
        let query_ngrams = extract_ngrams(&normalized_query, self.n);
        let mut candidate_ids = AHashSet::new();

        // Use compressed index if available, otherwise use uncompressed
        if let Some(compressed) = &self.compressed_index {
            for ngram in &query_ngrams {
                let hash = Self::hash_string(ngram);
                if let Some(posting_list) = compressed.get(&hash) {
                    candidate_ids.extend(posting_list.decode());
                }
            }
        } else {
            for ngram in &query_ngrams {
                let hash = Self::hash_string(ngram);
                if let Some(ids) = self.index.get(&hash) {
                    candidate_ids.extend(ids.iter().copied());
                }
            }
        }

        candidate_ids.into_iter().collect()
    }
    
    /// Get candidates with a minimum n-gram overlap ratio
    pub fn get_candidates_with_min_ratio(&self, query: &str, min_ratio: f64) -> Vec<usize> {
        // Normalize query to match indexed n-grams when normalize is enabled
        let normalized_query = if self.normalize {
            query.to_lowercase()
        } else {
            query.to_string()
        };
        let query_ngrams: AHashSet<String> = extract_ngrams(&normalized_query, self.n).into_iter().collect();
        let query_ngram_count = query_ngrams.len();

        if query_ngram_count == 0 {
            return Vec::new();
        }

        // Count how many query n-grams each candidate has
        let mut candidate_counts: AHashMap<usize, usize> = AHashMap::new();

        // Use compressed index if available, otherwise use uncompressed
        if let Some(compressed) = &self.compressed_index {
            for ngram in &query_ngrams {
                let hash = Self::hash_string(ngram);
                if let Some(posting_list) = compressed.get(&hash) {
                    for id in posting_list.decode() {
                        *candidate_counts.entry(id).or_insert(0) += 1;
                    }
                }
            }
        } else {
            for ngram in &query_ngrams {
                let hash = Self::hash_string(ngram);
                if let Some(ids) = self.index.get(&hash) {
                    for &id in ids {
                        *candidate_counts.entry(id).or_insert(0) += 1;
                    }
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
        let candidates = if self.min_ngram_ratio > 0.0 {
            self.get_candidates_with_min_ratio(query, self.min_ngram_ratio)
        } else {
            self.get_candidates(query)
        };
        
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
        let candidates = if self.min_ngram_ratio > 0.0 {
            self.get_candidates_with_min_ratio(query, self.min_ngram_ratio)
        } else {
            self.get_candidates(query)
        };
        
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
        // Normalize query to match indexed text when normalize is enabled
        let normalized_query = if self.normalize {
            query.to_lowercase()
        } else {
            query.to_string()
        };
        let hash = Self::hash_string(&normalized_query);
        // Check if hash exists and verify the actual text matches (handle hash collisions)
        // Compare normalized versions when normalize is enabled
        self.exact_lookup.get(&hash)
            .map(|&id| {
                self.entries.get(id).map(|e| {
                    if self.normalize {
                        e.text.to_lowercase() == normalized_query
                    } else {
                        e.text == query
                    }
                }).unwrap_or(false)
            })
            .unwrap_or(false)
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.index.clear();
        self.compressed_index = None;
        self.entries.clear();
        self.exact_lookup.clear();
    }

    // =========================================================================
    // Posting List Compression
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
    /// # Example
    /// ```ignore
    /// let mut index = NgramIndex::new(3);
    /// index.add_all(large_dataset);
    /// index.compress();  // Reduce memory usage
    /// // Searches still work, using compressed index
    /// ```
    pub fn compress(&mut self) {
        if self.index.is_empty() {
            return;
        }

        let mut compressed = AHashMap::with_capacity(self.index.len());

        for (hash, ids) in &self.index {
            // Sort IDs for delta encoding (should already be sorted due to add order, but ensure it)
            let mut sorted_ids = ids.clone();
            sorted_ids.sort_unstable();
            compressed.insert(*hash, CompressedPostingList::from_sorted_ids(&sorted_ids));
        }

        self.compressed_index = Some(compressed);
        self.index.clear();
        self.index.shrink_to_fit();
    }

    /// Decompress the posting lists to allow adding new items.
    ///
    /// This restores the uncompressed index from the compressed format.
    /// Call this before adding new items to a compressed index.
    pub fn decompress(&mut self) {
        if let Some(compressed) = self.compressed_index.take() {
            for (hash, posting_list) in compressed {
                self.index.insert(hash, posting_list.decode());
            }
        }
    }

    /// Check if the index is currently compressed.
    #[inline]
    pub fn is_compressed(&self) -> bool {
        self.compressed_index.is_some()
    }

    /// Get compression statistics.
    ///
    /// Returns (compressed_bytes, uncompressed_bytes, compression_ratio).
    /// Returns None if the index is not compressed.
    pub fn compression_stats(&self) -> Option<(usize, usize, f64)> {
        self.compressed_index.as_ref().map(|compressed| {
            let mut compressed_bytes = 0usize;
            let mut uncompressed_bytes = 0usize;

            for posting_list in compressed.values() {
                compressed_bytes += posting_list.compressed_size();
                uncompressed_bytes += posting_list.uncompressed_size();
            }

            let ratio = if uncompressed_bytes > 0 {
                compressed_bytes as f64 / uncompressed_bytes as f64
            } else {
                1.0
            };

            (compressed_bytes, uncompressed_bytes, ratio)
        })
    }
    
    /// Get entry by ID
    pub fn get(&self, id: usize) -> Option<&IndexEntry> {
        self.entries.get(id)
    }

    // =========================================================================
    // Serialization
    // =========================================================================

    /// Serialize the index to bytes using bincode.
    ///
    /// # Example
    /// ```ignore
    /// let bytes = index.to_bytes()?;
    /// std::fs::write("index.bin", &bytes)?;
    /// ```
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Deserialize an index from bytes.
    ///
    /// # Example
    /// ```ignore
    /// let bytes = std::fs::read("index.bin")?;
    /// let index = NgramIndex::from_bytes(&bytes)?;
    /// ```
    pub fn from_bytes(data: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(data)
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

    /// Create with minimum n-gram ratio for candidate filtering
    pub fn with_min_ngram_ratio(ngram_size: usize, min_ngram_ratio: f64) -> Self {
        Self {
            ngram_index: NgramIndex::with_params(ngram_size, 0.0, min_ngram_ratio, false),
        }
    }

    /// Create with all parameters including normalization
    ///
    /// # Arguments
    /// * `ngram_size` - Size of n-grams for indexing
    /// * `min_ngram_ratio` - Minimum ratio of query n-grams that must match (0.0 to 1.0)
    /// * `normalize` - Whether to lowercase text for case-insensitive n-gram matching
    pub fn with_params(ngram_size: usize, min_ngram_ratio: f64, normalize: bool) -> Self {
        Self {
            ngram_index: NgramIndex::with_params(ngram_size, 0.0, min_ngram_ratio, normalize),
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

    #[test]
    fn test_compressed_posting_list() {
        // Test basic compression/decompression
        let ids = vec![10, 15, 20, 100, 105, 1000];
        let compressed = CompressedPostingList::from_sorted_ids(&ids);

        assert_eq!(compressed.len(), 6);
        assert_eq!(compressed.decode(), ids);

        // Verify compression actually saves space
        assert!(compressed.compressed_size() < compressed.uncompressed_size());
        assert!(compressed.compression_ratio() < 1.0);
    }

    #[test]
    fn test_compressed_posting_list_empty() {
        let compressed = CompressedPostingList::from_sorted_ids(&[]);
        assert!(compressed.is_empty());
        assert_eq!(compressed.decode(), Vec::<usize>::new());
    }

    #[test]
    fn test_ngram_index_compress() {
        let mut index = NgramIndex::new(2);
        index.add_all(["hello", "hallo", "hullo", "world", "help"]);

        // Get candidates before compression
        let candidates_before = index.get_candidates("hello");
        assert!(!index.is_compressed());

        // Compress
        index.compress();
        assert!(index.is_compressed());

        // Get candidates after compression - should be same
        let candidates_after = index.get_candidates("hello");

        // Sort both for comparison (order may differ)
        let mut before_sorted = candidates_before.clone();
        let mut after_sorted = candidates_after.clone();
        before_sorted.sort();
        after_sorted.sort();

        assert_eq!(before_sorted, after_sorted);

        // Check compression stats
        let stats = index.compression_stats();
        assert!(stats.is_some());
        let (compressed, uncompressed, ratio) = stats.unwrap();
        assert!(compressed > 0);
        assert!(uncompressed > 0);
        assert!(ratio <= 1.0);
    }

    #[test]
    fn test_ngram_index_decompress() {
        let mut index = NgramIndex::new(2);
        index.add_all(["hello", "hallo", "world"]);

        let candidates_original = index.get_candidates("hello");

        // Compress then decompress
        index.compress();
        assert!(index.is_compressed());

        index.decompress();
        assert!(!index.is_compressed());

        // Should still work after decompress
        let candidates_after = index.get_candidates("hello");

        let mut orig_sorted = candidates_original.clone();
        let mut after_sorted = candidates_after.clone();
        orig_sorted.sort();
        after_sorted.sort();

        assert_eq!(orig_sorted, after_sorted);
    }

    #[test]
    fn test_compressed_search() {
        let mut index = NgramIndex::new(2);
        index.add_all(["hello", "hallo", "hullo", "world", "help"]);

        let jw = JaroWinkler::new();
        let results_before = index.search("hello", &jw, 0.8, Some(3));

        index.compress();
        let results_after = index.search("hello", &jw, 0.8, Some(3));

        // Same number of results
        assert_eq!(results_before.len(), results_after.len());

        // Same top result
        assert_eq!(results_before[0].text, results_after[0].text);
    }
}
