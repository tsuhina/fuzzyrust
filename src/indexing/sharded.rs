//! Sharded indices for very large datasets (>10M items).
//!
//! Sharding distributes data across multiple index instances, allowing:
//! - Better memory locality for large datasets
//! - Parallel search across shards
//! - Horizontal scaling
//!
//! # Example
//!
//! ```ignore
//! use fuzzyrust::indexing::sharded::ShardedBkTree;
//!
//! let mut tree = ShardedBkTree::new(8); // 8 shards
//! tree.add_all(items); // Items distributed across shards
//! let results = tree.search("query", 2); // Parallel search
//! ```

use ahash::AHasher;
use rayon::prelude::*;
use std::hash::{Hash, Hasher};

use super::bktree::{BkTree, SearchResult as BkSearchResult};
use super::ngram_index::{NgramIndex, SearchMatch};
use crate::algorithms::Similarity;

/// Compute a hash for sharding distribution.
///
/// Uses AHash for fast, high-quality hashing.
fn compute_shard_hash(text: &str) -> u64 {
    let mut hasher = AHasher::default();
    text.hash(&mut hasher);
    hasher.finish()
}

// ============================================================================
// ShardedBkTree
// ============================================================================

/// A sharded BK-tree for very large datasets.
///
/// Distributes items across multiple BK-tree shards based on text hash.
/// Searches are performed in parallel across all shards.
///
/// # Performance Characteristics
///
/// - **Insert**: O(log n / num_shards) average per shard
/// - **Search**: O(log n / num_shards) per shard, parallelized
/// - **Memory**: Slightly higher overhead due to multiple tree structures
///
/// # When to Use
///
/// - Datasets with >1M items
/// - When single-tree performance degrades
/// - When you have multiple CPU cores available
pub struct ShardedBkTree {
    shards: Vec<BkTree>,
    num_shards: usize,
    use_damerau: bool,
}

impl ShardedBkTree {
    /// Create a new sharded BK-tree with the specified number of shards.
    ///
    /// # Arguments
    /// * `num_shards` - Number of shards (typically 4-16, depending on CPU cores)
    pub fn new(num_shards: usize) -> Self {
        let num_shards = num_shards.max(1);
        Self {
            shards: (0..num_shards).map(|_| BkTree::new()).collect(),
            num_shards,
            use_damerau: false,
        }
    }

    /// Create a new sharded BK-tree using Damerau-Levenshtein distance.
    pub fn new_damerau(num_shards: usize) -> Self {
        let num_shards = num_shards.max(1);
        Self {
            shards: (0..num_shards).map(|_| BkTree::with_damerau()).collect(),
            num_shards,
            use_damerau: true,
        }
    }

    /// Get the shard index for a given text.
    fn shard_for(&self, text: &str) -> usize {
        (compute_shard_hash(text) as usize) % self.num_shards
    }

    /// Add a string to the appropriate shard.
    pub fn add(&mut self, text: impl Into<String>) -> bool {
        let text = text.into();
        let shard_idx = self.shard_for(&text);
        self.shards[shard_idx].add(text)
    }

    /// Add a string with associated data.
    pub fn add_with_data(&mut self, text: impl Into<String>, data: Option<u64>) -> bool {
        let text = text.into();
        let shard_idx = self.shard_for(&text);
        self.shards[shard_idx].add_with_data(text, data)
    }

    /// Add multiple strings, distributing across shards.
    pub fn add_all<I, S>(&mut self, iter: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for text in iter {
            self.add(text);
        }
    }

    /// Add multiple strings in parallel.
    ///
    /// Groups items by shard, then adds to each shard.
    pub fn add_all_parallel<I, S>(&mut self, iter: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String> + Send,
    {
        let items: Vec<String> = iter.into_iter().map(|s| s.into()).collect();

        // Group by shard
        let mut shard_items: Vec<Vec<String>> = vec![Vec::new(); self.num_shards];
        for text in items {
            let shard_idx = self.shard_for(&text);
            shard_items[shard_idx].push(text);
        }

        // Add to each shard (can't easily parallelize mutable access)
        for (shard_idx, items) in shard_items.into_iter().enumerate() {
            self.shards[shard_idx].add_all(items);
        }
    }

    /// Search all shards in parallel for strings within the given distance.
    pub fn search(&self, query: &str, max_distance: usize) -> Vec<BkSearchResult> {
        self.shards
            .par_iter()
            .flat_map(|shard| shard.search(query, max_distance))
            .collect()
    }

    /// Find the k nearest neighbors across all shards.
    pub fn find_nearest(&self, query: &str, k: usize) -> Vec<BkSearchResult> {
        // Get candidates from all shards
        let mut all_results: Vec<BkSearchResult> = self
            .shards
            .par_iter()
            .flat_map(|shard| shard.find_nearest(query, k))
            .collect();

        // Sort by distance and take top k
        all_results.sort_by_key(|r| r.distance);
        all_results.truncate(k);
        all_results
    }

    /// Check if any shard contains an exact match.
    pub fn contains(&self, query: &str) -> bool {
        // Check the specific shard where it would be stored
        let shard_idx = self.shard_for(query);
        self.shards[shard_idx].contains(query)
    }

    /// Remove an item from the appropriate shard by text.
    pub fn remove_text(&mut self, text: &str) -> bool {
        let shard_idx = self.shard_for(text);
        self.shards[shard_idx].remove_text(text)
    }

    /// Compact all shards to remove tombstones.
    pub fn compact(&mut self) {
        for shard in &mut self.shards {
            shard.compact();
        }
    }

    /// Get total number of items across all shards.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Get total number of active (non-deleted) items.
    pub fn active_count(&self) -> usize {
        self.shards.iter().map(|s| s.active_count()).sum()
    }

    /// Get total number of deleted items.
    pub fn deleted_count(&self) -> usize {
        self.shards.iter().map(|s| s.deleted_count()).sum()
    }

    /// Check if all shards are empty.
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.is_empty())
    }

    /// Get the number of shards.
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Get the distribution of items across shards.
    ///
    /// Returns a vector of item counts per shard.
    pub fn shard_distribution(&self) -> Vec<usize> {
        self.shards.iter().map(|s| s.len()).collect()
    }

    /// Whether this uses Damerau-Levenshtein distance.
    pub fn uses_damerau(&self) -> bool {
        self.use_damerau
    }
}

impl Default for ShardedBkTree {
    fn default() -> Self {
        // Default to number of CPU cores, capped at 16
        let num_shards = std::thread::available_parallelism()
            .map(|p| p.get().min(16))
            .unwrap_or(8);
        Self::new(num_shards)
    }
}

// ============================================================================
// ShardedNgramIndex
// ============================================================================

/// A sharded N-gram index for very large datasets.
///
/// Distributes items across multiple N-gram index shards based on text hash.
/// Searches are performed in parallel across all shards.
pub struct ShardedNgramIndex {
    shards: Vec<NgramIndex>,
    num_shards: usize,
    n: usize,
}

impl ShardedNgramIndex {
    /// Create a new sharded N-gram index.
    ///
    /// # Arguments
    /// * `num_shards` - Number of shards
    /// * `n` - N-gram size (typically 2-4)
    pub fn new(num_shards: usize, n: usize) -> Self {
        let num_shards = num_shards.max(1);
        Self {
            shards: (0..num_shards).map(|_| NgramIndex::new(n)).collect(),
            num_shards,
            n,
        }
    }

    /// Create with parameters.
    pub fn with_params(
        num_shards: usize,
        n: usize,
        min_similarity: f64,
        min_ngram_ratio: f64,
        normalize: bool,
    ) -> Self {
        let num_shards = num_shards.max(1);
        Self {
            shards: (0..num_shards)
                .map(|_| NgramIndex::with_params(n, min_similarity, min_ngram_ratio, normalize))
                .collect(),
            num_shards,
            n,
        }
    }

    /// Get the shard index for a given text.
    fn shard_for(&self, text: &str) -> usize {
        (compute_shard_hash(text) as usize) % self.num_shards
    }

    /// Add a string to the appropriate shard.
    pub fn add(&mut self, text: impl Into<String>) -> usize {
        let text = text.into();
        let shard_idx = self.shard_for(&text);
        // Encode shard in the high bits of the ID
        let local_id = self.shards[shard_idx].add(text);
        (shard_idx << 48) | local_id
    }

    /// Add a string with associated data.
    pub fn add_with_data(&mut self, text: impl Into<String>, data: Option<u64>) -> usize {
        let text = text.into();
        let shard_idx = self.shard_for(&text);
        let local_id = self.shards[shard_idx].add_with_data(text, data);
        (shard_idx << 48) | local_id
    }

    /// Add multiple strings.
    pub fn add_all<I, S>(&mut self, iter: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for text in iter {
            self.add(text);
        }
    }

    /// Add multiple strings in parallel.
    pub fn add_all_parallel<I, S>(&mut self, iter: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String> + Send,
    {
        let items: Vec<String> = iter.into_iter().map(|s| s.into()).collect();

        // Group by shard
        let mut shard_items: Vec<Vec<String>> = vec![Vec::new(); self.num_shards];
        for text in items {
            let shard_idx = self.shard_for(&text);
            shard_items[shard_idx].push(text);
        }

        // Add to each shard using parallel add
        for (shard_idx, items) in shard_items.into_iter().enumerate() {
            self.shards[shard_idx].add_all_parallel(items);
        }
    }

    /// Search all shards in parallel.
    pub fn search<S: Similarity + Send + Sync + ?Sized>(
        &self,
        query: &str,
        similarity: &S,
        min_similarity: f64,
        limit: Option<usize>,
    ) -> Vec<SearchMatch> {
        // Collect results from each shard in parallel
        let shard_results: Vec<Vec<SearchMatch>> = self
            .shards
            .par_iter()
            .enumerate()
            .map(|(shard_idx, shard)| {
                shard
                    .search_parallel(query, similarity, min_similarity, limit)
                    .into_iter()
                    .map(|mut m| {
                        // Encode shard in the ID
                        m.id |= shard_idx << 48;
                        m
                    })
                    .collect()
            })
            .collect();

        // Flatten and sort by similarity
        let mut all_results: Vec<SearchMatch> = shard_results.into_iter().flatten().collect();
        all_results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        if let Some(limit) = limit {
            all_results.truncate(limit);
        }
        all_results
    }

    /// Check if any shard contains an exact match.
    pub fn contains(&self, query: &str) -> bool {
        let shard_idx = self.shard_for(query);
        self.shards[shard_idx].contains(query)
    }

    /// Get text by ID (decodes shard from high bits).
    pub fn get_text(&self, id: usize) -> Option<String> {
        let shard_idx = id >> 48;
        let local_id = id & ((1 << 48) - 1);
        if shard_idx < self.num_shards {
            self.shards[shard_idx].get_text(local_id)
        } else {
            None
        }
    }

    /// Clear all shards.
    pub fn clear(&mut self) {
        for shard in &mut self.shards {
            shard.clear();
        }
    }

    /// Get total number of items.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Check if all shards are empty.
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.is_empty())
    }

    /// Get the number of shards.
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Get the N-gram size.
    pub fn ngram_size(&self) -> usize {
        self.n
    }

    /// Get the distribution of items across shards.
    pub fn shard_distribution(&self) -> Vec<usize> {
        self.shards.iter().map(|s| s.len()).collect()
    }
}

impl Default for ShardedNgramIndex {
    fn default() -> Self {
        let num_shards = std::thread::available_parallelism()
            .map(|p| p.get().min(16))
            .unwrap_or(8);
        Self::new(num_shards, 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::JaroWinkler;

    #[test]
    fn test_sharded_bktree_basic() {
        let mut tree = ShardedBkTree::new(4);
        tree.add_all(["hello", "hallo", "hullo", "world", "word"]);

        assert_eq!(tree.len(), 5);

        let results = tree.search("hello", 2);
        assert!(results.iter().any(|r| r.text == "hello"));
        assert!(results.iter().any(|r| r.text == "hallo"));
    }

    #[test]
    fn test_sharded_bktree_contains() {
        let mut tree = ShardedBkTree::new(4);
        tree.add("hello");
        tree.add("world");

        assert!(tree.contains("hello"));
        assert!(tree.contains("world"));
        assert!(!tree.contains("foo"));
    }

    #[test]
    fn test_sharded_bktree_distribution() {
        let mut tree = ShardedBkTree::new(4);
        for i in 0..1000 {
            tree.add(format!("item_{}", i));
        }

        let dist = tree.shard_distribution();
        assert_eq!(dist.len(), 4);
        assert_eq!(dist.iter().sum::<usize>(), 1000);

        // Check somewhat even distribution (each shard should have 100-400 items)
        for count in dist {
            assert!(count > 50, "Shard has too few items: {}", count);
            assert!(count < 500, "Shard has too many items: {}", count);
        }
    }

    #[test]
    fn test_sharded_bktree_find_nearest() {
        let mut tree = ShardedBkTree::new(4);
        tree.add_all(["hello", "hallo", "hullo", "world", "word", "help", "held"]);

        let nearest = tree.find_nearest("helo", 3);
        assert_eq!(nearest.len(), 3);
        assert!(nearest[0].distance <= nearest[1].distance);
        assert!(nearest[1].distance <= nearest[2].distance);
    }

    #[test]
    fn test_sharded_ngram_index_basic() {
        let mut index = ShardedNgramIndex::new(4, 3);
        index.add_all(["hello", "hallo", "hullo", "world", "word"]);

        assert_eq!(index.len(), 5);

        let jw = JaroWinkler::new();
        let results = index.search("hello", &jw, 0.7, Some(10));
        assert!(!results.is_empty());
    }

    #[test]
    fn test_sharded_ngram_index_get_text() {
        let mut index = ShardedNgramIndex::new(4, 3);
        let id1 = index.add("hello");
        let id2 = index.add("world");

        assert_eq!(index.get_text(id1), Some("hello".to_string()));
        assert_eq!(index.get_text(id2), Some("world".to_string()));
    }

    #[test]
    fn test_sharded_bktree_remove() {
        let mut tree = ShardedBkTree::new(4);
        tree.add("hello");
        tree.add("world");

        assert!(tree.contains("hello"));
        assert!(tree.remove_text("hello"));
        // Note: contains checks the shard, but the item is now deleted
        assert_eq!(tree.active_count(), 1);
    }
}
