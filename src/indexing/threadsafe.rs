//! Thread-safe wrappers for fuzzy search indices.
//!
//! These wrappers use `parking_lot::RwLock` to provide safe concurrent access
//! to indices that are otherwise not thread-safe for mutation.
//!
//! # Usage
//!
//! ```ignore
//! use fuzzyrust::indexing::threadsafe::ThreadSafeBkTree;
//!
//! let tree = ThreadSafeBkTree::new();
//!
//! // Can be safely shared across threads
//! let tree_clone = tree.clone();
//! std::thread::spawn(move || {
//!     tree_clone.add("hello");
//! });
//! ```
//!
//! # Performance Notes
//!
//! - Read operations (search, contains) acquire a shared read lock
//! - Write operations (add, remove) acquire an exclusive write lock
//! - Multiple readers can proceed concurrently
//! - Writers block all other access
//!
//! For read-heavy workloads, this provides good concurrency.
//! For write-heavy workloads, consider batching writes or using
//! separate indices per thread.

use parking_lot::RwLock;
use std::sync::Arc;

use super::bktree::{BkTree, SearchResult as BkSearchResult};
use super::ngram_index::{IndexEntry, NgramIndex, SearchMatch};
use crate::algorithms::Similarity;

/// Thread-safe wrapper for BkTree.
///
/// Provides concurrent read access with exclusive write access using RwLock.
/// All operations are safe to call from multiple threads simultaneously.
#[derive(Clone)]
pub struct ThreadSafeBkTree {
    inner: Arc<RwLock<BkTree>>,
}

impl ThreadSafeBkTree {
    /// Create a new empty thread-safe BK-tree using Levenshtein distance.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(BkTree::new())),
        }
    }

    /// Create a new thread-safe BK-tree using Damerau-Levenshtein distance.
    pub fn new_damerau() -> Self {
        Self {
            inner: Arc::new(RwLock::new(BkTree::with_damerau())),
        }
    }

    /// Wrap an existing BkTree in a thread-safe wrapper.
    pub fn from_tree(tree: BkTree) -> Self {
        Self {
            inner: Arc::new(RwLock::new(tree)),
        }
    }

    /// Add a string to the tree.
    ///
    /// Acquires an exclusive write lock.
    pub fn add(&self, text: impl Into<String>) -> bool {
        self.inner.write().add(text)
    }

    /// Add a string with associated data.
    ///
    /// Acquires an exclusive write lock.
    pub fn add_with_data(&self, text: impl Into<String>, data: u64) -> bool {
        self.inner.write().add_with_data(text, Some(data))
    }

    /// Add multiple strings.
    ///
    /// Acquires an exclusive write lock for the entire operation.
    pub fn add_all<I, S>(&self, iter: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner.write().add_all(iter);
    }

    /// Search for strings within a given edit distance.
    ///
    /// Acquires a shared read lock.
    pub fn search(&self, query: &str, max_distance: usize) -> Vec<BkSearchResult> {
        self.inner.read().search(query, max_distance)
    }

    /// Find the k nearest neighbors.
    ///
    /// Acquires a shared read lock.
    pub fn find_nearest(&self, query: &str, k: usize) -> Vec<BkSearchResult> {
        self.inner.read().find_nearest(query, k)
    }

    /// Check if the tree contains an exact match.
    ///
    /// Acquires a shared read lock.
    pub fn contains(&self, query: &str) -> bool {
        self.inner.read().contains(query)
    }

    /// Remove an entry by ID (tombstone deletion).
    ///
    /// Acquires an exclusive write lock.
    pub fn remove(&self, id: usize) -> bool {
        self.inner.write().remove(id)
    }

    /// Remove an entry by text value.
    ///
    /// Acquires an exclusive write lock.
    pub fn remove_text(&self, text: &str) -> bool {
        self.inner.write().remove_text(text)
    }

    /// Rebuild the tree without tombstones.
    ///
    /// Acquires an exclusive write lock.
    pub fn compact(&self) {
        self.inner.write().compact();
    }

    /// Get the number of entries (including deleted).
    ///
    /// Acquires a shared read lock.
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// Get the number of active (non-deleted) entries.
    ///
    /// Acquires a shared read lock.
    pub fn active_count(&self) -> usize {
        self.inner.read().active_count()
    }

    /// Get the number of deleted entries.
    ///
    /// Acquires a shared read lock.
    pub fn deleted_count(&self) -> usize {
        self.inner.read().deleted_count()
    }

    /// Check if the tree is empty.
    ///
    /// Acquires a shared read lock.
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    /// Serialize the tree to bytes.
    ///
    /// Acquires a shared read lock.
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        self.inner.read().to_bytes()
    }

    /// Create from serialized bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, bincode::Error> {
        BkTree::from_bytes(data).map(Self::from_tree)
    }

    /// Create from serialized bytes with Damerau-Levenshtein distance.
    pub fn from_bytes_damerau(data: &[u8]) -> Result<Self, bincode::Error> {
        BkTree::from_bytes_damerau(data).map(Self::from_tree)
    }

    /// Get direct access to the underlying tree (for advanced use).
    ///
    /// Returns a read guard that can be used for multiple operations.
    pub fn read(&self) -> parking_lot::RwLockReadGuard<'_, BkTree> {
        self.inner.read()
    }

    /// Get exclusive access to the underlying tree (for advanced use).
    ///
    /// Returns a write guard that can be used for multiple operations.
    pub fn write(&self) -> parking_lot::RwLockWriteGuard<'_, BkTree> {
        self.inner.write()
    }
}

impl Default for ThreadSafeBkTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe wrapper for NgramIndex.
///
/// Provides concurrent read access with exclusive write access using RwLock.
#[derive(Clone)]
pub struct ThreadSafeNgramIndex {
    inner: Arc<RwLock<NgramIndex>>,
}

impl ThreadSafeNgramIndex {
    /// Create a new thread-safe n-gram index.
    pub fn new(n: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(NgramIndex::new(n))),
        }
    }

    /// Create with minimum similarity threshold.
    pub fn with_min_similarity(n: usize, min_similarity: f64) -> Self {
        Self {
            inner: Arc::new(RwLock::new(NgramIndex::with_min_similarity(
                n,
                min_similarity,
            ))),
        }
    }

    /// Create with all parameters.
    pub fn with_params(
        n: usize,
        min_similarity: f64,
        min_ngram_ratio: f64,
        normalize: bool,
    ) -> Self {
        Self {
            inner: Arc::new(RwLock::new(NgramIndex::with_params(
                n,
                min_similarity,
                min_ngram_ratio,
                normalize,
            ))),
        }
    }

    /// Wrap an existing NgramIndex.
    pub fn from_index(index: NgramIndex) -> Self {
        Self {
            inner: Arc::new(RwLock::new(index)),
        }
    }

    /// Add a string to the index.
    ///
    /// Acquires an exclusive write lock.
    pub fn add(&self, text: impl Into<String>) -> usize {
        self.inner.write().add(text)
    }

    /// Add a string with associated data.
    ///
    /// Acquires an exclusive write lock.
    pub fn add_with_data(&self, text: impl Into<String>, data: Option<u64>) -> usize {
        self.inner.write().add_with_data(text, data)
    }

    /// Add multiple strings.
    ///
    /// Acquires an exclusive write lock.
    pub fn add_all<I, S>(&self, iter: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner.write().add_all(iter);
    }

    /// Add multiple strings in parallel.
    ///
    /// Acquires an exclusive write lock.
    pub fn add_all_parallel<I, S>(&self, iter: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String> + Send,
    {
        self.inner.write().add_all_parallel(iter);
    }

    /// Search with a similarity function.
    ///
    /// Acquires a shared read lock.
    pub fn search<S: Similarity + ?Sized>(
        &self,
        query: &str,
        similarity: &S,
        min_similarity: f64,
        limit: Option<usize>,
    ) -> Vec<SearchMatch> {
        self.inner
            .read()
            .search(query, similarity, min_similarity, limit)
    }

    /// Parallel search for large datasets.
    ///
    /// Acquires a shared read lock.
    pub fn search_parallel<S: Similarity + Send + Sync + ?Sized>(
        &self,
        query: &str,
        similarity: &S,
        min_similarity: f64,
        limit: Option<usize>,
    ) -> Vec<SearchMatch> {
        self.inner
            .read()
            .search_parallel(query, similarity, min_similarity, limit)
    }

    /// Check if the index contains an exact match.
    ///
    /// Acquires a shared read lock.
    pub fn contains(&self, query: &str) -> bool {
        self.inner.read().contains(query)
    }

    /// Get text for a specific ID.
    ///
    /// Acquires a shared read lock.
    pub fn get_text(&self, id: usize) -> Option<String> {
        self.inner.read().get_text(id)
    }

    /// Get entry by ID.
    ///
    /// Acquires a shared read lock. Returns cloned entry to avoid lifetime issues.
    pub fn get(&self, id: usize) -> Option<IndexEntry> {
        self.inner.read().get(id).cloned()
    }

    /// Get the number of entries.
    ///
    /// Acquires a shared read lock.
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// Check if the index is empty.
    ///
    /// Acquires a shared read lock.
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    /// Clear the index.
    ///
    /// Acquires an exclusive write lock.
    pub fn clear(&self) {
        self.inner.write().clear();
    }

    /// Serialize the index to bytes.
    ///
    /// Acquires a shared read lock.
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        self.inner.read().to_bytes()
    }

    /// Create from serialized bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, bincode::Error> {
        NgramIndex::from_bytes(data).map(Self::from_index)
    }

    /// Get direct access to the underlying index.
    pub fn read(&self) -> parking_lot::RwLockReadGuard<'_, NgramIndex> {
        self.inner.read()
    }

    /// Get exclusive access to the underlying index.
    pub fn write(&self) -> parking_lot::RwLockWriteGuard<'_, NgramIndex> {
        self.inner.write()
    }
}

impl Default for ThreadSafeNgramIndex {
    fn default() -> Self {
        Self::new(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::JaroWinkler;
    use std::thread;

    #[test]
    fn test_threadsafe_bktree_concurrent_reads() {
        let tree = ThreadSafeBkTree::new();
        tree.add_all(["hello", "hallo", "hullo", "world"]);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let tree = tree.clone();
                thread::spawn(move || {
                    let results = tree.search("hello", 2);
                    assert!(!results.is_empty());
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_threadsafe_bktree_concurrent_writes() {
        let tree = ThreadSafeBkTree::new();

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let tree = tree.clone();
                thread::spawn(move || {
                    for j in 0..100 {
                        tree.add(format!("item_{}_{}", i, j));
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(tree.len(), 400);
    }

    #[test]
    fn test_threadsafe_ngram_index_concurrent() {
        let index = ThreadSafeNgramIndex::new(3);
        index.add_all(["hello", "hallo", "hullo", "world"]);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let index = index.clone();
                thread::spawn(move || {
                    let jw = JaroWinkler::new();
                    let results = index.search("hello", &jw, 0.7, Some(10));
                    assert!(!results.is_empty());
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_threadsafe_mixed_read_write() {
        let tree = ThreadSafeBkTree::new();
        tree.add_all(["initial"]);

        let tree1 = tree.clone();
        let writer = thread::spawn(move || {
            for i in 0..50 {
                tree1.add(format!("write_{}", i));
                thread::yield_now();
            }
        });

        let tree2 = tree.clone();
        let reader = thread::spawn(move || {
            for _ in 0..100 {
                let _ = tree2.search("initial", 2);
                thread::yield_now();
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();

        assert!(!tree.is_empty()); // At least "initial"
    }
}
