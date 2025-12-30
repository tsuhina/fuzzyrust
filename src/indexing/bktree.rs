//! BK-tree (Burkhard-Keller tree) implementation
//!
//! A metric tree structure that enables fast fuzzy searching.
//! Works with any edit distance metric that satisfies triangle inequality.

use ahash::AHashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Maximum search distance for nearest neighbor queries.
///
/// Limits the search radius expansion in `find_nearest` to prevent
/// excessive computation when no close matches exist.
const MAX_NEAREST_SEARCH_DISTANCE: usize = 20;

/// Maximum tree depth to prevent infinite loops from buggy distance functions.
const MAX_TREE_DEPTH: usize = 1000;

/// A distance function type
pub type DistanceFn = Arc<dyn Fn(&str, &str) -> usize + Send + Sync>;

/// Result from a BK-tree search
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Unique ID assigned when the item was added to the tree
    pub id: usize,
    pub text: String,
    pub distance: usize,
    pub data: Option<u64>,
}

impl SearchResult {
    pub fn new(id: usize, text: String, distance: usize) -> Self {
        Self {
            id,
            text,
            distance,
            data: None,
        }
    }

    pub fn with_data(id: usize, text: String, distance: usize, data: u64) -> Self {
        Self {
            id,
            text,
            distance,
            data: Some(data),
        }
    }
}

/// A node in the BK-tree
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BkNode {
    /// Unique ID for this entry
    id: usize,
    /// The string stored at this node
    text: String,
    /// Optional associated data (e.g., user-provided ID)
    data: Option<u64>,
    /// Whether this node has been soft-deleted
    #[serde(default)]
    deleted: bool,
    /// Children indexed by distance
    children: AHashMap<usize, Box<BkNode>>,
}

impl BkNode {
    fn new(id: usize, text: String, data: Option<u64>) -> Self {
        Self {
            id,
            text,
            data,
            deleted: false,
            children: AHashMap::new(),
        }
    }
}

/// BK-tree for efficient fuzzy string search
///
/// # Capacity Limits
///
/// The tree uses `usize` for item IDs. On 64-bit systems, this allows up to
/// 18,446,744,073,709,551,615 items (18 quintillion). In practice, memory will
/// be exhausted long before this limit is reached. ID overflow is not checked
/// at runtime for performance reasons.
///
/// # Custom Distance Functions
///
/// When providing a custom distance function via `with_distance()`, ensure it:
/// - Returns actual edit distances (0 for identical strings)
/// - Does NOT return `usize::MAX` as a sentinel value
/// - Satisfies the triangle inequality for correct tree behavior
///
/// # Serialization
///
/// The tree can be serialized to bytes using `to_bytes()` and restored using
/// `from_bytes()`. Note that the distance function is not serialized; when
/// deserializing, a new distance function must be provided (defaults to
/// Levenshtein distance).
#[derive(Clone)]
pub struct BkTree {
    root: Option<Box<BkNode>>,
    distance_fn: DistanceFn,
    size: usize,
    /// Number of soft-deleted nodes (for deciding when to compact)
    deleted_count: usize,
    /// Next ID to assign to a new entry.
    /// Note: Overflow is not checked (would require 2^64 items on 64-bit systems).
    next_id: usize,
}

/// Serializable state for BkTree (excludes distance function)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BkTreeState {
    root: Option<Box<BkNode>>,
    size: usize,
    deleted_count: usize,
    next_id: usize,
}

impl BkTree {
    /// Create a new BK-tree with Levenshtein distance
    pub fn new() -> Self {
        Self::with_distance(Arc::new(|a: &str, b: &str| {
            crate::algorithms::levenshtein::levenshtein(a, b)
        }))
    }

    /// Create a new BK-tree with Damerau-Levenshtein distance
    pub fn with_damerau() -> Self {
        Self::with_distance(Arc::new(|a: &str, b: &str| {
            crate::algorithms::damerau::damerau_levenshtein(a, b)
        }))
    }

    /// Create a new BK-tree with a custom distance function
    pub fn with_distance(distance_fn: DistanceFn) -> Self {
        Self {
            root: None,
            distance_fn,
            size: 0,
            deleted_count: 0,
            next_id: 0,
        }
    }

    /// Create a BK-tree with any EditDistance trait implementation
    ///
    /// This provides a more flexible way to configure the tree using
    /// the trait system instead of raw function pointers.
    ///
    /// # Example
    /// ```ignore
    /// use fuzzyrust::algorithms::levenshtein::Levenshtein;
    /// use fuzzyrust::indexing::bktree::BkTree;
    ///
    /// let tree = BkTree::with_metric(Levenshtein::new());
    /// ```
    pub fn with_metric<T: crate::algorithms::EditDistance + 'static>(metric: T) -> Self {
        let metric = Arc::new(metric);
        Self::with_distance(Arc::new(move |a, b| metric.distance(a, b)))
    }

    /// Create a BK-tree with an Arc-wrapped EditDistance trait object
    ///
    /// Useful when you want to share the same metric instance across
    /// multiple trees or manage the lifetime explicitly.
    ///
    /// # Example
    /// ```ignore
    /// use std::sync::Arc;
    /// use fuzzyrust::algorithms::damerau::DamerauLevenshtein;
    /// use fuzzyrust::indexing::bktree::BkTree;
    ///
    /// let metric = Arc::new(DamerauLevenshtein::new());
    /// let tree = BkTree::with_metric_arc(metric);
    /// ```
    pub fn with_metric_arc(metric: Arc<dyn crate::algorithms::EditDistance>) -> Self {
        Self::with_distance(Arc::new(move |a, b| metric.distance(a, b)))
    }

    /// Add a string to the tree.
    ///
    /// Returns `true` if the string was added successfully, `false` if it was a duplicate
    /// or if the tree depth limit was exceeded (indicating a buggy distance function).
    pub fn add(&mut self, text: impl Into<String>) -> bool {
        self.add_with_data(text, None)
    }

    /// Add a string with associated data.
    ///
    /// Returns `true` if the string was added successfully, `false` if:
    /// - The string is a duplicate (already exists in the tree)
    /// - The tree depth limit was exceeded (likely a buggy distance function)
    ///
    /// # Note on Tree Depth
    ///
    /// The tree has a maximum depth of 1000 to prevent infinite loops from
    /// malformed distance functions. If this limit is exceeded, the item is
    /// rejected and `false` is returned. This should never happen with proper
    /// distance functions that satisfy the metric space properties.
    pub fn add_with_data(&mut self, text: impl Into<String>, data: Option<u64>) -> bool {
        let text = text.into();
        let id = self.next_id;

        let Some(mut node) = self.root.as_mut() else {
            self.root = Some(Box::new(BkNode::new(id, text, data)));
            self.size = 1;
            // ID increment with overflow check in debug mode
            #[cfg(debug_assertions)]
            {
                self.next_id = self
                    .next_id
                    .checked_add(1)
                    .expect("BK-tree ID overflow - this should never happen in practice");
            }
            #[cfg(not(debug_assertions))]
            {
                self.next_id = self.next_id.wrapping_add(1);
            }
            return true;
        };

        let mut depth = 0;

        loop {
            depth += 1;
            // Safety check: prevent infinite loops from buggy distance functions.
            // Always warn (even in release mode) and reject the item.
            if depth > MAX_TREE_DEPTH {
                // Log warning to stderr in release mode (eprintln! always outputs)
                eprintln!(
                    "[fuzzyrust warning] BK-tree exceeded maximum depth {} while adding item. \
                     This indicates a possible infinite loop or buggy distance function that violates \
                     metric space properties. The item was rejected.",
                    MAX_TREE_DEPTH
                );
                // In debug mode, also panic for easier debugging
                debug_assert!(
                    false,
                    "BK-tree exceeded maximum depth {} - possible infinite loop or buggy distance function",
                    MAX_TREE_DEPTH
                );
                // Note: Item is rejected to prevent infinite loops from buggy distance functions.
                // In production, this should never happen with proper metric distance functions.
                return false;
            }

            let dist = (self.distance_fn)(&text, &node.text);

            if dist == 0 {
                // Duplicate - item already exists
                return false;
            }

            match node.children.entry(dist) {
                std::collections::hash_map::Entry::Occupied(entry) => {
                    node = entry.into_mut();
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(Box::new(BkNode::new(id, text, data)));
                    self.size += 1;
                    // ID increment with overflow check in debug mode
                    #[cfg(debug_assertions)]
                    {
                        self.next_id = self
                            .next_id
                            .checked_add(1)
                            .expect("BK-tree ID overflow - this should never happen in practice");
                    }
                    #[cfg(not(debug_assertions))]
                    {
                        self.next_id = self.next_id.wrapping_add(1);
                    }
                    return true;
                }
            }
        }
    }

    /// Add multiple strings at once
    pub fn add_all<I, S>(&mut self, iter: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for text in iter {
            self.add(text);
        }
    }

    /// Search for strings within a given distance
    pub fn search(&self, query: &str, max_distance: usize) -> Vec<SearchResult> {
        let mut results = Vec::new();

        if let Some(root) = &self.root {
            self.search_recursive(root, query, max_distance, &mut results);
        }

        // Sort by distance
        results.sort_by_key(|r| r.distance);
        results
    }

    fn search_recursive(
        &self,
        node: &BkNode,
        query: &str,
        max_distance: usize,
        results: &mut Vec<SearchResult>,
    ) {
        let dist = (self.distance_fn)(query, &node.text);

        // Only include non-deleted nodes in results
        if dist <= max_distance && !node.deleted {
            results.push(SearchResult {
                id: node.id,
                text: node.text.clone(),
                distance: dist,
                data: node.data,
            });
        }

        // Use triangle inequality to prune search
        let min_dist = dist.saturating_sub(max_distance);
        let max_dist = dist + max_distance;

        for (&child_dist, child_node) in &node.children {
            if child_dist >= min_dist && child_dist <= max_dist {
                self.search_recursive(child_node, query, max_distance, results);
            }
        }
    }

    /// Find the k nearest neighbors
    pub fn find_nearest(&self, query: &str, k: usize) -> Vec<SearchResult> {
        if self.root.is_none() || k == 0 {
            return Vec::new();
        }

        // Start with a small radius and expand if needed
        let mut max_dist = 1;
        let mut results;

        loop {
            results = self.search(query, max_dist);

            if results.len() >= k || max_dist >= MAX_NEAREST_SEARCH_DISTANCE {
                break;
            }

            // Use controlled expansion: double but cap at MAX to avoid missing items
            // between the last power of 2 and MAX_NEAREST_SEARCH_DISTANCE
            let next_dist = max_dist.saturating_mul(2);
            max_dist = if next_dist > MAX_NEAREST_SEARCH_DISTANCE {
                MAX_NEAREST_SEARCH_DISTANCE
            } else {
                next_dist
            };
        }

        results.truncate(k);
        results
    }

    /// Check if the tree contains an exact match
    pub fn contains(&self, query: &str) -> bool {
        if let Some(root) = &self.root {
            self.contains_recursive(root, query)
        } else {
            false
        }
    }

    fn contains_recursive(&self, node: &BkNode, query: &str) -> bool {
        let dist = (self.distance_fn)(query, &node.text);

        // Check if this node matches (and is not deleted)
        if dist == 0 && !node.deleted {
            return true;
        }

        if let Some(child) = node.children.get(&dist) {
            self.contains_recursive(child, query)
        } else {
            false
        }
    }

    /// Get the number of items in the tree
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if tree is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Clear the tree
    pub fn clear(&mut self) {
        self.root = None;
        self.size = 0;
        self.deleted_count = 0;
        self.next_id = 0;
    }

    // =========================================================================
    // Serialization
    // =========================================================================

    /// Serialize the tree to bytes using bincode.
    ///
    /// Note: The distance function is NOT serialized. When deserializing,
    /// you must provide a distance function (or use the default Levenshtein).
    ///
    /// # Example
    /// ```ignore
    /// let bytes = tree.to_bytes()?;
    /// std::fs::write("index.bin", &bytes)?;
    /// ```
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        let state = BkTreeState {
            root: self.root.clone(),
            size: self.size,
            deleted_count: self.deleted_count,
            next_id: self.next_id,
        };
        bincode::serialize(&state)
    }

    /// Deserialize a tree from bytes, using Levenshtein distance.
    ///
    /// # Example
    /// ```ignore
    /// let bytes = std::fs::read("index.bin")?;
    /// let tree = BkTree::from_bytes(&bytes)?;
    /// ```
    pub fn from_bytes(data: &[u8]) -> Result<Self, bincode::Error> {
        Self::from_bytes_with_distance(
            data,
            Arc::new(|a: &str, b: &str| crate::algorithms::levenshtein::levenshtein(a, b)),
        )
    }

    /// Deserialize a tree from bytes using Damerau-Levenshtein distance.
    ///
    /// Use this when the tree was originally created with `BkTree::with_damerau()`.
    pub fn from_bytes_damerau(data: &[u8]) -> Result<Self, bincode::Error> {
        Self::from_bytes_with_distance(
            data,
            Arc::new(|a: &str, b: &str| crate::algorithms::damerau::damerau_levenshtein(a, b)),
        )
    }

    /// Deserialize a tree from bytes with a custom distance function.
    pub fn from_bytes_with_distance(
        data: &[u8],
        distance_fn: DistanceFn,
    ) -> Result<Self, bincode::Error> {
        let state: BkTreeState = bincode::deserialize(data)?;
        Ok(Self {
            root: state.root,
            distance_fn,
            size: state.size,
            deleted_count: state.deleted_count,
            next_id: state.next_id,
        })
    }

    // =========================================================================
    // Deletion (Tombstone-based)
    // =========================================================================

    /// Remove an item by its ID (soft delete using tombstone).
    ///
    /// The item is marked as deleted but remains in the tree structure.
    /// Use `compact()` to rebuild the tree without deleted items.
    ///
    /// Returns `true` if the item was found and deleted, `false` otherwise.
    pub fn remove(&mut self, id: usize) -> bool {
        if let Some(ref mut root) = self.root {
            if Self::remove_by_id_recursive(root, id) {
                self.deleted_count += 1;
                return true;
            }
        }
        false
    }

    fn remove_by_id_recursive(node: &mut BkNode, id: usize) -> bool {
        if node.id == id && !node.deleted {
            node.deleted = true;
            return true;
        }

        for child in node.children.values_mut() {
            if Self::remove_by_id_recursive(child, id) {
                return true;
            }
        }
        false
    }

    /// Remove an item by its text (soft delete using tombstone).
    ///
    /// Returns `true` if the item was found and deleted, `false` otherwise.
    pub fn remove_text(&mut self, text: &str) -> bool {
        let distance_fn = self.distance_fn.clone();
        if let Some(ref mut root) = self.root {
            if Self::remove_by_text_recursive(root, text, &distance_fn) {
                self.deleted_count += 1;
                return true;
            }
        }
        false
    }

    fn remove_by_text_recursive(node: &mut BkNode, text: &str, distance_fn: &DistanceFn) -> bool {
        let dist = distance_fn(text, &node.text);

        if dist == 0 && !node.deleted {
            node.deleted = true;
            return true;
        }

        if let Some(child) = node.children.get_mut(&dist) {
            if Self::remove_by_text_recursive(child, text, distance_fn) {
                return true;
            }
        }
        false
    }

    /// Get the number of deleted (tombstoned) items.
    pub fn deleted_count(&self) -> usize {
        self.deleted_count
    }

    /// Get the number of active (non-deleted) items.
    pub fn active_count(&self) -> usize {
        self.size.saturating_sub(self.deleted_count)
    }

    /// Rebuild the tree without deleted items.
    ///
    /// This is useful when many items have been deleted and you want to
    /// reclaim memory and improve search performance.
    ///
    /// # Note
    /// Item IDs will be reassigned during compaction.
    pub fn compact(&mut self) {
        // Collect all non-deleted items
        let items: Vec<(String, Option<u64>)> = self
            .iter_active()
            .map(|(text, data)| (text.to_string(), data))
            .collect();

        // Clear and rebuild
        self.clear();
        for (text, data) in items {
            self.add_with_data(text, data);
        }
    }

    /// Iterate over all active (non-deleted) items.
    pub fn iter_active(&self) -> impl Iterator<Item = (&str, Option<u64>)> {
        BkTreeActiveIter::new(self.root.as_deref())
    }
}

impl Default for BkTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch search across multiple queries using parallel processing
pub fn batch_search(
    tree: &BkTree,
    queries: &[String],
    max_distance: usize,
) -> Vec<Vec<SearchResult>> {
    queries
        .par_iter()
        .map(|query| tree.search(query, max_distance))
        .collect()
}

/// Minimum tree size to use parallel search.
/// Below this threshold, sequential search is faster due to parallelism overhead.
const PARALLEL_SEARCH_THRESHOLD: usize = 10_000;

/// Maximum depth for collecting subtree roots for parallel search.
/// Collecting at depth 2-3 provides good parallelism without excessive overhead.
const PARALLEL_SEARCH_DEPTH: usize = 3;

/// Wrapper type to make raw pointers Send for parallel iteration.
///
/// # Safety
/// This is safe because:
/// 1. The BkTree is not modified during search (all operations are read-only)
/// 2. The pointers are only valid during the lifetime of the search call
/// 3. Each parallel task reads from a disjoint subtree
struct SendableNodePtr(*const BkNode);

// SAFETY: BkNode is read-only during search operations, and each parallel
// task operates on a disjoint subtree, so no data races can occur.
unsafe impl Send for SendableNodePtr {}
unsafe impl Sync for SendableNodePtr {}

impl BkTree {
    /// Parallel search for large trees (>10K nodes).
    ///
    /// For trees with more than 10,000 nodes, this method partitions the search
    /// across multiple CPU cores by collecting subtree roots at depth 2-3 and
    /// searching each subtree in parallel.
    ///
    /// For smaller trees, falls back to sequential search to avoid overhead.
    ///
    /// # Arguments
    /// * `query` - The string to search for
    /// * `max_distance` - Maximum edit distance to consider
    /// * `limit` - Optional maximum number of results to return
    ///
    /// # Returns
    /// Vector of SearchResult sorted by distance (ascending)
    ///
    /// # Performance
    /// Typically provides 2-4x speedup on large trees (>10K nodes) with many cores.
    /// The speedup depends on tree structure and query characteristics.
    pub fn search_parallel(
        &self,
        query: &str,
        max_distance: usize,
        limit: Option<usize>,
    ) -> Vec<SearchResult> {
        // For small trees, use sequential search
        if self.size < PARALLEL_SEARCH_THRESHOLD {
            let mut results = self.search(query, max_distance);
            if let Some(lim) = limit {
                results.truncate(lim);
            }
            return results;
        }

        let Some(root) = &self.root else {
            return Vec::new();
        };

        // Collect subtree roots at depth 2-3 for parallel processing
        let subtree_roots = self.collect_subtree_roots(root, PARALLEL_SEARCH_DEPTH);

        // Also check the root node and nodes along the path to subtree roots
        let mut root_path_results = Vec::new();
        self.search_root_path(
            root,
            query,
            max_distance,
            PARALLEL_SEARCH_DEPTH,
            &mut root_path_results,
        );

        // Search each subtree in parallel
        // Wrap raw pointers in a newtype that implements Send to enable parallel iteration
        let query_owned = query.to_string();
        let distance_fn = self.distance_fn.clone();
        let sendable_roots: Vec<SendableNodePtr> =
            subtree_roots.into_iter().map(SendableNodePtr).collect();

        let parallel_results: Vec<SearchResult> = sendable_roots
            .into_par_iter()
            .flat_map(|SendableNodePtr(node_ptr)| {
                // SAFETY: We only read from the tree and nodes are not modified during search
                let node = unsafe { &*node_ptr };
                let mut results = Vec::new();
                Self::search_subtree_recursive(
                    node,
                    &query_owned,
                    max_distance,
                    &distance_fn,
                    &mut results,
                );
                results
            })
            .collect();

        // Combine results
        let mut all_results = root_path_results;
        all_results.extend(parallel_results);

        // Sort by distance and apply limit
        all_results.sort_by_key(|r| r.distance);

        if let Some(lim) = limit {
            all_results.truncate(lim);
        }

        all_results
    }

    /// Collect pointers to subtree roots at a specific depth for parallel processing.
    ///
    /// Returns raw pointers to avoid lifetime issues with parallel iteration.
    /// These pointers are only valid while the tree is not modified.
    fn collect_subtree_roots(&self, root: &BkNode, max_depth: usize) -> Vec<*const BkNode> {
        let mut result = Vec::new();
        self.collect_subtree_roots_recursive(root, 0, max_depth, &mut result);
        result
    }

    fn collect_subtree_roots_recursive(
        &self,
        node: &BkNode,
        current_depth: usize,
        max_depth: usize,
        result: &mut Vec<*const BkNode>,
    ) {
        if current_depth >= max_depth {
            // At target depth, add this node as a subtree root
            result.push(node as *const BkNode);
            return;
        }

        // Recurse into children
        for child in node.children.values() {
            self.collect_subtree_roots_recursive(child, current_depth + 1, max_depth, result);
        }
    }

    /// Search nodes from root down to the parallel search depth.
    /// These nodes are not included in the parallel subtree search.
    fn search_root_path(
        &self,
        node: &BkNode,
        query: &str,
        max_distance: usize,
        remaining_depth: usize,
        results: &mut Vec<SearchResult>,
    ) {
        let dist = (self.distance_fn)(query, &node.text);

        // Add this node if it matches and is not deleted
        if dist <= max_distance && !node.deleted {
            results.push(SearchResult {
                id: node.id,
                text: node.text.clone(),
                distance: dist,
                data: node.data,
            });
        }

        // If we're at the depth limit, don't recurse further
        // (subtrees will be handled by parallel search)
        if remaining_depth == 0 {
            return;
        }

        // Use triangle inequality to prune search
        let min_dist = dist.saturating_sub(max_distance);
        let max_dist = dist + max_distance;

        for (&child_dist, child_node) in &node.children {
            if child_dist >= min_dist && child_dist <= max_dist {
                self.search_root_path(
                    child_node,
                    query,
                    max_distance,
                    remaining_depth - 1,
                    results,
                );
            }
        }
    }

    /// Search a subtree recursively (used by parallel search).
    fn search_subtree_recursive(
        node: &BkNode,
        query: &str,
        max_distance: usize,
        distance_fn: &DistanceFn,
        results: &mut Vec<SearchResult>,
    ) {
        let dist = distance_fn(query, &node.text);

        // Add this node if it matches and is not deleted
        if dist <= max_distance && !node.deleted {
            results.push(SearchResult {
                id: node.id,
                text: node.text.clone(),
                distance: dist,
                data: node.data,
            });
        }

        // Use triangle inequality to prune search
        let min_dist = dist.saturating_sub(max_distance);
        let max_dist = dist + max_distance;

        for (&child_dist, child_node) in &node.children {
            if child_dist >= min_dist && child_dist <= max_dist {
                Self::search_subtree_recursive(
                    child_node,
                    query,
                    max_distance,
                    distance_fn,
                    results,
                );
            }
        }
    }

    /// Find the k nearest neighbors using parallel search.
    ///
    /// Similar to `find_nearest`, but uses parallel processing for large trees.
    pub fn find_nearest_parallel(&self, query: &str, k: usize) -> Vec<SearchResult> {
        if self.root.is_none() || k == 0 {
            return Vec::new();
        }

        // Start with a small radius and expand if needed
        let mut max_dist = 1;
        let mut results;

        loop {
            results = self.search_parallel(query, max_dist, Some(k));

            if results.len() >= k || max_dist >= MAX_NEAREST_SEARCH_DISTANCE {
                break;
            }

            // Use controlled expansion
            let next_dist = max_dist.saturating_mul(2);
            max_dist = if next_dist > MAX_NEAREST_SEARCH_DISTANCE {
                MAX_NEAREST_SEARCH_DISTANCE
            } else {
                next_dist
            };
        }

        results.truncate(k);
        results
    }
}

/// Iterator over active (non-deleted) items in a BkTree
struct BkTreeActiveIter<'a> {
    stack: Vec<&'a BkNode>,
}

impl<'a> BkTreeActiveIter<'a> {
    fn new(root: Option<&'a BkNode>) -> Self {
        let mut stack = Vec::new();
        if let Some(node) = root {
            stack.push(node);
        }
        Self { stack }
    }
}

impl<'a> Iterator for BkTreeActiveIter<'a> {
    type Item = (&'a str, Option<u64>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            // Push all children onto the stack
            for child in node.children.values() {
                self.stack.push(child.as_ref());
            }
            // Return this node if it's not deleted
            if !node.deleted {
                return Some((&node.text, node.data));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bktree_basic() {
        let mut tree = BkTree::new();
        tree.add("hello");
        tree.add("hallo");
        tree.add("hullo");
        tree.add("world");

        assert_eq!(tree.len(), 4);
        assert!(tree.contains("hello"));
        assert!(!tree.contains("helloo"));
    }

    #[test]
    fn test_bktree_search() {
        let mut tree = BkTree::new();
        tree.add_all(["book", "books", "boo", "cook", "cake"]);

        let results = tree.search("book", 1);
        assert!(results.iter().any(|r| r.text == "book" && r.distance == 0));
        assert!(results.iter().any(|r| r.text == "books" && r.distance == 1));
        assert!(results.iter().any(|r| r.text == "boo" && r.distance == 1));
        assert!(results.iter().any(|r| r.text == "cook" && r.distance == 1));
    }

    #[test]
    fn test_find_nearest() {
        let mut tree = BkTree::new();
        tree.add_all(["apple", "application", "apply", "banana", "bandana"]);

        let results = tree.find_nearest("appli", 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_parallel_search_small_tree_uses_sequential() {
        // Build a small tree (below PARALLEL_SEARCH_THRESHOLD)
        let mut tree = BkTree::new();
        for i in 0..100 {
            tree.add(format!("item_{}", i));
        }

        // Both should return same results for small trees
        let seq_results = tree.search("item_50", 2);
        let par_results = tree.search_parallel("item_50", 2, None);

        // Results should be identical
        assert_eq!(seq_results.len(), par_results.len());
        let seq_texts: std::collections::HashSet<_> = seq_results.iter().map(|r| &r.text).collect();
        let par_texts: std::collections::HashSet<_> = par_results.iter().map(|r| &r.text).collect();
        assert_eq!(seq_texts, par_texts);
    }

    #[test]
    fn test_parallel_search_with_limit() {
        let mut tree = BkTree::new();
        for i in 0..200 {
            tree.add(format!("item_{:04}", i));
        }

        // Search with limit
        let results = tree.search_parallel("item_0050", 3, Some(5));
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_parallel_search_empty_tree() {
        let tree = BkTree::new();
        let results = tree.search_parallel("query", 2, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_nearest_parallel() {
        let mut tree = BkTree::new();
        tree.add_all(["apple", "application", "apply", "banana", "bandana"]);

        let results = tree.find_nearest_parallel("appli", 2);
        assert_eq!(results.len(), 2);

        // Verify same results as sequential
        let seq_results = tree.find_nearest("appli", 2);
        let par_texts: std::collections::HashSet<_> = results.iter().map(|r| &r.text).collect();
        let seq_texts: std::collections::HashSet<_> = seq_results.iter().map(|r| &r.text).collect();
        assert_eq!(par_texts, seq_texts);
    }
}
