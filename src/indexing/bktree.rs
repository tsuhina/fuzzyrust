//! BK-tree (Burkhard-Keller tree) implementation
//! 
//! A metric tree structure that enables fast fuzzy searching.
//! Works with any edit distance metric that satisfies triangle inequality.

use ahash::AHashMap;
use rayon::prelude::*;
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
        Self { id, text, distance, data: None }
    }

    pub fn with_data(id: usize, text: String, distance: usize, data: u64) -> Self {
        Self { id, text, distance, data: Some(data) }
    }
}

/// A node in the BK-tree
#[derive(Debug, Clone)]
struct BkNode {
    /// Unique ID for this entry
    id: usize,
    /// The string stored at this node
    text: String,
    /// Optional associated data (e.g., user-provided ID)
    data: Option<u64>,
    /// Children indexed by distance
    children: AHashMap<usize, Box<BkNode>>,
}

impl BkNode {
    fn new(id: usize, text: String, data: Option<u64>) -> Self {
        Self {
            id,
            text,
            data,
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
#[derive(Clone)]
pub struct BkTree {
    root: Option<Box<BkNode>>,
    distance_fn: DistanceFn,
    size: usize,
    /// Next ID to assign to a new entry.
    /// Note: Overflow is not checked (would require 2^64 items on 64-bit systems).
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
            self.next_id += 1;
            return true;
        };

        let mut depth = 0;

        loop {
            depth += 1;
            // Safety check: prevent infinite loops from buggy distance functions.
            // In debug mode: panic with clear message for debugging.
            // In release mode: reject the item and return false.
            if depth > MAX_TREE_DEPTH {
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
                    self.next_id += 1;
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

        if dist <= max_distance {
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
        
        if dist == 0 {
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
        self.next_id = 0;
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
}
