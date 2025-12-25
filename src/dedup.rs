//! Deduplication utilities
//!
//! Provides functions for finding and grouping duplicate strings.

use crate::algorithms::Similarity;
use ahash::AHashMap;
use rayon::prelude::*;

/// Result from deduplication operation
#[derive(Debug, Clone)]
pub struct DeduplicationResult {
    /// Groups of duplicate items (each group contains similar strings)
    pub groups: Vec<Vec<String>>,
    /// Items that are unique (no duplicates found)
    pub unique: Vec<String>,
    /// Total number of duplicate items found
    pub total_duplicates: usize,
}

/// Deduplication methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupMethod {
    /// Compare all pairs (O(N^2)). Accurate but slow for large N.
    BruteForce,
    /// Sorted Neighborhood Method (O(N log N)). Fast, good for large N.
    /// Sorts items and compares within a sliding window.
    SortedNeighborhood { window_size: usize },
}

/// Union-Find data structure for efficient clustering
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            // Union by rank
            match self.rank[root_x].cmp(&self.rank[root_y]) {
                std::cmp::Ordering::Less => {
                    self.parent[root_x] = root_y;
                }
                std::cmp::Ordering::Greater => {
                    self.parent[root_y] = root_x;
                }
                std::cmp::Ordering::Equal => {
                    self.parent[root_y] = root_x;
                    self.rank[root_x] += 1;
                }
            }
        }
    }

    fn get_groups(&mut self, items: &[String]) -> Vec<Vec<String>> {
        let mut groups: AHashMap<usize, Vec<String>> = AHashMap::new();

        for (i, item) in items.iter().enumerate() {
            let root = self.find(i);
            groups.entry(root).or_insert_with(Vec::new).push(item.clone());
        }

        groups.into_values().collect()
    }
}

/// Find duplicate items in a list using the specified similarity algorithm.
///
/// # Arguments
/// * `items` - List of strings to deduplicate
/// * `similarity_fn` - Function that computes similarity between two strings
/// * `min_similarity` - Minimum similarity score to consider items as duplicates (0.0 to 1.0)
/// * `method` - Deduplication method to use
///
/// # Returns
/// DeduplicationResult containing groups of duplicates and unique items
pub fn find_duplicates<F>(
    items: &[String],
    similarity_fn: F,
    min_similarity: f64,
    method: DedupMethod,
) -> DeduplicationResult
where
    F: Fn(&str, &str) -> f64 + Sync + Send,
{
    match method {
        DedupMethod::BruteForce => find_duplicates_brute_force(items, similarity_fn, min_similarity),
        DedupMethod::SortedNeighborhood { window_size } => {
            find_duplicates_snm(items, similarity_fn, min_similarity, window_size)
        }
    }
}

/// Implementation of Brute Force deduplication (O(N^2))
fn find_duplicates_brute_force<F>(
    items: &[String],
    similarity_fn: F,
    min_similarity: f64,
) -> DeduplicationResult
where
    F: Fn(&str, &str) -> f64 + Sync + Send,
{
    if items.is_empty() {
        return DeduplicationResult {
            groups: vec![],
            unique: vec![],
            total_duplicates: 0,
        };
    }

    let n = items.len();
    if n == 1 {
        return DeduplicationResult {
            groups: vec![],
            unique: items.to_vec(),
            total_duplicates: 0,
        };
    }

    let mut uf = UnionFind::new(n);

    // Parallel processing with reduced memory footprint
    // Instead of collecting all pairs, we collect only the matches.
    // Chunking strategy: Iterate through rows `i` and compare with `j > i`.
    // Use references relative to the outer scope to overlap the lifetime with rayon's join
    // Note: rayon's par_iter closure requires `Send` + `Sync`.
    let similar_pairs: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            // We need to capture references for the inner closure
            let items_ref = items; 
            let sim_fn_ref = &similarity_fn;
            
            (i + 1..n).into_par_iter()
                .filter_map(move |j| {
                    let similarity = sim_fn_ref(&items_ref[i], &items_ref[j]);
                    if similarity >= min_similarity {
                        Some((i, j))
                    } else {
                        None
                    }
                })
        })
        .collect();

    // Union similar items
    for (i, j) in similar_pairs {
        uf.union(i, j);
    }

    // Get groups
    let all_groups = uf.get_groups(items);

    // Separate groups (size > 1) from unique items (size == 1)
    let mut groups = Vec::new();
    let mut unique = Vec::new();

    for group in all_groups {
        if group.len() > 1 {
            groups.push(group);
        } else {
            unique.extend(group);
        }
    }

    // Sort groups by size (largest first) for better readability
    groups.sort_by(|a, b| b.len().cmp(&a.len()));

    let total_duplicates = groups.iter().map(|g| g.len() - 1).sum();

    DeduplicationResult {
        groups,
        unique,
        total_duplicates,
    }
}

/// Implementation of Sorted Neighborhood Method (O(N log N))
fn find_duplicates_snm<F>(
    items: &[String],
    similarity_fn: F,
    min_similarity: f64,
    window_size: usize,
) -> DeduplicationResult
where
    F: Fn(&str, &str) -> f64 + Sync + Send,
{
    if items.is_empty() {
        return DeduplicationResult {
            groups: vec![],
            unique: vec![],
            total_duplicates: 0,
        };
    }

    if items.len() == 1 {
        return DeduplicationResult {
            groups: vec![],
            unique: items.to_vec(),
            total_duplicates: 0,
        };
    }

    let n = items.len();
    
    // 1. Create indexed items for sorting
    // We store (original_index, string_slice)
    let mut indexed_items: Vec<(usize, &str)> = items
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s.as_str()))
        .collect();

    // 2. Sort items lexicographically
    // This brings similar strings (sharing prefixes) close together
    indexed_items.par_sort_unstable_by(|(_, a), (_, b)| a.cmp(b));

    let mut uf = UnionFind::new(n);

    // 3. Sliding window comparison
    // For each item in the sorted list, compare with the next `window_size` items
    let window_size = window_size.max(1);
    
    // Parallel processing using direct range iteration
    // We iterate 0..n and for each item check the window ahead.
    
    // We need to share `indexed_items` across threads. Since it's a read-only Vec, &Vec is Sync.
    // However, into_par_iter() closures usually require move.
    // We can use a reference wrapper or just refer to the slice if we're careful.
    let indexed_items_ref = &indexed_items;

    let valid_matches: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let end = (i + 1 + window_size).min(n);
            let (orig_idx_i, str_i) = indexed_items_ref[i];
            let sim_fn_ref = &similarity_fn;
            let items_slice = indexed_items_ref;
            
            // Inner loop: check window
            (i + 1..end).into_par_iter().filter_map(move |j| {
                let (orig_idx_j, str_j) = items_slice[j];
                let similarity = sim_fn_ref(str_i, str_j);
                
                if similarity >= min_similarity {
                    Some((orig_idx_i, orig_idx_j))
                } else {
                    None
                }
            })
        })
        .collect();

    // Union valid matches
    for (i, j) in valid_matches {
        uf.union(i, j);
    }

    // 4. Group results (same as brute force)
    let all_groups = uf.get_groups(items);

    let mut groups = Vec::new();
    let mut unique = Vec::new();

    for group in all_groups {
        if group.len() > 1 {
            groups.push(group);
        } else {
            unique.extend(group);
        }
    }

    groups.sort_by(|a, b| b.len().cmp(&a.len()));
    let total_duplicates = groups.iter().map(|g| g.len() - 1).sum();

    DeduplicationResult {
        groups,
        unique,
        total_duplicates,
    }
}

/// Find duplicates using a trait-based similarity metric
pub fn find_duplicates_with_metric<S: Similarity + ?Sized>(
    items: &[String],
    metric: &S,
    min_similarity: f64,
    method: DedupMethod,
) -> DeduplicationResult {
    find_duplicates(items, |a, b| metric.similarity(a, b), min_similarity, method)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::jaro::JaroWinkler;

    #[test]
    fn test_find_duplicates_empty() {
        let items: Vec<String> = vec![];
        let result = find_duplicates(
            &items, 
            |a, b| a.len().min(b.len()) as f64, 
            0.8,
            DedupMethod::BruteForce
        );
        assert_eq!(result.groups.len(), 0);
        assert_eq!(result.unique.len(), 0);
        assert_eq!(result.total_duplicates, 0);
    }


    #[test]
    fn test_find_duplicates_single() {
        let items = vec!["hello".to_string()];
        let result = find_duplicates(
            &items, 
            |a, b| if a == b { 1.0 } else { 0.0 }, 
            0.8,
            DedupMethod::BruteForce
        );
        assert_eq!(result.groups.len(), 0);
        assert_eq!(result.unique.len(), 1);
        assert_eq!(result.total_duplicates, 0);
    }


    #[test]
    fn test_find_duplicates_basic() {
        let items = vec![
            "hello".to_string(),
            "helo".to_string(),
            "world".to_string(),
            "hello".to_string(),
        ];

        let jw = JaroWinkler::new();
        let result = find_duplicates_with_metric(
            &items, 
            &jw, 
            0.85, 
            DedupMethod::BruteForce
        );

        // "hello", "helo", and "hello" should be grouped
        // "world" should be unique
        assert!(result.groups.len() > 0);
        assert!(result.unique.contains(&"world".to_string()));
    }

    #[test]
    fn test_find_duplicates_all_unique() {
        let items = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ];

        let jw = JaroWinkler::new();
        let result = find_duplicates_with_metric(
            &items, 
            &jw, 
            0.9, 
            DedupMethod::BruteForce
        );

        assert_eq!(result.groups.len(), 0);
        assert_eq!(result.unique.len(), 3);
        assert_eq!(result.total_duplicates, 0);
    }

    #[test]
    fn test_find_duplicates_all_same() {
        let items = vec![
            "test".to_string(),
            "test".to_string(),
            "test".to_string(),
        ];

        let result = find_duplicates(
            &items, 
            |a, b| if a == b { 1.0 } else { 0.0 }, 
            0.99,
            DedupMethod::BruteForce
        );

        assert_eq!(result.groups.len(), 1);
        assert_eq!(result.groups[0].len(), 3);
        assert_eq!(result.unique.len(), 0);
        assert_eq!(result.total_duplicates, 2);
    }
    #[test]
    #[ignore]
    fn test_memory_bomb_fix_stress() {
        use crate::algorithms::levenshtein::levenshtein_similarity;
        
        // Simulating memory bomb scenario
        // 5000 items -> 12.5M pairs.
        // Original implementation allocated vector of 12.5M * 16 bytes ~ 200MB.
        // While 200MB isn't a crash on modern systems, 20k items would be 200M pairs (~3.2GB).
        // We use 5000 to keep test time reasonable but ensure logic holds.
        // Key is that if it tried to collect all pairs, memory usage would spike linearly with N^2.
        
        let n = 5000;
        let items: Vec<String> = (0..n).map(|i| format!("item_{}", i)).collect();
        
        let result = find_duplicates(
            &items,
            levenshtein_similarity,
            0.9,
            DedupMethod::BruteForce
        );
        
        // Just verify it completes successfully
        assert_eq!(result.total_duplicates, 0); // items distinct enough
    }
}
