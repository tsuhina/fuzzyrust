///! Field-specific index implementations for schema-based multi-field matching
///!
///! This module provides specialized index implementations for different field types.
///! Each field type uses an optimal index structure and algorithm.

use super::types::{Algorithm, FieldType};
use crate::algorithms::{self, Similarity};
use crate::indexing::{NgramIndex, HybridIndex};
use ahash::{AHashMap, AHashSet};

/// Result from a field index search
#[derive(Debug, Clone)]
pub struct FieldSearchMatch {
    /// Record ID
    pub id: usize,
    /// Field value
    pub value: String,
    /// Similarity score (0.0-1.0)
    pub score: f64,
    /// Optional user data
    pub data: Option<u64>,
}

/// Trait for field-specific indices
///
/// Each field type implements this trait to provide:
/// - Adding values to the index
/// - Searching for similar values
pub trait FieldIndex: Send + Sync {
    /// Add a value to the index
    fn add(&mut self, id: usize, value: &str, data: Option<u64>);

    /// Search for similar values
    ///
    /// Returns matches sorted by descending similarity score.
    fn search(&self, query: &str, min_similarity: f64, limit: Option<usize>) -> Vec<FieldSearchMatch>;

    /// Get the index size (number of entries)
    fn len(&self) -> usize;

    /// Check if index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Short text field index
///
/// Optimized for names, titles, short identifiers (typically ≤100 chars).
/// Uses N-gram index with Jaro-Winkler or Levenshtein similarity.
pub struct ShortTextIndex {
    inner: NgramIndex,
    algorithm: Algorithm,
    /// Maps internal index ID to record ID for correct lookups
    id_mapping: Vec<usize>,
}

impl ShortTextIndex {
    /// Create a new short text index with the specified algorithm
    pub fn new(algorithm: Algorithm) -> Self {
        Self {
            inner: NgramIndex::new(2), // 2-grams work well for short text
            algorithm,
            id_mapping: Vec::new(),
        }
    }
}

impl FieldIndex for ShortTextIndex {
    fn add(&mut self, id: usize, value: &str, data: Option<u64>) {
        // Store mapping from internal index ID to record ID
        self.id_mapping.push(id);
        self.inner.add_with_data(value.to_string(), data);
    }

    fn search(&self, query: &str, min_similarity: f64, limit: Option<usize>) -> Vec<FieldSearchMatch> {
        // Match on algorithm and call search with the concrete type
        let raw_matches = match &self.algorithm {
            Algorithm::Levenshtein => {
                let sim = algorithms::levenshtein::Levenshtein::new();
                self.inner.search(query, &sim, min_similarity, limit)
            }
            Algorithm::DamerauLevenshtein => {
                let sim = algorithms::damerau::DamerauLevenshtein::new();
                self.inner.search(query, &sim, min_similarity, limit)
            }
            Algorithm::JaroWinkler(config) => {
                let sim = algorithms::jaro::JaroWinkler::new()
                    .with_prefix_weight(config.prefix_weight);
                self.inner.search(query, &sim, min_similarity, limit)
            }
            Algorithm::Ngram { ngram_size } => {
                let sim = algorithms::ngram::Ngram::new(*ngram_size);
                self.inner.search(query, &sim, min_similarity, limit)
            }
            Algorithm::ExactMatch => {
                let sim = ExactMatchSimilarity;
                self.inner.search(query, &sim, min_similarity, limit)
            }
            _ => {
                // Default fallback to Jaro-Winkler
                let sim = algorithms::jaro::JaroWinkler::new();
                self.inner.search(query, &sim, min_similarity, limit)
            }
        };

        raw_matches
            .into_iter()
            .filter_map(|m| {
                // Map internal index ID to record ID
                let record_id = self.id_mapping.get(m.id).copied()?;
                Some(FieldSearchMatch {
                    id: record_id,
                    value: m.text,
                    score: m.similarity,
                    data: m.data,
                })
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Long text field index
///
/// Optimized for descriptions, documents, paragraphs (>100 chars).
/// Uses Hybrid index with N-gram similarity for efficiency.
pub struct LongTextIndex {
    inner: HybridIndex,
    algorithm: Algorithm,
    /// Maps internal index ID to record ID for correct lookups
    id_mapping: Vec<usize>,
}

impl LongTextIndex {
    /// Create a new long text index with the specified algorithm
    pub fn new(algorithm: Algorithm) -> Self {
        let ngram_size = match &algorithm {
            Algorithm::Ngram { ngram_size } => *ngram_size,
            _ => 3, // Default: 3-grams for long text
        };

        Self {
            inner: HybridIndex::new(ngram_size),
            algorithm,
            id_mapping: Vec::new(),
        }
    }
}

impl FieldIndex for LongTextIndex {
    fn add(&mut self, id: usize, value: &str, data: Option<u64>) {
        // Store mapping from internal index ID to record ID
        self.id_mapping.push(id);
        self.inner.add_with_data(value.to_string(), data);
    }

    fn search(&self, query: &str, min_similarity: f64, limit: Option<usize>) -> Vec<FieldSearchMatch> {
        // Match on algorithm and call search with the concrete type
        let raw_matches = match &self.algorithm {
            Algorithm::Ngram { ngram_size } => {
                let sim = algorithms::ngram::Ngram::new(*ngram_size);
                self.inner.search(query, &sim, min_similarity, limit)
            }
            Algorithm::Cosine => {
                let sim = algorithms::cosine::CosineSimilarity::new();
                self.inner.search(query, &sim, min_similarity, limit)
            }
            Algorithm::Levenshtein => {
                let sim = algorithms::levenshtein::Levenshtein::new();
                self.inner.search(query, &sim, min_similarity, limit)
            }
            Algorithm::ExactMatch => {
                let sim = ExactMatchSimilarity;
                self.inner.search(query, &sim, min_similarity, limit)
            }
            _ => {
                // Default fallback to 3-gram
                let sim = algorithms::ngram::Ngram::new(3);
                self.inner.search(query, &sim, min_similarity, limit)
            }
        };

        raw_matches
            .into_iter()
            .filter_map(|m| {
                // Map internal index ID to record ID
                let record_id = self.id_mapping.get(m.id).copied()?;
                Some(FieldSearchMatch {
                    id: record_id,
                    value: m.text,
                    score: m.similarity,
                    data: m.data,
                })
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Token set field index
///
/// Optimized for tags, categories, comma-separated values.
/// Uses inverted index with Jaccard similarity.
///
/// # Design Invariant
///
/// The `inverted_index` stores indices into the `entries` vector. These indices
/// are stored at insertion time as `self.entries.len()` (the next index).
/// **Entries are never removed or reordered**, so these indices remain valid
/// for the lifetime of the index. If this invariant is ever violated (e.g., by
/// adding a `remove()` method), the inverted index would need to be rebuilt.
pub struct TokenSetIndex {
    /// Token -> list of entry indices (into `entries` vector)
    inverted_index: AHashMap<String, Vec<usize>>,
    /// All indexed entries (append-only, never removed)
    entries: Vec<TokenSetEntry>,
    /// Separator string for tokenization (supports multi-character separators like ", " or " | ")
    separator: String,
}

#[derive(Debug, Clone)]
struct TokenSetEntry {
    id: usize,
    tokens: AHashSet<String>,
    original: String,
    data: Option<u64>,
}

impl TokenSetIndex {
    /// Create a new token set index with the specified separator
    ///
    /// # Arguments
    /// * `separator` - The string used to split token sets (e.g., "," or ", " or " | ")
    pub fn new(separator: impl Into<String>) -> Self {
        Self {
            inverted_index: AHashMap::new(),
            entries: Vec::new(),
            separator: separator.into(),
        }
    }

    /// Tokenize a string using the configured separator
    fn tokenize(&self, value: &str) -> AHashSet<String> {
        value
            .split(&self.separator)
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Compute Jaccard similarity between two token sets
    fn jaccard_similarity(&self, query_tokens: &AHashSet<String>, entry_tokens: &AHashSet<String>) -> f64 {
        if query_tokens.is_empty() && entry_tokens.is_empty() {
            return 1.0;
        }

        let intersection: usize = query_tokens.intersection(entry_tokens).count();
        let union: usize = query_tokens.union(entry_tokens).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

impl FieldIndex for TokenSetIndex {
    fn add(&mut self, id: usize, value: &str, data: Option<u64>) {
        let tokens = self.tokenize(value);

        // Add to inverted index
        for token in &tokens {
            self.inverted_index
                .entry(token.clone())
                .or_insert_with(Vec::new)
                .push(self.entries.len());
        }

        // Add entry
        self.entries.push(TokenSetEntry {
            id,
            tokens,
            original: value.to_string(),
            data,
        });
    }

    fn search(&self, query: &str, min_similarity: f64, limit: Option<usize>) -> Vec<FieldSearchMatch> {
        let query_tokens = self.tokenize(query);

        if query_tokens.is_empty() {
            return Vec::new();
        }

        // Get candidates from inverted index
        let mut candidate_ids = AHashSet::new();
        for token in &query_tokens {
            if let Some(ids) = self.inverted_index.get(token) {
                candidate_ids.extend(ids.iter().copied());
            }
        }

        // Score candidates
        let mut matches: Vec<FieldSearchMatch> = candidate_ids
            .into_iter()
            .filter_map(|idx| {
                // Use bounds-checked access to prevent potential panic if indices get out of sync
                let entry = self.entries.get(idx)?;
                let score = self.jaccard_similarity(&query_tokens, &entry.tokens);

                if score >= min_similarity {
                    Some(FieldSearchMatch {
                        id: entry.id,
                        value: entry.original.clone(),
                        score,
                        data: entry.data,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Debug assertion to catch NaN scores early
        #[cfg(debug_assertions)]
        for m in &matches {
            debug_assert!(!m.score.is_nan(), "NaN score detected for value: {}", m.value);
        }

        // Sort by descending score
        matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Apply limit
        if let Some(limit) = limit {
            matches.truncate(limit);
        }

        matches
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Create a field index based on field type
pub fn create_field_index(field_type: &FieldType, algorithm: &Algorithm) -> Box<dyn FieldIndex> {
    match field_type {
        FieldType::ShortText { .. } => Box::new(ShortTextIndex::new(algorithm.clone())),
        FieldType::LongText { .. } => Box::new(LongTextIndex::new(algorithm.clone())),
        FieldType::TokenSet { separator, .. } => Box::new(TokenSetIndex::new(separator.clone())),
    }
}

/// Exact match similarity (for ExactMatch algorithm)
struct ExactMatchSimilarity;

impl Similarity for ExactMatchSimilarity {
    fn similarity(&self, a: &str, b: &str) -> f64 {
        if a == b {
            1.0
        } else {
            0.0
        }
    }

    fn name(&self) -> &'static str {
        "exact_match"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_text_index() {
        let mut index = ShortTextIndex::new(Algorithm::JaroWinkler(Default::default()));

        index.add(0, "John Doe", Some(100));
        index.add(1, "Jane Doe", Some(101));
        index.add(2, "John Smith", Some(102));

        let matches = index.search("Jon Doe", 0.7, Some(10));

        assert!(!matches.is_empty());
        // "John Doe" should be the best match
        assert_eq!(matches[0].id, 0);
        assert!(matches[0].score > 0.8);
    }

    #[test]
    fn test_long_text_index() {
        let mut index = LongTextIndex::new(Algorithm::Ngram { ngram_size: 2 });

        index.add(
            0,
            "The quick brown fox jumps over the lazy dog",
            Some(200),
        );
        index.add(
            1,
            "A fast brown fox leaps over a sleepy dog",
            Some(201),
        );
        index.add(
            2,
            "Python is a high-level programming language",
            Some(202),
        );

        let matches = index.search("quick brown fox", 0.3, Some(10));

        assert!(!matches.is_empty());
        // First two should match, third shouldn't
        assert!(matches.iter().any(|m| m.id == 0 || m.id == 1));
    }

    #[test]
    fn test_token_set_index() {
        let mut index = TokenSetIndex::new(",");

        index.add(0, "python, rust, golang", Some(300));
        index.add(1, "python, java, c++", Some(301));
        index.add(2, "rust, c, assembly", Some(302));

        // With threshold 0.3:
        // - "python, rust, golang" = 2/3 = 0.667 (matches)
        // - "python, java, c++" = 1/4 = 0.25 (doesn't match)
        // - "rust, c, assembly" = 1/4 = 0.25 (doesn't match)
        let matches = index.search("python, rust", 0.3, Some(10));
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, 0);
        assert!((matches[0].score - 0.667).abs() < 0.01);

        // With lower threshold, all 3 should match
        let matches_all = index.search("python, rust", 0.2, Some(10));
        assert_eq!(matches_all.len(), 3);
        assert_eq!(matches_all[0].id, 0); // Highest score
    }

    #[test]
    fn test_token_set_exact_match() {
        let mut index = TokenSetIndex::new(",");

        index.add(0, "tag1, tag2, tag3", None);

        let matches = index.search("tag1, tag2, tag3", 0.9, Some(10));

        assert_eq!(matches.len(), 1);
        assert!((matches[0].score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_token_set_empty_query() {
        let mut index = TokenSetIndex::new(",");

        index.add(0, "tag1, tag2", None);

        let matches = index.search("", 0.0, Some(10));

        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_token_set_multi_char_separator() {
        // Test multi-character separator like " | "
        let mut index = TokenSetIndex::new(" | ");

        index.add(0, "python | rust | golang", Some(400));
        index.add(1, "python | java | c++", Some(401));

        let matches = index.search("python | rust", 0.3, Some(10));
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, 0);
    }

    #[test]
    fn test_field_index_len() {
        let mut index = ShortTextIndex::new(Algorithm::Levenshtein);

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        index.add(0, "test", None);
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_create_field_index() {
        let short_text_type = FieldType::ShortText {
            max_length: 100,
            default_algorithm: Algorithm::JaroWinkler(Default::default()),
        };
        let index = create_field_index(&short_text_type, &Algorithm::Levenshtein);
        assert!(index.is_empty());

        let token_set_type = FieldType::TokenSet {
            separator: ",".to_string(),
            default_algorithm: Algorithm::Jaccard,
        };
        let index = create_field_index(&token_set_type, &Algorithm::Jaccard);
        assert!(index.is_empty());
    }
}
