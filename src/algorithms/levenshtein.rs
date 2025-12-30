//! Levenshtein (edit) distance implementation
//!
//! Optimized with:
//! - Myers bit-parallel algorithm for O(⌈m/64⌉n) time complexity
//! - Single-row DP fallback for strings > 64 chars
//! - Early termination with max distance threshold
//! - Unicode-aware character handling

use super::EditDistance;
use ahash::AHashMap;
use smallvec::SmallVec;

/// Maximum pattern length for Myers bit-parallel algorithm (64 bits per block)
const MYERS_BLOCK_SIZE: usize = 64;

// ============================================================================
// Myers Bit-Parallel Algorithm
// ============================================================================

/// Myers bit-parallel Levenshtein distance for patterns up to 64 characters.
///
/// This is the fastest known algorithm for edit distance, running in O(⌈m/64⌉n) time.
/// For patterns <= 64 chars, it processes the entire pattern in a single 64-bit word.
///
/// Based on: Myers, G. (1999). "A fast bit-vector algorithm for approximate string matching"
#[inline]
fn myers_64(pattern: &[char], text: &[char]) -> usize {
    let m = pattern.len();
    let n = text.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    if m > MYERS_BLOCK_SIZE {
        return dp_distance(pattern, text);
    }

    // Build pattern character masks (Peq)
    // For each character c, Peq[c] has bit i set if pattern[i] == c
    let mut peq: AHashMap<char, u64> = AHashMap::with_capacity(m.min(26));
    for (i, &c) in pattern.iter().enumerate() {
        *peq.entry(c).or_insert(0) |= 1u64 << i;
    }

    // Initialize bit vectors
    // Vp = all 1s (vertical positive: all increases by 1)
    // Vn = all 0s (vertical negative: no decreases)
    let mut vp: u64 = !0u64;
    let mut vn: u64 = 0u64;
    let mut score = m;

    // Mask for the m-th bit (0-indexed, so bit m-1)
    let mask = 1u64 << (m - 1);

    // Process each character in text
    for &tc in text.iter() {
        // Get character match vector
        let eq = peq.get(&tc).copied().unwrap_or(0);

        // Standard Myers algorithm
        let xv = eq | vn;
        let eq_and_vp = eq & vp;
        let xh = ((eq_and_vp.wrapping_add(vp)) ^ vp) | eq;

        let hp = vn | !(xh | vp);
        let hn = vp & xh;

        // Update score based on carry out at position m-1
        if (hp & mask) != 0 {
            score += 1;
        } else if (hn & mask) != 0 {
            score -= 1;
        }

        // Update vertical vectors for next iteration
        // Shift hp and hn left by 1 (with implicit 0 at position 0)
        // This is correct because the first row is 0,1,2,3... so hp[0] and hn[0] are fixed
        let hp_shifted = (hp << 1) | 1;  // Set bit 0 to 1 (first column always increases)
        let hn_shifted = hn << 1;

        vp = hn_shifted | !(xv | hp_shifted);
        vn = hp_shifted & xv;
    }

    score
}

/// Myers bit-parallel with max_distance threshold for early termination.
/// Returns None if distance exceeds threshold.
#[inline]
fn myers_64_bounded(pattern: &[char], text: &[char], max_distance: usize) -> Option<usize> {
    let m = pattern.len();
    let n = text.len();

    if m == 0 {
        return if n <= max_distance { Some(n) } else { None };
    }
    if n == 0 {
        return if m <= max_distance { Some(m) } else { None };
    }

    // Early exit if length difference exceeds threshold
    if m.abs_diff(n) > max_distance {
        return None;
    }

    if m > MYERS_BLOCK_SIZE {
        return dp_distance_bounded(pattern, text, max_distance);
    }

    // Build pattern character masks
    let mut peq: AHashMap<char, u64> = AHashMap::with_capacity(m.min(26));
    for (i, &c) in pattern.iter().enumerate() {
        *peq.entry(c).or_insert(0) |= 1u64 << i;
    }

    let mut vp: u64 = !0u64;
    let mut vn: u64 = 0u64;
    let mut score = m;

    let mask = 1u64 << (m - 1);
    let threshold = max_distance;

    for (j, &tc) in text.iter().enumerate() {
        let eq = peq.get(&tc).copied().unwrap_or(0);

        let xv = eq | vn;
        let eq_and_vp = eq & vp;
        let xh = ((eq_and_vp.wrapping_add(vp)) ^ vp) | eq;

        let hp = vn | !(xh | vp);
        let hn = vp & xh;

        if (hp & mask) != 0 {
            score += 1;
        } else if (hn & mask) != 0 {
            score -= 1;
        }

        // Early termination: if current score minus remaining potential improvements
        // still exceeds threshold, we can stop
        let remaining = n - j - 1;
        if score > threshold + remaining {
            return None;
        }

        let hp_shifted = (hp << 1) | 1;
        let hn_shifted = hn << 1;

        vp = hn_shifted | !(xv | hp_shifted);
        vn = hp_shifted & xv;
    }

    if score <= threshold {
        Some(score)
    } else {
        None
    }
}

/// Standard DP distance for char slices (fallback for long strings)
#[inline]
fn dp_distance(a: &[char], b: &[char]) -> usize {
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Ensure shorter string is on the column axis
    let (target, source) = if m < n { (a, b) } else { (b, a) };
    let n_target = target.len();

    let mut row: SmallVec<[usize; 64]> = (0..=n_target).collect();

    for (i, &sc) in source.iter().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;

        for j in 0..n_target {
            let cost = if sc == target[j] { 0 } else { 1 };
            let deletion = row[j + 1] + 1;
            let insertion = row[j] + 1;
            let substitution = prev + cost;

            prev = row[j + 1];
            row[j + 1] = substitution.min(deletion).min(insertion);
        }
    }

    row[n_target]
}

/// Standard DP distance with max_distance threshold (fallback for long strings)
#[inline]
fn dp_distance_bounded(a: &[char], b: &[char], max_distance: usize) -> Option<usize> {
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return if n <= max_distance { Some(n) } else { None };
    }
    if n == 0 {
        return if m <= max_distance { Some(m) } else { None };
    }

    if m.abs_diff(n) > max_distance {
        return None;
    }

    let (target, source) = if m < n { (a, b) } else { (b, a) };
    let n_target = target.len();

    let mut row: SmallVec<[usize; 64]> = (0..=n_target).collect();

    for (i, &sc) in source.iter().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        let mut row_min = row[0];

        for j in 0..n_target {
            let cost = if sc == target[j] { 0 } else { 1 };
            let deletion = row[j + 1] + 1;
            let insertion = row[j] + 1;
            let substitution = prev + cost;

            prev = row[j + 1];
            let cell = substitution.min(deletion).min(insertion);
            row[j + 1] = cell;
            row_min = row_min.min(cell);
        }

        if row_min > max_distance {
            return None;
        }
    }

    let result = row[n_target];
    if result <= max_distance {
        Some(result)
    } else {
        None
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Levenshtein distance calculator with optional early termination
///
/// # Complexity
/// - Time: O(m*n) where m and n are string lengths
/// - Space: O(min(m,n)) using single-row DP optimization
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Levenshtein {
    /// Maximum distance to compute (for early termination)
    pub max_distance: Option<usize>,
}

impl Levenshtein {
    #[must_use]
    pub fn new() -> Self {
        Self { max_distance: None }
    }

    #[must_use]
    pub fn with_max_distance(max_distance: usize) -> Self {
        Self { max_distance: Some(max_distance) }
    }

    /// Compute distance with proper Option semantics.
    /// Returns `None` if distance exceeds max_distance threshold.
    /// Returns `Some(distance)` otherwise.
    #[must_use]
    pub fn compute(&self, a: &str, b: &str) -> Option<usize> {
        levenshtein_distance_bounded(a, b, self.max_distance)
    }
}

impl EditDistance for Levenshtein {
    fn distance(&self, a: &str, b: &str) -> usize {
        // When max_distance is set and exceeded, return max_distance + 1
        // to indicate "greater than threshold" without using sentinel values
        match self.max_distance {
            Some(max_d) => levenshtein_distance_bounded(a, b, Some(max_d))
                .unwrap_or(max_d.saturating_add(1)),
            None => levenshtein_distance_bounded(a, b, None).unwrap_or(0),
        }
    }

    fn name(&self) -> &'static str {
        "levenshtein"
    }
}

/// Compute Levenshtein distance with optional max threshold.
///
/// Uses Myers bit-parallel algorithm for strings <= 64 chars (O(n) time),
/// falls back to standard DP for longer strings.
///
/// Returns `None` if distance exceeds `max_distance` (early termination).
/// Returns `Some(distance)` if distance is within threshold or no threshold set.
///
/// # Example
/// ```
/// use fuzzyrust::algorithms::levenshtein::levenshtein_distance_bounded;
///
/// // No threshold - always returns Some
/// assert_eq!(levenshtein_distance_bounded("kitten", "sitting", None), Some(3));
///
/// // Within threshold - returns actual distance
/// assert_eq!(levenshtein_distance_bounded("abc", "abd", Some(2)), Some(1));
///
/// // Exceeds threshold - returns None
/// assert_eq!(levenshtein_distance_bounded("abcdef", "ghijkl", Some(3)), None);
/// ```
#[inline]
#[must_use]
pub fn levenshtein_distance_bounded(a: &str, b: &str, max_distance: Option<usize>) -> Option<usize> {
    if a == b {
        return Some(0);
    }

    // Collect chars once
    let a_chars: SmallVec<[char; 64]> = a.chars().collect();
    let b_chars: SmallVec<[char; 64]> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return match max_distance {
            Some(max_d) if n > max_d => None,
            _ => Some(n),
        };
    }
    if n == 0 {
        return match max_distance {
            Some(max_d) if m > max_d => None,
            _ => Some(m),
        };
    }

    // Use Myers bit-parallel for the shorter string as pattern
    // This gives best performance since Myers is O(⌈m/64⌉n)
    let (pattern, text) = if m <= n {
        (&a_chars[..], &b_chars[..])
    } else {
        (&b_chars[..], &a_chars[..])
    };

    match max_distance {
        Some(max_d) => myers_64_bounded(pattern, text, max_d),
        None => Some(myers_64(pattern, text)),
    }
}

/// Compute Levenshtein distance using exponential search.
///
/// This is faster than the standard algorithm when the expected edit distance
/// is small relative to the string length. It works by trying progressively
/// larger thresholds (1, 2, 4, 8, ...) until the actual distance is found.
///
/// Use this when you expect strings to be similar (small edit distance).
///
/// # Example
/// ```
/// use fuzzyrust::algorithms::levenshtein::levenshtein_exp;
///
/// // For similar strings, exponential search is faster
/// assert_eq!(levenshtein_exp("kitten", "kittens"), 1);
/// assert_eq!(levenshtein_exp("hello", "hallo"), 1);
/// ```
#[inline]
#[must_use]
pub fn levenshtein_exp(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }

    let a_chars: SmallVec<[char; 64]> = a.chars().collect();
    let b_chars: SmallVec<[char; 64]> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Start with threshold of 1 and double until we find the distance
    let mut threshold = 1usize;
    let max_possible = m.max(n);

    let (pattern, text) = if m <= n {
        (&a_chars[..], &b_chars[..])
    } else {
        (&b_chars[..], &a_chars[..])
    };

    loop {
        if let Some(dist) = myers_64_bounded(pattern, text, threshold) {
            return dist;
        }

        // Double the threshold
        threshold = threshold.saturating_mul(2);

        // If threshold exceeds max possible distance, compute without bound
        if threshold >= max_possible {
            return myers_64(pattern, text);
        }
    }
}

/// Compute Levenshtein distance with optional max threshold.
///
/// **Deprecated**: Use `levenshtein_distance_bounded` for proper Option semantics.
/// This function returns `usize::MAX` when threshold is exceeded, which can be
/// confused with legitimate distance values.
///
/// Returns `usize::MAX` if distance exceeds max_distance (when provided).
#[inline]
#[must_use]
#[deprecated(since = "0.2.0", note = "Use levenshtein_distance_bounded for proper Option semantics")]
pub fn levenshtein_distance(a: &str, b: &str, max_distance: Option<usize>) -> usize {
    levenshtein_distance_bounded(a, b, max_distance).unwrap_or(usize::MAX)
}

/// Convenience function for simple distance calculation
#[inline]
#[must_use]
pub fn levenshtein(a: &str, b: &str) -> usize {
    // This never fails because there's no threshold
    levenshtein_distance_bounded(a, b, None).unwrap_or(0)
}

/// Convenience function for normalized similarity (0.0 to 1.0)
#[inline]
#[must_use]
pub fn levenshtein_similarity(a: &str, b: &str) -> f64 {
    let dist = levenshtein(a, b);
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        1.0
    } else {
        1.0 - (dist as f64 / max_len as f64)
    }
}

/// Compute Levenshtein similarity between a string and a char slice.
///
/// This is an optimization for cases where you already have a char slice
/// (e.g., a window into a longer string's chars), avoiding the need to
/// collect the chars back into a String for comparison.
///
/// # Arguments
/// * `s` - The source string
/// * `chars` - A char slice to compare against
///
/// # Example
/// ```
/// use fuzzyrust::algorithms::levenshtein::levenshtein_similarity_with_chars;
///
/// let chars: Vec<char> = "hello".chars().collect();
/// let score = levenshtein_similarity_with_chars("hallo", &chars);
/// assert!(score > 0.7);
/// ```
#[inline]
#[must_use]
pub fn levenshtein_similarity_with_chars(s: &str, chars: &[char]) -> f64 {
    let s_chars: SmallVec<[char; 64]> = s.chars().collect();
    let s_len = s_chars.len();
    let chars_len = chars.len();

    if s_len == 0 && chars_len == 0 {
        return 1.0;
    }
    if s_len == 0 || chars_len == 0 {
        return 0.0;
    }

    // Use the shorter one as pattern for Myers algorithm
    let (pattern, text) = if s_len <= chars_len {
        (&s_chars[..], chars)
    } else {
        (chars, &s_chars[..])
    };

    let dist = if pattern.len() <= MYERS_BLOCK_SIZE {
        myers_64(pattern, text)
    } else {
        dp_distance(pattern, text)
    };

    let max_len = s_len.max(chars_len);
    1.0 - (dist as f64 / max_len as f64)
}

// ============================================================================
// Grapheme Cluster Mode
// ============================================================================

use unicode_segmentation::UnicodeSegmentation;

/// Levenshtein distance treating grapheme clusters as single units.
///
/// This is useful for text with emoji sequences or combining characters
/// where a single visual character may be multiple Unicode code points.
///
/// # Examples
/// ```
/// use fuzzyrust::algorithms::levenshtein::levenshtein_grapheme;
///
/// // 👨‍👩‍👧‍👦 is 7 code points but 1 grapheme cluster
/// let family = "👨‍👩‍👧‍👦";
/// let man = "👨";
///
/// // With graphemes, the family emoji counts as 1 unit
/// assert_eq!(levenshtein_grapheme(family, man), 1);
/// ```
#[must_use]
pub fn levenshtein_grapheme(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }

    // Collect grapheme clusters
    let a_graphemes: SmallVec<[&str; 64]> = a.graphemes(true).collect();
    let b_graphemes: SmallVec<[&str; 64]> = b.graphemes(true).collect();

    let m = a_graphemes.len();
    let n = b_graphemes.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use standard DP for grapheme mode (simpler, still fast for typical strings)
    dp_distance_generic(&a_graphemes, &b_graphemes)
}

/// DP distance for generic comparable slices (used for grapheme mode)
fn dp_distance_generic<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let m = a.len();
    let n = b.len();

    // Use single-row DP with two rows optimization
    let mut prev: SmallVec<[usize; 128]> = (0..=n).collect();
    let mut curr: SmallVec<[usize; 128]> = SmallVec::with_capacity(n + 1);

    for i in 1..=m {
        curr.clear();
        curr.push(i);

        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            let val = (prev[j] + 1) // deletion
                .min(curr[j - 1] + 1) // insertion
                .min(prev[j - 1] + cost); // substitution
            curr.push(val);
        }

        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Grapheme-aware Levenshtein similarity (0.0 to 1.0)
///
/// Normalizes edit distance by the maximum grapheme cluster count.
#[inline]
#[must_use]
pub fn levenshtein_similarity_grapheme(a: &str, b: &str) -> f64 {
    let dist = levenshtein_grapheme(a, b);
    let max_len = a.graphemes(true).count().max(b.graphemes(true).count());
    if max_len == 0 {
        1.0
    } else {
        1.0 - (dist as f64 / max_len as f64)
    }
}

// ============================================================================
// SIMD-Accelerated Functions (via triple_accel)
// ============================================================================

/// SIMD-accelerated Levenshtein distance.
///
/// Uses triple_accel's SIMD-optimized implementation which can be 20-30x faster
/// than the scalar implementation for longer strings. Automatically falls back
/// to scalar on CPUs without SIMD support.
///
/// Best for:
/// - Long strings (>100 characters)
/// - Batch processing many comparisons
/// - x86/x86-64 CPUs with AVX2 or SSE4.1
///
/// # Examples
/// ```
/// use fuzzyrust::algorithms::levenshtein::levenshtein_simd;
///
/// assert_eq!(levenshtein_simd("kitten", "sitting"), 3);
/// assert_eq!(levenshtein_simd("hello", "hallo"), 1);
/// ```
#[inline]
#[must_use]
pub fn levenshtein_simd(a: &str, b: &str) -> usize {
    if a == b {
        return 0;
    }
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }

    // triple_accel works on bytes, which is fine for ASCII
    // For Unicode, we still get correct edit distance on byte level
    triple_accel::levenshtein::levenshtein(a.as_bytes(), b.as_bytes()) as usize
}

/// SIMD-accelerated Levenshtein distance with max threshold.
///
/// Returns `None` if the distance exceeds `max_distance`, enabling early termination.
/// Uses SIMD acceleration when available for optimal performance.
///
/// # Examples
/// ```
/// use fuzzyrust::algorithms::levenshtein::levenshtein_simd_bounded;
///
/// // Within threshold
/// assert_eq!(levenshtein_simd_bounded("hello", "hallo", 2), Some(1));
///
/// // Exceeds threshold
/// assert_eq!(levenshtein_simd_bounded("abc", "xyz", 2), None);
/// ```
#[inline]
#[must_use]
pub fn levenshtein_simd_bounded(a: &str, b: &str, max_distance: usize) -> Option<usize> {
    if a == b {
        return Some(0);
    }

    let a_len = a.len();
    let b_len = b.len();

    if a_len == 0 {
        return if b_len <= max_distance { Some(b_len) } else { None };
    }
    if b_len == 0 {
        return if a_len <= max_distance { Some(a_len) } else { None };
    }

    // Early exit if length difference exceeds threshold
    if a_len.abs_diff(b_len) > max_distance {
        return None;
    }

    // triple_accel's levenshtein_simd_k returns Option<u32>
    triple_accel::levenshtein::levenshtein_simd_k(
        a.as_bytes(),
        b.as_bytes(),
        max_distance as u32,
    ).map(|d| d as usize)
}

/// SIMD-accelerated Levenshtein similarity (0.0 to 1.0).
///
/// Uses SIMD acceleration for the distance calculation, providing
/// significant speedups on x86/x86-64 CPUs.
#[inline]
#[must_use]
pub fn levenshtein_similarity_simd(a: &str, b: &str) -> f64 {
    let dist = levenshtein_simd(a, b);
    let max_len = a.len().max(b.len());
    if max_len == 0 {
        1.0
    } else {
        1.0 - (dist as f64 / max_len as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_basic() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("saturday", "sunday"), 3);
    }

    #[test]
    fn test_levenshtein_unicode() {
        assert_eq!(levenshtein("café", "cafe"), 1);
        assert_eq!(levenshtein("日本語", "日本"), 1);
    }

    #[test]
    fn test_levenshtein_bounded_returns_none_when_exceeded() {
        // Exceeds threshold - returns None
        assert_eq!(levenshtein_distance_bounded("abcdef", "ghijkl", Some(3)), None);
        // Within threshold - returns actual distance
        assert_eq!(levenshtein_distance_bounded("abc", "abd", Some(2)), Some(1));
        // Equal strings always return Some(0)
        assert_eq!(levenshtein_distance_bounded("abc", "abc", Some(0)), Some(0));
    }

    #[test]
    fn test_levenshtein_bounded_no_threshold() {
        // Without threshold, always returns Some
        assert_eq!(levenshtein_distance_bounded("kitten", "sitting", None), Some(3));
        assert_eq!(levenshtein_distance_bounded("", "", None), Some(0));
        assert_eq!(levenshtein_distance_bounded("abc", "", None), Some(3));
    }

    #[test]
    fn test_levenshtein_struct_compute() {
        let lev = Levenshtein::with_max_distance(2);
        assert_eq!(lev.compute("abc", "abd"), Some(1));
        assert_eq!(lev.compute("abc", "xyz"), None);

        let lev_unbounded = Levenshtein::new();
        assert_eq!(lev_unbounded.compute("abc", "xyz"), Some(3));
    }

    #[test]
    #[allow(deprecated)]
    fn test_deprecated_levenshtein_distance() {
        // Backward compatibility: deprecated function still works
        assert_eq!(levenshtein_distance("abcdef", "ghijkl", Some(3)), usize::MAX);
        assert_eq!(levenshtein_distance("abc", "abd", Some(2)), 1);
    }

    #[test]
    fn test_levenshtein_exp() {
        // Exponential search should give same results as regular
        assert_eq!(levenshtein_exp("", ""), 0);
        assert_eq!(levenshtein_exp("abc", "abc"), 0);
        assert_eq!(levenshtein_exp("kitten", "sitting"), 3);
        assert_eq!(levenshtein_exp("kitten", "kittens"), 1);
        assert_eq!(levenshtein_exp("hello", "hallo"), 1);
        assert_eq!(levenshtein_exp("abc", ""), 3);
        assert_eq!(levenshtein_exp("", "xyz"), 3);
    }

    #[test]
    fn test_myers_algorithm() {
        // Test that Myers gives correct results
        assert_eq!(levenshtein("algorithm", "altruistic"), 6);
        assert_eq!(levenshtein("intention", "execution"), 5);
        assert_eq!(levenshtein("a", "b"), 1);
        assert_eq!(levenshtein("ab", "ba"), 2);  // swap is 2 edits in Levenshtein
    }
}
