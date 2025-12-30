//! Evaluation metrics for assessing match quality.
//!
//! Provides precision, recall, F-score, and confusion matrix calculations
//! for validating fuzzy matching and deduplication results.

use std::collections::HashSet;

/// Confusion matrix values for classification evaluation.
#[derive(Debug, Clone, Default)]
pub struct ConfusionMatrix {
    /// True positives: correctly predicted matches
    pub true_positives: usize,
    /// False positives: incorrectly predicted matches
    pub false_positives: usize,
    /// False negatives: missed matches
    pub false_negatives: usize,
    /// True negatives: correctly rejected non-matches
    pub true_negatives: usize,
}

impl ConfusionMatrix {
    /// Calculate precision from confusion matrix values.
    pub fn precision(&self) -> f64 {
        let denominator = self.true_positives + self.false_positives;
        if denominator == 0 {
            if self.false_negatives == 0 {
                1.0 // No predictions and no actual positives = perfect
            } else {
                0.0 // No predictions but there were actual positives
            }
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    /// Calculate recall from confusion matrix values.
    pub fn recall(&self) -> f64 {
        let denominator = self.true_positives + self.false_negatives;
        if denominator == 0 {
            if self.false_positives == 0 {
                1.0 // No actual positives and no false positives = perfect
            } else {
                0.0 // No actual positives but there were predictions
            }
        } else {
            self.true_positives as f64 / denominator as f64
        }
    }

    /// Calculate F-beta score from confusion matrix values.
    pub fn f_score(&self, beta: f64) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            let beta_sq = beta * beta;
            (1.0 + beta_sq) * p * r / (beta_sq * p + r)
        }
    }
}

/// Compute precision: TP / (TP + FP)
///
/// Precision measures the accuracy of positive predictions.
/// A precision of 1.0 means no false positives.
///
/// # Arguments
/// * `true_matches` - Set of actual match pairs (ground truth)
/// * `predicted_matches` - Set of predicted match pairs
///
/// # Returns
/// Precision score between 0.0 and 1.0
pub fn precision(
    true_matches: &HashSet<(usize, usize)>,
    predicted_matches: &HashSet<(usize, usize)>,
) -> f64 {
    if predicted_matches.is_empty() {
        return if true_matches.is_empty() { 1.0 } else { 0.0 };
    }
    let tp = predicted_matches.intersection(true_matches).count();
    tp as f64 / predicted_matches.len() as f64
}

/// Compute recall: TP / (TP + FN)
///
/// Recall measures the completeness of positive predictions.
/// A recall of 1.0 means no false negatives (all true matches found).
///
/// # Arguments
/// * `true_matches` - Set of actual match pairs (ground truth)
/// * `predicted_matches` - Set of predicted match pairs
///
/// # Returns
/// Recall score between 0.0 and 1.0
pub fn recall(
    true_matches: &HashSet<(usize, usize)>,
    predicted_matches: &HashSet<(usize, usize)>,
) -> f64 {
    if true_matches.is_empty() {
        return if predicted_matches.is_empty() {
            1.0
        } else {
            0.0
        };
    }
    let tp = predicted_matches.intersection(true_matches).count();
    tp as f64 / true_matches.len() as f64
}

/// Compute F-beta score: weighted harmonic mean of precision and recall.
///
/// F1 score (beta=1.0) gives equal weight to precision and recall.
/// F0.5 (beta=0.5) weighs precision higher than recall.
/// F2 (beta=2.0) weighs recall higher than precision.
///
/// # Arguments
/// * `true_matches` - Set of actual match pairs (ground truth)
/// * `predicted_matches` - Set of predicted match pairs
/// * `beta` - Weight parameter (default 1.0 for F1 score)
///
/// # Returns
/// F-beta score between 0.0 and 1.0
pub fn f_score(
    true_matches: &HashSet<(usize, usize)>,
    predicted_matches: &HashSet<(usize, usize)>,
    beta: f64,
) -> f64 {
    let p = precision(true_matches, predicted_matches);
    let r = recall(true_matches, predicted_matches);
    if p + r == 0.0 {
        return 0.0;
    }
    let beta_sq = beta * beta;
    (1.0 + beta_sq) * p * r / (beta_sq * p + r)
}

/// Compute confusion matrix from match sets.
///
/// # Arguments
/// * `true_matches` - Set of actual match pairs (ground truth)
/// * `predicted_matches` - Set of predicted match pairs
/// * `total_pairs` - Total number of possible pairs (for computing TN)
///
/// # Returns
/// ConfusionMatrix with TP, FP, FN, TN counts
pub fn confusion_matrix(
    true_matches: &HashSet<(usize, usize)>,
    predicted_matches: &HashSet<(usize, usize)>,
    total_pairs: usize,
) -> ConfusionMatrix {
    let tp = predicted_matches.intersection(true_matches).count();
    let fp = predicted_matches.difference(true_matches).count();
    let fn_ = true_matches.difference(predicted_matches).count();
    let tn = total_pairs.saturating_sub(tp + fp + fn_);

    ConfusionMatrix {
        true_positives: tp,
        false_positives: fp,
        false_negatives: fn_,
        true_negatives: tn,
    }
}

/// Normalize a pair to ensure consistent ordering (smaller ID first).
///
/// This is useful for comparing match sets where (1, 2) and (2, 1)
/// should be considered the same pair.
pub fn normalize_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Convert a list of pairs to a HashSet with normalized ordering.
pub fn normalize_pairs(pairs: &[(usize, usize)]) -> HashSet<(usize, usize)> {
    pairs.iter().map(|&(a, b)| normalize_pair(a, b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_perfect() {
        let true_matches: HashSet<_> = [(0, 1), (1, 2)].into_iter().collect();
        let predicted: HashSet<_> = [(0, 1), (1, 2)].into_iter().collect();
        assert_eq!(precision(&true_matches, &predicted), 1.0);
    }

    #[test]
    fn test_precision_partial() {
        let true_matches: HashSet<_> = [(0, 1)].into_iter().collect();
        let predicted: HashSet<_> = [(0, 1), (2, 3)].into_iter().collect();
        assert_eq!(precision(&true_matches, &predicted), 0.5);
    }

    #[test]
    fn test_precision_empty() {
        let true_matches: HashSet<(usize, usize)> = HashSet::new();
        let predicted: HashSet<(usize, usize)> = HashSet::new();
        assert_eq!(precision(&true_matches, &predicted), 1.0);
    }

    #[test]
    fn test_recall_perfect() {
        let true_matches: HashSet<_> = [(0, 1), (1, 2)].into_iter().collect();
        let predicted: HashSet<_> = [(0, 1), (1, 2)].into_iter().collect();
        assert_eq!(recall(&true_matches, &predicted), 1.0);
    }

    #[test]
    fn test_recall_partial() {
        let true_matches: HashSet<_> = [(0, 1), (1, 2)].into_iter().collect();
        let predicted: HashSet<_> = [(0, 1)].into_iter().collect();
        assert_eq!(recall(&true_matches, &predicted), 0.5);
    }

    #[test]
    fn test_f_score_perfect() {
        let true_matches: HashSet<_> = [(0, 1)].into_iter().collect();
        let predicted: HashSet<_> = [(0, 1)].into_iter().collect();
        assert_eq!(f_score(&true_matches, &predicted, 1.0), 1.0);
    }

    #[test]
    fn test_f_score_zero() {
        let true_matches: HashSet<_> = [(0, 1)].into_iter().collect();
        let predicted: HashSet<_> = [(2, 3)].into_iter().collect();
        assert_eq!(f_score(&true_matches, &predicted, 1.0), 0.0);
    }

    #[test]
    fn test_confusion_matrix() {
        let true_matches: HashSet<_> = [(0, 1), (1, 2)].into_iter().collect();
        let predicted: HashSet<_> = [(0, 1), (2, 3)].into_iter().collect();
        let cm = confusion_matrix(&true_matches, &predicted, 10);
        assert_eq!(cm.true_positives, 1);
        assert_eq!(cm.false_positives, 1);
        assert_eq!(cm.false_negatives, 1);
        assert_eq!(cm.true_negatives, 7);
    }

    #[test]
    fn test_normalize_pair() {
        assert_eq!(normalize_pair(1, 2), (1, 2));
        assert_eq!(normalize_pair(2, 1), (1, 2));
        assert_eq!(normalize_pair(5, 5), (5, 5));
    }
}
