//! Scoring strategies for combining multi-field similarity scores
//!
//! This module provides different strategies for combining individual field scores
//! into a single overall similarity score for a multi-field match.
//!
//! Available strategies:
//! - WeightedAverage: Normalize and weight field scores
//! - MinMaxScaling: Scale scores to 0-1 before weighting

use serde::{Deserialize, Serialize};

/// Scoring strategy for combining field scores
///
/// Each strategy defines how individual field similarity scores are combined
/// into an overall record similarity score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScoringStrategy {
    /// Weighted average of field scores
    ///
    /// Computes: Σ(score_i × weight_i) / Σ(weight_i)
    ///
    /// Each field score is multiplied by its weight, then divided by the
    /// sum of all weights. This normalizes the result to 0-1 regardless
    /// of the absolute weight values.
    ///
    /// **Best for**: Most use cases. Simple, intuitive, well-normalized.
    WeightedAverage,

    /// Min-max normalization before weighted averaging
    ///
    /// First normalizes all field scores to the same 0-1 range using
    /// min-max scaling, then applies weighted average.
    ///
    /// Scaling: (score - min) / (max - min)
    ///
    /// **Best for**: When field scores have different natural ranges or
    /// some algorithms produce consistently higher/lower scores.
    MinMaxScaling,
}

/// Individual field score with metadata
#[derive(Debug, Clone, PartialEq)]
pub struct FieldScore {
    /// Field name
    pub field_name: String,

    /// Similarity score (0.0 - 1.0)
    pub score: f64,

    /// Field weight
    pub weight: f64,
}

impl FieldScore {
    /// Create a new field score
    pub fn new(field_name: impl Into<String>, score: f64, weight: f64) -> Self {
        Self {
            field_name: field_name.into(),
            score: score.clamp(0.0, 1.0),
            weight: weight.clamp(0.0, 10.0),
        }
    }

    /// Get the weighted score (score × weight)
    pub fn weighted_score(&self) -> f64 {
        self.score * self.weight
    }
}

/// Weighted average scoring strategy
pub struct WeightedAverage;

impl WeightedAverage {
    /// Combine field scores using weighted average
    ///
    /// Formula: Σ(score_i × weight_i) / Σ(weight_i)
    ///
    /// # Arguments
    ///
    /// * `field_scores` - Individual field scores with weights
    ///
    /// # Returns
    ///
    /// Overall similarity score in range [0.0, 1.0]
    pub fn combine(field_scores: &[FieldScore]) -> f64 {
        if field_scores.is_empty() {
            return 0.0;
        }

        let mut total_weighted_score = 0.0;
        let mut total_weight = 0.0;

        for field_score in field_scores {
            if field_score.weight > 0.0 {
                total_weighted_score += field_score.weighted_score();
                total_weight += field_score.weight;
            }
        }

        if total_weight > 0.0 {
            (total_weighted_score / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Min-max scaling scoring strategy
pub struct MinMaxScaling;

impl MinMaxScaling {
    /// Combine field scores using min-max normalization then weighted average
    ///
    /// First normalizes all scores to [0, 1] using min-max scaling,
    /// then applies weighted average.
    ///
    /// # Arguments
    ///
    /// * `field_scores` - Individual field scores with weights
    ///
    /// # Returns
    ///
    /// Overall similarity score in range [0.0, 1.0]
    pub fn combine(field_scores: &[FieldScore]) -> f64 {
        if field_scores.is_empty() {
            return 0.0;
        }

        // Find min and max scores directly from iterator (no intermediate allocation)
        let mut min_score = f64::INFINITY;
        let mut max_score = f64::NEG_INFINITY;
        for fs in field_scores {
            min_score = min_score.min(fs.score);
            max_score = max_score.max(fs.score);
        }

        // Handle edge case: all scores the same
        if (max_score - min_score).abs() < 1e-10 {
            return WeightedAverage::combine(field_scores);
        }

        // Normalize scores to [0, 1]
        let normalized_scores: Vec<FieldScore> = field_scores
            .iter()
            .map(|fs| {
                let normalized = (fs.score - min_score) / (max_score - min_score);
                FieldScore::new(&fs.field_name, normalized, fs.weight)
            })
            .collect();

        // Apply weighted average to normalized scores
        WeightedAverage::combine(&normalized_scores)
    }
}

impl ScoringStrategy {
    /// Combine field scores using this strategy
    pub fn combine(&self, field_scores: &[FieldScore]) -> f64 {
        match self {
            ScoringStrategy::WeightedAverage => WeightedAverage::combine(field_scores),
            ScoringStrategy::MinMaxScaling => MinMaxScaling::combine(field_scores),
        }
    }

    /// Get the strategy name
    pub fn name(&self) -> &'static str {
        match self {
            ScoringStrategy::WeightedAverage => "weighted_average",
            ScoringStrategy::MinMaxScaling => "minmax_scaling",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_score_clamping() {
        let score = FieldScore::new("test", 1.5, 15.0);
        assert_eq!(score.score, 1.0); // Clamped to 1.0
        assert_eq!(score.weight, 10.0); // Clamped to 10.0

        let score = FieldScore::new("test", -0.5, -5.0);
        assert_eq!(score.score, 0.0); // Clamped to 0.0
        assert_eq!(score.weight, 0.0); // Clamped to 0.0
    }

    #[test]
    fn test_weighted_score() {
        let score = FieldScore::new("test", 0.8, 5.0);
        assert_eq!(score.weighted_score(), 4.0); // 0.8 * 5.0
    }

    #[test]
    fn test_weighted_average_basic() {
        let scores = vec![
            FieldScore::new("field1", 0.8, 10.0),
            FieldScore::new("field2", 0.6, 5.0),
        ];

        // (0.8 * 10 + 0.6 * 5) / (10 + 5) = (8 + 3) / 15 = 11 / 15 ≈ 0.733
        let combined = WeightedAverage::combine(&scores);
        assert!((combined - 0.7333).abs() < 0.001);
    }

    #[test]
    fn test_weighted_average_equal_weights() {
        let scores = vec![
            FieldScore::new("field1", 0.8, 5.0),
            FieldScore::new("field2", 0.6, 5.0),
            FieldScore::new("field3", 0.4, 5.0),
        ];

        // (0.8 + 0.6 + 0.4) / 3 = 0.6
        let combined = WeightedAverage::combine(&scores);
        assert!((combined - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_weighted_average_with_zero_weight() {
        let scores = vec![
            FieldScore::new("field1", 0.8, 10.0),
            FieldScore::new("field2", 0.2, 0.0), // Zero weight = ignored
        ];

        // Only field1 counts: 0.8 * 10 / 10 = 0.8
        let combined = WeightedAverage::combine(&scores);
        assert!((combined - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_weighted_average_empty() {
        let scores: Vec<FieldScore> = vec![];
        let combined = WeightedAverage::combine(&scores);
        assert_eq!(combined, 0.0);
    }

    #[test]
    fn test_minmax_scaling_basic() {
        let scores = vec![
            FieldScore::new("field1", 0.4, 5.0),  // min
            FieldScore::new("field2", 0.9, 5.0),  // max
            FieldScore::new("field3", 0.65, 5.0), // middle
        ];

        // Normalized: (0.4-0.4)/(0.9-0.4)=0, (0.9-0.4)/(0.9-0.4)=1, (0.65-0.4)/(0.9-0.4)=0.5
        // Average: (0 + 1 + 0.5) / 3 = 0.5
        let combined = MinMaxScaling::combine(&scores);
        assert!((combined - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_minmax_scaling_all_same() {
        let scores = vec![
            FieldScore::new("field1", 0.7, 5.0),
            FieldScore::new("field2", 0.7, 5.0),
            FieldScore::new("field3", 0.7, 5.0),
        ];

        // All scores the same -> falls back to weighted average
        let combined = MinMaxScaling::combine(&scores);
        assert!((combined - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_strategy_enum_combine() {
        let scores = vec![
            FieldScore::new("field1", 0.8, 10.0),
            FieldScore::new("field2", 0.6, 5.0),
        ];

        let result1 = ScoringStrategy::WeightedAverage.combine(&scores);
        let result2 = WeightedAverage::combine(&scores);
        assert!((result1 - result2).abs() < 0.001);

        let result3 = ScoringStrategy::MinMaxScaling.combine(&scores);
        let result4 = MinMaxScaling::combine(&scores);
        assert!((result3 - result4).abs() < 0.001);
    }

    #[test]
    fn test_strategy_names() {
        assert_eq!(ScoringStrategy::WeightedAverage.name(), "weighted_average");
        assert_eq!(ScoringStrategy::MinMaxScaling.name(), "minmax_scaling");
    }

    #[test]
    fn test_weighted_average_with_different_weights() {
        let scores = vec![
            FieldScore::new("name", 0.9, 10.0), // High score, high weight
            FieldScore::new("tags", 0.3, 3.0),  // Low score, low weight
        ];

        // (0.9 * 10 + 0.3 * 3) / (10 + 3) = (9 + 0.9) / 13 = 9.9 / 13 ≈ 0.762
        let combined = WeightedAverage::combine(&scores);
        assert!((combined - 0.762).abs() < 0.001);
    }
}
