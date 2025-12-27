"""Tests for evaluation metrics (precision, recall, f_score, confusion_matrix)."""

import pytest

import fuzzyrust as fr


class TestPrecision:
    """Tests for precision metric."""

    def test_perfect_precision(self):
        """All predictions are correct."""
        true_matches = [(0, 1), (1, 2)]
        predicted = [(0, 1), (1, 2)]
        assert fr.precision(true_matches, predicted) == 1.0

    def test_partial_precision(self):
        """Half the predictions are correct."""
        true_matches = [(0, 1)]
        predicted = [(0, 1), (2, 3)]
        assert fr.precision(true_matches, predicted) == 0.5

    def test_zero_precision(self):
        """No predictions are correct."""
        true_matches = [(0, 1)]
        predicted = [(2, 3)]
        assert fr.precision(true_matches, predicted) == 0.0

    def test_empty_predictions(self):
        """No predictions made."""
        true_matches = [(0, 1)]
        predicted = []
        # No predictions but there are true matches = 0 precision
        assert fr.precision(true_matches, predicted) == 0.0

    def test_empty_both(self):
        """Both sets empty."""
        true_matches = []
        predicted = []
        # No predictions and no true matches = perfect precision
        assert fr.precision(true_matches, predicted) == 1.0

    def test_pair_order_normalized(self):
        """Pairs should be order-independent: (0,1) == (1,0)."""
        true_matches = [(0, 1)]
        predicted = [(1, 0)]  # Same pair, different order
        assert fr.precision(true_matches, predicted) == 1.0


class TestRecall:
    """Tests for recall metric."""

    def test_perfect_recall(self):
        """All true matches are found."""
        true_matches = [(0, 1), (1, 2)]
        predicted = [(0, 1), (1, 2)]
        assert fr.recall(true_matches, predicted) == 1.0

    def test_partial_recall(self):
        """Half the true matches are found."""
        true_matches = [(0, 1), (1, 2)]
        predicted = [(0, 1)]
        assert fr.recall(true_matches, predicted) == 0.5

    def test_zero_recall(self):
        """No true matches are found."""
        true_matches = [(0, 1)]
        predicted = [(2, 3)]
        assert fr.recall(true_matches, predicted) == 0.0

    def test_empty_true_matches(self):
        """No true matches to find."""
        true_matches = []
        predicted = [(0, 1)]
        # No true matches to find = 0 recall (can't find what doesn't exist)
        assert fr.recall(true_matches, predicted) == 0.0

    def test_empty_both(self):
        """Both sets empty."""
        true_matches = []
        predicted = []
        # No true matches and no predictions = perfect recall
        assert fr.recall(true_matches, predicted) == 1.0

    def test_overprediction(self):
        """More predictions than true matches."""
        true_matches = [(0, 1)]
        predicted = [(0, 1), (2, 3), (4, 5)]
        # All true matches found = perfect recall
        assert fr.recall(true_matches, predicted) == 1.0


class TestFScore:
    """Tests for F-beta score metric."""

    def test_perfect_f1(self):
        """Perfect precision and recall."""
        true_matches = [(0, 1)]
        predicted = [(0, 1)]
        assert fr.f_score(true_matches, predicted) == 1.0

    def test_zero_f1(self):
        """No correct predictions."""
        true_matches = [(0, 1)]
        predicted = [(2, 3)]
        assert fr.f_score(true_matches, predicted) == 0.0

    def test_balanced_f1(self):
        """Balanced precision and recall."""
        # precision = 1/2, recall = 1/2 -> F1 = 0.5
        true_matches = [(0, 1), (1, 2)]
        predicted = [(0, 1), (3, 4)]
        f1 = fr.f_score(true_matches, predicted)
        assert f1 == pytest.approx(0.5, abs=0.001)

    def test_f05_weights_precision(self):
        """F0.5 weighs precision more than recall."""
        # High precision, low recall
        true_matches = [(0, 1), (1, 2), (2, 3), (3, 4)]
        predicted = [(0, 1)]  # precision=1.0, recall=0.25

        f1 = fr.f_score(true_matches, predicted, beta=1.0)
        f05 = fr.f_score(true_matches, predicted, beta=0.5)

        # F0.5 should be higher because precision is perfect
        assert f05 > f1

    def test_f2_weights_recall(self):
        """F2 weighs recall more than precision."""
        # Low precision, high recall
        true_matches = [(0, 1)]
        predicted = [(0, 1), (1, 2), (2, 3), (3, 4)]  # precision=0.25, recall=1.0

        f1 = fr.f_score(true_matches, predicted, beta=1.0)
        f2 = fr.f_score(true_matches, predicted, beta=2.0)

        # F2 should be higher because recall is perfect
        assert f2 > f1

    def test_beta_zero(self):
        """Beta=0 should work (pure precision)."""
        true_matches = [(0, 1)]
        predicted = [(0, 1), (2, 3)]
        # With beta=0, only precision matters
        f0 = fr.f_score(true_matches, predicted, beta=0.0)
        assert f0 == pytest.approx(0.5, abs=0.001)  # precision = 0.5

    def test_negative_beta_raises(self):
        """Negative beta should raise ValueError."""
        with pytest.raises(ValueError):
            fr.f_score([(0, 1)], [(0, 1)], beta=-1.0)


class TestConfusionMatrix:
    """Tests for confusion matrix calculation."""

    def test_basic_confusion_matrix(self):
        """Basic confusion matrix calculation."""
        true_matches = [(0, 1), (1, 2)]
        predicted = [(0, 1), (2, 3)]
        cm = fr.confusion_matrix(true_matches, predicted, total_pairs=10)

        assert cm.tp == 1  # (0,1) is correct
        assert cm.fp == 1  # (2,3) is wrong
        assert cm.fn_count == 1  # (1,2) was missed
        assert cm.tn == 7  # remaining pairs

    def test_perfect_classification(self):
        """All predictions correct."""
        true_matches = [(0, 1)]
        predicted = [(0, 1)]
        cm = fr.confusion_matrix(true_matches, predicted, total_pairs=10)

        assert cm.tp == 1
        assert cm.fp == 0
        assert cm.fn_count == 0
        assert cm.tn == 9

    def test_confusion_matrix_precision_method(self):
        """ConfusionMatrixResult.precision() method."""
        true_matches = [(0, 1)]
        predicted = [(0, 1), (2, 3)]
        cm = fr.confusion_matrix(true_matches, predicted, total_pairs=10)

        assert cm.precision() == 0.5  # 1 TP, 1 FP

    def test_confusion_matrix_recall_method(self):
        """ConfusionMatrixResult.recall() method."""
        true_matches = [(0, 1), (1, 2)]
        predicted = [(0, 1)]
        cm = fr.confusion_matrix(true_matches, predicted, total_pairs=10)

        assert cm.recall() == 0.5  # 1 TP, 1 FN

    def test_confusion_matrix_f_score_method(self):
        """ConfusionMatrixResult.f_score() method."""
        true_matches = [(0, 1)]
        predicted = [(0, 1)]
        cm = fr.confusion_matrix(true_matches, predicted, total_pairs=10)

        assert cm.f_score() == 1.0
        assert cm.f_score(beta=0.5) == 1.0
        assert cm.f_score(beta=2.0) == 1.0

    def test_empty_confusion_matrix(self):
        """Empty sets."""
        cm = fr.confusion_matrix([], [], total_pairs=10)

        assert cm.tp == 0
        assert cm.fp == 0
        assert cm.fn_count == 0
        assert cm.tn == 10

    def test_confusion_matrix_repr(self):
        """String representation."""
        cm = fr.confusion_matrix([(0, 1)], [(0, 1), (2, 3)], total_pairs=10)
        repr_str = repr(cm)
        assert "ConfusionMatrixResult" in repr_str
        assert "tp=1" in repr_str
        assert "fp=1" in repr_str


class TestMetricsIntegration:
    """Integration tests using metrics with deduplication results."""

    def test_evaluate_dedup_results(self):
        """Evaluate deduplication quality using metrics."""
        # Simulate ground truth and predictions
        # Items: ["hello", "hallo", "world", "werld"]
        # True duplicates: hello-hallo (0,1), world-werld (2,3)
        true_matches = [(0, 1), (2, 3)]

        # Simulated predictions: found hello-hallo, missed world-werld
        predicted = [(0, 1)]

        precision = fr.precision(true_matches, predicted)
        recall = fr.recall(true_matches, predicted)
        f1 = fr.f_score(true_matches, predicted)

        assert precision == 1.0  # All predictions correct
        assert recall == 0.5  # Missed half
        # F1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 0.667
        assert f1 == pytest.approx(0.667, abs=0.01)

    def test_symmetric_pairs(self):
        """Pairs should be symmetric - (a,b) == (b,a)."""
        true_matches = [(0, 1), (2, 3)]
        predicted = [(1, 0), (3, 2)]  # Same pairs, reversed order

        assert fr.precision(true_matches, predicted) == 1.0
        assert fr.recall(true_matches, predicted) == 1.0
        assert fr.f_score(true_matches, predicted) == 1.0
