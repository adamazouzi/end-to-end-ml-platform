"""Unit tests for evaluation metric calculations."""

from churn_platform.models.training import _evaluate


def test_evaluation_metrics_include_class_aware_scores_and_confusion_matrix():
    result = _evaluate([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.7, 0.9])
    assert result["precision"] == 2 / 3
    assert result["recall"] == 1.0
    assert result["f1"] == 0.8
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert 0 <= result["roc_auc"] <= 1
