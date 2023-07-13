import pytest
from src.utils.eval_utils import evaluate_fbeta


def test_evaluate_fbeta():
    # Test case 1: Perfect prediction (precision=1.0, recall=1.0)
    threshold = 0.5
    normal_scores = [0.2, 0.3, 0.4]
    anomaly_scores = [0.7, 0.8, 0.9]
    expected_fbeta = 1.0

    assert evaluate_fbeta(threshold, normal_scores, anomaly_scores) == pytest.approx(expected_fbeta)

    # Test case 2: Random scores
    threshold = 0.5
    normal_scores = [0.2, 0.6, 0.4, 0.8]
    anomaly_scores = [0.7, 0.3, 0.9, 0.5]
    expected_fbeta = 0.6666666666666666

    assert evaluate_fbeta(threshold, normal_scores, anomaly_scores) == pytest.approx(expected_fbeta, abs=1e-6)

    # Test case 3: No anomalies detected (precision=0.0, recall=0.0)
    threshold = 0.5
    normal_scores = [0.2, 0.3, 0.4]
    anomaly_scores = [0.1, 0.1, 0.1]
    expected_fbeta = 0.0

    assert evaluate_fbeta(threshold, normal_scores, anomaly_scores) == pytest.approx(expected_fbeta)

