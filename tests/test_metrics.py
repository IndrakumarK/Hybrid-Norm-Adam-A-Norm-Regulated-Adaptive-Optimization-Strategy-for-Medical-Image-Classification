import numpy as np
from utils.metrics import calculate_metrics


def test_metrics_confusion_and_accuracy():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = calculate_metrics(y_true, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [
        [2, 0],
        [0, 2],
    ]


def test_metrics_with_incorrect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.8, 0.2, 0.3, 0.9])

    metrics = calculate_metrics(y_true, y_prob)

    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"] == [
        [1, 1],
        [1, 1],
    ]