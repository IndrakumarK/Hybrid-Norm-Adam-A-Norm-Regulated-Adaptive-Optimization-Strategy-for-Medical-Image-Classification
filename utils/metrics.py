"""
Binary classification metrics for HNAdam brain tumor evaluation.
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)


def expected_calibration_error(y_true, probabilities, n_bins=15):
    """
    Calculate Expected Calibration Error (ECE).
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)

    if len(y_true) != len(probabilities) or len(y_true) == 0:
        raise ValueError(
            "y_true and probabilities must have the same non-zero length."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must contain finite values.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    predictions = (probabilities >= 0.5).astype(int)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    correctness = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence > lower)
                & (confidence <= upper)
            )

        if np.any(mask):
            bin_accuracy = np.mean(correctness[mask])
            bin_confidence = np.mean(confidence[mask])

            ece += (
                np.mean(mask)
                * abs(bin_accuracy - bin_confidence)
            )

    return float(ece)


def calculate_metrics(y_true, probabilities, threshold=0.5):
    """
    Calculate binary classification performance metrics.

    Class mapping:
        0 = no_tumor
        1 = tumor

    Args:
        y_true: Ground-truth binary labels.
        probabilities: Predicted tumor probabilities.
        threshold: Classification threshold.

    Returns:
        Dictionary containing classification metrics.
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    probabilities = np.asarray(
        probabilities, dtype=float
    ).reshape(-1)

    if len(y_true) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("Ground-truth labels must be binary (0 or 1).")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must contain finite values.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    has_both_classes = len(np.unique(y_true)) == 2

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "sensitivity": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "specificity": float(specificity),

        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),

        "mcc": float(
            matthews_corrcoef(
                y_true,
                predictions,
            )
        ),

        "auroc": (
            float(roc_auc_score(y_true, probabilities))
            if has_both_classes
            else float("nan")
        ),

        "auprc": (
            float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            )
            if has_both_classes
            else float("nan")
        ),

        "ece": float(
            expected_calibration_error(
                y_true,
                probabilities,
            )
        ),

        "brier": float(
            np.mean((probabilities - y_true) ** 2)
        ),

        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }