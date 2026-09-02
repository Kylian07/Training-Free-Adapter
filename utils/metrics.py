"""
Evaluation metrics for medical imaging classification:
- Accuracy
- Area Under the ROC Curve (AUC-ROC)
- Macro F1-Score
- Expected Calibration Error (ECE)
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score


def calculate_expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE):
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def calculate_all_metrics(probs: np.ndarray, labels: np.ndarray, task: str = "multi-class") -> dict:
    """
    Calculates comprehensive diagnostic metrics.
    probs: (N, K) probabilities
    labels: (N,) or (N, 1) integer class targets
    """
    if labels.ndim > 1:
        labels = labels.squeeze()

    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    ece = calculate_expected_calibration_error(probs, labels)

    # Compute Multi-class AUC (One-vs-Rest)
    num_classes = probs.shape[1]
    try:
        if num_classes == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.5

    return {
        "accuracy": float(acc),
        "auc": float(auc),
        "f1_macro": float(macro_f1),
        "precision": float(precision),
        "recall": float(recall),
        "ece": float(ece)
    }
