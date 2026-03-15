"""Binary classification metric computation utilities."""

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_classification_metrics(
    labels: Iterable[int], probs: Iterable[float], threshold: float = 0.5
) -> Dict[str, float]:

    labels = np.asarray(list(labels))
    probs = np.asarray(list(probs), dtype=float)

    if not np.isfinite(probs).all():
        probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    preds = (probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy_score(labels, preds)),
        "mcc": (
            float(matthews_corrcoef(labels, preds))
            if len(np.unique(labels)) > 1
            else 0.0
        ),
    }
    if len(np.unique(labels)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        except ValueError:

            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(labels, probs))
        except ValueError:
            metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics
