"""Shared threshold search utilities."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .metrics import compute_classification_metrics


def search_best_threshold(
    labels: Iterable[int], probs: Iterable[float]
) -> Tuple[List[float], List[Dict[str, float]], Dict[str, float]]:

    labels_arr = np.asarray(list(labels), dtype=int)
    probs_arr = np.asarray(list(probs), dtype=float)
    if probs_arr.size == 0 or labels_arr.size == 0:
        empty_metrics = compute_classification_metrics(labels_arr, probs_arr, threshold=0.5)
        empty_entry = {"threshold": 0.5, **empty_metrics}
        return [0.5], [empty_entry], empty_entry


    if not np.isfinite(probs_arr).all():
        probs_arr = np.nan_to_num(probs_arr, nan=0.5, posinf=1.0, neginf=0.0)

    def pick_key(record: Dict[str, float]) -> Tuple[float, float, float]:
        return (record["f1"], record["precision"], record["threshold"])

    thresholds = sorted(
        set(float(np.round(v, 8)) for v in probs_arr.tolist()) | {0.0, 1.0}
    )


    static_auc = compute_classification_metrics(labels_arr, probs_arr, threshold=0.5)
    roc_auc = float(static_auc.get("roc_auc", float("nan")))
    pr_auc = float(static_auc.get("pr_auc", float("nan")))
    pos_total = int(np.sum(labels_arr == 1))
    neg_total = int(labels_arr.size - pos_total)

    records: List[Dict[str, float]] = []
    for th in thresholds:
        preds = probs_arr >= th
        tp = int(np.sum(preds & (labels_arr == 1)))
        fp = int(np.sum(preds & (labels_arr == 0)))
        fn = pos_total - tp
        tn = neg_total - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        mcc_den = math.sqrt(
            max(tp + fp, 0)
            * max(tp + fn, 0)
            * max(tn + fp, 0)
            * max(tn + fn, 0)
        )
        mcc = ((tp * tn - fp * fn) / mcc_den) if mcc_den > 0 else 0.0

        records.append(
            {
                "threshold": float(th),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "mcc": float(mcc),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

    best_entry = max(records, key=pick_key)
    return thresholds, records, best_entry
