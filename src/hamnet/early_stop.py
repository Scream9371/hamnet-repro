"""Early stopping helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopper:


    patience: int
    mode: str = "max"
    min_delta: float = 0.0
    best_score: float | None = None
    bad_epochs: int = 0

    def step(self, score: float | None) -> bool:

        if self.patience <= 0:
            return False
        if score is None:
            self.bad_epochs = 0
            return False
        if self.best_score is None:
            self.best_score = float(score)
            self.bad_epochs = 0
            return False

        improved = self._is_improved(float(score), self.best_score)
        if improved:
            self.best_score = float(score)
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def _is_improved(self, score: float, best: float) -> bool:
        if self.mode == "min":
            return score < (best - self.min_delta)
        return score > (best + self.min_delta)


def pick_monitor_key(metrics: dict | None) -> str:

    if metrics and "pr_auc" in metrics and metrics.get("pr_auc") is not None:
        return "pr_auc"
    return "roc_auc"
