"""Classification metrics and bootstrap confidence intervals."""

from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Ranking, threshold-dependent and probability-quality metrics for binary predictions."""
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    y_pred = (y_probs >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(y_true, y_probs)),
        "roc_auc": float(roc_auc_score(y_true, y_probs)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_probs)),
    }


class TabularEvaluator:
    """Computes held-out metrics with percentile bootstrap confidence intervals."""

    def __init__(self, bootstrap_rounds: int = 1000, ci_percentile: float = 95.0, seed: int = 42):
        self.bootstrap_rounds = bootstrap_rounds
        self.ci_percentile = ci_percentile
        self.seed = seed

    def compute_metrics(self, y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        return classification_metrics(y_true, y_probs, threshold)

    def bootstrap_cis(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        statistics: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
    ) -> Dict[str, Tuple[float, float]]:
        """Percentile bootstrap CIs for every value in the dict returned by `statistics`.

        Resamples lacking one of the classes are skipped because ranking metrics are undefined there.
        """
        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs)
        rng = np.random.default_rng(self.seed)
        n = len(y_true)
        samples = []
        for _ in range(self.bootstrap_rounds):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            samples.append(statistics(y_true[idx], y_probs[idx]))
        if not samples:
            raise ValueError("Bootstrap failed: no resample contained both classes")
        alpha = (100.0 - self.ci_percentile) / 2.0
        return {
            name: (
                float(np.percentile([s[name] for s in samples], alpha)),
                float(np.percentile([s[name] for s in samples], 100.0 - alpha)),
            )
            for name in samples[0]
        }

    def compute_bootstrap_cis(self, y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Tuple[float, float]]:
        """Bootstrap CIs for every metric returned by `compute_metrics`."""
        return self.bootstrap_cis(y_true, y_probs, lambda yt, yp: classification_metrics(yt, yp, threshold))
