"""Cost-sensitive decision threshold selection.

Thresholds must be selected on out-of-fold or validation predictions and then applied,
unchanged, to the final test set.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class CostSensitiveThresholdFinder:
    """Finds the decision threshold that minimises total misclassification cost."""

    def __init__(self, false_positive_cost: float = 1.0, false_negative_cost: float = 5.0):
        if false_positive_cost <= 0 or false_negative_cost <= 0:
            raise ValueError("Misclassification costs must be positive")
        self.fp_cost = false_positive_cost
        self.fn_cost = false_negative_cost

    @property
    def bayes_optimal_threshold(self) -> float:
        """Cost-minimising threshold if probabilities were perfectly calibrated: FP / (FP + FN)."""
        return self.fp_cost / (self.fp_cost + self.fn_cost)

    def compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        return float(fp * self.fp_cost + fn * self.fn_cost)

    def evaluate_threshold(self, y_true: np.ndarray, y_probs: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Confusion counts, precision, recall and cost when flagging probabilities >= threshold."""
        y_true = np.asarray(y_true)
        y_pred = (np.asarray(y_probs) >= threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        cost = fp * self.fp_cost + fn * self.fn_cost
        return {
            "threshold": float(threshold),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "flagged": tp + fp,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "total_cost": float(cost),
            "cost_per_business": float(cost / len(y_true)),
        }

    def cost_curve(self, y_true: np.ndarray, y_probs: np.ndarray, thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)
        return pd.DataFrame([self.evaluate_threshold(y_true, y_probs, t) for t in thresholds])

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        n_steps: int = 99,
        grid_min: float = 0.01,
        grid_max: float = 0.99,
    ) -> Dict[str, Any]:
        """Sweeps a threshold grid and returns the lowest threshold achieving the minimum cost."""
        curve = self.cost_curve(y_true, y_probs, np.linspace(grid_min, grid_max, n_steps))
        best = curve.loc[curve["total_cost"].idxmin()]
        default = self.evaluate_threshold(y_true, y_probs, 0.5)
        cost_savings = default["total_cost"] - best["total_cost"]
        return {
            "optimal_threshold": round(float(best["threshold"]), 4),
            "minimized_cost": float(best["total_cost"]),
            "cost_at_default_0_5": float(default["total_cost"]),
            "cost_savings": float(cost_savings),
            "savings_percent": float(cost_savings / default["total_cost"] * 100) if default["total_cost"] else 0.0,
            "confusion_at_optimal": {k: int(best[k]) for k in ("tn", "fp", "fn", "tp")},
            "curve": curve,
        }


def plot_cost_curves(curves: Dict[str, pd.DataFrame], selected_threshold: float, output_path: str, bayes_threshold: Optional[float] = None) -> None:
    """Plots cost per business against threshold for one or more prediction sets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, curve in curves.items():
        ax.plot(curve["threshold"], curve["cost_per_business"], label=label, linewidth=2)
    ax.axvline(selected_threshold, color="black", linestyle="--", linewidth=1, label=f"Selected threshold ({selected_threshold:.2f})")
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=1, label="Default threshold (0.50)")
    if bayes_threshold is not None:
        ax.axvline(bayes_threshold, color="tab:green", linestyle="-.", linewidth=1, label=f"FP/(FP+FN) = {bayes_threshold:.3f}")
    ax.set_xlabel("Decision threshold (flag if probability >= threshold)")
    ax.set_ylabel("Cost per business (illustrative $)")
    ax.set_title("Misclassification cost across thresholds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
