"""Probability calibration with Platt (sigmoid) scaling or isotonic regression.

Calibrators are fitted on out-of-fold development predictions only; the test set is used
solely to report how well the chosen calibration generalises.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

EPSILON = 1e-6


def _logit(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probs, dtype=float), EPSILON, 1 - EPSILON)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


class ProbabilityCalibrator(BaseEstimator):
    """Maps raw model probabilities to calibrated probabilities (method: none, sigmoid or isotonic)."""

    def __init__(self, method: str = "sigmoid"):
        self.method = method

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        if self.method == "sigmoid":
            self.mapper_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_probs), y_true)
        elif self.method == "isotonic":
            self.mapper_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw_probs, y_true)
        elif self.method == "none":
            self.mapper_ = None
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        raw_probs = np.asarray(raw_probs, dtype=float)
        if self.method == "sigmoid":
            return self.mapper_.predict_proba(_logit(raw_probs))[:, 1]
        if self.method == "isotonic":
            return self.mapper_.predict(raw_probs)
        return raw_probs

    @staticmethod
    def evaluate_calibration(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
        y_probs = np.clip(np.asarray(y_probs, dtype=float), EPSILON, 1 - EPSILON)
        return {
            "brier_score": float(brier_score_loss(y_true, y_probs)),
            "log_loss": float(log_loss(y_true, y_probs)),
            "mean_predicted": float(np.mean(y_probs)),
            "observed_rate": float(np.mean(y_true)),
        }


def cross_validated_calibration(raw_probs: np.ndarray, y_true: np.ndarray, method: str, cv_folds: int = 5, seed: int = 42) -> np.ndarray:
    """Calibrated probabilities where each sample's calibrator never saw that sample's label."""
    raw_probs = np.asarray(raw_probs, dtype=float)
    y_true = np.asarray(y_true)
    calibrated = np.zeros_like(raw_probs)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for fit_idx, apply_idx in splitter.split(raw_probs.reshape(-1, 1), y_true):
        calibrator = ProbabilityCalibrator(method).fit(raw_probs[fit_idx], y_true[fit_idx])
        calibrated[apply_idx] = calibrator.transform(raw_probs[apply_idx])
    return calibrated


def select_calibration_method(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    methods: List[str],
    cv_folds: int = 5,
    seed: int = 42,
) -> Tuple[str, pd.DataFrame, Dict[str, np.ndarray]]:
    """Chooses the method with the lowest cross-validated Brier score on development predictions."""
    rows = []
    calibrated_by_method = {}
    for method in methods:
        calibrated = cross_validated_calibration(raw_probs, y_true, method, cv_folds, seed)
        calibrated_by_method[method] = calibrated
        rows.append({"method": method, **ProbabilityCalibrator.evaluate_calibration(y_true, calibrated)})
    table = pd.DataFrame(rows).sort_values("brier_score").reset_index(drop=True)
    return str(table.loc[0, "method"]), table, calibrated_by_method


def plot_reliability_diagram(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: str, n_bins: int = 10) -> None:
    """Quantile-binned reliability diagram for one or more (y_true, y_probs) pairs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(6, 5.5))
    max_prob = 0.0
    for label, (y_true, y_probs) in curves.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.5, label=label)
        max_prob = max(max_prob, float(np.max(mean_pred)), float(np.max(frac_pos)))
    limit = min(1.0, max_prob * 1.1 + 0.02)
    ax.plot([0, limit], [0, limit], color="grey", linestyle="--", linewidth=1, label="Perfect calibration")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Mean predicted probability (quantile bins)")
    ax.set_ylabel("Observed insolvency rate")
    ax.set_title("Reliability diagram (held-out test set)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
