"""Data drift monitoring with the Population Stability Index (PSI) and the two-sample KS test.

A drift alert is a signal to investigate the incoming data. It does not by itself show that
model performance has deteriorated.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

MIN_SAMPLES = 20
PROPORTION_FLOOR = 1e-4


def _psi_from_proportions(ref_pct: np.ndarray, curr_pct: np.ndarray) -> float:
    ref_pct = np.clip(ref_pct, PROPORTION_FLOOR, None)
    curr_pct = np.clip(curr_pct, PROPORTION_FLOOR, None)
    return float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))


class DataDriftMonitor:
    """Compares a current batch of features against a reference (training) distribution."""

    def __init__(self, psi_moderate: float = 0.10, psi_critical: float = 0.25, ks_alpha: float = 0.01):
        self.psi_moderate = psi_moderate
        self.psi_critical = psi_critical
        self.ks_alpha = ks_alpha

    @staticmethod
    def calculate_psi(ref: np.ndarray, curr: np.ndarray, num_buckets: int = 10) -> float:
        """PSI for a numeric feature using reference quantile bins with open-ended outer bins."""
        ref = np.asarray(ref, dtype=float)
        curr = np.asarray(curr, dtype=float)
        if len(ref) == 0 or len(curr) == 0:
            raise ValueError("PSI requires non-empty reference and current samples")
        inner_edges = np.unique(np.percentile(ref, np.linspace(0, 100, num_buckets + 1)[1:-1]))
        edges = np.concatenate([[-np.inf], inner_edges, [np.inf]])
        ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
        curr_pct = np.histogram(curr, bins=edges)[0] / len(curr)
        return _psi_from_proportions(ref_pct, curr_pct)

    @staticmethod
    def calculate_categorical_psi(ref: pd.Series, curr: pd.Series) -> float:
        """PSI over category frequencies; categories unseen in the reference get their own bucket."""
        ref_freq = ref.astype(str).value_counts(normalize=True)
        curr_freq = curr.astype(str).value_counts(normalize=True)
        categories = ref_freq.index.union(curr_freq.index)
        return _psi_from_proportions(
            ref_freq.reindex(categories, fill_value=0).to_numpy(),
            curr_freq.reindex(categories, fill_value=0).to_numpy(),
        )

    def _status(self, psi: float) -> str:
        if psi > self.psi_critical:
            return "CRITICAL_DRIFT"
        if psi > self.psi_moderate:
            return "MODERATE_DRIFT"
        return "STABLE"

    def monitor_features(
        self,
        ref_df: pd.DataFrame,
        curr_df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Per-feature PSI, KS test (numeric only), missing and malformed value rates.

        Malformed values are entries that cannot be parsed as numbers in a numeric column.
        """
        report: Dict[str, Dict[str, Any]] = {}
        for col in numeric_cols + (categorical_cols or []):
            is_numeric = col in numeric_cols
            entry: Dict[str, Any] = {"type": "numeric" if is_numeric else "categorical"}
            if col not in curr_df.columns:
                report[col] = {**entry, "status": "MISSING_FEATURE"}
                continue
            if col not in ref_df.columns:
                report[col] = {**entry, "status": "NOT_IN_REFERENCE"}
                continue

            curr_raw = curr_df[col]
            entry["missing_rate"] = float(curr_raw.isna().mean())
            if is_numeric:
                ref_vals = pd.to_numeric(ref_df[col], errors="coerce").dropna().to_numpy()
                curr_parsed = pd.to_numeric(curr_raw, errors="coerce")
                entry["malformed_rate"] = float((curr_parsed.isna() & curr_raw.notna()).mean())
                curr_vals = curr_parsed.dropna().to_numpy()
                if len(curr_vals) < MIN_SAMPLES or len(ref_vals) < MIN_SAMPLES:
                    report[col] = {**entry, "status": "INSUFFICIENT_DATA"}
                    continue
                entry["psi"] = self.calculate_psi(ref_vals, curr_vals)
                ks_stat, ks_pvalue = ks_2samp(ref_vals, curr_vals)
                entry["ks_statistic"] = float(ks_stat)
                entry["ks_pvalue"] = float(ks_pvalue)
                entry["ks_significant"] = bool(ks_pvalue < self.ks_alpha)
                entry["out_of_reference_range_rate"] = float(np.mean((curr_vals < ref_vals.min()) | (curr_vals > ref_vals.max())))
            else:
                curr_vals = curr_raw.dropna()
                if len(curr_vals) < MIN_SAMPLES:
                    report[col] = {**entry, "status": "INSUFFICIENT_DATA"}
                    continue
                entry["psi"] = self.calculate_categorical_psi(ref_df[col].dropna(), curr_vals)
                reference_levels = set(ref_df[col].dropna().astype(str))
                entry["unseen_category_rate"] = float((~curr_vals.astype(str).isin(reference_levels)).mean())
            entry["status"] = self._status(entry["psi"])
            report[col] = entry

        alert_statuses = {"MODERATE_DRIFT", "CRITICAL_DRIFT", "MISSING_FEATURE"}
        return {
            "drift_detected": any(r["status"] in alert_statuses for r in report.values()),
            "feature_metrics": report,
        }
