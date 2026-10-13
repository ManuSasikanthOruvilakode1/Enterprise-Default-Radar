"""Subgroup performance and disparity reporting (e.g. by state or industry).

Disparity summaries only use groups that meet minimum size and positive-count thresholds;
smaller groups are still listed but their rates are too noisy to compare.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class FairnessAuditor:
    """Reports per-group error rates and disparity metrics for binary decisions."""

    @staticmethod
    def group_table(
        df: pd.DataFrame,
        sensitive_col: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None,
        min_group_size: int = 1,
        min_positives: int = 1,
    ) -> pd.DataFrame:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        groups = df[sensitive_col].astype(str).to_numpy()
        rows = []
        for group in sorted(np.unique(groups)):
            mask = groups == group
            g_true, g_pred = y_true[mask], y_pred[mask]
            positives = int(g_true.sum())
            negatives = int(mask.sum()) - positives
            flagged = int(g_pred.sum())
            row = {
                "attribute": sensitive_col,
                "group": group,
                "n": int(mask.sum()),
                "positives": positives,
                "prevalence": positives / mask.sum(),
                "selection_rate": flagged / mask.sum(),
                "true_positive_rate": float(np.mean(g_pred[g_true == 1])) if positives else np.nan,
                "false_positive_rate": float(np.mean(g_pred[g_true == 0])) if negatives else np.nan,
                "precision": float(np.mean(g_true[g_pred == 1])) if flagged else np.nan,
                "roc_auc": np.nan,
                "sufficient_sample": bool(mask.sum() >= min_group_size and positives >= min_positives),
            }
            if y_probs is not None and positives and negatives:
                row["roc_auc"] = float(roc_auc_score(g_true, np.asarray(y_probs)[mask]))
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def audit_subgroups(
        df: pd.DataFrame,
        sensitive_col: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None,
        min_group_size: int = 1,
        min_positives: int = 1,
    ) -> Dict[str, Any]:
        """Group table plus disparities computed over sufficiently large groups.

        Equalized odds difference is the larger of the TPR range and the FPR range.
        """
        table = FairnessAuditor.group_table(df, sensitive_col, y_true, y_pred, y_probs, min_group_size, min_positives)
        eligible = table[table["sufficient_sample"]]

        def value_range(column: str) -> float:
            values = eligible[column].dropna()
            return float(values.max() - values.min()) if len(values) >= 2 else float("nan")

        tpr_range = value_range("true_positive_rate")
        fpr_range = value_range("false_positive_rate")
        return {
            "sensitive_attribute": sensitive_col,
            "groups_compared": eligible["group"].tolist(),
            "groups_excluded_small": table.loc[~table["sufficient_sample"], "group"].tolist(),
            "demographic_parity_difference": value_range("selection_rate"),
            "true_positive_rate_difference": tpr_range,
            "false_positive_rate_difference": fpr_range,
            "equalized_odds_difference": float(np.nanmax([tpr_range, fpr_range])) if not (np.isnan(tpr_range) and np.isnan(fpr_range)) else float("nan"),
            "group_table": table,
        }
