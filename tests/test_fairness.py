"""Tests for subgroup evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.fairness import FairnessAuditor


def make_groups():
    df = pd.DataFrame({"state": ["A"] * 6 + ["B"] * 6 + ["C"] * 2})
    y_true = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1])
    return df, y_true, y_pred


def test_equalized_odds_uses_tpr_and_fpr():
    df, y_true, y_pred = make_groups()
    audit = FairnessAuditor.audit_subgroups(df, "state", y_true, y_pred, min_group_size=5, min_positives=1)
    # A: TPR 1.0, FPR 0.25; B: TPR 0.5, FPR 0.0
    assert audit["true_positive_rate_difference"] == 0.5
    assert audit["false_positive_rate_difference"] == 0.25
    assert audit["equalized_odds_difference"] == 0.5


def test_small_groups_excluded_from_disparities():
    df, y_true, y_pred = make_groups()
    audit = FairnessAuditor.audit_subgroups(df, "state", y_true, y_pred, min_group_size=5, min_positives=1)
    assert audit["groups_excluded_small"] == ["C"]
    assert audit["demographic_parity_difference"] == pytest.approx(0.5 - 1 / 6)


def test_group_without_positives_has_undefined_tpr():
    df, y_true, y_pred = make_groups()
    table = FairnessAuditor.group_table(df, "state", y_true, y_pred)
    row = table.set_index("group").loc["C"]
    assert np.isnan(row["true_positive_rate"])
    assert row["n"] == 2 and row["positives"] == 0
