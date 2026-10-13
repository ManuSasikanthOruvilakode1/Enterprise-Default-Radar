"""Tests for metric computation and bootstrap intervals."""

import numpy as np
import pytest

from src.evaluate import TabularEvaluator, classification_metrics


def test_perfect_ranking_metrics():
    metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["recall"] == 1.0 and metrics["precision"] == 1.0
    assert metrics["brier_score"] == pytest.approx((0.01 + 0.04 + 0.04 + 0.01) / 4)


def test_threshold_changes_only_threshold_metrics():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.3, 0.9])
    low, high = classification_metrics(y, p, 0.2), classification_metrics(y, p, 0.95)
    assert low["roc_auc"] == high["roc_auc"]
    assert low["recall"] == 1.0 and high["recall"] == 0.0


def test_bootstrap_ci_contains_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 300), 0, 1)
    evaluator = TabularEvaluator(bootstrap_rounds=200, seed=3)
    cis = evaluator.compute_bootstrap_cis(y, p)
    point = classification_metrics(y, p)
    for name, (low, high) in cis.items():
        assert low <= point[name] <= high
    assert cis == TabularEvaluator(bootstrap_rounds=200, seed=3).compute_bootstrap_cis(y, p)


def test_bootstrap_with_single_class_raises():
    with pytest.raises(ValueError):
        TabularEvaluator(bootstrap_rounds=100).compute_bootstrap_cis(np.zeros(10), np.linspace(0, 1, 10))
