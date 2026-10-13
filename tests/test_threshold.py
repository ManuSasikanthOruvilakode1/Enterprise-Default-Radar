"""Tests for cost-sensitive threshold selection."""

import numpy as np
import pytest

from src.threshold import CostSensitiveThresholdFinder


def test_threshold_finder():
    finder = CostSensitiveThresholdFinder(false_positive_cost=1.0, false_negative_cost=5.0)
    y_true = np.array([0, 0, 0, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.45, 0.8])
    result = finder.find_optimal_threshold(y_true, y_probs)
    assert result["optimal_threshold"] <= 0.45
    assert result["minimized_cost"] == 0.0
    assert result["cost_at_default_0_5"] == 5.0
    assert result["cost_savings"] == 5.0


def test_evaluate_threshold_counts_and_cost():
    finder = CostSensitiveThresholdFinder(1.0, 5.0)
    result = finder.evaluate_threshold(np.array([0, 0, 1, 1]), np.array([0.6, 0.1, 0.7, 0.2]), 0.5)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["total_cost"] == 6.0
    assert result["precision"] == 0.5 and result["recall"] == 0.5


def test_cost_curve_shape():
    finder = CostSensitiveThresholdFinder(1.0, 5.0)
    curve = finder.cost_curve(np.array([0, 1]), np.array([0.2, 0.8]), np.linspace(0.1, 0.9, 9))
    assert len(curve) == 9
    assert {"threshold", "total_cost", "cost_per_business"} <= set(curve.columns)


def test_bayes_threshold_matches_cost_ratio():
    assert CostSensitiveThresholdFinder(1.0, 5.0).bayes_optimal_threshold == pytest.approx(1 / 6)


def test_selected_threshold_near_cost_ratio_for_calibrated_probabilities():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 20_000)
    y_true = (rng.uniform(0, 1, 20_000) < probs).astype(int)
    result = CostSensitiveThresholdFinder(1.0, 5.0).find_optimal_threshold(y_true, probs)
    assert abs(result["optimal_threshold"] - 1 / 6) < 0.03


def test_non_positive_costs_rejected():
    with pytest.raises(ValueError):
        CostSensitiveThresholdFinder(0.0, 5.0)
