"""Tests for probability calibration."""

import numpy as np
import pytest

from src.calibrate import ProbabilityCalibrator, cross_validated_calibration, select_calibration_method


@pytest.fixture
def overconfident_scores():
    rng = np.random.default_rng(0)
    true_probs = rng.uniform(0.0, 0.3, 4000)
    y = (rng.uniform(size=4000) < true_probs).astype(int)
    inflated = np.clip(true_probs * 3, 0, 0.99)  # e.g. class-weighted model output
    return inflated, y


def test_sigmoid_calibration_reduces_brier(overconfident_scores):
    scores, y = overconfident_scores
    before = ProbabilityCalibrator.evaluate_calibration(y, scores)["brier_score"]
    calibrated = cross_validated_calibration(scores, y, "sigmoid")
    after = ProbabilityCalibrator.evaluate_calibration(y, calibrated)["brier_score"]
    assert after < before
    assert abs(calibrated.mean() - y.mean()) < 0.01


def test_calibration_preserves_ranking(overconfident_scores):
    scores, y = overconfident_scores
    calibrated = ProbabilityCalibrator("sigmoid").fit(scores, y).transform(scores)
    assert np.all(np.diff(calibrated[np.argsort(scores)]) >= -1e-12)


def test_none_method_is_identity(overconfident_scores):
    scores, y = overconfident_scores
    assert np.array_equal(ProbabilityCalibrator("none").fit(scores, y).transform(scores), scores)


def test_select_method_prefers_calibration_for_inflated_scores(overconfident_scores):
    scores, y = overconfident_scores
    method, table, calibrated = select_calibration_method(scores, y, ["none", "sigmoid", "isotonic"])
    assert method != "none"
    assert set(table["method"]) == {"none", "sigmoid", "isotonic"}
    assert set(calibrated) == {"none", "sigmoid", "isotonic"}


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        ProbabilityCalibrator("beta").fit(np.array([0.1, 0.9]), np.array([0, 1]))
