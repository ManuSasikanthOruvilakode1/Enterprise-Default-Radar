"""Tests for DataDriftMonitor."""

import numpy as np
import pandas as pd
import pytest

from src.drift import DataDriftMonitor


def test_psi_identical_distributions():
    rng = np.random.default_rng(42)
    assert DataDriftMonitor.calculate_psi(rng.normal(10, 2, 500), rng.normal(10, 2, 500)) < 0.10


def test_psi_drifted_distribution():
    rng = np.random.default_rng(42)
    assert DataDriftMonitor.calculate_psi(rng.normal(10, 2, 500), rng.normal(18, 2, 500)) > 0.25


def test_psi_counts_values_beyond_reference_range():
    ref = np.linspace(0, 1, 1000)
    assert DataDriftMonitor.calculate_psi(ref, np.full(500, 5.0)) > 0.25


def test_psi_constant_reference_does_not_crash():
    assert DataDriftMonitor.calculate_psi(np.ones(100), np.ones(100)) == pytest.approx(0.0)


def test_psi_empty_input_raises():
    with pytest.raises(ValueError):
        DataDriftMonitor.calculate_psi(np.array([]), np.array([1.0]))


def test_categorical_psi_detects_mix_shift():
    ref = pd.Series(["A"] * 80 + ["B"] * 20)
    assert DataDriftMonitor.calculate_categorical_psi(ref, ref) == pytest.approx(0.0)
    assert DataDriftMonitor.calculate_categorical_psi(ref, pd.Series(["A"] * 20 + ["B"] * 80)) > 0.25


@pytest.fixture
def reference_frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(5, 1, 500), "cat": rng.choice(["A", "B"], 500)})


def test_monitor_stable_batch(reference_frame):
    report = DataDriftMonitor().monitor_features(reference_frame.iloc[:250], reference_frame.iloc[250:], ["x", "y"], ["cat"])
    assert not report["drift_detected"]
    assert all(m["status"] == "STABLE" for m in report["feature_metrics"].values())


def test_monitor_shifted_batch(reference_frame):
    shifted = reference_frame.assign(x=reference_frame["x"] + 3)
    report = DataDriftMonitor().monitor_features(reference_frame, shifted, ["x", "y"])
    assert report["drift_detected"]
    assert report["feature_metrics"]["x"]["status"] == "CRITICAL_DRIFT"
    assert report["feature_metrics"]["x"]["ks_significant"]
    assert report["feature_metrics"]["y"]["status"] == "STABLE"


def test_monitor_missing_feature(reference_frame):
    report = DataDriftMonitor().monitor_features(reference_frame, reference_frame.drop(columns="y"), ["x", "y"])
    assert report["feature_metrics"]["y"]["status"] == "MISSING_FEATURE"
    assert report["drift_detected"]


def test_monitor_malformed_and_missing_values(reference_frame):
    batch = reference_frame.astype({"x": object})
    batch.loc[:49, "x"] = "not a number"
    batch.loc[50:99, "x"] = np.nan
    metrics = DataDriftMonitor().monitor_features(reference_frame, batch, ["x"])["feature_metrics"]["x"]
    assert metrics["malformed_rate"] == pytest.approx(0.1)
    assert metrics["missing_rate"] == pytest.approx(0.1)
    assert "psi" in metrics


def test_monitor_insufficient_data(reference_frame):
    report = DataDriftMonitor().monitor_features(reference_frame, reference_frame.head(5), ["x"])
    assert report["feature_metrics"]["x"]["status"] == "INSUFFICIENT_DATA"
