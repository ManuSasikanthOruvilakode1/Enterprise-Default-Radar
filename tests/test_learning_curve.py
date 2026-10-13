"""Tests for the learning-curve experiment."""

import pytest

from src.config import ModelConfig
from src.learning_curve import plot_learning_curve, run_learning_curve, summarise


@pytest.fixture(scope="module")
def curve():
    configs = [
        ModelConfig(name="linear", model_type="logistic_regression", role="reference", params={"max_iter": 1000}),
        ModelConfig(name="hgb", model_type="hist_gradient_boosting", role="candidate", params={"max_iter": 20}),
    ]
    return run_learning_curve(configs, [400, 800], test_rows=3000, repeats=2, test_seed=5, train_seed_start=100)


def test_one_row_per_size_repeat_and_model(curve):
    results, _ = curve
    assert len(results) == 2 * 2 * 2
    assert set(results["model"]) == {"linear", "hgb"}
    assert results["average_precision"].between(0, 1).all()


def test_models_share_training_samples(curve):
    results, _ = curve
    seeds = results.groupby(["train_size", "repeat"])["train_seed"].nunique()
    assert (seeds == 1).all()
    assert results["train_seed"].nunique() == 4


def test_ceiling_reported(curve):
    _, ceiling = curve
    assert 0 < ceiling["average_precision"] < 1 and ceiling["test_rows"] == 3000


def test_summary_and_plot(curve, tmp_path):
    results, ceiling = curve
    stats = summarise(results)
    assert set(stats.columns) == {"train_size", "model", "mean", "std"}
    plot_learning_curve(results, ceiling, str(tmp_path / "curve.png"), highlight_size=500)
    assert (tmp_path / "curve.png").stat().st_size > 0


def test_test_seed_must_not_reuse_a_training_seed():
    config = ModelConfig(name="linear", model_type="logistic_regression", role="reference")
    with pytest.raises(ValueError, match="test seed"):
        run_learning_curve([config], [400], test_rows=1000, repeats=1, test_seed=100, train_seed_start=100)
