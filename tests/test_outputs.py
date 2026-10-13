"""Tests for report formatting, plot outputs and the model factory."""

import math

import numpy as np
import pandas as pd
import pytest

from src.calibrate import plot_reliability_diagram
from src.config import ModelConfig
from src.reporting import fmt, fmt_ci, markdown_table, write_markdown
from src.schema import INPUT_COLUMNS, TARGET_COLUMN
from src.threshold import CostSensitiveThresholdFinder, plot_cost_curves
from src.train import build_estimator, build_model_pipeline, is_available


def test_fmt_handles_missing_and_booleans():
    assert fmt(float("nan")) == "n/a"
    assert fmt(None) == "n/a"
    assert fmt(True) == "yes"
    assert fmt(0.123456, 3) == "0.123"
    assert fmt_ci(0.5, (0.4, 0.6), 2) == "0.50 [0.40, 0.60]"


def test_markdown_table_layout(tmp_path):
    table = markdown_table(pd.DataFrame({"model": ["a", "b"], "score": [0.5, math.nan]}))
    lines = table.splitlines()
    assert lines[0] == "| model | score |"
    assert lines[3] == "| b | n/a |"
    path = tmp_path / "sub" / "report.md"
    write_markdown(str(path), ["# Title", table])
    assert path.read_text(encoding="utf-8").startswith("# Title")


def test_plots_are_written(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    probs = np.clip(y * 0.4 + rng.uniform(0, 0.6, 400), 0, 1)
    curve = CostSensitiveThresholdFinder().cost_curve(y, probs)
    plot_cost_curves({"dev": curve}, 0.2, str(tmp_path / "cost.png"), bayes_threshold=1 / 6)
    plot_reliability_diagram({"model": (y, probs)}, str(tmp_path / "calibration.png"))
    assert (tmp_path / "cost.png").stat().st_size > 0
    assert (tmp_path / "calibration.png").stat().st_size > 0


@pytest.mark.parametrize("model_type,params", [
    ("xgboost", {"n_estimators": 10, "n_jobs": 1}),
    ("lightgbm", {"n_estimators": 10, "verbose": -1, "n_jobs": 1}),
    ("catboost", {"iterations": 10, "verbose": 0, "allow_writing_files": False}),
    ("logistic_regression", {"max_iter": 500}),
])
def test_every_model_type_fits_in_pipeline(business_df, model_type, params):
    if not is_available(model_type):
        pytest.skip(f"{model_type} not installed")
    config = ModelConfig(name=model_type, model_type=model_type, role="candidate", params=params)
    pipeline = build_model_pipeline(config).fit(business_df[INPUT_COLUMNS], business_df[TARGET_COLUMN])
    probs = pipeline.predict_proba(business_df[INPUT_COLUMNS].head(5))[:, 1]
    assert probs.shape == (5,) and np.all((probs >= 0) & (probs <= 1))


def test_missing_optional_package_gives_install_hint(monkeypatch):
    from src import train
    monkeypatch.setattr(train, "is_available", lambda model_type: False)
    with pytest.raises(ImportError, match="uv sync --extra advanced"):
        build_estimator("xgboost", {})


def test_shap_summary_plot_written(trained_risk_model, business_df, tmp_path):
    from src import interpret
    if not interpret.shap_available():
        pytest.skip("SHAP not installed")
    explainer = interpret.ModelExplainer(trained_risk_model.model_pipeline, business_df[INPUT_COLUMNS], background_size=50)
    explainer.plot_summary(business_df[INPUT_COLUMNS].head(100), str(tmp_path / "shap.png"))
    assert (tmp_path / "shap.png").stat().st_size > 0
