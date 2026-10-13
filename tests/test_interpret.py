"""Tests for SHAP explanations and fallbacks."""

import numpy as np
import pytest

from src import interpret
from src.interpret import ExplanationUnavailable, ModelExplainer, grouped_importance, permutation_importance_table
from src.schema import INPUT_COLUMNS, TARGET_COLUMN

requires_shap = pytest.mark.skipif(not interpret.shap_available(), reason="SHAP not installed")


@pytest.fixture
def explainer(trained_risk_model, business_df):
    return ModelExplainer(trained_risk_model.model_pipeline, business_df[INPUT_COLUMNS], background_size=50)


@requires_shap
def test_shap_values_add_up_to_model_log_odds(explainer, trained_risk_model, business_df):
    sample = business_df[INPUT_COLUMNS].head(10)
    values, base, _ = explainer.shap_values(sample)
    raw = trained_risk_model.raw_probability(sample)
    assert np.allclose(values.sum(axis=1) + base, np.log(raw / (1 - raw)), atol=1e-4)


@requires_shap
def test_global_importance_covers_every_feature(explainer, business_df):
    table = explainer.global_importance(business_df[INPUT_COLUMNS].head(100))
    assert len(table) == len(explainer.feature_names)
    assert table["mean_abs_shap"].is_monotonic_decreasing
    grouped = grouped_importance(table)
    assert {"state", "industry", "profit_margin"} <= set(grouped["source_column"])


@requires_shap
def test_explain_record_requires_single_row(explainer, business_df):
    contributions, _ = explainer.explain_record(business_df[INPUT_COLUMNS].head(1), top_n=5)
    assert len(contributions) == 5
    with pytest.raises(ValueError):
        explainer.explain_record(business_df[INPUT_COLUMNS].head(2))


@requires_shap
def test_feature_name_mismatch_detected(explainer, business_df):
    explainer.feature_names = list(reversed(explainer.feature_names))
    with pytest.raises(ValueError):
        explainer.transform(business_df[INPUT_COLUMNS].head(3))


def test_missing_shap_raises_clear_error(monkeypatch, trained_risk_model, business_df):
    monkeypatch.setattr(interpret, "shap", None)
    with pytest.raises(ExplanationUnavailable):
        ModelExplainer(trained_risk_model.model_pipeline, business_df[INPUT_COLUMNS])


def test_permutation_fallback(trained_risk_model, business_df):
    table = permutation_importance_table(
        trained_risk_model.model_pipeline, business_df[INPUT_COLUMNS], business_df[TARGET_COLUMN].to_numpy(), n_repeats=2
    )
    assert set(table["feature"]) == set(INPUT_COLUMNS)
