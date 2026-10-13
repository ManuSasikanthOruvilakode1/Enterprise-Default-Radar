"""Tests for the deployable RiskModel and model persistence."""

import numpy as np
import pytest
from pydantic import ValidationError

from src.export import ModelPersistence
from src.pipeline import FLAGGED_LABEL, NOT_FLAGGED_LABEL
from src.schema import INPUT_COLUMNS


def test_score_profile_returns_probability_and_category(trained_risk_model, valid_profile):
    result = trained_risk_model.score_profile(valid_profile)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["category"] == (FLAGGED_LABEL if result["probability"] >= result["threshold"] else NOT_FLAGGED_LABEL)
    assert result["warnings"] == []


def test_higher_leverage_and_losses_raise_risk(trained_risk_model, valid_profile):
    safe = trained_risk_model.score_profile({**valid_profile, "debt_to_equity_ratio": 0.3, "current_ratio": 3.0, "profit_margin": 0.25})
    risky = trained_risk_model.score_profile({**valid_profile, "debt_to_equity_ratio": 5.0, "current_ratio": 0.4, "profit_margin": -0.3})
    assert risky["probability"] > safe["probability"]


def test_out_of_range_input_warns(trained_risk_model, valid_profile):
    result = trained_risk_model.score_profile({**valid_profile, "debt_to_equity_ratio": 50.0})
    assert any("debt_to_equity_ratio" in w for w in result["warnings"])


def test_invalid_profile_rejected(trained_risk_model, valid_profile):
    with pytest.raises(ValidationError):
        trained_risk_model.score_profile({**valid_profile, "state": "Atlantis"})


def test_batch_predictions_match_threshold(trained_risk_model, business_df):
    probs = trained_risk_model.predict_proba(business_df)
    assert np.array_equal(trained_risk_model.predict(business_df), (probs >= trained_risk_model.threshold).astype(int))


def test_save_and_load_round_trip(trained_risk_model, business_df, tmp_path):
    path = tmp_path / "risk_model.joblib"
    ModelPersistence.save_pipeline(trained_risk_model, str(path))
    loaded = ModelPersistence.load_pipeline(str(path))
    sample = business_df[INPUT_COLUMNS].head(20)
    assert np.allclose(loaded.predict_proba(sample), trained_risk_model.predict_proba(sample))
    assert (tmp_path / "risk_model_metadata.json").exists()


def test_load_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelPersistence.load_pipeline(str(tmp_path / "missing.joblib"))
