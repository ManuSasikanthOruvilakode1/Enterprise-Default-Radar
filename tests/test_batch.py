"""Tests for batch scoring."""

import numpy as np
import pandas as pd
import pytest

from src.batch import OUTPUT_COLUMNS, score_frame


@pytest.fixture
def batch(valid_profile):
    rows = [
        dict(valid_profile, business_id="A"),
        dict(valid_profile, business_id="B", state="Atlantis"),           # unknown category
        dict(valid_profile, business_id="C", current_ratio=np.nan),       # missing value
        dict(valid_profile, business_id="D", debt_to_equity_ratio=50.0),  # valid but out of training range
        dict(valid_profile, business_id="E", employee_count="many"),      # malformed
    ]
    return pd.DataFrame(rows)


def test_valid_rows_scored_and_invalid_rows_explained(trained_risk_model, batch):
    scored = score_frame(trained_risk_model, batch)
    assert list(scored.columns) == list(batch.columns) + OUTPUT_COLUMNS
    by_id = scored.set_index("business_id")
    assert by_id.loc[["A", "D"], "insolvency_probability"].between(0, 1).all()
    assert by_id.loc[["B", "C", "E"], "insolvency_probability"].isna().all()
    assert "state" in by_id.loc["B", "validation_error"]
    assert "current_ratio" in by_id.loc["C", "validation_error"]
    assert "employee_count" in by_id.loc["E", "validation_error"]
    assert "debt_to_equity_ratio" in by_id.loc["D", "warnings"]
    assert by_id.loc["A", "validation_error"] == ""


def test_batch_matches_single_record_scoring(trained_risk_model, valid_profile):
    scored = score_frame(trained_risk_model, pd.DataFrame([valid_profile]))
    single = trained_risk_model.score_profile(valid_profile)
    assert scored.loc[0, "insolvency_probability"] == pytest.approx(single["probability"])
    assert scored.loc[0, "flagged"] == single["flagged"]


def test_missing_column_rejected(trained_risk_model, batch):
    with pytest.raises(ValueError, match="missing required columns"):
        score_frame(trained_risk_model, batch.drop(columns="profit_margin"))


def test_existing_output_column_rejected(trained_risk_model, batch):
    with pytest.raises(ValueError, match="output columns"):
        score_frame(trained_risk_model, batch.assign(flagged=1))


def test_score_csv_script(tmp_path, trained_risk_model, batch, monkeypatch):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    import score_csv

    from src.export import ModelPersistence

    model_path, input_path = tmp_path / "model.joblib", tmp_path / "input.csv"
    ModelPersistence.save_pipeline(trained_risk_model, str(model_path))
    batch.to_csv(input_path, index=False)
    monkeypatch.setattr(sys, "argv", ["score_csv.py", str(input_path), "--model", str(model_path)])
    assert score_csv.main() == 0
    output = pd.read_csv(tmp_path / "input_scored.csv")
    assert len(output) == len(batch) and "insolvency_probability" in output.columns
