"""Shared fixtures: a small synthetic dataset and a quickly trained risk model."""

import pandas as pd
import pytest

from src.calibrate import ProbabilityCalibrator
from src.config import ModelConfig
from src.data_generation import generate_business_dataset
from src.pipeline import RiskModel
from src.schema import INPUT_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN
from src.train import build_model_pipeline


@pytest.fixture(scope="session")
def business_df() -> pd.DataFrame:
    return generate_business_dataset(n_samples=800, seed=7)


@pytest.fixture(scope="session")
def fast_model_config() -> ModelConfig:
    return ModelConfig(
        name="fast_hgb",
        model_type="hist_gradient_boosting",
        role="baseline",
        params={"max_iter": 30, "max_depth": 3, "class_weight": "balanced", "random_state": 0},
    )


@pytest.fixture(scope="session")
def trained_risk_model(business_df, fast_model_config) -> RiskModel:
    X, y = business_df[INPUT_COLUMNS], business_df[TARGET_COLUMN].to_numpy()
    pipeline = build_model_pipeline(fast_model_config).fit(X, y)
    calibrator = ProbabilityCalibrator("sigmoid").fit(pipeline.predict_proba(X)[:, 1], y)
    ranges = {col: [float(X[col].min()), float(X[col].max())] for col in NUMERIC_COLUMNS}
    return RiskModel(pipeline, calibrator, 0.2, ranges, X.head(50), {"selected_model": "fast_hgb"})


@pytest.fixture
def valid_profile() -> dict:
    return {
        "state": "NSW",
        "industry": "Construction",
        "employee_count": 12,
        "annual_turnover_aud": 1_500_000.0,
        "years_operating": 6,
        "r_and_d_expenditure_aud": 50_000.0,
        "debt_to_equity_ratio": 1.5,
        "current_ratio": 1.4,
        "profit_margin": 0.05,
    }
