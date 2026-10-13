"""Tests for FeatureEngineer."""

import pandas as pd

from src.feature_engineering import FeatureEngineer, dummy_column_name, source_column


def make_record(**overrides) -> pd.DataFrame:
    record = {
        "business_id": "BIZ_01",
        "state": "NSW",
        "industry": "Construction",
        "employee_count": 10,
        "annual_turnover_aud": 1000000.0,
        "years_operating": 5,
        "r_and_d_expenditure_aud": 50000.0,
        "debt_to_equity_ratio": 1.5,
        "current_ratio": 2.0,
        "profit_margin": 0.10,
        "insolvency_risk": 0,
    }
    record.update(overrides)
    return pd.DataFrame([record])


def test_feature_engineering_ratios():
    transformed = FeatureEngineer().fit_transform(make_record())
    assert transformed["revenue_per_employee"].iloc[0] == 100000.0
    assert transformed["r_and_d_intensity"].iloc[0] == 0.05
    assert abs(transformed["leverage_liquidity_ratio"].iloc[0] - 1.5 / 2.05) < 1e-12


def test_output_excludes_identifier_and_target(business_df):
    transformed = FeatureEngineer().fit_transform(business_df)
    assert "business_id" not in transformed.columns
    assert "insolvency_risk" not in transformed.columns
    assert all(pd.api.types.is_float_dtype(dtype) for dtype in transformed.dtypes)


def test_columns_stable_between_fit_and_transform(business_df):
    engineer = FeatureEngineer().fit(business_df)
    single = engineer.transform(business_df.head(1))
    assert list(single.columns) == list(engineer.get_feature_names_out())


def test_unseen_category_maps_to_reference_level(business_df):
    engineer = FeatureEngineer().fit(business_df)
    transformed = engineer.transform(make_record(state="Unknown"))
    state_columns = [c for c in transformed.columns if c.startswith("state_")]
    assert transformed[state_columns].sum(axis=1).iloc[0] == 0


def test_dummy_names_are_model_safe():
    assert dummy_column_name("industry", "Accommodation & Food") == "industry_Accommodation_and_Food"
    assert source_column("industry_Accommodation_and_Food") == "industry"
    assert source_column("profit_margin") == "profit_margin"
