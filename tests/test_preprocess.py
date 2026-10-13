"""Tests for splitting and outlier capping."""

import pandas as pd

from src.preprocess import OutlierCapper, TabularPreprocessor


def test_stratified_split():
    prep = TabularPreprocessor(random_state=42)
    df = pd.DataFrame({"feature": range(100), "insolvency_risk": [1] * 20 + [0] * 80})
    train_df, test_df = prep.stratified_split(df, target_col="insolvency_risk", test_size=0.25)
    assert len(train_df) == 75
    assert len(test_df) == 25
    assert test_df["insolvency_risk"].sum() == 5


def test_cap_outliers():
    capped = TabularPreprocessor().cap_outliers(pd.DataFrame({"val": [1, 2, 3, 4, 1000]}), ["val"], lower_q=0.0, upper_q=0.8)
    assert capped["val"].max() < 1000


def test_outlier_capper_applies_training_caps_to_new_data():
    train = pd.DataFrame({"val": list(range(1, 101))})
    capper = OutlierCapper(["val"], 0.05, 0.95).fit(train)
    new_data = pd.DataFrame({"val": [-500, 50, 10_000]})
    capped = capper.transform(new_data)
    low, high = capper.caps_["val"]
    assert capped["val"].tolist() == [low, 50, high]
    assert new_data["val"].tolist() == [-500, 50, 10_000]  # input not modified
