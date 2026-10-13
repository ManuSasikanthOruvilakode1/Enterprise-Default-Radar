"""Tests for dataset loading and schema validation."""

import pytest

from src.ingest import DataIngestor


def test_load_valid_dataset(tmp_path, business_df):
    path = tmp_path / "data.csv"
    business_df.to_csv(path, index=False)
    loaded = DataIngestor.load_and_validate(str(path))
    assert len(loaded) == len(business_df)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        DataIngestor.load_and_validate("does/not/exist.csv")


def test_missing_column_raises(business_df):
    with pytest.raises(ValueError, match="missing required"):
        DataIngestor.validate(business_df.drop(columns="current_ratio"))


def test_null_values_raise(business_df):
    df = business_df.copy()
    df.loc[0, "profit_margin"] = None
    with pytest.raises(ValueError, match="missing values"):
        DataIngestor.validate(df)


def test_non_binary_target_raises(business_df):
    df = business_df.copy()
    df.loc[0, "insolvency_risk"] = 2
    with pytest.raises(ValueError, match="binary"):
        DataIngestor.validate(df)


def test_non_numeric_feature_raises(business_df):
    df = business_df.copy()
    df["current_ratio"] = df["current_ratio"].astype(str)
    with pytest.raises(ValueError, match="non-numeric"):
        DataIngestor.validate(df)


def test_duplicate_ids_raise(business_df):
    df = business_df.copy()
    df.loc[1, "business_id"] = df.loc[0, "business_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        DataIngestor.validate(df)
