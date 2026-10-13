"""Data ingestion and schema validation for the business outcomes dataset."""

import os
from typing import List

import pandas as pd

from .schema import CATEGORICAL_COLUMNS, ID_COLUMN, NUMERIC_COLUMNS, TARGET_COLUMN


class DataIngestor:
    """Loads the raw CSV and verifies columns, types, missing values and target labels."""

    REQUIRED_COLUMNS: List[str] = [ID_COLUMN] + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET_COLUMN]

    @classmethod
    def load_and_validate(cls, csv_path: str) -> pd.DataFrame:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Dataset file not found at: {csv_path}. "
                "Run `uv run python scripts/generate_synthetic_business_data.py` to create it."
            )
        df = pd.read_csv(csv_path)
        cls.validate(df)
        return df

    @classmethod
    def validate(cls, df: pd.DataFrame) -> None:
        missing = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required schema columns: {missing}")

        null_counts = df[cls.REQUIRED_COLUMNS].isna().sum()
        columns_with_nulls = null_counts[null_counts > 0].to_dict()
        if columns_with_nulls:
            raise ValueError(f"Dataset contains missing values: {columns_with_nulls}")

        non_numeric = [col for col in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            raise ValueError(f"Expected numeric columns but found non-numeric values in: {non_numeric}")

        negative = [col for col in NUMERIC_COLUMNS if col != "profit_margin" and (df[col] < 0).any()]
        if negative:
            raise ValueError(f"Columns contain negative values: {negative}")

        labels = set(df[TARGET_COLUMN].unique())
        if not labels <= {0, 1}:
            raise ValueError(f"Target column '{TARGET_COLUMN}' must be binary 0/1, found {sorted(labels)}")
        if len(labels) < 2:
            raise ValueError(f"Target column '{TARGET_COLUMN}' contains only one class")

        if df[ID_COLUMN].duplicated().any():
            raise ValueError(f"Duplicate values found in '{ID_COLUMN}'")
