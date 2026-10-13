"""Leakage-free dataset splitting and outlier capping."""

from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


class TabularPreprocessor:
    """Splits the dataset and offers one-off outlier capping for exploratory use."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def cap_outliers(self, df: pd.DataFrame, numeric_cols: List[str], lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
        """Caps a frame at its own quantiles. Inside models use OutlierCapper instead."""
        return OutlierCapper(numeric_cols, lower_q, upper_q).fit_transform(df)

    def stratified_split(
        self,
        df: pd.DataFrame,
        target_col: str = "insolvency_risk",
        test_size: float = 0.25,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Performs a stratified split preserving the class distribution."""
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_state,
            stratify=df[target_col],
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Learns quantile caps on training data and applies the same caps to any later data."""

    def __init__(self, numeric_cols: Optional[List[str]] = None, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.numeric_cols = numeric_cols
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y=None) -> "OutlierCapper":
        columns = self.numeric_cols if self.numeric_cols is not None else X.select_dtypes("number").columns.tolist()
        self.caps_: Dict[str, Tuple[float, float]] = {
            col: (float(X[col].quantile(self.lower_quantile)), float(X[col].quantile(self.upper_quantile)))
            for col in columns
            if col in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_capped = X.copy()
        for col, (low, high) in self.caps_.items():
            if col in X_capped.columns:
                X_capped[col] = X_capped[col].clip(lower=low, upper=high)
        return X_capped
