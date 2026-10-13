"""Feature engineering for the business outcomes dataset."""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

ENGINEERED_COLUMNS = ["revenue_per_employee", "r_and_d_intensity", "leverage_liquidity_ratio", "log_turnover"]


def dummy_column_name(column: str, level: str) -> str:
    """Model-safe one-hot column name, e.g. ('industry', 'Accommodation & Food') -> 'industry_Accommodation_and_Food'."""
    return f"{column}_{level.replace('&', 'and').replace(' ', '_').replace('__', '_')}"


def source_column(feature_name: str) -> str:
    """Raw input column a model feature was derived from (engineered ratios map to themselves)."""
    for col in CATEGORICAL_COLUMNS:
        if feature_name.startswith(f"{col}_"):
            return col
    return feature_name


def output_feature_names(fitted_categories: Dict[str, List[str]]) -> List[str]:
    """Model feature order: raw numeric columns, engineered ratios, then one-hot columns."""
    return NUMERIC_COLUMNS + ENGINEERED_COLUMNS + [
        dummy_column_name(col, level) for col, levels in fitted_categories.items() for level in levels
    ]


def non_reference_levels(categories: Dict[str, Sequence[str]]) -> Dict[str, List[str]]:
    """Sorted levels per column, minus the alphabetically first (reference) level."""
    return {col: sorted(str(level) for level in levels)[1:] for col, levels in categories.items()}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds financial ratios and one-hot encodes state and industry.

    Category levels come from `categories` when given (e.g. the schema's known levels),
    otherwise they are learned in `fit`. Either way the output columns are identical for
    training, validation and scoring data. Unseen categories map to the reference level.
    """

    def __init__(self, categories: Optional[Dict[str, Sequence[str]]] = None):
        self.categories = categories

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        categories = self.categories or {col: X[col].astype(str).unique() for col in CATEGORICAL_COLUMNS if col in X.columns}
        self.fitted_categories: Dict[str, List[str]] = non_reference_levels(categories)
        self.feature_names_out_ = output_feature_names(self.fitted_categories)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_feat = X[NUMERIC_COLUMNS].astype(float).copy()

        df_feat["revenue_per_employee"] = df_feat["annual_turnover_aud"] / np.clip(df_feat["employee_count"], 1, None)
        df_feat["r_and_d_intensity"] = df_feat["r_and_d_expenditure_aud"] / np.clip(df_feat["annual_turnover_aud"], 1, None)
        df_feat["leverage_liquidity_ratio"] = df_feat["debt_to_equity_ratio"] / (df_feat["current_ratio"] + 0.05)
        df_feat["log_turnover"] = np.log1p(df_feat["annual_turnover_aud"])

        for col, levels in self.fitted_categories.items():
            values = X[col].astype(str).to_numpy()
            for level in levels:
                df_feat[dummy_column_name(col, level)] = (values == level).astype(float)

        return df_feat[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)
