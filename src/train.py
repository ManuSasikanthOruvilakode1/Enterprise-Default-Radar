"""Model factory: builds leakage-free scikit-learn pipelines for every candidate model."""

import importlib
from typing import Any, Dict, List, Optional

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig
from .feature_engineering import FeatureEngineer, non_reference_levels, output_feature_names
from .preprocess import OutlierCapper
from .schema import INDUSTRIES, NUMERIC_COLUMNS, STATES

KNOWN_CATEGORIES = {"state": STATES, "industry": INDUSTRIES}

# Derived feature -> {input: direction in which the derived feature moves when the input rises}.
DERIVED_FEATURE_SIGNS = {"leverage_liquidity_ratio": {"debt_to_equity_ratio": 1, "current_ratio": -1}}

# Optional dependency that provides each model type (None means scikit-learn only).
OPTIONAL_PACKAGES = {
    "hist_gradient_boosting": None,
    "logistic_regression": None,
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}


def library_version(model_type: str) -> Optional[str]:
    """Returns the installed library version for a model type, or None if it is not installed."""
    package = OPTIONAL_PACKAGES[model_type] or "sklearn"
    try:
        return importlib.import_module(package).__version__
    except ImportError:
        return None


def is_available(model_type: str) -> bool:
    return library_version(model_type) is not None


def build_estimator(model_type: str, params: Dict[str, Any]):
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**params)
    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    if not is_available(model_type):
        raise ImportError(
            f"Model type '{model_type}' requires the optional package '{OPTIONAL_PACKAGES[model_type]}'. "
            "Install it with `uv sync --extra advanced`."
        )
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    if model_type == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params)
    raise ValueError(f"Unknown model type: {model_type}")


def model_feature_names() -> List[str]:
    """Feature order produced by the pipeline's FeatureEngineer (category levels fixed by the schema)."""
    return output_feature_names(non_reference_levels(KNOWN_CATEGORIES))


def monotonic_params(model_type: str, constraints: Dict[str, int]) -> Dict[str, Any]:
    """Translates {feature: +1/-1} into each library's monotonic-constraint parameter."""
    if not constraints:
        return {}
    feature_names = model_feature_names()
    unknown = sorted(set(constraints) - set(feature_names))
    if unknown:
        raise ValueError(f"Monotonic constraints refer to unknown features: {unknown}")
    for derived, inputs in DERIVED_FEATURE_SIGNS.items():
        for raw, sign in inputs.items():
            if raw in constraints and constraints.get(derived) != constraints[raw] * sign:
                raise ValueError(
                    f"Constraining '{raw}' also requires '{derived}': {constraints[raw] * sign:+d}, because '{derived}' is "
                    "computed from it; otherwise the model can still respond non-monotonically through the derived feature."
                )
    vector = [int(constraints.get(name, 0)) for name in feature_names]
    if model_type == "lightgbm":
        return {"monotone_constraints": vector}
    if model_type == "xgboost":
        return {"monotone_constraints": "(" + ",".join(str(v) for v in vector) + ")"}
    if model_type == "catboost":
        return {"monotone_constraints": vector}
    if model_type == "hist_gradient_boosting":
        return {"monotonic_cst": vector}
    raise ValueError(f"Monotonic constraints are not supported for model type '{model_type}'")


def build_model_pipeline(model_config: ModelConfig, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> Pipeline:
    """Outlier capping -> feature engineering -> (scaling for linear models) -> estimator.

    Every step is fitted only on the data passed to `fit`, so cross-validation folds stay leakage-free.
    """
    steps = [
        ("outlier_capper", OutlierCapper(NUMERIC_COLUMNS, lower_quantile, upper_quantile)),
        ("features", FeatureEngineer(categories=KNOWN_CATEGORIES)),
    ]
    if model_config.model_type == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
    params = {**model_config.params, **monotonic_params(model_config.model_type, model_config.monotonic_constraints)}
    steps.append(("model", build_estimator(model_config.model_type, params)))
    return Pipeline(steps)
