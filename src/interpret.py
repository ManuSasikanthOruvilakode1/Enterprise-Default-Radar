"""SHAP explanations for the selected model.

SHAP values describe how the model uses each feature to move a prediction away from the
average (in log-odds of the uncalibrated model). They describe model behaviour and
associations in the training data, not causal effects on insolvency.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .feature_engineering import source_column

try:
    import shap
except ImportError:  # SHAP is an optional dependency
    shap = None


class ExplanationUnavailable(RuntimeError):
    """Raised when SHAP explanations cannot be produced for the model or environment."""


def shap_available() -> bool:
    return shap is not None


class ModelExplainer:
    """Explains a fitted pipeline whose last step is the estimator and earlier steps build features."""

    def __init__(self, pipeline: Pipeline, background_raw: pd.DataFrame, background_size: int = 200, seed: int = 42):
        if shap is None:
            raise ExplanationUnavailable("SHAP is not installed. Install it with `uv sync --extra advanced`.")
        self.seed = seed
        self.preprocessor = pipeline[:-1]
        self.model = pipeline[-1]
        self.feature_names = list(pipeline.named_steps["features"].feature_names_out_)

        sample = background_raw.sample(min(background_size, len(background_raw)), random_state=seed)
        background = self.transform(sample)
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.explainer_type = "tree"
        except Exception:
            if hasattr(self.model, "coef_"):
                self.explainer = shap.LinearExplainer(self.model, background)
                self.explainer_type = "linear"
            else:
                self.explainer = shap.Explainer(self._log_odds, background)
                self.explainer_type = "permutation"

    def _log_odds(self, X: np.ndarray) -> np.ndarray:
        probs = np.clip(self.model.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        return np.log(probs / (1 - probs))

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Runs the preprocessing steps and validates the result against the model's expected inputs."""
        transformed = self.preprocessor.transform(raw)
        transformed = pd.DataFrame(np.asarray(transformed, dtype=float), columns=self.feature_names, index=raw.index)
        expected_count = getattr(self.model, "n_features_in_", None)
        if expected_count is not None and expected_count != transformed.shape[1]:
            raise ValueError(f"Model expects {expected_count} features but preprocessing produced {transformed.shape[1]}")
        expected_names = getattr(self.model, "feature_names_in_", None)
        if expected_names is not None and list(expected_names) != self.feature_names:
            raise ValueError("Transformed feature names do not match the names the model was trained with")
        return transformed

    def shap_values(self, raw: pd.DataFrame) -> Tuple[np.ndarray, float, pd.DataFrame]:
        """SHAP values (n_samples x n_features), base value and the transformed features."""
        X = self.transform(raw)
        explanation = self.explainer(X)
        values = np.asarray(explanation.values)
        base = np.asarray(explanation.base_values)
        if values.ndim == 3:  # some explainers return one slice per class
            values = values[..., 1]
            base = base[..., 1] if base.ndim > 1 else base
        return values, float(np.ravel(base)[0]), X

    def global_importance(self, raw: pd.DataFrame) -> pd.DataFrame:
        values, _, _ = self.shap_values(raw)
        table = pd.DataFrame({
            "feature": self.feature_names,
            "source_column": [source_column(f) for f in self.feature_names],
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        })
        table = table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        table.insert(0, "rank", np.arange(1, len(table) + 1))
        return table

    def explain_record(self, raw_row: pd.DataFrame, top_n: Optional[int] = None) -> Tuple[pd.DataFrame, float]:
        """Per-feature contributions for one record, sorted by absolute size, plus the base value."""
        if len(raw_row) != 1:
            raise ValueError("explain_record expects exactly one row")
        values, base, X = self.shap_values(raw_row)
        table = pd.DataFrame({"feature": self.feature_names, "feature_value": X.iloc[0].to_numpy(), "shap_value": values[0]})
        table = table.reindex(table["shap_value"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        return (table.head(top_n) if top_n else table), base

    def plot_summary(self, raw: pd.DataFrame, output_path: str, max_display: int = 15) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        values, _, X = self.shap_values(raw)
        shap.summary_plot(values, X, max_display=max_display, show=False, rng=np.random.default_rng(self.seed))
        fig = plt.gcf()
        fig.suptitle("SHAP summary (impact on model log-odds)", fontsize=11)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def grouped_importance(feature_importance: pd.DataFrame) -> pd.DataFrame:
    """Sums mean |SHAP| of one-hot columns back to their raw input column (state, industry)."""
    grouped = feature_importance.groupby("source_column", as_index=False)["mean_abs_shap"].sum()
    return grouped.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def permutation_importance_table(pipeline: Pipeline, X_raw: pd.DataFrame, y: np.ndarray, seed: int = 42, n_repeats: int = 10) -> pd.DataFrame:
    """Fallback global importance on raw input columns when SHAP is unavailable."""
    result = permutation_importance(pipeline, X_raw, y, scoring="average_precision", n_repeats=n_repeats, random_state=seed)
    table = pd.DataFrame({
        "feature": X_raw.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)
