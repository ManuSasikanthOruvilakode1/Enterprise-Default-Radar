"""Hyperparameter search with Optuna, and the single entry point for fitting any configured model.

`fit_model` tunes on the data it is given (when the model config asks for it) and then fits
the final pipeline on that same data. Inside the tournament it is called once per outer
training fold, so tuning is nested and never sees the fold it is scored on.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import ModelConfig
from .train import build_model_pipeline

try:
    import optuna
except ImportError:  # Optuna is an optional dependency
    optuna = None


# Search spaces from the project handover. Fixed settings (seeds, threads, class weighting)
# stay in each model's YAML config and are merged underneath the sampled values.
def _xgboost_space(trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.60, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0, step=0.05),
        "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1.0, 3.0, 5.0, 10.0]),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _lightgbm_space(trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100, step=10),
        "subsample": trial.suggest_float("subsample", 0.60, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0, step=0.05),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _catboost_space(trial) -> Dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 150, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
    }


SEARCH_SPACES: Dict[str, Callable[[Any], Dict[str, Any]]] = {
    "xgboost": _xgboost_space,
    "lightgbm": _lightgbm_space,
    "catboost": _catboost_space,
}


def tune_hyperparameters(
    model_config: ModelConfig,
    X: pd.DataFrame,
    y: np.ndarray,
    quantiles: Tuple[float, float] = (0.01, 0.99),
    seed: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Maximises mean inner-CV average precision; returns the best full parameter set and every trial."""
    if optuna is None:
        raise ImportError("Hyperparameter tuning requires Optuna. Install it with `uv sync --extra advanced`.")
    if model_config.tuning is None:
        raise ValueError(f"Model '{model_config.name}' has no tuning settings")
    if model_config.model_type not in SEARCH_SPACES:
        raise ValueError(f"No search space defined for model type '{model_config.model_type}'")

    y = np.asarray(y)
    X = X.reset_index(drop=True)
    inner_splits = list(
        StratifiedKFold(model_config.tuning.inner_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    )
    space = SEARCH_SPACES[model_config.model_type]

    def objective(trial) -> float:
        candidate = model_config.model_copy(update={"params": {**model_config.params, **space(trial)}, "tuning": None})
        scores = []
        for step, (train_idx, val_idx) in enumerate(inner_splits):
            pipeline = build_model_pipeline(candidate, *quantiles).fit(X.iloc[train_idx], y[train_idx])
            scores.append(average_precision_score(y[val_idx], pipeline.predict_proba(X.iloc[val_idx])[:, 1]))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=model_config.tuning.n_trials)

    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials.columns = [c.replace("params_", "") for c in trials.columns]
    return {**model_config.params, **study.best_params}, trials


def fit_model(
    model_config: ModelConfig,
    X: pd.DataFrame,
    y: np.ndarray,
    quantiles: Tuple[float, float] = (0.01, 0.99),
    seed: int = 42,
) -> Tuple[Pipeline, Dict[str, Any], Optional[pd.DataFrame]]:
    """Tunes (if configured) and fits a pipeline. Returns the pipeline, the parameters used and any tuning trials."""
    params, trials = model_config.params, None
    if model_config.tuning is not None:
        params, trials = tune_hyperparameters(model_config, X, y, quantiles, seed)
    final_config = model_config.model_copy(update={"params": params, "tuning": None})
    return build_model_pipeline(final_config, *quantiles).fit(X, y), params, trials


def tuning_available() -> bool:
    return optuna is not None
