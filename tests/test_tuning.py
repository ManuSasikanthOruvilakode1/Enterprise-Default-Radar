"""Tests for monotonic constraints, nested hyperparameter tuning and parallel determinism."""

import numpy as np
import pandas as pd
import pytest

from src import tuning
from src.config import ModelConfig, TuningConfig
from src.schema import INPUT_COLUMNS, TARGET_COLUMN
from src.tournament import cross_validate_model, make_cv_splits
from src.train import build_model_pipeline, is_available, model_feature_names, monotonic_params

CONSTRAINTS = {"debt_to_equity_ratio": 1, "leverage_liquidity_ratio": 1, "current_ratio": -1, "profit_margin": -1}
requires_lightgbm = pytest.mark.skipif(not is_available("lightgbm"), reason="LightGBM not installed")
requires_optuna = pytest.mark.skipif(not tuning.tuning_available(), reason="Optuna not installed")


def lgbm_config(**overrides) -> ModelConfig:
    base = dict(name="lgbm", model_type="lightgbm", role="candidate",
                params={"n_estimators": 30, "verbose": -1, "n_jobs": 1, "random_state": 0})
    return ModelConfig(**{**base, **overrides})


def test_constraint_vector_follows_feature_order():
    names = model_feature_names()
    vector = monotonic_params("lightgbm", CONSTRAINTS)["monotone_constraints"]
    assert len(vector) == len(names)
    assert vector[names.index("debt_to_equity_ratio")] == 1
    assert vector[names.index("current_ratio")] == -1
    assert sum(abs(v) for v in vector) == len(CONSTRAINTS)


def test_constraint_parameter_names_per_library():
    assert "monotone_constraints" in monotonic_params("xgboost", CONSTRAINTS)
    assert "monotonic_cst" in monotonic_params("hist_gradient_boosting", CONSTRAINTS)
    assert monotonic_params("lightgbm", {}) == {}


def test_constraint_on_input_requires_matching_derived_constraint():
    with pytest.raises(ValueError, match="leverage_liquidity_ratio"):
        monotonic_params("lightgbm", {"debt_to_equity_ratio": 1})
    with pytest.raises(ValueError, match="leverage_liquidity_ratio"):
        monotonic_params("lightgbm", {"current_ratio": -1, "leverage_liquidity_ratio": -1})


def test_shipped_constraints_are_consistent():
    from src.config import load_model_config
    for name in ("lightgbm_monotonic", "lightgbm_tuned", "xgboost_monotonic", "xgboost_tuned"):
        config = load_model_config(f"configs/{name}.yaml")
        assert monotonic_params(config.model_type, config.monotonic_constraints)


def test_unknown_feature_or_unsupported_model_rejected():
    with pytest.raises(ValueError, match="unknown features"):
        monotonic_params("lightgbm", {"not_a_feature": 1})
    with pytest.raises(ValueError, match="not supported"):
        monotonic_params("logistic_regression", CONSTRAINTS)


@pytest.mark.parametrize("model_type,params", [
    ("hist_gradient_boosting", {"max_iter": 30}),
    ("lightgbm", {"n_estimators": 30, "verbose": -1}),
    ("xgboost", {"n_estimators": 30}),
])
def test_predictions_respect_constraints(business_df, model_type, params):
    if not is_available(model_type):
        pytest.skip(f"{model_type} not installed")
    config = ModelConfig(name=model_type, model_type=model_type, role="candidate", params=params, monotonic_constraints=CONSTRAINTS)
    pipeline = build_model_pipeline(config).fit(business_df[INPUT_COLUMNS], business_df[TARGET_COLUMN])
    rows = pd.concat([business_df[INPUT_COLUMNS].head(20)] * 30, ignore_index=True)
    rows["debt_to_equity_ratio"] = np.repeat(np.linspace(0.1, 6, 30), 20)
    probs = pipeline.predict_proba(rows)[:, 1].reshape(30, 20)
    assert np.all(np.diff(probs, axis=0) >= -1e-9)


def test_search_spaces_produce_valid_parameters():
    optuna = pytest.importorskip("optuna")
    for model_type, space in tuning.SEARCH_SPACES.items():
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        params = space(study.ask())
        assert params, model_type


@requires_lightgbm
@requires_optuna
def test_fit_model_tunes_and_records_trials(business_df):
    config = lgbm_config(tuning=TuningConfig(n_trials=4, inner_folds=2))
    pipeline, params, trials = tuning.fit_model(config, business_df[INPUT_COLUMNS], business_df[TARGET_COLUMN].to_numpy(), seed=0)
    assert len(trials) == 4
    assert {"number", "value", "state", "num_leaves"} <= set(trials.columns)
    assert params["verbose"] == -1  # fixed settings are kept
    assert pipeline.named_steps["model"].get_params()["num_leaves"] == params["num_leaves"]


@requires_lightgbm
@requires_optuna
def test_tuning_is_reproducible(business_df):
    config = lgbm_config(tuning=TuningConfig(n_trials=3, inner_folds=2))
    X, y = business_df[INPUT_COLUMNS], business_df[TARGET_COLUMN].to_numpy()
    assert tuning.fit_model(config, X, y, seed=1)[1] == tuning.fit_model(config, X, y, seed=1)[1]


@requires_lightgbm
def test_untuned_model_returns_config_params(business_df):
    config = lgbm_config()
    _, params, trials = tuning.fit_model(config, business_df[INPUT_COLUMNS], business_df[TARGET_COLUMN].to_numpy())
    assert params == config.params and trials is None


def test_tuning_without_settings_or_space_rejected(business_df):
    X, y = business_df[INPUT_COLUMNS], business_df[TARGET_COLUMN].to_numpy()
    if not tuning.tuning_available():
        pytest.skip("Optuna not installed")
    with pytest.raises(ValueError, match="no tuning settings"):
        tuning.tune_hyperparameters(lgbm_config(), X, y)
    linear = ModelConfig(name="lr", model_type="logistic_regression", role="reference", tuning=TuningConfig(n_trials=2, inner_folds=2))
    with pytest.raises(ValueError, match="No search space"):
        tuning.tune_hyperparameters(linear, X, y)


def test_missing_optuna_gives_install_hint(monkeypatch, business_df):
    monkeypatch.setattr(tuning, "optuna", None)
    config = lgbm_config(tuning=TuningConfig(n_trials=2, inner_folds=2))
    with pytest.raises(ImportError, match="uv sync --extra advanced"):
        tuning.tune_hyperparameters(config, business_df[INPUT_COLUMNS], business_df[TARGET_COLUMN].to_numpy())


@requires_lightgbm
@requires_optuna
def test_nested_cv_results_do_not_depend_on_parallelism(business_df):
    X, y = business_df[INPUT_COLUMNS], business_df[TARGET_COLUMN].to_numpy()
    splits = make_cv_splits(y, 2, 1, 0)
    config = lgbm_config(tuning=TuningConfig(n_trials=2, inner_folds=2), monotonic_constraints=CONSTRAINTS)
    serial = cross_validate_model(config, X, y, splits, seed=0, n_jobs=1)
    parallel = cross_validate_model(config, X, y, splits, seed=0, n_jobs=2)
    assert np.allclose(serial.oof_probabilities, parallel.oof_probabilities)
    assert list(serial.fold_metrics["params"]) == list(parallel.fold_metrics["params"])
    assert len(serial.tuning_trials) == 2 * len(splits)
