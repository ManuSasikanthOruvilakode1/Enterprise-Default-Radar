"""Tests for the model tournament."""

import numpy as np
import pandas as pd
import pytest

from src.config import ModelConfig
from src.schema import INPUT_COLUMNS, TARGET_COLUMN
from src.tournament import METRIC_NAMES, make_cv_splits, run_tournament


@pytest.fixture(scope="module")
def tournament(business_df, fast_model_config):
    reference = ModelConfig(name="linear", model_type="logistic_regression", role="reference", params={"max_iter": 1000})
    return run_tournament(
        business_df[INPUT_COLUMNS], business_df[TARGET_COLUMN].to_numpy(), [fast_model_config, reference],
        n_splits=3, n_repeats=2, seed=0, verbose=False,
    )


def test_splits_are_identical_for_same_seed(business_df):
    y = business_df[TARGET_COLUMN].to_numpy()
    first, second = make_cv_splits(y, 3, 2, 0), make_cv_splits(y, 3, 2, 0)
    assert all(np.array_equal(a.validation_index, b.validation_index) for a, b in zip(first, second, strict=True))


def test_every_sample_validated_once_per_repeat(business_df):
    y = business_df[TARGET_COLUMN].to_numpy()
    splits = make_cv_splits(y, 3, 2, 0)
    for repeat in (0, 1):
        indices = np.concatenate([s.validation_index for s in splits if s.repeat == repeat])
        assert sorted(indices) == list(range(len(y)))


def test_fold_table_has_one_row_per_model_and_fold(tournament):
    table = tournament.fold_table()
    assert len(table) == len(tournament.candidates) * 6
    assert set(METRIC_NAMES) <= set(table.columns)
    assert {"model", "role", "repeat", "fold", "library_version", "params", "cv_seed"} <= set(table.columns)
    assert table["average_precision"].between(0, 1).all()


def test_reference_model_is_never_selected(tournament):
    assert tournament.selected_model == "fast_hgb"
    assert tournament.candidates[tournament.selected_model].model_config.selectable


def test_oof_probabilities_cover_all_samples(tournament, business_df):
    oof = tournament.candidates["fast_hgb"].oof_probabilities
    assert len(oof) == len(business_df)
    assert np.all((oof >= 0) & (oof <= 1))


def test_summary_sorted_by_primary_metric(tournament):
    summary = tournament.summary_table()
    assert summary["average_precision_mean"].is_monotonic_decreasing
    assert "folds_better_than_baseline" in summary.columns


def test_tournament_csv_round_trip(tournament, tmp_path):
    path = tmp_path / "model_tournament.csv"
    tournament.fold_table().to_csv(path, index=False)
    assert len(pd.read_csv(path)) == len(tournament.fold_table())


def test_unavailable_model_is_skipped(business_df, fast_model_config, monkeypatch):
    from src import tournament as tournament_module
    monkeypatch.setattr(tournament_module, "is_available", lambda model_type: model_type != "xgboost")
    unavailable = ModelConfig(name="xgb", model_type="xgboost", role="candidate")
    result = run_tournament(business_df[INPUT_COLUMNS], business_df[TARGET_COLUMN].to_numpy(),
                            [fast_model_config, unavailable], n_splits=2, n_repeats=1, verbose=False)
    assert "xgb" in result.skipped and "xgb" not in result.candidates


def test_unknown_primary_metric_rejected(business_df, fast_model_config):
    with pytest.raises(ValueError):
        run_tournament(business_df[INPUT_COLUMNS], business_df[TARGET_COLUMN].to_numpy(), [fast_model_config],
                       primary_metric="accuracy", verbose=False)
