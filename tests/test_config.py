"""Tests that the shipped configuration files load and validate."""

import os

import pytest
from pydantic import ValidationError

from src.config import ModelConfig, load_model_config, load_project_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_project_config_loads():
    cfg = load_project_config(os.path.join(ROOT, "config.yaml"))
    assert cfg.cost_matrix.false_negative_cost > cfg.cost_matrix.false_positive_cost
    for path in cfg.tournament.model_configs:
        model_cfg = load_model_config(os.path.join(ROOT, path))
        assert model_cfg.name == os.path.splitext(os.path.basename(path))[0]


def test_exactly_one_baseline_model():
    cfg = load_project_config(os.path.join(ROOT, "config.yaml"))
    roles = [load_model_config(os.path.join(ROOT, p)).role for p in cfg.tournament.model_configs]
    assert roles.count("baseline") == 1


def test_invalid_model_type_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(name="x", model_type="random_forest", role="candidate")


def test_duplicate_model_configs_rejected():
    from src.config import TournamentConfig
    with pytest.raises(ValidationError, match="more than once"):
        TournamentConfig(model_configs=["configs/a.yaml", "configs/a.yaml"])


def test_invalid_constraint_direction_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(name="x", model_type="lightgbm", role="candidate", monotonic_constraints={"current_ratio": 2})


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_project_config(os.path.join(ROOT, "nope.yaml"))
