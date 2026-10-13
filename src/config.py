"""Typed loading of the project configuration files."""

import os
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    raw_csv_path: str
    test_size: float = Field(gt=0, lt=1)


class PreprocessingConfig(BaseModel):
    outlier_lower_quantile: float = Field(ge=0, lt=0.5)
    outlier_upper_quantile: float = Field(gt=0.5, le=1)


class CrossValidationConfig(BaseModel):
    n_splits: int = Field(ge=2)
    n_repeats: int = Field(ge=1)
    n_jobs: int = -1  # parallel workers across folds; results do not depend on it


class TournamentConfig(BaseModel):
    model_configs: List[str]

    @field_validator("model_configs")
    @classmethod
    def no_duplicates(cls, value: List[str]) -> List[str]:
        duplicates = sorted({path for path in value if value.count(path) > 1})
        if duplicates:
            raise ValueError(f"Model configs listed more than once: {duplicates}")
        return value


class CostMatrixConfig(BaseModel):
    false_positive_cost: float = Field(gt=0)
    false_negative_cost: float = Field(gt=0)


class ThresholdConfig(BaseModel):
    grid_min: float = Field(gt=0, lt=1)
    grid_max: float = Field(gt=0, lt=1)
    grid_steps: int = Field(ge=2)


class CalibrationConfig(BaseModel):
    methods: List[Literal["none", "sigmoid", "isotonic"]]
    cv_folds: int = Field(ge=2)


class EvaluationConfig(BaseModel):
    bootstrap_rounds: int = Field(ge=100)
    confidence_level: float = Field(gt=0, lt=100)


class SubgroupConfig(BaseModel):
    columns: List[str]
    min_group_size: int = Field(ge=1)
    min_positives: int = Field(ge=1)


class DriftConfig(BaseModel):
    psi_moderate: float = Field(gt=0)
    psi_critical: float = Field(gt=0)
    ks_alpha: float = Field(gt=0, lt=1)


class InterpretabilityConfig(BaseModel):
    background_size: int = Field(ge=10)
    explain_sample_size: int = Field(ge=10)


class LearningCurveConfig(BaseModel):
    train_sizes: List[int]
    test_rows: int = Field(ge=1000)
    repeats: int = Field(ge=1)
    test_seed: int
    train_seed_start: int
    model_configs: List[str]


class ProjectConfig(BaseModel):
    random_seed: int
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    cross_validation: CrossValidationConfig
    tournament: TournamentConfig
    cost_matrix: CostMatrixConfig
    threshold: ThresholdConfig
    calibration: CalibrationConfig
    evaluation: EvaluationConfig
    subgroups: SubgroupConfig
    drift: DriftConfig
    interpretability: InterpretabilityConfig
    learning_curve: LearningCurveConfig


class TuningConfig(BaseModel):
    """Optuna search run inside each training set (nested within the tournament's outer folds)."""

    n_trials: int = Field(ge=1)
    inner_folds: int = Field(ge=2)


class ModelConfig(BaseModel):
    name: str
    model_type: Literal["hist_gradient_boosting", "xgboost", "lightgbm", "catboost", "logistic_regression"]
    role: Literal["baseline", "candidate", "reference"]
    params: Dict[str, Any] = Field(default_factory=dict)
    # Feature name -> +1 (risk can only rise with the feature) or -1 (can only fall). Unlisted features are unconstrained.
    monotonic_constraints: Dict[str, Literal[-1, 1]] = Field(default_factory=dict)
    tuning: Optional[TuningConfig] = None

    @field_validator("params", "monotonic_constraints", mode="before")
    @classmethod
    def default_empty_mapping(cls, value: Any) -> Dict[str, Any]:
        return value or {}

    @property
    def selectable(self) -> bool:
        return self.role != "reference"


def _read_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return content


def load_project_config(path: str) -> ProjectConfig:
    return ProjectConfig(**_read_yaml(path))


def load_model_config(path: str) -> ModelConfig:
    content = _read_yaml(path)
    name = os.path.splitext(os.path.basename(path))[0]
    return ModelConfig(name=name, **content)
