"""Model tournament: compares candidate models on identical repeated stratified CV splits.

Preprocessing and feature engineering are refitted inside every training fold. The final
held-out test set is never passed to this module.
"""

import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError
from sklearn.model_selection import RepeatedStratifiedKFold

from .config import ModelConfig
from .evaluate import classification_metrics
from .train import is_available, library_version
from .tuning import fit_model, tuning_available

METRIC_NAMES = ["average_precision", "roc_auc", "precision", "recall", "f1", "balanced_accuracy", "brier_score"]


@dataclass
class CVSplit:
    repeat: int
    fold: int
    train_index: np.ndarray
    validation_index: np.ndarray


@dataclass
class CandidateResult:
    model_config: ModelConfig
    fold_metrics: pd.DataFrame
    oof_probabilities: np.ndarray
    runtime_seconds: float
    library_version: str
    tuning_trials: Optional[pd.DataFrame] = None


@dataclass
class TournamentResult:
    candidates: Dict[str, CandidateResult]
    skipped: Dict[str, str]
    selected_model: str
    primary_metric: str
    seed: int
    n_splits: int
    n_repeats: int
    baseline_model: Optional[str] = None
    splits: List[CVSplit] = field(default_factory=list, repr=False)

    def fold_table(self) -> pd.DataFrame:
        """One row per model, repeat and fold, with versions, params and seed for traceability."""
        frames = []
        for name, result in self.candidates.items():
            frame = result.fold_metrics.copy()
            frame.insert(0, "model", name)
            frame.insert(1, "role", result.model_config.role)
            frame["library_version"] = result.library_version
            frame["monotonic_constraints"] = json.dumps(result.model_config.monotonic_constraints, sort_keys=True)
            frame["cv_seed"] = self.seed
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def tuning_trials_table(self) -> Optional[pd.DataFrame]:
        """Every Optuna trial run inside every outer training fold, for tuned models."""
        frames = [
            result.tuning_trials.assign(model=name)
            for name, result in self.candidates.items()
            if result.tuning_trials is not None
        ]
        if not frames:
            return None
        table = pd.concat(frames, ignore_index=True)
        leading = ["model", "repeat", "fold", "number", "value", "state"]
        return table[leading + [c for c in table.columns if c not in leading]]

    def summary_table(self) -> pd.DataFrame:
        """Mean and standard deviation of each metric across all folds, best primary metric first."""
        rows = []
        baseline_scores = None
        if self.baseline_model in self.candidates:
            baseline_scores = self.candidates[self.baseline_model].fold_metrics[self.primary_metric].to_numpy()
        for name, result in self.candidates.items():
            metrics = result.fold_metrics
            row = {"model": name, "role": result.model_config.role}
            for metric in METRIC_NAMES:
                row[f"{metric}_mean"] = float(metrics[metric].mean())
                row[f"{metric}_std"] = float(metrics[metric].std(ddof=1))
            if baseline_scores is not None:
                row["folds_better_than_baseline"] = int(np.sum(metrics[self.primary_metric].to_numpy() > baseline_scores))
            row["n_folds"] = len(metrics)
            row["runtime_seconds"] = result.runtime_seconds
            row["library_version"] = result.library_version
            rows.append(row)
        summary = pd.DataFrame(rows)
        return summary.sort_values(f"{self.primary_metric}_mean", ascending=False).reset_index(drop=True)


def make_cv_splits(y: np.ndarray, n_splits: int, n_repeats: int, seed: int) -> List[CVSplit]:
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    placeholder = np.zeros(len(y))
    return [
        CVSplit(repeat=i // n_splits, fold=i % n_splits, train_index=train_idx, validation_index=val_idx)
        for i, (train_idx, val_idx) in enumerate(splitter.split(placeholder, y))
    ]


def split_seed(seed: int, split: CVSplit) -> int:
    """Distinct, reproducible seed for the tuning search inside each outer split."""
    return seed + 1000 * split.repeat + split.fold


def _evaluate_split(model_config: ModelConfig, X: pd.DataFrame, y: np.ndarray, split: CVSplit, quantiles, seed: int):
    pipeline, params, trials = fit_model(model_config, X.iloc[split.train_index], y[split.train_index], quantiles, split_seed(seed, split))
    probs = pipeline.predict_proba(X.iloc[split.validation_index])[:, 1]
    y_val = y[split.validation_index]
    row = {
        "repeat": split.repeat,
        "fold": split.fold,
        "n_validation": len(y_val),
        "n_validation_positives": int(y_val.sum()),
        **classification_metrics(y_val, probs, threshold=0.5),
        "params": json.dumps(params, sort_keys=True),
    }
    if trials is not None:
        trials = trials.assign(repeat=split.repeat, fold=split.fold)
    return probs, row, trials


def cross_validate_model(
    model_config: ModelConfig,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: List[CVSplit],
    quantiles: Tuple[float, float] = (0.01, 0.99),
    seed: int = 42,
    n_jobs: int = 1,
) -> CandidateResult:
    """Fits (and, if configured, tunes) a fresh pipeline per split.

    Returns fold metrics and out-of-fold probabilities averaged over repeats. Splits are
    independent and individually seeded, so results do not depend on `n_jobs`.
    """
    start = time.perf_counter()
    jobs = [delayed(_evaluate_split)(model_config, X, y, split, quantiles, seed) for split in splits]
    try:
        outputs = Parallel(n_jobs=n_jobs)(jobs)
    except TerminatedWorkerError:
        # A native library crashed a worker process (seen intermittently with XGBoost on Windows).
        # Every split is independently seeded, so a serial rerun gives identical results.
        warnings.warn(f"A worker process crashed while evaluating '{model_config.name}'; rerunning its splits serially.", RuntimeWarning, stacklevel=2)
        outputs = Parallel(n_jobs=1)(jobs)
    runtime = time.perf_counter() - start

    oof_sum = np.zeros(len(y))
    oof_count = np.zeros(len(y))
    for split, (probs, _, _) in zip(splits, outputs, strict=True):
        oof_sum[split.validation_index] += probs
        oof_count[split.validation_index] += 1
    if np.any(oof_count == 0):
        raise ValueError("Some samples never appeared in a validation fold")
    trials = [t for _, _, t in outputs if t is not None]
    return CandidateResult(
        model_config=model_config,
        fold_metrics=pd.DataFrame([row for _, row, _ in outputs]),
        oof_probabilities=oof_sum / oof_count,
        runtime_seconds=round(runtime, 2),
        library_version=library_version(model_config.model_type),
        tuning_trials=pd.concat(trials, ignore_index=True) if trials else None,
    )


def run_tournament(
    X: pd.DataFrame,
    y: np.ndarray,
    model_configs: List[ModelConfig],
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    quantiles: Tuple[float, float] = (0.01, 0.99),
    primary_metric: str = "average_precision",
    verbose: bool = True,
    n_jobs: int = 1,
) -> TournamentResult:
    """Evaluates every available model on the same splits and selects the best selectable one."""
    if primary_metric not in METRIC_NAMES:
        raise ValueError(f"Unknown primary metric: {primary_metric}")
    y = np.asarray(y)
    X = X.reset_index(drop=True)
    splits = make_cv_splits(y, n_splits, n_repeats, seed)

    candidates: Dict[str, CandidateResult] = {}
    skipped: Dict[str, str] = {}
    for model_config in model_configs:
        missing = not is_available(model_config.model_type) or (model_config.tuning is not None and not tuning_available())
        if missing:
            skipped[model_config.name] = "optional dependency not installed"
            if verbose:
                print(f"   - {model_config.name:28s} skipped (optional dependency not installed)")
            continue
        result = cross_validate_model(model_config, X, y, splits, quantiles, seed, n_jobs)
        candidates[model_config.name] = result
        if verbose:
            mean_score = result.fold_metrics[primary_metric].mean()
            print(f"   - {model_config.name:28s} {primary_metric}={mean_score:.4f}  ({result.runtime_seconds:.1f}s)")

    selectable = {name: res for name, res in candidates.items() if res.model_config.selectable}
    if not selectable:
        raise RuntimeError("No selectable model could be evaluated")
    selected = max(selectable, key=lambda name: selectable[name].fold_metrics[primary_metric].mean())
    baseline = next((name for name, res in candidates.items() if res.model_config.role == "baseline"), None)

    return TournamentResult(
        candidates=candidates,
        skipped=skipped,
        selected_model=selected,
        primary_metric=primary_metric,
        seed=seed,
        n_splits=n_splits,
        n_repeats=n_repeats,
        baseline_model=baseline,
        splits=splits,
    )
