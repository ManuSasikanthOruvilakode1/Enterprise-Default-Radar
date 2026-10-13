"""Learning-curve experiment: how model quality changes with the amount of training data.

Because the data is synthetic, fresh training samples of any size can be drawn from the same
generator and scored on one large, independent test sample. This shows where each model sits
relative to the best achievable score (the true generating probabilities) as data grows.
"""

import time
from typing import List, Tuple

import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import ModelConfig
from .data_generation import generate_business_dataset, oracle_probabilities
from .schema import INPUT_COLUMNS, TARGET_COLUMN
from .tuning import fit_model


def _score_one(model_config: ModelConfig, size: int, repeat: int, train_seed: int, test_rows: int, test_seed: int, quantiles) -> dict:
    train_df = generate_business_dataset(size, seed=train_seed)
    test_df = generate_business_dataset(test_rows, seed=test_seed)
    start = time.perf_counter()
    pipeline, _, _ = fit_model(model_config, train_df[INPUT_COLUMNS], train_df[TARGET_COLUMN].to_numpy(), quantiles, train_seed)
    fit_seconds = time.perf_counter() - start
    probs = pipeline.predict_proba(test_df[INPUT_COLUMNS])[:, 1]
    y_test = test_df[TARGET_COLUMN].to_numpy()
    return {
        "train_size": size,
        "repeat": repeat,
        "model": model_config.name,
        "train_positives": int(train_df[TARGET_COLUMN].sum()),
        "average_precision": float(average_precision_score(y_test, probs)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "fit_seconds": round(fit_seconds, 2),
        "train_seed": train_seed,
    }


def run_learning_curve(
    model_configs: List[ModelConfig],
    train_sizes: List[int],
    test_rows: int,
    repeats: int,
    test_seed: int,
    train_seed_start: int,
    quantiles: Tuple[float, float] = (0.01, 0.99),
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, dict]:
    """Returns one row per (size, repeat, model) and the ceiling scores on the shared test sample.

    Every model sees the same training sample for a given size and repeat.
    """
    if test_seed in {train_seed_start + 100 * i + r for i in range(len(train_sizes)) for r in range(repeats)}:
        raise ValueError("The test seed must differ from every training seed")
    jobs = [
        delayed(_score_one)(config, size, repeat, train_seed_start + 100 * i + repeat, test_rows, test_seed, quantiles)
        for i, size in enumerate(train_sizes)
        for repeat in range(repeats)
        for config in model_configs
    ]
    results = pd.DataFrame(Parallel(n_jobs=n_jobs)(jobs))

    test_df = generate_business_dataset(test_rows, seed=test_seed)
    oracle = oracle_probabilities(test_df)
    ceiling = {
        "average_precision": float(average_precision_score(test_df[TARGET_COLUMN], oracle)),
        "roc_auc": float(roc_auc_score(test_df[TARGET_COLUMN], oracle)),
        "test_rows": test_rows,
        "test_positives": int(test_df[TARGET_COLUMN].sum()),
    }
    return results, ceiling


def summarise(results: pd.DataFrame, metric: str = "average_precision") -> pd.DataFrame:
    """Mean and standard deviation of a metric per training size (rows) and model (columns)."""
    grouped = results.groupby(["train_size", "model"])[metric].agg(["mean", "std"]).reset_index()
    return grouped


def plot_learning_curve(results: pd.DataFrame, ceiling: dict, output_path: str, highlight_size: int = None, metric: str = "average_precision") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, frame in results.groupby("model", sort=False):
        stats = frame.groupby("train_size")[metric].agg(["mean", "min", "max"])
        ax.plot(stats.index, stats["mean"], marker="o", linewidth=2, label=model)
        ax.fill_between(stats.index, stats["min"], stats["max"], alpha=0.15)
    ax.axhline(ceiling[metric], color="black", linestyle="--", linewidth=1, label="True probabilities (ceiling)")
    if highlight_size:
        ax.axvline(highlight_size, color="grey", linestyle=":", linewidth=1, label=f"Project dataset ({highlight_size:,} rows)")
    ax.set_xscale("log")
    ax.set_xlabel("Training rows (log scale)")
    ax.set_ylabel("AUC-PR on independent test sample" if metric == "average_precision" else metric)
    ax.set_title("Learning curve: model quality vs training data size")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
