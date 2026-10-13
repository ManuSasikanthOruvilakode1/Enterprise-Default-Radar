"""Learning-curve experiment: model quality against training-set size (settings in config.yaml)."""

import argparse
import os
import sys
import warnings

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import load_model_config, load_project_config
from src.learning_curve import plot_learning_curve, run_learning_curve, summarise
from src.reporting import SYNTHETIC_DATA_NOTE, markdown_table, write_markdown

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_DATASET_ROWS = 2500


def main(config_path: str) -> None:
    cfg = load_project_config(config_path)
    lc = cfg.learning_curve
    results_dir = os.path.join(BASE_DIR, "results")
    model_configs = [load_model_config(os.path.join(BASE_DIR, p)) for p in lc.model_configs]
    quantiles = (cfg.preprocessing.outlier_lower_quantile, cfg.preprocessing.outlier_upper_quantile)

    print(f"Learning curve: sizes {lc.train_sizes}, {lc.repeats} repeats, models {[m.name for m in model_configs]}")
    print(f"Test sample: {lc.test_rows:,} rows (seed {lc.test_seed})")
    results, ceiling = run_learning_curve(
        model_configs, lc.train_sizes, lc.test_rows, lc.repeats, lc.test_seed, lc.train_seed_start,
        quantiles, cfg.cross_validation.n_jobs,
    )
    results.to_csv(os.path.join(results_dir, "learning_curve.csv"), index=False)
    plot_learning_curve(results, ceiling, os.path.join(results_dir, "learning_curve.png"), PROJECT_DATASET_ROWS)

    stats = summarise(results)
    table = stats.pivot(index="train_size", columns="model", values="mean")[[m.name for m in model_configs]]
    spread = stats.pivot(index="train_size", columns="model", values="std")[[m.name for m in model_configs]]
    display = pd.DataFrame({"Training rows": [f"{s:,}" for s in table.index]})
    positives = results.groupby("train_size")["train_positives"].mean()
    display["Insolvencies (avg)"] = [int(round(positives[s])) for s in table.index]
    for model in table.columns:
        display[model] = [f"{m:.3f} +/- {sd:.3f}" for m, sd in zip(table[model], spread[model], strict=True)]

    linear = next((m.name for m in model_configs if m.model_type == "logistic_regression"), None)
    boosters = [m.name for m in model_configs if m.model_type != "logistic_regression"]
    findings = []
    if linear and boosters:
        for size in table.index:
            best = table.loc[size, boosters].idxmax()
            gap = table.loc[size, linear] - table.loc[size, best]
            findings.append(
                f"- {size:,} rows: best booster `{best}` {table.loc[size, best]:.3f} vs `{linear}` {table.loc[size, linear]:.3f} "
                f"(gap {gap:+.3f}); ceiling {ceiling['average_precision']:.3f}."
            )

    print("\n".join(findings))
    write_markdown(os.path.join(results_dir, "learning_curve.md"), [
        "# Learning curve",
        SYNTHETIC_DATA_NOTE,
        "## Question",
        "The main pipeline found that, with 119 training insolvencies, gradient boosting trails a plain logistic regression. "
        "Is that a property of the method or of the data size? Because the data is synthetic, fresh samples of any size can "
        "be drawn from the same generator to find out.",
        "## Design",
        f"- Training sizes: {', '.join(f'{s:,}' for s in lc.train_sizes)} rows; {lc.repeats} independent samples per size; "
        "every model sees the same sample.\n"
        f"- One independent test sample of {ceiling['test_rows']:,} rows ({ceiling['test_positives']:,} insolvencies), so "
        "test noise is negligible and the spread comes from the training samples.\n"
        "- Models use the same pipeline and configs as the tournament (tuned models run their nested search on each training sample).\n"
        f"- Ceiling: the true generating probabilities score AUC-PR {ceiling['average_precision']:.3f} "
        f"(ROC-AUC {ceiling['roc_auc']:.3f}) on the test sample.",
        "## AUC-PR on the independent test sample (mean +/- sd over samples)",
        markdown_table(display),
        "![Learning curve](learning_curve.png)",
        "## Findings",
        "\n".join(findings),
        "The data generator is itself a logistic model, so logistic regression is correctly specified here and is expected "
        "to approach the ceiling quickly. Boosting has to discover the same smooth relationships from splits, which needs more "
        "data; monotonic constraints and tuning reduce that cost. On real data, where relationships are rarely exactly "
        "linear in the log-odds, the balance can differ, which is why the tournament keeps both kinds of model.",
        "Full per-sample results: `results/learning_curve.csv`.",
    ])
    print("Wrote results/learning_curve.md, .csv and .png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "config.yaml"))
    main(parser.parse_args().config)
