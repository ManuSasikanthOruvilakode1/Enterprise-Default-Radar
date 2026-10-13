"""Markdown report writers. Every number written here comes from the current pipeline run."""

import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

SYNTHETIC_DATA_NOTE = (
    "> **Synthetic data.** All results come from a synthetic dataset whose target was drawn from a known "
    "logistic model. They demonstrate the methodology only and say nothing about performance on real businesses."
)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_ci(point: float, interval: Tuple[float, float], digits: int = 4) -> str:
    return f"{point:.{digits}f} [{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v, digits) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider] + rows)


def write_markdown(path: str, sections: Iterable[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(sections).rstrip() + "\n")


def tournament_section(summary: pd.DataFrame, tournament: Any, skipped: Dict[str, str]) -> List[str]:
    columns = {
        "model": "Model",
        "role": "Role",
        "average_precision_mean": "AUC-PR mean",
        "average_precision_std": "AUC-PR std",
        "roc_auc_mean": "ROC-AUC mean",
        "brier_score_mean": "Brier mean",
        "recall_mean": "Recall@0.5",
        "precision_mean": "Precision@0.5",
        "balanced_accuracy_mean": "Bal. acc.@0.5",
        "folds_better_than_baseline": "Folds > baseline (AUC-PR)",
        "runtime_seconds": "Runtime (s)",
        "library_version": "Version",
    }
    summary = summary.copy()
    summary["runtime_seconds"] = summary["runtime_seconds"].map(lambda s: "" if pd.isna(s) else f"{s:.1f}")
    if "folds_better_than_baseline" in summary.columns:
        summary["folds_better_than_baseline"] = summary["folds_better_than_baseline"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    summary["library_version"] = summary["library_version"].fillna("")
    table = summary[[c for c in columns if c in summary.columns]].rename(columns=columns)
    n_folds = tournament.n_splits * tournament.n_repeats
    sections = [
        "## Cross-validated tournament (development set only)",
        f"- Design: {tournament.n_splits}-fold stratified CV repeated {tournament.n_repeats} times "
        f"({n_folds} folds, seed {tournament.seed}); identical splits for every model.\n"
        "- Outlier capping and feature engineering are refitted inside each training fold.\n"
        "- Primary metric: average precision (AUC-PR), because positives are rare (~6%).\n"
        "- Threshold-dependent metrics use 0.5 on uncalibrated, class-weighted scores and are shown for context only.\n"
        "- Reference rows are reported for context and cannot be selected. The last row scores each fold with the true "
        "probabilities used to generate the labels: the realistic ceiling for this data.",
        markdown_table(table),
    ]
    if skipped:
        sections.append("Skipped models: " + ", ".join(f"`{k}` ({v})" for k, v in skipped.items()))
    sections.append(f"**Selected model:** `{tournament.selected_model}` (highest mean AUC-PR among selectable models).")
    return sections


def ablation_section(tournament: Any) -> List[str]:
    """Effect of monotonic constraints and nested tuning, compared fold by fold with each library's defaults."""
    rows = []
    for base_name, base in tournament.candidates.items():
        variants = [n for n in (f"{base_name}_monotonic", f"{base_name}_tuned") if n in tournament.candidates]
        if not variants:
            continue
        base_scores = base.fold_metrics["average_precision"].to_numpy()
        for name in [base_name] + variants:
            result = tournament.candidates[name]
            scores = result.fold_metrics["average_precision"].to_numpy()
            config = result.model_config
            rows.append({
                "Model": name,
                "Monotonic": bool(config.monotonic_constraints),
                "Tuned (nested)": bool(config.tuning),
                "AUC-PR mean": float(scores.mean()),
                "Change vs defaults": float(scores.mean() - base_scores.mean()),
                "Folds better than defaults": "" if name == base_name else f"{int((scores > base_scores).sum())}/{len(scores)}",
                "Brier mean": float(result.fold_metrics["brier_score"].mean()),
            })
    if not rows:
        return []
    tuned = [r.model_config for r in tournament.candidates.values() if r.model_config.tuning]
    design = ""
    if tuned:
        t = tuned[0].tuning
        design = (
            f"Tuned models run an Optuna TPE search ({t.n_trials} trials, median pruning) scored by {t.inner_folds}-fold "
            "AUC-PR on the outer training fold only, so each outer validation fold is scored by a model whose "
            "hyperparameters never saw it. Every trial is recorded in `results/tuning_trials.csv`; the parameters "
            "chosen in each fold are in the `params` column of `results/model_tournament.csv`."
        )
    return [
        "## Do monotonic constraints and tuning help?",
        "Monotonic constraints encode domain knowledge: predicted risk may only rise with debt-to-equity and the "
        "leverage/liquidity ratio, and only fall with current ratio, profit margin and years operating. "
        + design,
        markdown_table(pd.DataFrame(rows)),
    ]


def final_comparison_section(rows: pd.DataFrame, difference: Optional[Dict[str, Any]], n_test: int, n_positives: int) -> List[str]:
    sections = [
        "## Final held-out test set",
        f"Scored once, after model, calibration and threshold choices were fixed. "
        f"{n_test} businesses, {n_positives} insolvent. 95% percentile bootstrap intervals in brackets.",
        markdown_table(rows),
    ]
    if difference:
        sections.append(
            f"Paired bootstrap difference in AUC-PR (selected minus baseline): "
            f"{fmt_ci(difference['point'], difference['ci'])}. "
            + ("The interval includes zero, so the test set cannot distinguish the two models."
               if difference["ci"][0] <= 0 <= difference["ci"][1]
               else "The interval excludes zero.")
        )
    return sections
