"""End-to-end run: tournament, calibration, cost threshold, held-out evaluation, subgroups, drift and SHAP.

Order of operations (the final test set is untouched until step 6):
  1. Load and validate data, split into development (75%) and final test (25%) sets.
  2. Model tournament with repeated stratified CV on the development set.
  3. Calibration method chosen by cross-validated Brier score on out-of-fold predictions.
  4. Cost-minimising threshold chosen on calibrated out-of-fold predictions.
  5. Final model refitted on the full development set.
  6. Single evaluation on the test set, plus subgroup, drift and explainability reports.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.calibrate import ProbabilityCalibrator, plot_reliability_diagram, select_calibration_method
from src.config import ProjectConfig, load_model_config, load_project_config
from src.data_generation import oracle_probabilities
from src.drift import DataDriftMonitor
from src.evaluate import TabularEvaluator, classification_metrics
from src.export import ModelPersistence
from src.fairness import FairnessAuditor
from src.ingest import DataIngestor
from src.interpret import ExplanationUnavailable, ModelExplainer, grouped_importance, permutation_importance_table
from src.pipeline import RiskModel
from src.preprocess import TabularPreprocessor
from src.reporting import (
    SYNTHETIC_DATA_NOTE,
    ablation_section,
    final_comparison_section,
    fmt,
    fmt_ci,
    markdown_table,
    tournament_section,
    write_markdown,
)
from src.schema import CATEGORICAL_COLUMNS, ID_COLUMN, INPUT_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN
from src.threshold import CostSensitiveThresholdFinder, plot_cost_curves
from src.tournament import run_tournament
from src.train import build_model_pipeline, library_version
from src.tuning import fit_model

warnings.filterwarnings("ignore", category=UserWarning)


def file_sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def section(title: str) -> None:
    print(f"\n{title}")


def main(config_path: str, output_dir: str = BASE_DIR) -> Dict[str, Any]:
    cfg: ProjectConfig = load_project_config(config_path)
    seed = cfg.random_seed
    results_dir = os.path.join(output_dir, "results")
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(results_dir, exist_ok=True)
    quantiles = (cfg.preprocessing.outlier_lower_quantile, cfg.preprocessing.outlier_upper_quantile)
    evaluator = TabularEvaluator(cfg.evaluation.bootstrap_rounds, cfg.evaluation.confidence_level, seed)
    finder = CostSensitiveThresholdFinder(cfg.cost_matrix.false_positive_cost, cfg.cost_matrix.false_negative_cost)

    print("=" * 70)
    print(" Enterprise Default Radar - end-to-end run")
    print("=" * 70)

    # 1. Data -------------------------------------------------------------------------------
    csv_path = os.path.join(BASE_DIR, cfg.dataset.raw_csv_path)
    section(f"1. Loading and validating {cfg.dataset.raw_csv_path}")
    df = DataIngestor.load_and_validate(csv_path)
    dev_df, test_df = TabularPreprocessor(seed).stratified_split(df, TARGET_COLUMN, cfg.dataset.test_size)
    X_dev, y_dev = dev_df[INPUT_COLUMNS], dev_df[TARGET_COLUMN].to_numpy()
    X_test, y_test = test_df[INPUT_COLUMNS], test_df[TARGET_COLUMN].to_numpy()
    overlapping_ids = len(set(dev_df[ID_COLUMN]) & set(test_df[ID_COLUMN]))
    overlapping_rows = len(dev_df[INPUT_COLUMNS].merge(test_df[INPUT_COLUMNS], how="inner"))
    if overlapping_ids or overlapping_rows:
        raise RuntimeError(f"Leakage check failed: {overlapping_ids} shared IDs, {overlapping_rows} duplicated feature rows")
    print(f"   {len(df)} rows, positive rate {y_dev.mean() * 100:.2f}% (dev) / {y_test.mean() * 100:.2f}% (test)")
    print(f"   Development: {len(dev_df)} rows ({y_dev.sum()} positives) | Test: {len(test_df)} rows ({y_test.sum()} positives)")
    print("   Leakage checks passed: no shared IDs or duplicated feature rows across splits")

    # 2. Tournament -------------------------------------------------------------------------
    section(f"2. Model tournament ({cfg.cross_validation.n_splits}-fold CV x {cfg.cross_validation.n_repeats} repeats)")
    model_configs = [load_model_config(os.path.join(BASE_DIR, p)) for p in cfg.tournament.model_configs]
    tournament = run_tournament(
        X_dev, y_dev, model_configs,
        n_splits=cfg.cross_validation.n_splits,
        n_repeats=cfg.cross_validation.n_repeats,
        seed=seed,
        quantiles=quantiles,
        n_jobs=cfg.cross_validation.n_jobs,
    )
    tournament.fold_table().to_csv(os.path.join(results_dir, "model_tournament.csv"), index=False)
    tuning_trials = tournament.tuning_trials_table()
    if tuning_trials is not None:
        tuning_trials.to_csv(os.path.join(results_dir, "tuning_trials.csv"), index=False)
    summary = tournament.summary_table()
    dev_oracle = oracle_probabilities(dev_df)
    oracle_folds = pd.DataFrame([classification_metrics(y_dev[s.validation_index], dev_oracle[s.validation_index]) for s in tournament.splits])
    summary = pd.concat([summary, pd.DataFrame([{
        "model": "true probabilities (ceiling)",
        "role": "reference",
        **{f"{m}_mean": oracle_folds[m].mean() for m in ("average_precision", "roc_auc", "brier_score")},
        "average_precision_std": oracle_folds["average_precision"].std(ddof=1),
    }])], ignore_index=True)
    selected = tournament.candidates[tournament.selected_model]
    print(f"   Selected: {tournament.selected_model}")

    # 3. Calibration (development out-of-fold predictions only) ---------------------------------
    section("3. Choosing a calibration method on out-of-fold predictions")
    oof_raw = selected.oof_probabilities
    calibration_method, calibration_table, calibrated_oof = select_calibration_method(
        oof_raw, y_dev, cfg.calibration.methods, cfg.calibration.cv_folds, seed
    )
    oof_calibrated = calibrated_oof[calibration_method]
    print(f"   Selected calibration: {calibration_method} (CV Brier {calibration_table.loc[0, 'brier_score']:.4f})")

    # 4. Threshold (development out-of-fold predictions only) -----------------------------------
    section("4. Choosing the cost-minimising threshold on calibrated out-of-fold predictions")
    threshold_search = finder.find_optimal_threshold(
        y_dev, oof_calibrated, cfg.threshold.grid_steps, cfg.threshold.grid_min, cfg.threshold.grid_max
    )
    threshold = threshold_search["optimal_threshold"]
    print(f"   Selected threshold: {threshold:.2f} (cost-ratio reference FP/(FP+FN) = {finder.bayes_optimal_threshold:.3f})")

    # 5. Final model -------------------------------------------------------------------------
    section("5. Refitting the selected model on the full development set")
    final_pipeline, final_params, final_trials = fit_model(selected.model_config, X_dev, y_dev, quantiles, seed)
    if final_trials is not None:
        final_trials.to_csv(os.path.join(results_dir, "final_model_tuning_trials.csv"), index=False)
        print(f"   Tuned on all development data: best inner-CV AUC-PR {final_trials['value'].max():.4f} "
              f"({len(final_trials)} trials)")
    calibrator = ProbabilityCalibrator(calibration_method).fit(oof_raw, y_dev)

    # 6. Final test evaluation (first and only use of the test labels) -------------------------
    section("6. Evaluating once on the held-out test set")
    test_raw = final_pipeline.predict_proba(X_test)[:, 1]
    test_calibrated = calibrator.transform(test_raw)
    final_metrics = classification_metrics(y_test, test_calibrated, threshold)
    final_cis = evaluator.compute_bootstrap_cis(y_test, test_calibrated, threshold)
    print(f"   AUC-PR  {fmt_ci(final_metrics['average_precision'], final_cis['average_precision'])}")
    print(f"   ROC-AUC {fmt_ci(final_metrics['roc_auc'], final_cis['roc_auc'])}")
    print(f"   Brier   {fmt_ci(final_metrics['brier_score'], final_cis['brier_score'])}")

    comparison_rows = [("Selected: " + tournament.selected_model + f" ({calibration_method} calibration)", test_calibrated)]
    baseline_test = None
    if tournament.baseline_model and tournament.baseline_model != tournament.selected_model:
        baseline_config = tournament.candidates[tournament.baseline_model].model_config
        baseline_pipeline = build_model_pipeline(baseline_config, *quantiles).fit(X_dev, y_dev)
        baseline_test = baseline_pipeline.predict_proba(X_test)[:, 1]
        comparison_rows.append((f"Baseline: {tournament.baseline_model} (uncalibrated)", baseline_test))
    comparison_rows.append(("Reference: true data-generating probabilities", oracle_probabilities(test_df)))

    table_rows = []
    for label, probs in comparison_rows:
        point = classification_metrics(y_test, probs)
        cis = evaluator.compute_bootstrap_cis(y_test, probs)
        table_rows.append({
            "Model": label,
            "AUC-PR": fmt_ci(point["average_precision"], cis["average_precision"]),
            "ROC-AUC": fmt_ci(point["roc_auc"], cis["roc_auc"]),
            "Brier": fmt_ci(point["brier_score"], cis["brier_score"]),
        })
    difference = None
    if baseline_test is not None:
        from sklearn.metrics import average_precision_score

        paired = np.column_stack([test_calibrated, baseline_test])
        diff_ci = evaluator.bootstrap_cis(
            y_test, paired,
            lambda yt, pp: {"diff": average_precision_score(yt, pp[:, 0]) - average_precision_score(yt, pp[:, 1])},
        )["diff"]
        difference = {
            "point": average_precision_score(y_test, test_calibrated) - average_precision_score(y_test, baseline_test),
            "ci": diff_ci,
        }

    write_markdown(os.path.join(results_dir, "model_comparison.md"), [
        "# Model comparison",
        SYNTHETIC_DATA_NOTE,
        *tournament_section(summary, tournament, tournament.skipped),
        *ablation_section(tournament),
        *final_comparison_section(pd.DataFrame(table_rows), difference, len(y_test), int(y_test.sum())),
        "The reference row scores the test set with the probabilities used to generate the labels. "
        "It is the best any model could expect on average, so differences between models should be read against it.",
        "Fold-level results with versions, hyperparameters and seeds: `results/model_tournament.csv`.",
    ])

    # Threshold report ---------------------------------------------------------------------
    original_probs = baseline_test if baseline_test is not None else test_raw
    operating_points = [
        (f"Original baseline ({tournament.baseline_model or tournament.selected_model}), uncalibrated, 0.50", original_probs, 0.5),
        (f"Selected model ({tournament.selected_model}), uncalibrated, 0.50", test_raw, 0.5),
        ("Selected model, calibrated, 0.50", test_calibrated, 0.5),
        (f"Selected model, calibrated, selected threshold {threshold:.2f}", test_calibrated, threshold),
    ]
    op_rows = []
    for label, probs, t in operating_points:
        result = finder.evaluate_threshold(y_test, probs, t)
        op_rows.append({
            "Operating point": label,
            **{k: result[k] for k in ("tp", "fp", "fn", "tn", "flagged", "precision", "recall")},
            "total_cost ($)": int(result["total_cost"]),
        })
    op_table = pd.DataFrame(op_rows)

    def cost_saving(reference_probs: np.ndarray, reference_threshold: float):
        paired = np.column_stack([reference_probs, test_calibrated])
        statistic = lambda yt, pp: {"saving": finder.compute_cost(yt, (pp[:, 0] >= reference_threshold).astype(int))
                                    - finder.compute_cost(yt, (pp[:, 1] >= threshold).astype(int))}
        point = statistic(y_test, paired)["saving"]
        return point, evaluator.bootstrap_cis(y_test, paired, statistic)["saving"]

    saving_vs_raw = cost_saving(original_probs, 0.5)
    saving_vs_cal = cost_saving(test_calibrated, 0.5)
    test_curve = finder.cost_curve(y_test, test_calibrated, np.linspace(cfg.threshold.grid_min, cfg.threshold.grid_max, cfg.threshold.grid_steps))
    plot_cost_curves(
        {"Development out-of-fold (used to select)": threshold_search["curve"], "Final test set (reporting only)": test_curve},
        threshold, os.path.join(results_dir, "cost_threshold_curve.png"), finder.bayes_optimal_threshold,
    )
    write_markdown(os.path.join(results_dir, "threshold_report.md"), [
        "# Cost-sensitive threshold report",
        SYNTHETIC_DATA_NOTE,
        "## Cost assumptions",
        f"- False positive (unnecessary review of a healthy business): ${cfg.cost_matrix.false_positive_cost:.2f}\n"
        f"- False negative (missed business that becomes insolvent): ${cfg.cost_matrix.false_negative_cost:.2f}\n"
        "- These are illustrative assumptions, not measured costs. Change them in `config.yaml`.",
        "## How the threshold was chosen",
        f"- Thresholds from {cfg.threshold.grid_min} to {cfg.threshold.grid_max} ({cfg.threshold.grid_steps} steps) were scored on "
        f"cross-validated, calibrated out-of-fold predictions for the {len(y_dev)} development businesses.\n"
        f"- Selected threshold: **{threshold:.2f}** (development cost ${threshold_search['minimized_cost']:.0f} vs "
        f"${threshold_search['cost_at_default_0_5']:.0f} at 0.50).\n"
        f"- With perfectly calibrated probabilities the cost-minimising threshold is FP/(FP+FN) = {finder.bayes_optimal_threshold:.3f}; "
        "the data-driven choice is a sanity check against that value.\n"
        "- The test set was not used to choose the threshold.",
        f"## Held-out test results ({len(y_test)} businesses, {int(y_test.sum())} insolvent)",
        markdown_table(op_table),
        f"- Cost saving of the selected operating point vs the original baseline (uncalibrated, 0.50): "
        f"${saving_vs_raw[0]:.0f} (95% bootstrap CI ${saving_vs_raw[1][0]:.0f} to ${saving_vs_raw[1][1]:.0f}).\n"
        f"- Cost saving vs calibrated probabilities at 0.50: ${saving_vs_cal[0]:.0f} "
        f"(95% bootstrap CI ${saving_vs_cal[1][0]:.0f} to ${saving_vs_cal[1][1]:.0f}).",
        "![Cost versus threshold](cost_threshold_curve.png)",
        "## Trade-off",
        "Lowering the threshold flags more businesses: fewer insolvencies are missed (fewer false negatives) at the price of "
        "more unnecessary reviews (more false positives). Because a miss is assumed to cost five times a review, the selected "
        "threshold accepts many false positives. If reviews are expensive or capacity is limited, raise the false positive cost "
        "and rerun; the selected threshold will rise.",
        "## Limitations",
        "- The test set has few positives, so costs and recall move in large steps and the intervals are wide.\n"
        "- The cost curve is flat near its minimum; nearby thresholds give similar costs.\n"
        "- Real review costs, capacity limits and the cost of a missed insolvency vary by business and are not modelled.",
    ])

    # Calibration report -------------------------------------------------------------------
    test_calibration_rows = pd.DataFrame([
        {"predictions": "Uncalibrated", **ProbabilityCalibrator.evaluate_calibration(y_test, test_raw)},
        {"predictions": f"Calibrated ({calibration_method})", **ProbabilityCalibrator.evaluate_calibration(y_test, test_calibrated)},
    ])
    plot_reliability_diagram(
        {"Uncalibrated": (y_test, test_raw), f"Calibrated ({calibration_method})": (y_test, test_calibrated)},
        os.path.join(results_dir, "calibration_curve.png"),
    )
    write_markdown(os.path.join(results_dir, "calibration_report.md"), [
        "# Probability calibration report",
        SYNTHETIC_DATA_NOTE,
        f"The selected model (`{tournament.selected_model}`) is trained with balanced class weights, so its raw scores are not "
        f"probabilities. On the development set its out-of-fold scores average {np.mean(oof_raw):.3f} against an observed "
        f"insolvency rate of {np.mean(y_dev):.3f}, but the averages hide how the scores are spread: the reliability diagram "
        "below shows whether high scores are over- or under-confident. Calibration maps scores to probabilities that can be "
        "read as risk estimates and compared with the cost-based threshold.",
        "## Method selection (development set, cross-validated)",
        f"Each method was fitted and scored with {cfg.calibration.cv_folds}-fold CV on the out-of-fold predictions of the development set. "
        "Lowest Brier score wins.",
        markdown_table(calibration_table),
        f"**Selected:** `{calibration_method}`.",
        "## Held-out test set (reporting only)",
        markdown_table(test_calibration_rows),
        "![Reliability diagram](calibration_curve.png)",
        "## Limitations",
        "- With few test positives, each reliability bin holds only a handful of insolvencies, so the curve is noisy.\n"
        "- The calibrator is fitted on out-of-fold scores, while the deployed model is refitted on all development data; its "
        "score distribution can differ slightly.\n"
        "- Calibration changes probability values, not the ranking, so AUC-PR and ROC-AUC are unaffected.",
    ])

    # Subgroups ----------------------------------------------------------------------------
    section("7. Subgroup evaluation")
    test_pred = (test_calibrated >= threshold).astype(int)
    subgroup_sections, subgroup_tables = [], []
    for column in cfg.subgroups.columns:
        audit = FairnessAuditor.audit_subgroups(
            test_df, column, y_test, test_pred, test_calibrated, cfg.subgroups.min_group_size, cfg.subgroups.min_positives
        )
        subgroup_tables.append(audit["group_table"])
        print(f"   {column}: equalized odds diff {fmt(audit['equalized_odds_difference'])} over {len(audit['groups_compared'])} groups")
        subgroup_sections += [
            f"## By {column}",
            markdown_table(audit["group_table"].drop(columns="attribute")),
            f"- Groups compared (n >= {cfg.subgroups.min_group_size} and >= {cfg.subgroups.min_positives} positives): "
            f"{', '.join(audit['groups_compared']) or 'none'}\n"
            f"- Too small to compare: {', '.join(audit['groups_excluded_small']) or 'none'}\n"
            f"- Demographic parity difference (selection rate range): {fmt(audit['demographic_parity_difference'])}\n"
            f"- True positive rate range: {fmt(audit['true_positive_rate_difference'])}\n"
            f"- False positive rate range: {fmt(audit['false_positive_rate_difference'])}\n"
            f"- Equalized odds difference (max of TPR and FPR ranges): {fmt(audit['equalized_odds_difference'])}",
        ]
    pd.concat(subgroup_tables).to_csv(os.path.join(results_dir, "subgroup_metrics.csv"), index=False)
    write_markdown(os.path.join(results_dir, "subgroup_report.md"), [
        "# Subgroup evaluation",
        SYNTHETIC_DATA_NOTE,
        f"Held-out test set, selected threshold {threshold:.2f}. Rates within a group are based on very few insolvencies, "
        "so differences between groups are mostly sampling noise and should not be read as evidence of unfair treatment.",
        "By construction, `state` has **no effect** on the synthetic target, while `industry` does (Construction and "
        "Accommodation & Food carry higher risk). Differences in selection rate across industries therefore partly reflect "
        "real differences in prevalence. These patterns are properties of the generator, not of real populations.",
        *subgroup_sections,
    ])

    # Drift --------------------------------------------------------------------------------
    section("8. Drift monitoring")
    monitor = DataDriftMonitor(cfg.drift.psi_moderate, cfg.drift.psi_critical, cfg.drift.ks_alpha)
    rng = np.random.default_rng(seed)
    downturn = X_test.copy()
    downturn["profit_margin"] = downturn["profit_margin"] - 0.06
    downturn["debt_to_equity_ratio"] = downturn["debt_to_equity_ratio"] * 1.35
    downturn["current_ratio"] = downturn["current_ratio"] * 0.85
    quality_issue = X_test.copy().astype({"current_ratio": object})
    quality_issue.loc[rng.random(len(quality_issue)) < 0.15, "current_ratio"] = np.nan
    quality_issue.loc[rng.random(len(quality_issue)) < 0.05, "current_ratio"] = "n/a"
    quality_issue = quality_issue.drop(columns="r_and_d_expenditure_aud")
    scenarios = [
        ("Held-out test set (expected stable)", X_test),
        ("Simulated downturn: margins -6 pts, leverage x1.35, liquidity x0.85", downturn),
        ("Simulated data-quality issue: missing and malformed current_ratio, dropped R&D column", quality_issue),
    ]
    drift_sections = []
    for label, batch in scenarios:
        result = monitor.monitor_features(X_dev, batch, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)
        rows = [{"feature": col, **{k: m.get(k) for k in ("status", "psi", "ks_statistic", "ks_pvalue", "missing_rate", "malformed_rate")}}
                for col, m in result["feature_metrics"].items()]
        print(f"   {label[:60]:60s} drift detected: {result['drift_detected']}")
        drift_sections += [f"## {label}", f"Drift detected: **{fmt(result['drift_detected'])}**", markdown_table(pd.DataFrame(rows))]
        if batch is downturn:
            flag_rate = float(np.mean(calibrator.transform(final_pipeline.predict_proba(batch)[:, 1]) >= threshold))
            drift_sections.append(
                f"Share of businesses flagged: {np.mean(test_pred):.1%} on the original test set vs {flag_rate:.1%} on the shifted batch. "
                "Labels for the shifted batch are unknown, so whether accuracy changed cannot be measured here."
            )
    write_markdown(os.path.join(results_dir, "drift_report.md"), [
        "# Data drift report",
        SYNTHETIC_DATA_NOTE,
        "Reference distribution: development set. Population Stability Index (PSI) uses ten reference quantile bins with open "
        "outer bins; categorical PSI uses category frequencies. The KS test is reported for numeric features.",
        f"- PSI <= {cfg.drift.psi_moderate}: stable; {cfg.drift.psi_moderate}-{cfg.drift.psi_critical}: moderate drift; "
        f"> {cfg.drift.psi_critical}: critical drift (common industry rules of thumb, not statistical tests).\n"
        f"- KS p-value < {cfg.drift.ks_alpha} marks a statistically detectable shift; with large batches tiny, harmless shifts become significant.\n"
        "- Missing features, missing values and malformed (non-numeric) entries are reported separately.\n"
        "- A drift alert is a reason to investigate the data, not proof that model performance has dropped.",
        *drift_sections,
    ])

    # Explainability -----------------------------------------------------------------------
    section("9. Explainability")
    explain_sample = X_test.sample(min(cfg.interpretability.explain_sample_size, len(X_test)), random_state=seed)
    try:
        explainer = ModelExplainer(final_pipeline, X_dev, cfg.interpretability.background_size, seed)
        importance = explainer.global_importance(explain_sample)
        importance.to_csv(os.path.join(results_dir, "feature_importance.csv"), index=False)
        explainer.plot_summary(explain_sample, os.path.join(results_dir, "shap_summary.png"))
        grouped = grouped_importance(importance)
        # Inputs the synthetic generator ignores; ratios built only from them carry no signal either.
        no_effect_inputs = {"state", "employee_count", "annual_turnover_aud", "r_and_d_expenditure_aud",
                            "log_turnover", "revenue_per_employee", "r_and_d_intensity"}
        noise_share = grouped.loc[grouped["source_column"].isin(no_effect_inputs), "mean_abs_shap"].sum() / grouped["mean_abs_shap"].sum()

        order = np.argsort(test_calibrated)
        examples = [
            ("Highest predicted risk", order[-1]),
            ("Closest to the decision threshold", int(np.argmin(np.abs(test_calibrated - threshold)))),
            ("Lowest predicted risk", order[0]),
        ]
        example_sections = []
        for label, idx in examples:
            record = X_test.iloc[[idx]]
            contributions, base = explainer.explain_record(record, top_n=6)
            example_sections += [
                f"## {label}",
                f"Calibrated probability {test_calibrated[idx]:.3f} (threshold {threshold:.2f}); actual outcome: "
                f"{'insolvent' if y_test[idx] else 'not insolvent'}. Model base value {base:.3f} log-odds.",
                "Inputs: " + ", ".join(f"{c}={record.iloc[0][c]}" for c in INPUT_COLUMNS),
                markdown_table(contributions, digits=3),
            ]
        write_markdown(os.path.join(results_dir, "example_explanations.md"), [
            "# Example prediction explanations",
            SYNTHETIC_DATA_NOTE,
            f"SHAP values ({explainer.explainer_type} explainer) for three test businesses, top six contributions each. Positive "
            "values push the uncalibrated model's log-odds towards insolvency. They describe how the model uses its inputs and "
            "do not establish causes of insolvency.",
            "## Global importance by input column (mean |SHAP|, one-hot columns summed)",
            markdown_table(grouped),
            f"**Sanity check against the known generator.** {noise_share:.0%} of total attribution goes to inputs that have no "
            "effect on the synthetic target (state, employee count, turnover, R&D spend and ratios built only from them). "
            "A model that had learned only the true signal would put close to 0% there; a large share means the model is "
            "partly fitting noise in a small training set (119 insolvencies), which is consistent with the linear reference "
            "model outperforming the boosters in `model_comparison.md`. The true drivers are leverage, liquidity, profit "
            "margin, years operating and industry.",
            "![SHAP summary](shap_summary.png)",
            *example_sections,
        ])
        print(f"   Top features: {', '.join(importance['feature'].head(5))}")
    except ExplanationUnavailable as error:
        print(f"   SHAP unavailable ({error}); writing permutation importance instead")
        permutation_importance_table(final_pipeline, X_test, y_test, seed).to_csv(
            os.path.join(results_dir, "feature_importance.csv"), index=False
        )

    # Persist ------------------------------------------------------------------------------
    section("10. Saving model and run metadata")
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": library_version("hist_gradient_boosting"),
        **{name: library_version(name) for name in ("xgboost", "lightgbm", "catboost")},
    }
    metadata = {
        "selected_model": tournament.selected_model,
        "model_type": selected.model_config.model_type,
        "model_params": final_params,
        "monotonic_constraints": selected.model_config.monotonic_constraints,
        "calibration_method": calibration_method,
        "decision_threshold": threshold,
        "cost_assumptions": cfg.cost_matrix.model_dump(),
        "test_metrics": {k: {"value": final_metrics[k], "ci95": final_cis[k]} for k in final_metrics},
        "test_rows": len(y_test),
        "test_positives": int(y_test.sum()),
        "data_sha256": file_sha256(csv_path),
        "versions": versions,
        "random_seed": seed,
        "trained_on": "synthetic data only",
    }
    training_ranges = {col: [float(X_dev[col].min()), float(X_dev[col].max())] for col in NUMERIC_COLUMNS}
    risk_model = RiskModel(
        final_pipeline, calibrator, threshold, training_ranges,
        X_dev.sample(cfg.interpretability.background_size, random_state=seed).reset_index(drop=True), metadata,
    )
    model_path = os.path.join(models_dir, "risk_model.joblib")
    ModelPersistence.save_pipeline(risk_model, model_path)
    with open(os.path.join(results_dir, "run_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump({**metadata, "config": cfg.model_dump()}, handle, indent=2, default=str)
    print(f"   Model: {model_path}")
    print(f"   Reports: {results_dir}")
    print("\nDone.")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "config.yaml"), help="Path to the project config file")
    parser.add_argument("--output-dir", default=BASE_DIR, help="Folder that receives results/ and models/ (default: project root)")
    args = parser.parse_args()
    main(args.config, args.output_dir)
