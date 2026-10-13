# Baseline verification

This records how the original baseline was checked before any changes were made, what was found, and what was changed as a result.

## 1. Reproducing the original baseline

The original `scripts/run_pipeline.py` was run unchanged in a clean `uv` environment (Python 3.12). Every documented figure was reproduced exactly:

| Metric | Documented | Reproduced |
| --- | --- | --- |
| Cost at threshold 0.50 | $200.00 | $200.00 |
| Cost at threshold 0.8029 | $163.00 | $163.00 |
| Cost reduction | $37.00 (18.5%) | $37.00 (18.5%) |
| AUC-ROC | 0.7291 [0.6499, 0.8092] | 0.7291 [0.6499, 0.8092] |
| AUC-PR | 0.2089 [0.1237, 0.3604] | 0.2089 [0.1237, 0.3604] |
| Brier score | 0.1192 | 0.1192 |
| Balanced accuracy | 61.4% | 61.4% |
| Demographic parity difference (state) | 0.0794 | 0.0794 |
| Equalized odds difference (state) | 0.0667 | 0.0667 |
| PSI, all numeric features | < 0.04 | 0.0138 to 0.0398 |
| Unit tests | 6 passing | 6 passing |

The original report is archived in [`results/baseline/original_report.md`](../results/baseline/original_report.md). The data generator reproduces the CSV exactly (verified by byte comparison).

Reproducing a number does not make it valid. The checks below explain why several of these figures should not be reported.

## 2. Data checks

| Check | Result |
| --- | --- |
| Rows / columns | 2,500 rows, 11 columns, no missing values |
| Target rate | 6.36% (159 insolvent). The generator comment claimed ~15%; corrected. |
| Duplicate IDs / duplicate feature rows | None |
| Split | Stratified 75/25: development 1,875 rows (119 positives), test 625 rows (40 positives) |
| Overlap between splits | No shared IDs or identical feature rows |
| Target leakage | None. Every feature is a firm characteristic used by the generator before the label is drawn; no feature is derived from the label. |
| Smallest groups in test set | NT: 7 rows, 0 positives; ACT: 15 rows, 0 positives; Mining: 32 rows, 0 positives |

**Known ground truth.** The target is drawn from a logistic model of debt-to-equity, current ratio, profit margin, years operating and two industry effects ([`src/data_generation.py`](../src/data_generation.py)). State, employee count, turnover and R&D spend have no effect. Scoring the test set with these true probabilities gives AUC-ROC 0.754 and AUC-PR 0.235, the realistic ceiling for this test set. The baseline's 0.729 AUC-ROC was already close to it.

## 3. Issues found

| # | Issue | Impact | Fix |
| --- | --- | --- | --- |
| 1 | **Decision threshold tuned on the test set** (`find_optimal_threshold(y_test, ...)`) | The 0.8029 threshold and the $37 / 18.5% saving are in-sample. They overstate what the threshold would achieve on new data. | Threshold chosen on cross-validated out-of-fold predictions from the development set; the test set is only scored once. |
| 2 | Outlier caps learned on the training set but never applied to the test set | Train and test went through different preprocessing | `OutlierCapper` transformer learns caps on training folds and applies the same caps everywhere |
| 3 | Preprocessing not inside a cross-validation loop | Cross-validation would leak preprocessing statistics | Capping, feature engineering and scaling are steps of a scikit-learn `Pipeline` refitted per fold |
| 4 | "Equalized odds difference" used only the false positive rate | Understated the metric; groups without positives were treated as TPR = 0 | Max of the TPR range and FPR range; undefined rates are reported as n/a; groups below a minimum size are excluded from disparity summaries |
| 5 | Calibration module existed but was never used; it relied on the removed `cv="prefit"` API | Class-weighted scores reported as probabilities (Brier 0.119) | Platt and isotonic calibration selected by cross-validated Brier on development predictions |
| 6 | PSI dropped current values outside the reference range | Strong shifts could be under-reported | Open-ended outer bins; missing, malformed and missing-column cases handled |
| 7 | XGBoost, LightGBM and CatBoost configs existed but no code used them | Documented tournament did not exist | `src/tournament.py` |
| 8 | README described files and tools that do not exist (ONNX export, MLflow, notebooks, Evidently, `download_data.sh`) | Misleading documentation | README rewritten from the actual code and results |
| 9 | Pickled model referenced the training class layout and a stub FastAPI app with only `/health` | Not a usable inference path | `RiskModel` artifact saved with `joblib`, reused by the Streamlit app |

## 4. Reproducible commands

```bash
uv sync --all-extras
uv run python scripts/generate_synthetic_business_data.py   # optional: recreates data/raw/business_outcomes.csv
uv run python scripts/run_pipeline.py
uv run pytest tests
```

## 5. Baseline after the fixes

With the leakage-free design, the original `HistGradientBoostingClassifier` configuration is the `hist_gradient_boosting` baseline in the tournament. Its corrected figures are in [`results/model_comparison.md`](../results/model_comparison.md). The threshold comparison on untouched test predictions is in [`results/threshold_report.md`](../results/threshold_report.md).
