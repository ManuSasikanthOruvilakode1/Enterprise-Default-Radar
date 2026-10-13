# Learning curve

> **Synthetic data.** All results come from a synthetic dataset whose target was drawn from a known logistic model. They demonstrate the methodology only and say nothing about performance on real businesses.

## Question

The main pipeline found that, with 119 training insolvencies, gradient boosting trails a plain logistic regression. Is that a property of the method or of the data size? Because the data is synthetic, fresh samples of any size can be drawn from the same generator to find out.

## Design

- Training sizes: 500, 1,000, 2,500, 10,000, 50,000 rows; 5 independent samples per size; every model sees the same sample.
- One independent test sample of 100,000 rows (6,206 insolvencies), so test noise is negligible and the spread comes from the training samples.
- Models use the same pipeline and configs as the tournament (tuned models run their nested search on each training sample).
- Ceiling: the true generating probabilities score AUC-PR 0.232 (ROC-AUC 0.765) on the test sample.

## AUC-PR on the independent test sample (mean +/- sd over samples)

| Training rows | Insolvencies (avg) | logistic_regression | lightgbm | lightgbm_monotonic | lightgbm_tuned |
| --- | --- | --- | --- | --- | --- |
| 500 | 28 | 0.161 +/- 0.011 | 0.145 +/- 0.016 | 0.152 +/- 0.017 | 0.145 +/- 0.016 |
| 1,000 | 60 | 0.162 +/- 0.014 | 0.137 +/- 0.019 | 0.150 +/- 0.010 | 0.154 +/- 0.020 |
| 2,500 | 165 | 0.209 +/- 0.003 | 0.161 +/- 0.007 | 0.168 +/- 0.004 | 0.199 +/- 0.004 |
| 10,000 | 617 | 0.223 +/- 0.002 | 0.192 +/- 0.006 | 0.201 +/- 0.004 | 0.212 +/- 0.005 |
| 50,000 | 3097 | 0.229 +/- 0.001 | 0.221 +/- 0.001 | 0.223 +/- 0.001 | 0.225 +/- 0.001 |

![Learning curve](learning_curve.png)

## Findings

- 500 rows: best booster `lightgbm_monotonic` 0.152 vs `logistic_regression` 0.161 (gap +0.009); ceiling 0.232.
- 1,000 rows: best booster `lightgbm_tuned` 0.154 vs `logistic_regression` 0.162 (gap +0.008); ceiling 0.232.
- 2,500 rows: best booster `lightgbm_tuned` 0.199 vs `logistic_regression` 0.209 (gap +0.011); ceiling 0.232.
- 10,000 rows: best booster `lightgbm_tuned` 0.212 vs `logistic_regression` 0.223 (gap +0.011); ceiling 0.232.
- 50,000 rows: best booster `lightgbm_tuned` 0.225 vs `logistic_regression` 0.229 (gap +0.004); ceiling 0.232.

The data generator is itself a logistic model, so logistic regression is correctly specified here and is expected to approach the ceiling quickly. Boosting has to discover the same smooth relationships from splits, which needs more data; monotonic constraints and tuning reduce that cost. On real data, where relationships are rarely exactly linear in the log-odds, the balance can differ, which is why the tournament keeps both kinds of model.

Full per-sample results: `results/learning_curve.csv`.
