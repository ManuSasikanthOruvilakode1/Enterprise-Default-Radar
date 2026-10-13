# Model comparison

> **Synthetic data.** All results come from a synthetic dataset whose target was drawn from a known logistic model. They demonstrate the methodology only and say nothing about performance on real businesses.

## Cross-validated tournament (development set only)

- Design: 5-fold stratified CV repeated 3 times (15 folds, seed 42); identical splits for every model.
- Outlier capping and feature engineering are refitted inside each training fold.
- Primary metric: average precision (AUC-PR), because positives are rare (~6%).
- Threshold-dependent metrics use 0.5 on uncalibrated, class-weighted scores and are shown for context only.
- Reference rows are reported for context and cannot be selected. The last row scores each fold with the true probabilities used to generate the labels: the realistic ceiling for this data.

| Model | Role | AUC-PR mean | AUC-PR std | ROC-AUC mean | Brier mean | Recall@0.5 | Precision@0.5 | Bal. acc.@0.5 | Folds > baseline (AUC-PR) | Runtime (s) | Version |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | reference | 0.2877 | 0.0932 | 0.7551 | 0.1821 | 0.6424 | 0.1426 | 0.6903 | 14 | 0.1 | 1.9.1 |
| lightgbm_tuned | candidate | 0.2429 | 0.0681 | 0.7410 | 0.1532 | 0.5256 | 0.1551 | 0.6676 | 12 | 21.6 | 4.7.0 |
| xgboost_tuned | candidate | 0.2398 | 0.0885 | 0.7373 | 0.0752 | 0.1310 | 0.2328 | 0.5492 | 12 | 16.0 | 3.4.1 |
| xgboost_monotonic | candidate | 0.2254 | 0.0583 | 0.7168 | 0.0645 | 0.1682 | 0.3223 | 0.5704 | 10 | 0.8 | 3.4.1 |
| lightgbm_monotonic | candidate | 0.2237 | 0.0665 | 0.7176 | 0.0749 | 0.2580 | 0.2581 | 0.6032 | 10 | 0.3 | 4.7.0 |
| lightgbm | candidate | 0.1972 | 0.0527 | 0.6861 | 0.0636 | 0.1627 | 0.3711 | 0.5703 | 8 | 0.3 | 4.7.0 |
| hist_gradient_boosting | baseline | 0.1934 | 0.0667 | 0.6780 | 0.1092 | 0.3089 | 0.1602 | 0.6003 | 0 | 3.8 | 1.9.1 |
| xgboost | candidate | 0.1897 | 0.0483 | 0.6842 | 0.0614 | 0.1205 | 0.4100 | 0.5531 | 8 | 2.6 | 3.4.1 |
| catboost | candidate | 0.1716 | 0.0389 | 0.6724 | 0.0727 | 0.1738 | 0.2488 | 0.5682 | 7 | 2.1 | 1.2.10 |
| true probabilities (ceiling) | reference | 0.2980 | 0.1113 | 0.7715 | 0.0528 | n/a | n/a | n/a |  |  |  |

**Selected model:** `lightgbm_tuned` (highest mean AUC-PR among selectable models).

## Do monotonic constraints and tuning help?

Monotonic constraints encode domain knowledge: predicted risk may only rise with debt-to-equity and the leverage/liquidity ratio, and only fall with current ratio, profit margin and years operating. Tuned models run an Optuna TPE search (30 trials, median pruning) scored by 3-fold AUC-PR on the outer training fold only, so each outer validation fold is scored by a model whose hyperparameters never saw it. Every trial is recorded in `results/tuning_trials.csv`; the parameters chosen in each fold are in the `params` column of `results/model_tournament.csv`.

| Model | Monotonic | Tuned (nested) | AUC-PR mean | Change vs defaults | Folds better than defaults | Brier mean |
| --- | --- | --- | --- | --- | --- | --- |
| xgboost | no | no | 0.1897 | 0.0000 |  | 0.0614 |
| xgboost_monotonic | yes | no | 0.2254 | 0.0357 | 12/15 | 0.0645 |
| xgboost_tuned | yes | yes | 0.2398 | 0.0501 | 15/15 | 0.0752 |
| lightgbm | no | no | 0.1972 | 0.0000 |  | 0.0636 |
| lightgbm_monotonic | yes | no | 0.2237 | 0.0265 | 11/15 | 0.0749 |
| lightgbm_tuned | yes | yes | 0.2429 | 0.0456 | 13/15 | 0.1532 |

## Final held-out test set

Scored once, after model, calibration and threshold choices were fixed. 625 businesses, 40 insolvent. 95% percentile bootstrap intervals in brackets.

| Model | AUC-PR | ROC-AUC | Brier |
| --- | --- | --- | --- |
| Selected: lightgbm_tuned (sigmoid calibration) | 0.2193 [0.1264, 0.3659] | 0.7362 [0.6447, 0.8209] | 0.0565 [0.0430, 0.0715] |
| Baseline: hist_gradient_boosting (uncalibrated) | 0.2242 [0.1248, 0.3798] | 0.7267 [0.6466, 0.8103] | 0.1209 [0.1064, 0.1353] |
| Reference: true data-generating probabilities | 0.2349 [0.1347, 0.3862] | 0.7535 [0.6787, 0.8292] | 0.0552 [0.0413, 0.0706] |

Paired bootstrap difference in AUC-PR (selected minus baseline): -0.0048 [-0.0729, 0.0490]. The interval includes zero, so the test set cannot distinguish the two models.

The reference row scores the test set with the probabilities used to generate the labels. It is the best any model could expect on average, so differences between models should be read against it.

Fold-level results with versions, hyperparameters and seeds: `results/model_tournament.csv`.
