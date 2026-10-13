# Cost-sensitive threshold report

> **Synthetic data.** All results come from a synthetic dataset whose target was drawn from a known logistic model. They demonstrate the methodology only and say nothing about performance on real businesses.

## Cost assumptions

- False positive (unnecessary review of a healthy business): $1.00
- False negative (missed business that becomes insolvent): $5.00
- These are illustrative assumptions, not measured costs. Change them in `config.yaml`.

## How the threshold was chosen

- Thresholds from 0.01 to 0.99 (99 steps) were scored on cross-validated, calibrated out-of-fold predictions for the 1875 development businesses.
- Selected threshold: **0.19** (development cost $518 vs $596 at 0.50).
- With perfectly calibrated probabilities the cost-minimising threshold is FP/(FP+FN) = 0.167; the data-driven choice is a sanity check against that value.
- The test set was not used to choose the threshold.

## Held-out test results (625 businesses, 40 insolvent)

| Operating point | tp | fp | fn | tn | flagged | precision | recall | total_cost ($) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original baseline (hist_gradient_boosting), uncalibrated, 0.50 | 16 | 76 | 24 | 509 | 92 | 0.1739 | 0.4000 | 196 |
| Selected model (lightgbm_tuned), uncalibrated, 0.50 | 21 | 129 | 19 | 456 | 150 | 0.1400 | 0.5250 | 224 |
| Selected model, calibrated, 0.50 | 0 | 0 | 40 | 585 | 0 | 0.0000 | 0.0000 | 200 |
| Selected model, calibrated, selected threshold 0.19 | 15 | 61 | 25 | 524 | 76 | 0.1974 | 0.3750 | 186 |

- Cost saving of the selected operating point vs the original baseline (uncalibrated, 0.50): $10 (95% bootstrap CI $-17 to $37).
- Cost saving vs calibrated probabilities at 0.50: $14 (95% bootstrap CI $-25 to $60).

![Cost versus threshold](cost_threshold_curve.png)

## Trade-off

Lowering the threshold flags more businesses: fewer insolvencies are missed (fewer false negatives) at the price of more unnecessary reviews (more false positives). Because a miss is assumed to cost five times a review, the selected threshold accepts many false positives. If reviews are expensive or capacity is limited, raise the false positive cost and rerun; the selected threshold will rise.

## Limitations

- The test set has few positives, so costs and recall move in large steps and the intervals are wide.
- The cost curve is flat near its minimum; nearby thresholds give similar costs.
- Real review costs, capacity limits and the cost of a missed insolvency vary by business and are not modelled.
