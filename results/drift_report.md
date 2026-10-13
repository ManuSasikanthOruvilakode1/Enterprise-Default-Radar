# Data drift report

> **Synthetic data.** All results come from a synthetic dataset whose target was drawn from a known logistic model. They demonstrate the methodology only and say nothing about performance on real businesses.

Reference distribution: development set. Population Stability Index (PSI) uses ten reference quantile bins with open outer bins; categorical PSI uses category frequencies. The KS test is reported for numeric features.

- PSI <= 0.1: stable; 0.1-0.25: moderate drift; > 0.25: critical drift (common industry rules of thumb, not statistical tests).
- KS p-value < 0.01 marks a statistically detectable shift; with large batches tiny, harmless shifts become significant.
- Missing features, missing values and malformed (non-numeric) entries are reported separately.
- A drift alert is a reason to investigate the data, not proof that model performance has dropped.

## Held-out test set (expected stable)

Drift detected: **no**

| feature | status | psi | ks_statistic | ks_pvalue | missing_rate | malformed_rate |
| --- | --- | --- | --- | --- | --- | --- |
| employee_count | STABLE | 0.0111 | 0.0256 | 0.9147 | 0.0000 | 0.0000 |
| annual_turnover_aud | STABLE | 0.0335 | 0.0459 | 0.2734 | 0.0000 | 0.0000 |
| years_operating | STABLE | 0.0151 | 0.0485 | 0.2160 | 0.0000 | 0.0000 |
| r_and_d_expenditure_aud | STABLE | 0.0139 | 0.0325 | 0.6978 | 0.0000 | 0.0000 |
| debt_to_equity_ratio | STABLE | 0.0125 | 0.0373 | 0.5247 | 0.0000 | 0.0000 |
| current_ratio | STABLE | 0.0305 | 0.0608 | 0.0612 | 0.0000 | 0.0000 |
| profit_margin | STABLE | 0.0256 | 0.0299 | 0.7916 | 0.0000 | 0.0000 |
| state | STABLE | 0.0099 | n/a | n/a | 0.0000 | n/a |
| industry | STABLE | 0.0091 | n/a | n/a | 0.0000 | n/a |

## Simulated downturn: margins -6 pts, leverage x1.35, liquidity x0.85

Drift detected: **yes**

| feature | status | psi | ks_statistic | ks_pvalue | missing_rate | malformed_rate |
| --- | --- | --- | --- | --- | --- | --- |
| employee_count | STABLE | 0.0111 | 0.0256 | 0.9147 | 0.0000 | 0.0000 |
| annual_turnover_aud | STABLE | 0.0335 | 0.0459 | 0.2734 | 0.0000 | 0.0000 |
| years_operating | STABLE | 0.0151 | 0.0485 | 0.2160 | 0.0000 | 0.0000 |
| r_and_d_expenditure_aud | STABLE | 0.0139 | 0.0325 | 0.6978 | 0.0000 | 0.0000 |
| debt_to_equity_ratio | MODERATE_DRIFT | 0.1768 | 0.1808 | 0.0000 | 0.0000 | 0.0000 |
| current_ratio | CRITICAL_DRIFT | 0.4029 | 0.2256 | 0.0000 | 0.0000 | 0.0000 |
| profit_margin | CRITICAL_DRIFT | 0.3563 | 0.2491 | 0.0000 | 0.0000 | 0.0000 |
| state | STABLE | 0.0099 | n/a | n/a | 0.0000 | n/a |
| industry | STABLE | 0.0091 | n/a | n/a | 0.0000 | n/a |

Share of businesses flagged: 12.2% on the original test set vs 29.0% on the shifted batch. Labels for the shifted batch are unknown, so whether accuracy changed cannot be measured here.

## Simulated data-quality issue: missing and malformed current_ratio, dropped R&D column

Drift detected: **yes**

| feature | status | psi | ks_statistic | ks_pvalue | missing_rate | malformed_rate |
| --- | --- | --- | --- | --- | --- | --- |
| employee_count | STABLE | 0.0111 | 0.0256 | 0.9147 | 0.0000 | 0.0000 |
| annual_turnover_aud | STABLE | 0.0335 | 0.0459 | 0.2734 | 0.0000 | 0.0000 |
| years_operating | STABLE | 0.0151 | 0.0485 | 0.2160 | 0.0000 | 0.0000 |
| r_and_d_expenditure_aud | MISSING_FEATURE | n/a | n/a | n/a | n/a | n/a |
| debt_to_equity_ratio | STABLE | 0.0125 | 0.0373 | 0.5247 | 0.0000 | 0.0000 |
| current_ratio | STABLE | 0.0381 | 0.0681 | 0.0515 | 0.1616 | 0.0544 |
| profit_margin | STABLE | 0.0256 | 0.0299 | 0.7916 | 0.0000 | 0.0000 |
| state | STABLE | 0.0099 | n/a | n/a | 0.0000 | n/a |
| industry | STABLE | 0.0091 | n/a | n/a | 0.0000 | n/a |
