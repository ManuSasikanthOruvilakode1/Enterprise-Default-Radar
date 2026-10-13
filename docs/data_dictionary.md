# Data dictionary

File: `data/raw/business_outcomes.csv`, created by `scripts/generate_synthetic_business_data.py` (seed 42).

> **The dataset is synthetic.** Its marginal distributions are loosely shaped after public economic statistics (state and industry mix, firm size, financial ratios), but every row is generated. It does not contain real company records, and results on it do not validate performance on real businesses.

## Target

| Column | Type | Definition |
| --- | --- | --- |
| `insolvency_risk` | int (0/1) | 1 if the business is labelled insolvent. Drawn as Bernoulli(p), where p = sigmoid(z) and z = -2.8 + 0.65 (debt_to_equity - 1.5) - 0.90 (current_ratio - 1.5) - 2.50 profit_margin - 0.03 years_operating + 0.45 [Construction] + 0.35 [Accommodation & Food]. Positive rate 6.36% (159 of 2,500). |

Because labels are drawn randomly from p, even a perfect model of p cannot predict every outcome. The true probabilities give a test-set AUC-ROC of about 0.75; that is the realistic ceiling.

## Features

| Column | Type | Range in data | Generation | Effect on target |
| --- | --- | --- | --- | --- |
| `business_id` | string | `BIZ_000001` to `BIZ_002500` | Sequential identifier | None (never used as a feature) |
| `state` | category | NSW, VIC, QLD, WA, SA, TAS, ACT, NT | Fixed population shares (NSW 32% ... NT 1%) | None |
| `industry` | category | 7 industries | Fixed shares | Construction +0.45, Accommodation & Food +0.35 log-odds |
| `employee_count` | int | 1 to 285 | Log-normal, clipped to 1 to 350 | None |
| `annual_turnover_aud` | float (AUD) | 84,600 to 49.8M | Employees x uniform(75k, 220k) + noise | None |
| `years_operating` | int | 1 to 45 | Exponential (mean 8), clipped | -0.03 log-odds per year |
| `r_and_d_expenditure_aud` | float (AUD) | 0 to 4.6M | 0 for ~37% of firms, else 1 to 12% of turnover | None |
| `debt_to_equity_ratio` | float | 0.1 to 6.0 | Gamma(2, 0.8), clipped | +0.65 log-odds per unit |
| `current_ratio` | float | 0.2 to 3.8 | Normal(1.6, 0.6), clipped | -0.90 log-odds per unit |
| `profit_margin` | float (fraction) | -0.27 to 0.40 | Normal(0.08, 0.10), clipped | -2.50 log-odds per unit |

## Engineered features (`src/feature_engineering.py`)

| Feature | Definition |
| --- | --- |
| `revenue_per_employee` | annual_turnover_aud / max(employee_count, 1) |
| `r_and_d_intensity` | r_and_d_expenditure_aud / max(annual_turnover_aud, 1) |
| `leverage_liquidity_ratio` | debt_to_equity_ratio / (current_ratio + 0.05) |
| `log_turnover` | log(1 + annual_turnover_aud) |
| `state_*`, `industry_*` | One-hot indicators; the alphabetically first level (ACT, Accommodation & Food) is the reference |

Numeric inputs are capped at the 1st and 99th percentiles learned on the training data before feature engineering.

## Validation rules

- Loading (`src/ingest.py`): required columns present, no missing values, numeric types, non-negative values (except `profit_margin`), binary target with both classes, unique IDs.
- Single-record scoring (`src/schema.py`, used by the app): known state and industry, employees 1 to 100,000, turnover > 0, years 0 to 200, debt-to-equity 0 to 100, current ratio > 0 and <= 100, profit margin -1 to 1, R&D not above turnover. Valid values outside the training range are accepted with an extrapolation warning.
