# Model card: Enterprise Default Radar insolvency risk model

## Model details

- **Type:** LightGBM gradient-boosted trees with monotonic constraints, hyperparameters tuned by Optuna (30 trials, 3-fold inner CV), followed by sigmoid (Platt) calibration and a cost-based decision threshold of 0.19.
- **Selected by:** a tournament of nine configurations on repeated stratified CV (5 folds x 3 repeats), with tuning nested inside each training fold. The selection metric was mean AUC-PR.
- **Artifact:** `models/risk_model.joblib` (`src/pipeline.RiskModel`), recreated by `scripts/run_pipeline.py`. Its hyperparameters, data hash and library versions are in `results/run_metadata.json`.
- **Inputs:** state, industry, employee count, annual turnover, years operating, R&D expenditure, debt-to-equity ratio, current ratio and profit margin, validated by `src/schema.BusinessProfile`.
- **Output:** a calibrated probability of insolvency, and a flag when the probability is at least 0.19.

## Intended use

- To demonstrate a leakage-free, reproducible risk-modelling workflow: model comparison, calibration, cost-sensitive decisions, explanation, subgroup reporting and drift monitoring.
- For teaching, code review and as a template to adapt to real data after proper validation.

## Out-of-scope use

- Assessing real businesses, credit decisions, lending, insurance, investment or any decision about real people or companies.
- Treating the probability or the flag as financial advice.
- Applying the model to data from another source without retraining and re-validating it.

## Training data

- 2,500 **synthetic** businesses (`data/raw/business_outcomes.csv`), 6.36% labelled insolvent. Distributions are loosely shaped after public economic statistics. Labels are drawn from a known logistic model of leverage, liquidity, profit margin, firm age and industry. State, size, turnover and R&D spend have no effect by construction. See `docs/data_dictionary.md`.
- The data was split 75% for development (1,875 rows, 119 insolvent) and 25% for the final test (625 rows, 40 insolvent).

## Evaluation (held-out test set, 95% bootstrap intervals)

| Metric | Model | Original baseline | True probabilities (ceiling) |
| --- | --- | --- | --- |
| AUC-PR | 0.219 [0.126, 0.366] | 0.224 [0.125, 0.380] | 0.235 [0.135, 0.386] |
| ROC-AUC | 0.736 [0.645, 0.821] | 0.727 [0.647, 0.810] | 0.754 [0.679, 0.829] |
| Brier score | 0.057 [0.043, 0.072] | 0.121 [0.106, 0.135] | 0.055 [0.041, 0.071] |

- **Ranking:** the test set cannot separate the model from the original baseline; the AUC-PR difference is -0.005 [-0.073, 0.049]. In cross-validation the model scores 0.243 against the baseline's 0.193 and beats it in 12 of 15 folds.
- **Probabilities:** calibration brings the Brier score close to the ceiling.
- **Decisions:** at the 0.19 threshold the model flags 76 of 625 businesses and catches 15 of the 40 insolvencies, at a total illustrative cost of $186. The original rule costs $196 at the same $1 FP / $5 FN assumptions.
- **Reference:** a logistic regression scores higher in cross-validation (0.288). The data generator is itself logistic, so that model is correctly specified. The learning-curve experiment (`results/learning_curve.md`) shows the boosters closing the gap as training data grows.

## Subgroups

- Performance is reported by state and industry in `results/subgroup_report.md`.
- Most groups have fewer than 15 insolvencies in the test set, and groups below 50 rows or 5 insolvencies are excluded from disparity summaries. The remaining differences are dominated by sampling noise.
- State has no effect in the generator, so any state disparity is noise. Industry differences partly reflect real differences in prevalence.
- No fairness claim is made.

## Explainability

- SHAP values are reported in `results/example_explanations.md` and `shap_summary.png`, and shown per prediction in the app.
- About 30% of the attribution goes to inputs that have no real effect, which shows the model still fits some noise. Explanations describe the model's behaviour, not the causes of insolvency.

## Monitoring

- `src/drift.py` reports PSI and KS statistics, missing columns and malformed values. Thresholds are 0.10 (moderate) and 0.25 (critical).
- A simulated downturn triggers critical alerts and raises the share of flagged businesses from 12.2% to 29.0%.
- An alert is a prompt to investigate the data, not evidence that accuracy has fallen.

## Ethical considerations and limitations

- **Synthetic data:** none of the results transfer to real businesses.
- **Location as an input:** using a business's state as a feature could encode geographic disadvantage in real data. It is kept here only to show subgroup reporting.
- **Costs:** they are illustrative. Real review costs, capacity limits and the harm of a wrong flag differ between organisations and must be set by the people accountable for the decision.
- **Sample size:** with 40 test insolvencies, every metric has wide uncertainty. Read the intervals, not just the point estimates.
- **Human oversight:** a flag should prompt a human review, never trigger an automatic action.
