# Original baseline report (archived)

> Archived output of the original pipeline, kept for comparison. The threshold (0.8029) and the cost saving were selected
> on the test set, so they are in-sample figures. See `docs/baseline_verification.md` for the issues found and
> `results/model_comparison.md` / `results/threshold_report.md` for the corrected evaluation.

- **AUC-ROC**: 0.7291 (95% CI: [0.6499, 0.8092])
- **AUC-PR**: 0.2089 (95% CI: [0.1237, 0.3604])
- **Optimal Decision Threshold**: 0.8029
- **Cost Savings vs Default (0.5)**: $37.00 (18.5% reduction)
- **Brier Score**: 0.1192
- **Balanced Accuracy**: 61.4%

## Fairness Across States

- Demographic Parity Difference: 0.0794
- Equalized Odds Difference: 0.0667
