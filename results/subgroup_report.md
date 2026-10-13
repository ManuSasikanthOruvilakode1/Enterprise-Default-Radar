# Subgroup evaluation

> **Synthetic data.** All results come from a synthetic dataset whose target was drawn from a known logistic model. They demonstrate the methodology only and say nothing about performance on real businesses.

Held-out test set, selected threshold 0.19. Rates within a group are based on very few insolvencies, so differences between groups are mostly sampling noise and should not be read as evidence of unfair treatment.

By construction, `state` has **no effect** on the synthetic target, while `industry` does (Construction and Accommodation & Food carry higher risk). Differences in selection rate across industries therefore partly reflect real differences in prevalence. These patterns are properties of the generator, not of real populations.

## By state

| group | n | positives | prevalence | selection_rate | true_positive_rate | false_positive_rate | precision | roc_auc | sufficient_sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACT | 15 | 0 | 0.0000 | 0.1333 | n/a | 0.1333 | 0.0000 | n/a | no |
| NSW | 191 | 13 | 0.0681 | 0.1257 | 0.5385 | 0.0955 | 0.2917 | 0.8349 | yes |
| NT | 7 | 0 | 0.0000 | 0.0000 | n/a | 0.0000 | n/a | n/a | no |
| QLD | 130 | 9 | 0.0692 | 0.1308 | 0.3333 | 0.1157 | 0.1765 | 0.7309 | yes |
| SA | 35 | 4 | 0.1143 | 0.0571 | 0.0000 | 0.0645 | 0.0000 | 0.5887 | no |
| TAS | 19 | 2 | 0.1053 | 0.1053 | 0.5000 | 0.0588 | 0.5000 | 0.6471 | no |
| VIC | 165 | 6 | 0.0364 | 0.1152 | 0.1667 | 0.1132 | 0.0526 | 0.6226 | yes |
| WA | 63 | 6 | 0.0952 | 0.1587 | 0.5000 | 0.1228 | 0.3000 | 0.7807 | yes |

- Groups compared (n >= 50 and >= 5 positives): NSW, QLD, VIC, WA
- Too small to compare: ACT, NT, SA, TAS
- Demographic parity difference (selection rate range): 0.0436
- True positive rate range: 0.3718
- False positive rate range: 0.0273
- Equalized odds difference (max of TPR and FPR ranges): 0.3718

## By industry

| group | n | positives | prevalence | selection_rate | true_positive_rate | false_positive_rate | precision | roc_auc | sufficient_sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Accommodation & Food | 67 | 9 | 0.1343 | 0.1194 | 0.5556 | 0.0517 | 0.6250 | 0.8563 | yes |
| Construction | 138 | 8 | 0.0580 | 0.1522 | 0.5000 | 0.1308 | 0.1905 | 0.7798 | yes |
| Health Care | 49 | 5 | 0.1020 | 0.1429 | 0.2000 | 0.1364 | 0.1429 | 0.6591 | no |
| Manufacturing | 93 | 4 | 0.0430 | 0.0968 | 0.2500 | 0.0899 | 0.1111 | 0.5169 | no |
| Mining | 32 | 0 | 0.0000 | 0.0938 | n/a | 0.0938 | 0.0000 | n/a | no |
| Professional Services | 128 | 9 | 0.0703 | 0.1094 | 0.4444 | 0.0840 | 0.2857 | 0.8571 | yes |
| Retail Trade | 118 | 5 | 0.0424 | 0.1186 | 0.0000 | 0.1239 | 0.0000 | 0.5327 | yes |

- Groups compared (n >= 50 and >= 5 positives): Accommodation & Food, Construction, Professional Services, Retail Trade
- Too small to compare: Health Care, Manufacturing, Mining
- Demographic parity difference (selection rate range): 0.0428
- True positive rate range: 0.5556
- False positive rate range: 0.0790
- Equalized odds difference (max of TPR and FPR ranges): 0.5556
