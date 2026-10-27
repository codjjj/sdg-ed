# Step B Predictive Protocol v1

Features are computed from training-only spectral/covariance diagnostics; targets come from Step B non-oracle cached evaluation.
The evaluation unit is a dataset-shot group (`n=54` = 9 datasets x 6 shot regimes); non-full groups are 5-seed means.

| target | fold | n | Pearson [95% CI] | Spearman [95% CI] | MAE pp [95% CI] | RMSE pp [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| target_best_whitening_minus_cos_pp | leave_one_group | 54 | 0.832 [0.755, 0.895] | 0.692 [0.467, 0.825] | 4.65 [3.66, 5.74] | 6.04 [4.67, 7.41] |
| target_best_whitening_minus_cos_pp | leave_one_dataset | 54 | 0.752 [0.638, 0.846] | 0.678 [0.479, 0.799] | 5.46 [4.23, 6.78] | 7.24 [5.39, 9.01] |
| target_ridge_minus_cos_pp | leave_one_group | 54 | 0.801 [0.730, 0.876] | 0.673 [0.459, 0.809] | 4.12 [3.14, 5.16] | 5.58 [4.17, 6.96] |
| target_ridge_minus_cos_pp | leave_one_dataset | 54 | 0.713 [0.604, 0.828] | 0.630 [0.414, 0.763] | 4.87 [3.70, 6.13] | 6.66 [4.82, 8.45] |

## Routing

### target_best_whitening_minus_cos_pp / leave_one_group
| threshold | chosen acc [95% CI] | oracle acc [95% CI] | regret pp [95% CI] | whitening frac [95% CI] |
|---:|---:|---:|---:|---:|
| 0.0 | 76.67 [71.62, 81.34] | 76.92 [71.84, 81.64] | 0.25 [0.12, 0.39] | 0.74 [0.61, 0.85] |
| 2.0 | 76.57 [71.57, 81.01] | 76.92 [72.03, 81.32] | 0.35 [0.17, 0.56] | 0.54 [0.41, 0.67] |
| 5.0 | 76.59 [71.90, 81.26] | 76.92 [72.34, 81.49] | 0.33 [0.17, 0.53] | 0.43 [0.30, 0.56] |
| 10.0 | 75.79 [70.44, 80.32] | 76.92 [71.97, 81.19] | 1.13 [0.47, 1.93] | 0.26 [0.15, 0.37] |

Trivial baselines:
| policy | mean acc [95% CI] | regret pp [95% CI] |
|---|---:|---:|
| always cosine | 70.21 [64.43, 75.93] | 6.71 [4.05, 9.56] |
| always WCos | 75.42 [70.46, 80.09] | 1.50 [1.01, 2.07] |
| always LDA | 75.71 [70.65, 80.28] | 1.21 [0.83, 1.65] |
| oracle best of cosine/whitening | 76.92 [71.88, 81.31] | 0.00 [0.00, 0.00] |

### target_best_whitening_minus_cos_pp / leave_one_dataset
| threshold | chosen acc [95% CI] | oracle acc [95% CI] | regret pp [95% CI] | whitening frac [95% CI] |
|---:|---:|---:|---:|---:|
| 0.0 | 76.68 [71.63, 81.35] | 76.92 [71.84, 81.64] | 0.24 [0.12, 0.39] | 0.72 [0.59, 0.83] |
| 2.0 | 76.35 [71.22, 80.90] | 76.92 [72.03, 81.32] | 0.58 [0.20, 1.19] | 0.54 [0.41, 0.67] |
| 5.0 | 76.36 [71.57, 81.17] | 76.92 [72.34, 81.49] | 0.56 [0.19, 1.11] | 0.43 [0.30, 0.56] |
| 10.0 | 75.85 [70.50, 80.40] | 76.92 [71.97, 81.19] | 1.08 [0.41, 1.88] | 0.30 [0.19, 0.43] |

Trivial baselines:
| policy | mean acc [95% CI] | regret pp [95% CI] |
|---|---:|---:|
| always cosine | 70.21 [64.43, 75.93] | 6.71 [4.05, 9.56] |
| always WCos | 75.42 [70.46, 80.09] | 1.50 [1.01, 2.07] |
| always LDA | 75.71 [70.65, 80.28] | 1.21 [0.83, 1.65] |
| oracle best of cosine/whitening | 76.92 [71.88, 81.31] | 0.00 [0.00, 0.00] |
