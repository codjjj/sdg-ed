# Predictive Router Feature Ablation

All rows reuse the 54 dataset-shot groups from the train-only protocol. The `full` row is the original diagnostic set used in the main paper; the other rows ablate diagnostic families without changing the fold protocol.

## Best-whitening minus cosine target

| feature set | fold | Pearson | Spearman | MAE pp | RMSE pp | 5pp router regret |
|---|---|---:|---:|---:|---:|---:|
| full | leave_one_group | 0.832 | 0.692 | 4.65 | 6.04 | 0.33 |
| full | leave_one_dataset | 0.752 | 0.678 | 5.46 | 7.24 | 0.56 |
| spectral_only | leave_one_group | 0.766 | 0.665 | 5.12 | 7.05 | 0.42 |
| spectral_only | leave_one_dataset | 0.601 | 0.589 | 6.45 | 8.83 | 0.57 |
| covariance_only | leave_one_group | 0.818 | 0.665 | 4.95 | 6.25 | 0.31 |
| covariance_only | leave_one_dataset | 0.762 | 0.676 | 5.58 | 7.09 | 0.29 |
| variance_only | leave_one_group | 0.327 | 0.601 | 7.03 | 10.53 | 0.37 |
| variance_only | leave_one_dataset | 0.077 | 0.404 | 8.01 | 11.78 | 2.81 |
| tail_fisher_only | leave_one_group | 0.670 | 0.482 | 5.08 | 8.08 | 0.77 |
| tail_fisher_only | leave_one_dataset | 0.598 | 0.421 | 5.60 | 8.74 | 0.75 |
| tail128_only | leave_one_group | 0.612 | 0.438 | 5.51 | 8.60 | 0.51 |
| tail128_only | leave_one_dataset | 0.493 | 0.393 | 6.12 | 9.51 | 1.45 |
| condition_only | leave_one_group | 0.669 | 0.573 | 6.46 | 8.09 | 0.55 |
| condition_only | leave_one_dataset | 0.581 | 0.554 | 7.04 | 8.87 | 0.53 |
| no_tail_fisher | leave_one_group | 0.815 | 0.651 | 5.02 | 6.30 | 0.31 |
| no_tail_fisher | leave_one_dataset | 0.756 | 0.657 | 5.72 | 7.17 | 0.29 |
| no_covariance | leave_one_group | 0.682 | 0.597 | 5.19 | 7.99 | 0.76 |
| no_covariance | leave_one_dataset | 0.652 | 0.535 | 5.54 | 8.33 | 0.76 |
| no_sample_regime | leave_one_group | 0.773 | 0.541 | 5.04 | 6.92 | 0.42 |
| no_sample_regime | leave_one_dataset | 0.662 | 0.504 | 5.83 | 8.22 | 0.85 |

## Ridge-reference minus cosine target

| feature set | fold | Pearson | Spearman | MAE pp | RMSE pp |
|---|---|---:|---:|---:|---:|
| full | leave_one_group | 0.801 | 0.673 | 4.12 | 5.58 |
| full | leave_one_dataset | 0.713 | 0.630 | 4.87 | 6.66 |
| spectral_only | leave_one_group | 0.699 | 0.576 | 4.67 | 6.74 |
| spectral_only | leave_one_dataset | 0.528 | 0.453 | 5.75 | 8.16 |
| covariance_only | leave_one_group | 0.805 | 0.674 | 4.13 | 5.52 |
| covariance_only | leave_one_dataset | 0.744 | 0.654 | 4.66 | 6.26 |
| variance_only | leave_one_group | 0.288 | 0.444 | 6.01 | 9.19 |
| variance_only | leave_one_dataset | -0.007 | 0.193 | 6.87 | 10.44 |
| tail_fisher_only | leave_one_group | 0.624 | 0.547 | 4.55 | 7.29 |
| tail_fisher_only | leave_one_dataset | 0.569 | 0.507 | 4.89 | 7.70 |
| tail128_only | leave_one_group | 0.568 | 0.563 | 4.94 | 7.65 |
| tail128_only | leave_one_dataset | 0.482 | 0.528 | 5.25 | 8.18 |
| condition_only | leave_one_group | 0.659 | 0.639 | 5.38 | 7.00 |
| condition_only | leave_one_dataset | 0.586 | 0.621 | 5.69 | 7.54 |
| no_tail_fisher | leave_one_group | 0.802 | 0.661 | 4.24 | 5.56 |
| no_tail_fisher | leave_one_dataset | 0.740 | 0.628 | 4.77 | 6.30 |
| no_covariance | leave_one_group | 0.619 | 0.639 | 4.70 | 7.36 |
| no_covariance | leave_one_dataset | 0.597 | 0.590 | 5.03 | 7.66 |
| no_sample_regime | leave_one_group | 0.735 | 0.530 | 4.62 | 6.34 |
| no_sample_regime | leave_one_dataset | 0.620 | 0.461 | 5.27 | 7.40 |
