# OOD11 frozen-router replay

The frozen router is fit only on the nine core datasets; Stanford Dogs, FoodX-251, and NABirds are never used for fitting or refitting.
iNaturalist-2021 mini produced no accepted result and is excluded; non-full NABirds groups use min(available, K) support images per class.

| block | datasets / groups | Pearson | router / WCos / LDA regret (pp) |
|---|---:|---:|---:|
| original_ood4 | 4 / 24 | 0.472 | 1.150 / 0.395 / 0.931 |
| coarse_ood4 | 4 / 24 | 0.914 | 1.856 / 1.014 / 0.549 |
| finegrained_ood3 | 3 / 18 | 0.500 | 0.168 / 2.264 / 0.335 |
| ood8 | 8 / 48 | 0.791 | 1.503 / 0.705 / 0.740 |
| ood11 | 11 / 66 | 0.523 | 1.139 / 1.130 / 0.630 |
