# Results Manifest

Paths are relative to `artifacts/results/`. Table, figure, and appendix
references follow the camera-ready paper.

## Core full-shot and few-shot results

- `stepb_cached_eval_v1/aggregate_groups.json`: train-validated cached-feature
  evaluators for the 54 DINOv2-B dataset-shot groups (Table 1, Table 8).
- `stepb_cached_eval_v1/*/shot_*/result.json`: per-dataset, per-shot,
  per-seed records (few-shot means and SDs; the five-seed cells of Table 26,
  Appendix I).
- `method_matrix_aggregate.json`: method matrix with cosine,
  whitening-aware, FeCAM-style, logistic, and MLP references, plus the
  seed-level spectral diagnostics plotted in Figure 3.
- `bootstrap_ci_v1/fullshot_ci.json`: test-example bootstrap intervals for the
  full-shot gaps (Table 13, Appendix C).

## Spectral mechanism and diagnostics

- `p0_mechanism_v2/summary_with_pred.json`: full-shot spectral diagnostics
  including tail Fisher statistics (Table 2).
- `p0_mechanism_v2/{cub200,tiny_imagenet}/shot_full/seed_0/spectrum_fisher.npz`:
  PC-wise Fisher spectra (Figure 4).
- `supplement_stats_v1/aggregate_groups.json`,
  `supplement_stats_v1/diagnostic_correlations.json`: high-shot and
  regularization supplement (Tables 21-23, Figure 8, Appendix F).

## Estimator, routing, and unseen-dataset checks

- `p3_estimator_defense/aggregate.json`: estimator decomposition and FeCAM-style
  comparisons (Table 5; Table 17, Appendix D).
- `stepb_predictive_protocol_v1/predictive_results.json`,
  `stepb_predictive_protocol_v2/feature_ablation.json`: predictive router
  metrics and feature ablations (Tables 3-4; Table 14, Appendix C).
- `stepb_predictive_protocol_reproduced{,_ablation}/`: the same protocol
  recomputed from the aggregates by
  `scripts/run_predictive_protocol_from_artifacts.sh`; `joined_records.json`
  holds the 54 training rows of the frozen router.
- `heldout_router_v1/heldout_router_generalization.json`: router on the four
  original unseen datasets (OOD4 rows of Table 15, Appendix C).
- `router_ood11_replay/router_ood11_replay.json` (+ `.md` summary):
  frozen-router replay on 11 unseen datasets (66 dataset-shot groups) in three
  blocks: `original_ood4` (Caltech-101, DTD, EuroSAT, Flowers-102),
  `coarse_ood4` (MNIST, Fashion-MNIST, SVHN, GTSRB), and `finegrained_ood3`
  (Stanford Dogs, FoodX-251, NABirds); plus the pooled `ood8` and `ood11`
  suites. Each group record carries the router inputs, the prediction, and the
  evaluator accuracies; each block carries point metrics and 10,000-draw
  dataset-cluster bootstrap intervals (expanded router evaluation, Tables 15-16,
  Appendix C). The `inputs` section lists the hashes of the per-seed prediction
  arrays used to build it; those arrays are not bundled.

## Backbone and synthetic analyses

- `backbone/<backbone>/<dataset>/result.json`: backbone robustness sweep
  (Figure 6; Tables 18 and 20, Appendix E).
- `backbone/clip_vit_b32/backbone_comparison.json`: per-dataset DINOv2-B and
  CLIP comparison (also the DINOv2-B point references used by the backbone
  scripts).
- `backbone/ranking_instability.json`: backbone ranking and regret
  (Table 19, Appendix E).
- `p2_synthetic_case/cub200/summary.json`: synthetic replay case study
  (Table 25, Figure 10, Appendix H).
- `synthetic_gaussian_v4/summary.json` and plots: controlled Gaussian mechanism
  (Table 24, Figure 9, Appendix G).

## Reference outputs of the recomputation scripts

`recompute_reference/` holds the outputs of `scripts/recompute/*.py` and
`src/sdg_router_uncertainty_audit.py` run on the files above, for comparison
with your own runs:

- `table34_dataset_cluster_ci.json`: dataset-cluster intervals for every cell
  of Tables 3 and 4.
- `router_uncertainty_audit.json`: dataset-cluster router audit
  (`src/sdg_router_uncertainty_audit.py`); Tables 3-4 take the LODO Spearman
  interval and the LODO router mean-accuracy interval from its
  `main_54_groups` block, and Table 14 takes its "no tail Fisher" row from the
  `feature_incremental_value` block.
- `fixed_targets_cluster_ci.json`: gap targets and fixed policies
  (Table 11, Appendix C).
- `paired_statistics.json`: paired t-intervals and BH q-values for the
  five-seed cells of the support-regime matrix (Table 26, Appendix I; also the
  interval and q-value columns of Table 7).
- `ood8_per_dataset.json`: per-dataset rows of the added coarse block
  (Table 15, Appendix C).
- `spectral_diagnostics.json`, `highshot_diagnostics.json`: diagnostic
  correlations with and without CUB-200 (Figure 3; Table 22, Appendix F).

## Not included

- Result records of the ImageNet-1K, adaptation, and class-incremental
  analyses (Appendices J-L). The repository includes their code.
- The five-split values in the Full column of Table 26 (Appendix I): the mean
  ± SD and 95% t-interval of LDA−Cos over five train-only validation splits
  come from the outputs of the adaptation study (the frozen arm of
  `src/sdg_lightweight_adaptation_audit.py`, Appendix K) and cannot be
  recomputed from the bundled files. The full-shot point estimates are
  bundled.

## Generated outputs

`bash scripts/reproduce_paper_assets.sh` writes `outputs/paper_assets/`;
`bash scripts/run_predictive_protocol_from_artifacts.sh` writes
`outputs/predictive_protocol/`; `bash scripts/reproduce_new_analyses.sh` writes
`outputs/new_analyses/`.
