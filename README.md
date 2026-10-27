# When Do Cosine Prototypes Mislead? A Whitening-Aware Benchmark for Frozen-Feature Image Classification

Code and result artifacts for the NeurIPS 2026 paper (Evaluations & Datasets
track) by Jian Ding and Su Yang (Fudan University).

Cosine nearest-class-mean evaluation is a common default for frozen-feature
image classification. The paper measures when this choice changes benchmark
conclusions: it holds frozen features, splits, and tuning budgets fixed,
compares cosine prototypes with train-validated whitening-aware references
(whitened cosine, WCos-TV, and shared-covariance LDA, LDA-TV), relates their
gap (the Spectral Decision Gap, SDG) to training-set feature geometry, and
quantifies uncertainty and boundary cases.

This repository contains the evaluator code, per-seed result JSONs, manifests,
cached-feature format notes, dataset and backbone preparation recipes, and
rendering scripts for the core results and the expanded router evaluation, as
well as the code for the ImageNet-1K, adaptation, and class-incremental
analyses. The core results and the expanded router evaluation can be checked
from the bundled result files on a CPU, without raw images. The ImageNet-1K,
adaptation, and class-incremental analyses (Appendices J-L) are released as
code only; their result records are not included.

## Contents

| Paper | Analysis | Code | Bundled results |
|---|---|---|---|
| Sec. 3; Table 1, Figure 2 | Full-shot evaluator matrix on DINOv2-B (nine datasets) | `src/sdg_stepb_cached_eval.py`, `src/spectral_decision_gap_ed.py` | `stepb_cached_eval_v1/`, `method_matrix_aggregate.json` |
| Sec. 3; Table 2, Figures 3-4 | Spectral diagnostics (tail Fisher mass, Fisher spectra) | `src/sdg_p0_mechanism.py`, `scripts/recompute/spectral_diagnostics.py` | `p0_mechanism_v2/`, `method_matrix_aggregate.json` |
| Sec. 3; Tables 3-4 | Predictive router (exploratory) with dataset-cluster intervals | `src/sdg_stepb_predictive_protocol.py`, `scripts/recompute/table34_dataset_cluster_ci.py`, `src/sdg_router_uncertainty_audit.py` | `stepb_predictive_protocol_*/` |
| Sec. 4; Tables 5-7; App. D, Table 17 | Estimator decomposition, reference-probe ranking, regime boundaries, estimator taxonomy | `src/sdg_p3_estimator_defense.py`, `scripts/render_assets.py` | `p3_estimator_defense/` |
| Sec. 4; Table 8, Figure 5 | Few-shot support-regime matrix (1-20 shots, five seeds) | `src/sdg_stepb_cached_eval.py`, `src/spectral_decision_gap_ed.py` | `stepb_cached_eval_v1/*/shot_*/result.json` |
| Sec. 4, Figure 6; App. E, Tables 18-20 | Backbone robustness across six backbones, ranking instability | `src/sdg_backbone_compare.py`, `src/sdg_backbone_bootstrap_ci.py`, `src/sdg_backbone_trainval_audit.py` | `backbone/` |
| App. C, Table 11 | Fixed-evaluator gap targets and fixed policies with dataset-cluster CIs | `src/sdg_paired_statistics.py`, `scripts/recompute/fixed_targets_cluster_ci.py` | `stepb_predictive_protocol_reproduced/` |
| App. C, Tables 12-14 | Hyperparameter grids, test-example bootstrap intervals, router feature ablation | `src/sdg_stepb_cached_eval.py`, `src/sdg_bootstrap_ci.py`, `src/sdg_stepb_predictive_protocol.py` | `bootstrap_ci_v1/`, `stepb_predictive_protocol_v2/` |
| App. C, Tables 15-16 | Expanded router evaluation: frozen router on 4, 8, and 11 unseen datasets (66 groups) | `src/sdg_heldout_router_eval.py` (4), `src/sdg_router_ood8_eval.py` (8), `src/sdg_router_ood11_replay.py` and `src/verify_sdg_router_ood11_replay.py` (11), `scripts/recompute/ood8_per_dataset.py` | `heldout_router_v1/`, `router_ood11_replay/` |
| App. F, Tables 21-23, Figure 8 | High-shot and regularization supplement | `src/sdg_supplement_stats.py`, `scripts/recompute/highshot_diagnostics.py` | `supplement_stats_v1/` |
| App. G, Table 24, Figure 9 | Controlled Gaussian mechanism | `src/sdg_synthetic_gaussian.py` | `synthetic_gaussian_v4/` |
| App. H, Table 25, Figure 10 | Synthetic replay case study | results only | `p2_synthetic_case/` |
| App. I, Table 26 | Support-regime matrix with paired t-intervals and BH q-values (five-seed cells; see the note below) | `scripts/recompute/paired_statistics.py`, `src/sdg_paired_statistics.py` | `stepb_cached_eval_v1/` |
| App. J, Tables 27-28 | ImageNet-100, ImageNet-1K, and WordNet subgroups (six backbones) | `src/sdg_imagenet1k_controlled_multibackbone.py`, `src/verify_sdg_imagenet1k_controlled_multibackbone.py`, `src/sdg_materialize_imagenet1k_224.py`, `src/sdg_imagenet100_original_cache_multiseed.py` | code only |
| App. K, Tables 29-30 | Rank-16 cached-feature adapter; last-block and full-encoder fine-tuning | `src/sdg_lightweight_adaptation_audit.py`; `src/sdg_encoder_finetune_audit.py`, `src/sdg_encoder_finetune_comparison.py`, `src/sdg_encoder_finetune_unified_ci.py` | code only |
| App. L, Table 31 | SimpleCIL-aligned full-data CIL, CLS-FSCIL, and patch-mean FSCIL on CUB-200 | `src/sdg_simplecil_matched.py`, `src/sdg_simplecil_fulldata_evaluators.py` (each with `verify_*` and `run_*`), `src/sdg_fscil_prototype_audit.py` | code only |

Bundled results are under `artifacts/results/`; `artifacts/RESULTS_MANIFEST.md`
describes each file. Data preparation and feature extraction are documented in
`DATA_AND_FEATURES.md`.

## Installation

Python 3.10 or newer. To check the core results and the expanded router
evaluation from the bundled files and to rerun the cached-feature evaluators:

```bash
pip install -r requirements.txt
```

For feature extraction, the GPU analyses (Appendices J-L), dataset
preparation, and the tests:

```bash
pip install -r requirements-full.txt
pytest
```

The code was tested with Python 3.11, NumPy 1.26, SciPy 1.17, PyTorch 2.6,
torchvision 0.21, transformers 4.56, datasets 3.6, and pyarrow 17.

## Reproducing the core results from the bundled files

These commands need no raw images, features, checkpoints, GPU, or internet
access, and finish in a few minutes on a laptop CPU.

```bash
python scripts/smoke_check.py                       # files present, headline numbers match
bash scripts/reproduce_paper_assets.sh              # tables and figures -> outputs/paper_assets/
bash scripts/run_predictive_protocol_from_artifacts.sh   # router protocol -> outputs/predictive_protocol/
bash scripts/reproduce_new_analyses.sh              # camera-ready additions -> outputs/new_analyses/
```

`reproduce_paper_assets.sh` regenerates the LaTeX table snippets and figures of
the main paper and Appendices C-H from the result JSONs, in the layout of the
submitted version; file names follow that version (for example,
`fig1_fullshot_sdg.pdf` is Figure 2 of the final paper, and
`fig4_spectral_scatter.pdf` shows the data of Figure 3 in an earlier layout).
`reproduce_new_analyses.sh` runs the analyses added for the camera-ready
version, which supply the intervals and rows that the final tables add:

| Result | Command | Output |
|---|---|---|
| Tables 3-4, dataset-cluster intervals (9 datasets resampled with all shot groups, 10,000 draws); "no tail Fisher" row of Table 14 | `python scripts/recompute/table34_dataset_cluster_ci.py` and `python src/sdg_router_uncertainty_audit.py --output ...` (LODO Spearman and router mean-accuracy intervals, no-tail-Fisher ablation) | `table34_dataset_cluster_ci.json`, `router_uncertainty_audit.json` |
| Table 11 (Appendix C), gap targets (fixed WCos, fixed LDA, per-group max), inflation, LODO Pearson, fixed-policy regret | `python scripts/recompute/fixed_targets_cluster_ci.py` | `fixed_targets_cluster_ci.json` |
| Tables 15-16 (Appendix C), router on 4/8/11 unseen datasets: per-dataset rows, pooled rows, fine-grained block | `python scripts/recompute/ood8_per_dataset.py --block {original_ood4,coarse_ood4,finegrained_ood3}`; `python src/verify_sdg_router_ood11_replay.py --records-only ...` | printed tables; validation log |
| Table 26 (Appendix I) and the interval and q-value columns of Table 7: paired 95% t-intervals and BH q-values over the 45 five-seed cells | `python scripts/recompute/paired_statistics.py` | `paired_statistics.json` |
| Appendices C and I, full statistics audit (adds exact sign-flip tests and target sensitivity) | `python src/sdg_paired_statistics.py` | `paired_statistics_audit/` |
| Figure 3 and Table 22 (Appendix F), diagnostic correlations with and without CUB-200 | `python scripts/recompute/spectral_diagnostics.py`, `python scripts/recompute/highshot_diagnostics.py` | `spectral_diagnostics.json`, `highshot_diagnostics.json` |

`artifacts/results/recompute_reference/` holds the outputs we obtained, for
comparison. The OOD11 replay file stores, for each of the 66 unseen
dataset-shot groups, the router inputs, the frozen router's prediction, and the
evaluator accuracies; `--records-only` verification refits the router on the 54
released training rows, replays all 66 predictions, and recomputes every point
metric and bootstrap interval in the file.

One part of Table 26 cannot be recomputed from the bundled files. Its Full
column gives, next to the point estimate, the mean ± SD and 95% t-interval of
LDA−Cos over five train-only validation splits. These values come from the
outputs of the adaptation study (the frozen arm of
`src/sdg_lightweight_adaptation_audit.py`, Appendix K), which are not bundled;
the same holds for the Tiny-ImageNet five-split interval quoted in Section 4.

Headline values to expect: Aircraft full-shot cosine 42.33, LDA-TV 79.96,
SDG-TV +37.62 pp; CUB-200 cosine 48.90, LDA-TV 80.46, SDG-TV +31.57 pp; Cars
cosine 78.07, LDA-TV 90.16, SDG-TV +12.09 pp; regret of always-cosine 6.71 pp
[0.94, 13.07]; added coarse unseen block router / always-WCos / always-LDA
regret 1.856 / 1.014 / 0.549 pp.

## Rerunning from features or raw images

**Cached-feature evaluators (CPU).** Prepare `train.pt`/`test.pt` caches as in
`DATA_AND_FEATURES.md`, then:

```bash
SDG_FEATURE_ROOT=/path/to/features bash scripts/run_cached_eval_example.sh aircraft full 0
python src/sdg_stepb_cached_eval.py --mode aggregate --out-dir outputs
```

**Feature extraction (GPU).** `src/spectral_generalization.py --mode dataset`
extracts DINOv2-B CLS features (and the other backbones) for the core and
unseen datasets; see `DATA_AND_FEATURES.md` for the checkpoints, dataset
sources, and preparation scripts.

**Unseen-dataset router checks (Appendix C).** OOD4: prepare Caltech-101, DTD,
EuroSAT, and Flowers-102 with `src/prepare_heldout_data.py`, extract features,
run `src/sdg_stepb_cached_eval.py --exp-name heldout_stepb_eval_v1` for every
shot and seed, aggregate, then run `src/sdg_heldout_router_eval.py`.
OOD8: prepare MNIST/Fashion-MNIST/SVHN/GTSRB with
`src/prepare_scope_expansion_data.py`, extract features, run
`src/sdg_stepb_cached_eval.py --train-cache ... --test-cache ...` for every
shot and seed, aggregate, then `src/sdg_router_ood8_eval.py
--scope-aggregate-groups <aggregate_groups.json> --scope-feature-root <features>`.
OOD11 fine-grained block: `src/prepare_finegrained_heldout_data.py` (data and
manifest), `src/seal_finegrained_data_inputs.py` (input binding),
`src/sdg_finegrained_features.py` (features),
`src/sdg_finegrained_eval.py` (one job per dataset and shot), then
`src/sdg_router_ood11_replay.py` and its verifier. The router is always fit
on the nine core datasets only.

The analyses of Appendices J-L are released as code only: the commands below
rerun them, but their result records are not included in this repository.

**ImageNet-1K study (Appendix J).**

```bash
python src/sdg_materialize_imagenet1k_224.py --target-root data/imagenet1k_224
# six workers, one backbone each; rank 0 aggregates once all six results exist
for rank in 1 2 3 4 5 0; do
  RANK=$rank WORLD_SIZE=6 python src/sdg_imagenet1k_controlled_multibackbone.py \
    --data-root data/imagenet1k_224 --output-root outputs/imagenet1k
done
python src/verify_sdg_imagenet1k_controlled_multibackbone.py --output-root outputs/imagenet1k
# ImageNet-100 column: three selection seeds on the six-backbone ImageNet-100 caches
python src/sdg_imagenet100_original_cache_multiseed.py
```

**Adaptation (Appendix K).**

```bash
# rank-16 adapter on the nine DINOv2-B caches, five matched seeds
SEEDS="20260724 20260725 20260726 20260727 20260728"   # evaluator-selection seeds
python src/sdg_lightweight_adaptation_audit.py --dataset aircraft \
  --evaluator-validation-seeds $SEEDS
python src/sdg_lightweight_adaptation_audit.py --aggregate-only \
  --evaluator-validation-seeds $SEEDS
# encoder fine-tuning from raw images (CUB-200, Aircraft, CIFAR-100; three runs)
python src/sdg_encoder_finetune_audit.py --dataset aircraft \
  --finetune-scope last_block --output-root outputs/finetune_last_block
python src/sdg_encoder_finetune_audit.py --dataset aircraft \
  --finetune-scope full_encoder --output-root outputs/finetune_full_encoder
python src/sdg_encoder_finetune_comparison.py \
  --last-block-root outputs/finetune_last_block --full-encoder-root outputs/finetune_full_encoder \
  --output-root outputs/finetune_comparison
python src/sdg_encoder_finetune_unified_ci.py \
  --last-block-root outputs/finetune_last_block --full-encoder-root outputs/finetune_full_encoder
```

**Class-incremental protocols (Appendix L).**

```bash
# SimpleCIL-aligned full-data CIL reproduction and CLS-FSCIL (DINOv2-B without registers)
python src/run_sdg_simplecil_matched.py --cub-root data/CUB_200_2011 \
  --output-root outputs/simplecil_matched
# matched Cos/WCos/LDA in the full-data CIL schedule
python src/run_sdg_simplecil_fulldata_evaluators.py --source-root outputs/simplecil_matched \
  --protocol configs/simplecil_fulldata_evaluators_protocol.json \
  --protocol-sha256 9472fb3cf2529dcac7ae9103dc092cbefc3132fc2a778ee19af3e045cee25c52 \
  --output-root outputs/simplecil_fulldata
# patch-mean FSCIL on the CUB-200 core cache
python src/sdg_fscil_prototype_audit.py --source-root <patch-mean source cache>
```

The full-data protocol file pins the hashes of our canonical CLS feature cache.
If you regenerate the features, copy the file, update its `source` hashes to
your cache, and pass the new file's SHA-256.

Most producers refuse to overwrite an existing output directory, record the
hashes of their inputs, and accept `--help` for all options.

## Configurable paths

All defaults are relative to the repository and can be moved with environment
variables (or the corresponding command-line flags):

| Variable | Default | Contents |
|---|---|---|
| `SDG_RESULTS_ROOT` | `artifacts/results` | released result files |
| `SDG_OUTPUT_ROOT` | `outputs` | regenerated outputs |
| `SDG_FEATURE_ROOT` | `features` | cached features (`<dataset>/` and `backbones/<backbone>/<dataset>/`) |
| `SDG_DATA_ROOT` | `data` | prepared datasets (`<dataset>/hf_dataset/{train,test}`) |
| `SDG_MODEL_ROOT` | `models` | local Hugging Face model snapshots |

## Repository layout

```text
.
├── artifacts/results/     released result JSONs and reference outputs
├── configs/               preregistered protocol for the full-data CIL evaluator audit
├── scripts/               reproduction entry points
│   └── recompute/         CPU recomputations of the camera-ready numbers
├── splits/fscil_cub200/   standard CUB-200 FSCIL session files
├── src/                   evaluators, analyses, data preparation, feature extraction
├── tests/                 unit tests (pytest)
├── DATA_AND_FEATURES.md   datasets, checkpoints, feature definitions and format
├── LICENSE                MIT
└── NOTICE                 third-party notices
```

Analysis code lives in one flat `src/` directory so that modules import each
other by name; `pytest.ini` puts `src/` on the path for the tests, and the
scripts add it themselves.

## Data, features, and licenses

The core suite uses the final-layer CLS token of DINOv2 ViT-B/14 with
registers; for CUB-200, the core cache is the spatial mean of the same
encoder's patch-token latents. `DATA_AND_FEATURES.md` lists the features of
every analysis (Table 10 of the paper) and notes how the MAE features of the
six-backbone sweep were extracted.

Raw images, cached feature tensors, and pretrained checkpoints are not
redistributed. Obtain datasets from their original sources and checkpoints from
the Hugging Face Hub; both remain under their providers' terms.

The code is released under the MIT License (`LICENSE`). The CUB-200 FSCIL
session files in `splits/fscil_cub200/` come from the CEC repository (MIT); see
`NOTICE`.

## Citation

```bibtex
@inproceedings{ding2026cosine,
  title     = {When Do Cosine Prototypes Mislead? A Whitening-Aware Benchmark for Frozen-Feature Image Classification},
  author    = {Ding, Jian and Yang, Su},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026},
  note      = {To appear}
}
```

## Contact

Jian Ding (jding24@m.fudan.edu.cn) and Su Yang (suyang@fudan.edu.cn).
