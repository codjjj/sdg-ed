# Reproducibility Checklist

The repository contains the evaluator code, per-seed result JSONs, manifests,
cached-feature format notes, dataset and backbone preparation recipes, and
rendering scripts for the core results and the expanded router evaluation, as
well as the code for the ImageNet-1K, adaptation, and class-incremental
analyses (Appendices J-L). The result records of Appendices J-L are not
included.

## Reproducible from the bundled result files (CPU, minutes)

- Full-shot and few-shot evaluator tables (Tables 1, 5-8) and figures.
- Spectral diagnostics (Table 2, Figures 3-4) and their correlations with and
  without CUB-200 (Figure 3, Table 22).
- Predictive router metrics, feature ablations (Table 14), and dataset-cluster
  intervals for every cell of Tables 3 and 4.
- Gap targets and fixed reporting policies with dataset-cluster intervals
  (Table 11, Appendix C).
- Expanded router evaluation on 4, 8, and 11 unseen datasets, including all
  per-dataset rows and the fine-grained block (Tables 15-16, Appendix C),
  verified from the replay records.
- Paired uncertainty for the five-seed cells of the support-regime matrix:
  means, SDs, 95% paired t-intervals, and BH q-values (Table 26, Appendix I).
- Full-shot test-example bootstrap intervals (Table 13), estimator taxonomy,
  backbone robustness, high-shot supplement, Gaussian mechanism, and synthetic
  replay (Appendices D-H).

```bash
python scripts/smoke_check.py
bash scripts/reproduce_paper_assets.sh
bash scripts/run_predictive_protocol_from_artifacts.sh
bash scripts/reproduce_new_analyses.sh
```

Not reproducible from the bundled files: the five-split mean ± SD and 95%
t-interval in the Full column of Table 26 (Appendix I). They come from the
outputs of the adaptation study (the frozen arm of
`src/sdg_lightweight_adaptation_audit.py`, Appendix K), which are not bundled.

## Reproducible with locally prepared cached features (CPU)

- Train-validated cosine, WCos, LDA, ridge, QDA-diag, and kNN evaluator reruns
  for any dataset-shot-seed cell, and their aggregation.
- Backbone bootstrap and train-validated backbone ranking
  (`src/sdg_backbone_bootstrap_ci.py`, `src/sdg_backbone_trainval_audit.py`).
- ImageNet-100 three-seed check on the six-backbone caches
  (`src/sdg_imagenet100_original_cache_multiseed.py`).

## Code only: rerun with raw images and a GPU

The analyses of Appendices J-L are released as code only; their result
records are not included.

- Feature extraction for the core suite, the six-backbone sweep, and the
  unseen datasets (`src/spectral_generalization.py`,
  `src/sdg_finegrained_features.py`).
- ImageNet-1K and WordNet-subgroup study (Appendix J;
  `src/sdg_imagenet1k_controlled_multibackbone.py`).
- Rank-16 adapter and encoder fine-tuning (Appendix K;
  `src/sdg_lightweight_adaptation_audit.py`,
  `src/sdg_encoder_finetune_audit.py`).
- SimpleCIL-aligned CIL and FSCIL runs (Appendix L;
  `src/sdg_simplecil_matched.py`,
  `src/sdg_simplecil_fulldata_evaluators.py`,
  `src/sdg_fscil_prototype_audit.py`).

Each producer script writes a result JSON with its inputs' hashes; several have an
independent `verify_*.py` checker.

## Not bundled

- Raw images and cached feature tensors.
- Pretrained checkpoints.
- Result records of the ImageNet-1K, adaptation, and class-incremental
  analyses (Appendices J-L).
- Job-scheduler scripts and logs of the original compute environment.
