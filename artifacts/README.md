# Bundled Artifacts

`artifacts/results/` holds the small result files behind the core results and
the expanded router evaluation: per-seed evaluator records, aggregates,
diagnostics, and a few generated assets. It does not include raw images,
cached feature tensors, checkpoints, or job logs. The ImageNet-1K, adaptation,
and class-incremental analyses (Appendices J-L) are released as code only;
their result records are not included.

`RESULTS_MANIFEST.md` lists every result file and what it reproduces.
`../DATA_AND_FEATURES.md` documents how to prepare features when rerunning
evaluators locally.

Regenerate the tables and figures of the main paper and Appendices C-H (in the
layout of the submitted version):

```bash
bash scripts/reproduce_paper_assets.sh
```

Recompute the camera-ready additions (dataset-cluster intervals, paired
statistics, fixed targets, expanded router tables):

```bash
bash scripts/reproduce_new_analyses.sh
```

Rerun the evaluator on your own cached features (format in
`DATA_AND_FEATURES.md`):

```bash
SDG_FEATURE_ROOT=features bash scripts/run_cached_eval_example.sh cub200 full 0
```
