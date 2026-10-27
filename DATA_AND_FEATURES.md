# Data and Cached Features

The core results and the expanded router evaluation can be checked from the
result files bundled under `artifacts/results/` (see the README); the
ImageNet-1K, adaptation, and class-incremental analyses (Appendices J-L) are
released as code only. This document is for users who want to go further and
regenerate features or rerun evaluators. The repository does not redistribute
raw images, cached feature tensors, or pretrained checkpoints: they are large
and governed by the terms of the original dataset and model providers.

## Feature definitions

This table follows Table 10 of the paper.

| Result | Encoder | Feature |
|---|---|---|
| Core suite (Sections 3-4; Appendices C, D, F, I; input of the rank-16 adapter in Appendix K) and unseen-dataset router checks (Appendix C) | DINOv2 ViT-B/14 with registers | final-layer CLS token |
| Core suite, CUB-200 | same | spatial mean of the patch-token latents |
| Six-backbone sweep (Appendix E), DINOv2-B column | DINOv2 ViT-B/14 with registers | the core-suite features above |
| Six-backbone sweep (Appendix E), other backbones | CLIP ViT-B/32, MAE ViT-B/16, SigLIP-B/16, ConvNeXt-B, supervised ViT-B/16 | standard pooled feature |
| ImageNet-1K study (Appendix J) | the six backbones | standard pooled feature (CLS token for DINOv2-B) |
| Encoder fine-tuning (Appendix K) | DINOv2 ViT-B/14 with registers | CLS token re-extracted after tuning |
| CIL and CLS-FSCIL (Appendix L) | DINOv2 ViT-B/14 without registers | CLS token |
| Patch-mean FSCIL (Appendix L) | DINOv2 ViT-B/14 with registers | the CUB-200 core-suite feature |

"Standard pooled feature" is the checkpoint's image embedding: the projected
image embedding for CLIP and SigLIP, the CLS token for ViT-style encoders, and
the global average of the final feature map for ConvNeXt.

MAE masking: in the six-backbone sweep, MAE ViT-B/16 features were extracted
with the Hugging Face default forward pass (`src/spectral_generalization.py
--backbone hf_vision`), which randomly masks 75% of the patches; each cache
holds one such draw per image, shared by all evaluators. Re-extracting without
masking (`mask_ratio = 0`, as in the ImageNet-1K study) changes the ImageNet-100
gaps only slightly: +30.30/+28.05 pp (WCos/LDA, exact-100 task of Table 28)
versus +30.15/+28.10 pp with the original caches (Table 27).

The core suite therefore uses the final-layer CLS token of DINOv2 ViT-B/14 with
registers for eight of the nine datasets. For CUB-200, the core cache is the
spatial mean of the same encoder's patch-token latents. The extraction script
in this repository (`src/spectral_generalization.py`) produces CLS tokens;
CUB-200 CLS features are therefore a different representation from the core
CUB-200 cache.

## Checkpoints

All checkpoints are public on the Hugging Face Hub. Scripts that hash or load
local snapshots read them from `$SDG_MODEL_ROOT/<name>` (default `models/<name>`):

```bash
huggingface-cli download facebook/dinov2-with-registers-base --local-dir models/dinov2-with-registers-base
```

| Name in code | Hub checkpoint | Local directory |
|---|---|---|
| `dinov2` (core suite) | `facebook/dinov2-with-registers-base` | `dinov2-with-registers-base` |
| `clip_vit_b32` | `openai/clip-vit-base-patch32` | `clip-vit-base-patch32` |
| `mae_vit_base` | `facebook/vit-mae-base` | `vit-mae-base` |
| `siglip_b16` | `google/siglip-base-patch16-224` | `siglip-base-patch16-224` |
| `convnext_base` | `facebook/convnext-base-224` | `convnext-base-224` |
| `vit_sup_b16` | `google/vit-base-patch16-224` | `vit-base-patch16-224` |
| SimpleCIL-aligned runs | `facebook/dinov2-base`, revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415` | `dinov2-base` |

## Datasets

Obtain every dataset from its original source and use it under its license.
Prepared datasets live under `$SDG_DATA_ROOT` (default `data/`); the
extraction scripts first look for `<data-root>/<dataset>/hf_dataset/{train,test}`
(Hugging Face `save_to_disk` format), then fall back to the Hugging Face Hub or
torchvision.

Core suite (train / test images used by the evaluators):

| Dataset | Split | Train / test | Source |
|---|---|---|---|
| CUB-200-2011 | official | 5,994 / 5,794 | Caltech-UCSD Birds-200-2011 |
| FGVC Aircraft (variant, 100 classes) | trainval / test | 6,667 / 3,333 | torchvision `FGVCAircraft` |
| Stanford Cars | official | 8,144 / 8,041 | official split (torchvision's download link is offline; use any copy of the official split as an `hf_dataset`) |
| Food-101 | train / validation | 75,750 / 25,250 | HF `food101` |
| Oxford-IIIT Pets | trainval / test | 3,680 / 3,669 | HF `timm/oxford-iiit-pet` |
| CIFAR-10, CIFAR-100 | official | 50,000 / 10,000 | torchvision |
| Tiny-ImageNet | train / val | 100,000 / 10,000 | `tiny-imagenet-200.zip` (CS231n) |
| ImageNet-100 | train / validation | 126,689 / 5,000 | HF `clane9/imagenet-100` |

The released full-shot records list the number of training examples in each
core cache (`n_train` in `artifacts/results/stepb_cached_eval_v1/*/shot_full/result.json`).

Unseen datasets for the router checks (Appendix C):

| Block | Datasets | Preparation |
|---|---|---|
| OOD4 | Caltech-101, DTD, EuroSAT, Flowers-102 | `src/prepare_heldout_data.py` (splits documented in its docstring) |
| added coarse block | MNIST, Fashion-MNIST, SVHN, GTSRB | `src/prepare_scope_expansion_data.py` (official splits) |
| added fine-grained block | Stanford Dogs, FoodX-251, NABirds | `src/prepare_finegrained_heldout_data.py` (official splits, archive hashes pinned) |

NABirds has no public download URL in the code: download `nabirds.tar.gz`
from the official NABirds page and pass `--source-root <dir>` with the archive
at `<dir>/nabirds/nabirds.tar.gz`. NABirds is limited to non-commercial use and
may not be redistributed. iNaturalist-2021 mini is supported by the
preparation script but produced no accepted result and is not in the paper.

ImageNet-1K study (Appendix J): the public 224-pixel derivative
`mlnomad/imagenet-1k-224` at revision `d9e99f3b3b356abef801094154a0595729ac0818`
(1,281,167 train / 50,000 validation images), downloaded and verified by
`src/sdg_materialize_imagenet1k_224.py`. ImageNet terms of access apply.

Raw images are also needed for encoder fine-tuning (CUB-200, Aircraft,
CIFAR-100; Appendix K) and for the SimpleCIL-aligned CIL/FSCIL runs (CUB-200;
Appendix L). Both read the official `CUB_200_2011/` directory
(`images.txt`, `train_test_split.txt`, `image_class_labels.txt`, `images/`).
The FSCIL session files are shipped in `splits/fscil_cub200/`.

## Feature extraction

`src/spectral_generalization.py` extracts one pooled vector per image after
resize-256 (bicubic), center-crop-224 and ImageNet normalization, and writes
`<out-dir>/<dataset>/features/{train,test}.pt`:

```bash
# DINOv2-B core features (CLS token) for one dataset
python src/spectral_generalization.py --mode dataset --dataset aircraft \
  --backbone dinov2 --model-path models/dinov2-with-registers-base \
  --data-root data --out-dir features

# another backbone of the six-backbone sweep
python src/spectral_generalization.py --mode dataset --dataset aircraft \
  --backbone hf_vision --model-path models/siglip-base-patch16-224 \
  --data-root data --out-dir features/backbones/siglip_b16
```

Use `--backbone clip` for CLIP ViT-B/32. The ImageNet-1K study and the
fine-grained unseen datasets have dedicated extractors that hash their inputs
(`src/sdg_imagenet1k_controlled_multibackbone.py`,
`src/sdg_finegrained_features.py`).

## Cached feature format

Each `.pt` file is a PyTorch dictionary:

```python
{
    "pooled": FloatTensor[num_examples, feature_dim],
    "labels": LongTensor[num_examples],
}
```

- `pooled` stores one frozen feature vector per image; `float32` or `float64`.
- `labels` are integer class labels, consistent between the train and test
  files of a dataset.

Layout read by the evaluators (`$SDG_FEATURE_ROOT`, default `features/`):

```text
features/
  <dataset>/train.pt, test.pt                         # DINOv2-B core caches
  backbones/<backbone>/<dataset>/train.pt, test.pt    # six-backbone sweep
```

The six-backbone sweep reads its DINOv2-B column from the core caches. Each
directory may also keep the two files under a `features/` subdirectory, which
is what `spectral_generalization.py` writes.

## Rerunning the cached-feature evaluators

```bash
SDG_FEATURE_ROOT=/path/to/features bash scripts/run_cached_eval_example.sh cub200 full 0
```

Full grid of the paper (few-shot cells use seeds 0-4, full-shot uses seed 0):

```bash
for dataset in aircraft cub200 cars food101 pets cifar100 cifar10 imagenet100 tiny_imagenet; do
  for shot in 1 2 5 10 20 full; do
    if [ "$shot" = "full" ]; then
      bash scripts/run_cached_eval_example.sh "$dataset" "$shot" 0
    else
      for seed in 0 1 2 3 4; do
        bash scripts/run_cached_eval_example.sh "$dataset" "$shot" "$seed"
      done
    fi
  done
done
python src/sdg_stepb_cached_eval.py --mode aggregate --out-dir outputs
```

The aggregate is written to `outputs/stepb_cached_eval_v1/aggregate_groups.json`.
`sdg_stepb_cached_eval.py` also accepts explicit `--train-cache` and
`--test-cache` files (used for the MNIST/Fashion-MNIST/SVHN/GTSRB block).

## What is not included

- Raw dataset images and cached feature tensors.
- Pretrained checkpoints.
- Result records of the ImageNet-1K, adaptation, and class-incremental
  analyses (Appendices J-L); their code is included.
- Job-scheduler scripts and logs of the original compute environment.
