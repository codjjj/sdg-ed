#!/usr/bin/env python3
"""Controlled ImageNet-100 -> ImageNet-1K SDG audit across frozen backbones.

It produces the ImageNet-1K results of Appendix J.  Every backbone is
evaluated on the same public 224px ImageNet-1K derivative, with the same
train/validation rows, the same exact 100-class subset, and the same
train-only evaluator protocol.

`--data-root` is the tree written by sdg_materialize_imagenet1k_224.py.
Model snapshots are read with `local_files_only=True` from
`$SDG_MODEL_ROOT/<name>` (default `models/<name>`); download each `hf_repo`
below into that directory first, e.g.
`huggingface-cli download facebook/dinov2-with-registers-base --local-dir models/dinov2-with-registers-base`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Callable, Iterable

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch

from sdg_paths import MODEL_ROOT
from sdg_imagenet1k_siglip_audit import (
    REG_GRID,
    SEMANTIC_GROUPS,
    add_statistics,
    class_cluster_gap_ci,
    covariance_bundle,
    family_parameters,
    local_label_map,
    predict_cosine,
    predict_family,
    prediction_sha256,
    select_base_regularizer,
    sufficient_statistics,
    splitmix64,
    write_new,
)


SOURCE_REPO = "mlnomad/imagenet-1k-224"
SOURCE_REVISION = "d9e99f3b3b356abef801094154a0595729ac0818"
EXPECTED_TRAIN_ROWS = 1_281_167
EXPECTED_VALIDATION_ROWS = 50_000
EXPECTED_CLASSES = 1_000
EXPECTED_TRAIN_SHARDS = 33
EXPECTED_VALIDATION_SHARDS = 2
EXPECTED_SOURCE_FILES = 37
EXPECTED_SOURCE_BYTES = 16_689_509_469
REPEATED_FORWARD_MAX_ABS_TOL = 1e-5

BACKBONES = {
    "dinov2": {
        "label": "DINOv2-B/14-reg",
        "mode": "dinov2",
        "hf_repo": "facebook/dinov2-with-registers-base",
        "model_path": str(MODEL_ROOT / "dinov2-with-registers-base"),
        "expected_dim": 768,
        "batch_size": 128,
    },
    "clip_vit_b32": {
        "label": "CLIP ViT-B/32",
        "mode": "clip",
        "hf_repo": "openai/clip-vit-base-patch32",
        "model_path": str(MODEL_ROOT / "clip-vit-base-patch32"),
        "expected_dim": 512,
        "batch_size": 192,
    },
    "mae_vit_base": {
        "label": "MAE ViT-B/16 (deterministic no-mask eval)",
        "mode": "mae",
        "hf_repo": "facebook/vit-mae-base",
        "model_path": str(MODEL_ROOT / "vit-mae-base"),
        "expected_dim": 768,
        "batch_size": 128,
    },
    "siglip_b16": {
        "label": "SigLIP-B/16",
        "mode": "hf_vision",
        "hf_repo": "google/siglip-base-patch16-224",
        "model_path": str(MODEL_ROOT / "siglip-base-patch16-224"),
        "expected_dim": 768,
        "batch_size": 160,
    },
    "convnext_base": {
        "label": "ConvNeXt-B",
        "mode": "hf_vision",
        "hf_repo": "facebook/convnext-base-224",
        "model_path": str(MODEL_ROOT / "convnext-base-224"),
        "expected_dim": 1024,
        "batch_size": 160,
    },
    "vit_sup_b16": {
        "label": "Supervised ViT-B/16",
        "mode": "hf_vision",
        "hf_repo": "google/vit-base-patch16-224",
        "model_path": str(MODEL_ROOT / "vit-base-patch16-224"),
        "expected_dim": 768,
        "batch_size": 160,
    },
}

# The class names of clane9/imagenet-100 were matched against the canonical
# ImageNet-1K class order.  The two ambiguous aliases are frozen explicitly:
# "chime, bell, gong" -> 494 (chime), not 577 (gong);
# "harmonica, mouth organ, harp, mouth harp" -> 593 (harmonica), not 594 (harp).
# The order below is the clane9/imagenet-100 label order.
IMAGENET100_LABELS = (
    452, 64, 374, 236, 993, 176, 882, 904, 503, 74,
    57, 959, 953, 508, 872, 228, 122, 421, 599, 858,
    157, 449, 994, 608, 151, 209, 15, 876, 246, 766,
    455, 857, 131, 119, 234, 90, 45, 936, 479, 272,
    665, 653, 659, 158, 960, 765, 908, 703, 407, 560,
    317, 938, 724, 748, 331, 619, 120, 267, 155, 708,
    368, 772, 167, 494, 180, 431, 342, 854, 305, 54,
    268, 667, 166, 277, 662, 798, 313, 498, 299, 222,
    682, 593, 775, 674, 592, 137, 758, 717, 606, 281,
    796, 211, 620, 830, 544, 275, 242, 570, 99, 169,
)
IMAGENET100_CLASS_NAMES_SHA256 = (
    "3ce797b5c31dbc3e5e014ba81840d0ccff13b1f144b920e210c063066179ee7a"
)

ARMS = {
    "exact_imagenet100": {
        "wordnet_root": None,
        "labels": IMAGENET100_LABELS,
        "definition": "exact clane9/imagenet-100 class universe mapped to canonical ImageNet-1K labels",
    },
    **SEMANTIC_GROUPS,
}
ARM_ORDER = ("exact_imagenet100", "global_1k", "dogs", "birds", "vehicles")
SPLIT_SEEDS = (20260724, 20260725, 20260726)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, default=None)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--backbone", choices=tuple(BACKBONES), default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--stats-batch-size", type=int, default=16_384)
    parser.add_argument("--eval-batch-size", type=int, default=2_048)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--wait-seconds", type=int, default=43_200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--model-smoke", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_ready_manifest(data_root: Path) -> dict:
    marker = data_root / "_READY.json"
    if not marker.is_file():
        raise FileNotFoundError(f"source ready marker missing: {marker}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    expected = {
        "repo_id": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "train_shards": EXPECTED_TRAIN_SHARDS,
        "validation_shards": EXPECTED_VALIDATION_SHARDS,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "validation_rows": EXPECTED_VALIDATION_ROWS,
        "total_bytes": EXPECTED_SOURCE_BYTES,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"source ready manifest mismatch for {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    files = payload.get("files")
    if not isinstance(files, list) or len(files) != EXPECTED_SOURCE_FILES:
        found = len(files) if isinstance(files, list) else type(files).__name__
        raise ValueError(
            f"source ready manifest expected {EXPECTED_SOURCE_FILES} files, found {found}"
        )
    expected_paths = {".gitattributes", "README.md"}
    expected_paths.update(
        f"data/train-{index:05d}-of-{EXPECTED_TRAIN_SHARDS:05d}.parquet"
        for index in range(EXPECTED_TRAIN_SHARDS)
    )
    expected_paths.update(
        f"data/validation-{index:05d}-of-{EXPECTED_VALIDATION_SHARDS:05d}.parquet"
        for index in range(EXPECTED_VALIDATION_SHARDS)
    )
    observed_paths = {row.get("path") for row in files}
    if observed_paths != expected_paths:
        raise ValueError("source ready manifest file paths do not match the frozen shard set")
    if sum(int(row["size"]) for row in files) != EXPECTED_SOURCE_BYTES:
        raise ValueError("source ready manifest file sizes do not sum to total_bytes")
    for row in files:
        path = data_root / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != row["size"]:
            raise ValueError(f"source file size mismatch: {path}")
    return payload


def verify_source_content(data_root: Path, ready_manifest: dict) -> dict:
    verified = []
    for row in ready_manifest["files"]:
        path = data_root / row["path"]
        expected_sha256 = row.get("sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"invalid source sha256 in ready manifest: {row['path']}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"source sha256 mismatch for {row['path']}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        verified.append(
            {
                "path": row["path"],
                "size": int(row["size"]),
                "sha256": actual_sha256,
            }
        )
    return {
        "schema_version": 1,
        "kind": "sdg_imagenet1k_source_verification",
        "repo_id": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "ready_manifest_sha256": sha256_file(data_root / "_READY.json"),
        "files": verified,
    }


def distributed_source_gate(
    data_root: Path,
    output_root: Path,
    *,
    rank: int,
    wait_seconds: int,
) -> dict:
    gate = output_root / "_SOURCE_VERIFIED.json"
    ready_manifest = validate_source_ready_manifest(data_root)
    ready_sha256 = sha256_file(data_root / "_READY.json")
    if rank == 0 and not gate.exists():
        payload = verify_source_content(data_root, ready_manifest)
        write_new(
            gate,
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if gate.is_file():
            try:
                payload = json.loads(gate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                time.sleep(2)
                continue
            if payload.get("repo_id") != SOURCE_REPO:
                raise ValueError("source gate repo mismatch")
            if payload.get("revision") != SOURCE_REVISION:
                raise ValueError("source gate revision mismatch")
            if payload.get("ready_manifest_sha256") != ready_sha256:
                raise ValueError("source gate ready-manifest hash mismatch")
            files = payload.get("files")
            if not isinstance(files, list) or len(files) != EXPECTED_SOURCE_FILES:
                raise ValueError("source gate file-count mismatch")
            if files != ready_manifest["files"]:
                raise ValueError("source gate file manifest mismatch")
            return payload
        time.sleep(2)
    raise TimeoutError(f"timed out waiting for source verification gate: {gate}")


def parquet_files(data_root: Path, split: str) -> list[Path]:
    expected = EXPECTED_TRAIN_SHARDS if split == "train" else EXPECTED_VALIDATION_SHARDS
    files = sorted((data_root / "data").glob(f"{split}-*.parquet"))
    if len(files) != expected:
        raise ValueError(f"{split}: expected {expected} parquet shards, found {len(files)}")
    return files


def parquet_source_metadata(data_root: Path, *, hash_files: bool) -> dict:
    result = {
        "repo": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "root": str(data_root),
        "splits": {},
    }
    for split, expected_rows in (
        ("train", EXPECTED_TRAIN_ROWS),
        ("validation", EXPECTED_VALIDATION_ROWS),
    ):
        files = parquet_files(data_root, split)
        rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
        if rows != expected_rows:
            raise ValueError(f"{split}: expected {expected_rows} rows, found {rows}")
        shards = []
        for path in files:
            record = {
                "name": path.name,
                "bytes": int(path.stat().st_size),
            }
            if hash_files:
                record["sha256"] = sha256_file(path)
            shards.append(record)
        result["splits"][split] = {
            "rows": int(rows),
            "shards": shards,
            "sha256_verified_by_this_worker": bool(hash_files),
        }
    return result


def model_identity(model_path: Path) -> dict:
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    names = {
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
        "model.safetensors",
    }
    files = [path for path in sorted(model_path.rglob("*")) if path.is_file() and path.name in names]
    if not files:
        raise ValueError(f"no model identity files found in {model_path}")
    return {
        "path": str(model_path),
        "files": [
            {
                "name": str(path.relative_to(model_path)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }


def make_model_and_transform(
    backbone: str, device: torch.device
) -> tuple[torch.nn.Module, Callable[[Image.Image], torch.Tensor], dict]:
    from spectral_generalization import (
        make_clip_transform,
        make_hf_vision_transform,
        make_transform,
    )
    from transformers import AutoModel, CLIPModel

    spec = BACKBONES[backbone]
    model_path = Path(spec["model_path"])
    mode = spec["mode"]
    if mode == "dinov2":
        model = AutoModel.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
        transform = make_transform()
    elif mode == "clip":
        model = CLIPModel.from_pretrained(str(model_path), local_files_only=True)
        transform = make_clip_transform()
    else:
        model = AutoModel.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
        transform = make_hf_vision_transform(model_path)
        if mode == "mae":
            # ViTMAE defaults to random 75% masking even in eval mode.  A frozen
            # representation audit must be deterministic, so use all patches.
            model.config.mask_ratio = 0.0
    model = model.to(device).eval()
    return model, transform, model_identity(model_path)


def pool_features(backbone: str, model: torch.nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    from spectral_generalization import hf_vision_pooled

    mode = BACKBONES[backbone]["mode"]
    with torch.inference_mode():
        if mode == "dinov2":
            pooled = model(pixel_values=pixels).last_hidden_state[:, 0, :]
        elif mode == "clip":
            pooled = model.get_image_features(pixel_values=pixels)
        elif hasattr(model, "get_image_features"):
            pooled = model.get_image_features(pixel_values=pixels)
        else:
            output = model(pixel_values=pixels)
            pooled = hf_vision_pooled(output)
            if mode == "mae":
                expected_tokens = (224 // 16) ** 2 + 1
                if output.last_hidden_state.shape[1] != expected_tokens:
                    raise AssertionError(
                        "MAE no-mask contract failed: "
                        f"{output.last_hidden_state.shape[1]} != {expected_tokens}"
                    )
    if pooled.ndim != 2:
        raise ValueError(f"expected 2D pooled features, found {tuple(pooled.shape)}")
    return pooled.detach().cpu().float()


def repeated_forward_max_abs(
    first: torch.Tensor | np.ndarray,
    repeated: torch.Tensor | np.ndarray,
) -> float:
    first_array = np.asarray(first)
    repeated_array = np.asarray(repeated)
    if first_array.shape != repeated_array.shape:
        raise ValueError(
            f"repeated-forward shape mismatch: {first_array.shape} != "
            f"{repeated_array.shape}"
        )
    max_error = float(np.max(np.abs(first_array - repeated_array)))
    if max_error > REPEATED_FORWARD_MAX_ABS_TOL:
        raise AssertionError(
            "frozen feature extraction exceeds numerical repeatability tolerance: "
            f"{max_error} > {REPEATED_FORWARD_MAX_ABS_TOL}"
        )
    return max_error


def run_model_smoke(args: argparse.Namespace) -> Path:
    output = args.output_root / "model_smoke.json"
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {args.output_root}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    images = [
        Image.new("RGB", (224, 224), color=(30, 90, 150)),
        Image.new("RGB", (224, 224), color=(180, 70, 20)),
    ]
    rows = []
    for backbone, spec in BACKBONES.items():
        model, transform, model_meta = make_model_and_transform(backbone, device)
        pixels = torch.stack([transform(image) for image in images]).to(device)
        first = pool_features(backbone, model, pixels)
        repeated = pool_features(backbone, model, pixels)
        expected_shape = (len(images), int(spec["expected_dim"]))
        if tuple(first.shape) != expected_shape:
            raise ValueError(
                f"{backbone} smoke shape mismatch: {tuple(first.shape)} != {expected_shape}"
            )
        if not torch.isfinite(first).all():
            raise ValueError(f"{backbone} smoke produced non-finite features")
        max_error = repeated_forward_max_abs(first.numpy(), repeated.numpy())
        rows.append(
            {
                "backbone": backbone,
                "backbone_label": spec["label"],
                "feature_shape": list(first.shape),
                "feature_sha256": hashlib.sha256(first.numpy().tobytes()).hexdigest(),
                "repeated_forward_max_abs": max_error,
                "repeated_forward_max_abs_tolerance": REPEATED_FORWARD_MAX_ABS_TOL,
                "model_identity": model_meta,
            }
        )
        print(
            f"[model-smoke] backbone={backbone} shape={tuple(first.shape)} "
            f"repeat_max_abs={max_error:.3e}",
            flush=True,
        )
        del model, pixels, first, repeated
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "kind": "sdg_imagenet1k_multibackbone_model_smoke",
        "git_commit": args.commit,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "rows": rows,
    }
    write_new(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_new(args.output_root / "_SUCCESS", "ok\n")
    return output


def decode_image(record: dict, transform: Callable[[Image.Image], torch.Tensor]) -> torch.Tensor:
    image_bytes = record.get("bytes")
    if image_bytes is None:
        raise ValueError("public derivative row does not contain embedded image bytes")
    with Image.open(BytesIO(image_bytes)) as image:
        return transform(image.convert("RGB"))


def iter_parquet_batches(
    files: Iterable[Path],
    batch_size: int,
) -> Iterable[tuple[list[dict], np.ndarray]]:
    for path in files:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema.names)
        if not {"bytes", "label"}.issubset(names):
            # Arrow flattens the Image struct into bytes/path in schema.names.
            if "image" not in names or "label" not in names:
                raise ValueError(f"unexpected parquet schema for {path}: {parquet.schema}")
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["image", "label"]):
            records = batch.column(0).to_pylist()
            labels = np.asarray(batch.column(1).to_numpy(), dtype=np.int64)
            yield records, labels


def extract_split(
    backbone: str,
    model: torch.nn.Module,
    transform: Callable[[Image.Image], torch.Tensor],
    files: list[Path],
    expected_rows: int,
    expected_dim: int,
    output_dir: Path,
    *,
    device: torch.device,
    batch_size: int,
    decode_workers: int,
) -> dict:
    feature_path = output_dir / "features.npy"
    label_path = output_dir / "labels.npy"
    if feature_path.exists() or label_path.exists():
        raise FileExistsError(f"refusing to overwrite feature cache in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32, shape=(expected_rows, expected_dim)
    )
    labels_out = np.lib.format.open_memmap(
        label_path, mode="w+", dtype=np.int16, shape=(expected_rows,)
    )
    offset = 0
    started = time.time()
    repeated_forward_error = None
    with ThreadPoolExecutor(max_workers=decode_workers) as pool:
        for records, labels in iter_parquet_batches(files, batch_size):
            tensors = list(pool.map(lambda item: decode_image(item, transform), records))
            pixels = torch.stack(tensors).to(device=device, non_blocking=True)
            batch_features = pool_features(backbone, model, pixels).numpy()
            if offset == 0:
                repeated = pool_features(backbone, model, pixels).numpy()
                repeated_forward_error = repeated_forward_max_abs(
                    batch_features, repeated
                )
            stop = offset + len(labels)
            if batch_features.shape != (len(labels), expected_dim):
                raise ValueError(
                    f"feature shape mismatch: {batch_features.shape} "
                    f"!= {(len(labels), expected_dim)}"
                )
            if labels.min() < 0 or labels.max() >= EXPECTED_CLASSES:
                raise ValueError(f"label range mismatch: {labels.min()}..{labels.max()}")
            features[offset:stop] = batch_features
            labels_out[offset:stop] = labels
            offset = stop
            if offset == len(labels) or offset % (batch_size * 200) < batch_size:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[extract] backbone={backbone} rows={offset}/{expected_rows} "
                    f"rate={offset/elapsed:.1f}/s",
                    flush=True,
                )
    features.flush()
    labels_out.flush()
    if offset != expected_rows:
        raise ValueError(f"row count mismatch: {offset} != {expected_rows}")
    counts = np.bincount(np.asarray(labels_out, dtype=np.int64), minlength=EXPECTED_CLASSES)
    if len(counts) != EXPECTED_CLASSES or np.any(counts <= 0):
        raise ValueError("not all ImageNet-1K labels are represented")
    return {
        "features_path": str(feature_path),
        "labels_path": str(label_path),
        "rows": int(expected_rows),
        "dim": int(expected_dim),
        "class_count_min": int(counts.min()),
        "class_count_median": float(np.median(counts)),
        "class_count_max": int(counts.max()),
        "repeated_forward_max_abs": repeated_forward_error,
        "repeated_forward_max_abs_tolerance": REPEATED_FORWARD_MAX_ABS_TOL,
        "feature_sha256": sha256_file(feature_path),
        "label_sha256": sha256_file(label_path),
    }


def stable_train_validation_mask(
    labels: np.ndarray,
    seed: int,
    *,
    fraction: float = 0.20,
    cap_per_class: int = 50,
) -> np.ndarray:
    """Return True for the train-only selection split, class balanced."""

    labels = np.asarray(labels, dtype=np.int64)
    selected = np.zeros(len(labels), dtype=bool)
    seed_value = np.uint64(seed) ^ np.uint64(0xA0761D6478BD642F)
    original_indices = np.arange(len(labels), dtype=np.uint64)
    class_order = np.argsort(labels, kind="stable")
    class_counts = np.bincount(labels, minlength=EXPECTED_CLASSES)
    class_offsets = np.concatenate(([0], np.cumsum(class_counts)))
    for cls in range(EXPECTED_CLASSES):
        indices = class_order[class_offsets[cls] : class_offsets[cls + 1]]
        if len(indices) < 2:
            raise ValueError(f"class {cls} has too few training rows")
        n_validation = min(cap_per_class, max(1, int(round(fraction * len(indices)))))
        keys = splitmix64(original_indices[indices] ^ seed_value)
        ordered = indices[np.argsort(keys, kind="stable")]
        selected[ordered[:n_validation]] = True
    return selected


def _arm_indices(labels: np.ndarray, class_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mapping = local_label_map(class_labels)
    local = mapping[np.asarray(labels, dtype=np.int64)]
    return local, np.flatnonzero(local >= 0)


def prepare_refit_arm(
    features: np.ndarray,
    labels: np.ndarray,
    class_labels: np.ndarray,
    *,
    device: torch.device,
    stats_batch_size: int,
) -> dict:
    local, indices = _arm_indices(labels, class_labels)
    stats = sufficient_statistics(
        features,
        local,
        indices,
        len(class_labels),
        device=device,
        batch_size=stats_batch_size,
    )
    bundle = covariance_bundle(stats)
    return {
        "local_labels": local,
        "indices": indices,
        "bundle": bundle,
        "family_parameters": {
            family: family_parameters(family, bundle) for family in ("wcos", "lda")
        },
    }


def evaluate_arm_seed(
    name: str,
    class_labels: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    validation_mask: np.ndarray,
    refit: dict,
    *,
    device: torch.device,
    stats_batch_size: int,
    eval_batch_size: int,
    n_boot: int,
    seed: int,
) -> tuple[dict, dict]:
    train_arm = refit["local_labels"] >= 0
    inner_indices = np.flatnonzero(train_arm & ~validation_mask)
    selection_indices = np.flatnonzero(train_arm & validation_mask)
    inner_stats = sufficient_statistics(
        train_features,
        refit["local_labels"],
        inner_indices,
        len(class_labels),
        device=device,
        batch_size=stats_batch_size,
    )
    inner_bundle = covariance_bundle(inner_stats)
    selections = {
        family: select_base_regularizer(
            family,
            train_features,
            selection_indices,
            train_labels[selection_indices],
            family_parameters(family, inner_bundle),
            class_labels,
            device=device,
            batch_size=eval_batch_size,
            score_dtype=torch.float32,
        )
        for family in ("wcos", "lda")
    }

    _, test_indices = _arm_indices(test_labels, class_labels)
    truth = test_labels[test_indices]
    predictions = {
        "cos": predict_cosine(
            test_features,
            test_indices,
            refit["bundle"]["means"],
            class_labels,
            device=device,
            batch_size=eval_batch_size,
            score_dtype=torch.float32,
        )
    }
    actual_regularizers = {}
    for family in ("wcos", "lda"):
        predictions[family], actual_regularizers[family] = predict_family(
            family,
            test_features,
            test_indices,
            refit["family_parameters"][family],
            selections[family]["selected_base"],
            class_labels,
            device=device,
            batch_size=eval_batch_size,
            score_dtype=torch.float32,
        )
    accuracies = {
        method: float(np.mean(prediction == truth))
        for method, prediction in predictions.items()
    }
    result = {
        "arm": name,
        "seed": int(seed),
        "n_classes": int(len(class_labels)),
        "split_counts": {
            "inner": int(len(inner_indices)),
            "selection": int(len(selection_indices)),
            "refit_support": int(len(refit["indices"])),
            "test": int(len(test_indices)),
        },
        "selection": selections,
        "refit_actual_regularizers": actual_regularizers,
        "test_accuracies": accuracies,
        "fixed_evaluator_gaps": {
            family: class_cluster_gap_ci(
                truth,
                predictions["cos"],
                predictions[family],
                class_labels,
                n_boot=n_boot,
                seed=seed + (101 if family == "wcos" else 211),
            )
            for family in ("wcos", "lda")
        },
        "best_whitening_minus_cos_pp": float(
            100.0 * (max(accuracies["wcos"], accuracies["lda"]) - accuracies["cos"])
        ),
        "prediction_sha256": {
            method: prediction_sha256(prediction)
            for method, prediction in predictions.items()
        },
    }
    arrays = {
        "indices": test_indices,
        "truth": truth,
        **predictions,
    }
    return result, arrays


def paired_shared100_scale_contrast(
    exact: dict,
    global_: dict,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    exact_indices = np.asarray(exact["indices"])
    if not np.array_equal(exact_indices, np.flatnonzero(np.isin(global_["truth"], IMAGENET100_LABELS))):
        raise AssertionError("shared-100 test row identity mismatch")
    truth = exact["truth"]
    if not np.array_equal(truth, global_["truth"][exact_indices]):
        raise AssertionError("shared-100 test labels mismatch")
    output = {}
    rng = np.random.RandomState(seed + 919)
    classes = np.asarray(IMAGENET100_LABELS, dtype=np.int64)
    for family in ("wcos", "lda"):
        global_diff = (
            (global_[family][exact_indices] == truth).astype(np.float64)
            - (global_["cos"][exact_indices] == truth).astype(np.float64)
        )
        exact_diff = (
            (exact[family] == truth).astype(np.float64)
            - (exact["cos"] == truth).astype(np.float64)
        )
        difference = global_diff - exact_diff
        class_sums = np.asarray([difference[truth == cls].sum() for cls in classes])
        class_counts = np.asarray([np.sum(truth == cls) for cls in classes])
        samples = []
        for _ in range(n_boot):
            draw = rng.randint(0, len(classes), size=len(classes))
            samples.append(100.0 * class_sums[draw].sum() / class_counts[draw].sum())
        output[family] = {
            "point_pp": float(100.0 * difference.mean()),
            "ci95_pp": [
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)),
            ],
            "estimand": (
                "(1K-way evaluator gap - exact-100-way evaluator gap) on the "
                "same validation images from the exact 100 classes"
            ),
            "bootstrap_unit": "shared ImageNet class",
            "n_boot": int(n_boot),
        }
    return output


def paired_subgroup_global_gap_contrast(
    subgroup: dict,
    global_: dict,
    class_labels: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    subgroup_indices = np.asarray(subgroup["indices"])
    expected_indices = np.flatnonzero(np.isin(global_["truth"], class_labels))
    if not np.array_equal(subgroup_indices, expected_indices):
        raise AssertionError("subgroup/global test row identity mismatch")
    truth = np.asarray(subgroup["truth"])
    if not np.array_equal(truth, np.asarray(global_["truth"])[subgroup_indices]):
        raise AssertionError("subgroup/global test labels mismatch")

    output = {}
    rng = np.random.RandomState(seed + 1297)
    class_labels = np.asarray(class_labels, dtype=np.int64)
    for family in ("wcos", "lda"):
        subgroup_gap = (
            (np.asarray(subgroup[family]) == truth).astype(np.float64)
            - (np.asarray(subgroup["cos"]) == truth).astype(np.float64)
        )
        global_gap = (
            (np.asarray(global_[family])[subgroup_indices] == truth).astype(np.float64)
            - (np.asarray(global_["cos"])[subgroup_indices] == truth).astype(np.float64)
        )
        difference = subgroup_gap - global_gap
        class_sums = np.asarray(
            [difference[truth == cls].sum() for cls in class_labels],
            dtype=np.float64,
        )
        class_counts = np.asarray(
            [np.sum(truth == cls) for cls in class_labels],
            dtype=np.int64,
        )
        samples = []
        for _ in range(n_boot):
            draw = rng.randint(0, len(class_labels), size=len(class_labels))
            samples.append(100.0 * class_sums[draw].sum() / class_counts[draw].sum())
        output[family] = {
            "point_pp": float(100.0 * difference.mean()),
            "ci95_pp": [
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)),
            ],
            "estimand": (
                "within-subgroup evaluator gap minus global-1K evaluator gap "
                "on the same subgroup validation images"
            ),
            "bootstrap_unit": "subgroup ImageNet class",
            "n_boot": int(n_boot),
        }
    return output


def copy_feature_artifacts(local_root: Path, output_root: Path, metadata: dict) -> dict:
    destination = output_root / "feature_cache"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=False)
    copied = {}
    for split in ("train", "validation"):
        copied[split] = {}
        source_dir = local_root / split
        target_dir = destination / split
        target_dir.mkdir()
        for name in ("features.npy", "labels.npy"):
            source = source_dir / name
            target = target_dir / name
            shutil.copyfile(source, target)
            if target.stat().st_size != source.stat().st_size:
                raise IOError(f"copied feature size mismatch: {target}")
            copied[split][name] = {
                "path": str(target),
                "bytes": int(target.stat().st_size),
                "sha256": metadata[split][
                    "feature_sha256" if name == "features.npy" else "label_sha256"
                ],
            }
    return copied


def run_backbone(
    args: argparse.Namespace,
    backbone: str,
    source_verification: dict,
) -> Path:
    spec = BACKBONES[backbone]
    ready_manifest = validate_source_ready_manifest(args.data_root)
    output_root = args.output_root / backbone
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    local_parent = args.local_root or Path(tempfile.mkdtemp(prefix=f"sdg-e13-{backbone}-"))
    local_root = local_parent / backbone
    if local_root.exists():
        raise FileExistsError(local_root)
    local_root.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, transform, model_meta = make_model_and_transform(backbone, device)
    batch_size = int(args.batch_size or spec["batch_size"])
    feature_meta = {}
    for split, rows in (
        ("train", EXPECTED_TRAIN_ROWS),
        ("validation", EXPECTED_VALIDATION_ROWS),
    ):
        feature_meta[split] = extract_split(
            backbone,
            model,
            transform,
            parquet_files(args.data_root, split),
            rows,
            int(spec["expected_dim"]),
            local_root / split,
            device=device,
            batch_size=batch_size,
            decode_workers=args.decode_workers,
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_features = np.load(local_root / "train/features.npy", mmap_mode="r")
    train_labels = np.load(local_root / "train/labels.npy", mmap_mode="r")
    test_features = np.load(local_root / "validation/features.npy", mmap_mode="r")
    test_labels = np.load(local_root / "validation/labels.npy", mmap_mode="r")
    arms = {}
    refits = {}
    for name in ARM_ORDER:
        class_labels = np.asarray(ARMS[name]["labels"], dtype=np.int64)
        refits[name] = prepare_refit_arm(
            train_features,
            train_labels,
            class_labels,
            device=device,
            stats_batch_size=args.stats_batch_size,
        )
        arms[name] = []
    contrasts = []
    for seed in SPLIT_SEEDS:
        validation_mask = stable_train_validation_mask(train_labels, seed)
        arrays = {}
        for name in ARM_ORDER:
            result, arrays[name] = evaluate_arm_seed(
                name,
                np.asarray(ARMS[name]["labels"], dtype=np.int64),
                train_features,
                train_labels,
                test_features,
                test_labels,
                validation_mask,
                refits[name],
                device=device,
                stats_batch_size=args.stats_batch_size,
                eval_batch_size=args.eval_batch_size,
                n_boot=args.n_boot,
                seed=seed,
            )
            arms[name].append(result)
        contrasts.append(
            {
                "seed": int(seed),
                "fixed_evaluator_contrasts": paired_shared100_scale_contrast(
                    arrays["exact_imagenet100"],
                    arrays["global_1k"],
                    n_boot=args.n_boot,
                    seed=seed,
                ),
                "paired_subgroup_global_contrasts": {
                    name: paired_subgroup_global_gap_contrast(
                        arrays[name],
                        arrays["global_1k"],
                        np.asarray(ARMS[name]["labels"], dtype=np.int64),
                        n_boot=args.n_boot,
                        seed=seed + 3001 * (index + 1),
                    )
                    for index, name in enumerate(("dogs", "birds", "vehicles"))
                },
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_root.mkdir(parents=True, exist_ok=False)
    copied = copy_feature_artifacts(local_root, output_root, feature_meta)
    payload = {
        "schema_version": 1,
        "kind": "sdg_imagenet1k_controlled_multibackbone_backbone_result",
        "git_commit": args.commit,
        "backbone": backbone,
        "backbone_label": spec["label"],
        "backbone_spec": spec,
        "model_identity": model_meta,
        "source": parquet_source_metadata(
            args.data_root,
            hash_files=int(os.environ.get("RANK", args.rank or 0)) == 0,
        ),
        "source_ready_manifest_sha256": sha256_file(args.data_root / "_READY.json"),
        "source_ready_manifest_total_bytes": ready_manifest["total_bytes"],
        "source_verification": source_verification,
        "feature_cache": copied,
        "feature_metadata": feature_meta,
        "protocol": {
            "source_split": {
                "support": "public derivative train (1,281,167 rows)",
                "test": "public derivative validation (50,000 rows)",
            },
            "selection": (
                "20% class-wise inner split capped at 50/class; validation labels "
                "select relative-trace regularizer; refit on every train row"
            ),
            "split_seeds": list(SPLIT_SEEDS),
            "regularization_grid": list(REG_GRID),
            "score_dtype": "float32",
            "covariance_and_eigendecomposition_dtype": "float64",
            "bootstrap_unit": "ImageNet class",
            "n_boot": int(args.n_boot),
            "exact_imagenet100_labels": list(IMAGENET100_LABELS),
            "exact_imagenet100_class_names_sha256": IMAGENET100_CLASS_NAMES_SHA256,
            "ambiguity_resolutions": {
                "chime, bell, gong": 494,
                "harmonica, mouth organ, harp, mouth harp": 593,
            },
            "arms": {
                name: {
                    "n_classes": len(ARMS[name]["labels"]),
                    "wordnet_root": ARMS[name].get("wordnet_root"),
                    "definition": ARMS[name].get("definition"),
                }
                for name in ARM_ORDER
            },
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "platform": platform.platform(),
            "rank": int(os.environ.get("RANK", args.rank or 0)),
            "world_size": int(os.environ.get("WORLD_SIZE", args.world_size or 1)),
        },
        "scope_limits": [
            "post-submission evidence",
            "public 224px ImageNet-1K derivative, not redistributed official ImageNet",
            "frozen encoders; no representation training",
            "subgroup tasks classify only among classes inside each subgroup",
            "absolute accuracy is not directly equated with the submitted ImageNet-100 cell",
        ],
        "arms": arms,
        "paired_shared100_scale_contrasts": contrasts,
    }
    write_new(
        output_root / "result.json",
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    write_new(output_root / "_SUCCESS", "ok\n")
    if args.local_root is None:
        shutil.rmtree(local_parent)
    return output_root / "result.json"


def summarize_backbones(payloads: list[dict]) -> dict:
    rows = []
    for payload in payloads:
        for seed_index, seed in enumerate(SPLIT_SEEDS):
            row = {
                "backbone": payload["backbone"],
                "backbone_label": payload["backbone_label"],
                "seed": int(seed),
            }
            for arm in ("exact_imagenet100", "global_1k", "dogs", "birds", "vehicles"):
                result = payload["arms"][arm][seed_index]
                row[arm] = {
                    "cos": result["test_accuracies"]["cos"],
                    "wcos": result["test_accuracies"]["wcos"],
                    "lda": result["test_accuracies"]["lda"],
                    "wcos_gap_pp": result["fixed_evaluator_gaps"]["wcos"]["point_pp"],
                    "lda_gap_pp": result["fixed_evaluator_gaps"]["lda"]["point_pp"],
                }
            row["paired_shared100_scale_contrast"] = payload[
                "paired_shared100_scale_contrasts"
            ][seed_index]["fixed_evaluator_contrasts"]
            row["paired_subgroup_global_contrasts"] = payload[
                "paired_shared100_scale_contrasts"
            ][seed_index]["paired_subgroup_global_contrasts"]
            rows.append(row)
    return {
        "schema_version": 1,
        "kind": "sdg_imagenet1k_controlled_multibackbone_aggregate",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "backbones": [payload["backbone"] for payload in payloads],
        "split_seeds": list(SPLIT_SEEDS),
        "rows": rows,
    }


def aggregate_report(payload: dict) -> str:
    lines = [
        "# Controlled ImageNet-100 → ImageNet-1K multi-backbone SDG audit",
        "",
        "Evidence on one fixed public 224px ImageNet-1K derivative.",
        "Cells are mean pp over three train-only selection seeds.",
        "",
        "| backbone | exact-100 WCos/LDA gap | global-1K WCos/LDA gap | dogs WCos/LDA gap | birds WCos/LDA gap | vehicles WCos/LDA gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for backbone in payload["backbones"]:
        rows = [row for row in payload["rows"] if row["backbone"] == backbone]
        label = rows[0]["backbone_label"]
        cells = []
        for arm in ("exact_imagenet100", "global_1k", "dogs", "birds", "vehicles"):
            wcos = np.mean([row[arm]["wcos_gap_pp"] for row in rows])
            lda = np.mean([row[arm]["lda_gap_pp"] for row in rows])
            cells.append(f"{wcos:+.2f}/{lda:+.2f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "- Exact-100 and global-1K use identical source rows, preprocessing, split rule, and evaluator grid within each backbone.",
            "- Dogs/birds/vehicles are within-subgroup classification arms.",
            "- Absolute accuracies are not presented as the core-suite ImageNet-100 Table 1 cell because this source is a resized public derivative.",
        ]
    )
    return "\n".join(lines) + "\n"


def wait_and_aggregate(args: argparse.Namespace) -> Path:
    deadline = time.time() + args.wait_seconds
    paths = [args.output_root / name / "result.json" for name in BACKBONES]
    while time.time() < deadline:
        ready = [path for path in paths if path.exists()]
        print(f"[aggregate-wait] ready={len(ready)}/{len(paths)}", flush=True)
        if len(ready) == len(paths):
            break
        time.sleep(30)
    else:
        missing = [str(path) for path in paths if not path.exists()]
        raise TimeoutError(f"timed out waiting for backbone results: {missing}")
    payloads = [json.loads(path.read_text()) for path in paths]
    aggregate = summarize_backbones(payloads)
    write_new(
        args.output_root / "aggregate.json",
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    write_new(args.output_root / "aggregate.md", aggregate_report(aggregate))
    write_new(args.output_root / "_SUCCESS", "ok\n")
    return args.output_root / "aggregate.json"


def main() -> None:
    args = parse_args()
    if args.model_smoke:
        result = run_model_smoke(args)
        print(json.dumps({"model_smoke": str(result)}), flush=True)
        return
    rank = int(os.environ.get("RANK", args.rank if args.rank is not None else 0))
    world_size = int(
        os.environ.get("WORLD_SIZE", args.world_size if args.world_size is not None else 1)
    )
    if args.backbone is None:
        if world_size != len(BACKBONES):
            raise ValueError(f"matrix mode requires world_size={len(BACKBONES)}, found {world_size}")
        backbone = tuple(BACKBONES)[rank]
    else:
        backbone = args.backbone
    source_verification = distributed_source_gate(
        args.data_root,
        args.output_root,
        rank=rank,
        wait_seconds=args.wait_seconds,
    )
    result = run_backbone(args, backbone, source_verification)
    print(json.dumps({"backbone": backbone, "result": str(result)}), flush=True)
    if rank == 0 and not args.extract_only and world_size == len(BACKBONES):
        aggregate = wait_and_aggregate(args)
        print(json.dumps({"aggregate": str(aggregate)}), flush=True)


if __name__ == "__main__":
    main()
