#!/usr/bin/env python3
"""Canonical DINOv2 CLS SimpleCIL reproduction and matched CUB FSCIL audit.

Arm A reproduces the SimpleCIL cosine-prototype baseline on CUB-200 (ten
20-class tasks, class orders seeded as in the public SimpleCIL code) with the
CLS token of DINOv2 ViT-B/14 *without* registers (facebook/dinov2-base at a
pinned revision). Arm B runs the 11-session CUB-200 FSCIL protocol on the same
CLS features. `--cub-root` is the official CUB_200_2011 directory and
`--index-root` the FSCIL session files (default: splits/fscil_cub200).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sdg_paths import MODEL_ROOT, RELEASE_ROOT
from sdg_fscil_prototype_audit import (
    BASE_CLASSES,
    METHODS,
    NUM_CLASSES,
    arm_metrics,
    array_sha256,
    file_sha256,
    fit_base_geometries,
    official_bootstrap,
    official_support_indices,
    path_key,
    resampled_support_indices,
    seed_paired_summary,
    support_arrays,
)


MODEL_REPO = "facebook/dinov2-base"
MODEL_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
ANACP_COMMIT = "7801c59284e2880c0c911bb6e0c7afc0b15f1df7"
SIMPLECIL_COMMIT = "a2e6b71f6a2fed4421609a0917eea861fa387c76"
ARM_A_SEEDS = (0, 1, 2)
ARM_B_SEEDS = tuple(range(20))
ARM_B_METHODS = ("cosine", "wcos_base_frozen", "lda_base_frozen")
EXPECTED_IMAGES = 11788
EXPECTED_TRAIN = 5994
EXPECTED_TEST = 5794


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha256(payload: object) -> str:
    return sha256_bytes(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    )


def ensure_new_output_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing non-empty output root: {path}")
    path.mkdir(parents=True, exist_ok=True)


def write_json_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_torch_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        shutil.copy2(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_npz_new(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        shutil.copy2(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_cub_inventory(root: Path, *, require_files: bool = True) -> list[dict]:
    root = Path(root)
    metadata = {}
    for filename in ("images.txt", "image_class_labels.txt", "train_test_split.txt"):
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            key, value = line.split(maxsplit=1)
            rows[int(key)] = value.strip()
        metadata[filename] = rows
    ids = sorted(metadata["images.txt"])
    if ids != list(range(1, EXPECTED_IMAGES + 1)):
        raise ValueError("CUB image ids must be exactly 1..11788")
    if any(set(rows) != set(ids) for rows in metadata.values()):
        raise ValueError("CUB metadata id sets differ")
    records = []
    for image_id in ids:
        relative = metadata["images.txt"][image_id]
        label = int(metadata["image_class_labels.txt"][image_id]) - 1
        split_token = int(metadata["train_test_split.txt"][image_id])
        if label not in range(NUM_CLASSES) or split_token not in (0, 1):
            raise ValueError(f"invalid CUB metadata at image id {image_id}")
        image_path = root / "images" / relative
        if require_files and not image_path.is_file():
            raise FileNotFoundError(image_path)
        records.append(
            {
                "image_id": image_id,
                "relative_path": relative,
                "path": str(image_path),
                "label": label,
                "split": "train" if split_token == 1 else "test",
            }
        )
    train = [row for row in records if row["split"] == "train"]
    test = [row for row in records if row["split"] == "test"]
    if len(train) != EXPECTED_TRAIN or len(test) != EXPECTED_TEST:
        raise ValueError(f"CUB split counts mismatch: train={len(train)} test={len(test)}")
    if set(row["label"] for row in train) != set(range(NUM_CLASSES)):
        raise ValueError("CUB train split does not cover 200 classes")
    if set(row["label"] for row in test) != set(range(NUM_CLASSES)):
        raise ValueError("CUB test split does not cover 200 classes")
    return records


class CubImageDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, int, str, str, int]:
        row = self.records[index]
        data = Path(row["path"]).read_bytes()
        with Image.open(io.BytesIO(data)) as source:
            image = source.convert("RGB")
        return image, row["label"], row["relative_path"], sha256_bytes(data), len(data)


def make_collate(processor):
    def collate(batch):
        images, labels, paths, hashes, sizes = zip(*batch)
        processed = processor(images=list(images), return_tensors="pt")
        pixel_values = processed["pixel_values"]
        if pixel_values.ndim != 4 or pixel_values.shape[0] != len(batch):
            raise ValueError(f"processor batch shape mismatch: {tuple(pixel_values.shape)}")
        return pixel_values, np.asarray(labels), list(paths), list(hashes), list(sizes)

    return collate


def select_cls_token(last_hidden_state: torch.Tensor) -> torch.Tensor:
    if last_hidden_state.ndim != 3 or last_hidden_state.shape[1] < 1:
        raise ValueError(f"invalid DINOv2 hidden state shape: {tuple(last_hidden_state.shape)}")
    return last_hidden_state[:, 0]


def materialize_model(model_root: Path, repo_id: str, revision: str) -> dict:
    model_root = Path(model_root)
    required = ("config.json", "preprocessor_config.json", "model.safetensors")
    if not all((model_root / name).is_file() for name in required):
        from huggingface_hub import snapshot_download

        model_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id,
            revision=revision,
            local_dir=model_root,
            allow_patterns=["*.json", "*.safetensors", "README.md", ".gitattributes"],
            ignore_patterns=["*.bin", "*.msgpack"],
        )
    missing = [name for name in required if not (model_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"model materialization missing files: {missing}")
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "dinov2" or config.get("hidden_size") != 768:
        raise ValueError(f"unexpected DINOv2-base config: {config}")
    files = {}
    for path in sorted(model_root.iterdir()):
        if path.is_file():
            files[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return {
        "repo_id": repo_id,
        "revision": revision,
        "root": str(model_root),
        "config": config,
        "files": files,
    }


def extract_split(
    records: list[dict], processor, model, device: torch.device, batch_size: int, workers: int
) -> dict:
    loader = DataLoader(
        CubImageDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=make_collate(processor),
    )
    features, labels, paths, hashes, sizes = [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for pixel_values, batch_labels, batch_paths, batch_hashes, batch_sizes in loader:
            pixel_values = pixel_values.to(device=device, dtype=torch.float32, non_blocking=True)
            hidden = model(pixel_values=pixel_values).last_hidden_state
            cls = select_cls_token(hidden).float().cpu()
            if cls.shape != (len(batch_labels), 768):
                raise ValueError(f"feature batch shape mismatch: {tuple(cls.shape)}")
            features.append(cls)
            labels.append(torch.from_numpy(batch_labels).long())
            paths.extend(batch_paths)
            hashes.extend(batch_hashes)
            sizes.extend(int(value) for value in batch_sizes)
    feature = torch.cat(features, dim=0).contiguous()
    label = torch.cat(labels, dim=0).contiguous()
    expected_paths = [row["relative_path"] for row in records]
    expected_labels = torch.tensor([row["label"] for row in records], dtype=torch.long)
    if paths != expected_paths or not torch.equal(label, expected_labels):
        raise AssertionError("feature extraction changed image order or labels")
    if feature.shape != (len(records), 768) or not torch.isfinite(feature).all():
        raise ValueError("invalid extracted CLS feature matrix")
    return {
        "pooled": feature,
        "labels": label,
        "paths": paths,
        "image_sha256": hashes,
        "image_bytes": sizes,
    }


def simplecil_task_classes(seed: int, n_tasks: int = 10) -> list[list[int]]:
    classes = list(range(NUM_CLASSES))
    random.Random(seed).shuffle(classes)
    if NUM_CLASSES % n_tasks:
        raise ValueError("class count must be divisible by task count")
    tasks = [classes[index::n_tasks] for index in range(n_tasks)]
    if any(len(task) != NUM_CLASSES // n_tasks for task in tasks):
        raise AssertionError("unequal SimpleCIL task sizes")
    if sorted(label for task in tasks for label in task) != list(range(NUM_CLASSES)):
        raise AssertionError("SimpleCIL tasks are not a class partition")
    return tasks


def normalized_class_means(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    means = np.stack([features[labels == label].mean(axis=0) for label in range(NUM_CLASSES)])
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("zero class prototype")
    return means / norms


def normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("zero query feature")
    return features / norms


def evaluate_simplecil_seed(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> tuple[dict, list[np.ndarray]]:
    tasks = simplecil_task_classes(seed)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = normalize_rows(test_features.astype(np.float64)) @ normalized_class_means(
            train_features.astype(np.float64), train_labels
        ).T
    if not np.isfinite(scores).all():
        raise FloatingPointError("non-finite SimpleCIL cosine scores")
    matrix = np.full((len(tasks), len(tasks)), np.nan, dtype=np.float64)
    predictions = []
    sessions = []
    for session in range(len(tasks)):
        seen = [label for task in tasks[: session + 1] for label in task]
        eval_mask = np.isin(test_labels, seen)
        prediction = np.asarray(seen, dtype=np.int64)[
            np.argmax(scores[eval_mask][:, seen], axis=1)
        ]
        stored = np.full(len(test_labels), -1, dtype=np.int16)
        stored[eval_mask] = prediction.astype(np.int16)
        predictions.append(stored)
        for task_index in range(session + 1):
            mask = np.isin(test_labels, tasks[task_index])
            matrix[session, task_index] = float((stored[mask] == test_labels[mask]).mean())
        row_average = float(np.mean(matrix[session, : session + 1]))
        sessions.append(
            {
                "session": session,
                "learned_task_classes": tasks[session],
                "seen_classes": seen,
                "per_task_accuracy": matrix[session, : session + 1].tolist(),
                "row_average_accuracy": row_average,
                "num_eval": int(eval_mask.sum()),
                "prediction_sha256": array_sha256(stored),
            }
        )
    row_averages = [row["row_average_accuracy"] for row in sessions]
    return (
        {
            "seed": int(seed),
            "task_classes": tasks,
            "accuracy_matrix": [[None if np.isnan(v) else float(v) for v in row] for row in matrix],
            "sessions": sessions,
            "average_accuracy": float(np.mean(row_averages)),
            "final_accuracy": row_averages[-1],
        },
        predictions,
    )


def mean_sample_sd(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        raise ValueError("sample SD requires at least two values")
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "values": array.tolist(),
    }


def run_arm_b(train: dict, test: dict, index_root: Path, bootstrap_seed: int) -> tuple[dict, dict, dict]:
    train_features = train["pooled"].numpy()
    train_labels = train["labels"].numpy()
    test_features = test["pooled"].numpy()
    test_labels = test["labels"].numpy()
    support, order, official_manifest = official_support_indices(
        train["paths"], train_labels, index_root
    )
    base_rows = np.concatenate([support[label] for label in range(BASE_CLASSES)])
    geometry = fit_base_geometries(train_features, train_labels, base_rows, validation_seed=0)
    official_features, official_labels = support_arrays(train_features, train_labels, support)
    metrics_all, predictions_all = arm_metrics(
        official_features, official_labels, test_features, test_labels, geometry, order
    )
    metrics = {name: metrics_all[name] for name in ARM_B_METHODS}
    predictions = {name: predictions_all[name] for name in ARM_B_METHODS}
    bootstrap_all = official_bootstrap(
        metrics_all, predictions_all, test_labels, n_boot=2000, seed=bootstrap_seed
    )
    bootstrap = {
        key: value for key, value in bootstrap_all.items() if key.split("_minus_")[0] in ARM_B_METHODS
    }
    rows = []
    support_manifests = [official_manifest]
    prediction_arrays = {}
    for method, method_predictions in predictions.items():
        for session, prediction in enumerate(method_predictions):
            prediction_arrays[f"arm_b_official__{method}__s{session:02d}"] = prediction
    for seed in ARM_B_SEEDS:
        sampled, sampled_order, manifest = resampled_support_indices(
            train["paths"], train_labels, support, seed
        )
        sampled_features, sampled_labels = support_arrays(train_features, train_labels, sampled)
        seed_metrics_all, seed_predictions_all = arm_metrics(
            sampled_features,
            sampled_labels,
            test_features,
            test_labels,
            geometry,
            sampled_order,
        )
        seed_metrics = {name: seed_metrics_all[name] for name in ARM_B_METHODS}
        rows.append({"seed": seed, "class_order": sampled_order, "metrics": seed_metrics})
        support_manifests.append(manifest)
        for method in ARM_B_METHODS:
            for session, prediction in enumerate(seed_predictions_all[method]):
                prediction_arrays[f"arm_b_seed{seed:02d}__{method}__s{session:02d}"] = prediction
    paired = {}
    for challenger in ARM_B_METHODS[1:]:
        paired[challenger + "_minus_cosine"] = {}
        for metric in (
            "average_accuracy",
            "final_accuracy",
            "final_base_accuracy",
            "final_incremental_accuracy",
        ):
            values = [
                100.0 * (row["metrics"][challenger][metric] - row["metrics"]["cosine"][metric])
                for row in rows
            ]
            paired[challenger + "_minus_cosine"][metric] = seed_paired_summary(
                values, seed=bootstrap_seed + 2039 * METHODS.index(challenger), n_boot=10000
            )
    geometry_manifest = {
        "validation_split": geometry["validation_split"],
        "base_rows": geometry["base_rows"],
        "base_rows_sha256": geometry["base_rows_sha256"],
        "lda": {key: value for key, value in geometry["lda"].items() if key != "inverse"},
        "wcos": {
            key: value
            for key, value in geometry["wcos"].items()
            if key not in {"mean", "vectors", "scale"}
        },
    }
    return (
        {
            "protocol": "CUB official 100-base + 10x10-way/5-shot",
            "methods": list(ARM_B_METHODS),
            "official": {
                "class_order": order,
                "metrics": metrics,
                "paired_test_example_bootstrap": bootstrap,
            },
            "resampled_sensitivity": {
                "seeds": list(ARM_B_SEEDS),
                "rows": rows,
                "paired_seed_summary": paired,
            },
            "geometry": geometry_manifest,
        },
        {"official": official_manifest, "resampled": support_manifests[1:]},
        prediction_arrays,
    )


def render_summary(result: dict) -> str:
    lines = [
        "# SimpleCIL-aligned CIL and CLS-FSCIL audit",
        "",
        "> Evaluator-only audit on frozen CLS features.",
        "",
        "## Arm A: published full-data 10-task protocol",
        "",
        "| metric | mean (%) | sample SD (pp) | seeds |",
        "|---|---:|---:|---|",
    ]
    for metric, row in result["arm_a"]["summary"].items():
        lines.append(
            f"| {metric} | {100*row['mean']:.3f} | {100*row['sample_sd']:.3f} | 0,1,2 |"
        )
    lines.extend(
        [
            "",
            "## Arm B: matched FSCIL transfer",
            "",
            "| method | AA (%) | final (%) | final base (%) | final novel (%) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in ARM_B_METHODS:
        row = result["arm_b"]["official"]["metrics"][method]
        lines.append(
            f"| {method} | {100*row['average_accuracy']:.3f} | "
            f"{100*row['final_accuracy']:.3f} | {100*row['final_base_accuracy']:.3f} | "
            f"{100*row['final_incremental_accuracy']:.3f} |"
        )
    lines.extend(["", "Paired seed uncertainty is stored in `result.json`.", ""])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    ensure_new_output_root(args.output_root)
    model_manifest = materialize_model(args.model_root, args.model_repo, args.model_revision)
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(args.model_root, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_root, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("feature extraction requires a CUDA GPU")
    model = model.to(device=device, dtype=torch.float32)

    inventory = load_cub_inventory(args.cub_root)
    train_records = [row for row in inventory if row["split"] == "train"]
    test_records = [row for row in inventory if row["split"] == "test"]
    train = extract_split(train_records, processor, model, device, args.batch_size, args.workers)
    test = extract_split(test_records, processor, model, device, args.batch_size, args.workers)
    del model
    torch.cuda.empty_cache()

    image_rows = []
    for split, records, payload in (("train", train_records, train), ("test", test_records, test)):
        for row, digest, size in zip(records, payload["image_sha256"], payload["image_bytes"]):
            image_rows.append(
                {
                    "image_id": row["image_id"],
                    "relative_path": row["relative_path"],
                    "label": row["label"],
                    "split": split,
                    "bytes": size,
                    "sha256": digest,
                }
            )
    image_manifest = {
        "schema_version": 1,
        "dataset": "CUB_200_2011",
        "root": str(args.cub_root),
        "metadata": {
            name: {"bytes": (args.cub_root / name).stat().st_size, "sha256": file_sha256(args.cub_root / name)}
            for name in ("images.txt", "image_class_labels.txt", "train_test_split.txt")
        },
        "counts": {"all": len(image_rows), "train": len(train_records), "test": len(test_records)},
        "images": image_rows,
    }
    image_manifest["inventory_sha256"] = canonical_json_sha256(image_rows)

    prediction_arrays = {"test_labels": test["labels"].numpy().astype(np.int16)}
    arm_a_rows = []
    for seed in ARM_A_SEEDS:
        metrics, predictions = evaluate_simplecil_seed(
            train["pooled"].numpy(),
            train["labels"].numpy(),
            test["pooled"].numpy(),
            test["labels"].numpy(),
            seed,
        )
        arm_a_rows.append(metrics)
        for session, prediction in enumerate(predictions):
            prediction_arrays[f"arm_a_seed{seed}__s{session:02d}"] = prediction
    arm_a = {
        "protocol": "AnaCP/SimpleCIL CUB full-data 10-task class-incremental",
        "seeds": list(ARM_A_SEEDS),
        "rows": arm_a_rows,
        "summary": {
            "average_accuracy": mean_sample_sd([row["average_accuracy"] for row in arm_a_rows]),
            "final_accuracy": mean_sample_sd([row["final_accuracy"] for row in arm_a_rows]),
        },
        "published_anchor": {
            "average_accuracy_mean_percent": 91.20,
            "average_accuracy_sd_pp": 0.48,
            "final_accuracy_percent": 86.87,
            "use": "reproduction check only; no tuning to the anchor",
        },
    }
    arm_b, arm_b_support, arm_b_predictions = run_arm_b(
        train, test, args.index_root, args.bootstrap_seed
    )
    prediction_arrays.update(arm_b_predictions)

    cache_path = args.output_root / "canonical_cls_features.pt"
    image_manifest_path = args.output_root / "image_manifest.json"
    model_manifest_path = args.output_root / "model_manifest.json"
    support_manifest_path = args.output_root / "support_manifest.json"
    predictions_path = args.output_root / "session_predictions.npz"
    result_path = args.output_root / "result.json"
    summary_path = args.output_root / "summary.md"
    write_torch_new(
        cache_path,
        {
            "train": {key: train[key] for key in ("pooled", "labels", "paths")},
            "test": {key: test[key] for key in ("pooled", "labels", "paths")},
            "feature_token": "last_hidden_state[:,0]",
            "model_repo": args.model_repo,
            "model_revision": args.model_revision,
        },
    )
    write_json_new(image_manifest_path, image_manifest)
    write_json_new(model_manifest_path, model_manifest)
    write_json_new(
        support_manifest_path,
        {
            "arm_a": {str(row["seed"]): row["task_classes"] for row in arm_a_rows},
            "arm_b": arm_b_support,
        },
    )
    write_npz_new(predictions_path, prediction_arrays)
    result = {
        "schema_version": 1,
        "kind": "sdg_caev_simplecil_matched_followup",
        "claim_scope": "representation/protocol audit; not a transported published baseline",
        "commit": args.commit,
        "upstream": {
            "AnaCP": ANACP_COMMIT,
            "RevisitingCIL": SIMPLECIL_COMMIT,
        },
        "protocol": {
            "model_repo": args.model_repo,
            "model_revision": args.model_revision,
            "processor": "AutoImageProcessor official config",
            "feature_token": "last_hidden_state[:,0]",
            "feature_dtype": "float32",
            "cosine": "L2-normalized query dot L2-normalized arithmetic class mean",
        },
        "arm_a": arm_a,
        "arm_b": arm_b,
        "artifacts": {},
        "runtime": {
            "seconds": time.time() - started,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
    }
    for path in (cache_path, image_manifest_path, model_manifest_path, support_manifest_path, predictions_path):
        result["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    write_json_new(result_path, result)
    write_text_new(summary_path, render_summary(result))
    print(json.dumps({"result": str(result_path), "runtime_seconds": result["runtime"]["seconds"]}))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, default=RELEASE_ROOT / "splits" / "fscil_cub200")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT / "dinov2-base")
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--commit", default=None)
    args = parser.parse_args()
    if args.model_repo != MODEL_REPO or args.model_revision != MODEL_REVISION:
        parser.error("model repo/revision must match the frozen canonical DINOv2-base identity")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("invalid loader parameters")
    return args


if __name__ == "__main__":
    run(parse_args())
