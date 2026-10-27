#!/usr/bin/env python3
"""CUB FSCIL prototype-head geometry audit (patch-mean FSCIL, Appendix L).

The official arm follows the frozen 11-session CUB protocol: 100 base classes
followed by ten 10-way, 5-shot sessions.  Frozen DINOv2-B features and the
official session-index files are the only inputs.  Regularizers and covariance
geometry are selected/refit using base-session support only and then frozen.

This is an evaluator isolation audit, not a reproduction or comparison of
end-to-end FSCIL algorithms.

Inputs: the CUB-200 core-suite cache (patch-token mean; `--frozen-slim-root`)
and a source cache with the same features plus image paths for the complete
official training split (`--source-root`: train.pt and val.pt with keys
pooled/labels/paths). The session files are the standard CUB-200 FSCIL splits
shipped in `splits/fscil_cub200/` (from the CEC repository, MIT license).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from sdg_paths import OUTPUT_ROOT, RELEASE_ROOT, core_feature_dir
from sdg_audit_core import (
    REG_GRID,
    accuracy,
    environment_manifest,
    lda_grid_predictions,
    paired_bootstrap_difference,
    pooled_within_stats,
    row_normalize,
    select_index_from_scores,
    sorted_eigh,
    total_covariance_stats,
    wcos_grid_predictions,
    write_json_new,
    write_text_new,
)
from sdg_stepb_cached_eval import train_val_split


BASE_CLASSES = 100
WAY = 10
SESSIONS = 11
SHOT = 5
NUM_CLASSES = 200
METHODS = ("cosine", "euclidean", "wcos_base_frozen", "lda_base_frozen")

DEFAULT_FROZEN_SLIM_ROOT = core_feature_dir("cub200")
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT / "fscil_prototype"
DEFAULT_INDEX_ROOT = RELEASE_ROOT / "splits" / "fscil_cub200"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def path_key(value: str | Path) -> str:
    path = Path(str(value))
    if len(path.parts) < 2:
        raise ValueError(f"CUB image path lacks class directory: {value}")
    parent = path.parent.name
    official_match = re.fullmatch(r"(?P<class_id>\d{3})\.(?P<class_name>.+)", parent)
    cache_match = re.fullmatch(
        r"(?P<zero_based>\d{3})_(?P<class_id>\d{3})_(?P<class_name>.+)",
        parent,
    )
    if official_match is not None:
        class_id = int(official_match.group("class_id"))
        class_name = official_match.group("class_name")
    elif cache_match is not None:
        zero_based = int(cache_match.group("zero_based"))
        class_id = int(cache_match.group("class_id"))
        class_name = cache_match.group("class_name")
        if class_id != zero_based + 1:
            raise ValueError(f"inconsistent CUB cache class directory: {value}")
    else:
        raise ValueError(f"unrecognized CUB class directory: {value}")
    if class_id < 1 or class_id > NUM_CLASSES:
        raise ValueError(f"invalid CUB class directory: {value}")
    return f"{class_id:03d}.{class_name}/{path.name}"


def label_from_index_line(line: str) -> int:
    parent = Path(line.strip()).parent.name
    token = parent.split(".", 1)[0]
    if not token.isdigit():
        raise ValueError(f"cannot infer CUB class from index line: {line}")
    label = int(token) - 1
    if label < 0 or label >= NUM_CLASSES:
        raise ValueError(f"invalid CUB class in index line: {line}")
    return label


def audit_index_files(index_root: Path) -> dict:
    index_root = Path(index_root)
    files = []
    seen_paths: set[str] = set()
    class_order = []
    for session in range(SESSIONS):
        path = index_root / f"session_{session + 1}.txt"
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        labels = [label_from_index_line(line) for line in lines]
        expected_labels = (
            list(range(BASE_CLASSES))
            if session == 0
            else list(range(BASE_CLASSES + (session - 1) * WAY, BASE_CLASSES + session * WAY))
        )
        counts = Counter(labels)
        expected_per_class = 30 if session == 0 else SHOT
        if sorted(counts) != expected_labels:
            raise ValueError(
                f"session {session + 1} classes {sorted(counts)} do not match {expected_labels}"
            )
        if any(counts[label] != expected_per_class for label in expected_labels):
            raise ValueError(f"session {session + 1} has invalid per-class counts: {dict(counts)}")
        keys = [path_key(line) for line in lines]
        duplicates = seen_paths.intersection(keys)
        if duplicates:
            raise ValueError(f"official support path reused across sessions: {sorted(duplicates)[:3]}")
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate path inside session {session + 1}")
        seen_paths.update(keys)
        class_order.extend(expected_labels)
        files.append(
            {
                "session": session,
                "file": str(path),
                "sha256": file_sha256(path),
                "rows": len(lines),
                "classes": expected_labels,
                "per_class": expected_per_class,
            }
        )
    if class_order != list(range(NUM_CLASSES)):
        raise AssertionError("official CUB class order is not 0..199")
    return {
        "schema_version": 1,
        "kind": "official_cub_fscil_index_audit",
        "base_classes": BASE_CLASSES,
        "way": WAY,
        "sessions": SESSIONS,
        "shot": SHOT,
        "class_order": class_order,
        "files": files,
        "unique_support_paths": len(seen_paths),
    }


def _load_source_split(path: Path) -> dict:
    # The legacy source files also contain multi-GiB spatial latents.  mmap
    # keeps those unrequested storages off RAM while pooled/path metadata is
    # copied into the small working cache.
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    required = {"pooled", "labels", "paths"}
    if not required.issubset(payload):
        raise ValueError(f"{path} lacks keys {sorted(required - set(payload))}")
    pooled = payload["pooled"].float().contiguous()
    labels = payload["labels"].long().contiguous()
    paths = [str(value) for value in payload["paths"]]
    if pooled.ndim != 2 or labels.ndim != 1 or pooled.shape[0] != labels.numel():
        raise ValueError(f"invalid source shapes at {path}")
    if len(paths) != labels.numel():
        raise ValueError(f"path count mismatch at {path}")
    if not torch.isfinite(pooled).all():
        raise FloatingPointError(f"non-finite source features at {path}")
    return {"pooled": pooled, "labels": labels, "paths": paths}


def _load_frozen_slim(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    pooled = payload["pooled"].float().contiguous()
    labels = payload["labels"].long().contiguous()
    if pooled.ndim != 2 or labels.ndim != 1 or pooled.shape[0] != labels.numel():
        raise ValueError(f"invalid frozen slim shapes at {path}")
    if not torch.isfinite(pooled).all():
        raise FloatingPointError(f"non-finite frozen slim features at {path}")
    return {"pooled": pooled, "labels": labels}


def _check_frozen_slim_lineage(source: dict, frozen_path: Path) -> dict:
    frozen = torch.load(frozen_path, map_location="cpu", weights_only=False)
    frozen_features = frozen["pooled"].float().contiguous()
    frozen_labels = frozen["labels"].long().contiguous()
    feature_exact = torch.equal(source["pooled"], frozen_features)
    label_exact = torch.equal(source["labels"], frozen_labels)
    if not feature_exact or not label_exact:
        raise AssertionError(f"source/frozen slim lineage mismatch: {frozen_path}")
    return {
        "frozen_path": str(frozen_path),
        "frozen_sha256": file_sha256(frozen_path),
        "rows": int(frozen_labels.numel()),
        "feature_exact": feature_exact,
        "label_exact": label_exact,
    }


def _write_torch_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        shutil.copy2(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_full_official_cache(
    source_root: Path,
    frozen_slim_root: Path,
    cache_root: Path,
    index_root: Path,
) -> dict:
    """Materialize train+validation as the full official train split."""

    train_output = cache_root / "train_full.pt"
    test_output = cache_root / "test.pt"
    manifest_output = cache_root / "manifest.json"
    if train_output.exists() or test_output.exists() or manifest_output.exists():
        raise FileExistsError(f"refusing partial/existing cache at {cache_root}")

    source_train = _load_source_split(source_root / "train.pt")
    train_lineage = _check_frozen_slim_lineage(source_train, frozen_slim_root / "train.pt")
    source_validation = _load_source_split(source_root / "val.pt")
    train_full = {
        "pooled": torch.cat([source_train["pooled"], source_validation["pooled"]], dim=0),
        "labels": torch.cat([source_train["labels"], source_validation["labels"]], dim=0),
        "paths": source_train["paths"] + source_validation["paths"],
    }
    del source_validation
    gc.collect()

    if train_full["labels"].numel() != 5994:
        raise ValueError(f"full CUB train split has {train_full['labels'].numel()} rows, expected 5994")
    train_keys = [path_key(value) for value in train_full["paths"]]
    if len(train_keys) != len(set(train_keys)):
        raise ValueError("full CUB train cache has duplicate image identities")
    train_by_key = {key: index for index, key in enumerate(train_keys)}
    index_audit = audit_index_files(index_root)
    missing = []
    label_mismatch = []
    for file_row in index_audit["files"]:
        lines = [
            line.strip()
            for line in Path(file_row["file"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for line in lines:
            key = path_key(line)
            if key not in train_by_key:
                missing.append(key)
            elif int(train_full["labels"][train_by_key[key]]) != label_from_index_line(line):
                label_mismatch.append(key)
    if missing or label_mismatch:
        raise ValueError(
            f"official index/cache mismatch: missing={missing[:3]} label_mismatch={label_mismatch[:3]}"
        )

    # The test cache already is the exact frozen paper artifact.  Loading the
    # 4.57-GB legacy test source solely to recover unused paths adds no
    # scientific check and caused avoidable infrastructure pressure.
    frozen_test_path = frozen_slim_root / "test.pt"
    source_test = _load_frozen_slim(frozen_test_path)
    if source_test["labels"].numel() != 5794:
        raise ValueError(f"CUB test split has {source_test['labels'].numel()} rows, expected 5794")
    test_lineage = {
        "frozen_path": str(frozen_test_path),
        "frozen_sha256": file_sha256(frozen_test_path),
        "rows": int(source_test["labels"].numel()),
        "source": "exact frozen paper slim cache",
        "legacy_multigiB_test_source_loaded": False,
    }

    _write_torch_new(train_output, train_full)
    _write_torch_new(test_output, source_test)
    manifest = {
        "schema_version": 1,
        "kind": "official_cub_fscil_full_slim_cache",
        "source_root": str(source_root),
        "frozen_slim_root": str(frozen_slim_root),
        "train_composition": "source train.pt + source val.pt",
        "train_rows": int(train_full["labels"].numel()),
        "test_rows": int(source_test["labels"].numel()),
        "feature_dim": int(train_full["pooled"].shape[1]),
        "train_class_counts": {
            str(label): int((train_full["labels"] == label).sum()) for label in range(NUM_CLASSES)
        },
        "test_class_counts": {
            str(label): int((source_test["labels"] == label).sum()) for label in range(NUM_CLASSES)
        },
        "train_lineage": train_lineage,
        "test_lineage": test_lineage,
        "index_audit": index_audit,
        "artifacts": {
            "train_full.pt": {
                "bytes": train_output.stat().st_size,
                "sha256": file_sha256(train_output),
            },
            "test.pt": {
                "bytes": test_output.stat().st_size,
                "sha256": file_sha256(test_output),
            },
        },
    }
    write_json_new(manifest_output, manifest)
    return manifest


def load_prepared_cache(cache_root: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, list[str], dict]:
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename in ("train_full.pt", "test.pt"):
        path = cache_root / filename
        expected = manifest["artifacts"][filename]
        if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"prepared cache artifact mismatch: {path}")
    train = torch.load(cache_root / "train_full.pt", map_location="cpu", weights_only=False)
    test = torch.load(cache_root / "test.pt", map_location="cpu", weights_only=False)
    return (
        train["pooled"].float().numpy(),
        train["labels"].long().numpy(),
        [str(value) for value in train["paths"]],
        test["pooled"].float().numpy(),
        test["labels"].long().numpy(),
        [str(value) for value in test.get("paths", [])],
        manifest,
    )


def official_support_indices(
    train_paths: list[str],
    train_labels: np.ndarray,
    index_root: Path,
) -> tuple[dict[int, np.ndarray], list[int], dict]:
    keys = [path_key(value) for value in train_paths]
    if len(keys) != len(set(keys)):
        raise ValueError("training identities are not unique")
    index_by_key = {key: index for index, key in enumerate(keys)}
    support: dict[int, list[int]] = {label: [] for label in range(NUM_CLASSES)}
    order = []
    sessions = []
    for session in range(SESSIONS):
        path = index_root / f"session_{session + 1}.txt"
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        session_labels = []
        for line in lines:
            label = label_from_index_line(line)
            key = path_key(line)
            if key not in index_by_key:
                raise KeyError(f"official support missing from prepared cache: {key}")
            row = index_by_key[key]
            if int(train_labels[row]) != label:
                raise ValueError(f"official support label mismatch: {key}")
            support[label].append(row)
            session_labels.append(label)
        current = sorted(set(session_labels))
        order.extend(current)
        sessions.append(
            {
                "session": session,
                "classes": current,
                "paths_by_class": {
                    str(label): [path_key(train_paths[row]) for row in support[label]]
                    for label in current
                },
            }
        )
    frozen = {}
    for label, rows in support.items():
        expected = 30 if label < BASE_CLASSES else SHOT
        if len(rows) != expected or len(rows) != len(set(rows)):
            raise ValueError(f"official class {label} support size={len(rows)}, expected {expected}")
        frozen[label] = np.asarray(rows, dtype=np.int64)
    if order != list(range(NUM_CLASSES)):
        raise ValueError("official class order mismatch")
    return frozen, order, {"kind": "official_index", "sessions": sessions}


def _hash_rank(seed: int, label: int, key: str) -> bytes:
    return hashlib.sha256(f"sdg-fscil-support-v1\0{seed}\0{label}\0{key}".encode()).digest()


def resampled_support_indices(
    train_paths: list[str],
    train_labels: np.ndarray,
    base_support: dict[int, np.ndarray],
    seed: int,
) -> tuple[dict[int, np.ndarray], list[int], dict]:
    support = {label: np.asarray(rows, dtype=np.int64).copy() for label, rows in base_support.items()}
    selected_paths = {}
    for label in range(BASE_CLASSES, NUM_CLASSES):
        pool = np.flatnonzero(train_labels == label)
        if len(pool) < SHOT:
            raise ValueError(f"class {label} has only {len(pool)} training rows")
        ranked = sorted(pool.tolist(), key=lambda row: _hash_rank(seed, label, path_key(train_paths[row])))
        chosen = np.asarray(sorted(ranked[:SHOT]), dtype=np.int64)
        support[label] = chosen
        selected_paths[str(label)] = [path_key(train_paths[row]) for row in chosen]
    novel_order = sorted(
        range(BASE_CLASSES, NUM_CLASSES),
        key=lambda label: hashlib.sha256(
            f"sdg-fscil-order-v1\0{seed}\0{label}".encode()
        ).digest(),
    )
    order = list(range(BASE_CLASSES)) + novel_order
    return support, order, {
        "kind": "support_and_novel_order_sensitivity",
        "seed": int(seed),
        "selection": "sha256 rank over full official training pool, five rows per novel class",
        "novel_class_order": novel_order,
        "paths_by_class": selected_paths,
    }


def support_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    support: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.concatenate([support[label] for label in range(NUM_CLASSES)])
    selected_features = features[rows]
    selected_labels = labels[rows]
    expected = np.concatenate(
        [
            np.full(len(support[label]), label, dtype=np.int64)
            for label in range(NUM_CLASSES)
        ]
    )
    if not np.array_equal(selected_labels, expected):
        raise ValueError("support rows do not align with class keys")
    return selected_features, selected_labels


def fit_base_geometries(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    base_rows: np.ndarray,
    validation_seed: int,
) -> dict:
    base_features = np.asarray(train_features[base_rows], dtype=np.float64)
    base_labels = np.asarray(train_labels[base_rows], dtype=np.int64)
    if set(base_labels.tolist()) != set(range(BASE_CLASSES)):
        raise ValueError("base support does not cover exactly classes 0..99")
    inner, inner_labels, validation, validation_labels, split = train_val_split(
        base_features, base_labels, seed=validation_seed
    )
    if validation is None:
        raise ValueError("base session unexpectedly lacks validation")
    classes = list(range(BASE_CLASSES))
    lda_validation, _ = lda_grid_predictions(
        inner,
        inner_labels,
        validation,
        classes,
        REG_GRID,
        regularization_mode="absolute",
    )
    wcos_validation, _ = wcos_grid_predictions(
        inner,
        inner_labels,
        validation,
        classes,
        REG_GRID,
        regularization_mode="absolute",
    )
    lda_scores = [accuracy(prediction, validation_labels) for prediction in lda_validation]
    wcos_scores = [accuracy(prediction, validation_labels) for prediction in wcos_validation]
    lda_regularizer = float(REG_GRID[select_index_from_scores(lda_scores)])
    wcos_regularizer = float(REG_GRID[select_index_from_scores(wcos_scores)])

    _, within_covariance = pooled_within_stats(base_features, base_labels, classes)
    lda_values, lda_vectors = sorted_eigh(within_covariance)
    lda_inverse = (lda_vectors * (1.0 / (lda_values + lda_regularizer))) @ lda_vectors.T

    total_mean, total_covariance = total_covariance_stats(base_features)
    wcos_values, wcos_vectors = sorted_eigh(total_covariance)
    wcos_scale = 1.0 / np.sqrt(wcos_values + wcos_regularizer)
    return {
        "validation_split": split,
        "base_rows": int(len(base_rows)),
        "base_rows_sha256": array_sha256(base_rows),
        "lda": {
            "selected_regularizer": lda_regularizer,
            "validation_scores": lda_scores,
            "inverse": lda_inverse,
            "covariance_sha256": array_sha256(within_covariance),
        },
        "wcos": {
            "selected_regularizer": wcos_regularizer,
            "validation_scores": wcos_scores,
            "mean": total_mean,
            "vectors": wcos_vectors,
            "scale": wcos_scale,
            "covariance_sha256": array_sha256(total_covariance),
        },
    }


def class_means(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    means = []
    for label in range(NUM_CLASSES):
        subset = features[labels == label]
        if len(subset) == 0:
            raise ValueError(f"support lacks class {label}")
        means.append(subset.mean(axis=0))
    return np.asarray(means, dtype=np.float64)


def all_class_scores(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    test_features: np.ndarray,
    geometry: dict,
) -> dict[str, np.ndarray]:
    raw_means = class_means(support_features, support_labels)
    test64 = np.asarray(test_features, dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = {
            "cosine": row_normalize(test64) @ row_normalize(raw_means).T,
            "euclidean": (
                2.0 * test64 @ raw_means.T
                - np.sum(raw_means * raw_means, axis=1)[None, :]
            ),
        }

    wcos = geometry["wcos"]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        support_wcos = (
            (np.asarray(support_features, dtype=np.float64) - wcos["mean"]) @ wcos["vectors"]
        ) * wcos["scale"][None, :]
        test_wcos = ((test64 - wcos["mean"]) @ wcos["vectors"]) * wcos["scale"][None, :]
    wcos_means = class_means(support_wcos, support_labels)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores["wcos_base_frozen"] = row_normalize(test_wcos) @ row_normalize(wcos_means).T

    weights = raw_means @ geometry["lda"]["inverse"]
    bias = -0.5 * np.sum(raw_means * weights, axis=1)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores["lda_base_frozen"] = test64 @ weights.T + bias[None, :]
    if set(scores) != set(METHODS):
        raise AssertionError("method score set mismatch")
    if any(value.shape != (len(test_features), NUM_CLASSES) for value in scores.values()):
        raise AssertionError("all-class score shape mismatch")
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise FloatingPointError("non-finite FSCIL score matrix")
    return scores


def evaluate_score_sessions(
    score: np.ndarray,
    test_labels: np.ndarray,
    class_order: list[int],
    *,
    base_classes: int = BASE_CLASSES,
    way: int = WAY,
) -> tuple[list[dict], list[np.ndarray]]:
    if sorted(class_order) != list(range(score.shape[1])):
        raise ValueError("class_order must be a permutation of score columns")
    session_count = 1 + (score.shape[1] - base_classes) // way
    rows = []
    stored = []
    for session in range(session_count):
        seen = class_order[: base_classes + session * way]
        current = class_order[:base_classes] if session == 0 else class_order[
            base_classes + (session - 1) * way : base_classes + session * way
        ]
        old = [] if session == 0 else class_order[: base_classes + (session - 1) * way]
        eval_mask = np.isin(test_labels, seen)
        eval_labels = test_labels[eval_mask]
        restricted = score[eval_mask][:, seen]
        prediction = np.asarray(seen, dtype=np.int64)[np.argmax(restricted, axis=1)]
        full_prediction = np.full(len(test_labels), -1, dtype=np.int16)
        full_prediction[eval_mask] = prediction.astype(np.int16)
        stored.append(full_prediction)
        base_mask = eval_labels < base_classes
        incremental_mask = eval_labels >= base_classes
        current_mask = np.isin(eval_labels, current)
        old_mask = np.isin(eval_labels, old)
        rows.append(
            {
                "session": session,
                "seen_classes": len(seen),
                "current_classes": current,
                "num_eval": int(eval_mask.sum()),
                "all_acc": accuracy(prediction, eval_labels),
                "base_acc": accuracy(prediction[base_mask], eval_labels[base_mask]),
                "incremental_acc": (
                    accuracy(prediction[incremental_mask], eval_labels[incremental_mask])
                    if incremental_mask.any()
                    else None
                ),
                "current_acc": accuracy(prediction[current_mask], eval_labels[current_mask]),
                "old_acc": (
                    accuracy(prediction[old_mask], eval_labels[old_mask]) if old_mask.any() else None
                ),
                "prediction_sha256": array_sha256(full_prediction),
            }
        )
    return rows, stored


def arm_metrics(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    geometry: dict,
    class_order: list[int],
) -> tuple[dict, dict[str, list[np.ndarray]]]:
    score_by_method = all_class_scores(
        support_features,
        support_labels,
        test_features,
        geometry,
    )
    metrics = {}
    predictions = {}
    for method in METHODS:
        session_rows, session_predictions = evaluate_score_sessions(
            score_by_method[method], test_labels, class_order
        )
        metrics[method] = {
            "sessions": session_rows,
            "average_accuracy": float(np.mean([row["all_acc"] for row in session_rows])),
            "final_accuracy": session_rows[-1]["all_acc"],
            "final_base_accuracy": session_rows[-1]["base_acc"],
            "final_incremental_accuracy": session_rows[-1]["incremental_acc"],
        }
        predictions[method] = session_predictions
    return metrics, predictions


def seed_paired_summary(values: list[float], seed: int, n_boot: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        raise ValueError("paired seed summary requires at least two values")
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    critical = float(stats.t.ppf(0.975, df=len(array) - 1))
    mean = float(array.mean())
    rng = np.random.RandomState(seed)
    bootstrap = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        bootstrap[index] = float(array[rng.randint(0, len(array), size=len(array))].mean())
    return {
        "n_seeds": int(len(array)),
        "mean_pp": mean,
        "sd_pp": float(array.std(ddof=1)),
        "t_ci95_pp": [mean - critical * standard_error, mean + critical * standard_error],
        "bootstrap_ci95_pp": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "positive_seeds": int((array > 0).sum()),
        "zero_seeds": int((array == 0).sum()),
        "negative_seeds": int((array < 0).sum()),
        "values_pp": array.tolist(),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
    }


def official_bootstrap(
    official_metrics: dict,
    official_predictions: dict[str, list[np.ndarray]],
    test_labels: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    result = {}
    for challenger in METHODS:
        if challenger == "cosine":
            continue
        rows = []
        for session in range(SESSIONS):
            mask = official_predictions["cosine"][session] >= 0
            if not np.array_equal(mask, official_predictions[challenger][session] >= 0):
                raise AssertionError("paired evaluator masks differ")
            paired = paired_bootstrap_difference(
                official_predictions[challenger][session][mask],
                official_predictions["cosine"][session][mask],
                test_labels[mask],
                n_boot=n_boot,
                seed=seed + 1009 * session + 9176 * METHODS.index(challenger),
            )
            paired["session"] = session
            paired["challenger_accuracy"] = official_metrics[challenger]["sessions"][session][
                "all_acc"
            ]
            paired["cosine_accuracy"] = official_metrics["cosine"]["sessions"][session][
                "all_acc"
            ]
            rows.append(paired)
        result[f"{challenger}_minus_cosine"] = rows
    return result


def _write_npz_new(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        shutil.copy2(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_summary(payload: dict) -> str:
    official = payload["official"]["metrics"]
    seed_summary = payload["resampled_sensitivity"]["paired_seed_summary"]
    lines = [
        "# CUB official FSCIL prototype-head geometry audit",
        "",
        "> Additional analysis; evaluator geometry only, on frozen features.",
        "",
        "## Protocol",
        "",
        "- CUB-200: 100 base classes, then 10 sessions of 10-way 5-shot.",
        "- Frozen DINOv2-B features; official session indices are the primary arm.",
        "- WCos/LDA regularizers are selected on a class-balanced base-only validation split,",
        "  refit on full base support, and frozen before any incremental class arrives.",
        "- This isolates evaluator geometry and is not an end-to-end FSCIL leaderboard comparison.",
        "",
        "## Official-index arm",
        "",
        "| method | average acc. (%) | final acc. (%) | final base (%) | final novel (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = official[method]
        lines.append(
            f"| {method} | {100*row['average_accuracy']:.2f} | "
            f"{100*row['final_accuracy']:.2f} | {100*row['final_base_accuracy']:.2f} | "
            f"{100*row['final_incremental_accuracy']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Resampled support/order sensitivity",
            "",
            "| contrast | metric | mean delta (pp) | t-CI (pp) | signs (+/0/-) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for contrast, metrics in seed_summary.items():
        for metric, row in metrics.items():
            lines.append(
                f"| {contrast} | {metric} | {row['mean_pp']:+.3f} | "
                f"[{row['t_ci95_pp'][0]:+.3f}, {row['t_ci95_pp'][1]:+.3f}] | "
                f"{row['positive_seeds']}/{row['zero_seeds']}/{row['negative_seeds']} |"
            )
    lines.extend(
        [
            "",
            "Interpretation must follow the signed intervals: evaluator geometry is load-bearing only",
            "for metrics whose paired interval excludes zero. Mixed or negative results narrow the claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(args: argparse.Namespace) -> dict:
    output_root = Path(args.output_root)
    cache_root = output_root / "cache"
    if not (cache_root / "manifest.json").is_file():
        prepare_full_official_cache(
            Path(args.source_root),
            Path(args.frozen_slim_root),
            cache_root,
            Path(args.index_root),
        )
    (
        train_features,
        train_labels,
        train_paths,
        test_features,
        test_labels,
        _test_paths,
        cache_manifest,
    ) = load_prepared_cache(cache_root)

    official_support, official_order, official_support_manifest = official_support_indices(
        train_paths, train_labels, Path(args.index_root)
    )
    base_rows = np.concatenate([official_support[label] for label in range(BASE_CLASSES)])
    geometry = fit_base_geometries(
        train_features, train_labels, base_rows, validation_seed=args.validation_seed
    )
    geometry_manifest = {
        "validation_split": geometry["validation_split"],
        "base_rows": geometry["base_rows"],
        "base_rows_sha256": geometry["base_rows_sha256"],
        "lda": {
            key: value for key, value in geometry["lda"].items() if key != "inverse"
        },
        "wcos": {
            key: value
            for key, value in geometry["wcos"].items()
            if key not in {"mean", "vectors", "scale"}
        },
        "future_session_rows_used_for_selection_or_covariance": 0,
        "test_rows_used_for_selection_or_covariance": 0,
    }

    official_features, official_labels = support_arrays(
        train_features, train_labels, official_support
    )
    official_metrics, official_predictions = arm_metrics(
        official_features,
        official_labels,
        test_features,
        test_labels,
        geometry,
        official_order,
    )
    prediction_arrays = {"test_labels": test_labels.astype(np.int16)}
    for method in METHODS:
        for session, prediction in enumerate(official_predictions[method]):
            prediction_arrays[f"official__{method}__s{session:02d}"] = prediction

    resampled_rows = []
    support_manifests = {"official": official_support_manifest, "resampled": []}
    for seed in args.sensitivity_seeds:
        support, order, support_manifest = resampled_support_indices(
            train_paths,
            train_labels,
            official_support,
            seed,
        )
        support_features, support_labels = support_arrays(train_features, train_labels, support)
        metrics, predictions = arm_metrics(
            support_features,
            support_labels,
            test_features,
            test_labels,
            geometry,
            order,
        )
        resampled_rows.append({"seed": int(seed), "class_order": order, "metrics": metrics})
        support_manifests["resampled"].append(support_manifest)
        for method in METHODS:
            for session, prediction in enumerate(predictions[method]):
                prediction_arrays[f"seed{seed}__{method}__s{session:02d}"] = prediction

    paired_seed_summary = {}
    summary_metrics = (
        "average_accuracy",
        "final_accuracy",
        "final_base_accuracy",
        "final_incremental_accuracy",
    )
    for challenger in METHODS:
        if challenger == "cosine":
            continue
        contrast = f"{challenger}_minus_cosine"
        paired_seed_summary[contrast] = {}
        for metric in summary_metrics:
            deltas = [
                100.0
                * (row["metrics"][challenger][metric] - row["metrics"]["cosine"][metric])
                for row in resampled_rows
            ]
            paired_seed_summary[contrast][metric] = seed_paired_summary(
                deltas,
                seed=args.bootstrap_seed + 2039 * METHODS.index(challenger),
                n_boot=args.seed_bootstrap,
            )

    official_paired = official_bootstrap(
        official_metrics,
        official_predictions,
        test_labels,
        n_boot=args.test_bootstrap,
        seed=args.bootstrap_seed,
    )
    result = {
        "schema_version": 1,
        "kind": "post_submission_cub_fscil_prototype_geometry_audit",
        "claim_scope": (
            "Evaluator-geometry isolation on frozen features; not an end-to-end FSCIL method comparison"
        ),
        "protocol": {
            "dataset": "CUB-200-2011",
            "base_classes": BASE_CLASSES,
            "incremental_sessions": SESSIONS - 1,
            "way": WAY,
            "shot": SHOT,
            "official_index_root": str(Path(args.index_root)),
            "index_audit": cache_manifest["index_audit"],
            "feature_backbone": "DINOv2-B",
            "methods": list(METHODS),
        },
        "cache_manifest": cache_manifest,
        "geometry": geometry_manifest,
        "official": {
            "support": "official session_1..session_11 index files",
            "class_order": official_order,
            "metrics": official_metrics,
            "paired_test_example_bootstrap": official_paired,
        },
        "resampled_sensitivity": {
            "scope": (
                "same base/novel class split; five supports per novel class sampled from the full "
                "official train pool and the 100 novel classes deterministically reordered"
            ),
            "seeds": [int(value) for value in args.sensitivity_seeds],
            "rows": resampled_rows,
            "paired_seed_summary": paired_seed_summary,
        },
        "selection_guards": {
            "regularizer_selection": "base-support inner validation only",
            "covariance_refit": "full official base support only, frozen across all sessions",
            "novel_support_used_for": "class prototypes only",
            "test_labels_used_for": "final metrics and paired uncertainty only",
        },
        "artifacts": {},
        "environment": environment_manifest(args.commit),
    }

    predictions_path = output_root / "session_predictions.npz"
    support_path = output_root / "support_manifest.json"
    result_path = output_root / "fscil_prototype_audit.json"
    summary_path = output_root / "fscil_prototype_audit.md"
    _write_npz_new(predictions_path, prediction_arrays)
    write_json_new(support_path, support_manifests)
    result["artifacts"] = {
        "session_predictions.npz": {
            "bytes": predictions_path.stat().st_size,
            "sha256": file_sha256(predictions_path),
            "arrays": len(prediction_arrays),
        },
        "support_manifest.json": {
            "bytes": support_path.stat().st_size,
            "sha256": file_sha256(support_path),
        },
    }
    write_json_new(result_path, result)
    write_text_new(summary_path, render_summary(result))
    print(json.dumps(result["official"]["metrics"], indent=2))
    print(f"saved {result_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="CUB-200 patch-mean source cache with train.pt/val.pt (pooled, labels, paths)",
    )
    parser.add_argument("--frozen-slim-root", type=Path, default=DEFAULT_FROZEN_SLIM_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument("--sensitivity-seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--test-bootstrap", type=int, default=2000)
    parser.add_argument("--seed-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--commit", default=None)
    args = parser.parse_args()
    if len(args.sensitivity_seeds) < 5 or len(set(args.sensitivity_seeds)) != len(
        args.sensitivity_seeds
    ):
        parser.error("sensitivity seeds must contain at least five unique values")
    if args.test_bootstrap < 1 or args.seed_bootstrap < 1:
        parser.error("bootstrap counts must be positive")
    return args


if __name__ == "__main__":
    run_audit(parse_args())
