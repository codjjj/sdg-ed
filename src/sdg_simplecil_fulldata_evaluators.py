#!/usr/bin/env python3
"""Matched Cos/WCos/LDA audit in the standard full-data SimpleCIL schedule.

Reads the canonical DINOv2 CLS cache written by sdg_simplecil_matched.py
(`--source-root`) and a preregistered protocol file whose SHA-256 must match
`--protocol-sha256`; see configs/simplecil_fulldata_evaluators_protocol.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from sdg_simplecil_matched import (
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    MODEL_REPO,
    MODEL_REVISION,
    mean_sample_sd,
    simplecil_task_classes,
    write_json_new,
    write_npz_new,
    write_text_new,
)
from sdg_fscil_prototype_audit import array_sha256, file_sha256
from sdg_audit_core import REG_GRID


METHODS = ("cosine", "wcos", "lda")
SENSITIVITY_SEEDS = tuple(range(20))
PUBLISHED_SEEDS = (0, 1, 2)
SPLIT_SALT = "sdg-caev-simplecil-fulldata-evaluator-v1"
BOUNDARY_TOLERANCE = 1e-12


def canonical_json_sha256(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def validate_protocol(path: Path, expected_sha256: str) -> dict:
    if file_sha256(path) != expected_sha256:
        raise ValueError("protocol preregistration hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not payload.get("frozen_before_result_reveal"):
        raise ValueError("invalid preregistration schema or reveal state")
    schedule = payload.get("schedule", {})
    if tuple(schedule.get("published_reproduction_seeds", [])) != PUBLISHED_SEEDS:
        raise ValueError("published seed contract mismatch")
    if tuple(schedule.get("sensitivity_seeds", [])) != SENSITIVITY_SEEDS:
        raise ValueError("sensitivity seed contract mismatch")
    if payload.get("memory_assumption", {}).get("retains_historical_train_features") is not True:
        raise ValueError("history-memory assumption must be explicit")
    if payload.get("memory_assumption", {}).get("retains_only_sufficient_statistics") is not False:
        raise ValueError("sufficient-statistics assumption must remain false")
    candidates = payload.get("train_only_selection", {}).get("candidate_grid")
    if candidates != [float(value) for value in REG_GRID]:
        raise ValueError("regularization candidate grid mismatch")
    return payload


def load_canonical_cache(source_root: Path, protocol: dict) -> tuple[dict, dict]:
    source = protocol["source"]
    cache_path = source_root / source["cache_file"]
    result_path = source_root / "result.json"
    image_manifest_path = source_root / "image_manifest.json"
    for path in (cache_path, result_path, image_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if cache_path.stat().st_size != int(source["cache_bytes"]):
        raise ValueError("canonical cache byte count mismatch")
    if file_sha256(cache_path) != source["cache_sha256"]:
        raise ValueError("canonical cache SHA-256 mismatch")
    if file_sha256(result_path) != source["result_sha256"]:
        raise ValueError("source result SHA-256 mismatch")
    if file_sha256(image_manifest_path) != source["image_manifest_sha256"]:
        raise ValueError("source image-manifest SHA-256 mismatch")
    source_result = json.loads(result_path.read_text(encoding="utf-8"))
    if source_result["protocol"]["model_repo"] != MODEL_REPO:
        raise ValueError("source model repo mismatch")
    if source_result["protocol"]["model_revision"] != MODEL_REVISION:
        raise ValueError("source model revision mismatch")
    if source_result["protocol"]["feature_token"] != "last_hidden_state[:,0]":
        raise ValueError("source feature token mismatch")
    if source_result["artifacts"][source["cache_file"]]["sha256"] != source["cache_sha256"]:
        raise ValueError("source result does not bind canonical cache")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    train = payload["train"]
    test = payload["test"]
    if payload.get("model_repo") != MODEL_REPO or payload.get("model_revision") != MODEL_REVISION:
        raise ValueError("cache model identity mismatch")
    if payload.get("feature_token") != "last_hidden_state[:,0]":
        raise ValueError("cache feature-token mismatch")
    for split, expected in ((train, EXPECTED_TRAIN), (test, EXPECTED_TEST)):
        if split["pooled"].shape != (expected, 768):
            raise ValueError("canonical feature shape mismatch")
        if split["labels"].shape != (expected,) or len(split["paths"]) != expected:
            raise ValueError("canonical label/path shape mismatch")
        if not torch.isfinite(split["pooled"]).all():
            raise FloatingPointError("canonical cache contains non-finite features")
        if set(split["labels"].tolist()) != set(range(200)):
            raise ValueError("canonical cache does not cover 200 classes")
    if len(set(train["paths"])) != EXPECTED_TRAIN or len(set(test["paths"])) != EXPECTED_TEST:
        raise ValueError("canonical cache paths are not unique")
    return payload, {
        "cache_path": str(cache_path),
        "cache_bytes": cache_path.stat().st_size,
        "cache_sha256": file_sha256(cache_path),
        "source_result_sha256": file_sha256(result_path),
        "source_image_manifest_sha256": file_sha256(image_manifest_path),
    }


def fixed_train_validation_split(paths: list[str], labels: np.ndarray) -> dict:
    if len(paths) != len(labels) or len(paths) != len(set(paths)):
        raise ValueError("train split requires unique aligned paths")
    inner_rows: list[int] = []
    validation_rows: list[int] = []
    per_class = {}
    for label in sorted(set(int(value) for value in labels.tolist())):
        rows = np.flatnonzero(labels == label).tolist()
        if len(rows) < 2:
            raise ValueError(f"class {label} has fewer than two train examples")
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{SPLIT_SALT}\0{label}\0{paths[row]}".encode("utf-8")
            ).digest(),
        )
        n_validation = min(50, max(1, int(round(0.2 * len(rows)))))
        n_validation = min(n_validation, len(rows) - 1)
        selected_validation = sorted(ranked[:n_validation])
        selected_inner = sorted(ranked[n_validation:])
        validation_rows.extend(selected_validation)
        inner_rows.extend(selected_inner)
        per_class[str(label)] = {
            "all": len(rows),
            "inner": len(selected_inner),
            "validation": len(selected_validation),
            "inner_rows_sha256": array_sha256(np.asarray(selected_inner, dtype=np.int64)),
            "validation_rows_sha256": array_sha256(
                np.asarray(selected_validation, dtype=np.int64)
            ),
        }
    inner = np.asarray(sorted(inner_rows), dtype=np.int64)
    validation = np.asarray(sorted(validation_rows), dtype=np.int64)
    if len(set(inner.tolist()) & set(validation.tolist())) or len(inner) + len(validation) != len(paths):
        raise AssertionError("train-only split is not a disjoint partition")
    return {
        "inner_rows": inner,
        "validation_rows": validation,
        "inner_rows_sha256": array_sha256(inner),
        "validation_rows_sha256": array_sha256(validation),
        "per_class": per_class,
    }


def _class_means(features: torch.Tensor, labels: torch.Tensor, classes: list[int]) -> torch.Tensor:
    rows = []
    for label in classes:
        subset = features[labels == label]
        if subset.shape[0] == 0:
            raise ValueError(f"class {label} has no fitting rows")
        rows.append(subset.mean(dim=0))
    return torch.stack(rows)


def _row_normalize(features: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
    if torch.any(norms <= 0):
        raise ValueError("zero row norm")
    return features / norms


def _total_stats(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.T @ centered / max(1, features.shape[0] - 1)
    return mean, covariance


def _within_stats(
    features: torch.Tensor, labels: torch.Tensor, classes: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    means = _class_means(features, labels, classes)
    scatter = torch.zeros(
        (features.shape[1], features.shape[1]), dtype=features.dtype, device=features.device
    )
    for index, label in enumerate(classes):
        centered = features[labels == label] - means[index]
        scatter = scatter + centered.T @ centered
    covariance = scatter / max(1, features.shape[0] - len(classes))
    return means, covariance


def _safe_eigh(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.isfinite(covariance).all():
        raise FloatingPointError("non-finite covariance")
    values, vectors = torch.linalg.eigh(covariance)
    return values.clamp_min(0.0), vectors


def _cosine_predictions(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    query_features: torch.Tensor,
    classes: list[int],
) -> torch.Tensor:
    means = _class_means(train_features, train_labels, classes)
    scores = _row_normalize(query_features) @ _row_normalize(means).T
    if not torch.isfinite(scores).all():
        raise FloatingPointError("non-finite cosine scores")
    class_tensor = torch.as_tensor(classes, dtype=torch.long, device=query_features.device)
    return class_tensor[torch.argmax(scores, dim=1)]


def _wcos_predictions(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    query_features: torch.Tensor,
    classes: list[int],
    regularizer: float,
) -> torch.Tensor:
    mean, covariance = _total_stats(train_features)
    values, vectors = _safe_eigh(covariance)
    scale = torch.rsqrt(values + float(regularizer))
    train_white = ((train_features - mean) @ vectors) * scale
    query_white = ((query_features - mean) @ vectors) * scale
    return _cosine_predictions(train_white, train_labels, query_white, classes)


def _lda_predictions(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    query_features: torch.Tensor,
    classes: list[int],
    regularizer: float,
) -> torch.Tensor:
    means, covariance = _within_stats(train_features, train_labels, classes)
    values, vectors = _safe_eigh(covariance)
    inverse = (vectors * torch.reciprocal(values + float(regularizer))) @ vectors.T
    weights = means @ inverse
    bias = -0.5 * torch.sum(means * weights, dim=1)
    scores = query_features @ weights.T + bias[None, :]
    if not torch.isfinite(scores).all():
        raise FloatingPointError("non-finite LDA scores")
    class_tensor = torch.as_tensor(classes, dtype=torch.long, device=query_features.device)
    return class_tensor[torch.argmax(scores, dim=1)]


def select_regularizer(
    family: str,
    inner_features: torch.Tensor,
    inner_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    classes: list[int],
    candidates: list[float],
) -> tuple[float, list[float]]:
    if family not in {"wcos", "lda"}:
        raise ValueError(f"unsupported family: {family}")
    scores = []
    class_tensor = torch.as_tensor(classes, dtype=torch.long, device=inner_features.device)
    if family == "wcos":
        mean, covariance = _total_stats(inner_features)
        values, vectors = _safe_eigh(covariance)
        inner_pc = (inner_features - mean) @ vectors
        validation_pc = (validation_features - mean) @ vectors
        for candidate in candidates:
            scale = torch.rsqrt(values + float(candidate))
            means = _class_means(inner_pc * scale, inner_labels, classes)
            scores_matrix = _row_normalize(validation_pc * scale) @ _row_normalize(means).T
            prediction = class_tensor[torch.argmax(scores_matrix, dim=1)]
            scores.append(float((prediction == validation_labels).double().mean().item()))
    else:
        means, covariance = _within_stats(inner_features, inner_labels, classes)
        values, vectors = _safe_eigh(covariance)
        projected_means = means @ vectors
        projected_validation = validation_features @ vectors
        for candidate in candidates:
            inverse_values = torch.reciprocal(values + float(candidate))
            weights = projected_means * inverse_values
            bias = -0.5 * torch.sum(projected_means * weights, dim=1)
            scores_matrix = projected_validation @ weights.T + bias[None, :]
            prediction = class_tensor[torch.argmax(scores_matrix, dim=1)]
            scores.append(float((prediction == validation_labels).double().mean().item()))
    if not np.isfinite(np.asarray(scores, dtype=np.float64)).all():
        raise FloatingPointError("non-finite train-only validation scores")
    selected_index = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    return float(candidates[selected_index]), scores


def paired_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(SENSITIVITY_SEEDS),) or not np.isfinite(array).all():
        raise ValueError("paired summary requires twenty finite values")
    mean = float(array.mean())
    sample_sd = float(array.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(array) - 1) * sample_sd / math.sqrt(len(array)))
    return {
        "n": len(array),
        "mean_pp": mean,
        "sample_sd_pp": sample_sd,
        "ci95_pp": [mean - half, mean + half],
        "positive": int(np.sum(array > BOUNDARY_TOLERANCE)),
        "negative": int(np.sum(array < -BOUNDARY_TOLERANCE)),
        "boundary": int(np.sum(np.abs(array) <= BOUNDARY_TOLERANCE)),
        "values_pp": array.tolist(),
    }


def evaluate_seed(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    split: dict,
    seed: int,
    candidates: list[float],
) -> tuple[dict, dict[str, np.ndarray]]:
    tasks = simplecil_task_classes(seed)
    inner_mask_all = torch.zeros(len(train_labels), dtype=torch.bool, device=train_labels.device)
    validation_mask_all = torch.zeros_like(inner_mask_all)
    inner_mask_all[torch.as_tensor(split["inner_rows"], device=train_labels.device)] = True
    validation_mask_all[torch.as_tensor(split["validation_rows"], device=train_labels.device)] = True
    predictions: dict[str, np.ndarray] = {}
    sessions = []
    for session in range(10):
        classes = [label for task in tasks[: session + 1] for label in task]
        class_tensor = torch.as_tensor(classes, dtype=torch.long, device=train_labels.device)
        seen_train = torch.isin(train_labels, class_tensor)
        seen_test = torch.isin(test_labels, class_tensor)
        inner = seen_train & inner_mask_all
        validation = seen_train & validation_mask_all
        if int(inner.sum()) + int(validation.sum()) != int(seen_train.sum()):
            raise AssertionError("session train rows do not match fixed split")
        selected = {}
        validation_scores = {}
        for family in ("wcos", "lda"):
            selected[family], validation_scores[family] = select_regularizer(
                family,
                train_features[inner],
                train_labels[inner],
                train_features[validation],
                train_labels[validation],
                classes,
                candidates,
            )
        query = test_features[seen_test]
        method_predictions = {
            "cosine": _cosine_predictions(
                train_features[seen_train], train_labels[seen_train], query, classes
            ),
            "wcos": _wcos_predictions(
                train_features[seen_train],
                train_labels[seen_train],
                query,
                classes,
                selected["wcos"],
            ),
            "lda": _lda_predictions(
                train_features[seen_train],
                train_labels[seen_train],
                query,
                classes,
                selected["lda"],
            ),
        }
        method_rows = {}
        for method, prediction in method_predictions.items():
            full = torch.full(
                (len(test_labels),), -1, dtype=torch.int16, device=test_labels.device
            )
            full[seen_test] = prediction.to(torch.int16)
            array = full.cpu().numpy()
            key = f"seed{seed:02d}__{method}__s{session:02d}"
            predictions[key] = array
            method_rows[method] = {
                "accuracy": float((prediction == test_labels[seen_test]).double().mean().item()),
                "prediction_sha256": array_sha256(array),
            }
            if method in selected:
                method_rows[method]["selected_regularizer"] = selected[method]
                method_rows[method]["validation_scores"] = validation_scores[method]
        sessions.append(
            {
                "session": session,
                "learned_task_classes": tasks[session],
                "seen_classes": classes,
                "num_train": int(seen_train.sum()),
                "num_inner_train": int(inner.sum()),
                "num_validation": int(validation.sum()),
                "num_eval": int(seen_test.sum()),
                "methods": method_rows,
            }
        )
    metrics = {}
    for method in METHODS:
        values = [row["methods"][method]["accuracy"] for row in sessions]
        metrics[method] = {"average_accuracy": float(np.mean(values)), "final_accuracy": values[-1]}
    return {
        "seed": seed,
        "task_classes": tasks,
        "sessions": sessions,
        "metrics": metrics,
    }, predictions


def render_summary(result: dict) -> str:
    lines = [
        "# Standard full-data CIL matched evaluator audit",
        "",
        "> Evaluator-only diagnostic; full historical train features are retained.",
        "",
        "| evaluator | AA mean (%) | AA SD (pp) | Final mean (%) | Final SD (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = result["sensitivity_20_seed"]["summary"][method]
        lines.append(
            f"| {method} | {100*row['average_accuracy']['mean']:.3f} | "
            f"{100*row['average_accuracy']['sample_sd']:.3f} | "
            f"{100*row['final_accuracy']['mean']:.3f} | "
            f"{100*row['final_accuracy']['sample_sd']:.3f} |"
        )
    lines.extend(["", "Paired t intervals and every seed/session are in `result.json`.", ""])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = validate_protocol(args.protocol, args.protocol_sha256)
    cache, source_identity = load_canonical_cache(args.source_root, protocol)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("full evaluator audit requires a CUDA GPU")
    train_features = cache["train"]["pooled"].to(device=device, dtype=torch.float64)
    train_labels = cache["train"]["labels"].to(device=device, dtype=torch.long)
    test_features = cache["test"]["pooled"].to(device=device, dtype=torch.float64)
    test_labels = cache["test"]["labels"].to(device=device, dtype=torch.long)
    split = fixed_train_validation_split(
        [str(value) for value in cache["train"]["paths"]], train_labels.cpu().numpy()
    )
    candidates = [float(value) for value in protocol["train_only_selection"]["candidate_grid"]]
    rows = []
    prediction_arrays: dict[str, np.ndarray] = {
        "test_labels": test_labels.cpu().numpy().astype(np.int16)
    }
    for seed in SENSITIVITY_SEEDS:
        row, predictions = evaluate_seed(
            train_features,
            train_labels,
            test_features,
            test_labels,
            split,
            seed,
            candidates,
        )
        rows.append(row)
        prediction_arrays.update(predictions)
        print(f"[seed-complete] {seed}", flush=True)
    summary = {}
    for method in METHODS:
        summary[method] = {
            metric: mean_sample_sd([row["metrics"][method][metric] for row in rows])
            for metric in ("average_accuracy", "final_accuracy")
        }
    paired = {}
    for challenger in ("wcos", "lda"):
        paired[f"{challenger}_minus_cosine"] = {}
        for metric in ("average_accuracy", "final_accuracy"):
            values = [
                100.0 * (row["metrics"][challenger][metric] - row["metrics"]["cosine"][metric])
                for row in rows
            ]
            paired[f"{challenger}_minus_cosine"][metric] = paired_summary(values)
    prediction_manifest = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_sha256(value)}
        for key, value in sorted(prediction_arrays.items())
    }
    predictions_path = args.output_root / "session_predictions.npz"
    write_npz_new(predictions_path, prediction_arrays)
    result = {
        "schema_version": 1,
        "kind": "sdg_caev_simplecil_fulldata_cil_evaluators",
        "claim_scope": protocol["scientific_scope"],
        "commit": args.commit,
        "protocol_sha256": args.protocol_sha256,
        "source_identity": source_identity,
        "memory_assumption": protocol["memory_assumption"],
        "split": {
            "salt": SPLIT_SALT,
            "inner_rows": len(split["inner_rows"]),
            "validation_rows": len(split["validation_rows"]),
            "inner_rows_sha256": split["inner_rows_sha256"],
            "validation_rows_sha256": split["validation_rows_sha256"],
            "per_class": split["per_class"],
        },
        "published_seed_subset": {
            "seeds": list(PUBLISHED_SEEDS),
            "rows": [row for row in rows if row["seed"] in PUBLISHED_SEEDS],
        },
        "sensitivity_20_seed": {
            "seeds": list(SENSITIVITY_SEEDS),
            "rows": rows,
            "summary": summary,
            "paired": paired,
        },
        "prediction_manifest": prediction_manifest,
        "artifacts": {
            predictions_path.name: {
                "bytes": predictions_path.stat().st_size,
                "sha256": file_sha256(predictions_path),
            }
        },
        "runtime": {
            "seconds": time.time() - started,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
    }
    result_path = args.output_root / "result.json"
    write_json_new(result_path, result)
    write_text_new(args.output_root / "summary.md", render_summary(result))
    print(json.dumps({"result": str(result_path), "sha256": file_sha256(result_path)}))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--commit", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
