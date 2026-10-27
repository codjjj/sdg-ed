#!/usr/bin/env python3
"""Independent artifact and metric verifier for the full-data CIL evaluator audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from sdg_simplecil_matched import simplecil_task_classes, write_json_new
from sdg_fscil_prototype_audit import array_sha256, file_sha256
from sdg_audit_core import (
    REG_GRID,
    accuracy,
    cosine_predictions,
    lda_grid_predictions,
    select_index_from_scores,
    wcos_grid_predictions,
)


METHODS = ("cosine", "wcos", "lda")
SENSITIVITY_SEEDS = tuple(range(20))
PUBLISHED_SEEDS = (0, 1, 2)
SPLIT_SALT = "sdg-caev-simplecil-fulldata-evaluator-v1"


def assert_close(observed: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if abs(float(observed) - float(expected)) > tolerance:
        raise AssertionError(f"{label}: observed={observed} expected={expected}")


def fixed_split(paths: list[str], labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inner, validation = [], []
    for label in sorted(set(int(value) for value in labels.tolist())):
        rows = np.flatnonzero(labels == label).tolist()
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{SPLIT_SALT}\0{label}\0{paths[row]}".encode("utf-8")
            ).digest(),
        )
        n_validation = min(50, max(1, int(round(0.2 * len(rows)))))
        n_validation = min(n_validation, len(rows) - 1)
        validation.extend(ranked[:n_validation])
        inner.extend(ranked[n_validation:])
    return np.asarray(sorted(inner), dtype=np.int64), np.asarray(sorted(validation), dtype=np.int64)


def independent_predictions(
    family: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    inner_rows: np.ndarray,
    validation_rows: np.ndarray,
    classes: list[int],
) -> np.ndarray:
    seen_train = np.isin(train_labels, classes)
    seen_test = np.isin(test_labels, classes)
    inner = np.intersect1d(np.flatnonzero(seen_train), inner_rows, assume_unique=False)
    validation = np.intersect1d(
        np.flatnonzero(seen_train), validation_rows, assume_unique=False
    )
    if family == "cosine":
        prediction = cosine_predictions(
            train_features[seen_train], train_labels[seen_train], test_features[seen_test], classes
        )
    else:
        grid_function = wcos_grid_predictions if family == "wcos" else lda_grid_predictions
        validation_predictions, _ = grid_function(
            train_features[inner],
            train_labels[inner],
            train_features[validation],
            classes,
            REG_GRID,
            regularization_mode="absolute",
        )
        scores = [accuracy(row, train_labels[validation]) for row in validation_predictions]
        selected = float(REG_GRID[select_index_from_scores(scores)])
        test_predictions, _ = grid_function(
            train_features[seen_train],
            train_labels[seen_train],
            test_features[seen_test],
            classes,
            [selected],
            regularization_mode="absolute",
        )
        prediction = test_predictions[0]
    full = np.full(len(test_labels), -1, dtype=np.int16)
    full[seen_test] = np.asarray(prediction, dtype=np.int16)
    return full


def verify(args: argparse.Namespace) -> dict:
    root = args.root
    result_path = root / "result.json"
    predictions_path = root / "session_predictions.npz"
    for path in (result_path, predictions_path, args.protocol, args.source_cache):
        if not path.is_file():
            raise FileNotFoundError(path)
    if file_sha256(args.protocol) != args.protocol_sha256:
        raise ValueError("protocol hash mismatch")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("kind") != "sdg_caev_simplecil_fulldata_cil_evaluators":
        raise ValueError("unexpected result kind")
    if result.get("protocol_sha256") != args.protocol_sha256:
        raise ValueError("result does not bind protocol")
    if file_sha256(args.source_cache) != protocol["source"]["cache_sha256"]:
        raise ValueError("independent source-cache hash mismatch")
    if file_sha256(predictions_path) != result["artifacts"][predictions_path.name]["sha256"]:
        raise ValueError("prediction archive hash mismatch")
    cache = torch.load(args.source_cache, map_location="cpu", weights_only=False)
    train_features = cache["train"]["pooled"].double().numpy()
    train_labels = cache["train"]["labels"].long().numpy()
    test_features = cache["test"]["pooled"].double().numpy()
    test_labels = cache["test"]["labels"].long().numpy()
    train_paths = [str(value) for value in cache["train"]["paths"]]
    inner_rows, validation_rows = fixed_split(train_paths, train_labels)
    if array_sha256(inner_rows) != result["split"]["inner_rows_sha256"]:
        raise ValueError("independent inner split hash mismatch")
    if array_sha256(validation_rows) != result["split"]["validation_rows_sha256"]:
        raise ValueError("independent validation split hash mismatch")
    with np.load(predictions_path, allow_pickle=False) as archive:
        expected_keys = set(result["prediction_manifest"])
        if set(archive.files) != expected_keys:
            raise ValueError("prediction key set mismatch")
        arrays = {key: archive[key] for key in archive.files}
    for key, manifest in result["prediction_manifest"].items():
        array = arrays[key]
        if list(array.shape) != manifest["shape"] or str(array.dtype) != manifest["dtype"]:
            raise ValueError(f"prediction shape/dtype mismatch: {key}")
        if array_sha256(array) != manifest["sha256"]:
            raise ValueError(f"prediction array hash mismatch: {key}")
    if not np.array_equal(arrays["test_labels"], test_labels.astype(np.int16)):
        raise ValueError("test-label prediction artifact mismatch")
    rows = result["sensitivity_20_seed"]["rows"]
    if [row["seed"] for row in rows] != list(SENSITIVITY_SEEDS):
        raise ValueError("result seed order mismatch")
    for row in rows:
        seed = row["seed"]
        if row["task_classes"] != simplecil_task_classes(seed):
            raise ValueError(f"task partition mismatch for seed {seed}")
        for session_row in row["sessions"]:
            session = session_row["session"]
            seen = session_row["seen_classes"]
            eval_mask = np.isin(test_labels, seen)
            for method in METHODS:
                key = f"seed{seed:02d}__{method}__s{session:02d}"
                prediction = arrays[key]
                if np.any(prediction[~eval_mask] != -1):
                    raise ValueError(f"unseen query received prediction: {key}")
                observed = float((prediction[eval_mask] == test_labels[eval_mask]).mean())
                assert_close(observed, session_row["methods"][method]["accuracy"], key)
        for method in METHODS:
            observed_sessions = [
                session["methods"][method]["accuracy"] for session in row["sessions"]
            ]
            assert_close(
                float(np.mean(observed_sessions)),
                row["metrics"][method]["average_accuracy"],
                f"seed{seed}/{method}/AA",
            )
            assert_close(
                observed_sessions[-1],
                row["metrics"][method]["final_accuracy"],
                f"seed{seed}/{method}/Final",
            )
    for method in METHODS:
        for metric in ("average_accuracy", "final_accuracy"):
            values = np.asarray(
                [row["metrics"][method][metric] for row in rows], dtype=np.float64
            )
            summary = result["sensitivity_20_seed"]["summary"][method][metric]
            assert_close(float(values.mean()), summary["mean"], f"{method}/{metric}/mean")
            assert_close(
                float(values.std(ddof=1)), summary["sample_sd"], f"{method}/{metric}/sd"
            )
    for challenger in ("wcos", "lda"):
        for metric in ("average_accuracy", "final_accuracy"):
            values = np.asarray(
                [
                    100.0
                    * (
                        row["metrics"][challenger][metric]
                        - row["metrics"]["cosine"][metric]
                    )
                    for row in rows
                ],
                dtype=np.float64,
            )
            paired = result["sensitivity_20_seed"]["paired"][
                f"{challenger}_minus_cosine"
            ][metric]
            mean = float(values.mean())
            sample_sd = float(values.std(ddof=1))
            half = float(stats.t.ppf(0.975, len(values) - 1) * sample_sd / math.sqrt(len(values)))
            assert_close(mean, paired["mean_pp"], f"{challenger}/{metric}/paired-mean")
            assert_close(sample_sd, paired["sample_sd_pp"], f"{challenger}/{metric}/paired-sd")
            assert_close(mean - half, paired["ci95_pp"][0], f"{challenger}/{metric}/ci-low")
            assert_close(mean + half, paired["ci95_pp"][1], f"{challenger}/{metric}/ci-high")
            if paired["positive"] != int(np.sum(values > 1e-12)):
                raise AssertionError("paired positive-count mismatch")
            if paired["negative"] != int(np.sum(values < -1e-12)):
                raise AssertionError("paired negative-count mismatch")
            if paired["boundary"] != int(np.sum(np.abs(values) <= 1e-12)):
                raise AssertionError("paired boundary-count mismatch")
    semantic_arrays = 0
    for seed in PUBLISHED_SEEDS:
        tasks = simplecil_task_classes(seed)
        for session in range(10):
            classes = [label for task in tasks[: session + 1] for label in task]
            for family in METHODS:
                expected = independent_predictions(
                    family,
                    train_features,
                    train_labels,
                    test_features,
                    test_labels,
                    inner_rows,
                    validation_rows,
                    classes,
                )
                key = f"seed{seed:02d}__{family}__s{session:02d}"
                if not np.array_equal(expected, arrays[key]):
                    mismatch = int(np.sum(expected != arrays[key]))
                    raise AssertionError(f"independent prediction mismatch {key}: {mismatch}")
                semantic_arrays += 1
    validation = {
        "schema_version": 1,
        "kind": "sdg_caev_simplecil_fulldata_evaluator_independent_validation",
        "accepted": True,
        "protocol_sha256": args.protocol_sha256,
        "result_sha256": file_sha256(result_path),
        "prediction_archive_sha256": file_sha256(predictions_path),
        "prediction_arrays_checked": len(arrays),
        "semantic_prediction_arrays_recomputed": semantic_arrays,
        "semantic_recompute_seeds": list(PUBLISHED_SEEDS),
        "metric_rows_recomputed": len(SENSITIVITY_SEEDS) * 10 * len(METHODS),
        "test_labels_used_for_selection": False,
    }
    write_json_new(args.output, validation)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(verify(parse_args()), sort_keys=True))
