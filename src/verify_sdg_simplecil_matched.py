#!/usr/bin/env python3
"""Independent metric and cosine-path verifier for the SimpleCIL-aligned runs.

Covers the full-data CIL reproduction and CLS-FSCIL (Appendix L).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


NUM_CLASSES = 200
BASE_CLASSES = 100
WAY = 10
ARM_A_SEEDS = (0, 1, 2)
ARM_B_SEEDS = tuple(range(20))
ARM_B_METHODS = ("cosine", "wcos_base_frozen", "lda_base_frozen")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def key_for_path(value: str) -> str:
    path = Path(value)
    if len(path.parts) < 2:
        raise ValueError(f"invalid CUB path: {value}")
    return f"{path.parent.name}/{path.name}"


def normalized_rows(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("zero vector in cosine verifier")
    return value / norm


def class_means(features: np.ndarray, labels: np.ndarray, rows_by_class: dict[int, list[int]]) -> np.ndarray:
    values = []
    for label in range(NUM_CLASSES):
        rows = rows_by_class[label]
        if not rows or any(int(labels[row]) != label for row in rows):
            raise ValueError(f"invalid support rows for class {label}")
        values.append(features[rows].mean(axis=0))
    return np.stack(values)


def assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=1e-12, rtol=0.0):
        raise AssertionError(f"{label}: actual={actual} expected={expected}")


def metric_from_prediction(
    prediction: np.ndarray, labels: np.ndarray, seen: list[int], base_classes: int = BASE_CLASSES
) -> dict:
    mask = np.isin(labels, seen)
    if not np.array_equal(prediction >= 0, mask):
        raise AssertionError("stored prediction mask differs from seen-class mask")
    y = labels[mask]
    pred = prediction[mask]
    base = y < base_classes
    novel = y >= base_classes
    return {
        "all_acc": float((pred == y).mean()),
        "base_acc": float((pred[base] == y[base]).mean()) if base.any() else None,
        "incremental_acc": float((pred[novel] == y[novel]).mean()) if novel.any() else None,
    }


def merge_paths_by_class(manifest: dict) -> dict[int, list[str]]:
    result = {label: [] for label in range(NUM_CLASSES)}
    if manifest.get("kind") == "official_index":
        for session in manifest["sessions"]:
            for label, paths in session["paths_by_class"].items():
                result[int(label)] = list(paths)
    else:
        for label, paths in manifest["paths_by_class"].items():
            result[int(label)] = list(paths)
    return result


def support_rows(
    train_paths: list[str], official: dict, override: dict | None = None
) -> dict[int, list[int]]:
    index = {key_for_path(value): row for row, value in enumerate(train_paths)}
    if len(index) != len(train_paths):
        raise ValueError("duplicate train image identities")
    base_paths = merge_paths_by_class(official)
    if override is not None:
        novel_paths = merge_paths_by_class(override)
        for label in range(BASE_CLASSES, NUM_CLASSES):
            base_paths[label] = novel_paths[label]
    result = {}
    for label, paths in base_paths.items():
        expected = 30 if label < BASE_CLASSES else 5
        if len(paths) != expected or len(paths) != len(set(paths)):
            raise ValueError(f"class {label} support count mismatch")
        try:
            result[label] = [index[key_for_path(value)] for value in paths]
        except KeyError as error:
            raise KeyError(f"support identity absent from feature cache: {error}") from error
    return result


def verify_arm_a(result: dict, features: dict, predictions: dict[str, np.ndarray], support: dict) -> dict:
    train_x = features["train"]["pooled"].numpy()
    train_y = features["train"]["labels"].numpy()
    test_x = features["test"]["pooled"].numpy()
    test_y = features["test"]["labels"].numpy()
    prototypes = np.stack([train_x[train_y == label].mean(axis=0) for label in range(NUM_CLASSES)])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = normalized_rows(test_x) @ normalized_rows(prototypes).T
    if not np.isfinite(scores).all():
        raise FloatingPointError("non-finite Arm A cosine scores")
    verified = []
    for row in result["arm_a"]["rows"]:
        seed = int(row["seed"])
        if seed not in ARM_A_SEEDS or support["arm_a"][str(seed)] != row["task_classes"]:
            raise AssertionError("Arm A seed/task manifest mismatch")
        tasks = row["task_classes"]
        session_averages = []
        for session in range(10):
            seen = [label for task in tasks[: session + 1] for label in task]
            mask = np.isin(test_y, seen)
            direct = np.full(len(test_y), -1, dtype=np.int16)
            direct[mask] = np.asarray(seen, dtype=np.int64)[
                np.argmax(scores[mask][:, seen], axis=1)
            ].astype(np.int16)
            stored = predictions[f"arm_a_seed{seed}__s{session:02d}"]
            if not np.array_equal(direct, stored):
                raise AssertionError(f"Arm A cosine prediction mismatch seed={seed} session={session}")
            per_task = []
            for task in tasks[: session + 1]:
                task_mask = np.isin(test_y, task)
                per_task.append(float((stored[task_mask] == test_y[task_mask]).mean()))
            average = float(np.mean(per_task))
            assert_close(average, row["sessions"][session]["row_average_accuracy"], "Arm A row")
            session_averages.append(average)
        assert_close(float(np.mean(session_averages)), row["average_accuracy"], "Arm A AA")
        assert_close(session_averages[-1], row["final_accuracy"], "Arm A final")
        verified.append(seed)
    for metric in ("average_accuracy", "final_accuracy"):
        values = np.asarray([row[metric] for row in result["arm_a"]["rows"]])
        summary = result["arm_a"]["summary"][metric]
        assert_close(float(values.mean()), summary["mean"], f"Arm A {metric} mean")
        assert_close(float(values.std(ddof=1)), summary["sample_sd"], f"Arm A {metric} SD")
    return {"seeds": verified, "cosine_predictions_recomputed": 30}


def verify_arm_b(result: dict, features: dict, predictions: dict[str, np.ndarray], support: dict) -> dict:
    train_x = features["train"]["pooled"].numpy()
    train_y = features["train"]["labels"].numpy()
    train_paths = list(features["train"]["paths"])
    test_x = features["test"]["pooled"].numpy()
    test_y = features["test"]["labels"].numpy()
    official_support = support["arm_b"]["official"]
    settings = [("official", result["arm_b"]["official"], official_support, None)]
    resampled_by_seed = {
        int(row["seed"]): row for row in support["arm_b"]["resampled"]
    }
    for row in result["arm_b"]["resampled_sensitivity"]["rows"]:
        seed = int(row["seed"])
        settings.append((f"seed{seed:02d}", row, official_support, resampled_by_seed[seed]))
    recomputed_cosine = 0
    metric_checks = 0
    for prefix, row, base_manifest, override in settings:
        rows_by_class = support_rows(train_paths, base_manifest, override)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = normalized_rows(test_x) @ normalized_rows(
                class_means(train_x, train_y, rows_by_class)
            ).T
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"non-finite Arm B cosine scores for {prefix}")
        order = row["class_order"]
        metrics = row["metrics"]
        for method in ARM_B_METHODS:
            session_acc = []
            for session in range(11):
                seen = order[: BASE_CLASSES + session * WAY]
                stored = predictions[f"arm_b_{prefix}__{method}__s{session:02d}"]
                if method == "cosine":
                    mask = np.isin(test_y, seen)
                    direct = np.full(len(test_y), -1, dtype=np.int16)
                    direct[mask] = np.asarray(seen, dtype=np.int64)[
                        np.argmax(scores[mask][:, seen], axis=1)
                    ].astype(np.int16)
                    if not np.array_equal(direct, stored):
                        raise AssertionError(f"Arm B cosine mismatch {prefix} session={session}")
                    recomputed_cosine += 1
                observed = metric_from_prediction(stored, test_y, seen)
                expected = metrics[method]["sessions"][session]
                for name in ("all_acc", "base_acc", "incremental_acc"):
                    if observed[name] is None or expected[name] is None:
                        if observed[name] != expected[name]:
                            raise AssertionError(f"Arm B null metric mismatch {prefix}/{method}/{name}")
                    else:
                        assert_close(observed[name], expected[name], f"Arm B {prefix}/{method}/{name}")
                        metric_checks += 1
                session_acc.append(observed["all_acc"])
            assert_close(float(np.mean(session_acc)), metrics[method]["average_accuracy"], "Arm B AA")
            assert_close(session_acc[-1], metrics[method]["final_accuracy"], "Arm B final")
    return {
        "settings": len(settings),
        "cosine_predictions_recomputed": recomputed_cosine,
        "stored_prediction_metric_checks": metric_checks,
    }


def verify(root: Path, output: Path) -> dict:
    result_path = root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for filename, expected in result["artifacts"].items():
        path = root / filename
        if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"artifact identity mismatch: {filename}")
    features = torch.load(root / "canonical_cls_features.pt", map_location="cpu", weights_only=False)
    image_manifest = json.loads((root / "image_manifest.json").read_text(encoding="utf-8"))
    support = json.loads((root / "support_manifest.json").read_text(encoding="utf-8"))
    with np.load(root / "session_predictions.npz") as archive:
        predictions = {name: archive[name] for name in archive.files}
    if features["feature_token"] != "last_hidden_state[:,0]":
        raise AssertionError("feature token identity mismatch")
    if features["train"]["pooled"].shape != (5994, 768) or features["test"]["pooled"].shape != (5794, 768):
        raise ValueError("canonical feature cache shape mismatch")
    if image_manifest["counts"] != {"all": 11788, "train": 5994, "test": 5794}:
        raise ValueError("image manifest count mismatch")
    for split in ("train", "test"):
        image_rows = [row for row in image_manifest["images"] if row["split"] == split]
        if [row["relative_path"] for row in image_rows] != list(features[split]["paths"]):
            raise ValueError(f"{split} image/feature path identity mismatch")
        if not np.array_equal(
            np.asarray([row["label"] for row in image_rows], dtype=np.int64),
            features[split]["labels"].numpy(),
        ):
            raise ValueError(f"{split} image/feature label identity mismatch")
    expected_arrays = 1 + 3 * 10 + 3 * 11 + 20 * 3 * 11
    if len(predictions) != expected_arrays or not np.array_equal(
        predictions["test_labels"], features["test"]["labels"].numpy()
    ):
        raise ValueError("prediction archive identity/count mismatch")
    arm_a = verify_arm_a(result, features, predictions, support)
    arm_b = verify_arm_b(result, features, predictions, support)
    payload = {
        "schema_version": 1,
        "kind": "independent_sdg_caev_simplecil_matched_validation",
        "accepted": True,
        "result_sha256": file_sha256(result_path),
        "artifact_hashes_verified": len(result["artifacts"]),
        "prediction_arrays_verified": len(predictions),
        "arm_a": arm_a,
        "arm_b": arm_b,
    }
    write_json_new(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    verify(arguments.root, arguments.output)
