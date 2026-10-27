#!/usr/bin/env python3
"""ImageNet-1K scale/subgroup SDG audit on public SigLIP features.

The source artifact contains L2-normalized SigLIP-B/16 embeddings for the
1,281,167 ImageNet-1K training images.  Since it does not contain the official
validation split, this audit uses an index-stable within-class
inner/validation/test split and labels the result accordingly.  It is not a
replacement for an official ImageNet-1K validation-set evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


SOURCE_REPO = "mlnomad/imagenet-1k-224-clip-embeddings"
SOURCE_REVISION = "3015896bc84a32b6aa87b25a781540efea6b4945"
SOURCE_URL = (
    "https://huggingface.co/datasets/"
    f"{SOURCE_REPO}/resolve/{SOURCE_REVISION}/train_embeddings.npz"
)
SOURCE_BYTES = 3_426_505_979
SOURCE_SHA256 = "94464d5c10dafe16672f3bba1e45b2d765c7abdfb07eac1ca89d4f4dd6b258b2"
EXPECTED_ROWS = 1_281_167
EXPECTED_DIM = 768
EXPECTED_CLASSES = 1_000

REG_GRID = (
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
)

# These are the ImageNet-1K labels whose WordNet synsets descend from the
# stated root synset under WordNet 3.0.  The mapping is the canonical
# ImageNet class-index order (n01440764 ... n15075141).
SEMANTIC_GROUPS = {
    "global_1k": {
        "wordnet_root": None,
        "labels": tuple(range(1_000)),
    },
    "dogs": {
        "wordnet_root": "dog.n.01",
        "labels": tuple(range(151, 269)),
    },
    "birds": {
        "wordnet_root": "bird.n.01",
        "labels": (
            *range(7, 25),
            *range(80, 101),
            *range(127, 147),
        ),
    },
    "vehicles": {
        "wordnet_root": "vehicle.n.01",
        "labels": (
            403,
            404,
            405,
            407,
            408,
            417,
            428,
            436,
            444,
            450,
            468,
            472,
            484,
            510,
            511,
            537,
            547,
            554,
            555,
            561,
            565,
            569,
            573,
            575,
            576,
            586,
            603,
            609,
            612,
            625,
            627,
            628,
            656,
            657,
            660,
            661,
            665,
            670,
            671,
            675,
            690,
            705,
            717,
            724,
            734,
            751,
            757,
            780,
            791,
            802,
            803,
            812,
            814,
            817,
            820,
            829,
            833,
            847,
            864,
            866,
            867,
            870,
            871,
            880,
            895,
            913,
            914,
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--stats-batch-size", type=int, default=16_384)
    parser.add_argument("--eval-batch-size", type=int, default=2_048)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_source(path: Path) -> dict:
    """Download once into the local source directory and verify exact identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_name(path.name + ".partial")
        command = [
            "curl",
            "-L",
            "-C",
            "-",
            "--connect-timeout",
            "20",
            "--max-time",
            "7200",
            "-o",
            str(partial),
            SOURCE_URL,
        ]
        subprocess.run(command, check=True)
        if partial.stat().st_size != SOURCE_BYTES:
            raise ValueError(
                f"downloaded source size mismatch: {partial.stat().st_size} != {SOURCE_BYTES}"
            )
        digest = sha256_file(partial)
        if digest != SOURCE_SHA256:
            raise ValueError(f"downloaded source sha256 mismatch: {digest}")
        os.replace(partial, path)

    if path.stat().st_size != SOURCE_BYTES:
        raise ValueError(f"source size mismatch: {path.stat().st_size} != {SOURCE_BYTES}")
    digest = sha256_file(path)
    if digest != SOURCE_SHA256:
        raise ValueError(f"source sha256 mismatch: {digest}")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": digest,
        "url": SOURCE_URL,
        "repo": SOURCE_REPO,
        "revision": SOURCE_REVISION,
    }


def load_source(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"embeddings", "labels", "original_indices"}:
            raise ValueError(f"unexpected NPZ keys: {archive.files}")
        features = archive["embeddings"]
        labels = archive["labels"]
        original_indices = archive["original_indices"]

    if features.shape != (EXPECTED_ROWS, EXPECTED_DIM):
        raise ValueError(f"unexpected embedding shape: {features.shape}")
    if labels.shape != (EXPECTED_ROWS,) or original_indices.shape != (EXPECTED_ROWS,):
        raise ValueError(
            f"unexpected metadata shapes: labels={labels.shape}, indices={original_indices.shape}"
        )
    if features.dtype != np.float32:
        raise ValueError(f"expected float32 embeddings, found {features.dtype}")
    labels = np.asarray(labels, dtype=np.int64)
    original_indices = np.asarray(original_indices, dtype=np.int64)
    if labels.min() != 0 or labels.max() != EXPECTED_CLASSES - 1:
        raise ValueError(f"unexpected label range: {labels.min()}..{labels.max()}")
    counts = np.bincount(labels, minlength=EXPECTED_CLASSES)
    if len(counts) != EXPECTED_CLASSES or np.any(counts < 1):
        raise ValueError("not all ImageNet-1K labels are represented")
    if len(np.unique(original_indices)) != EXPECTED_ROWS:
        raise ValueError("original_indices are not unique")
    if not np.isfinite(features).all():
        raise FloatingPointError("source embeddings contain non-finite values")

    norms = np.linalg.norm(features, axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    if max_norm_error > 1e-3:
        raise ValueError(f"source embeddings are not unit normalized: max error={max_norm_error}")
    metadata = {
        "rows": int(len(features)),
        "dim": int(features.shape[1]),
        "n_classes": int(len(counts)),
        "class_count_min": int(counts.min()),
        "class_count_median": float(np.median(counts)),
        "class_count_max": int(counts.max()),
        "embedding_dtype": str(features.dtype),
        "unit_norm_max_abs_error": max_norm_error,
    }
    return features, labels, original_indices, metadata


def splitmix64(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))


def stable_within_class_split(
    labels: np.ndarray,
    original_indices: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return 0=inner, 1=validation, 2=test with 80/10/10 class-wise splits."""

    labels = np.asarray(labels, dtype=np.int64)
    original_indices = np.asarray(original_indices, dtype=np.int64)
    if labels.shape != original_indices.shape:
        raise ValueError("labels and original_indices must have the same shape")
    assignment = np.full(len(labels), 255, dtype=np.uint8)
    seed_value = np.uint64(seed) ^ np.uint64(0xD1B54A32D192ED03)
    for cls in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == cls)
        keys = splitmix64(original_indices[indices].astype(np.uint64) ^ seed_value)
        ordered = indices[np.argsort(keys, kind="stable")]
        n_test = max(1, int(round(0.10 * len(ordered))))
        n_validation = max(1, int(round(0.10 * len(ordered))))
        if n_test + n_validation >= len(ordered):
            raise ValueError(f"class {cls} has too few examples for a three-way split")
        assignment[ordered[:n_test]] = 2
        assignment[ordered[n_test : n_test + n_validation]] = 1
        assignment[ordered[n_test + n_validation :]] = 0
    if np.any(assignment == 255):
        raise RuntimeError("split assignment left unassigned examples")
    return assignment


def local_label_map(class_labels: Iterable[int]) -> np.ndarray:
    class_labels = tuple(int(value) for value in class_labels)
    mapping = np.full(EXPECTED_CLASSES, -1, dtype=np.int64)
    mapping[np.asarray(class_labels, dtype=np.int64)] = np.arange(len(class_labels))
    return mapping


def tensor_indices(indices: np.ndarray, start: int, stop: int) -> np.ndarray:
    return np.ascontiguousarray(indices[start:stop], dtype=np.int64)


def sufficient_statistics(
    features: np.ndarray,
    local_labels: np.ndarray,
    indices: np.ndarray,
    n_classes: int,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    dim = features.shape[1]
    counts = torch.zeros(n_classes, dtype=torch.float64, device=device)
    sums = torch.zeros((n_classes, dim), dtype=torch.float64, device=device)
    cross = torch.zeros((dim, dim), dtype=torch.float64, device=device)
    for start in range(0, len(indices), batch_size):
        selected = tensor_indices(indices, start, min(start + batch_size, len(indices)))
        batch = torch.from_numpy(np.ascontiguousarray(features[selected])).to(
            device=device, dtype=torch.float64
        )
        labels = torch.from_numpy(np.ascontiguousarray(local_labels[selected])).to(device=device)
        if torch.any(labels < 0):
            raise ValueError("statistics indices contain labels outside the arm")
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))
        sums.index_add_(0, labels, batch)
        cross.add_(batch.T @ batch)
    if torch.any(counts <= 0):
        raise ValueError("every arm class must have support examples")
    return {"counts": counts, "sums": sums, "cross": cross}


def add_statistics(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> dict:
    return {key: first[key] + second[key] for key in first}


def covariance_bundle(stats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    counts = stats["counts"]
    sums = stats["sums"]
    cross = stats["cross"]
    n_total = counts.sum()
    means = sums / counts[:, None]
    global_mean = sums.sum(0) / n_total
    total_covariance = (
        cross - n_total * torch.outer(global_mean, global_mean)
    ) / torch.clamp(n_total - 1.0, min=1.0)
    between = (means.T * counts[None, :]) @ means
    within_covariance = (cross - between) / torch.clamp(
        n_total - float(len(counts)), min=1.0
    )
    total_covariance = 0.5 * (total_covariance + total_covariance.T)
    within_covariance = 0.5 * (within_covariance + within_covariance.T)
    return {
        "means": means,
        "global_mean": global_mean,
        "total_covariance": total_covariance,
        "within_covariance": within_covariance,
        "counts": counts,
    }


def descending_eigh(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh(covariance)
    values = torch.clamp(torch.flip(values, dims=(0,)), min=0.0)
    vectors = torch.flip(vectors, dims=(1,))
    return values, vectors


def normalized_rows(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values, p=2.0, dim=1, eps=1e-12)


def predict_cosine(
    features: np.ndarray,
    indices: np.ndarray,
    prototypes: torch.Tensor,
    class_labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    score_dtype: torch.dtype,
) -> np.ndarray:
    prototype_scores = normalized_rows(prototypes.to(dtype=score_dtype))
    output = []
    for start in range(0, len(indices), batch_size):
        selected = tensor_indices(indices, start, min(start + batch_size, len(indices)))
        batch = torch.from_numpy(np.ascontiguousarray(features[selected])).to(
            device=device, dtype=score_dtype
        )
        scores = normalized_rows(batch) @ prototype_scores.T
        output.append(torch.argmax(scores, dim=1).cpu().numpy())
    local = np.concatenate(output)
    return class_labels[local]


def family_parameters(
    family: str,
    bundle: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor | float]:
    if family == "wcos":
        covariance = bundle["total_covariance"]
        values, vectors = descending_eigh(covariance)
        prototype_pc = (bundle["means"] - bundle["global_mean"]) @ vectors
        return {
            "values": values,
            "vectors": vectors,
            "prototype_pc": prototype_pc,
            "center": bundle["global_mean"],
            "scale": float(torch.trace(covariance).item() / covariance.shape[0]),
        }
    if family == "lda":
        covariance = bundle["within_covariance"]
        values, vectors = descending_eigh(covariance)
        mean_pc = bundle["means"] @ vectors
        return {
            "values": values,
            "vectors": vectors,
            "mean_pc": mean_pc,
            "center": torch.zeros_like(bundle["global_mean"]),
            "scale": float(torch.trace(covariance).item() / covariance.shape[0]),
        }
    raise ValueError(f"unknown evaluator family: {family}")


def predict_family(
    family: str,
    features: np.ndarray,
    indices: np.ndarray,
    parameters: dict[str, torch.Tensor | float],
    base_regularizer: float,
    class_labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    score_dtype: torch.dtype,
) -> tuple[np.ndarray, float]:
    actual_regularizer = float(base_regularizer * max(float(parameters["scale"]), 1e-12))
    values = parameters["values"]
    vectors = parameters["vectors"]
    center = parameters["center"]
    if not isinstance(values, torch.Tensor) or not isinstance(vectors, torch.Tensor):
        raise TypeError("invalid spectral parameter bundle")
    denominator = values + actual_regularizer

    if family == "wcos":
        prototype_pc = parameters["prototype_pc"]
        if not isinstance(prototype_pc, torch.Tensor):
            raise TypeError("missing whitened-cosine prototypes")
        reference = normalized_rows(
            (prototype_pc / torch.sqrt(denominator)[None, :]).to(dtype=score_dtype)
        )
        bias = None
    elif family == "lda":
        mean_pc = parameters["mean_pc"]
        if not isinstance(mean_pc, torch.Tensor):
            raise TypeError("missing LDA class means")
        reference = (mean_pc / denominator[None, :]).to(dtype=score_dtype)
        bias = (-0.5 * torch.sum(mean_pc * mean_pc / denominator[None, :], dim=1)).to(
            dtype=score_dtype
        )
    else:
        raise ValueError(f"unknown evaluator family: {family}")

    vectors = vectors.to(dtype=score_dtype)
    center = center.to(dtype=score_dtype)
    output = []
    for start in range(0, len(indices), batch_size):
        selected = tensor_indices(indices, start, min(start + batch_size, len(indices)))
        batch = torch.from_numpy(np.ascontiguousarray(features[selected])).to(
            device=device, dtype=score_dtype
        )
        pc = (batch - center[None, :]) @ vectors
        if family == "wcos":
            query = normalized_rows(pc / torch.sqrt(denominator).to(score_dtype)[None, :])
            scores = query @ reference.T
        else:
            scores = pc @ reference.T + bias[None, :]
        output.append(torch.argmax(scores, dim=1).cpu().numpy())
    local = np.concatenate(output)
    return class_labels[local], actual_regularizer


def select_base_regularizer(
    family: str,
    features: np.ndarray,
    validation_indices: np.ndarray,
    validation_labels: np.ndarray,
    parameters: dict[str, torch.Tensor | float],
    class_labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    score_dtype: torch.dtype,
) -> dict:
    scores = []
    actual = []
    for candidate in REG_GRID:
        prediction, actual_regularizer = predict_family(
            family,
            features,
            validation_indices,
            parameters,
            candidate,
            class_labels,
            device=device,
            batch_size=batch_size,
            score_dtype=score_dtype,
        )
        scores.append(float(np.mean(prediction == validation_labels)))
        actual.append(actual_regularizer)
    best = int(np.flatnonzero(np.asarray(scores) == np.max(scores))[0])
    return {
        "selected_base": float(REG_GRID[best]),
        "selected_actual_on_inner": float(actual[best]),
        "validation_accuracies": scores,
        "candidate_bases": list(REG_GRID),
        "regularization_mode": "relative_trace",
    }


def class_cluster_gap_ci(
    truth: np.ndarray,
    baseline: np.ndarray,
    method: np.ndarray,
    class_labels: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    truth = np.asarray(truth)
    diff = (method == truth).astype(np.float64) - (baseline == truth).astype(np.float64)
    sums = np.asarray([diff[truth == cls].sum() for cls in class_labels], dtype=np.float64)
    counts = np.asarray([np.sum(truth == cls) for cls in class_labels], dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError("every arm class must have test examples")
    point = float(100.0 * sums.sum() / counts.sum())
    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(n_boot):
        draw = rng.randint(0, len(class_labels), size=len(class_labels))
        samples.append(float(100.0 * sums[draw].sum() / counts[draw].sum()))
    return {
        "point_pp": point,
        "ci95_pp": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
        "bootstrap_unit": "ImageNet class; all test examples for a sampled class retained",
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def prediction_sha256(predictions: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(predictions, dtype=np.int16).tobytes()).hexdigest()


def run_arm(
    name: str,
    group: dict,
    features: np.ndarray,
    labels: np.ndarray,
    assignment: np.ndarray,
    *,
    device: torch.device,
    stats_batch_size: int,
    eval_batch_size: int,
    n_boot: int,
    seed: int,
) -> dict:
    class_labels = np.asarray(group["labels"], dtype=np.int64)
    mapping = local_label_map(class_labels)
    local_labels = mapping[labels]
    arm = local_labels >= 0
    inner_indices = np.flatnonzero(arm & (assignment == 0))
    validation_indices = np.flatnonzero(arm & (assignment == 1))
    test_indices = np.flatnonzero(arm & (assignment == 2))

    inner_stats = sufficient_statistics(
        features,
        local_labels,
        inner_indices,
        len(class_labels),
        device=device,
        batch_size=stats_batch_size,
    )
    validation_stats = sufficient_statistics(
        features,
        local_labels,
        validation_indices,
        len(class_labels),
        device=device,
        batch_size=stats_batch_size,
    )
    inner_bundle = covariance_bundle(inner_stats)
    validation_truth = labels[validation_indices]
    selections = {}
    for family in ("wcos", "lda"):
        selections[family] = select_base_regularizer(
            family,
            features,
            validation_indices,
            validation_truth,
            family_parameters(family, inner_bundle),
            class_labels,
            device=device,
            batch_size=eval_batch_size,
            score_dtype=torch.float32,
        )

    refit_bundle = covariance_bundle(add_statistics(inner_stats, validation_stats))
    test_truth = labels[test_indices]
    predictions = {
        "cos": predict_cosine(
            features,
            test_indices,
            refit_bundle["means"],
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
            features,
            test_indices,
            family_parameters(family, refit_bundle),
            selections[family]["selected_base"],
            class_labels,
            device=device,
            batch_size=eval_batch_size,
            score_dtype=torch.float32,
        )

    accuracies = {
        method: float(np.mean(prediction == test_truth))
        for method, prediction in predictions.items()
    }
    gaps = {
        family: class_cluster_gap_ci(
            test_truth,
            predictions["cos"],
            predictions[family],
            class_labels,
            n_boot=n_boot,
            seed=seed + (101 if family == "wcos" else 211),
        )
        for family in ("wcos", "lda")
    }
    return {
        "arm": name,
        "wordnet_root": group["wordnet_root"],
        "class_labels": class_labels.tolist(),
        "n_classes": int(len(class_labels)),
        "split_counts": {
            "inner": int(len(inner_indices)),
            "validation": int(len(validation_indices)),
            "refit_support": int(len(inner_indices) + len(validation_indices)),
            "test": int(len(test_indices)),
        },
        "selection": selections,
        "refit_actual_regularizers": actual_regularizers,
        "test_accuracies": accuracies,
        "fixed_evaluator_gaps": gaps,
        "best_whitening_minus_cos_pp": float(
            100.0 * (max(accuracies["wcos"], accuracies["lda"]) - accuracies["cos"])
        ),
        "prediction_sha256": {
            method: prediction_sha256(prediction)
            for method, prediction in predictions.items()
        },
    }


def report_markdown(payload: dict) -> str:
    lines = [
        "# ImageNet-1K SigLIP scale/subgroup SDG audit",
        "",
        "This is a train-split-holdout stress test on public "
        "precomputed unit-normalized SigLIP-B/16 embeddings. It is not an "
        "official ImageNet validation-set result and not a DINOv2 result.",
        "",
        "| arm | classes | refit support | test | Cos | WCos-TV | LDA-TV | WCos−Cos pp [class CI] | LDA−Cos pp [class CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in payload["arms"]:
        accuracy = arm["test_accuracies"]
        wcos = arm["fixed_evaluator_gaps"]["wcos"]
        lda = arm["fixed_evaluator_gaps"]["lda"]
        lines.append(
            f"| {arm['arm']} | {arm['n_classes']} | {arm['split_counts']['refit_support']} | "
            f"{arm['split_counts']['test']} | {100*accuracy['cos']:.2f} | "
            f"{100*accuracy['wcos']:.2f} | {100*accuracy['lda']:.2f} | "
            f"{wcos['point_pp']:+.2f} [{wcos['ci95_pp'][0]:+.2f}, {wcos['ci95_pp'][1]:+.2f}] | "
            f"{lda['point_pp']:+.2f} [{lda['ci95_pp'][0]:+.2f}, {lda['ci95_pp'][1]:+.2f}] |"
        )
    lines.extend(
        [
            "",
            f"- split seed: `{payload['protocol']['split_seed']}`",
            "- split: class-wise 80% inner / 10% validation / 10% test, keyed by original image index",
            "- regularization: inner-support fit, validation-only selection, inner+validation refit",
            "- uncertainty: class-cluster bootstrap; all test examples of a sampled class stay together",
            f"- source sha256: `{payload['source']['sha256']}`",
            f"- git commit: `{payload['git_commit']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="sdg-imagenet1k-",
        suffix=path.suffix,
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        shutil.copyfile(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {args.output_root}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    source = ensure_source(args.data_path)
    features, labels, original_indices, source_metadata = load_source(args.data_path)
    assignment = stable_within_class_split(labels, original_indices, args.seed)

    arms = []
    for name in ("global_1k", "dogs", "birds", "vehicles"):
        arms.append(
            run_arm(
                name,
                SEMANTIC_GROUPS[name],
                features,
                labels,
                assignment,
                device=device,
                stats_batch_size=args.stats_batch_size,
                eval_batch_size=args.eval_batch_size,
                n_boot=args.n_boot,
                seed=args.seed,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "kind": "sdg_imagenet1k_siglip_train_holdout_audit",
        "git_commit": args.commit,
        "source": {
            **source,
            **source_metadata,
            "source_split": "ImageNet-1K train",
            "feature_model": "google/siglip-base-patch16-224",
            "feature_normalization": "L2-normalized by artifact author",
            "contains_raw_images": False,
            "license_field": "other",
        },
        "protocol": {
            "split_seed": int(args.seed),
            "split_unit": "within ImageNet class, stable hash of original_index",
            "split_fractions": {"inner": 0.8, "validation": 0.1, "test": 0.1},
            "selection": "validation labels select relative-trace regularizer; test labels never select",
            "regularization_grid": list(REG_GRID),
            "score_dtype": "float32",
            "covariance_and_eigendecomposition_dtype": "float64",
            "n_boot": int(args.n_boot),
            "bootstrap_unit": "ImageNet class",
            "semantic_groups": {
                name: {
                    "wordnet_root": value["wordnet_root"],
                    "n_classes": len(value["labels"]),
                }
                for name, value in SEMANTIC_GROUPS.items()
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
        },
        "scope_limits": [
            "post-submission evidence",
            "public precomputed unit-normalized SigLIP-B/16 features",
            "within-train holdout, not the official ImageNet validation split",
            "not DINOv2 and not end-to-end representation training",
            "subgroup tasks classify only among classes inside each WordNet subgroup",
        ],
        "arms": arms,
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_new(
        args.output_root / "result.json",
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    write_new(args.output_root / "report.md", report_markdown(payload))
    print(json.dumps({"output_root": str(args.output_root), "arms": len(arms)}))


if __name__ == "__main__":
    main()
