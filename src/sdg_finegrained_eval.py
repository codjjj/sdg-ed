#!/usr/bin/env python3
"""Chunked frozen-protocol evaluator for the fine-grained unseen datasets.

Runs the core train-validated Cos/WCos/LDA protocol (same candidate grids and
inner validation split as sdg_stepb_cached_eval.py) on one dataset-shot group,
in query chunks so that large test splits fit in GPU memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from sdg_finegrained_manifest import (
    NONFULL_SEEDS,
    NONFULL_SHOTS,
    canonical_json_bytes,
    load_and_verify_manifest,
    sha256_file,
)
from sdg_stepb_cached_eval import LDA_REGS, WCOS_EPS
from spectral_decision_gap_ed import spearman


DEFAULT_REG = 1e-4


def write_json_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def write_npy_new(path: Path, array: np.ndarray) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as handle:
            np.save(handle, np.asarray(array, dtype=np.int64), allow_pickle=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return sha256_file(path)


def load_feature_bundle(feature_root: Path, dataset_manifest: dict) -> tuple[torch.Tensor, ...]:
    ready_path = feature_root / "FEATURE_READY.json"
    feature_manifest_path = feature_root / "feature_manifest.json"
    if not ready_path.exists() or not feature_manifest_path.exists():
        raise FileNotFoundError(f"feature readiness artifacts missing: {feature_root}")
    ready = json.loads(ready_path.read_text())
    feature_manifest = json.loads(feature_manifest_path.read_text())
    if ready["feature_manifest_sha256"] != sha256_file(feature_manifest_path):
        raise ValueError("feature READY/manifest hash mismatch")
    if feature_manifest["dataset_identity_sha256"] != dataset_manifest["dataset_identity_sha256"]:
        raise ValueError("feature/dataset identity mismatch")
    if feature_manifest["feature_token"] != "last_hidden_state[:,0,:]":
        raise ValueError("unexpected feature-token contract")
    outputs = []
    for split in ("train", "test"):
        path = feature_root / f"{split}.pt"
        expected = feature_manifest["splits"][split]["cache_sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"{split} feature cache hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        features = payload["pooled"].float().contiguous()
        labels = payload["labels"].long().contiguous()
        indices = payload["manifest_split_indices"].long().contiguous()
        if not torch.equal(indices, torch.arange(len(labels), dtype=torch.long)):
            raise ValueError(f"{split} split-index identity mismatch")
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
            raise ValueError(f"invalid {split} feature shapes")
        if not torch.isfinite(features).all():
            raise FloatingPointError(f"non-finite {split} feature cache")
        outputs.extend([features, labels])
    return tuple(outputs)


def train_val_indices(labels: np.ndarray, seed: int, val_frac: float = 0.2, max_val_per_class: int = 50):
    """Index-only form of the frozen Step-B train/validation split."""

    labels = np.asarray(labels)
    rng = np.random.RandomState(int(seed) + 7919)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            return np.arange(len(labels)), None, {
                "mode": "no_validation",
                "reason": "class_has_less_than_2_examples",
            }
        rng.shuffle(indices)
        if len(indices) <= 5:
            n_validation = 1
        else:
            n_validation = max(1, int(round(len(indices) * val_frac)))
            n_validation = min(n_validation, max_val_per_class, len(indices) - 1)
        validation_indices.extend(indices[:n_validation].tolist())
        train_indices.extend(indices[n_validation:].tolist())
    train_array = np.asarray(sorted(train_indices), dtype=np.int64)
    validation_array = np.asarray(sorted(validation_indices), dtype=np.int64)
    return train_array, validation_array, {
        "mode": "class_balanced_holdout",
        "val_per_class_cap": int(max_val_per_class),
        "val_frac": float(val_frac),
        "n_inner_train": int(len(train_array)),
        "n_val": int(len(validation_array)),
    }


def class_means(features: torch.Tensor, labels: torch.Tensor, n_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros((n_classes, features.shape[1]), dtype=features.dtype, device=features.device)
    sums.index_add_(0, labels, features)
    counts = torch.bincount(labels, minlength=n_classes)
    if len(counts) != n_classes or torch.any(counts == 0):
        raise ValueError("every contiguous class must occur in the support set")
    return sums / counts.to(features.dtype)[:, None], counts


def total_statistics(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.T @ centered / max(1, len(features) - 1)
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True)
    return mean, torch.clamp(values[order], min=0.0), vectors[:, order]


def pooled_statistics(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    means, _ = class_means(features, labels, n_classes)
    centered = features - means[labels]
    covariance = centered.T @ centered / max(1, len(features) - n_classes)
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True)
    return means, covariance, torch.clamp(values[order], min=0.0), vectors[:, order]


def chunked_linear_predictions(
    query_cpu: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    query_chunk: int,
) -> tuple[np.ndarray, dict]:
    predictions = []
    maximum_shape = (0, int(weights.shape[0]))
    for start in range(0, len(query_cpu), query_chunk):
        query = query_cpu[start : start + query_chunk].to(
            device=weights.device,
            dtype=weights.dtype,
            non_blocking=True,
        )
        scores = query @ weights.T
        if bias is not None:
            scores = scores + bias[None, :]
        predictions.append(torch.argmax(scores, dim=1).cpu().numpy())
        maximum_shape = max(maximum_shape, tuple(scores.shape), key=lambda shape: shape[0] * shape[1])
        del query, scores
    return np.concatenate(predictions).astype(np.int64), {
        "query_chunk": int(query_chunk),
        "n_classes": int(weights.shape[0]),
        "max_score_matrix_shape": list(maximum_shape),
        "full_score_matrix_materialized": False,
    }


def cosine_model(features: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    means, _ = class_means(features, labels, n_classes)
    return torch.nn.functional.normalize(means, dim=1)


def cosine_predictions(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    query_cpu: torch.Tensor,
    n_classes: int,
    query_chunk: int,
) -> tuple[np.ndarray, dict]:
    prototypes = cosine_model(train_features, train_labels, n_classes)
    predictions = []
    maximum_shape = (0, int(prototypes.shape[0]))
    for start in range(0, len(query_cpu), query_chunk):
        query = query_cpu[start : start + query_chunk].to(
            device=prototypes.device,
            dtype=prototypes.dtype,
            non_blocking=True,
        )
        query = torch.nn.functional.normalize(query, dim=1)
        scores = query @ prototypes.T
        predictions.append(torch.argmax(scores, dim=1).cpu().numpy())
        maximum_shape = max(maximum_shape, tuple(scores.shape), key=lambda shape: shape[0] * shape[1])
        del query, scores
    return np.concatenate(predictions).astype(np.int64), {
        "query_chunk": int(query_chunk),
        "n_classes": int(prototypes.shape[0]),
        "max_score_matrix_shape": list(maximum_shape),
        "full_score_matrix_materialized": False,
    }


def lda_weights_from_stats(
    means: torch.Tensor,
    values: torch.Tensor,
    vectors: torch.Tensor,
    regularizer: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inverse = (vectors * (1.0 / (values + float(regularizer)))) @ vectors.T
    weights = means @ inverse
    bias = -0.5 * torch.sum(means * weights, dim=1)
    return weights, bias


def wcos_weights_from_stats(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    mean: torch.Tensor,
    values: torch.Tensor,
    vectors: torch.Tensor,
    epsilon: float,
    n_classes: int,
) -> torch.Tensor:
    means, _ = class_means(train_features, train_labels, n_classes)
    projected_means = (means - mean) @ vectors
    return wcos_prototypes_from_projected_means(projected_means, values, epsilon)


def wcos_prototypes_from_projected_means(
    projected_means: torch.Tensor,
    values: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    scale = 1.0 / torch.sqrt(values + float(epsilon))
    return torch.nn.functional.normalize(projected_means * scale[None, :], dim=1)


def predict_wcos(
    query_cpu: torch.Tensor,
    prototypes: torch.Tensor,
    mean: torch.Tensor,
    values: torch.Tensor,
    vectors: torch.Tensor,
    epsilon: float,
    query_chunk: int,
) -> tuple[np.ndarray, dict]:
    predictions = []
    maximum_shape = (0, int(prototypes.shape[0]))
    scale = 1.0 / torch.sqrt(values + float(epsilon))
    for start in range(0, len(query_cpu), query_chunk):
        query = query_cpu[start : start + query_chunk].to(
            device=prototypes.device,
            dtype=prototypes.dtype,
            non_blocking=True,
        )
        transformed = ((query - mean) @ vectors) * scale[None, :]
        transformed = torch.nn.functional.normalize(transformed, dim=1)
        scores = transformed @ prototypes.T
        predictions.append(torch.argmax(scores, dim=1).cpu().numpy())
        maximum_shape = max(maximum_shape, tuple(scores.shape), key=lambda shape: shape[0] * shape[1])
        del query, transformed, scores
    return np.concatenate(predictions).astype(np.int64), {
        "query_chunk": int(query_chunk),
        "n_classes": int(prototypes.shape[0]),
        "max_score_matrix_shape": list(maximum_shape),
        "full_score_matrix_materialized": False,
    }


def accuracy(predictions: np.ndarray, labels: torch.Tensor) -> float:
    return float(np.mean(np.asarray(predictions) == labels.cpu().numpy()))


def select_regularizer(
    family: str,
    inner_features: torch.Tensor,
    inner_labels: torch.Tensor,
    validation_cpu: torch.Tensor | None,
    validation_labels: torch.Tensor | None,
    n_classes: int,
    candidates: list[float],
    query_chunk: int,
) -> dict:
    if validation_cpu is None:
        return {
            "selected": DEFAULT_REG,
            "selection": "fixed_default_no_validation",
            "validation_scores": [],
        }
    scores = []
    if family == "lda":
        means, _, values, vectors = pooled_statistics(inner_features, inner_labels, n_classes)
        for candidate in candidates:
            weights, bias = lda_weights_from_stats(means, values, vectors, candidate)
            predictions, _ = chunked_linear_predictions(
                validation_cpu,
                weights,
                bias,
                query_chunk=query_chunk,
            )
            scores.append(accuracy(predictions, validation_labels))
    elif family == "wcos":
        mean, values, vectors = total_statistics(inner_features)
        means, _ = class_means(inner_features, inner_labels, n_classes)
        projected_means = (means - mean) @ vectors
        for candidate in candidates:
            prototypes = wcos_prototypes_from_projected_means(
                projected_means,
                values,
                candidate,
            )
            predictions, _ = predict_wcos(
                validation_cpu,
                prototypes,
                mean,
                values,
                vectors,
                candidate,
                query_chunk,
            )
            scores.append(accuracy(predictions, validation_labels))
    else:
        raise ValueError(f"unsupported family: {family}")
    score_array = np.asarray(scores, dtype=np.float64)
    selected_index = int(np.flatnonzero(score_array == score_array.max())[0])
    return {
        "selected": float(candidates[selected_index]),
        "selection": "support_validation",
        "validation_scores": [float(value) for value in scores],
    }


def router_diagnostics(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    n_classes: int,
    shot: str,
) -> dict:
    mean, values, vectors = total_statistics(support_features)
    projected = (support_features - mean) @ vectors
    class_projected_means, counts = class_means(projected, support_labels, n_classes)
    global_mean = projected.mean(dim=0)
    between = torch.sum(
        counts.to(projected.dtype)[:, None] * (class_projected_means - global_mean) ** 2,
        dim=0,
    ) / max(1, n_classes - 1)
    within = torch.sum((projected - class_projected_means[support_labels]) ** 2, dim=0)
    within = within / max(1, len(projected) - n_classes)
    fisher = between / (within + 1e-12)
    mass = fisher / (torch.sum(fisher) + 1e-12)
    total = torch.sum(values)
    positive_threshold = max(float(torch.max(values).item()) * 1e-12, 1e-18)
    positive = values[values > positive_threshold]
    if len(positive) == 0:
        log_condition = 0.0
        effective_rank = 0.0
    else:
        condition = float((positive.max() / positive.min()).item())
        probability = positive / positive.sum()
        effective_rank = float(torch.exp(-torch.sum(probability * torch.log(probability + 1e-12))).item())
        log_condition = float(np.log10(condition + 1.0))
    eigen_log = np.log(np.maximum(values.detach().cpu().numpy(), 1e-18))
    fisher_log = np.log(np.maximum(fisher.detach().cpu().numpy(), 1e-18))
    return {
        "top1_var": float(values[:1].sum().item() / total.item()),
        "top5_var": float(values[:5].sum().item() / total.item()),
        "top10_var": float(values[:10].sum().item() / total.item()),
        "top50_var": float(values[:50].sum().item() / total.item()),
        "tail128_fisher": float(mass[128:].sum().item()),
        "tail256_fisher": float(mass[256:].sum().item()),
        "bottom_half_fisher": float(mass[len(mass) // 2 :].sum().item()),
        "spearman_logeig_logfisher": spearman(eigen_log, fisher_log),
        "log10_condition": log_condition,
        "effective_rank": effective_rank,
        "log10_n_train": float(np.log10(max(1, len(support_labels)))),
        "shot_numeric": 999.0 if shot == "full" else float(shot),
    }


def evaluate_seed(
    train_cpu: torch.Tensor,
    train_labels_cpu: torch.Tensor,
    test_cpu: torch.Tensor,
    test_labels_cpu: torch.Tensor,
    support_indices: np.ndarray,
    *,
    shot: str,
    seed: int,
    device: str,
    query_chunk: int,
    lda_regs: list[float],
    wcos_eps: list[float],
) -> tuple[dict, dict[str, np.ndarray]]:
    support_cpu = train_cpu[torch.from_numpy(support_indices)]
    support_labels_cpu = train_labels_cpu[torch.from_numpy(support_indices)]
    n_classes = int(torch.unique(train_labels_cpu).numel())
    expected_classes = torch.arange(n_classes, dtype=torch.long)
    if not torch.equal(torch.unique(support_labels_cpu), expected_classes):
        raise ValueError("support does not cover every contiguous class")

    support = support_cpu.to(device=device, dtype=torch.float64)
    support_labels = support_labels_cpu.to(device=device)
    inner_indices, validation_indices, split = train_val_indices(
        support_labels_cpu.numpy(), seed
    )
    inner = support[torch.from_numpy(inner_indices).to(device)]
    inner_labels = support_labels[torch.from_numpy(inner_indices).to(device)]
    if validation_indices is None:
        validation_cpu = None
        validation_labels = None
    else:
        validation_cpu = support_cpu[torch.from_numpy(validation_indices)]
        validation_labels = support_labels_cpu[torch.from_numpy(validation_indices)]

    lda_selection = select_regularizer(
        "lda",
        inner,
        inner_labels,
        validation_cpu,
        validation_labels,
        n_classes,
        lda_regs,
        query_chunk,
    )
    wcos_selection = select_regularizer(
        "wcos",
        inner,
        inner_labels,
        validation_cpu,
        validation_labels,
        n_classes,
        wcos_eps,
        query_chunk,
    )

    cos_predictions, cos_chunk = cosine_predictions(
        support,
        support_labels,
        test_cpu,
        n_classes,
        query_chunk,
    )
    means, _, lda_values, lda_vectors = pooled_statistics(support, support_labels, n_classes)
    lda_weights, lda_bias = lda_weights_from_stats(
        means,
        lda_values,
        lda_vectors,
        lda_selection["selected"],
    )
    lda_predictions, lda_chunk = chunked_linear_predictions(
        test_cpu,
        lda_weights,
        lda_bias,
        query_chunk=query_chunk,
    )
    wcos_mean, wcos_values, wcos_vectors = total_statistics(support)
    wcos_prototypes = wcos_weights_from_stats(
        support,
        support_labels,
        wcos_mean,
        wcos_values,
        wcos_vectors,
        wcos_selection["selected"],
        n_classes,
    )
    wcos_predictions, wcos_chunk = predict_wcos(
        test_cpu,
        wcos_prototypes,
        wcos_mean,
        wcos_values,
        wcos_vectors,
        wcos_selection["selected"],
        query_chunk,
    )
    diagnostics = router_diagnostics(support, support_labels, n_classes, shot)
    accuracies = {
        "cos": accuracy(cos_predictions, test_labels_cpu),
        "wcos_trainval": accuracy(wcos_predictions, test_labels_cpu),
        "lda_trainval": accuracy(lda_predictions, test_labels_cpu),
    }
    best_whitening = max(accuracies["wcos_trainval"], accuracies["lda_trainval"])
    record = {
        "shot": shot,
        "seed": int(seed),
        "n_train": int(len(support)),
        "n_test": int(len(test_cpu)),
        "n_classes": n_classes,
        "validation_split": split,
        "metrics": {
            **accuracies,
            "best_whitening": best_whitening,
            "best_whitening_minus_cos_pp": 100.0 * (best_whitening - accuracies["cos"]),
            "wcos_minus_cos_pp": 100.0 * (accuracies["wcos_trainval"] - accuracies["cos"]),
            "lda_minus_cos_pp": 100.0 * (accuracies["lda_trainval"] - accuracies["cos"]),
        },
        "selection": {"wcos": wcos_selection, "lda": lda_selection},
        "router_diagnostics": diagnostics,
        "chunking": {"cos": cos_chunk, "wcos": wcos_chunk, "lda": lda_chunk},
    }
    predictions = {
        "cos": cos_predictions,
        "wcos_trainval": wcos_predictions,
        "lda_trainval": lda_predictions,
    }
    return record, predictions


def mean_record(dataset: str, shot: str, records: list[dict]) -> dict:
    def average(path):
        values = []
        for record in records:
            value = record
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values))

    diagnostics = {
        key: average(("router_diagnostics", key))
        for key in records[0]["router_diagnostics"]
    }
    return {
        "dataset": dataset,
        "shot": shot,
        "n_seed_records": len(records),
        "seeds": [record["seed"] for record in records],
        "cos": average(("metrics", "cos")),
        "wcos_trainval": average(("metrics", "wcos_trainval")),
        "lda_trainval": average(("metrics", "lda_trainval")),
        "best_whitening": average(("metrics", "best_whitening")),
        "target_best_whitening_minus_cos_pp": average(
            ("metrics", "best_whitening_minus_cos_pp")
        ),
        **diagnostics,
    }


def run(args) -> None:
    dataset_manifest_path = Path(args.dataset_manifest)
    feature_root = Path(args.feature_root)
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"evaluation output root must be new: {output_root}")
    manifest = load_and_verify_manifest(dataset_manifest_path)
    train_cpu, train_labels_cpu, test_cpu, test_labels_cpu = load_feature_bundle(
        feature_root, manifest
    )
    if int(torch.unique(train_labels_cpu).numel()) != manifest["n_classes"]:
        raise ValueError("train feature class count differs from dataset manifest")
    if not torch.equal(torch.unique(train_labels_cpu), torch.unique(test_labels_cpu)):
        raise ValueError("train/test feature class identities differ")
    if args.shot == "full":
        seeds = [0]
    elif args.shot in NONFULL_SHOTS:
        seeds = list(NONFULL_SEEDS)
    else:
        raise ValueError(f"unsupported shot: {args.shot}")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the requested evaluator device")
        torch.backends.cuda.matmul.allow_tf32 = False
    if manifest["n_classes"] == 10_000 and args.query_chunk >= len(test_cpu):
        raise ValueError("iNaturalist evaluator must use strict query chunking")

    support_by_key = {
        (str(row["shot"]), int(row["seed"])): row for row in manifest["supports"]
    }
    output_root.mkdir(parents=True)
    records = []
    for seed in seeds:
        support = support_by_key[(args.shot, seed)]
        support_path = dataset_manifest_path.parent / support["path"]
        if sha256_file(support_path) != support["sha256"]:
            raise ValueError("support index hash mismatch before evaluation")
        indices = np.load(support_path, allow_pickle=False)
        print(
            f"[eval] dataset={manifest['dataset']} shot={args.shot} seed={seed} "
            f"classes={manifest['n_classes']} support={len(indices)} test={len(test_cpu)}",
            flush=True,
        )
        record, predictions = evaluate_seed(
            train_cpu,
            train_labels_cpu,
            test_cpu,
            test_labels_cpu,
            indices,
            shot=args.shot,
            seed=seed,
            device=args.device,
            query_chunk=args.query_chunk,
            lda_regs=list(args.lda_regs),
            wcos_eps=list(args.wcos_eps),
        )
        record["support"] = {
            "path": str(support_path),
            "sha256": support["sha256"],
            "n_indices": support["n_indices"],
        }
        record["predictions"] = {}
        for method, values in predictions.items():
            relative = Path("predictions") / f"seed_{seed}_{method}.npy"
            digest = write_npy_new(output_root / relative, values)
            record["predictions"][method] = {
                "path": relative.as_posix(),
                "sha256": digest,
                "n_rows": len(values),
            }
        records.append(record)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "kind": "sdg_caev_finegrained_dataset_shot_result",
        "dataset": manifest["dataset"],
        "shot": args.shot,
        "commit": args.commit,
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "dataset_identity_sha256": manifest["dataset_identity_sha256"],
        "feature_root": str(feature_root),
        "feature_manifest_sha256": sha256_file(feature_root / "feature_manifest.json"),
        "protocol": {
            "evaluator": "Cos/WCos/LDA",
            "regularization_selection": "frozen Step-B class-balanced train-only holdout",
            "lda_regs": list(args.lda_regs),
            "wcos_eps": list(args.wcos_eps),
            "default_no_validation": DEFAULT_REG,
            "numeric_dtype": "torch.float64",
            "tf32": False,
            "score_materialization": "query-chunked argmax only",
            "full_example_by_class_matrix_persisted": False,
        },
        "records": records,
        "group": mean_record(manifest["dataset"], args.shot, records),
    }
    write_json_new(output_root / "result.json", payload)
    write_json_new(
        output_root / "RESULT_READY.json",
        {
            "kind": "sdg_caev_finegrained_result_ready",
            "dataset": manifest["dataset"],
            "shot": args.shot,
            "result_sha256": sha256_file(output_root / "result.json"),
            "dataset_identity_sha256": manifest["dataset_identity_sha256"],
            "n_records": len(records),
        },
    )
    print(f"[result-ready] root={output_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shot", choices=[*NONFULL_SHOTS, "full"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-chunk", type=int, default=512)
    parser.add_argument("--lda-regs", nargs="+", type=float, default=LDA_REGS)
    parser.add_argument("--wcos-eps", nargs="+", type=float, default=WCOS_EPS)
    parser.add_argument("--commit", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
