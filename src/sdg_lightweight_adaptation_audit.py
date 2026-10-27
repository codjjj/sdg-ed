#!/usr/bin/env python3
"""Train-only low-rank representation-adaptation stress test for SDG.

Each dataset starts from the exact frozen DINOv2-B cache of the paper's core
suite.  A zero-initialized rank-r residual adapter and a temporary linear head
are trained only on support features.  The epoch budget is selected on an
inner support validation split and the adapter is then retrained on full
support.  The temporary head is discarded; cosine, WCos-TV, and LDA-TV are
evaluated on the same adapted representation.

This is cached-feature representation adaptation, not end-to-end encoder
fine-tuning.  The distinction is retained in every artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from sdg_backbone_bootstrap_ci import (
    DATASETS,
    DATASET_LABELS,
    dino_feature_dir,
    reference_classifier_matrix,
)
from sdg_paths import FEATURE_ROOT, OUTPUT_ROOT, RESULTS_ROOT
from sdg_audit_core import (
    REG_GRID,
    accuracy,
    environment_manifest,
    paired_bootstrap_difference,
    select_index_from_scores,
    select_then_refit_evaluator,
    write_json_new,
    write_text_new,
)
from sdg_backbone_trainval_audit import (
    file_identity,
    frozen_cosine_predictions,
    prediction_digest,
)
from sdg_stepb_cached_eval import train_val_split


METHODS = ("cos", "wcos_tv", "lda_tv", "validation_whitening")
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT / "lightweight_adaptation_9x5"


def dataset_for_rank(rank: int) -> str:
    if rank < 0 or rank >= len(DATASETS):
        raise ValueError(f"rank {rank} is outside the {len(DATASETS)}-dataset matrix")
    return DATASETS[rank]


def dataset_or_idle_for_rank(rank: int) -> str | None:
    if rank < 0:
        raise ValueError("matrix rank must be nonnegative")
    return DATASETS[rank] if rank < len(DATASETS) else None


class LowRankResidualAdapter(nn.Module):
    """Near-identity adapter with a zero-initialized residual branch."""

    def __init__(self, dimension: int, rank: int, residual_scale: float):
        super().__init__()
        self.norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        self.residual_scale = float(residual_scale)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        return self.residual_scale * self.up(F.gelu(self.down(self.norm(features))))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.residual(features)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def remap_labels(labels: np.ndarray, classes: list[int]) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(classes)}
    if set(labels.tolist()) - set(mapping):
        raise ValueError("labels contain a class outside the mapping")
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)


def _epoch_batches(size: int, batch_size: int, seed: int, epoch: int) -> list[np.ndarray]:
    generator = np.random.RandomState(seed + 104729 * epoch)
    order = generator.permutation(size)
    return [order[start : start + batch_size] for start in range(0, size, batch_size)]


@torch.no_grad()
def _head_accuracy(
    adapter: nn.Module,
    head: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    adapter.eval()
    head.eval()
    correct = 0
    total = 0
    for start in range(0, len(labels), batch_size):
        stop = min(len(labels), start + batch_size)
        batch = torch.from_numpy(np.asarray(features[start:stop], dtype=np.float32)).to(device)
        target = torch.from_numpy(np.asarray(labels[start:stop], dtype=np.int64)).to(device)
        prediction = head(adapter(batch)).argmax(dim=1)
        correct += int((prediction == target).sum().item())
        total += len(target)
    return correct / total


def select_training_epoch(
    inner_features: np.ndarray,
    inner_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    rank: int,
    residual_scale: float,
    residual_penalty: float,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: torch.device,
    train_adapter: bool,
) -> dict:
    """Select epoch using support validation only."""

    seed_everything(seed)
    dimension = int(inner_features.shape[1])
    classes = sorted(set(inner_labels.tolist()))
    train_targets = remap_labels(inner_labels, classes)
    validation_targets = remap_labels(validation_labels, classes)
    adapter = LowRankResidualAdapter(dimension, rank, residual_scale).to(device)
    if not train_adapter:
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
    head = nn.Linear(dimension, len(classes)).to(device)
    parameters = list(head.parameters())
    if train_adapter:
        parameters += [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    history = []
    best = None

    for epoch in range(1, max_epochs + 1):
        adapter.train(train_adapter)
        head.train()
        loss_sum = 0.0
        seen = 0
        for indices in _epoch_batches(len(inner_labels), batch_size, seed, epoch):
            batch = torch.from_numpy(np.asarray(inner_features[indices], dtype=np.float32)).to(device)
            target = torch.from_numpy(train_targets[indices]).to(device)
            adapted = adapter(batch)
            loss = F.cross_entropy(head(adapted), target)
            if train_adapter and residual_penalty > 0:
                residual = adapted - batch
                ratio = residual.square().sum(dim=1) / batch.square().sum(dim=1).clamp_min(1e-12)
                loss = loss + residual_penalty * ratio.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        validation_accuracy = _head_accuracy(
            adapter,
            head,
            validation_features,
            validation_targets,
            device,
            batch_size,
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "validation_head_accuracy": validation_accuracy,
        }
        history.append(row)
        if best is None or validation_accuracy > best["validation_head_accuracy"]:
            best = {
                "epoch": epoch,
                "validation_head_accuracy": validation_accuracy,
            }
    if best is None:
        raise AssertionError("epoch selection produced no checkpoint")
    return {
        "selected_epoch": int(best["epoch"]),
        "selected_validation_head_accuracy": float(best["validation_head_accuracy"]),
        "history": history,
        "selection_uses": "inner_support_validation_only",
    }


def train_full_support(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    rank: int,
    residual_scale: float,
    residual_penalty: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: torch.device,
    train_adapter: bool,
) -> tuple[LowRankResidualAdapter, nn.Linear, list[dict]]:
    """Retrain from the same initialization family on all support."""

    if epochs < 1:
        raise ValueError("full-support retraining requires at least one epoch")
    seed_everything(seed)
    dimension = int(features.shape[1])
    classes = sorted(set(labels.tolist()))
    targets = remap_labels(labels, classes)
    adapter = LowRankResidualAdapter(dimension, rank, residual_scale).to(device)
    if not train_adapter:
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
    head = nn.Linear(dimension, len(classes)).to(device)
    parameters = list(head.parameters())
    if train_adapter:
        parameters += [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        adapter.train(train_adapter)
        head.train()
        loss_sum = 0.0
        seen = 0
        for indices in _epoch_batches(len(labels), batch_size, seed, epoch):
            batch = torch.from_numpy(np.asarray(features[indices], dtype=np.float32)).to(device)
            target = torch.from_numpy(targets[indices]).to(device)
            adapted = adapter(batch)
            loss = F.cross_entropy(head(adapted), target)
            if train_adapter and residual_penalty > 0:
                residual = adapted - batch
                ratio = residual.square().sum(dim=1) / batch.square().sum(dim=1).clamp_min(1e-12)
                loss = loss + residual_penalty * ratio.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        history.append({"epoch": epoch, "train_loss": loss_sum / seen})
    return adapter, head, history


@torch.no_grad()
def transform_features(
    adapter: nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    adapter.eval()
    outputs = []
    numerator = 0.0
    denominator = 0.0
    max_ratio = 0.0
    for start in range(0, len(features), batch_size):
        stop = min(len(features), start + batch_size)
        batch = torch.from_numpy(np.asarray(features[start:stop], dtype=np.float32)).to(device)
        adapted = adapter(batch)
        residual = adapted - batch
        per_row = residual.norm(dim=1) / batch.norm(dim=1).clamp_min(1e-12)
        numerator += float(residual.square().sum().cpu())
        denominator += float(batch.square().sum().cpu())
        max_ratio = max(max_ratio, float(per_row.max().cpu()))
        outputs.append(adapted.cpu().numpy())
    transformed = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    if transformed.shape != features.shape or not np.isfinite(transformed).all():
        raise ValueError("adapted feature output is invalid")
    return transformed, {
        "global_l2_ratio": math.sqrt(numerator / max(denominator, 1e-30)),
        "max_row_l2_ratio": max_ratio,
    }


def evaluate_geometry(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    validation_seed: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    classes = sorted(set(train_labels.tolist()) & set(test_labels.tolist()))
    inner, inner_labels, validation, validation_labels, split = train_val_split(
        np.asarray(train_features, dtype=np.float64),
        train_labels,
        seed=validation_seed,
    )
    if validation is None:
        raise ValueError("full-shot adaptation audit requires support validation")
    split = {**split, "seed": int(validation_seed)}
    train64 = np.asarray(train_features, dtype=np.float64)
    test64 = np.asarray(test_features, dtype=np.float64)
    cosine = frozen_cosine_predictions(train_features, train_labels, test_features, classes)
    cosine_validation = accuracy(
        frozen_cosine_predictions(inner, inner_labels, validation, classes),
        validation_labels,
    )
    wcos = select_then_refit_evaluator(
        "wcos",
        train64,
        train_labels,
        inner,
        inner_labels,
        validation,
        validation_labels,
        test64,
        classes,
        REG_GRID,
        regularization_mode="absolute",
    )
    lda = select_then_refit_evaluator(
        "lda",
        train64,
        train_labels,
        inner,
        inner_labels,
        validation,
        validation_labels,
        test64,
        classes,
        REG_GRID,
        regularization_mode="absolute",
    )
    wcos_validation = max(wcos["validation_scores"])
    lda_validation = max(lda["validation_scores"])
    if wcos_validation >= lda_validation:
        selected_family = "wcos"
        selected_predictions = wcos["predictions"]
    else:
        selected_family = "lda"
        selected_predictions = lda["predictions"]
    predictions = {
        "cos": cosine,
        "wcos_tv": wcos["predictions"],
        "lda_tv": lda["predictions"],
        "validation_whitening": selected_predictions,
    }
    result = {
        "validation_split": split,
        "cosine_validation_accuracy": cosine_validation,
        "validation_whitening_family": selected_family,
        "selection": {
            "wcos_tv": {key: value for key, value in wcos.items() if key != "predictions"},
            "lda_tv": {key: value for key, value in lda.items() if key != "predictions"},
        },
        "accuracies": {method: accuracy(prediction, test_labels) for method, prediction in predictions.items()},
        "gaps_pp": {
            method: 100.0
            * (accuracy(predictions[method], test_labels) - accuracy(cosine, test_labels))
            for method in METHODS
            if method != "cos"
        },
        "prediction_sha256": {
            method: prediction_digest(prediction) for method, prediction in predictions.items()
        },
    }
    return result, predictions


def resolve_evaluator_validation_seeds(
    adapter_seeds: list[int],
    *,
    evaluator_validation_seed: int,
    evaluator_validation_seeds: list[int] | None,
) -> tuple[list[int], bool]:
    """Resolve a one-to-one evaluator-selection schedule for adapter seeds."""

    if evaluator_validation_seeds is None:
        return [int(evaluator_validation_seed)] * len(adapter_seeds), False
    resolved = [int(seed) for seed in evaluator_validation_seeds]
    if len(resolved) != len(adapter_seeds):
        raise ValueError(
            "--evaluator-validation-seeds must have exactly one value per adapter seed"
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            "--evaluator-validation-seeds must be unique for a genuine multi-seed audit"
        )
    return resolved, True


def evaluate_matched_representation_pair(
    frozen_train_features: np.ndarray,
    train_labels: np.ndarray,
    frozen_test_features: np.ndarray,
    test_labels: np.ndarray,
    adapted_train_features: np.ndarray,
    adapted_test_features: np.ndarray,
    *,
    validation_seed: int,
) -> tuple[dict, dict[str, np.ndarray], dict, dict[str, np.ndarray]]:
    """Evaluate frozen and adapted features under one identical selection split."""

    frozen_result, frozen_predictions = evaluate_geometry(
        frozen_train_features,
        train_labels,
        frozen_test_features,
        test_labels,
        validation_seed=validation_seed,
    )
    adapted_result, adapted_predictions = evaluate_geometry(
        adapted_train_features,
        train_labels,
        adapted_test_features,
        test_labels,
        validation_seed=validation_seed,
    )
    frozen_split = frozen_result["validation_split"]
    adapted_split = adapted_result["validation_split"]
    if frozen_split != adapted_split:
        raise AssertionError("frozen and adapted evaluator-selection splits are not matched")
    return frozen_result, frozen_predictions, adapted_result, adapted_predictions


def paired_gap_change_bootstrap(
    adapted_whitening: np.ndarray,
    adapted_cosine: np.ndarray,
    frozen_whitening: np.ndarray,
    frozen_cosine: np.ndarray,
    labels: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    arrays = [adapted_whitening, adapted_cosine, frozen_whitening, frozen_cosine]
    if any(np.asarray(value).shape != labels.shape for value in arrays):
        raise ValueError("gap-change bootstrap shape mismatch")
    correctness = [(np.asarray(value) == labels).astype(np.float64) for value in arrays]
    per_example = 100.0 * (
        correctness[0] - correctness[1] - correctness[2] + correctness[3]
    )
    rng = np.random.RandomState(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sampled = rng.randint(0, len(labels), size=len(labels))
        samples[index] = float(per_example[sampled].mean())
    return {
        "point_pp": float(per_example.mean()),
        "ci95_pp": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "n_test": int(len(labels)),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def seed_summary(values: list[float], *, seed: int, n_boot: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        raise ValueError("seed summary requires at least two values")
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    critical = float(stats.t.ppf(0.975, df=len(array) - 1))
    rng = np.random.RandomState(seed)
    bootstrap = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        bootstrap[index] = float(array[rng.randint(0, len(array), size=len(array))].mean())
    return {
        "n_seeds": len(array),
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
    }


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


def _load_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = payload["pooled"].float().numpy()
    labels = payload["labels"].long().numpy()
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError(f"invalid cache at {path}")
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite cache at {path}")
    return features, labels


def dataset_report(payload: dict) -> str:
    lines = [
        f"# {payload['dataset_label']} lightweight representation-adaptation audit",
        "",
        "> Cached-feature adapter evidence; this is not end-to-end encoder fine-tuning.",
        "",
        "## Matched evaluator-selection seeds",
        "",
        "| adapter seed | evaluator split seed | frozen family | adapted family | frozen gap | adapted gap | paired change |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["seeds"]:
        lines.append(
            f"| {row['seed']} | {row['evaluator_validation_seed']} | "
            f"{row['frozen_evaluator']['validation_whitening_family']} | "
            f"{row['adapted_evaluator']['validation_whitening_family']} | "
            f"{row['frozen_evaluator']['gaps_pp']['validation_whitening']:+.2f} | "
            f"{row['adapted_evaluator']['gaps_pp']['validation_whitening']:+.2f} | "
            f"{row['gap_change_bootstrap']['validation_whitening']['point_pp']:+.2f} |"
        )
    summary = payload["seed_summary"]["validation_whitening"]
    lines.extend(
        [
            "",
            f"Frozen WVal−Cos mean gap: {summary['frozen_gap_pp']['mean_pp']:+.3f} pp "
            f"(t-CI {summary['frozen_gap_pp']['t_ci95_pp']}).",
            f"Adapted WVal−Cos mean gap: {summary['adapted_gap_pp']['mean_pp']:+.3f} pp "
            f"(t-CI {summary['adapted_gap_pp']['t_ci95_pp']}).",
            f"Matched change from frozen gap: {summary['gap_change_pp']['mean_pp']:+.3f} pp "
            f"(t-CI {summary['gap_change_pp']['t_ci95_pp']}).",
            "",
        ]
    )
    return "\n".join(lines)


def run_dataset(args: argparse.Namespace) -> dict:
    output_root = Path(args.output_root)
    dataset_root = output_root / args.dataset
    result_path = dataset_root / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    feature_root = dino_feature_dir(Path(args.feature_root), args.dataset)
    train_path = feature_root / "train.pt"
    test_path = feature_root / "test.pt"
    train_features, train_labels = _load_cache(train_path)
    test_features, test_labels = _load_cache(test_path)
    classes = sorted(set(train_labels.tolist()) & set(test_labels.tolist()))
    if len(classes) < 2:
        raise ValueError("adaptation dataset has fewer than two classes")

    evaluator_validation_seeds, matched_evaluator_multiseed = (
        resolve_evaluator_validation_seeds(
            args.seeds,
            evaluator_validation_seed=args.evaluator_validation_seed,
            evaluator_validation_seeds=args.evaluator_validation_seeds,
        )
    )
    reference = reference_classifier_matrix(Path(args.results_root), "dinov2", args.dataset)
    expected_cosine = float(reference["cosine"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inner, inner_labels, validation, validation_labels, adapter_split = train_val_split(
        train_features,
        train_labels,
        seed=args.adapter_validation_seed,
    )
    if validation is None:
        raise ValueError("adapter requires a support validation split")
    prediction_arrays = {"labels": test_labels.astype(np.int16)}
    seed_rows = []
    cosine_errors = []
    for seed, evaluator_validation_seed in zip(
        args.seeds, evaluator_validation_seeds, strict=True
    ):
        run_seed = args.base_seed + 1000003 * seed
        adapter_selection = select_training_epoch(
            inner,
            inner_labels,
            validation,
            validation_labels,
            rank=args.rank,
            residual_scale=args.residual_scale,
            residual_penalty=args.residual_penalty,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed=run_seed,
            device=device,
            train_adapter=True,
        )
        adapter, adapter_head, adapter_history = train_full_support(
            train_features,
            train_labels,
            rank=args.rank,
            residual_scale=args.residual_scale,
            residual_penalty=args.residual_penalty,
            epochs=adapter_selection["selected_epoch"],
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed=run_seed,
            device=device,
            train_adapter=True,
        )
        adapted_train, train_change = transform_features(
            adapter, train_features, device=device, batch_size=args.eval_batch_size
        )
        adapted_test, test_change = transform_features(
            adapter, test_features, device=device, batch_size=args.eval_batch_size
        )
        test_targets = remap_labels(test_labels, classes)
        adapter_head_accuracy = _head_accuracy(
            adapter,
            adapter_head,
            test_features,
            test_targets,
            device,
            args.eval_batch_size,
        )

        head_only_selection = select_training_epoch(
            inner,
            inner_labels,
            validation,
            validation_labels,
            rank=args.rank,
            residual_scale=args.residual_scale,
            residual_penalty=0.0,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed=run_seed,
            device=device,
            train_adapter=False,
        )
        identity, head_only, head_only_history = train_full_support(
            train_features,
            train_labels,
            rank=args.rank,
            residual_scale=args.residual_scale,
            residual_penalty=0.0,
            epochs=head_only_selection["selected_epoch"],
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed=run_seed,
            device=device,
            train_adapter=False,
        )
        head_only_accuracy = _head_accuracy(
            identity,
            head_only,
            test_features,
            test_targets,
            device,
            args.eval_batch_size,
        )

        (
            frozen_result,
            frozen_predictions,
            adapted_result,
            adapted_predictions,
        ) = evaluate_matched_representation_pair(
            train_features,
            train_labels,
            test_features,
            test_labels,
            adapted_train,
            adapted_test,
            validation_seed=evaluator_validation_seed,
        )
        cosine_error = abs(frozen_result["accuracies"]["cos"] - expected_cosine)
        cosine_errors.append(cosine_error)
        if cosine_error > args.point_tolerance:
            raise AssertionError(
                "frozen cosine alignment failed: "
                f"{frozen_result['accuracies']['cos']} vs {expected_cosine}"
            )
        paired = {}
        gap_change = {}
        for method in METHODS:
            if method == "cos":
                continue
            paired[method] = paired_bootstrap_difference(
                adapted_predictions[method],
                adapted_predictions["cos"],
                test_labels,
                n_boot=args.test_bootstrap,
                seed=args.bootstrap_seed + seed * 1009 + METHODS.index(method) * 9176,
            )
            gap_change[method] = paired_gap_change_bootstrap(
                adapted_predictions[method],
                adapted_predictions["cos"],
                frozen_predictions[method],
                frozen_predictions["cos"],
                test_labels,
                n_boot=args.test_bootstrap,
                seed=args.bootstrap_seed + seed * 2029 + METHODS.index(method) * 7919,
            )
        for method, prediction in frozen_predictions.items():
            prediction_arrays[f"seed{seed}__frozen__{method}"] = np.asarray(
                prediction, dtype=np.int16
            )
        for method, prediction in adapted_predictions.items():
            prediction_arrays[f"seed{seed}__adapted__{method}"] = np.asarray(
                prediction, dtype=np.int16
            )
        seed_rows.append(
            {
                "seed": int(seed),
                "effective_seed": int(run_seed),
                "evaluator_validation_seed": int(evaluator_validation_seed),
                "adapter_training": {
                    **adapter_selection,
                    "full_support_history": adapter_history,
                    "test_head_accuracy": adapter_head_accuracy,
                    "test_head_labels_used_for_selection": False,
                },
                "head_only_control": {
                    **head_only_selection,
                    "full_support_history": head_only_history,
                    "test_head_accuracy": head_only_accuracy,
                    "test_head_labels_used_for_selection": False,
                },
                "representation_change": {"train": train_change, "test": test_change},
                "frozen_evaluator": frozen_result,
                "adapted_evaluator": adapted_result,
                "paired_test_bootstrap": paired,
                "gap_change_bootstrap": gap_change,
            }
        )
        del adapter, adapter_head, identity, head_only
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summaries = {}
    for method in METHODS:
        if method == "cos":
            continue
        adapted_gaps = [
            row["adapted_evaluator"]["gaps_pp"][method] for row in seed_rows
        ]
        frozen_gaps = [
            row["frozen_evaluator"]["gaps_pp"][method] for row in seed_rows
        ]
        gap_changes = [
            adapted - frozen
            for adapted, frozen in zip(adapted_gaps, frozen_gaps, strict=True)
        ]
        summaries[method] = {
            "frozen_gap_pp": seed_summary(
                frozen_gaps,
                seed=args.bootstrap_seed + 20000 + METHODS.index(method),
                n_boot=args.seed_bootstrap,
            ),
            "adapted_gap_pp": seed_summary(
                adapted_gaps,
                seed=args.bootstrap_seed + METHODS.index(method),
                n_boot=args.seed_bootstrap,
            ),
            "gap_change_pp": seed_summary(
                gap_changes,
                seed=args.bootstrap_seed + 10000 + METHODS.index(method),
                n_boot=args.seed_bootstrap,
            ),
        }

    predictions_path = dataset_root / "predictions.npz"
    _write_npz_new(predictions_path, prediction_arrays)
    payload = {
        "schema_version": 2,
        "kind": "post_submission_lightweight_representation_adaptation_audit",
        "dataset": args.dataset,
        "dataset_label": DATASET_LABELS[args.dataset],
        "scope": (
            "Zero-initialized low-rank residual adapter on cached frozen features; "
            "not end-to-end encoder fine-tuning"
        ),
        "protocol": {
            "backbone": "DINOv2-B",
            "rank": args.rank,
            "residual_scale": args.residual_scale,
            "residual_penalty": args.residual_penalty,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "seeds": args.seeds,
            "adapter_validation_seed": args.adapter_validation_seed,
            "evaluator_validation_seeds": evaluator_validation_seeds,
            "matched_evaluator_multiseed": matched_evaluator_multiseed,
            "regularization_grid": REG_GRID,
            "test_bootstrap": args.test_bootstrap,
            "seed_bootstrap": args.seed_bootstrap,
        },
        "data": {
            "n_train": int(len(train_labels)),
            "n_test": int(len(test_labels)),
            "n_classes": len(classes),
            "feature_dim": int(train_features.shape[1]),
            "files": [file_identity(train_path), file_identity(test_path)],
            "frozen_reference": file_identity(reference_path),
            "frozen_cosine_max_abs_error": max(cosine_errors),
        },
        "selection_guards": {
            "adapter_epoch_selection": "inner support validation only",
            "adapter_retrain": "full support for selected epoch count",
            "evaluator_selection": "support inner validation only",
            "frozen_adapted_pairing": (
                "same evaluator validation split within each adapter seed"
            ),
            "test_labels": "metrics and paired uncertainty only",
            "adapter_split": adapter_split,
        },
        "frozen_baseline": seed_rows[0]["frozen_evaluator"],
        "seeds": seed_rows,
        "seed_summary": summaries,
        "artifacts": {
            "predictions.npz": {
                "bytes": predictions_path.stat().st_size,
                "sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
                "arrays": len(prediction_arrays),
            }
        },
        "environment": environment_manifest(args.commit),
    }
    write_json_new(result_path, payload)
    write_text_new(dataset_root / "report.md", dataset_report(payload))
    print(f"[lightweight-adaptation] complete dataset={args.dataset} output={dataset_root}", flush=True)
    return payload


def aggregate_report(payload: dict) -> str:
    lines = [
        "# 9-dataset lightweight representation-adaptation audit",
        "",
        "> Cached-feature adapter evidence; not end-to-end encoder fine-tuning.",
        "",
        "| dataset | frozen WVal−Cos | frozen t-CI | adapted WVal−Cos | adapted t-CI | paired change | change t-CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["datasets"]:
        row = result["seed_summary"]["validation_whitening"]
        lines.append(
            f"| {result['dataset_label']} | {row['frozen_gap_pp']['mean_pp']:+.2f} | "
            f"[{row['frozen_gap_pp']['t_ci95_pp'][0]:+.2f},"
            f"{row['frozen_gap_pp']['t_ci95_pp'][1]:+.2f}] | "
            f"{row['adapted_gap_pp']['mean_pp']:+.2f} | "
            f"[{row['adapted_gap_pp']['t_ci95_pp'][0]:+.2f},"
            f"{row['adapted_gap_pp']['t_ci95_pp'][1]:+.2f}] | "
            f"{row['gap_change_pp']['mean_pp']:+.2f} | "
            f"[{row['gap_change_pp']['t_ci95_pp'][0]:+.2f},"
            f"{row['gap_change_pp']['t_ci95_pp'][1]:+.2f}] |"
        )
    lines.extend(
        [
            "",
            f"Dataset-cluster mean frozen gap: {payload['dataset_cluster_summary']['frozen_gap_pp']['mean_pp']:+.3f} pp.",
            f"Dataset-cluster mean adapted gap: {payload['dataset_cluster_summary']['adapted_gap_pp']['mean_pp']:+.3f} pp.",
            f"Dataset-cluster mean gap change: {payload['dataset_cluster_summary']['gap_change_pp']['mean_pp']:+.3f} pp.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(args: argparse.Namespace) -> dict:
    output_root = Path(args.output_root)
    result_path = output_root / "lightweight_adaptation_audit.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    results = []
    for dataset in DATASETS:
        path = output_root / dataset / "result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("schema_version") != 2:
            raise ValueError(f"matched multi-seed aggregate requires schema 2: {path}")
        results.append(result)
    frozen_dataset_means = [
        result["seed_summary"]["validation_whitening"]["frozen_gap_pp"]["mean_pp"]
        for result in results
    ]
    adapted_dataset_means = [
        result["seed_summary"]["validation_whitening"]["adapted_gap_pp"]["mean_pp"]
        for result in results
    ]
    change_dataset_means = [
        result["seed_summary"]["validation_whitening"]["gap_change_pp"]["mean_pp"]
        for result in results
    ]
    payload = {
        "schema_version": 2,
        "kind": "post_submission_lightweight_representation_adaptation_9dataset_aggregate",
        "scope": "cached-feature low-rank adapter, not end-to-end encoder fine-tuning",
        "datasets": results,
        "dataset_cluster_summary": {
            "frozen_gap_pp": seed_summary(
                frozen_dataset_means,
                seed=args.bootstrap_seed - 1,
                n_boot=args.seed_bootstrap,
            ),
            "adapted_gap_pp": seed_summary(
                adapted_dataset_means,
                seed=args.bootstrap_seed,
                n_boot=args.seed_bootstrap,
            ),
            "gap_change_pp": seed_summary(
                change_dataset_means,
                seed=args.bootstrap_seed + 1,
                n_boot=args.seed_bootstrap,
            ),
            "unit": "dataset mean over five matched evaluator-selection seeds",
        },
        "environment": environment_manifest(args.commit),
    }
    write_json_new(result_path, payload)
    write_text_new(output_root / "lightweight_adaptation_audit.md", aggregate_report(payload))
    print(f"[lightweight-adaptation] aggregate complete output={output_root}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", default=str(FEATURE_ROOT))
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--dataset-from-rank", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--residual-penalty", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--base-seed", type=int, default=20260724)
    parser.add_argument("--adapter-validation-seed", type=int, default=17)
    parser.add_argument("--evaluator-validation-seed", type=int, default=0)
    parser.add_argument("--evaluator-validation-seeds", type=int, nargs="+")
    parser.add_argument("--test-bootstrap", type=int, default=2000)
    parser.add_argument("--seed-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--point-tolerance", type=float, default=5e-6)
    parser.add_argument("--commit", default=None)
    args = parser.parse_args()
    mode_count = sum(
        [bool(args.aggregate_only), args.dataset is not None, bool(args.dataset_from_rank)]
    )
    if mode_count != 1:
        parser.error(
            "provide exactly one of --dataset, --dataset-from-rank, or --aggregate-only"
        )
    if args.dataset_from_rank:
        try:
            rank = int(os.environ.get("RANK", "0"))
            args.dataset = dataset_or_idle_for_rank(rank)
            args.idle_rank = args.dataset is None
        except ValueError as error:
            parser.error(str(error))
    if args.rank < 1 or args.max_epochs < 1 or args.batch_size < 1:
        parser.error("rank, max epochs, and batch size must be positive")
    if len(args.seeds) < 5 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain at least five unique values")
    try:
        resolve_evaluator_validation_seeds(
            args.seeds,
            evaluator_validation_seed=args.evaluator_validation_seed,
            evaluator_validation_seeds=args.evaluator_validation_seeds,
        )
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if getattr(parsed, "idle_rank", False):
        print(f"[lightweight-adaptation] idle matrix rank={os.environ.get('RANK')}", flush=True)
    elif parsed.aggregate_only:
        aggregate(parsed)
    else:
        run_dataset(parsed)
