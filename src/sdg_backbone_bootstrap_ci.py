#!/usr/bin/env python3
"""Paired test-example bootstrap for the frozen SDG backbone diagnostic.

The reporting pass reuses cached features and the fixed evaluator of the
paper's six-backbone sweep (Appendix E): raw cosine NCM versus pooled-covariance
Mahalanobis with ridge 1e-4.  Within each dataset, every evaluator/backbone
receives the same bootstrap test indices so model-selection uncertainty remains
paired.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sdg_paths import FEATURE_ROOT, OUTPUT_ROOT, RESULTS_ROOT, backbone_feature_dir, core_feature_dir


DATASETS = [
    "aircraft",
    "cub200",
    "cars",
    "food101",
    "pets",
    "cifar100",
    "cifar10",
    "imagenet100",
    "tiny_imagenet",
]
DATASET_LABELS = {
    "aircraft": "Aircraft",
    "cub200": "CUB-200",
    "cars": "Cars",
    "food101": "Food-101",
    "pets": "Pets",
    "cifar100": "CIFAR-100",
    "cifar10": "CIFAR-10",
    "imagenet100": "ImageNet-100",
    "tiny_imagenet": "Tiny-ImageNet",
}
BACKBONES = {
    "dinov2": "DINOv2-B",
    "clip_vit_b32": "CLIP-B/32",
    "mae_vit_base": "MAE ViT-B",
    "siglip_b16": "SigLIP-B/16",
    "convnext_base": "ConvNeXt-B",
    "vit_sup_b16": "Sup. ViT-B/16",
}


def write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(suffix=path.suffix))
    tmp.write_text(content)
    shutil.copy2(tmp, path)
    tmp.unlink(missing_ok=True)


def write_json_new(path: Path, payload: dict) -> None:
    write_new(path, json.dumps(payload, indent=2, default=float) + "\n")


def load_cache(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["pooled"].float(), payload["labels"].long().numpy()


def dino_feature_dir(feature_root: Path, dataset: str) -> Path:
    """DINOv2-B core cache: <feature_root>/<dataset>/{train,test}.pt."""

    return core_feature_dir(dataset, feature_root)


def feature_dir(feature_root: Path, backbone: str, dataset: str) -> Path:
    """Backbone cache: <feature_root>/backbones/<backbone>/<dataset>/{train,test}.pt."""

    if backbone == "dinov2":
        return dino_feature_dir(feature_root, dataset)
    return backbone_feature_dir(backbone, dataset, feature_root)


def reference_classifier_matrix(results_root: Path, backbone: str, dataset: str) -> dict:
    """Released full-shot cosine/Mahalanobis accuracies used as point checks.

    DINOv2-B values come from the released backbone comparison; the other
    backbones have one released result.json per dataset.
    """

    results_root = Path(results_root)
    if backbone == "dinov2":
        path = results_root / "backbone/clip_vit_b32/backbone_comparison.json"
        rows = {row["dataset"]: row for row in json.loads(path.read_text())}
        return rows[dataset]["dino"]
    path = results_root / "backbone" / backbone / dataset / "result.json"
    return json.loads(path.read_text())["classifier_matrix"]


def classes_from(train_labels: np.ndarray, test_labels: np.ndarray) -> list[int]:
    return sorted(set(train_labels.tolist()) & set(test_labels.tolist()))


def predict_cosine(
    train_features: torch.Tensor,
    train_labels: np.ndarray,
    test_features: torch.Tensor,
    classes: list[int],
) -> np.ndarray:
    labels = torch.from_numpy(train_labels)
    prototypes = torch.stack([train_features[labels == cls].mean(0) for cls in classes])
    prototypes = F.normalize(prototypes, dim=1)
    normalized_test = F.normalize(test_features, dim=1)
    indices = (normalized_test @ prototypes.T).argmax(1).numpy()
    return np.asarray([classes[int(index)] for index in indices])


def predict_mahalanobis(
    train_features: torch.Tensor,
    train_labels: np.ndarray,
    test_features: torch.Tensor,
    classes: list[int],
    ridge: float,
) -> np.ndarray:
    train = train_features.numpy().astype(np.float64)
    test = test_features.numpy().astype(np.float64)
    dim = train.shape[1]
    scatter = np.zeros((dim, dim), dtype=np.float64)
    prototypes = []
    n_total = 0
    for cls in classes:
        class_features = train[train_labels == cls]
        mean = class_features.mean(0)
        prototypes.append(mean)
        centered = class_features - mean
        # NumPy/Accelerate on macOS can emit spurious matmul floating warnings
        # for this finite calculation; explicit finiteness gates below retain
        # the semantic failure check without polluting task logs.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scatter += centered.T @ centered
        n_total += len(class_features)
    if not np.isfinite(scatter).all():
        raise FloatingPointError("non-finite pooled within-class scatter")
    covariance = scatter / max(1, n_total - len(classes))
    inverse = np.linalg.inv(covariance + ridge * np.eye(dim))
    prototypes = np.stack(prototypes)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        weights = prototypes @ inverse
    bias = -0.5 * np.sum(prototypes * weights, axis=1)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = test @ weights.T + bias[None, :]
    if not np.isfinite(weights).all() or not np.isfinite(scores).all():
        raise FloatingPointError("non-finite Mahalanobis weights or scores")
    indices = np.argmax(scores, axis=1)
    return np.asarray([classes[int(index)] for index in indices])


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def paired_bootstrap(
    cosine_correct: np.ndarray,
    diagnostic_correct: np.ndarray,
    backbone_names: list[str],
    n_boot: int,
    seed: int,
) -> tuple[list[dict], dict, dict]:
    if cosine_correct.shape != diagnostic_correct.shape:
        raise ValueError("cosine and diagnostic correctness matrices must have identical shape")
    if cosine_correct.ndim != 2 or cosine_correct.shape[0] != len(backbone_names):
        raise ValueError("correctness matrices must be backbone x test-example")
    n_backbones, n_test = cosine_correct.shape
    rng = np.random.RandomState(seed)
    cosine_boot = np.empty((n_backbones, n_boot), dtype=np.float64)
    diagnostic_boot = np.empty((n_backbones, n_boot), dtype=np.float64)
    gap_boot = np.empty((n_backbones, n_boot), dtype=np.float64)
    regret_boot = np.empty(n_boot, dtype=np.float64)
    flip_boot = np.empty(n_boot, dtype=np.bool_)
    cosine_winner_boot = np.empty(n_boot, dtype=np.int64)
    diagnostic_winner_boot = np.empty(n_boot, dtype=np.int64)

    for index in range(n_boot):
        sampled = rng.randint(0, n_test, size=n_test)
        cosine_acc = cosine_correct[:, sampled].mean(1)
        diagnostic_acc = diagnostic_correct[:, sampled].mean(1)
        cosine_boot[:, index] = cosine_acc
        diagnostic_boot[:, index] = diagnostic_acc
        gap_boot[:, index] = 100.0 * (diagnostic_acc - cosine_acc)
        cosine_winner = int(np.argmax(cosine_acc))
        diagnostic_winner = int(np.argmax(diagnostic_acc))
        cosine_winner_boot[index] = cosine_winner
        diagnostic_winner_boot[index] = diagnostic_winner
        flip_boot[index] = cosine_winner != diagnostic_winner
        regret_boot[index] = 100.0 * (
            diagnostic_acc[diagnostic_winner] - diagnostic_acc[cosine_winner]
        )

    cosine_point = cosine_correct.mean(1)
    diagnostic_point = diagnostic_correct.mean(1)
    cosine_winner = int(np.argmax(cosine_point))
    diagnostic_winner = int(np.argmax(diagnostic_point))
    cells = []
    for index, backbone in enumerate(backbone_names):
        cells.append(
            {
                "backbone": backbone,
                "backbone_label": BACKBONES[backbone],
                "cosine_acc": float(cosine_point[index]),
                "cosine_ci95": percentile_ci(cosine_boot[index]),
                "diagnostic_acc": float(diagnostic_point[index]),
                "diagnostic_ci95": percentile_ci(diagnostic_boot[index]),
                "diagnostic_minus_cosine_pp": float(
                    100.0 * (diagnostic_point[index] - cosine_point[index])
                ),
                "gap_ci95_pp": percentile_ci(gap_boot[index]),
            }
        )

    ranking = {
        "cosine_winner": backbone_names[cosine_winner],
        "cosine_winner_label": BACKBONES[backbone_names[cosine_winner]],
        "diagnostic_winner": backbone_names[diagnostic_winner],
        "diagnostic_winner_label": BACKBONES[backbone_names[diagnostic_winner]],
        "observed_flip": bool(cosine_winner != diagnostic_winner),
        "observed_regret_pp": float(
            100.0 * (diagnostic_point[diagnostic_winner] - diagnostic_point[cosine_winner])
        ),
        "bootstrap_flip_rate": float(flip_boot.mean()),
        "bootstrap_regret_mean_pp": float(regret_boot.mean()),
        "bootstrap_regret_ci95_pp": percentile_ci(regret_boot),
        "bootstrap_positive_regret_rate": float(np.mean(regret_boot > 0.0)),
        "cosine_winner_stability": float(np.mean(cosine_winner_boot == cosine_winner)),
        "diagnostic_winner_stability": float(
            np.mean(diagnostic_winner_boot == diagnostic_winner)
        ),
    }
    bootstrap_samples = {
        "regret_pp": regret_boot.tolist(),
        "flip": flip_boot.tolist(),
    }
    return cells, ranking, bootstrap_samples


def run_dataset(
    feature_root: Path,
    output_root: Path,
    dataset: str,
    backbone_names: list[str],
    ridge: float,
    n_boot: int,
    seed: int,
    point_tolerance: float,
    results_root: Path = RESULTS_ROOT,
) -> dict:
    reference_test_labels = None
    cosine_correct = []
    diagnostic_correct = []
    point_checks = []
    n_train = {}

    for backbone in backbone_names:
        print(f"[backbone-ci] dataset={dataset} backbone={backbone}", flush=True)
        directory = feature_dir(feature_root, backbone, dataset)
        train_features, train_labels = load_cache(directory / "train.pt")
        test_features, test_labels = load_cache(directory / "test.pt")
        if reference_test_labels is None:
            reference_test_labels = test_labels
        elif not np.array_equal(reference_test_labels, test_labels):
            raise ValueError(f"test-label ordering mismatch for {dataset}/{backbone}")
        classes = classes_from(train_labels, test_labels)
        cosine_prediction = predict_cosine(
            train_features, train_labels, test_features, classes
        )
        diagnostic_prediction = predict_mahalanobis(
            train_features, train_labels, test_features, classes, ridge
        )
        cosine_acc = float(np.mean(cosine_prediction == test_labels))
        diagnostic_acc = float(np.mean(diagnostic_prediction == test_labels))

        reference = reference_classifier_matrix(results_root, backbone, dataset)
        cosine_error = abs(cosine_acc - float(reference["cosine"]))
        diagnostic_error = abs(diagnostic_acc - float(reference["mahalanobis"]))
        if cosine_error > point_tolerance or diagnostic_error > point_tolerance:
            raise AssertionError(
                f"point estimate mismatch for {dataset}/{backbone}: "
                f"cos={cosine_error:.3g}, diagnostic={diagnostic_error:.3g}"
            )
        point_checks.append(
            {
                "backbone": backbone,
                "cosine_abs_error": cosine_error,
                "diagnostic_abs_error": diagnostic_error,
            }
        )
        cosine_correct.append(cosine_prediction == test_labels)
        diagnostic_correct.append(diagnostic_prediction == test_labels)
        n_train[backbone] = int(len(train_labels))

    cosine_correct = np.stack(cosine_correct)
    diagnostic_correct = np.stack(diagnostic_correct)
    cells, ranking, bootstrap_samples = paired_bootstrap(
        cosine_correct,
        diagnostic_correct,
        backbone_names,
        n_boot=n_boot,
        seed=seed,
    )
    if min(bootstrap_samples["regret_pp"]) < -1e-12:
        raise AssertionError("bootstrap selection regret must be nonnegative")
    payload = {
        "protocol": {
            "evaluation": "raw cosine vs fixed pooled-covariance Mahalanobis",
            "ridge": ridge,
            "bootstrap_unit": "shared test-example index across all backbones/evaluators",
            "n_boot": n_boot,
            "seed": seed,
            "point_tolerance": point_tolerance,
        },
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "n_train_by_backbone": n_train,
        "n_test": int(cosine_correct.shape[1]),
        "point_estimate_checks": point_checks,
        "cells": cells,
        "ranking": ranking,
        "bootstrap_samples": bootstrap_samples,
    }
    write_json_new(output_root / dataset / "result.json", payload)
    write_new(output_root / dataset / "result.md", dataset_report(payload))
    return payload


def dataset_report(payload: dict) -> str:
    lines = [f"# {payload['dataset_label']} cross-backbone paired bootstrap\n"]
    lines.append("| Backbone | Cos [95% CI] | Diagnostic [95% CI] | Gap pp [95% CI] |")
    lines.append("|---|---:|---:|---:|")
    for cell in payload["cells"]:
        cos_ci = cell["cosine_ci95"]
        diagnostic_ci = cell["diagnostic_ci95"]
        gap_ci = cell["gap_ci95_pp"]
        lines.append(
            f"| {cell['backbone_label']} | {100*cell['cosine_acc']:.2f} "
            f"[{100*cos_ci[0]:.2f},{100*cos_ci[1]:.2f}] | "
            f"{100*cell['diagnostic_acc']:.2f} "
            f"[{100*diagnostic_ci[0]:.2f},{100*diagnostic_ci[1]:.2f}] | "
            f"{cell['diagnostic_minus_cosine_pp']:.2f} "
            f"[{gap_ci[0]:.2f},{gap_ci[1]:.2f}] |"
        )
    ranking = payload["ranking"]
    lines += [
        "",
        f"- observed winners: Cos={ranking['cosine_winner_label']}; "
        f"diagnostic={ranking['diagnostic_winner_label']}",
        f"- observed diagnostic regret: {ranking['observed_regret_pp']:.2f}pp",
        f"- bootstrap flip rate: {100*ranking['bootstrap_flip_rate']:.1f}%",
        f"- bootstrap regret: {ranking['bootstrap_regret_mean_pp']:.2f}pp "
        f"[{ranking['bootstrap_regret_ci95_pp'][0]:.2f},"
        f"{ranking['bootstrap_regret_ci95_pp'][1]:.2f}]",
    ]
    return "\n".join(lines) + "\n"


def aggregate(output_root: Path, datasets: list[str], backbone_names: list[str]) -> dict:
    results = []
    for dataset in datasets:
        path = output_root / dataset / "result.json"
        if not path.exists():
            raise FileNotFoundError(f"missing dataset result: {path}")
        results.append(json.loads(path.read_text()))
    n_boot_values = {result["protocol"]["n_boot"] for result in results}
    if len(n_boot_values) != 1:
        raise ValueError("all dataset results must use the same bootstrap count")
    n_boot = n_boot_values.pop()
    regret = np.stack(
        [np.asarray(result["bootstrap_samples"]["regret_pp"]) for result in results]
    )
    flips = np.stack(
        [np.asarray(result["bootstrap_samples"]["flip"], dtype=np.int64) for result in results]
    )
    observed_regret = [result["ranking"]["observed_regret_pp"] for result in results]
    summary = {
        "n_datasets": len(results),
        "n_backbones": len(backbone_names),
        "n_boot": n_boot,
        "observed_top1_flips": int(
            sum(result["ranking"]["observed_flip"] for result in results)
        ),
        "observed_mean_regret_pp": float(np.mean(observed_regret)),
        "bootstrap_mean_regret_pp": float(regret.mean(0).mean()),
        "bootstrap_mean_regret_ci95_pp": percentile_ci(regret.mean(0)),
        "bootstrap_flip_count_mean": float(flips.sum(0).mean()),
        "bootstrap_flip_count_ci95": percentile_ci(flips.sum(0)),
    }
    payload = {
        "protocol": results[0]["protocol"],
        "summary": summary,
        "datasets": [
            {
                "dataset": result["dataset"],
                "dataset_label": result["dataset_label"],
                "n_test": result["n_test"],
                "cells": result["cells"],
                "ranking": result["ranking"],
                "point_estimate_checks": result["point_estimate_checks"],
            }
            for result in results
        ],
    }
    write_json_new(output_root / "aggregate.json", payload)
    write_new(output_root / "aggregate.md", aggregate_report(payload, backbone_names))
    return payload


def aggregate_report(payload: dict, backbone_names: list[str]) -> str:
    lines = ["# SDG six-backbone paired bootstrap\n"]
    lines.append("## Ranking consequences\n")
    lines.append(
        "| Dataset | Cos winner | Diagnostic winner | Observed regret pp | "
        "Bootstrap P(flip) | Regret pp [95% CI] | Winner stability Cos / diagnostic |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for result in payload["datasets"]:
        ranking = result["ranking"]
        lines.append(
            f"| {result['dataset_label']} | {ranking['cosine_winner_label']} | "
            f"{ranking['diagnostic_winner_label']} | {ranking['observed_regret_pp']:.2f} | "
            f"{100*ranking['bootstrap_flip_rate']:.1f}% | "
            f"{ranking['bootstrap_regret_mean_pp']:.2f} "
            f"[{ranking['bootstrap_regret_ci95_pp'][0]:.2f},"
            f"{ranking['bootstrap_regret_ci95_pp'][1]:.2f}] | "
            f"{100*ranking['cosine_winner_stability']:.1f}% / "
            f"{100*ranking['diagnostic_winner_stability']:.1f}% |"
        )
    summary = payload["summary"]
    lines += [
        "",
        f"- observed top-1 flips: {summary['observed_top1_flips']}/{summary['n_datasets']}",
        f"- observed mean diagnostic regret: {summary['observed_mean_regret_pp']:.2f}pp",
        f"- bootstrap mean diagnostic regret: {summary['bootstrap_mean_regret_pp']:.2f}pp "
        f"[{summary['bootstrap_mean_regret_ci95_pp'][0]:.2f},"
        f"{summary['bootstrap_mean_regret_ci95_pp'][1]:.2f}]",
        "",
        "## Per-cell diagnostic gap\n",
        "| Dataset | Backbone | Gap pp [95% CI] |",
        "|---|---|---:|",
    ]
    for result in payload["datasets"]:
        by_backbone = {cell["backbone"]: cell for cell in result["cells"]}
        for backbone in backbone_names:
            cell = by_backbone[backbone]
            interval = cell["gap_ci95_pp"]
            lines.append(
                f"| {result['dataset_label']} | {cell['backbone_label']} | "
                f"{cell['diagnostic_minus_cosine_pp']:.2f} "
                f"[{interval[0]:.2f},{interval[1]:.2f}] |"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "aggregate", "all"], default="all")
    parser.add_argument("--feature-root", default=str(FEATURE_ROOT))
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT / "backbone_bootstrap"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument(
        "--backbones", nargs="+", choices=list(BACKBONES), default=list(BACKBONES)
    )
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--point-tolerance", type=float, default=5e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_root = Path(args.feature_root)
    output_root = Path(args.output_root)
    if args.mode in {"run", "all"}:
        for dataset_index, dataset in enumerate(args.datasets):
            run_dataset(
                feature_root,
                output_root,
                dataset,
                args.backbones,
                ridge=args.ridge,
                n_boot=args.n_boot,
                seed=args.seed + 1009 * dataset_index,
                point_tolerance=args.point_tolerance,
                results_root=Path(args.results_root),
            )
    if args.mode in {"aggregate", "all"}:
        aggregate(output_root, args.datasets, args.backbones)
    print(f"[backbone-ci] complete output={output_root}")


if __name__ == "__main__":
    main()
