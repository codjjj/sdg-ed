#!/usr/bin/env python3
"""Train-validated 9-dataset x 6-backbone SDG ranking audit.

This runner reuses the frozen six-backbone feature caches.  It evaluates both
the original absolute regularization grid and a scale-normalized grid where
the actual ridge is ``alpha * trace(covariance) / dimension``.  Every ranking
bootstrap draw shares test indices across all backbones and evaluators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sdg_backbone_bootstrap_ci import (
    BACKBONES,
    DATASETS,
    DATASET_LABELS,
    feature_dir,
    predict_cosine as frozen_predict_cosine,
    reference_classifier_matrix,
)
from sdg_audit_core import (
    REG_GRID,
    accuracy,
    classes_from,
    environment_manifest,
    load_cache,
    select_then_refit_evaluator,
    write_json_new,
    write_text_new,
)
from sdg_paths import FEATURE_ROOT, OUTPUT_ROOT, RESULTS_ROOT
from sdg_stepb_cached_eval import train_val_split


METHODS = [
    "cos",
    "wcos_absolute",
    "lda_absolute",
    "validation_whitening_absolute",
    "validation_all_absolute",
    "wcos_relative",
    "lda_relative",
    "validation_whitening_relative",
    "validation_all_relative",
]
PRIMARY_METHOD = "validation_whitening_absolute"
SENSITIVITY_METHOD = "validation_whitening_relative"
SUBMITTED_FLIP_DATASETS = ["aircraft", "tiny_imagenet"]


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def prediction_digest(predictions: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(predictions).tobytes()).hexdigest()


def frozen_cosine_predictions(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    classes: list[int],
) -> np.ndarray:
    """Match the paper's Torch-float32 cosine classifier exactly."""

    train = torch.as_tensor(np.asarray(train_features), dtype=torch.float32)
    evaluation = torch.as_tensor(np.asarray(eval_features), dtype=torch.float32)
    return frozen_predict_cosine(train, train_labels, evaluation, classes)


def first_argmax(items: list[tuple[str, float]]) -> str:
    maximum = max(value for _, value in items)
    return next(name for name, value in items if value == maximum)


def evaluator(
    family: str,
    mode: str,
    xtr: np.ndarray,
    ytr: np.ndarray,
    xinner: np.ndarray,
    yinner: np.ndarray,
    xval: np.ndarray,
    yval: np.ndarray,
    xte: np.ndarray,
    classes: list[int],
) -> dict:
    return select_then_refit_evaluator(
        family,
        xtr,
        ytr,
        xinner,
        yinner,
        xval,
        yval,
        xte,
        classes,
        REG_GRID,
        regularization_mode=mode,
    )


def evaluate_cell(
    feature_root: Path,
    backbone: str,
    dataset: str,
    seed: int,
    tolerance: float,
    results_root: Path = RESULTS_ROOT,
) -> tuple[dict, dict]:
    directory = feature_dir(feature_root, backbone, dataset)
    train_path = directory / "train.pt"
    test_path = directory / "test.pt"
    xtr, ytr = load_cache(train_path)
    xte, yte = load_cache(test_path)
    classes = classes_from(ytr, yte)
    xinner, yinner, xval, yval, split = train_val_split(xtr, ytr, seed=seed)
    if xval is None:
        raise ValueError(f"full-shot backbone cell unexpectedly has no validation split: {dataset}/{backbone}")

    cos_predictions = frozen_cosine_predictions(xtr, ytr, xte, classes)
    cos_validation = accuracy(frozen_cosine_predictions(xinner, yinner, xval, classes), yval)
    outputs = {}
    for mode in ["absolute", "relative_trace"]:
        suffix = "absolute" if mode == "absolute" else "relative"
        outputs[f"wcos_{suffix}"] = evaluator(
            "wcos", mode, xtr, ytr, xinner, yinner, xval, yval, xte, classes
        )
        outputs[f"lda_{suffix}"] = evaluator(
            "lda", mode, xtr, ytr, xinner, yinner, xval, yval, xte, classes
        )

    predictions = {"cos": cos_predictions}
    selection = {}
    for suffix in ["absolute", "relative"]:
        wcos = outputs[f"wcos_{suffix}"]
        lda = outputs[f"lda_{suffix}"]
        wcos_val = max(wcos["validation_scores"])
        lda_val = max(lda["validation_scores"])
        whitening_family = first_argmax([("wcos", wcos_val), ("lda", lda_val)])
        all_family = first_argmax([("cos", cos_validation), ("wcos", wcos_val), ("lda", lda_val)])
        predictions[f"wcos_{suffix}"] = wcos["predictions"]
        predictions[f"lda_{suffix}"] = lda["predictions"]
        predictions[f"validation_whitening_{suffix}"] = predictions[f"{whitening_family}_{suffix}"]
        predictions[f"validation_all_{suffix}"] = (
            predictions["cos"] if all_family == "cos" else predictions[f"{all_family}_{suffix}"]
        )
        selection[suffix] = {
            "cos_validation_acc": cos_validation,
            "wcos_validation_best_acc": wcos_val,
            "lda_validation_best_acc": lda_val,
            "validation_whitening_family": whitening_family,
            "validation_all_family": all_family,
            "wcos": {key: value for key, value in wcos.items() if key != "predictions"},
            "lda": {key: value for key, value in lda.items() if key != "predictions"},
        }

    reference = reference_classifier_matrix(results_root, backbone, dataset)
    expected_cos = float(reference["cosine"])
    observed_cos = accuracy(predictions["cos"], yte)
    if abs(observed_cos - expected_cos) > tolerance:
        raise AssertionError(
            f"frozen cosine mismatch {dataset}/{backbone}: {observed_cos} vs {expected_cos}"
        )

    cell = {
        "dataset": dataset,
        "backbone": backbone,
        "backbone_label": BACKBONES[backbone],
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_classes": int(len(classes)),
        "validation_split": split,
        "input_files": [file_identity(train_path), file_identity(test_path)],
        "frozen_cosine_abs_error": abs(observed_cos - expected_cos),
        "selection": selection,
        "accuracies": {method: accuracy(predictions[method], yte) for method in METHODS},
        "prediction_sha256": {method: prediction_digest(predictions[method]) for method in METHODS},
    }
    return cell, {"labels": yte, "predictions": predictions}


def percentile(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def paired_ranking_bootstrap(
    correct_by_method: dict[str, np.ndarray],
    backbone_names: list[str],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    shapes = {array.shape for array in correct_by_method.values()}
    if len(shapes) != 1:
        raise ValueError("all correctness matrices must share shape")
    n_backbones, n_test = next(iter(shapes))
    if n_backbones != len(backbone_names):
        raise ValueError("backbone dimension mismatch")
    rng = np.random.RandomState(seed)
    method_samples = {
        method: {
            "regret_pp": np.empty(n_boot),
            "flip": np.empty(n_boot, dtype=np.bool_),
            "gap_pp": np.empty((n_backbones, n_boot)),
        }
        for method in METHODS
        if method != "cos"
    }
    for index in range(n_boot):
        sample = rng.randint(0, n_test, size=n_test)
        cos_accuracy = correct_by_method["cos"][:, sample].mean(1)
        cos_winner = int(np.argmax(cos_accuracy))
        for method, samples in method_samples.items():
            method_accuracy = correct_by_method[method][:, sample].mean(1)
            winner = int(np.argmax(method_accuracy))
            samples["gap_pp"][:, index] = 100.0 * (method_accuracy - cos_accuracy)
            samples["regret_pp"][index] = 100.0 * (
                method_accuracy[winner] - method_accuracy[cos_winner]
            )
            samples["flip"][index] = winner != cos_winner

    cos_point = correct_by_method["cos"].mean(1)
    cos_winner = int(np.argmax(cos_point))
    results = {}
    for method, samples in method_samples.items():
        point = correct_by_method[method].mean(1)
        winner = int(np.argmax(point))
        regret = samples["regret_pp"]
        if float(regret.min()) < -1e-12:
            raise AssertionError("selection regret must be nonnegative")
        results[method] = {
            "cosine_winner": backbone_names[cos_winner],
            "method_winner": backbone_names[winner],
            "observed_flip": bool(winner != cos_winner),
            "observed_regret_pp": float(100.0 * (point[winner] - point[cos_winner])),
            "bootstrap_flip_rate": float(samples["flip"].mean()),
            "bootstrap_regret_mean_pp": float(regret.mean()),
            "bootstrap_regret_ci95_pp": percentile(regret),
            "cell_gap_by_backbone": {
                backbone: {
                    "point_pp": float(
                        100.0
                        * (
                            correct_by_method[method][backbone_index].mean()
                            - correct_by_method["cos"][backbone_index].mean()
                        )
                    ),
                    "ci95_pp": percentile(samples["gap_pp"][backbone_index]),
                }
                for backbone_index, backbone in enumerate(backbone_names)
            },
            "bootstrap_samples": {
                "regret_pp": regret.tolist(),
                "flip": samples["flip"].tolist(),
            },
        }
    return results


def run_dataset(feature_root: Path, output_root: Path, dataset: str, args) -> dict:
    print(f"[backbone-trainval] dataset={dataset}", flush=True)
    backbone_names = list(BACKBONES)
    cells = []
    raw = []
    reference_labels = None
    for backbone in backbone_names:
        print(f"[backbone-trainval] dataset={dataset} backbone={backbone}", flush=True)
        cell, prediction_bundle = evaluate_cell(
            feature_root,
            backbone,
            dataset,
            args.validation_seed,
            args.point_tolerance,
            results_root=Path(args.results_root),
        )
        labels = prediction_bundle["labels"]
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise AssertionError(f"test label ordering mismatch: {dataset}/{backbone}")
        cells.append(cell)
        raw.append(prediction_bundle["predictions"])

    correct_by_method = {
        method: np.stack([bundle[method] == reference_labels for bundle in raw]) for method in METHODS
    }
    ranking = paired_ranking_bootstrap(
        correct_by_method,
        backbone_names,
        n_boot=args.n_boot,
        seed=args.seed + DATASETS.index(dataset) * 1009,
    )
    cell_intervals = []
    for backbone_index, backbone in enumerate(backbone_names):
        for method in METHODS:
            if method == "cos":
                continue
            interval = ranking[method]["cell_gap_by_backbone"][backbone]
            cell_intervals.append(
                {
                    "backbone": backbone,
                    "method": method,
                    **interval,
                    "n_test": int(len(reference_labels)),
                    "n_boot": args.n_boot,
                    "seed": args.seed + DATASETS.index(dataset) * 1009,
                }
            )
    payload = {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "n_test": int(len(reference_labels)),
        "backbones": backbone_names,
        "methods": METHODS,
        "cells": cells,
        "ranking": ranking,
        "cell_gap_intervals": cell_intervals,
    }
    write_json_new(output_root / dataset / "result.json", payload)
    write_text_new(output_root / dataset / "result.md", dataset_report(payload))
    return payload


def dataset_report(payload: dict) -> str:
    lines = [f"# {payload['dataset_label']} train-validated cross-backbone audit\n"]
    lines.append("| Backbone | Cos | WVal-abs | AllVal-abs | WVal-rel | AllVal-rel |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cell in payload["cells"]:
        acc = cell["accuracies"]
        lines.append(
            f"| {cell['backbone_label']} | {100*acc['cos']:.2f} | "
            f"{100*acc['validation_whitening_absolute']:.2f} | {100*acc['validation_all_absolute']:.2f} | "
            f"{100*acc['validation_whitening_relative']:.2f} | {100*acc['validation_all_relative']:.2f} |"
        )
    lines.append("\n## Ranking consequences")
    for method in [PRIMARY_METHOD, SENSITIVITY_METHOD]:
        row = payload["ranking"][method]
        lines.append(
            f"- {method}: Cos winner={BACKBONES[row['cosine_winner']]}; evaluator winner={BACKBONES[row['method_winner']]}; "
            f"regret={row['observed_regret_pp']:.2f}pp; CI={row['bootstrap_regret_ci95_pp']}."
        )
    return "\n".join(lines) + "\n"


def aggregate(results: list[dict], output_root: Path, args) -> dict:
    primary = [result["ranking"][PRIMARY_METHOD] for result in results]
    sensitivity = [result["ranking"][SENSITIVITY_METHOD] for result in results]
    retained_primary = [
        result["dataset"]
        for result in results
        if result["dataset"] in SUBMITTED_FLIP_DATASETS
        and result["ranking"][PRIMARY_METHOD]["bootstrap_regret_ci95_pp"][0] > 0.0
    ]
    retained_sensitivity = [
        result["dataset"]
        for result in results
        if result["dataset"] in SUBMITTED_FLIP_DATASETS
        and result["ranking"][SENSITIVITY_METHOD]["bootstrap_regret_ci95_pp"][0] > 0.0
    ]
    payload = {
        "experiment": "sdg_backbone_trainval_audit",
        "post_submission": True,
        "protocol": {
            "datasets": DATASETS,
            "backbones": list(BACKBONES),
            "methods": METHODS,
            "primary_method": PRIMARY_METHOD,
            "sensitivity_method": SENSITIVITY_METHOD,
            "regularization_grid": REG_GRID,
            "absolute": "ridge = grid value",
            "relative": "ridge = grid value * trace(training covariance) / dimension",
            "validation_seed": args.validation_seed,
            "n_boot": args.n_boot,
            "bootstrap_seed": args.seed,
            "bootstrap_unit": "shared test-example index across all backbones/evaluators within dataset",
            "cosine_numeric_semantics": "submitted Torch float32 classifier",
        },
        "environment": environment_manifest(args.commit),
        "decision_facts": {
            "primary_observed_flips": int(sum(row["observed_flip"] for row in primary)),
            "relative_observed_flips": int(sum(row["observed_flip"] for row in sensitivity)),
            "submitted_flip_datasets_with_primary_ci_strictly_positive": retained_primary,
            "submitted_flip_datasets_with_relative_ci_strictly_positive": retained_sensitivity,
            "at_least_one_submitted_flip_survives_primary": bool(retained_primary),
        },
        "datasets": results,
    }
    write_json_new(output_root / "backbone_trainval_audit.json", payload)
    write_text_new(output_root / "backbone_trainval_audit.md", aggregate_report(payload))
    return payload


def aggregate_report(payload: dict) -> str:
    lines = ["# SDG 9x6 train-validated backbone audit\n"]
    lines.append("| Dataset | Cos winner | WVal-abs winner | Regret pp [95% CI] | WVal-rel winner | Regret pp [95% CI] |")
    lines.append("|---|---|---|---:|---|---:|")
    for result in payload["datasets"]:
        primary = result["ranking"][PRIMARY_METHOD]
        relative = result["ranking"][SENSITIVITY_METHOD]
        lines.append(
            f"| {result['dataset_label']} | {BACKBONES[primary['cosine_winner']]} | "
            f"{BACKBONES[primary['method_winner']]} | {primary['observed_regret_pp']:.2f} "
            f"[{primary['bootstrap_regret_ci95_pp'][0]:.2f},{primary['bootstrap_regret_ci95_pp'][1]:.2f}] | "
            f"{BACKBONES[relative['method_winner']]} | {relative['observed_regret_pp']:.2f} "
            f"[{relative['bootstrap_regret_ci95_pp'][0]:.2f},{relative['bootstrap_regret_ci95_pp'][1]:.2f}] |"
        )
    lines.append("\n## Decision facts")
    for key, value in payload["decision_facts"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", default=str(FEATURE_ROOT))
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT / "backbone_trainval_9x6"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument("--point-tolerance", type=float, default=5e-6)
    parser.add_argument("--commit", default=None)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    results = [run_dataset(Path(args.feature_root), output_root, dataset, args) for dataset in DATASETS]
    aggregate(results, output_root, args)
    print(f"[backbone-trainval] complete output={output_root}", flush=True)


if __name__ == "__main__":
    main()
