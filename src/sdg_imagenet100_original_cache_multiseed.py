#!/usr/bin/env python3
"""Three-seed WCos/LDA audit on the six-backbone ImageNet-100 feature caches.

Produces the ImageNet-100 column of Table 27 (Appendix J) from the original
ImageNet-100 caches of the six-backbone sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sdg_backbone_bootstrap_ci import BACKBONES
from sdg_paths import FEATURE_ROOT, OUTPUT_ROOT, RESULTS_ROOT
from sdg_audit_core import environment_manifest, write_json_new, write_text_new
from sdg_backbone_trainval_audit import evaluate_cell


DATASET = "imagenet100"
SELECTION_SEEDS = (20260724, 20260725, 20260726)
METHODS = ("wcos_relative", "lda_relative")


def summarize_cells(cells: list[dict], seeds: tuple[int, ...] = SELECTION_SEEDS) -> list[dict]:
    """Validate the seed/backbone matrix and return gap mean and sample SD."""

    expected = {(backbone, seed) for backbone in BACKBONES for seed in seeds}
    observed = {(cell["backbone"], int(cell["selection_seed"])) for cell in cells}
    if len(observed) != len(cells):
        raise ValueError("duplicate backbone/selection-seed cell")
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"incomplete cell matrix: missing={missing}, extra={extra}")

    summaries = []
    for backbone, label in BACKBONES.items():
        backbone_cells = sorted(
            (cell for cell in cells if cell["backbone"] == backbone),
            key=lambda cell: int(cell["selection_seed"]),
        )
        per_seed = []
        for cell in backbone_cells:
            accuracies = cell["accuracies"]
            cosine = float(accuracies["cos"])
            row = {
                "selection_seed": int(cell["selection_seed"]),
                "cosine_accuracy": cosine,
            }
            for method in METHODS:
                row[f"{method}_accuracy"] = float(accuracies[method])
                row[f"{method}_minus_cosine_pp"] = 100.0 * (
                    float(accuracies[method]) - cosine
                )
            per_seed.append(row)

        summary = {
            "backbone": backbone,
            "backbone_label": label,
            "selection_seeds": list(seeds),
            "per_seed": per_seed,
        }
        for method in METHODS:
            gaps = np.asarray(
                [row[f"{method}_minus_cosine_pp"] for row in per_seed], dtype=np.float64
            )
            summary[f"{method}_minus_cosine_pp"] = {
                "mean": float(gaps.mean()),
                "sample_sd": float(gaps.std(ddof=1)),
            }
        summaries.append(summary)
    return summaries


def report(payload: dict) -> str:
    lines = [
        "# ImageNet-100 original-cache three-seed evaluator audit",
        "",
        "The table reports mean ± sample SD across the three train-only "
        "regularizer-selection seeds.",
        "",
        "| Backbone | WCos−Cos (pp) | LDA−Cos (pp) |",
        "|---|---:|---:|",
    ]
    for row in payload["summary"]:
        wcos = row["wcos_relative_minus_cosine_pp"]
        lda = row["lda_relative_minus_cosine_pp"]
        lines.append(
            f"| {row['backbone_label']} | {wcos['mean']:+.2f}±{wcos['sample_sd']:.2f} | "
            f"{lda['mean']:+.2f}±{lda['sample_sd']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", default=str(FEATURE_ROOT))
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument(
        "--output-root", default=str(OUTPUT_ROOT / "imagenet100_original_cache_multiseed")
    )
    parser.add_argument("--point-tolerance", type=float, default=5e-6)
    parser.add_argument("--commit", default=None)
    args = parser.parse_args()

    feature_root = Path(args.feature_root)
    output_root = Path(args.output_root)
    cells = []
    for seed in SELECTION_SEEDS:
        for backbone in BACKBONES:
            print(
                f"[imagenet100-multiseed] seed={seed} backbone={backbone}",
                flush=True,
            )
            cell, _ = evaluate_cell(
                feature_root,
                backbone,
                DATASET,
                seed,
                args.point_tolerance,
                results_root=Path(args.results_root),
            )
            cell["selection_seed"] = seed
            cells.append(cell)

    payload = {
        "experiment": "sdg_imagenet100_original_cache_multiseed",
        "protocol": {
            "dataset": DATASET,
            "data": "same frozen feature caches and test split used by submitted Appendix E",
            "backbones": list(BACKBONES),
            "selection_seeds": list(SELECTION_SEEDS),
            "selection": (
                "class-balanced train-only 20% validation split capped at 50 "
                "examples per class; refit on all training examples"
            ),
            "regularization": (
                "relative-trace ridge grid; actual ridge = grid value * "
                "trace(training covariance) / dimension"
            ),
            "reported_uncertainty": "mean and sample SD across selection seeds",
            "cosine_numeric_semantics": "submitted Torch float32 classifier",
        },
        "environment": environment_manifest(args.commit),
        "summary": summarize_cells(cells),
        "cells": cells,
    }
    write_json_new(output_root / "result.json", payload)
    write_text_new(output_root / "result.md", report(payload))
    write_text_new(output_root / "_SUCCESS", "artifact_gate_candidate=true\n")
    print(f"[imagenet100-multiseed] complete output={output_root}", flush=True)


if __name__ == "__main__":
    main()
