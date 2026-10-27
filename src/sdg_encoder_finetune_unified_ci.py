#!/usr/bin/env python3
"""One bootstrap interval per gap for the encoder fine-tuning runs (Appendix K).

Combines the frozen, last-block (E11) and full-encoder (E14) runs: three-run
mean paired gap with a bootstrap over runs and paired test examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DATASETS = ("cub200", "aircraft", "cifar100")
METHODS = ("wcos_tv", "lda_tv")
STAGES = (
    ("frozen", "last_block", "frozen"),
    ("last_block", "last_block", "finetuned"),
    ("full_encoder", "full_encoder", "finetuned"),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runs(root: Path, dataset: str) -> list[dict[str, np.ndarray]]:
    runs = []
    for seed in range(3):
        path = root / dataset / f"seed_{seed}" / "predictions.npz"
        with np.load(path, allow_pickle=False) as bundle:
            arrays = {name: np.asarray(bundle[name]) for name in bundle.files}
        expected = {"labels"} | {
            f"{prefix}__{method}"
            for prefix in ("frozen", "finetuned")
            for method in ("cos", *METHODS, "validation_whitening")
        }
        if set(arrays) != expected:
            raise ValueError(f"unexpected prediction arrays: {path}")
        runs.append(arrays)
    labels = runs[0]["labels"]
    if any(not np.array_equal(labels, run["labels"]) for run in runs[1:]):
        raise ValueError(f"test labels differ across runs: {dataset}")
    return runs


def hierarchical_ci(
    deltas_pp: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> list[float]:
    """Resample training runs and paired test examples in each replicate."""

    deltas_pp = np.asarray(deltas_pp, dtype=np.float64)
    if deltas_pp.ndim != 2 or deltas_pp.shape[0] != 3:
        raise ValueError("expected three runs by test examples")
    n_runs, n_test = deltas_pp.shape
    run_rng = np.random.RandomState(seed)
    example_rng = np.random.RandomState(seed + 1_000_003)
    run_counts = run_rng.multinomial(
        n_runs,
        np.full(n_runs, 1.0 / n_runs),
        size=n_boot,
    )
    bootstrap = np.empty(n_boot, dtype=np.float64)
    filled = np.zeros(n_boot, dtype=bool)
    for composition in np.unique(run_counts, axis=0):
        mask = np.all(run_counts == composition, axis=1)
        weights = composition.astype(np.float64) / n_runs
        per_example = np.sum(weights[:, None] * deltas_pp, axis=0)
        values, counts = np.unique(per_example, return_counts=True)
        draws = example_rng.multinomial(
            n_test,
            counts.astype(np.float64) / n_test,
            size=int(mask.sum()),
        )
        bootstrap[mask] = (
            np.sum(draws.astype(np.float64) * values[None, :], axis=1) / n_test
        )
        filled[mask] = True
    if not np.all(filled) or not np.all(np.isfinite(bootstrap)):
        raise RuntimeError("bootstrap output is incomplete or non-finite")
    return [
        float(np.percentile(bootstrap, 2.5)),
        float(np.percentile(bootstrap, 97.5)),
    ]


def gap_matrix(
    runs: list[dict[str, np.ndarray]],
    *,
    prefix: str,
    method: str,
) -> np.ndarray:
    labels = runs[0]["labels"]
    return np.stack(
        [
            100.0
            * (
                (run[f"{prefix}__{method}"] == labels).astype(np.float64)
                - (run[f"{prefix}__cos"] == labels).astype(np.float64)
            )
            for run in runs
        ]
    )


def summarize(args: argparse.Namespace) -> dict:
    roots = {
        "last_block": Path(args.last_block_root),
        "full_encoder": Path(args.full_encoder_root),
    }
    output = {
        "schema_version": 1,
        "kind": "sdg_encoder_finetune_unified_hierarchical_ci",
        "method": (
            "three-run mean paired accuracy gap with a percentile bootstrap "
            "that resamples training runs and paired test examples"
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "cells": {},
        "source_sha256": {},
    }
    for dataset_index, dataset in enumerate(DATASETS):
        loaded = {
            scope: load_runs(root, dataset) for scope, root in roots.items()
        }
        labels = loaded["last_block"][0]["labels"]
        if not np.array_equal(labels, loaded["full_encoder"][0]["labels"]):
            raise ValueError(f"test labels differ across scopes: {dataset}")
        for method in METHODS:
            reference = loaded["last_block"][0]
            for scope in loaded.values():
                for run in scope:
                    for name in (method, "cos"):
                        if not np.array_equal(
                            run[f"frozen__{name}"],
                            reference[f"frozen__{name}"],
                        ):
                            raise ValueError(
                                f"frozen predictions differ: {dataset}/{name}"
                            )
        output["cells"][dataset] = {}
        for stage_index, (stage, scope, prefix) in enumerate(STAGES):
            runs = loaded[scope]
            for method_index, method in enumerate(METHODS):
                deltas = gap_matrix(runs, prefix=prefix, method=method)
                bootstrap_seed = (
                    args.bootstrap_seed
                    + dataset_index * 100
                    + stage_index * 10
                    + method_index
                )
                output["cells"][dataset].setdefault(stage, {})[method] = {
                    "mean_gap_pp": float(deltas.mean()),
                    "ci95_pp": hierarchical_ci(
                        deltas,
                        n_boot=args.bootstrap_replicates,
                        seed=bootstrap_seed,
                    ),
                    "seed_points_pp": deltas.mean(axis=1).tolist(),
                    "n_runs": int(deltas.shape[0]),
                    "n_test": int(deltas.shape[1]),
                    "bootstrap_seed": bootstrap_seed,
                }
        for scope, root in roots.items():
            for seed in range(3):
                path = root / dataset / f"seed_{seed}" / "predictions.npz"
                output["source_sha256"][
                    f"{scope}/{dataset}/seed_{seed}/predictions.npz"
                ] = sha256_file(path)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last-block-root", required=True)
    parser.add_argument("--full-encoder-root", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2_026_072_600)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(summarize(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
