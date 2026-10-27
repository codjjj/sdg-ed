#!/usr/bin/env python3
"""Per-dataset rows of the expanded router table (Appendix C).

Reads the OOD11 replay JSON and aggregates one block's group records by
dataset: number of dataset-shot groups, mean measured gap (target), mean router
prediction, MAE, and mean router regret (all in pp). The default block,
`coarse_ood4`, holds the four datasets that extend OOD4 to OOD8 (MNIST,
Fashion-MNIST, SVHN, GTSRB). The block-level policy regrets are printed as a
consistency check against the published OOD8 numbers.

    python scripts/recompute/ood8_per_dataset.py
    python scripts/recompute/ood8_per_dataset.py --block finegrained_ood3
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RELEASE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY = RELEASE_ROOT / "artifacts/results/router_ood11_replay/router_ood11_replay.json"
LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "svhn": "SVHN",
    "gtsrb": "GTSRB",
    "eurosat": "EuroSAT",
    "caltech101": "Caltech-101",
    "flowers102": "Flowers-102",
    "dtd": "DTD",
    "stanford_dogs": "Stanford Dogs",
    "foodx251": "FoodX-251",
    "nabirds": "NABirds",
}


def per_dataset_rows(records: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["dataset"]].append(record)
    rows = []
    for dataset, items in groups.items():
        target = np.array([r["target_best_whitening_minus_cos_pp"] for r in items], dtype=float)
        prediction = np.array([r["router_prediction_pp"] for r in items], dtype=float)
        regret = np.array([r["router_policy_regret_pp"] for r in items], dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "groups": len(items),
                "target_mean_pp": float(target.mean()),
                "prediction_mean_pp": float(prediction.mean()),
                "mae_pp": float(np.abs(prediction - target).mean()),
                "router_regret_pp": float(regret.mean()),
            }
        )
    return rows


def block_regrets(records: list[dict]) -> dict[str, float]:
    """Mean regret to the per-group oracle max(cos, best whitening), in pp."""

    cos = np.array([r["cos"] for r in records])
    whitening = np.array([r["best_whitening"] for r in records])
    oracle = np.maximum(cos, whitening)
    routed = np.where(np.array([r["router_prediction_pp"] for r in records]) >= 5.0, whitening, cos)
    return {
        "router": float(100 * np.mean(oracle - routed)),
        "always_wcos": float(100 * np.mean(oracle - np.array([r["wcos_trainval"] for r in records]))),
        "always_lda": float(100 * np.mean(oracle - np.array([r["lda_trainval"] for r in records]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY, help="router_ood11_replay.json")
    parser.add_argument(
        "--block",
        default="coarse_ood4",
        choices=["original_ood4", "coarse_ood4", "finegrained_ood3"],
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args()

    payload = json.loads(args.replay.read_text())
    records = payload["blocks"][args.block]["records"]
    rows = per_dataset_rows(records)
    block = block_regrets(records)
    if args.json:
        print(json.dumps({"block": args.block, "rows": rows, "block_regret_pp": block}, indent=2))
        return
    print(f"{args.block}: per-dataset router results (pp)")
    print(f"{'Dataset':<15} {'Groups':>6} {'Target mean':>12} {'Pred. mean':>11} {'MAE':>7} {'Router regret':>14}")
    for row in rows:
        print(
            f"{LABELS.get(row['dataset'], row['dataset']):<15} {row['groups']:>6d} "
            f"{row['target_mean_pp']:>12.2f} {row['prediction_mean_pp']:>11.2f} "
            f"{row['mae_pp']:>7.2f} {row['router_regret_pp']:>14.2f}"
        )
    print(
        f"block mean regret: router {block['router']:.3f} | always-WCos {block['always_wcos']:.3f} "
        f"| always-LDA {block['always_lda']:.3f}"
    )


if __name__ == "__main__":
    main()
