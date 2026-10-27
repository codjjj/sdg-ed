#!/usr/bin/env python3
"""Spectral-diagnostic correlations behind Figure 3 (with and without CUB-200).

For the 54 DINOv2-B dataset-shot groups, pairs the training-set statistics
tail-128 Fisher mass and top-10 variance concentration (seed means, %) with the
per-group best-whitening gap and reports Pearson r pooled and per shot, over
all nine datasets and without the six CUB-200 groups (whose core cache is the
patch-token mean). Inputs are released artifacts:

    artifacts/results/method_matrix_aggregate.json              (seed-level diagnostics)
    artifacts/results/stepb_cached_eval_v1/aggregate_groups.json (54 group targets)
    artifacts/results/stepb_cached_eval_v1/*/shot_*/result.json  (per-seed evaluator records)

    python scripts/recompute/spectral_diagnostics.py
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy import stats

RELEASE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = RELEASE_ROOT / "artifacts/results"
DEFAULT_OUTPUT = RELEASE_ROOT / "outputs/recompute/spectral_diagnostics.json"
SHOTS = ["1", "2", "5", "10", "20", "full"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.results_root

    method_rows = json.loads((root / "method_matrix_aggregate.json").read_text())
    groups = json.loads((root / "stepb_cached_eval_v1/aggregate_groups.json").read_text())
    diag = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in method_rows:
        for k in ("tail128_fisher", "top10_var"):
            diag[(r["dataset"], str(r["shot"]))][k].append(r["sdgi"][k])

    def per_seed_targets(dataset: str, shot: str) -> tuple[float, float, float]:
        records = json.loads((root / f"stepb_cached_eval_v1/{dataset}/shot_{shot}/result.json").read_text())["records"]
        c = np.array([x["metrics"]["cos"] for x in records])
        w = np.array([x["metrics"]["wcos_trainval"]["acc"] for x in records])
        l = np.array([x["metrics"]["lda_trainval"]["acc"] for x in records])
        return 100 * float(np.mean(np.maximum(w, l) - c)), 100 * float(np.mean(w - c)), 100 * float(np.mean(l - c))

    rows = []
    for g in groups:
        key = (g["dataset"], str(g["shot"]))
        per_seed_max, wcos_gap, lda_gap = per_seed_targets(*key)
        rows.append(
            {
                "dataset": key[0],
                "shot": key[1],
                "tail": 100 * np.mean(diag[key]["tail128_fisher"]),
                "top10": 100 * np.mean(diag[key]["top10_var"]),
                "y_groupmax": g["best_whitening_minus_cos_pp"],
                "y_perseed": per_seed_max,
                "y_wcos": wcos_gap,
                "y_lda": lda_gap,
                "n_seed": len(diag[key]["tail128_fisher"]),
            }
        )

    def pearson(sub, x, y) -> float:
        return float(stats.pearsonr([q[x] for q in sub], [q[y] for q in sub])[0])

    correlations = {}
    for target in ("y_groupmax", "y_perseed", "y_wcos", "y_lda"):
        result = {}
        for label, sub in (("all", rows), ("no_cub", [q for q in rows if q["dataset"] != "cub200"])):
            result[label] = {
                "n": len(sub),
                "tail": pearson(sub, "tail", target),
                "top10": pearson(sub, "top10", target),
                "by_shot": {
                    s: {
                        "n": len([q for q in sub if q["shot"] == s]),
                        "tail": pearson([q for q in sub if q["shot"] == s], "tail", target),
                        "top10": pearson([q for q in sub if q["shot"] == s], "top10", target),
                    }
                    for s in SHOTS
                },
            }
        correlations[target] = result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows, "correlations": correlations}, indent=1, default=float) + "\n")
    print("Pearson r with the best-whitening gap (tail-128 Fisher / top-10 variance)")
    for target, result in correlations.items():
        for label in ("all", "no_cub"):
            v = result[label]
            per_shot = " ".join(f"{s}:{v['by_shot'][s]['tail']:.2f}/{v['by_shot'][s]['top10']:.2f}" for s in SHOTS)
            print(f"  {target:10s} {label:6s} n={v['n']:2d} pooled {v['tail']:.3f}/{v['top10']:.3f} | {per_shot}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
