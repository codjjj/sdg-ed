#!/usr/bin/env python3
"""High-shot diagnostic correlations with and without CUB-200 (Appendix F).

Pearson correlation of four training-set diagnostics (tail-128 Fisher mass,
tail-128 enrichment, effective rank, log10 condition number) with the oracle
gap over the 36 high-shot supplement groups, for all nine datasets and without
the four CUB-200 groups. Input: the released
artifacts/results/supplement_stats_v1/aggregate_groups.json.

    python scripts/recompute/highshot_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

RELEASE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = RELEASE_ROOT / "artifacts/results/supplement_stats_v1/aggregate_groups.json"
DEFAULT_OUTPUT = RELEASE_ROOT / "outputs/recompute/highshot_diagnostics.json"
STATISTICS = {
    "tail128_fisher": lambda r: r["tail128_fisher"],
    "tail128_enrichment": lambda r: r["tail128_enrichment"],
    "effective_rank": lambda r: r["effective_rank"],
    "log10_condition": lambda r: np.log10(r["condition_positive"]),
}


def correlations(rows: list[dict]) -> dict:
    y = np.array([r["gap_best_pp"] for r in rows])
    out = {"n": len(rows)}
    for name, value in STATISTICS.items():
        out[name] = float(stats.pearsonr(np.array([value(r) for r in rows]), y)[0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text())
    result = {
        "all": correlations(rows),
        "no_cub": correlations([r for r in rows if r["dataset"] != "cub200"]),
        "datasets": sorted({r["dataset"] for r in rows}),
        "cub_groups": sum(r["dataset"] == "cub200" for r in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
