#!/usr/bin/env python3
"""Write mean/std few-shot summaries from the SDG-ED method aggregate."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(os.environ.get("SDG_RESULTS_ROOT", RELEASE_ROOT / "artifacts/results"))
OUT_ROOT = Path(os.environ.get("SDG_OUTPUT_ROOT", RELEASE_ROOT / "outputs"))
AGG = RESULTS_ROOT / "method_matrix_aggregate.json"
OUT = OUT_ROOT / "fewshot_summary.md"
SHOT_ORDER = {"1": 1, "2": 2, "5": 5, "10": 10, "20": 20, "full": 999}


def metric(row, name):
    m = row["metrics"]
    if name == "mahal":
        return m["mahal"]["acc"]
    if name == "logreg":
        v = m.get("logreg")
        return None if v is None else v["acc"]
    return m.get(name)


def fmt(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    arr = np.array(xs, dtype=float) * 100
    if len(arr) == 1:
        return f"{arr.mean():.2f}"
    return f"{arr.mean():.2f}±{arr.std(ddof=1):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", default=str(AGG))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rows = json.load(open(args.aggregate))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["shot"])].append(row)

    lines = ["# SDG-ED Few-shot Summary\n"]
    lines.append("| dataset | shot | n | Cos | Mahal | Gap | WCos | LogReg | FeCAM diag | SDGI-tail128 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (dataset, shot), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], SHOT_ORDER.get(kv[0][1], 1000))):
        cos = [metric(r, "cos") for r in rs]
        mah = [metric(r, "mahal") for r in rs]
        gap = [100 * (b - a) for a, b in zip(cos, mah)]
        wcos = [metric(r, "wcos") for r in rs]
        logreg = [metric(r, "logreg") for r in rs]
        fecam = [metric(r, "fecam_diag") for r in rs]
        sdgi = [r["sdgi"]["tail128_fisher"] for r in rs]
        gap_text = f"{np.mean(gap):.2f}" if len(gap) == 1 else f"{np.mean(gap):.2f}±{np.std(gap, ddof=1):.2f}"
        lines.append(
            f"| {dataset} | {shot} | {len(rs)} | {fmt(cos)} | {fmt(mah)} | {gap_text} | "
            f"{fmt(wcos)} | {fmt(logreg)} | {fmt(fecam)} | {fmt(sdgi)} |"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[fewshot-summary] wrote {out}")


if __name__ == "__main__":
    main()
