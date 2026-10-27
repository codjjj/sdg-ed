#!/usr/bin/env python3
"""Paired uncertainty for the full support-regime matrix (Appendix I).

For every DINOv2-B dataset-shot cell, reads the released seed-level records
(artifacts/results/stepb_cached_eval_v1/<dataset>/shot_<shot>/result.json) and
computes the paired LDA-Cos and WCos-Cos gaps (pp): mean, sample SD, 95%
Student-t interval, two-sided and one-sided ("less") one-sample t-test
p-values, and Benjamini-Hochberg q-values over the 45 five-seed cells.
Full-shot cells use the complete training split once and are point estimates.

    python scripts/recompute/paired_statistics.py

src/sdg_paired_statistics.py runs the same analysis with exact sign-flip
tests and the fixed-target comparison added.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

RELEASE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEPB = RELEASE_ROOT / "artifacts/results/stepb_cached_eval_v1"
DEFAULT_OUTPUT = RELEASE_ROOT / "outputs/recompute/paired_statistics.json"
DATASETS = ["aircraft", "cub200", "cars", "food101", "pets", "cifar100", "cifar10", "imagenet100", "tiny_imagenet"]
SHOTS = ["1", "2", "5", "10", "20", "full"]


def benjamini_hochberg(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(len(p))
    running = 1.0
    for i in range(len(p) - 1, -1, -1):
        running = min(running, ranked[i] * len(p) / (i + 1))
        adjusted[i] = running
    out = np.empty(len(p))
    out[order] = adjusted
    return out


def cell_statistics(stepb_root: Path, dataset: str, shot: str) -> dict:
    records = json.loads((stepb_root / dataset / f"shot_{shot}" / "result.json").read_text())["records"]
    records = sorted(records, key=lambda r: int(r["seed"]))
    cos = np.array([r["metrics"]["cos"] for r in records])
    wcos = np.array([r["metrics"]["wcos_trainval"]["acc"] for r in records])
    lda = np.array([r["metrics"]["lda_trainval"]["acc"] for r in records])
    row = {
        "dataset": dataset,
        "shot": shot,
        "n": len(records),
        "cos": 100 * cos.mean(),
        "wcos": 100 * wcos.mean(),
        "lda": 100 * lda.mean(),
    }
    for name, gap in (("lda", 100 * (lda - cos)), ("wcos", 100 * (wcos - cos))):
        mean = gap.mean()
        row[f"{name}_mean"] = mean
        if len(gap) > 1:
            sd = gap.std(ddof=1)
            row[f"{name}_sd"] = sd
            row[f"{name}_ci"] = stats.t.interval(0.95, df=len(gap) - 1, loc=mean, scale=sd / np.sqrt(len(gap)))
            row[f"{name}_p2"] = stats.ttest_1samp(gap, 0).pvalue
            row[f"{name}_pless"] = stats.ttest_1samp(gap, 0, alternative="less").pvalue
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stepb-root", type=Path, default=DEFAULT_STEPB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = [cell_statistics(args.stepb_root, d, s) for d in DATASETS for s in SHOTS]
    seeded = [r for r in rows if r["n"] > 1]
    for name in ("lda", "wcos"):
        q_two = benjamini_hochberg([r[f"{name}_p2"] for r in seeded])
        q_less = benjamini_hochberg([r[f"{name}_pless"] for r in seeded])
        for r, a, b in zip(seeded, q_two, q_less):
            r[f"{name}_q2"] = a
            r[f"{name}_qless"] = b

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1, default=float) + "\n")

    print("dataset        shot n | LDA-Cos mean  SD   [95% CI]          q(BH, two-sided) | WCos-Cos mean [95% CI]")
    for r in rows:
        if r["n"] > 1:
            print(
                f"{r['dataset']:13s} {r['shot']:>5s} {r['n']} | {r['lda_mean']:7.2f} {r['lda_sd']:5.2f} "
                f"[{r['lda_ci'][0]:6.2f}, {r['lda_ci'][1]:6.2f}] q={r['lda_q2']:.2e} | "
                f"{r['wcos_mean']:6.2f} [{r['wcos_ci'][0]:6.2f}, {r['wcos_ci'][1]:6.2f}]"
            )
        else:
            print(f"{r['dataset']:13s} {r['shot']:>5s} {r['n']} | {r['lda_mean']:7.2f} (point) | {r['wcos_mean']:6.2f} (point)")
    negative = [r for r in seeded if r["lda_mean"] < 0]
    print(
        f"five-seed cells: {len(seeded)}; negative mean LDA-Cos: {len(negative)} "
        f"(BH q<0.05: {sum(r['lda_q2'] < 0.05 for r in negative)})"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
