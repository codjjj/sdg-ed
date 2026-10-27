#!/usr/bin/env python3
"""Gap targets and fixed reporting policies with dataset-cluster CIs (Appendix C).

Over the 54 core DINOv2-B dataset-shot groups this computes:
  * the macro-mean gap over cosine for three targets: fixed WCos-TV, fixed
    LDA-TV, and the per-group max(WCos-TV, LDA-TV);
  * the paired inflation of the max target over each fixed target (the same
    bootstrap draws are differenced, not two marginal intervals);
  * the leave-one-dataset-out router Pearson correlation for each target;
  * the regret of always-Cos / always-WCos / always-LDA to the per-group best
    of {Cos, WCos, LDA}.
Intervals resample whole datasets (all six shot groups kept together), 10,000
draws, seed 20260724.

    python scripts/recompute/fixed_targets_cluster_ci.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from sdg_paired_statistics import target_sensitivity  # noqa: E402

DEFAULT_JOINED = RELEASE_ROOT / "artifacts/results/stepb_predictive_protocol_reproduced/joined_records.json"
DEFAULT_OUTPUT = RELEASE_ROOT / "outputs/recompute/fixed_targets_cluster_ci.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    joined = json.loads(args.joined.read_text())
    datasets = np.array([r["dataset"] for r in joined])
    groups = [np.where(datasets == name)[0] for name in sorted(set(datasets))]

    def boot(stat) -> list[float]:
        rng = np.random.RandomState(args.seed)
        values = []
        for _ in range(args.n_boot):
            drawn = rng.randint(0, len(groups), size=len(groups))
            values.append(stat(np.concatenate([groups[i] for i in drawn])))
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    cos = np.array([r["cos"] for r in joined])
    wcos = np.array([r["wcos_trainval"] for r in joined])
    lda = np.array([r["lda_trainval"] for r in joined])
    best_whitening = np.array([r["best_whitening"] for r in joined])
    targets = {
        "fixed_wcos": 100 * (wcos - cos),
        "fixed_lda": 100 * (lda - cos),
        "max_wcos_lda": 100 * (best_whitening - cos),
    }
    lodo = {row["target"]: row["lodo_metrics"]["pearson"] for row in target_sensitivity(joined)}
    lodo["max_wcos_lda"] = lodo.pop("best_of_wcos_lda")

    out = {"gap_targets": [], "paired_inflation": [], "fixed_policies": []}
    for name, values in targets.items():
        out["gap_targets"].append(
            {
                "target": name,
                "mean_pp": float(values.mean()),
                "ci95_pp": boot(lambda idx: values[idx].mean()),
                "lodo_pearson": float(lodo[name]),
            }
        )
    for name in ("fixed_wcos", "fixed_lda"):
        diff = targets["max_wcos_lda"] - targets[name]
        out["paired_inflation"].append(
            {"contrast": f"max_wcos_lda - {name}", "mean_pp": float(diff.mean()), "ci95_pp": boot(lambda idx: diff[idx].mean())}
        )
    best = np.maximum(best_whitening, cos)
    for name, acc in (("always_cos", cos), ("always_wcos", wcos), ("always_lda", lda)):
        regret = 100 * (best - acc)
        out["fixed_policies"].append(
            {"policy": name, "regret_pp": float(regret.mean()), "ci95_pp": boot(lambda idx: regret[idx].mean())}
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print("Gap target         mean [95% dataset-cluster CI]   LODO Pearson")
    for r in out["gap_targets"]:
        print(f"  {r['target']:16s} {r['mean_pp']:6.3f} [{r['ci95_pp'][0]:6.3f}, {r['ci95_pp'][1]:6.3f}]   {r['lodo_pearson']:.3f}")
    print("Paired inflation")
    for r in out["paired_inflation"]:
        print(f"  {r['contrast']:28s} {r['mean_pp']:6.3f} [{r['ci95_pp'][0]:6.3f}, {r['ci95_pp'][1]:6.3f}]")
    print("Fixed-policy regret to per-group best of {Cos, WCos, LDA}")
    for r in out["fixed_policies"]:
        print(f"  {r['policy']:12s} {r['regret_pp']:5.2f} [{r['ci95_pp'][0]:5.2f}, {r['ci95_pp'][1]:5.2f}]")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
