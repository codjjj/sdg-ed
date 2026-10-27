#!/usr/bin/env python3
"""Dataset-cluster 95% bootstrap intervals for every cell of Tables 3 and 4.

Inputs are the reproduced predictive-protocol outputs (joined_records.json and
predictive_results.json), which `scripts/run_predictive_protocol_from_artifacts.sh`
regenerates from the released aggregates; a copy is bundled under
artifacts/results/stepb_predictive_protocol_reproduced/.

Each bootstrap draw resamples the nine datasets with replacement and keeps all
six shot groups of every drawn dataset (10,000 draws). Router metrics use seed
20260720 and the fixed policies seed 20260724, matching the published
intervals. Runs on CPU in about a minute.

    python scripts/recompute/table34_dataset_cluster_ci.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / "src"))

from sdg_stepb_predictive_protocol import metric_bundle, routing_arrays  # noqa: E402

DEFAULT_PROTOCOL_DIR = RELEASE_ROOT / "artifacts/results/stepb_predictive_protocol_reproduced"
DEFAULT_OUTPUT = RELEASE_ROOT / "outputs/recompute/table34_dataset_cluster_ci.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--router-seed", type=int, default=20260720)
    parser.add_argument("--policy-seed", type=int, default=20260724)
    args = parser.parse_args()

    joined = json.loads((args.protocol_dir / "joined_records.json").read_text())
    results = json.loads((args.protocol_dir / "predictive_results.json").read_text())
    key = {(r["dataset"], str(r["shot"])): i for i, r in enumerate(joined)}
    datasets = np.array([r["dataset"] for r in joined])
    groups = [np.where(datasets == name)[0] for name in sorted(set(datasets))]

    def boot(stat, seed: int) -> list[float]:
        rng = np.random.RandomState(seed)
        values = []
        for _ in range(args.n_boot):
            drawn = rng.randint(0, len(groups), size=len(groups))
            values.append(stat(np.concatenate([groups[i] for i in drawn])))
        return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]

    out = {"table3": [], "table4": []}
    predictions = {}
    for result in results:
        pred = np.empty(len(joined))
        target = np.empty(len(joined))
        for p in result["predictions"]:
            i = key[(p["dataset"], str(p["shot"]))]
            pred[i] = p["pred_pp"]
            target[i] = p["target_pp"]
        predictions[(result["target"], result["fold"])] = pred
        point = metric_bundle(pred, target)
        out["table3"].append(
            {
                "target": result["target"],
                "fold": result["fold"],
                "pearson": point["pearson"],
                "pearson_ci": boot(lambda idx: metric_bundle(pred[idx], target[idx])["pearson"], args.router_seed),
                "spearman": point["spearman"],
                "spearman_ci": boot(lambda idx: metric_bundle(pred[idx], target[idx])["spearman"], args.router_seed),
                "mae_pp": point["mae_pp"],
                "rmse_pp": point["rmse_pp"],
            }
        )

    cos = np.array([r["cos"] for r in joined])
    wcos = np.array([r["wcos_trainval"] for r in joined])
    lda = np.array([r["lda_trainval"] for r in joined])
    best = np.array([max(r["best_whitening"], r["cos"]) for r in joined])

    def policy(name: str, acc: np.ndarray) -> None:
        out["table4"].append(
            {
                "policy": name,
                "mean_acc": float(acc.mean() * 100),
                "regret_pp": float((best.mean() - acc.mean()) * 100),
                "mean_acc_ci": boot(lambda idx: acc[idx].mean() * 100, args.policy_seed),
                "regret_ci": boot(lambda idx: (best[idx].mean() - acc[idx].mean()) * 100, args.policy_seed),
            }
        )

    for fold, label in (("leave_one_group", "LOGO router, 5pp"), ("leave_one_dataset", "LODO router, 5pp")):
        chosen, _, _ = routing_arrays(joined, predictions[("target_best_whitening_minus_cos_pp", fold)], 5.0)
        policy(label, chosen)
    policy("always cosine", cos)
    policy("always WCos", wcos)
    policy("always LDA", lda)
    policy("oracle best of cosine/whitening", best)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print("Table 3: router prediction (Pearson / Spearman with 95% dataset-cluster CIs)")
    for r in out["table3"]:
        print(
            f"  {r['target'][7:]:28s} {r['fold']:18s} r={r['pearson']:.3f} "
            f"[{r['pearson_ci'][0]:.3f}, {r['pearson_ci'][1]:.3f}]  rho={r['spearman']:.3f} "
            f"[{r['spearman_ci'][0]:.3f}, {r['spearman_ci'][1]:.3f}]"
        )
    print("Table 4: routing policies (mean accuracy %, regret pp)")
    for r in out["table4"]:
        print(
            f"  {r['policy']:34s} acc={r['mean_acc']:.2f} [{r['mean_acc_ci'][0]:.2f}, {r['mean_acc_ci'][1]:.2f}]  "
            f"regret={r['regret_pp']:.2f} [{r['regret_ci'][0]:.2f}, {r['regret_ci'][1]:.2f}]"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
