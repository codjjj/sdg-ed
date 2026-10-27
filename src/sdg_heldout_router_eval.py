#!/usr/bin/env python3
"""Held-out dataset validation for the SDG-ED predictive router.

The main predictive protocol trains on the 9-dataset benchmark suite.  This
script keeps that training set fixed, computes the same training-only
diagnostics for new dataset-shot groups, and evaluates router/regret behavior
without refitting on the held-out datasets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from spectral_decision_gap_ed import load_features, sample_shot, sdgi_metrics
from sdg_stepb_predictive_protocol import (
    FEATURES,
    fit_predict_linear,
    load_rows,
    metric_bundle,
    routing_baselines,
    routing_eval,
    save_json,
)


RELEASE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("SDG_RESULTS_ROOT", RELEASE_ROOT / "artifacts/results"))
OUT_ROOT = Path(os.environ.get("SDG_OUTPUT_ROOT", RELEASE_ROOT / "outputs"))


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def covariance_diagnostics(x):
    xc = x - x.mean(0, keepdims=True)
    cov = (xc.T @ xc) / max(1, len(xc) - 1)
    vals = np.linalg.eigvalsh(cov)
    vals = np.maximum(vals, 0.0)
    positive = vals[vals > max(vals.max() * 1e-12, 1e-18)]
    if len(positive) == 0:
        condition = 0.0
        effective_rank = 0.0
    else:
        condition = float(positive.max() / positive.min())
        mass = positive / positive.sum()
        effective_rank = float(np.exp(-(mass * np.log(mass + 1e-12)).sum()))
    return {
        "log10_condition": float(np.log10(condition + 1.0)),
        "effective_rank": effective_rank,
    }


def shot_numeric(shot):
    return 999.0 if str(shot) == "full" else float(shot)


def mean_diag(values):
    out = {}
    for key in values[0].keys():
        out[key] = float(np.mean([v[key] for v in values]))
    return out


def diagnostics_for_group(dataset, shot, seeds):
    xtr_all, ytr_all, _, _ = load_features(dataset)
    rows = []
    for seed in seeds:
        xtr, ytr = sample_shot(xtr_all, ytr_all, str(shot), seed)
        row = {}
        row.update(sdgi_metrics(xtr, ytr))
        row.update(covariance_diagnostics(xtr))
        row["log10_n_train"] = float(np.log10(max(1, len(ytr))))
        row["shot_numeric"] = shot_numeric(shot)
        rows.append(row)
    return mean_diag(rows)


def heldout_rows(args):
    stepb_root = Path(args.out_root) / args.heldout_exp_name
    groups = read_json(stepb_root / "aggregate_groups.json")
    rows = []
    for r in groups:
        seeds = [0] if str(r["shot"]) == "full" else args.seeds
        row = dict(r)
        row.update(diagnostics_for_group(r["dataset"], r["shot"], seeds))
        row["target_best_whitening_minus_cos_pp"] = float(row["best_whitening_minus_cos_pp"])
        row["target_ridge_minus_cos_pp"] = float(row["ridge_minus_cos_pp"])
        rows.append(row)
    return rows


def predict_from_main(main_rows, target_key, eval_rows, features):
    x_train = np.array([[r[f] for f in features] for r in main_rows], dtype=np.float64)
    y_train = np.array([r[target_key] for r in main_rows], dtype=np.float64)
    x_eval = np.array([[r[f] for f in features] for r in eval_rows], dtype=np.float64)
    return fit_predict_linear(x_train, y_train, x_eval)


def summarize_by_dataset(rows, pred):
    groups = {}
    for r, p in zip(rows, pred):
        groups.setdefault(r["dataset"], []).append((r, p))
    out = []
    for dataset, items in sorted(groups.items()):
        target = np.array([r["target_best_whitening_minus_cos_pp"] for r, _ in items])
        pred_arr = np.array([p for _, p in items])
        regret = []
        for r, p in items:
            use_whitening = p >= 5.0
            chosen = r["best_whitening"] if use_whitening else r["cos"]
            regret.append((max(r["best_whitening"], r["cos"]) - chosen) * 100.0)
        out.append({
            "dataset": dataset,
            "n_groups": len(items),
            "target_mean_pp": float(target.mean()),
            "pred_mean_pp": float(pred_arr.mean()),
            "mae_pp": float(np.mean(np.abs(pred_arr - target))),
            "router_regret_pp": float(np.mean(regret)),
        })
    return out


def write_report(result, path: Path):
    lines = ["# Held-out Dataset Router Generalization\n"]
    lines.append(
        f"Main router training groups: {result['n_main_groups']}; held-out groups: {result['n_heldout_groups']}."
    )
    lines.append("The router is fit only on the original 9-dataset suite and evaluated on held-out datasets without refitting.")
    lines.append("\n## Best-whitening minus cosine target")
    m = result["best_whitening_target"]["metrics"]
    rr = result["best_whitening_target"]["routing"]
    lines.append(
        f"- Pearson={m['pearson']:.3f}, Spearman={m['spearman']:.3f}, "
        f"MAE={m['mae_pp']:.2f}pp, RMSE={m['rmse_pp']:.2f}pp."
    )
    lines.append(
        f"- 5pp router mean acc={rr['mean_chosen_acc']:.2f}, oracle={rr['mean_oracle_acc']:.2f}, "
        f"regret={rr['routing_regret_pp']:.2f}pp, routes-to-whitening={rr['route_to_whitening_frac']:.2f}."
    )
    lines.append("\n| dataset | groups | target mean | pred mean | MAE | router regret |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in result["by_dataset"]:
        lines.append(
            f"| {row['dataset']} | {row['n_groups']} | {row['target_mean_pp']:.2f} | "
            f"{row['pred_mean_pp']:.2f} | {row['mae_pp']:.2f} | {row['router_regret_pp']:.2f} |"
        )
    lines.append("\n## Trivial policies on held-out groups")
    lines.append("| policy | mean acc | regret |")
    lines.append("|---|---:|---:|")
    for row in result["routing_baselines"]:
        lines.append(f"| {row['policy']} | {row['mean_acc']:.2f} | {row['regret_pp']:.2f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(args):
    main_args = SimpleNamespace(
        stepb_root=args.main_stepb_root, results_root=args.results_root, out_root=args.out_root
    )
    main = load_rows(main_args)
    held = heldout_rows(args)

    pred = predict_from_main(main, "target_best_whitening_minus_cos_pp", held, FEATURES)
    target = np.array([r["target_best_whitening_minus_cos_pp"] for r in held], dtype=np.float64)
    ridge_pred = predict_from_main(main, "target_ridge_minus_cos_pp", held, FEATURES)
    ridge_target = np.array([r["target_ridge_minus_cos_pp"] for r in held], dtype=np.float64)

    result = {
        "n_main_groups": len(main),
        "n_heldout_groups": len(held),
        "features": FEATURES,
        "heldout_records": held,
        "best_whitening_target": {
            "metrics": metric_bundle(pred, target),
            "routing": routing_eval(held, pred, "target_best_whitening_minus_cos_pp", threshold=5.0),
            "predictions": pred.tolist(),
        },
        "ridge_reference_target": {
            "metrics": metric_bundle(ridge_pred, ridge_target),
            "predictions": ridge_pred.tolist(),
        },
        "routing_baselines": routing_baselines(held),
        "by_dataset": summarize_by_dataset(held, pred),
    }
    out_dir = Path(args.out_root) / args.exp_name
    save_json(result, out_dir / "heldout_router_generalization.json")
    write_report(result, out_dir / "heldout_router_generalization.md")
    print(f"[heldout-router] wrote {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root",
        default=str(OUT_ROOT),
        help="holds <heldout-exp-name>/aggregate_groups.json and receives <exp-name>/",
    )
    ap.add_argument("--results-root", default=str(ROOT), help="released main-suite aggregates")
    ap.add_argument("--main-stepb-root", default=str(ROOT / "stepb_cached_eval_v1"))
    ap.add_argument("--heldout-exp-name", default="heldout_stepb_eval_v1")
    ap.add_argument("--exp-name", default="heldout_router_v1")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
