#!/usr/bin/env python3
"""Leave-one-group-out predictive protocol for SDG-ED Step B.

It tests the paper's exploratory prediction claim (Tables 3-4, Appendix C):
can training-only spectral/covariance diagnostics warn that raw cosine will
disagree with whitening-aware or reference evaluators?
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("SDG_RESULTS_ROOT", RELEASE_ROOT / "artifacts/results"))
FEATURES = [
    "tail128_fisher",
    "tail256_fisher",
    "bottom_half_fisher",
    "spearman_logeig_logfisher",
    "log10_condition",
    "effective_rank",
    "log10_n_train",
    "shot_numeric",
]

ABLATION_FEATURE_SETS = [
    ("full", FEATURES),
    ("spectral_only", ["tail128_fisher", "tail256_fisher", "bottom_half_fisher", "spearman_logeig_logfisher", "top10_var", "top50_var"]),
    ("covariance_only", ["log10_condition", "effective_rank", "log10_n_train", "shot_numeric"]),
    ("variance_only", ["top10_var", "top50_var"]),
    ("tail_fisher_only", ["tail128_fisher", "tail256_fisher", "bottom_half_fisher"]),
    ("tail128_only", ["tail128_fisher"]),
    ("condition_only", ["log10_condition"]),
    ("no_tail_fisher", ["spearman_logeig_logfisher", "log10_condition", "effective_rank", "log10_n_train", "shot_numeric"]),
    ("no_covariance", ["tail128_fisher", "tail256_fisher", "bottom_half_fisher", "spearman_logeig_logfisher", "log10_n_train", "shot_numeric"]),
    ("no_sample_regime", ["tail128_fisher", "tail256_fisher", "bottom_half_fisher", "spearman_logeig_logfisher", "log10_condition", "effective_rank"]),
]


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


def mean_group(rows, key_fn):
    groups = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(r)
    return groups


def shot_numeric(shot):
    return 999.0 if str(shot) == "full" else float(shot)


def load_rows(args):
    results_root = Path(args.results_root)
    stepb = read_json(Path(args.stepb_root) / "aggregate_groups.json")
    method_rows = read_json(results_root / "method_matrix_aggregate.json")
    p3_rows = read_json(results_root / "p3_estimator_defense/aggregate.json")

    method_groups = mean_group(method_rows, lambda r: (r["dataset"], str(r["shot"])))
    p3_groups = mean_group(p3_rows, lambda r: (r["dataset"], str(r["shot"])))
    joined = []
    for r in stepb:
        key = (r["dataset"], str(r["shot"]))
        ms = method_groups[key]
        ps = p3_groups[key]
        sdgi = {
            "tail128_fisher": float(np.mean([m["sdgi"]["tail128_fisher"] for m in ms])),
            "tail256_fisher": float(np.mean([m["sdgi"]["tail256_fisher"] for m in ms])),
            "bottom_half_fisher": float(np.mean([m["sdgi"]["bottom_half_fisher"] for m in ms])),
            "spearman_logeig_logfisher": float(np.mean([m["sdgi"]["spearman_logeig_logfisher"] for m in ms])),
            "top10_var": float(np.mean([m["sdgi"]["top10_var"] for m in ms])),
            "top50_var": float(np.mean([m["sdgi"]["top50_var"] for m in ms])),
        }
        cov = {
            "log10_condition": float(np.mean([np.log10(p["covariance"]["condition_positive"] + 1.0) for p in ps])),
            "effective_rank": float(np.mean([p["covariance"]["effective_rank"] for p in ps])),
        }
        row = dict(r)
        row.update(sdgi)
        row.update(cov)
        row["shot_numeric"] = shot_numeric(r["shot"])
        row["log10_n_train"] = float(np.log10(max(1, np.mean([m["n_train"] for m in ms]))))
        row["target_best_whitening_minus_cos_pp"] = float(r["best_whitening_minus_cos_pp"])
        row["target_ridge_minus_cos_pp"] = float(r["ridge_minus_cos_pp"])
        joined.append(row)
    return joined


def rankdata(a):
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan")
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return float(np.mean(x * y))


def metric_bundle(pred, y):
    return {
        "pearson": safe_corr(pred, y),
        "spearman": safe_corr(rankdata(pred), rankdata(y)),
        "mae_pp": float(np.mean(np.abs(pred - y))),
        "rmse_pp": float(np.sqrt(np.mean((pred - y) ** 2))),
    }


def bootstrap_ci(pred, y, n_boot=2000, seed=0):
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.RandomState(seed)
    values = {k: [] for k in ["pearson", "spearman", "mae_pp", "rmse_pp"]}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        m = metric_bundle(pred[idx], y[idx])
        for k, v in m.items():
            values[k].append(v)
    return {
        k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        for k, v in values.items()
    }


def routing_arrays(rows, pred, threshold):
    chosen = []
    oracle = []
    route_to_whitening = []
    for r, p in zip(rows, pred):
        use_whitening = p >= threshold
        chosen_acc = r["best_whitening"] if use_whitening else r["cos"]
        oracle_acc = max(r["best_whitening"], r["cos"])
        chosen.append(chosen_acc)
        oracle.append(oracle_acc)
        route_to_whitening.append(use_whitening)
    return (
        np.asarray(chosen, dtype=np.float64),
        np.asarray(oracle, dtype=np.float64),
        np.asarray(route_to_whitening, dtype=np.float64),
    )


def routing_ci(chosen, oracle, route_to_whitening=None, n_boot=2000, seed=1):
    rng = np.random.RandomState(seed)
    n = len(chosen)
    values = {
        "mean_chosen_acc": [],
        "mean_oracle_acc": [],
        "routing_regret_pp": [],
    }
    if route_to_whitening is not None:
        values["route_to_whitening_frac"] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        chosen_m = chosen[idx].mean()
        oracle_m = oracle[idx].mean()
        values["mean_chosen_acc"].append(chosen_m * 100.0)
        values["mean_oracle_acc"].append(oracle_m * 100.0)
        values["routing_regret_pp"].append((oracle_m - chosen_m) * 100.0)
        if route_to_whitening is not None:
            values["route_to_whitening_frac"].append(route_to_whitening[idx].mean())
    return {
        k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        for k, v in values.items()
    }


def fit_predict_linear(train_x, train_y, test_x, alpha=1.0):
    mu = train_x.mean(0)
    std = train_x.std(0) + 1e-12
    x = (train_x - mu) / std
    xt = (test_x - mu) / std
    y = train_y - train_y.mean()
    a = alpha * np.eye(x.shape[1])
    w = np.linalg.solve(x.T @ x + a, x.T @ y)
    return xt @ w + train_y.mean()


def predict_cv(rows, target_key, fold_key, features=FEATURES):
    x = np.array([[r[f] for f in features] for r in rows], dtype=np.float64)
    y = np.array([r[target_key] for r in rows], dtype=np.float64)
    folds = [fold_key(r) for r in rows]
    pred = np.zeros_like(y)
    for fold in sorted(set(folds)):
        test = np.array([f == fold for f in folds])
        train = ~test
        pred[test] = fit_predict_linear(x[train], y[train], x[test])
    return pred, y


def routing_eval(rows, pred, target_key, threshold):
    chosen, oracle, route_to_whitening = routing_arrays(rows, pred, threshold)
    return {
        "threshold": float(threshold),
        "mean_chosen_acc": float(np.mean(chosen) * 100.0),
        "mean_oracle_acc": float(np.mean(oracle) * 100.0),
        "routing_regret_pp": float((np.mean(oracle) - np.mean(chosen)) * 100.0),
        "route_to_whitening_frac": float(np.mean(pred >= threshold)),
        "ci95": routing_ci(chosen, oracle, route_to_whitening, seed=int(1000 + threshold * 37)),
    }


def routing_baselines(rows):
    cos = np.array([r["cos"] for r in rows], dtype=np.float64)
    wcos = np.array([r["wcos_trainval"] for r in rows], dtype=np.float64)
    lda = np.array([r["lda_trainval"] for r in rows], dtype=np.float64)
    best = np.array([max(r["best_whitening"], r["cos"]) for r in rows], dtype=np.float64)

    def row(policy, acc, seed):
        ci = routing_ci(acc, best, seed=seed)
        return {
            "policy": policy,
            "mean_acc": float(acc.mean() * 100.0),
            "regret_pp": float((best.mean() - acc.mean()) * 100.0),
            "ci95": {
                "mean_acc": ci["mean_chosen_acc"],
                "regret_pp": ci["routing_regret_pp"],
            },
        }

    return [
        row("always cosine", cos, 11),
        row("always WCos", wcos, 12),
        row("always LDA", lda, 13),
        row("oracle best of cosine/whitening", best, 14),
    ]


def summarize_prediction(rows, target_key, fold_name, fold_key, features=FEATURES):
    pred, y = predict_cv(rows, target_key, fold_key, features=features)
    metrics = metric_bundle(pred, y)
    out = {
        "target": target_key,
        "fold": fold_name,
        "n": len(rows),
        "unit": "dataset-shot groups; non-full groups are 5-seed means",
        **metrics,
        "ci95": bootstrap_ci(pred, y),
        "predictions": [
            {
                "dataset": r["dataset"],
                "shot": r["shot"],
                "target_pp": float(t),
                "pred_pp": float(p),
            }
            for r, t, p in zip(rows, y, pred)
        ],
    }
    if target_key == "target_best_whitening_minus_cos_pp":
        out["routing"] = [routing_eval(rows, pred, target_key, t) for t in [0.0, 2.0, 5.0, 10.0]]
        out["routing_baselines"] = routing_baselines(rows)
    return out


def run_feature_ablation(rows):
    results = []
    for name, features in ABLATION_FEATURE_SETS:
        for target in ["target_best_whitening_minus_cos_pp", "target_ridge_minus_cos_pp"]:
            for fold_name, fold_key in [
                ("leave_one_group", lambda r: (r["dataset"], r["shot"])),
                ("leave_one_dataset", lambda r: r["dataset"]),
            ]:
                result = summarize_prediction(rows, target, fold_name, fold_key, features=features)
                result["feature_set"] = name
                result["features"] = features
                if target == "target_best_whitening_minus_cos_pp":
                    route5 = next(r for r in result["routing"] if abs(r["threshold"] - 5.0) < 1e-9)
                    result["routing_regret_5pp"] = route5["routing_regret_pp"]
                    result["routing_acc_5pp"] = route5["mean_chosen_acc"]
                result.pop("predictions", None)
                results.append(result)
    return results


def write_ablation_report(results, path: Path):
    lines = ["# Predictive Router Feature Ablation\n"]
    lines.append("All rows reuse the 54 dataset-shot groups from the train-only protocol. The `full` row is the original diagnostic set used in the main paper; the other rows ablate diagnostic families without changing the fold protocol.")
    lines.append("")
    lines.append("## Best-whitening minus cosine target")
    lines.append("")
    lines.append("| feature set | fold | Pearson | Spearman | MAE pp | RMSE pp | 5pp router regret |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in results:
        if row["target"] != "target_best_whitening_minus_cos_pp":
            continue
        lines.append(
            f"| {row['feature_set']} | {row['fold']} | {row['pearson']:.3f} | {row['spearman']:.3f} | "
            f"{row['mae_pp']:.2f} | {row['rmse_pp']:.2f} | {row['routing_regret_5pp']:.2f} |"
        )
    lines.append("")
    lines.append("## Ridge-reference minus cosine target")
    lines.append("")
    lines.append("| feature set | fold | Pearson | Spearman | MAE pp | RMSE pp |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in results:
        if row["target"] != "target_ridge_minus_cos_pp":
            continue
        lines.append(
            f"| {row['feature_set']} | {row['fold']} | {row['pearson']:.3f} | {row['spearman']:.3f} | "
            f"{row['mae_pp']:.2f} | {row['rmse_pp']:.2f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_report(results, path: Path):
    lines = ["# Step B Predictive Protocol v1\n"]
    lines.append("Features are computed from training-only spectral/covariance diagnostics; targets come from Step B non-oracle cached evaluation.")
    lines.append("The evaluation unit is a dataset-shot group (`n=54` = 9 datasets x 6 shot regimes); non-full groups are 5-seed means.")
    lines.append("")
    lines.append("| target | fold | n | Pearson [95% CI] | Spearman [95% CI] | MAE pp [95% CI] | RMSE pp [95% CI] |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        ci = r["ci95"]
        lines.append(
            f"| {r['target']} | {r['fold']} | {r['n']} | "
            f"{r['pearson']:.3f} [{ci['pearson'][0]:.3f}, {ci['pearson'][1]:.3f}] | "
            f"{r['spearman']:.3f} [{ci['spearman'][0]:.3f}, {ci['spearman'][1]:.3f}] | "
            f"{r['mae_pp']:.2f} [{ci['mae_pp'][0]:.2f}, {ci['mae_pp'][1]:.2f}] | "
            f"{r['rmse_pp']:.2f} [{ci['rmse_pp'][0]:.2f}, {ci['rmse_pp'][1]:.2f}] |"
        )
    lines.append("\n## Routing")
    for r in results:
        if "routing" not in r:
            continue
        lines.append(f"\n### {r['target']} / {r['fold']}")
        lines.append("| threshold | chosen acc [95% CI] | oracle acc [95% CI] | regret pp [95% CI] | whitening frac [95% CI] |")
        lines.append("|---:|---:|---:|---:|---:|")
        for row in r["routing"]:
            ci = row["ci95"]
            lines.append(
                f"| {row['threshold']:.1f} | "
                f"{row['mean_chosen_acc']:.2f} [{ci['mean_chosen_acc'][0]:.2f}, {ci['mean_chosen_acc'][1]:.2f}] | "
                f"{row['mean_oracle_acc']:.2f} [{ci['mean_oracle_acc'][0]:.2f}, {ci['mean_oracle_acc'][1]:.2f}] | "
                f"{row['routing_regret_pp']:.2f} [{ci['routing_regret_pp'][0]:.2f}, {ci['routing_regret_pp'][1]:.2f}] | "
                f"{row['route_to_whitening_frac']:.2f} [{ci['route_to_whitening_frac'][0]:.2f}, {ci['route_to_whitening_frac'][1]:.2f}] |"
            )
        lines.append("\nTrivial baselines:")
        lines.append("| policy | mean acc [95% CI] | regret pp [95% CI] |")
        lines.append("|---|---:|---:|")
        for row in r["routing_baselines"]:
            ci = row["ci95"]
            lines.append(
                f"| {row['policy']} | {row['mean_acc']:.2f} [{ci['mean_acc'][0]:.2f}, {ci['mean_acc'][1]:.2f}] | "
                f"{row['regret_pp']:.2f} [{ci['regret_pp'][0]:.2f}, {ci['regret_pp'][1]:.2f}] |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(args):
    rows = load_rows(args)
    results = []
    for target in ["target_best_whitening_minus_cos_pp", "target_ridge_minus_cos_pp"]:
        results.append(summarize_prediction(rows, target, "leave_one_group", lambda r: (r["dataset"], r["shot"])))
        results.append(summarize_prediction(rows, target, "leave_one_dataset", lambda r: r["dataset"]))
    out_dir = Path(args.out_root) / args.exp_name
    save_json(rows, out_dir / "joined_records.json")
    save_json(results, out_dir / "predictive_results.json")
    write_report(results, out_dir / "summary.md")
    if args.ablation:
        ablation = run_feature_ablation(rows)
        save_json(ablation, out_dir / "feature_ablation.json")
        write_ablation_report(ablation, out_dir / "feature_ablation.md")
    print(f"[predictive] wrote {out_dir / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-root",
        default=os.environ.get("SDG_OUTPUT_ROOT", str(RELEASE_ROOT / "outputs")),
        help="outputs go to <out-root>/<exp-name>/ (never into the released artifacts by default)",
    )
    ap.add_argument("--results-root", default=str(ROOT))
    ap.add_argument("--stepb-root", default=str(ROOT / "stepb_cached_eval_v1"))
    ap.add_argument("--exp-name", default="stepb_predictive_protocol_v1")
    ap.add_argument("--ablation", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
