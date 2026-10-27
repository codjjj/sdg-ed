#!/usr/bin/env python3
"""Estimator taxonomy and FeCAM defense experiments for SDG-ED."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spectral_decision_gap_ed import (
    OUT_ROOT,
    classes_from,
    fecam_diag,
    fecam_shrink,
    load_features,
    mahalanobis,
    ncm_cosine,
    sample_shot,
    save_json,
    whitened_cosine,
)


def pooled_stats(x, y, classes):
    d = x.shape[1]
    scatter = np.zeros((d, d), dtype=np.float64)
    n_total = 0
    means = []
    for c in classes:
        xc = x[y == c]
        mu = xc.mean(0)
        means.append(mu)
        centered = xc - mu
        scatter += centered.T @ centered
        n_total += len(xc)
    cov = scatter / max(1, n_total - len(classes))
    return np.stack(means), cov


def lda_from_cov(means, cov, xte, yte, classes, reg=1e-4):
    cov = cov + reg * np.eye(cov.shape[0])
    si = np.linalg.pinv(cov)
    ws = means @ si
    bias = -0.5 * np.sum(means * ws, axis=1)
    pred = np.argmax(xte @ ws.T + bias[None, :], axis=1)
    pred_lab = np.array([classes[i] for i in pred])
    return float(np.mean(pred_lab == yte))


def shared_diag_lda(means, cov, xte, yte, classes, reg=1e-4):
    return lda_from_cov(means, np.diag(np.diag(cov)), xte, yte, classes, reg)


def shrink_to_diag_lda(means, cov, xte, yte, classes, alpha, reg=1e-4):
    diag = np.diag(np.diag(cov))
    shrunk = (1.0 - alpha) * cov + alpha * diag
    return lda_from_cov(means, shrunk, xte, yte, classes, reg)


def lowrank_diag_cov(cov, rank):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    k = min(rank, len(vals))
    low = (vecs[:, :k] * vals[:k]) @ vecs[:, :k].T
    residual = np.maximum(np.diag(cov - low), 0.0)
    return low + np.diag(residual)


def cov_summary(cov):
    vals = np.linalg.eigvalsh(cov)
    vals = np.maximum(vals, 0.0)
    trace = float(vals.sum())
    vmax = float(vals.max())
    positive = vals[vals > max(vmax * 1e-10, 1e-12)]
    vmin = float(positive.min()) if len(positive) else 0.0
    p = vals / (trace + 1e-12)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {
        "trace": trace,
        "max_eig": vmax,
        "min_positive_eig": vmin,
        "condition_positive": float(vmax / (vmin + 1e-12)),
        "effective_rank": float(np.exp(entropy)),
        "rank_positive": int(len(positive)),
    }


def run_one(dataset, shot, seed, args):
    xtr_all, ytr_all, xte, yte = load_features(dataset)
    xtr, ytr = sample_shot(xtr_all, ytr_all, shot, seed)
    classes = classes_from(ytr, yte)
    print(f"[p3] dataset={dataset} shot={shot} seed={seed} train={xtr.shape} classes={len(classes)}")
    means, cov = pooled_stats(xtr, ytr, classes)
    metrics = {
        "cos": ncm_cosine(xtr, ytr, xte, yte, classes),
        "mahal_full": {},
        "shared_diag": shared_diag_lda(means, cov, xte, yte, classes, reg=args.reg),
        "shared_shrink": {},
        "lowrank_diag": {},
        "wcos_eps": {},
        "fecam_diag": fecam_diag(xtr, ytr, xte, yte, classes, reg=args.fecam_reg),
        "fecam_shrink": {},
    }
    for reg in args.mahal_regs:
        metrics["mahal_full"][str(reg)] = mahalanobis(xtr, ytr, xte, yte, classes, reg=reg)
    for alpha in args.shrink_alphas:
        metrics["shared_shrink"][str(alpha)] = shrink_to_diag_lda(means, cov, xte, yte, classes, alpha=alpha, reg=args.reg)
        metrics["fecam_shrink"][str(alpha)] = fecam_shrink(xtr, ytr, xte, yte, classes, alpha=alpha, reg=args.fecam_reg)
    for rank in args.lowrank_ranks:
        metrics["lowrank_diag"][str(rank)] = lda_from_cov(means, lowrank_diag_cov(cov, rank), xte, yte, classes, reg=args.reg)
    for eps in args.wcos_eps:
        metrics["wcos_eps"][str(eps)] = whitened_cosine(xtr, ytr, xte, yte, classes, eps=eps)
    metrics["mahal_best"] = max(metrics["mahal_full"].values())
    metrics["shared_shrink_best"] = max(metrics["shared_shrink"].values())
    metrics["fecam_shrink_best"] = max(metrics["fecam_shrink"].values())
    metrics["lowrank_diag_best"] = max(metrics["lowrank_diag"].values())
    metrics["wcos_best"] = max(metrics["wcos_eps"].values())
    return {
        "dataset": dataset,
        "shot": shot,
        "seed": int(seed),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_classes": int(len(classes)),
        "covariance": cov_summary(cov),
        "metrics": metrics,
    }


def run(args):
    seeds = [0] if args.shot == "full" else args.seeds
    records = [run_one(args.dataset, args.shot, seed, args) for seed in seeds]
    out_dir = Path(args.out_dir) / "p3_estimator_defense" / args.dataset / f"shot_{args.shot}"
    save_json({"dataset": args.dataset, "shot": args.shot, "records": records}, out_dir / "result.json")
    write_report(records, out_dir / "result.md")
    print(f"[p3] wrote {out_dir}")


def mean_std(vals):
    arr = np.array(vals, dtype=float) * 100
    if len(arr) == 1:
        return f"{arr.mean():.2f}"
    return f"{arr.mean():.2f}±{arr.std(ddof=1):.2f}"


def write_report(records, path: Path):
    lines = [f"# P3 Estimator Defense: {records[0]['dataset']} shot={records[0]['shot']}\n"]
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    keys = ["cos", "mahal_best", "shared_diag", "shared_shrink_best", "lowrank_diag_best", "wcos_best", "fecam_diag", "fecam_shrink_best"]
    for k in keys:
        lines.append(f"| {k} | {mean_std([r['metrics'][k] for r in records])} |")
    lines.append(f"| condition_positive | {np.mean([r['covariance']['condition_positive'] for r in records]):.3e} |")
    lines.append(f"| effective_rank | {np.mean([r['covariance']['effective_rank'] for r in records]):.2f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def aggregate(args):
    root = Path(args.out_dir) / "p3_estimator_defense"
    rows = []
    for p in sorted(root.glob("*/*/result.json")):
        obj = json.load(open(p))
        for r in obj["records"]:
            rows.append(r)
    save_json(rows, root / "aggregate.json")
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["shot"]), []).append(r)
    lines = ["# P3 Estimator Defense Aggregate\n"]
    lines.append("| dataset | shot | n | Cos | Mahal | Shared diag | Shared shrink | Lowrank+diag | WCos | FeCAM diag | FeCAM shrink | cond | eff rank |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    order = {"1": 1, "2": 2, "5": 5, "10": 10, "20": 20, "full": 999}
    for (dataset, shot), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 1000))):
        lines.append(
            f"| {dataset} | {shot} | {len(rs)} | "
            f"{mean_std([r['metrics']['cos'] for r in rs])} | "
            f"{mean_std([r['metrics']['mahal_best'] for r in rs])} | "
            f"{mean_std([r['metrics']['shared_diag'] for r in rs])} | "
            f"{mean_std([r['metrics']['shared_shrink_best'] for r in rs])} | "
            f"{mean_std([r['metrics']['lowrank_diag_best'] for r in rs])} | "
            f"{mean_std([r['metrics']['wcos_best'] for r in rs])} | "
            f"{mean_std([r['metrics']['fecam_diag'] for r in rs])} | "
            f"{mean_std([r['metrics']['fecam_shrink_best'] for r in rs])} | "
            f"{np.mean([r['covariance']['condition_positive'] for r in rs]):.2e} | "
            f"{np.mean([r['covariance']['effective_rank'] for r in rs]):.1f} |"
        )
    (root / "aggregate.md").write_text("\n".join(lines) + "\n")
    print(f"[p3-aggregate] rows={len(rows)} -> {root / 'aggregate.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "aggregate"], required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--shot", default="full")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--mahal-regs", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    ap.add_argument("--shrink-alphas", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    ap.add_argument("--lowrank-ranks", nargs="+", type=int, default=[16, 64, 128, 256])
    ap.add_argument("--wcos-eps", nargs="+", type=float, default=[1e-6, 1e-4, 1e-3, 1e-2, 1e-1])
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--fecam-reg", type=float, default=1e-3)
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    args = ap.parse_args()
    if args.mode == "run":
        if args.dataset is None:
            raise ValueError("--dataset required")
        run(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
