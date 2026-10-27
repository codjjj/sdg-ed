#!/usr/bin/env python3
"""Supplementary stats-only experiments for the SDG-ED paper.

This script reads cached frozen features and expands the evidence around three
review-sensitive questions:
  1. Does the shared-covariance gain persist at higher shots?
  2. Is it stable to dense covariance / whitening regularization sweeps?
  3. Which PCA frequency bands carry the decision gap?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spectral_decision_gap_ed import (
    OUT_ROOT,
    classes_from,
    covariance_pca,
    fisher_by_pc,
    load_features,
    ncm_cosine,
    sample_shot,
    save_json,
    spearman,
)


def pooled_stats(x, y, classes):
    d = x.shape[1]
    scatter = np.zeros((d, d), dtype=np.float64)
    n_total = 0
    means = []
    for c in classes:
        xc = x[y == c]
        mu = xc.mean(0)
        centered = xc - mu
        scatter += centered.T @ centered
        means.append(mu)
        n_total += len(xc)
    cov = scatter / max(1, n_total - len(classes))
    return np.stack(means), cov


def lda_acc_from_cov(means, cov, xte, yte, classes, reg):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    inv_cov = (vecs * (1.0 / (vals + reg))) @ vecs.T
    weights = means @ inv_cov
    bias = -0.5 * np.sum(means * weights, axis=1)
    pred = np.argmax(xte @ weights.T + bias[None, :], axis=1)
    pred_lab = np.array([classes[i] for i in pred])
    return float(np.mean(pred_lab == yte))


def mahal_sweep(xtr, ytr, xte, yte, classes, regs):
    means, cov = pooled_stats(xtr, ytr, classes)
    by_reg = {str(reg): lda_acc_from_cov(means, cov, xte, yte, classes, reg) for reg in regs}
    best_reg, best_acc = max(by_reg.items(), key=lambda kv: kv[1])
    return {
        "by_reg": by_reg,
        "best_reg": float(best_reg),
        "best_acc": float(best_acc),
    }, cov, means


def shared_diag_acc(means, cov, xte, yte, classes, reg):
    return lda_acc_from_cov(means, np.diag(np.diag(cov)), xte, yte, classes, reg)


def shrink_sweep(means, cov, xte, yte, classes, alphas, reg):
    diag = np.diag(np.diag(cov))
    rows = {}
    for alpha in alphas:
        shrunk = (1.0 - alpha) * cov + alpha * diag
        rows[str(alpha)] = lda_acc_from_cov(means, shrunk, xte, yte, classes, reg)
    best_alpha, best_acc = max(rows.items(), key=lambda kv: kv[1])
    return {"by_alpha": rows, "best_alpha": float(best_alpha), "best_acc": float(best_acc)}


def wcos_sweep(xtr, ytr, xte, yte, classes, eps_values):
    mean, eigvals, pcs = covariance_pca(xtr)
    ztr = (xtr - mean) @ pcs
    zte = (xte - mean) @ pcs
    rows = {}
    for eps in eps_values:
        scale = 1.0 / np.sqrt(np.maximum(eigvals, 0.0) + eps)
        rows[str(eps)] = ncm_cosine(ztr * scale[None, :], ytr, zte * scale[None, :], yte, classes)
    best_eps, best_acc = max(rows.items(), key=lambda kv: kv[1])
    return {"by_eps": rows, "best_eps": float(best_eps), "best_acc": float(best_acc)}


def band_indices(dim):
    spans = [
        ("all", 0, dim),
        ("top8", 0, min(8, dim)),
        ("top16", 0, min(16, dim)),
        ("top32", 0, min(32, dim)),
        ("top64", 0, min(64, dim)),
        ("top128", 0, min(128, dim)),
        ("top256", 0, min(256, dim)),
        ("top512", 0, min(512, dim)),
        ("band0_16", 0, min(16, dim)),
        ("band16_32", min(16, dim), min(32, dim)),
        ("band32_64", min(32, dim), min(64, dim)),
        ("band64_128", min(64, dim), min(128, dim)),
        ("band128_256", min(128, dim), min(256, dim)),
        ("band256_512", min(256, dim), min(512, dim)),
        ("band512_end", min(512, dim), dim),
        ("tail128", max(0, dim - 128), dim),
        ("tail256", max(0, dim - 256), dim),
        ("drop_top16", min(16, dim), dim),
        ("drop_top64", min(64, dim), dim),
        ("drop_tail128", 0, max(1, dim - 128)),
    ]
    return [(name, np.arange(lo, hi)) for name, lo, hi in spans if hi > lo]


def pca_band_sweep(xtr, ytr, xte, yte, classes, regs):
    mean, eigvals, pcs = covariance_pca(xtr)
    ztr = (xtr - mean) @ pcs
    zte = (xte - mean) @ pcs
    rows = []
    for name, idx in band_indices(pcs.shape[1]):
        xb_tr = ztr[:, idx]
        xb_te = zte[:, idx]
        cos = ncm_cosine(xb_tr, ytr, xb_te, yte, classes)
        mah, _, _ = mahal_sweep(xb_tr, ytr, xb_te, yte, classes, regs)
        rows.append({
            "band": name,
            "dim": int(len(idx)),
            "cos": cos,
            "mahal_best": mah["best_acc"],
            "mahal_best_reg": mah["best_reg"],
            "gap_pp": 100.0 * (mah["best_acc"] - cos),
            "var_mass": float(eigvals[idx].sum() / (eigvals.sum() + 1e-12)),
        })
        print(
            f"[band] {name:<12} d={len(idx):>3} cos={cos*100:.2f} "
            f"mahal={mah['best_acc']*100:.2f} gap={100*(mah['best_acc']-cos):+.2f}"
        )
    return rows, eigvals


def sdgi_panel(xtr, ytr):
    mean, eigvals, pcs = covariance_pca(xtr)
    fisher, mass = fisher_by_pc(xtr, ytr, mean, pcs)
    eig_log = np.log(np.maximum(eigvals, 1e-18))
    fisher_log = np.log(np.maximum(fisher, 1e-18))
    total = eigvals.sum() + 1e-12
    cum = np.cumsum(mass)
    median_pc = int(np.searchsorted(cum, 0.5) + 1)

    def tail(k):
        if k >= len(eigvals):
            return 0.0, 0.0, 0.0
        fisher_tail = float(mass[k:].sum())
        var_tail = float(eigvals[k:].sum() / total)
        return fisher_tail, var_tail, fisher_tail / (var_tail + 1e-12)

    tail64_f, tail64_v, tail64_e = tail(64)
    tail128_f, tail128_v, tail128_e = tail(128)
    tail256_f, tail256_v, tail256_e = tail(256)
    half = len(eigvals) // 2
    half_f, half_v, half_e = tail(half)
    corr = float(np.mean(
        ((eig_log - eig_log.mean()) / (eig_log.std() + 1e-12))
        * ((fisher_log - fisher_log.mean()) / (fisher_log.std() + 1e-12))
    ))
    return {
        "dim": int(len(eigvals)),
        "top1_var": float(eigvals[:1].sum() / total),
        "top5_var": float(eigvals[:5].sum() / total),
        "top10_var": float(eigvals[:10].sum() / total),
        "top50_var": float(eigvals[:50].sum() / total),
        "top128_var": float(eigvals[:128].sum() / total),
        "tail64_fisher": tail64_f,
        "tail64_var": tail64_v,
        "tail64_enrichment": tail64_e,
        "tail128_fisher": tail128_f,
        "tail128_var": tail128_v,
        "tail128_enrichment": tail128_e,
        "tail256_fisher": tail256_f,
        "tail256_var": tail256_v,
        "tail256_enrichment": tail256_e,
        "bottom_half_fisher": half_f,
        "bottom_half_var": half_v,
        "bottom_half_enrichment": half_e,
        "fisher_median_pc": median_pc,
        "fisher_argmax_pc": int(np.argmax(fisher) + 1),
        "spearman_logeig_logfisher": spearman(eig_log, fisher_log),
        "corr_logeig_logfisher": corr,
    }


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


def write_report(result, path: Path):
    m = result["metrics"]
    lines = [f"# SDG Supplement Stats: {result['dataset']} shot={result['shot']} seed={result['seed']}\n"]
    lines.append(f"- train/test/classes: {result['n_train']} / {result['n_test']} / {result['n_classes']}")
    lines.append(f"- Cos: {m['cos']*100:.2f}")
    lines.append(f"- Mahal best: {m['mahal']['best_acc']*100:.2f} @ reg={m['mahal']['best_reg']}")
    lines.append(f"- WCos best: {m['wcos']['best_acc']*100:.2f} @ eps={m['wcos']['best_eps']}")
    lines.append(f"- Shared shrink best: {m['shared_shrink']['best_acc']*100:.2f} @ alpha={m['shared_shrink']['best_alpha']}")
    lines.append(f"- Gap(best): {100*(m['mahal']['best_acc']-m['cos']):+.2f} pp")
    lines.append("\n## SDGI")
    for key in [
        "tail128_fisher",
        "tail128_enrichment",
        "bottom_half_fisher",
        "fisher_median_pc",
        "spearman_logeig_logfisher",
        "corr_logeig_logfisher",
    ]:
        val = result["sdgi"][key]
        lines.append(f"- {key}: {val:.6f}" if isinstance(val, float) else f"- {key}: {val}")
    lines.append("\n## PCA Bands")
    lines.append("| band | dim | var mass | Cos | Mahal(best) | Gap |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in result["pca_bands"]:
        lines.append(
            f"| {row['band']} | {row['dim']} | {row['var_mass']*100:.2f} | "
            f"{row['cos']*100:.2f} | {row['mahal_best']*100:.2f} | {row['gap_pp']:.2f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(args):
    xtr_all, ytr_all, xte, yte = load_features(args.dataset)
    xtr, ytr = sample_shot(xtr_all, ytr_all, args.shot, args.seed)
    classes = classes_from(ytr, yte)
    print(
        f"[supp] dataset={args.dataset} shot={args.shot} seed={args.seed} "
        f"train={xtr.shape} test={xte.shape} classes={len(classes)}"
    )

    cos = ncm_cosine(xtr, ytr, xte, yte, classes)
    mahal, cov, means = mahal_sweep(xtr, ytr, xte, yte, classes, args.mahal_regs)
    result = {
        "dataset": args.dataset,
        "shot": args.shot,
        "seed": int(args.seed),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_classes": int(len(classes)),
        "metrics": {
            "cos": cos,
            "mahal": mahal,
            "shared_diag": shared_diag_acc(means, cov, xte, yte, classes, args.diag_reg),
            "shared_shrink": shrink_sweep(means, cov, xte, yte, classes, args.shrink_alphas, args.diag_reg),
            "wcos": wcos_sweep(xtr, ytr, xte, yte, classes, args.wcos_eps),
        },
        "covariance": cov_summary(cov),
        "sdgi": sdgi_panel(xtr, ytr),
    }
    result["metrics"]["gap_best_pp"] = 100.0 * (result["metrics"]["mahal"]["best_acc"] - cos)
    bands, _ = pca_band_sweep(xtr, ytr, xte, yte, classes, args.band_regs)
    result["pca_bands"] = bands

    out_dir = Path(args.out_dir) / args.exp_name / args.dataset / f"shot_{args.shot}" / f"seed_{args.seed}"
    save_json(result, out_dir / "result.json")
    write_report(result, out_dir / "result.md")
    print(f"[supp] wrote {out_dir}")


def mean_std(vals, scale=100.0):
    arr = np.array(vals, dtype=float) * scale
    if len(arr) == 1:
        return f"{arr.mean():.2f}"
    return f"{arr.mean():.2f}±{arr.std(ddof=1):.2f}"


def safe_corr(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return float(np.mean(x * y))


def aggregate(args):
    root = Path(args.out_dir) / args.exp_name
    records = [json.load(open(p)) for p in sorted(root.glob("*/*/*/result.json"))]
    save_json(records, root / "aggregate_records.json")

    groups = {}
    for r in records:
        groups.setdefault((r["dataset"], str(r["shot"])), []).append(r)

    group_rows = []
    order = {"30": 30, "50": 50, "100": 100, "full": 999}
    lines = [f"# SDG Supplement Stats Aggregate ({args.exp_name})\n"]
    lines.append("| dataset | shot | n | Cos | Mahal(best) | Gap(best) | WCos(best) | Shrink(best) | tail128 Fisher | tail128 enrich | median PC | cond | eff rank |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (dataset, shot), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 1000))):
        row = {
            "dataset": dataset,
            "shot": shot,
            "n": len(rs),
            "cos": float(np.mean([r["metrics"]["cos"] for r in rs])),
            "mahal_best": float(np.mean([r["metrics"]["mahal"]["best_acc"] for r in rs])),
            "gap_best_pp": float(np.mean([r["metrics"]["gap_best_pp"] for r in rs])),
            "wcos_best": float(np.mean([r["metrics"]["wcos"]["best_acc"] for r in rs])),
            "shared_shrink_best": float(np.mean([r["metrics"]["shared_shrink"]["best_acc"] for r in rs])),
            "tail128_fisher": float(np.mean([r["sdgi"]["tail128_fisher"] for r in rs])),
            "tail128_enrichment": float(np.mean([r["sdgi"]["tail128_enrichment"] for r in rs])),
            "fisher_median_pc": float(np.mean([r["sdgi"]["fisher_median_pc"] for r in rs])),
            "condition_positive": float(np.mean([r["covariance"]["condition_positive"] for r in rs])),
            "effective_rank": float(np.mean([r["covariance"]["effective_rank"] for r in rs])),
        }
        group_rows.append(row)
        lines.append(
            f"| {dataset} | {shot} | {len(rs)} | "
            f"{mean_std([r['metrics']['cos'] for r in rs])} | "
            f"{mean_std([r['metrics']['mahal']['best_acc'] for r in rs])} | "
            f"{mean_std([r['metrics']['gap_best_pp'] for r in rs], scale=1.0)} | "
            f"{mean_std([r['metrics']['wcos']['best_acc'] for r in rs])} | "
            f"{mean_std([r['metrics']['shared_shrink']['best_acc'] for r in rs])} | "
            f"{mean_std([r['sdgi']['tail128_fisher'] for r in rs])} | "
            f"{mean_std([r['sdgi']['tail128_enrichment'] for r in rs], scale=1.0)} | "
            f"{np.mean([r['sdgi']['fisher_median_pc'] for r in rs]):.1f} | "
            f"{np.mean([r['covariance']['condition_positive'] for r in rs]):.2e} | "
            f"{np.mean([r['covariance']['effective_rank'] for r in rs]):.1f} |"
        )
    save_json(group_rows, root / "aggregate_groups.json")
    (root / "aggregate.md").write_text("\n".join(lines) + "\n")

    gaps = [r["gap_best_pp"] for r in group_rows]
    diagnostics = {
        "n_groups": len(group_rows),
        "corr_tail128_fisher_vs_gap": safe_corr([r["tail128_fisher"] for r in group_rows], gaps),
        "corr_tail128_enrichment_vs_gap": safe_corr([r["tail128_enrichment"] for r in group_rows], gaps),
        "corr_effective_rank_vs_gap": safe_corr([r["effective_rank"] for r in group_rows], gaps),
        "corr_log_condition_vs_gap": safe_corr([np.log10(r["condition_positive"] + 1.0) for r in group_rows], gaps),
    }
    save_json(diagnostics, root / "diagnostic_correlations.json")
    corr_lines = ["# Supplement Diagnostic Correlations\n"]
    for k, v in diagnostics.items():
        corr_lines.append(f"- {k}: {v:.6f}" if isinstance(v, float) else f"- {k}: {v}")
    (root / "diagnostic_correlations.md").write_text("\n".join(corr_lines) + "\n")
    print(f"[supp-aggregate] records={len(records)} groups={len(group_rows)} -> {root / 'aggregate.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "aggregate"], required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--shot", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp-name", default="supplement_stats_v1")
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    ap.add_argument("--mahal-regs", nargs="+", type=float, default=[1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
    ap.add_argument("--band-regs", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    ap.add_argument("--wcos-eps", nargs="+", type=float, default=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ap.add_argument("--shrink-alphas", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    ap.add_argument("--diag-reg", type=float, default=1e-4)
    args = ap.parse_args()

    if args.mode == "run":
        if args.dataset is None:
            raise ValueError("--dataset required")
        run(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
