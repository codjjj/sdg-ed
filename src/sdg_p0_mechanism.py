#!/usr/bin/env python3
"""Paper-grade mechanism figures for Spectral Decision Gap."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from spectral_decision_gap_ed import (
    OUT_ROOT,
    classes_from,
    covariance_pca,
    fisher_by_pc,
    load_features,
    mahalanobis,
    ncm_cosine,
    sample_shot,
    save_json,
    spearman,
)


def save_fig(fig, path: Path):
    tmp = Path(tempfile.mktemp(suffix=".png"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(tmp, dpi=220)
    path.unlink(missing_ok=True)
    shutil.copy2(tmp, path)
    tmp.unlink(missing_ok=True)


def project(mean, pcs, x, idx):
    return (x - mean) @ pcs[:, idx]


def pca_bands(dim: int):
    bands = [
        ("all", 0, dim),
        ("top16", 0, min(16, dim)),
        ("top64", 0, min(64, dim)),
        ("top128", 0, min(128, dim)),
        ("top256", 0, min(256, dim)),
        ("drop_top16", min(16, dim), dim),
        ("drop_top64", min(64, dim), dim),
        ("mid64_128", 64, min(128, dim)),
        ("mid128_256", 128, min(256, dim)),
        ("tail128", max(0, dim - 128), dim),
        ("tail256", max(0, dim - 256), dim),
    ]
    return [(name, np.arange(lo, hi)) for name, lo, hi in bands if hi > lo]


def safe_pearson(a, b):
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.mean(a * b))


def pca_ablation(xtr, ytr, xte, yte, classes, mean, pcs, reg):
    rows = []
    for name, idx in pca_bands(pcs.shape[1]):
        tr = project(mean, pcs, xtr, idx)
        te = project(mean, pcs, xte, idx)
        cos = ncm_cosine(tr, ytr, te, yte, classes)
        mah = mahalanobis(tr, ytr, te, yte, classes, reg=reg)
        rows.append({"band": name, "dim": int(len(idx)), "cos": cos, "mahal": mah, "gap_pp": 100 * (mah - cos)})
        print(f"[pca] {name:<12} d={len(idx):>3} cos={cos*100:.2f} mah={mah*100:.2f} gap={100*(mah-cos):+.2f}")
    return rows


def write_report(dataset, shot, seed, metrics, pca_rows, out_dir: Path):
    lines = [f"# P0 Mechanism: {dataset} shot={shot} seed={seed}\n"]
    lines.append("## Metrics")
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.6f}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("\n## PCA Ablation")
    lines.append("| band | dim | Cos | Mahal | Gap |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in pca_rows:
        lines.append(f"| {r['band']} | {r['dim']} | {r['cos']*100:.2f} | {r['mahal']*100:.2f} | {r['gap_pp']:.2f} |")
    (out_dir / "mechanism_report.md").write_text("\n".join(lines) + "\n")


def plot_all(dataset, eigvals, fisher, mass, pca_rows, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(1, len(eigvals) + 1)
    ev = eigvals / eigvals.sum()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(x, ev, lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("PC index")
    ax.set_ylabel("Explained variance ratio (log)")
    ax.set_title(f"{dataset}: eigen spectrum")
    fig.tight_layout()
    save_fig(fig, out_dir / "eigenspectrum.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(x, fisher, lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("PC index")
    ax.set_ylabel("Fisher ratio (log)")
    ax.set_title(f"{dataset}: per-PC Fisher ratio")
    fig.tight_layout()
    save_fig(fig, out_dir / "fisher_vs_pc.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(x, np.cumsum(mass), lw=1.6)
    ax.axvline(128, color="0.65", lw=1)
    ax.axvline(256, color="0.75", lw=1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("PC index")
    ax.set_ylabel("Cumulative Fisher mass")
    ax.set_title(f"{dataset}: decision signal by spectrum")
    fig.tight_layout()
    save_fig(fig, out_dir / "cumulative_fisher.png")
    plt.close(fig)

    labels = [r["band"] for r in pca_rows]
    cos = np.array([r["cos"] * 100 for r in pca_rows])
    mah = np.array([r["mahal"] * 100 for r in pca_rows])
    pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(labels)), 4.2))
    ax.bar(pos - 0.18, cos, width=0.36, label="Cos")
    ax.bar(pos + 0.18, mah, width=0.36, label="Mahal")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{dataset}: PCA truncation / band ablation")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_dir / "pca_ablation.png")
    plt.close(fig)


def run(args):
    xtr_all, ytr_all, xte, yte = load_features(args.dataset)
    xtr, ytr = sample_shot(xtr_all, ytr_all, args.shot, args.seed)
    classes = classes_from(ytr, yte)
    out_dir = Path(args.out_dir) / args.exp_name / args.dataset / f"shot_{args.shot}" / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[p0] dataset={args.dataset} shot={args.shot} seed={args.seed} train={xtr.shape} test={xte.shape} classes={len(classes)}")

    mean, eigvals, pcs = covariance_pca(xtr)
    fisher, mass = fisher_by_pc(xtr, ytr, mean, pcs)
    total = eigvals.sum()
    metrics = {
        "dataset": args.dataset,
        "shot": args.shot,
        "seed": args.seed,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_classes": int(len(classes)),
        "dim": int(xtr.shape[1]),
        "cos": ncm_cosine(xtr, ytr, xte, yte, classes),
        "mahal": mahalanobis(xtr, ytr, xte, yte, classes, reg=args.pca_reg),
        "top1_var": float(eigvals[:1].sum() / total),
        "top5_var": float(eigvals[:5].sum() / total),
        "top10_var": float(eigvals[:10].sum() / total),
        "top50_var": float(eigvals[:50].sum() / total),
        "tail128_fisher": float(mass[128:].sum()),
        "tail256_fisher": float(mass[256:].sum()),
        "bottom_half_fisher": float(mass[len(mass) // 2:].sum()),
    }
    eig_log = np.log(np.maximum(eigvals, 1e-18))
    fisher_log = np.log(np.maximum(fisher, 1e-18))
    metrics["sdgi_spearman_logeig_logfisher"] = spearman(eig_log, fisher_log)
    metrics["sdgi_corr_logeig_logfisher"] = safe_pearson(eig_log, fisher_log)
    metrics["mahal_reg_sweep"] = {str(reg): mahalanobis(xtr, ytr, xte, yte, classes, reg=reg) for reg in args.mahal_regs}
    best_reg, best_acc = max(metrics["mahal_reg_sweep"].items(), key=lambda kv: kv[1])
    metrics["mahal_best_reg"] = float(best_reg)
    metrics["mahal_best"] = float(best_acc)
    metrics["gap_pp"] = 100 * (metrics["mahal"] - metrics["cos"])
    metrics["gap_best_pp"] = 100 * (metrics["mahal_best"] - metrics["cos"])
    pca_rows = pca_ablation(xtr, ytr, xte, yte, classes, mean, pcs, args.pca_reg)
    save_json({"metrics": metrics, "pca_ablation": pca_rows}, out_dir / "metrics.json")

    tmp_npz = Path(tempfile.mktemp(suffix=".npz"))
    np.savez(tmp_npz, eigvals=eigvals, fisher=fisher, fisher_mass=mass)
    shutil.copy2(tmp_npz, out_dir / "spectrum_fisher.npz")
    tmp_npz.unlink(missing_ok=True)
    plot_all(args.dataset, eigvals, fisher, mass, pca_rows, out_dir)
    write_report(args.dataset, args.shot, args.seed, metrics, pca_rows, out_dir)
    print(f"[p0] wrote {out_dir}")


def aggregate(args):
    root = Path(args.out_dir) / args.exp_name
    rows = []
    for p in sorted(root.glob("*/*/*/metrics.json")):
        obj = json.load(open(p))
        row = obj["metrics"]
        npz_path = p.parent / "spectrum_fisher.npz"
        if npz_path.exists():
            arr = np.load(npz_path)
            eigvals = arr["eigvals"]
            mass = arr["fisher_mass"]
            total = eigvals.sum() + 1e-12
            row["tail128_var"] = float(eigvals[128:].sum() / total)
            row["tail256_var"] = float(eigvals[256:].sum() / total)
            row["bottom_half_var"] = float(eigvals[len(eigvals) // 2:].sum() / total)
            row["tail128_fisher_enrichment"] = float(row["tail128_fisher"] / (row["tail128_var"] + 1e-12))
            row["tail256_fisher_enrichment"] = float(row["tail256_fisher"] / (row["tail256_var"] + 1e-12))
            row["bottom_half_fisher_enrichment"] = float(row["bottom_half_fisher"] / (row["bottom_half_var"] + 1e-12))
        rows.append(row)
    if len(rows) >= 3 and all("sdgi_spearman_logeig_logfisher" in r for r in rows):
        x = np.array([
            [
                1.0,
                r["tail128_fisher"],
                r["bottom_half_fisher"],
                r["sdgi_spearman_logeig_logfisher"],
            ]
            for r in rows
        ], dtype=np.float64)
        y = np.array([r["gap_best_pp"] for r in rows], dtype=np.float64)
        coef = np.linalg.pinv(x) @ y
        for r, pred in zip(rows, x @ coef):
            r["sdgi_gap_pred_pp"] = float(pred)
        pred_meta = {
            "features": ["bias", "tail128_fisher", "bottom_half_fisher", "sdgi_spearman_logeig_logfisher"],
            "coef": coef.tolist(),
        }
    else:
        pred_meta = {"features": [], "coef": []}
    save_json(rows, root / "summary.json")
    save_json({"prediction": pred_meta, "rows": rows}, root / "summary_with_pred.json")

    def write_rows_md(path: Path, title: str, selected):
        lines = [f"# {title}\n"]
        lines.append("| dataset | shot | seed | Cos | Mahal@1e-4 | Gap@1e-4 | Mahal(best) | Gap(best) | SDGI pred | top10 var | tail128 Fisher | tail128 enrich | bottom-half Fisher | Spearman(logeig,logF) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in selected:
            lines.append(
                f"| {r['dataset']} | {r['shot']} | {r['seed']} | {r['cos']*100:.2f} | {r['mahal']*100:.2f} | "
                f"{r['gap_pp']:.2f} | {r.get('mahal_best', r['mahal'])*100:.2f} | {r.get('gap_best_pp', r['gap_pp']):.2f} | "
                f"{r.get('sdgi_gap_pred_pp', float('nan')):.2f} | "
                f"{r['top10_var']*100:.2f} | {r['tail128_fisher']*100:.2f} | "
                f"{r.get('tail128_fisher_enrichment', float('nan')):.2f} | "
                f"{r['bottom_half_fisher']*100:.2f} | "
                f"{r.get('sdgi_spearman_logeig_logfisher', float('nan')):.3f} |"
            )
        path.write_text("\n".join(lines) + "\n")

    write_rows_md(root / "summary.md", "P0 Mechanism Summary", rows)
    full_rows = [r for r in rows if str(r["shot"]) == "full"]
    if full_rows:
        write_rows_md(root / "summary_full.md", "P0 Full-Shot Mechanism Summary", full_rows)

    groups = {}
    for r in rows:
        if str(r["shot"]) != "full":
            groups.setdefault((r["dataset"], str(r["shot"])), []).append(r)
    if groups:
        lines = ["# P0 Seed Stability Summary\n"]
        lines.append("| dataset | shot | n | Cos | Mahal@1e-4 | Gap@1e-4 | Mahal(best) | Gap(best) | tail128 Fisher | tail128 enrich | Spearman(logeig,logF) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for (dataset, shot), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
            def ms(key, scale=1.0):
                vals = np.array([r[key] for r in rs], dtype=float) * scale
                if len(vals) == 1:
                    return f"{vals.mean():.2f}"
                return f"{vals.mean():.2f}±{vals.std(ddof=1):.2f}"
            lines.append(
                f"| {dataset} | {shot} | {len(rs)} | {ms('cos', 100)} | {ms('mahal', 100)} | "
                f"{ms('gap_pp')} | {ms('mahal_best', 100)} | {ms('gap_best_pp')} | {ms('tail128_fisher', 100)} | "
                f"{ms('tail128_fisher_enrichment')} | {ms('sdgi_spearman_logeig_logfisher')} |"
            )
        (root / "seed_stability.md").write_text("\n".join(lines) + "\n")
    print(f"[p0-aggregate] rows={len(rows)} -> {root / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "aggregate"], required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--shot", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mahal-regs", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    ap.add_argument("--pca-reg", type=float, default=1e-4)
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    ap.add_argument("--exp-name", default="p0_mechanism")
    args = ap.parse_args()
    if args.mode == "run":
        if args.dataset is None:
            raise ValueError("--dataset required")
        run(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
