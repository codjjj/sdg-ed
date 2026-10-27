#!/usr/bin/env python3
"""Controlled Gaussian-mixture mechanism check for SDG-ED.

The real-data analysis is correlational: high tail Fisher mass tends to predict
cosine-vs-whitening disagreement. This synthetic experiment keeps the shared
covariance spectrum fixed and only moves class-mean signal across principal
components. If the SDG story is right, placing the same discriminative signal in
low-variance directions should hurt raw cosine much more than whitening-aware
rules.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("SDG_OUTPUT_ROOT", RELEASE_ROOT / "outputs"))


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def cosine_ncm(xtr, ytr, xte, yte, classes):
    means = np.stack([xtr[ytr == c].mean(0) for c in classes])
    pred = np.argmax(l2_normalize(xte) @ l2_normalize(means).T, axis=1)
    pred_lab = np.array([classes[i] for i in pred])
    return float(np.mean(pred_lab == yte))


def pooled_covariance(x, y, classes):
    d = x.shape[1]
    means = []
    scatter = np.zeros((d, d), dtype=np.float64)
    n = 0
    for c in classes:
        xc = x[y == c]
        mu = xc.mean(0)
        means.append(mu)
        scatter += (xc - mu).T @ (xc - mu)
        n += len(xc)
    return np.stack(means), scatter / max(1, n - len(classes))


def lda(xtr, ytr, xte, yte, classes, reg):
    means, cov = pooled_covariance(xtr, ytr, classes)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    inv_cov = (vecs * (1.0 / (vals + reg))) @ vecs.T
    weights = means @ inv_cov
    bias = -0.5 * np.sum(means * weights, axis=1)
    pred = np.argmax(xte @ weights.T + bias[None, :], axis=1)
    pred_lab = np.array([classes[i] for i in pred])
    return float(np.mean(pred_lab == yte))


def whitened_cosine(xtr, ytr, xte, yte, classes, eps):
    mean = xtr.mean(0)
    xc = xtr - mean
    cov = (xc.T @ xc) / max(1, len(xc) - 1)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    scale = 1.0 / np.sqrt(vals + eps)
    ztr = ((xtr - mean) @ vecs) * scale[None, :]
    zte = ((xte - mean) @ vecs) * scale[None, :]
    return cosine_ncm(ztr, ytr, zte, yte, classes)


def bayes_diag(xte, yte, means, eigvals):
    inv = 1.0 / eigvals
    scores = []
    for mu in means:
        scores.append(-0.5 * (((xte - mu) ** 2) * inv[None, :]).sum(1))
    pred = np.argmax(np.stack(scores, axis=1), axis=1)
    pred_lab = pred.astype(yte.dtype)
    return float(np.mean(pred_lab == yte))


def fisher_diagnostics(xtr, ytr, eigvals, classes):
    global_mean = xtr.mean(0)
    between = np.zeros(xtr.shape[1], dtype=np.float64)
    within = np.zeros(xtr.shape[1], dtype=np.float64)
    n = 0
    for c in classes:
        xc = xtr[ytr == c]
        mu = xc.mean(0)
        between += len(xc) * (mu - global_mean) ** 2
        within += ((xc - mu) ** 2).sum(0)
        n += len(xc)
    between /= max(1, len(classes) - 1)
    within /= max(1, n - len(classes))
    fisher = between / (within + 1e-12)
    fisher_mass = fisher / (fisher.sum() + 1e-12)
    tail = np.arange(len(eigvals)) >= len(eigvals) - 64
    tail_fisher = float(fisher_mass[tail].sum())
    tail_var = float(eigvals[tail].sum() / eigvals.sum())
    return {
        "tail64_fisher": tail_fisher,
        "tail64_var": tail_var,
        "tail64_enrichment": tail_fisher / max(tail_var, 1e-12),
        "top16_var": float(eigvals[:16].sum() / eigvals.sum()),
    }


def true_fisher_diagnostics(means, eigvals):
    between = ((means - means.mean(0)) ** 2).mean(0)
    fisher = between / (eigvals + 1e-12)
    fisher_mass = fisher / (fisher.sum() + 1e-12)
    tail = np.arange(len(eigvals)) >= len(eigvals) - 64
    tail_fisher = float(fisher_mass[tail].sum())
    tail_var = float(eigvals[tail].sum() / eigvals.sum())
    rank_frac = np.arange(len(eigvals), dtype=np.float64) / max(1, len(eigvals) - 1)
    return {
        "true_tail64_fisher": tail_fisher,
        "true_tail64_var": tail_var,
        "true_tail64_enrichment": tail_fisher / max(tail_var, 1e-12),
        "true_fisher_rank_frac": float((rank_frac * fisher_mass).sum()),
    }


def band_indices(name, dim):
    if name == "head":
        return np.arange(0, 16)
    if name == "mid":
        lo = dim // 2 - 8
        return np.arange(lo, lo + 16)
    if name == "tail":
        return np.arange(dim - 16, dim)
    if name == "mixed":
        return np.r_[np.arange(0, 8), np.arange(dim - 8, dim)]
    raise ValueError(name)


def make_dataset(args, band, seed):
    rng = np.random.RandomState(seed)
    classes = np.arange(args.n_classes)
    dim = args.dim
    eigvals = np.geomspace(1.0, args.min_eig, dim)
    active = band_indices(band, dim)
    means = np.zeros((args.n_classes, dim), dtype=np.float64)
    raw = rng.randn(args.n_classes, len(active))
    raw = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12)
    means[:, active] = args.mean_scale * raw

    def sample(n_per_class):
        xs = []
        ys = []
        noise_scale = np.sqrt(eigvals)
        for c in classes:
            x = means[c][None, :] + rng.randn(n_per_class, dim) * noise_scale[None, :]
            xs.append(x)
            ys.extend([c] * n_per_class)
        return np.vstack(xs), np.array(ys, dtype=np.int64)

    xtr, ytr = sample(args.train_per_class)
    xte, yte = sample(args.test_per_class)
    return xtr, ytr, xte, yte, classes.tolist(), means, eigvals


def run(args):
    rows = []
    for seed in range(args.seeds):
        for band in ["head", "mid", "tail", "mixed"]:
            xtr, ytr, xte, yte, classes, means, eigvals = make_dataset(args, band, args.seed + seed)
            cos = cosine_ncm(xtr, ytr, xte, yte, classes)
            wcos = whitened_cosine(xtr, ytr, xte, yte, classes, args.eps)
            lda_acc = lda(xtr, ytr, xte, yte, classes, args.reg)
            bayes = bayes_diag(xte, yte, means, eigvals)
            diag = fisher_diagnostics(xtr, ytr, eigvals, classes)
            true_diag = true_fisher_diagnostics(means, eigvals)
            rows.append(
                {
                    "band": band,
                    "seed": seed,
                    "cos": cos,
                    "wcos": wcos,
                    "lda": lda_acc,
                    "bayes_diag": bayes,
                    "sdg_lda_pp": 100.0 * (lda_acc - cos),
                    "sdg_wcos_pp": 100.0 * (wcos - cos),
                    **diag,
                    **true_diag,
                }
            )

    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json({"config": vars(args), "rows": rows}, out_dir / "summary.json")
    write_report(rows, out_dir / "summary.md")
    plot(rows, out_dir / "synthetic_gaussian_mechanism.pdf")
    plot(rows, out_dir / "synthetic_gaussian_mechanism.png")
    print(f"[synthetic] wrote {out_dir}")


def summarize(rows, key):
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)


def write_report(rows, path: Path):
    bands = ["head", "mid", "tail", "mixed"]
    lines = ["# Synthetic Gaussian Mechanism\n"]
    lines.append("Only the location of class-mean signal changes; the shared covariance spectrum is fixed.")
    lines.append("")
    lines.append("| signal PCs | Cos | WCos | LDA | Bayes diag | LDA-Cos pp | true Fisher rank | true tail64 Fisher |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for band in bands:
        sub = [r for r in rows if r["band"] == band]
        cos, cos_s = summarize(sub, "cos")
        wcos, wcos_s = summarize(sub, "wcos")
        lda_m, lda_s = summarize(sub, "lda")
        bayes, bayes_s = summarize(sub, "bayes_diag")
        gap, gap_s = summarize(sub, "sdg_lda_pp")
        rank, _ = summarize(sub, "true_fisher_rank_frac")
        tail, _ = summarize(sub, "true_tail64_fisher")
        lines.append(
            f"| {band} | {100*cos:.2f}±{100*cos_s:.2f} | {100*wcos:.2f}±{100*wcos_s:.2f} | "
            f"{100*lda_m:.2f}±{100*lda_s:.2f} | {100*bayes:.2f}±{100*bayes_s:.2f} | "
            f"{gap:.2f}±{gap_s:.2f} | {rank:.2f} | {100*tail:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot(rows, path: Path):
    bands = ["head", "mid", "tail", "mixed"]
    labels = ["Head", "Mid", "Tail", "Mixed"]
    cos = []
    wcos = []
    lda_vals = []
    gap = []
    rank_scores = []
    for band in bands:
        sub = [r for r in rows if r["band"] == band]
        cos.append(100.0 * summarize(sub, "cos")[0])
        wcos.append(100.0 * summarize(sub, "wcos")[0])
        lda_vals.append(100.0 * summarize(sub, "lda")[0])
        gap.append(summarize(sub, "sdg_lda_pp")[0])
        rank_scores.append(summarize(sub, "true_fisher_rank_frac")[0])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    x = np.arange(len(bands))
    axes[0].bar(x - 0.24, cos, width=0.24, label="Cos", color="#999999")
    axes[0].bar(x, wcos, width=0.24, label="WCos", color="#66aa66")
    axes[0].bar(x + 0.24, lda_vals, width=0.24, label="LDA", color="#4477aa")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_title("Moving signal into tail PCs")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].scatter(rank_scores, gap, s=55, color="#7b3294")
    for e, g, lab in zip(rank_scores, gap, labels):
        axes[1].text(e + 0.02, g, lab, fontsize=8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("true Fisher-weighted PC rank")
    axes[1].set_ylabel("LDA - Cosine (pp)")
    axes[1].set_title("Controlled SDG mechanism")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=220 if path.suffix == ".png" else None)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    ap.add_argument("--exp-name", default="synthetic_gaussian_v1")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--n-classes", type=int, default=50)
    ap.add_argument("--train-per-class", type=int, default=20)
    ap.add_argument("--test-per-class", type=int, default=100)
    ap.add_argument("--min-eig", type=float, default=1e-3)
    ap.add_argument("--mean-scale", type=float, default=0.9)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
