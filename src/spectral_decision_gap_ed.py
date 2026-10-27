#!/usr/bin/env python3
"""Cached-feature experiments for Spectral Decision Gap.

This runner only reads cached frozen features. It is designed so each
dataset/shot/seed/method bundle can be run as an independent local or cluster
job without any project-specific infrastructure.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


RELEASE_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = Path(os.environ.get("SDG_FEATURE_ROOT", RELEASE_ROOT / "features"))
RESULTS_ROOT = Path(os.environ.get("SDG_RESULTS_ROOT", RELEASE_ROOT / "artifacts/results"))
OUT_ROOT = Path(os.environ.get("SDG_OUTPUT_ROOT", RELEASE_ROOT / "outputs"))


DATASETS = ["cub200", "aircraft", "cars", "food101", "pets", "cifar100", "cifar10", "tiny_imagenet", "imagenet100"]


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text(json.dumps(obj, indent=2, default=float) + "\n")
    shutil.copy2(tmp, path)
    tmp.unlink(missing_ok=True)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(suffix=".pt"))
    torch.save(obj, tmp)
    shutil.copy2(tmp, path)
    tmp.unlink(missing_ok=True)


def load_pt(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["pooled"].float(), obj["labels"].long()


def feature_paths(dataset: str):
    candidates = [
        FEATURE_ROOT / dataset,
        FEATURE_ROOT / dataset / "features",
    ]
    for root in candidates:
        train_path = root / "train.pt"
        test_path = root / "test.pt"
        if train_path.exists() and test_path.exists():
            return train_path, test_path
    raise FileNotFoundError(
        f"Missing cached features for {dataset}. Expected train.pt/test.pt under "
        f"{FEATURE_ROOT / dataset} or {FEATURE_ROOT / dataset / 'features'}. "
        "Set SDG_FEATURE_ROOT to your cached-feature directory."
    )


def cache_slim_cub():
    raise RuntimeError(
        "cache_cub is not used in the release. "
        "Place CUB cached features at $SDG_FEATURE_ROOT/cub200/{train,test}.pt."
    )


def load_features(dataset: str):
    tr, te = feature_paths(dataset)
    train_feat, train_lab = load_pt(tr)
    test_feat, test_lab = load_pt(te)
    return train_feat.numpy().astype(np.float64), train_lab.numpy(), test_feat.numpy().astype(np.float64), test_lab.numpy()


def sample_shot(x, y, shot: str, seed: int):
    if shot == "full":
        return x, y
    k = int(shot)
    rng = np.random.RandomState(seed)
    idx = []
    for c in sorted(np.unique(y).tolist()):
        where = np.flatnonzero(y == c)
        rng.shuffle(where)
        idx.extend(where[: min(k, len(where))].tolist())
    idx = np.array(sorted(idx))
    return x[idx], y[idx]


def classes_from(train_lab, test_lab):
    return sorted(set(train_lab.tolist()) & set(test_lab.tolist()))


def ncm_cosine(train_feat, train_lab, test_feat, test_lab, classes):
    protos = np.stack([train_feat[train_lab == c].mean(0) for c in classes])
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
    test = test_feat / (np.linalg.norm(test_feat, axis=1, keepdims=True) + 1e-12)
    pred = np.argmax(test @ protos.T, axis=1)
    pred_lab = np.array([classes[i] for i in pred])
    return float(np.mean(pred_lab == test_lab))


def pooled_cov(train_feat, train_lab, classes):
    d = train_feat.shape[1]
    scatter = np.zeros((d, d), dtype=np.float64)
    n_total = 0
    means = []
    for c in classes:
        xc = train_feat[train_lab == c]
        mu = xc.mean(0)
        means.append(mu)
        centered = xc - mu
        scatter += centered.T @ centered
        n_total += len(xc)
    cov = scatter / max(1, n_total - len(classes))
    return np.stack(means), cov


def mahalanobis(train_feat, train_lab, test_feat, test_lab, classes, reg=1e-4):
    protos, cov = pooled_cov(train_feat, train_lab, classes)
    cov = cov + reg * np.eye(cov.shape[0])
    si = np.linalg.pinv(cov)
    ws = protos @ si
    bias = -0.5 * np.sum(protos * ws, axis=1)
    pred = np.argmax(test_feat @ ws.T + bias[None, :], axis=1)
    pred_lab = np.array([classes[i] for i in pred])
    return float(np.mean(pred_lab == test_lab))


def whitened_cosine(train_feat, train_lab, test_feat, test_lab, classes, eps=1e-6):
    mu = train_feat.mean(0)
    xc = train_feat - mu
    cov = (xc.T @ xc) / max(1, len(xc) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    w = vecs @ np.diag(1.0 / np.sqrt(vals + eps)) @ vecs.T
    return ncm_cosine((train_feat - mu) @ w.T, train_lab, (test_feat - mu) @ w.T, test_lab, classes)


def fecam_diag(train_feat, train_lab, test_feat, test_lab, classes, reg=1e-3):
    means = []
    vars_ = []
    for c in classes:
        xc = train_feat[train_lab == c]
        means.append(xc.mean(0))
        vars_.append(xc.var(0) + reg)
    means = np.stack(means)
    vars_ = np.stack(vars_)
    scores = []
    for mu, var in zip(means, vars_):
        scores.append(-0.5 * (((test_feat - mu) ** 2 / var).sum(1) + np.log(var).sum()))
    pred = np.argmax(np.stack(scores, axis=1), axis=1)
    pred_lab = np.array([classes[i] for i in pred])
    return float(np.mean(pred_lab == test_lab))


def fecam_shrink(train_feat, train_lab, test_feat, test_lab, classes, alpha=0.5, reg=1e-3):
    shared_diag = train_feat.var(0) + reg
    means = []
    vars_ = []
    for c in classes:
        xc = train_feat[train_lab == c]
        means.append(xc.mean(0))
        class_diag = xc.var(0) + reg
        vars_.append((1.0 - alpha) * class_diag + alpha * shared_diag)
    means = np.stack(means)
    vars_ = np.stack(vars_)
    scores = []
    for mu, var in zip(means, vars_):
        scores.append(-0.5 * (((test_feat - mu) ** 2 / var).sum(1) + np.log(var).sum()))
    pred = np.argmax(np.stack(scores, axis=1), axis=1)
    pred_lab = np.array([classes[i] for i in pred])
    return float(np.mean(pred_lab == test_lab))


def logreg_acc(train_feat, train_lab, test_feat, test_lab, c_values):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    best = {"acc": -1.0, "C": None}
    for c in c_values:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=300, solver="lbfgs", multi_class="auto", n_jobs=8),
        )
        clf.fit(train_feat, train_lab)
        acc = float(clf.score(test_feat, test_lab))
        print(f"  [logreg] C={c} acc={acc*100:.2f}")
        if acc > best["acc"]:
            best = {"acc": acc, "C": c}
    return best


def mlp_acc(train_feat, train_lab, test_feat, test_lab, seed):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(512,), activation="relu", alpha=1e-4,
                      batch_size=256, learning_rate_init=1e-3, max_iter=80,
                      early_stopping=False, random_state=seed),
    )
    clf.fit(train_feat, train_lab)
    return float(clf.score(test_feat, test_lab))


def covariance_pca(train_feat):
    mu = train_feat.mean(0)
    xc = train_feat - mu
    cov = (xc.T @ xc) / max(1, len(xc) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return mu, vals[order], vecs[:, order]


def fisher_by_pc(train_feat, train_lab, mean, pcs):
    z = (train_feat - mean) @ pcs
    classes = sorted(np.unique(train_lab).tolist())
    global_mean = z.mean(0)
    between = np.zeros(z.shape[1], dtype=np.float64)
    within = np.zeros(z.shape[1], dtype=np.float64)
    n_total = 0
    for c in classes:
        zc = z[train_lab == c]
        mc = zc.mean(0)
        between += len(zc) * (mc - global_mean) ** 2
        within += ((zc - mc) ** 2).sum(0)
        n_total += len(zc)
    between /= max(1, len(classes) - 1)
    within /= max(1, n_total - len(classes))
    fisher = between / (within + 1e-12)
    mass = fisher / (fisher.sum() + 1e-12)
    return fisher, mass


def rankdata(a):
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float(np.mean(ra * rb))


def sdgi_metrics(train_feat, train_lab):
    mean, eigvals, pcs = covariance_pca(train_feat)
    fisher, mass = fisher_by_pc(train_feat, train_lab, mean, pcs)
    eig_log = np.log(np.maximum(eigvals, 1e-18))
    fisher_log = np.log(np.maximum(fisher, 1e-18))
    total = eigvals.sum()
    return {
        "top1_var": float(eigvals[:1].sum() / total),
        "top5_var": float(eigvals[:5].sum() / total),
        "top10_var": float(eigvals[:10].sum() / total),
        "top50_var": float(eigvals[:50].sum() / total),
        "tail128_fisher": float(mass[128:].sum()),
        "tail256_fisher": float(mass[256:].sum()),
        "bottom_half_fisher": float(mass[len(mass) // 2:].sum()),
        "spearman_logeig_logfisher": spearman(eig_log, fisher_log),
    }


def run_methods(args):
    xtr, ytr, xte, yte = load_features(args.dataset)
    xtr, ytr = sample_shot(xtr, ytr, args.shot, args.seed)
    classes = classes_from(ytr, yte)
    print(f"[run] dataset={args.dataset} shot={args.shot} seed={args.seed} train={xtr.shape} test={xte.shape} classes={len(classes)}")
    out = {
        "dataset": args.dataset,
        "shot": args.shot,
        "seed": args.seed,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_classes": int(len(classes)),
        "metrics": {},
        "sdgi": sdgi_metrics(xtr, ytr),
    }
    methods = set(args.methods)
    if "cos" in methods:
        out["metrics"]["cos"] = ncm_cosine(xtr, ytr, xte, yte, classes)
        print(f"  cos={out['metrics']['cos']*100:.2f}")
    if "mahal" in methods:
        out["metrics"]["mahal_by_reg"] = {}
        best = {"acc": -1.0, "reg": None}
        for reg in args.regs:
            acc = mahalanobis(xtr, ytr, xte, yte, classes, reg=reg)
            out["metrics"]["mahal_by_reg"][str(reg)] = acc
            print(f"  mahal reg={reg} acc={acc*100:.2f}")
            if acc > best["acc"]:
                best = {"acc": acc, "reg": reg}
        out["metrics"]["mahal"] = best
    if "wcos" in methods:
        out["metrics"]["wcos"] = whitened_cosine(xtr, ytr, xte, yte, classes)
        print(f"  wcos={out['metrics']['wcos']*100:.2f}")
    if "fecam_diag" in methods:
        out["metrics"]["fecam_diag"] = fecam_diag(xtr, ytr, xte, yte, classes, reg=args.fecam_reg)
        print(f"  fecam_diag={out['metrics']['fecam_diag']*100:.2f}")
    if "fecam_shrink" in methods:
        out["metrics"]["fecam_shrink"] = {}
        for alpha in args.alphas:
            acc = fecam_shrink(xtr, ytr, xte, yte, classes, alpha=alpha, reg=args.fecam_reg)
            out["metrics"]["fecam_shrink"][str(alpha)] = acc
            print(f"  fecam_shrink alpha={alpha} acc={acc*100:.2f}")
    if "logreg" in methods:
        out["metrics"]["logreg"] = logreg_acc(xtr, ytr, xte, yte, args.c_values)
    if "mlp" in methods:
        out["metrics"]["mlp"] = mlp_acc(xtr, ytr, xte, yte, args.seed)
        print(f"  mlp={out['metrics']['mlp']*100:.2f}")

    out_dir = Path(args.out_dir) / "method_matrix" / args.dataset / f"shot_{args.shot}" / f"seed_{args.seed}"
    save_json(out, out_dir / "result.json")
    write_method_report(out, out_dir / "result.md")


def write_method_report(result, path: Path):
    lines = [f"# {result['dataset']} shot={result['shot']} seed={result['seed']}\n"]
    lines.append(f"- train/test/classes: {result['n_train']} / {result['n_test']} / {result['n_classes']}")
    lines.append("## SDGI")
    for k, v in result["sdgi"].items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("\n## Metrics")
    for k, v in result["metrics"].items():
        lines.append(f"- {k}: {v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(suffix=".md"))
    tmp.write_text("\n".join(lines) + "\n")
    shutil.copy2(tmp, path)
    tmp.unlink(missing_ok=True)


def aggregate(args):
    rows = []
    root = Path(args.out_dir) / "method_matrix"
    for p in sorted(root.glob("*/*/*/result.json")):
        rows.append(read_json(p))
    save_json(rows, Path(args.out_dir) / "method_matrix_aggregate.json")
    lines = ["# Spectral Decision Gap Method Matrix\n"]
    lines.append("| dataset | shot | seed | classes | SDGI-tail128 | Cos | Mahal(best) | WCos | LogReg | MLP | FeCAM diag |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        m = r["metrics"]
        def pct(x):
            return "-" if x is None else f"{x*100:.2f}"
        mahal = m.get("mahal", {}).get("acc") if isinstance(m.get("mahal"), dict) else None
        logreg = m.get("logreg", {}).get("acc") if isinstance(m.get("logreg"), dict) else None
        lines.append(
            f"| {r['dataset']} | {r['shot']} | {r['seed']} | {r['n_classes']} | "
            f"{r['sdgi']['tail128_fisher']*100:.2f} | {pct(m.get('cos'))} | {pct(mahal)} | "
            f"{pct(m.get('wcos'))} | {pct(logreg)} | {pct(m.get('mlp'))} | {pct(m.get('fecam_diag'))} |"
        )
    out = Path(args.out_dir) / "method_matrix_aggregate.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[aggregate] rows={len(rows)} -> {out}")


def plot_existing(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_fig(fig, path: Path):
        tmp = Path(tempfile.mktemp(suffix=".png"))
        fig.savefig(tmp, dpi=200)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        shutil.copy2(tmp, path)
        tmp.unlink(missing_ok=True)

    rows = read_json(RESULTS_ROOT / "aggregate.json")
    xs = np.array([r["tail128_fisher_mass"] * 100 for r in rows])
    ys = np.array([r["mahal_minus_cos_pp"] for r in rows])
    labels = [r["dataset"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(xs, ys, s=70)
    for x, y, lab in zip(xs, ys, labels):
        ax.text(x + 0.4, y, lab, fontsize=8)
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xlabel("SDGI-tail128: Fisher mass in PC tail (%)")
    ax.set_ylabel("Mahalanobis - Cosine accuracy gap (pp)")
    ax.set_title("Spectral Decision Gap predicts prototype evaluation failure")
    fig.tight_layout()
    out_dir = Path(args.out_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_dir / "sdgi_tail128_vs_gap.png")

    xs2 = np.array([r["top10_var_pct"] * 100 for r in rows])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(xs2, ys, s=70)
    for x, y, lab in zip(xs2, ys, labels):
        ax.text(x + 0.4, y, lab, fontsize=8)
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xlabel("Top-10 variance explained (%)")
    ax.set_ylabel("Mahalanobis - Cosine accuracy gap (pp)")
    ax.set_title("Variance alone is not the decision diagnostic")
    fig.tight_layout()
    save_fig(fig, out_dir / "top10var_vs_gap.png")
    print(f"[plot] wrote {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cache_cub", "methods", "aggregate", "plot_existing"], required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--shot", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["cos", "mahal", "wcos", "fecam_diag", "fecam_shrink", "logreg"])
    ap.add_argument("--regs", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--c-values", nargs="+", type=float, default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--fecam-reg", type=float, default=1e-3)
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    args = ap.parse_args()

    if args.mode == "cache_cub":
        cache_slim_cub()
    elif args.mode == "methods":
        if args.dataset is None:
            raise ValueError("--dataset required for methods")
        run_methods(args)
    elif args.mode == "aggregate":
        aggregate(args)
    else:
        plot_existing(args)


if __name__ == "__main__":
    main()
