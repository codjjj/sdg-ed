#!/usr/bin/env python3
"""Train-only cached-feature evaluation for the SDG-ED benchmark.

This script runs the train-only protocol on cached features (no new feature
extraction is needed):
  1. LDA regularization is selected on training data only, never on test labels;
  2. an equal-budget estimator decomposition around cosine prototypes;
  3. reference probes for the ranking/regret analysis.

One run handles one dataset/shot group and loops over the relevant seeds.
Full-shot uses one seed; few-shot uses five seeds by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spectral_decision_gap_ed import (
    OUT_ROOT,
    classes_from,
    load_features,
    load_pt,
    ncm_cosine,
    sample_shot,
    save_json,
)


LDA_REGS = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
WCOS_EPS = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
RIDGE_ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
KNN_KS = [1, 3, 5, 10, 20]
DEFAULT_LDA_REG = 1e-4
DEFAULT_WCOS_EPS = 1e-4
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_KNN_K = 5


def pooled_stats(x, y, classes):
    d = x.shape[1]
    scatter = np.zeros((d, d), dtype=np.float64)
    means = []
    n_total = 0
    for c in classes:
        xc = x[y == c]
        mu = xc.mean(0)
        means.append(mu)
        scatter += (xc - mu).T @ (xc - mu)
        n_total += len(xc)
    cov = scatter / max(1, n_total - len(classes))
    return np.stack(means), cov


def lda_acc_from_stats(means, cov, x_eval, y_eval, classes, reg):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    inv_cov = (vecs * (1.0 / (vals + reg))) @ vecs.T
    weights = means @ inv_cov
    bias = -0.5 * np.sum(means * weights, axis=1)
    pred = np.argmax(x_eval @ weights.T + bias[None, :], axis=1)
    pred_lab = np.array([classes[i] for i in pred])
    return float(np.mean(pred_lab == y_eval))


def lda_sweep(x_train, y_train, x_eval, y_eval, classes, regs):
    means, cov = pooled_stats(x_train, y_train, classes)
    rows = {str(reg): lda_acc_from_stats(means, cov, x_eval, y_eval, classes, reg) for reg in regs}
    best_reg, best_acc = max(rows.items(), key=lambda kv: kv[1])
    return rows, float(best_reg), float(best_acc)


def qda_diag_acc(x_train, y_train, x_eval, y_eval, classes, reg):
    means = []
    vars_ = []
    for c in classes:
        xc = x_train[y_train == c]
        means.append(xc.mean(0))
        vars_.append(xc.var(0) + reg)
    means = np.stack(means)
    vars_ = np.stack(vars_)
    scores = []
    for mu, var in zip(means, vars_):
        scores.append(-0.5 * (((x_eval - mu) ** 2 / var).sum(1) + np.log(var).sum()))
    pred = np.argmax(np.stack(scores, axis=1), axis=1)
    pred_lab = np.array([classes[i] for i in pred])
    return float(np.mean(pred_lab == y_eval))


def qda_diag_sweep(x_train, y_train, x_eval, y_eval, classes, regs):
    rows = {str(reg): qda_diag_acc(x_train, y_train, x_eval, y_eval, classes, reg) for reg in regs}
    best_reg, best_acc = max(rows.items(), key=lambda kv: kv[1])
    return rows, float(best_reg), float(best_acc)


def centered_cosine(x_train, y_train, x_test, y_test, classes):
    mean = x_train.mean(0)
    return ncm_cosine(x_train - mean, y_train, x_test - mean, y_test, classes)


def euclidean_ncm(x_train, y_train, x_test, y_test, classes):
    means = np.stack([x_train[y_train == c].mean(0) for c in classes])
    x2 = np.sum(x_test * x_test, axis=1, keepdims=True)
    m2 = np.sum(means * means, axis=1)[None, :]
    pred = np.argmin(x2 + m2 - 2.0 * x_test @ means.T, axis=1)
    pred_lab = np.array([classes[i] for i in pred])
    return float(np.mean(pred_lab == y_test))


def whiten_projectors(x_train):
    mean = x_train.mean(0)
    xc = x_train - mean
    cov = (xc.T @ xc) / max(1, len(xc) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return mean, vals[order], vecs[:, order]


def wcos_acc_from_pca(x_train, y_train, x_eval, y_eval, classes, mean, vals, vecs, eps):
    z_train = (x_train - mean) @ vecs
    z_eval = (x_eval - mean) @ vecs
    scale = 1.0 / np.sqrt(np.maximum(vals, 0.0) + eps)
    return ncm_cosine(z_train * scale[None, :], y_train, z_eval * scale[None, :], y_eval, classes)


def wcos_sweep(x_train, y_train, x_eval, y_eval, classes, eps_values):
    mean, vals, vecs = whiten_projectors(x_train)
    rows = {str(eps): wcos_acc_from_pca(x_train, y_train, x_eval, y_eval, classes, mean, vals, vecs, eps) for eps in eps_values}
    best_eps, best_acc = max(rows.items(), key=lambda kv: kv[1])
    return rows, float(best_eps), float(best_acc)


def whiten_transform(x_train, x_eval, eps):
    mean, vals, vecs = whiten_projectors(x_train)
    scale = 1.0 / np.sqrt(np.maximum(vals, 0.0) + eps)
    return ((x_train - mean) @ vecs) * scale[None, :], ((x_eval - mean) @ vecs) * scale[None, :]


def knn_acc(x_train, y_train, x_eval, y_eval, classes, k, metric="cosine", batch_size=512):
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([label_to_idx[int(c)] for c in y_train], dtype=np.int64)
    train = np.asarray(x_train, dtype=np.float32)
    query = np.asarray(x_eval, dtype=np.float32)
    if metric == "cosine":
        train = train / (np.linalg.norm(train, axis=1, keepdims=True) + 1e-12)
        query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
        train_norm = None
    else:
        train_norm = np.sum(train * train, axis=1)[None, :]
    preds = []
    for lo in range(0, len(query), batch_size):
        q = query[lo:lo + batch_size]
        sim = q @ train.T
        if metric == "cosine":
            top = np.argpartition(-sim, min(k, train.shape[0]) - 1, axis=1)[:, :k]
        else:
            dist = np.sum(q * q, axis=1, keepdims=True) + train_norm - 2.0 * sim
            top = np.argpartition(dist, min(k, train.shape[0]) - 1, axis=1)[:, :k]
        for row in top:
            counts = np.bincount(y_idx[row], minlength=len(classes))
            preds.append(classes[int(np.argmax(counts))])
    return float(np.mean(np.array(preds) == y_eval))


def knn_sweep(x_train, y_train, x_eval, y_eval, classes, ks, metric="cosine"):
    rows = {str(k): knn_acc(x_train, y_train, x_eval, y_eval, classes, k, metric=metric) for k in ks if k <= len(y_train)}
    best_k, best_acc = max(rows.items(), key=lambda kv: kv[1])
    return rows, int(best_k), float(best_acc)


def ridge_acc(x_train, y_train, x_eval, y_eval, alpha):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha))
    clf.fit(x_train, y_train)
    return float(clf.score(x_eval, y_eval))


def ridge_sweep(x_train, y_train, x_eval, y_eval, classes, alphas):
    rows = {str(alpha): ridge_acc(x_train, y_train, x_eval, y_eval, alpha) for alpha in alphas}
    best_alpha, best_acc = max(rows.items(), key=lambda kv: kv[1])
    return rows, float(best_alpha), float(best_acc)


def train_val_split(x, y, seed, val_frac=0.2, max_val_per_class=50):
    rng = np.random.RandomState(seed + 7919)
    train_idx = []
    val_idx = []
    for c in sorted(np.unique(y).tolist()):
        idx = np.flatnonzero(y == c)
        if len(idx) < 2:
            return x, y, None, None, {"mode": "no_validation", "reason": "class_has_less_than_2_examples"}
        rng.shuffle(idx)
        if len(idx) <= 5:
            n_val = 1
        else:
            n_val = max(1, int(round(len(idx) * val_frac)))
            n_val = min(n_val, max_val_per_class, len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    train_idx = np.array(sorted(train_idx))
    val_idx = np.array(sorted(val_idx))
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], {
        "mode": "class_balanced_holdout",
        "val_per_class_cap": int(max_val_per_class),
        "val_frac": float(val_frac),
        "n_inner_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
    }


def select_then_refit(name, x_all, y_all, x_inner, y_inner, x_val, y_val, x_test, y_test, classes, candidates, default, sweep_fn):
    if x_val is None:
        test_rows, _, _ = sweep_fn(x_all, y_all, x_test, y_test, classes, [default])
        return {
            "acc": float(test_rows[str(default)]),
            "selected": default,
            "selection": "default_no_validation",
            "val_scores": {},
            "budget": 0,
        }
    val_rows, best, _ = sweep_fn(x_inner, y_inner, x_val, y_val, classes, candidates)
    test_rows, _, _ = sweep_fn(x_all, y_all, x_test, y_test, classes, [best])
    return {
        "acc": float(test_rows[str(best)]),
        "selected": best,
        "selection": f"{name}_validation",
        "val_scores": val_rows,
        "budget": len(candidates),
    }


def load_features_for_args(dataset, args):
    """Load features from --train-cache/--test-cache if given, else from SDG_FEATURE_ROOT."""
    if getattr(args, "train_cache", None) or getattr(args, "test_cache", None):
        if not (args.train_cache and args.test_cache):
            raise ValueError("--train-cache and --test-cache must be provided together")
        train_feat, train_lab = load_pt(Path(args.train_cache))
        test_feat, test_lab = load_pt(Path(args.test_cache))
        return (
            train_feat.numpy().astype(np.float64),
            train_lab.numpy(),
            test_feat.numpy().astype(np.float64),
            test_lab.numpy(),
        )
    return load_features(dataset)


def run_one(dataset, shot, seed, args):
    xtr_all, ytr_all, xte, yte = load_features_for_args(dataset, args)
    xtr, ytr = sample_shot(xtr_all, ytr_all, shot, seed)
    classes = classes_from(ytr, yte)
    x_inner, y_inner, x_val, y_val, split = train_val_split(
        xtr, ytr, seed=seed, val_frac=args.val_frac, max_val_per_class=args.max_val_per_class
    )
    print(
        f"[stepb] dataset={dataset} shot={shot} seed={seed} "
        f"train={xtr.shape} test={xte.shape} classes={len(classes)} val={split['mode']}"
    )

    lda_oracle_rows, lda_oracle_reg, lda_oracle_acc = lda_sweep(xtr, ytr, xte, yte, classes, args.lda_regs)
    wcos_oracle_rows, wcos_oracle_eps, wcos_oracle_acc = wcos_sweep(xtr, ytr, xte, yte, classes, args.wcos_eps)

    lda_trainval = select_then_refit("lda", xtr, ytr, x_inner, y_inner, x_val, y_val, xte, yte, classes, args.lda_regs, args.default_lda_reg, lda_sweep)
    wcos_trainval = select_then_refit("wcos", xtr, ytr, x_inner, y_inner, x_val, y_val, xte, yte, classes, args.wcos_eps, args.default_wcos_eps, wcos_sweep)
    qda_diag_trainval = select_then_refit("qda_diag", xtr, ytr, x_inner, y_inner, x_val, y_val, xte, yte, classes, args.lda_regs, args.default_lda_reg, qda_diag_sweep)
    ridge_trainval = select_then_refit("ridge", xtr, ytr, x_inner, y_inner, x_val, y_val, xte, yte, classes, args.ridge_alphas, args.default_ridge_alpha, ridge_sweep)
    knn_cos_trainval = select_then_refit("knn_cosine", xtr, ytr, x_inner, y_inner, x_val, y_val, xte, yte, classes, args.knn_ks, args.default_knn_k, knn_sweep)

    xtr_w, xte_w = whiten_transform(xtr, xte, wcos_trainval["selected"])
    knn_wcos = {}
    for k in args.knn_ks:
        if k <= len(ytr):
            knn_wcos[str(k)] = knn_acc(xtr_w, ytr, xte_w, yte, classes, k, metric="cosine")
    knn_wcos_best_k, knn_wcos_best_acc = max(knn_wcos.items(), key=lambda kv: kv[1])

    metrics = {
        "cos": ncm_cosine(xtr, ytr, xte, yte, classes),
        "centered_cos": centered_cosine(xtr, ytr, xte, yte, classes),
        "euclidean_ncm": euclidean_ncm(xtr, ytr, xte, yte, classes),
        "lda_oracle": {"acc": lda_oracle_acc, "reg": lda_oracle_reg, "test_scores": lda_oracle_rows},
        "lda_trainval": lda_trainval,
        "wcos_oracle": {"acc": wcos_oracle_acc, "eps": wcos_oracle_eps, "test_scores": wcos_oracle_rows},
        "wcos_trainval": wcos_trainval,
        "qda_diag_trainval": qda_diag_trainval,
        "ridge_trainval": ridge_trainval,
        "knn_cos_trainval": knn_cos_trainval,
        "knn_wcos_oracle": {
            "acc": float(knn_wcos_best_acc),
            "selected": int(knn_wcos_best_k),
            "test_scores": knn_wcos,
            "selection": "test_oracle_on_wcos_space",
        },
    }
    metrics["gaps_pp"] = {
        "lda_oracle_minus_cos": 100.0 * (metrics["lda_oracle"]["acc"] - metrics["cos"]),
        "lda_trainval_minus_cos": 100.0 * (metrics["lda_trainval"]["acc"] - metrics["cos"]),
        "wcos_trainval_minus_cos": 100.0 * (metrics["wcos_trainval"]["acc"] - metrics["cos"]),
        "ridge_minus_cos": 100.0 * (metrics["ridge_trainval"]["acc"] - metrics["cos"]),
        "oracle_minus_trainval_lda": 100.0 * (metrics["lda_oracle"]["acc"] - metrics["lda_trainval"]["acc"]),
    }
    reference_acc = metrics["ridge_trainval"]["acc"]
    metrics["reference"] = {
        "name": "ridge_trainval",
        "acc": reference_acc,
        "cos_regret_pp": 100.0 * (reference_acc - metrics["cos"]),
        "best_whitening_regret_pp": 100.0 * (reference_acc - max(metrics["lda_trainval"]["acc"], metrics["wcos_trainval"]["acc"])),
    }
    return {
        "dataset": dataset,
        "shot": shot,
        "seed": int(seed),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_classes": int(len(classes)),
        "validation_split": split,
        "tuning_protocol": {
            "train_only": True,
            "equal_budget_note": "LDA, WCos, QDA-diag, and Ridge use explicit training-validation selection when validation is available; KNN uses its discrete k grid.",
            "lda_budget": len(args.lda_regs) if x_val is not None else 0,
            "wcos_budget": len(args.wcos_eps) if x_val is not None else 0,
            "ridge_budget": len(args.ridge_alphas) if x_val is not None else 0,
            "knn_budget": len(args.knn_ks) if x_val is not None else 0,
        },
        "metrics": metrics,
    }


def mean_std(vals, scale=100.0):
    arr = np.array(vals, dtype=float) * scale
    if len(arr) == 1:
        return f"{arr.mean():.2f}"
    return f"{arr.mean():.2f}±{arr.std(ddof=1):.2f}"


def write_report(records, path: Path):
    lines = [f"# Step B Cached Eval: {records[0]['dataset']} shot={records[0]['shot']}\n"]
    lines.append(f"- records: {len(records)}")
    lines.append(f"- validation: {records[0]['validation_split']['mode']}")
    lines.append("\n| metric | value |")
    lines.append("|---|---:|")
    keys = [
        ("cos", lambda r: r["metrics"]["cos"]),
        ("centered_cos", lambda r: r["metrics"]["centered_cos"]),
        ("euclidean_ncm", lambda r: r["metrics"]["euclidean_ncm"]),
        ("lda_oracle", lambda r: r["metrics"]["lda_oracle"]["acc"]),
        ("lda_trainval", lambda r: r["metrics"]["lda_trainval"]["acc"]),
        ("wcos_trainval", lambda r: r["metrics"]["wcos_trainval"]["acc"]),
        ("ridge_trainval", lambda r: r["metrics"]["ridge_trainval"]["acc"]),
        ("qda_diag_trainval", lambda r: r["metrics"]["qda_diag_trainval"]["acc"]),
        ("knn_cos_trainval", lambda r: r["metrics"]["knn_cos_trainval"]["acc"]),
        ("oracle_minus_trainval_lda_pp", lambda r: r["metrics"]["gaps_pp"]["oracle_minus_trainval_lda"] / 100.0),
        ("ridge_minus_cos_pp", lambda r: r["metrics"]["gaps_pp"]["ridge_minus_cos"] / 100.0),
    ]
    for name, fn in keys:
        scale = 1.0 if name.endswith("_pp") else 100.0
        lines.append(f"| {name} | {mean_std([fn(r) for r in records], scale=scale)} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(args):
    seeds = [0] if args.shot == "full" else args.seeds
    records = [run_one(args.dataset, args.shot, seed, args) for seed in seeds]
    out_dir = Path(args.out_dir) / args.exp_name / args.dataset / f"shot_{args.shot}"
    save_json({"dataset": args.dataset, "shot": args.shot, "records": records}, out_dir / "result.json")
    write_report(records, out_dir / "result.md")
    print(f"[stepb] wrote {out_dir}")


def aggregate(args):
    root = Path(args.out_dir) / args.exp_name
    records = []
    for p in sorted(root.glob("*/*/result.json")):
        obj = json.load(open(p))
        records.extend(obj["records"])
    save_json(records, root / "aggregate_records.json")
    groups = {}
    for r in records:
        groups.setdefault((r["dataset"], str(r["shot"])), []).append(r)

    order = {"1": 1, "2": 2, "5": 5, "10": 10, "20": 20, "full": 999}
    group_rows = []
    lines = [f"# Step B Cached Eval Aggregate ({args.exp_name})\n"]
    lines.append("| dataset | shot | n | Cos | C-Cos | Euc | WCos train | LDA train | LDA oracle | oracle-train | Ridge ref | QDA diag | kNN cos | Ridge-Cos |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (dataset, shot), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 1000))):
        row = {
            "dataset": dataset,
            "shot": shot,
            "n": len(rs),
            "cos": float(np.mean([r["metrics"]["cos"] for r in rs])),
            "centered_cos": float(np.mean([r["metrics"]["centered_cos"] for r in rs])),
            "euclidean_ncm": float(np.mean([r["metrics"]["euclidean_ncm"] for r in rs])),
            "wcos_trainval": float(np.mean([r["metrics"]["wcos_trainval"]["acc"] for r in rs])),
            "lda_trainval": float(np.mean([r["metrics"]["lda_trainval"]["acc"] for r in rs])),
            "lda_oracle": float(np.mean([r["metrics"]["lda_oracle"]["acc"] for r in rs])),
            "ridge_trainval": float(np.mean([r["metrics"]["ridge_trainval"]["acc"] for r in rs])),
            "qda_diag_trainval": float(np.mean([r["metrics"]["qda_diag_trainval"]["acc"] for r in rs])),
            "knn_cos_trainval": float(np.mean([r["metrics"]["knn_cos_trainval"]["acc"] for r in rs])),
            "oracle_minus_trainval_lda_pp": float(np.mean([r["metrics"]["gaps_pp"]["oracle_minus_trainval_lda"] for r in rs])),
            "ridge_minus_cos_pp": float(np.mean([r["metrics"]["gaps_pp"]["ridge_minus_cos"] for r in rs])),
            "best_whitening": float(np.mean([max(r["metrics"]["lda_trainval"]["acc"], r["metrics"]["wcos_trainval"]["acc"]) for r in rs])),
        }
        row["best_whitening_minus_cos_pp"] = 100.0 * (row["best_whitening"] - row["cos"])
        group_rows.append(row)
        lines.append(
            f"| {dataset} | {shot} | {len(rs)} | "
            f"{mean_std([r['metrics']['cos'] for r in rs])} | "
            f"{mean_std([r['metrics']['centered_cos'] for r in rs])} | "
            f"{mean_std([r['metrics']['euclidean_ncm'] for r in rs])} | "
            f"{mean_std([r['metrics']['wcos_trainval']['acc'] for r in rs])} | "
            f"{mean_std([r['metrics']['lda_trainval']['acc'] for r in rs])} | "
            f"{mean_std([r['metrics']['lda_oracle']['acc'] for r in rs])} | "
            f"{mean_std([r['metrics']['gaps_pp']['oracle_minus_trainval_lda'] for r in rs], scale=1.0)} | "
            f"{mean_std([r['metrics']['ridge_trainval']['acc'] for r in rs])} | "
            f"{mean_std([r['metrics']['qda_diag_trainval']['acc'] for r in rs])} | "
            f"{mean_std([r['metrics']['knn_cos_trainval']['acc'] for r in rs])} | "
            f"{mean_std([r['metrics']['gaps_pp']['ridge_minus_cos'] for r in rs], scale=1.0)} |"
        )
    save_json(group_rows, root / "aggregate_groups.json")
    (root / "aggregate.md").write_text("\n".join(lines) + "\n")

    review_lines = ["# Step B Summary\n"]
    review_lines.append("- Main column for LDA should be `lda_trainval`, not `lda_oracle`.")
    review_lines.append("- `oracle-train` quantifies how much the previous test-selected upper bound inflated LDA.")
    review_lines.append("- `Ridge-Cos` is the first reference-evaluator regret proxy; Step B predictive/ranking analysis will use it explicitly.")
    review_lines.append(f"- records: {len(records)}")
    review_lines.append(f"- groups: {len(group_rows)}")
    (root / "review_summary.md").write_text("\n".join(review_lines) + "\n")
    print(f"[stepb-aggregate] records={len(records)} groups={len(group_rows)} -> {root / 'aggregate.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "aggregate"], required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--shot", default="full")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--exp-name", default="stepb_cached_eval_v1")
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    ap.add_argument("--train-cache", help="Optional train.pt cache with pooled/labels tensors.")
    ap.add_argument("--test-cache", help="Optional test.pt cache with pooled/labels tensors.")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-val-per-class", type=int, default=50)
    ap.add_argument("--lda-regs", nargs="+", type=float, default=LDA_REGS)
    ap.add_argument("--wcos-eps", nargs="+", type=float, default=WCOS_EPS)
    ap.add_argument("--ridge-alphas", nargs="+", type=float, default=RIDGE_ALPHAS)
    ap.add_argument("--knn-ks", nargs="+", type=int, default=KNN_KS)
    ap.add_argument("--default-lda-reg", type=float, default=DEFAULT_LDA_REG)
    ap.add_argument("--default-wcos-eps", type=float, default=DEFAULT_WCOS_EPS)
    ap.add_argument("--default-ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    ap.add_argument("--default-knn-k", type=int, default=DEFAULT_KNN_K)
    args = ap.parse_args()
    if args.mode == "run":
        if args.dataset is None:
            raise ValueError("--dataset required")
        run(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
