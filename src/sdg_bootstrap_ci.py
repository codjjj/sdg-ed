#!/usr/bin/env python3
"""Bootstrap confidence intervals for SDG-ED cached-feature evaluators.

This is a reporting pass: it reuses the train-only hyperparameters selected by
`sdg_stepb_cached_eval.py` and only recomputes per-example predictions so that
headline full-shot accuracies and gaps can receive test-set bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spectral_decision_gap_ed import OUT_ROOT, classes_from, load_features


DATASETS = ["aircraft", "cub200", "cars", "food101", "pets", "cifar100", "cifar10", "imagenet100", "tiny_imagenet"]


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


def load_stepb_record(root: Path, dataset: str):
    path = root / "stepb_cached_eval_v1" / dataset / "shot_full" / "result.json"
    payload = json.load(open(path))
    return payload["records"][0]


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def pred_cosine(xtr, ytr, xte, classes):
    means = np.stack([xtr[ytr == c].mean(0) for c in classes])
    pred = np.argmax(l2_normalize(xte) @ l2_normalize(means).T, axis=1)
    return np.array([classes[i] for i in pred])


def pred_centered_cosine(xtr, ytr, xte, classes):
    mean = xtr.mean(0)
    return pred_cosine(xtr - mean, ytr, xte - mean, classes)


def pred_euclidean(xtr, ytr, xte, classes):
    means = np.stack([xtr[ytr == c].mean(0) for c in classes])
    dist = np.sum(xte * xte, axis=1, keepdims=True) + np.sum(means * means, axis=1)[None, :] - 2.0 * xte @ means.T
    pred = np.argmin(dist, axis=1)
    return np.array([classes[i] for i in pred])


def pooled_stats(xtr, ytr, classes):
    d = xtr.shape[1]
    scatter = np.zeros((d, d), dtype=np.float64)
    means = []
    n = 0
    for c in classes:
        xc = xtr[ytr == c]
        mu = xc.mean(0)
        means.append(mu)
        scatter += (xc - mu).T @ (xc - mu)
        n += len(xc)
    return np.stack(means), scatter / max(1, n - len(classes))


def pred_lda(xtr, ytr, xte, classes, reg):
    means, cov = pooled_stats(xtr, ytr, classes)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    inv_cov = (vecs * (1.0 / (vals + reg))) @ vecs.T
    weights = means @ inv_cov
    bias = -0.5 * np.sum(means * weights, axis=1)
    pred = np.argmax(xte @ weights.T + bias[None, :], axis=1)
    return np.array([classes[i] for i in pred])


def whiten_pca(xtr, xte, eps):
    mean = xtr.mean(0)
    xc = xtr - mean
    cov = (xc.T @ xc) / max(1, len(xc) - 1)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    scale = 1.0 / np.sqrt(vals + eps)
    return ((xtr - mean) @ vecs) * scale[None, :], ((xte - mean) @ vecs) * scale[None, :]


def pred_wcos(xtr, ytr, xte, classes, eps):
    ztr, zte = whiten_pca(xtr, xte, eps)
    return pred_cosine(ztr, ytr, zte, classes)


def pred_qda_diag(xtr, ytr, xte, classes, reg):
    means = []
    vars_ = []
    for c in classes:
        xc = xtr[ytr == c]
        means.append(xc.mean(0))
        vars_.append(xc.var(0) + reg)
    means = np.stack(means)
    vars_ = np.stack(vars_)
    scores = []
    for mu, var in zip(means, vars_):
        scores.append(-0.5 * (((xte - mu) ** 2 / var).sum(1) + np.log(var).sum()))
    pred = np.argmax(np.stack(scores, axis=1), axis=1)
    return np.array([classes[i] for i in pred])


def pred_ridge(xtr, ytr, xte, alpha):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha))
    clf.fit(xtr, ytr)
    return clf.predict(xte)


def pred_knn_cos(xtr, ytr, xte, classes, k, batch_size=512):
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([label_to_idx[int(c)] for c in ytr], dtype=np.int64)
    train = l2_normalize(np.asarray(xtr, dtype=np.float32))
    test = l2_normalize(np.asarray(xte, dtype=np.float32))
    preds = []
    kk = min(k, len(ytr))
    for lo in range(0, len(test), batch_size):
        sim = test[lo:lo + batch_size] @ train.T
        top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        for row in top:
            counts = np.bincount(y_idx[row], minlength=len(classes))
            preds.append(classes[int(np.argmax(counts))])
    return np.array(preds)


def bootstrap(values, n_boot, seed):
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[i] = values[idx].mean()
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def fmt_ci(mean, ci):
    return f"{100*mean:.2f} [{100*ci[0]:.2f},{100*ci[1]:.2f}]"


def run_dataset(dataset, args):
    print(f"[ci] {dataset}")
    xtr, ytr, xte, yte = load_features(dataset)
    classes = classes_from(ytr, yte)
    record = load_stepb_record(Path(args.root), dataset)
    metrics = record["metrics"]

    selected = {
        "wcos_eps": float(metrics["wcos_trainval"]["selected"]),
        "lda_reg": float(metrics["lda_trainval"]["selected"]),
        "ridge_alpha": float(metrics["ridge_trainval"]["selected"]),
        "qda_diag_reg": float(metrics["qda_diag_trainval"]["selected"]),
        "knn_k": int(metrics["knn_cos_trainval"]["selected"]),
    }
    preds = {
        "cos": pred_cosine(xtr, ytr, xte, classes),
        "centered_cos": pred_centered_cosine(xtr, ytr, xte, classes),
        "euclidean_ncm": pred_euclidean(xtr, ytr, xte, classes),
        "wcos_tv": pred_wcos(xtr, ytr, xte, classes, selected["wcos_eps"]),
        "lda_tv": pred_lda(xtr, ytr, xte, classes, selected["lda_reg"]),
        "ridge": pred_ridge(xtr, ytr, xte, selected["ridge_alpha"]),
        "qda_diag": pred_qda_diag(xtr, ytr, xte, classes, selected["qda_diag_reg"]),
        "knn_cos": pred_knn_cos(xtr, ytr, xte, classes, selected["knn_k"], batch_size=args.knn_batch_size),
    }
    correct = {k: (v == yte).astype(np.float64) for k, v in preds.items()}
    rows = {}
    for name, vals in correct.items():
        rows[name] = {
            "acc": float(vals.mean()),
            "ci95": bootstrap(vals, args.n_boot, args.seed),
        }
    for name in ["centered_cos", "euclidean_ncm", "wcos_tv", "lda_tv", "ridge", "qda_diag", "knn_cos"]:
        diff = correct[name] - correct["cos"]
        rows[f"{name}_minus_cos_pp"] = {
            "mean_pp": float(100.0 * diff.mean()),
            "ci95_pp": [100.0 * x for x in bootstrap(diff, args.n_boot, args.seed + 17)],
        }
    return {
        "dataset": dataset,
        "n_test": int(len(yte)),
        "selected": selected,
        "metrics": rows,
    }


def write_report(results, path: Path):
    lines = ["# Full-shot Bootstrap CIs\n"]
    lines.append("CIs are test-example bootstrap intervals using the train-only hyperparameters already selected in StepB; no test labels are used for tuning.")
    lines.append("")
    lines.append("| dataset | Cos | WCos-TV | LDA-TV | LDA-Cos pp | Ridge-Cos pp | QDA-Cos pp | kNN-Cos pp |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        m = r["metrics"]
        lines.append(
            f"| {r['dataset']} | {fmt_ci(m['cos']['acc'], m['cos']['ci95'])} | "
            f"{fmt_ci(m['wcos_tv']['acc'], m['wcos_tv']['ci95'])} | "
            f"{fmt_ci(m['lda_tv']['acc'], m['lda_tv']['ci95'])} | "
            f"{m['lda_tv_minus_cos_pp']['mean_pp']:.2f} [{m['lda_tv_minus_cos_pp']['ci95_pp'][0]:.2f},{m['lda_tv_minus_cos_pp']['ci95_pp'][1]:.2f}] | "
            f"{m['ridge_minus_cos_pp']['mean_pp']:.2f} [{m['ridge_minus_cos_pp']['ci95_pp'][0]:.2f},{m['ridge_minus_cos_pp']['ci95_pp'][1]:.2f}] | "
            f"{m['qda_diag_minus_cos_pp']['mean_pp']:.2f} [{m['qda_diag_minus_cos_pp']['ci95_pp'][0]:.2f},{m['qda_diag_minus_cos_pp']['ci95_pp'][1]:.2f}] | "
            f"{m['knn_cos_minus_cos_pp']['mean_pp']:.2f} [{m['knn_cos_minus_cos_pp']['ci95_pp'][0]:.2f},{m['knn_cos_minus_cos_pp']['ci95_pp'][1]:.2f}] |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(OUT_ROOT))
    ap.add_argument("--exp-name", default="bootstrap_ci_v1")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--knn-batch-size", type=int, default=512)
    args = ap.parse_args()

    out_dir = Path(args.root) / args.exp_name
    results = [run_dataset(ds, args) for ds in args.datasets]
    save_json(results, out_dir / "fullshot_ci.json")
    write_report(results, out_dir / "aggregate.md")
    print(f"[ci] wrote {out_dir / 'aggregate.md'}")


if __name__ == "__main__":
    main()
