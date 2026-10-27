#!/usr/bin/env python3
"""Aggregate SDG scope-expansion datasets (additional unseen domains).

The scope-expansion experiments add closed-set domains beyond the original
nine natural-image datasets while keeping the same frozen-feature evaluator
intervention.  This script consumes cached Step-B outputs and full-feature
spectral diagnostics, then writes a compact markdown summary and an optional
LaTeX table for the paper appendix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sdg_paths import FEATURE_ROOT, OUTPUT_ROOT


DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "svhn": "SVHN",
    "gtsrb": "GTSRB",
    "stl10": "STL-10",
    "pcam": "PCAM",
}

DATASET_FAMILY = {
    "mnist": "digits",
    "fashion_mnist": "apparel",
    "svhn": "street digits",
    "gtsrb": "traffic signs",
    "stl10": "small objects",
    "pcam": "histology",
}


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def fmt(x, suffix="", missing="--"):
    if x is None or not np.isfinite(float(x)):
        return missing
    return f"{float(x):.2f}{suffix}"


def load_feature_result(feature_root: Path, dataset: str):
    path = feature_root / dataset / "result.json"
    if not path.exists():
        return None
    r = read_json(path)
    return {
        "n_train": r.get("n_train"),
        "n_test": r.get("n_test"),
        "n_classes": r.get("n_classes"),
        "top10_var_pct": 100.0 * r["anisotropy"]["top10_var_pct"],
        "tail128_fisher_pct": 100.0 * r["fisher"]["tail128_mass"],
        "spectral_full_cos": 100.0 * r["classifier_matrix"]["cosine"],
        "spectral_full_mahal": 100.0 * r["classifier_matrix"]["mahalanobis"],
        "spectral_full_wcos": 100.0 * r["classifier_matrix"]["whitened_cosine"],
        "spectral_full_gap": r["classifier_matrix"]["mahal_minus_cos_pp"],
    }


def load_stepb_groups(stepb_root: Path):
    path = stepb_root / "aggregate_groups.json"
    if path.exists():
        return read_json(path)
    return []


def summarize_dataset(dataset: str, groups, feature_row):
    by_shot = {str(g["shot"]): g for g in groups if g["dataset"] == dataset}
    full = by_shot.get("full")
    finite = [g for shot, g in by_shot.items() if shot != "full"]
    row = {
        "dataset": dataset,
        "label": DATASET_LABELS.get(dataset, dataset),
        "family": DATASET_FAMILY.get(dataset, "extra"),
        "n_train": feature_row.get("n_train") if feature_row else None,
        "n_test": feature_row.get("n_test") if feature_row else None,
        "n_classes": feature_row.get("n_classes") if feature_row else None,
        "tail128_fisher_pct": feature_row.get("tail128_fisher_pct") if feature_row else None,
        "top10_var_pct": feature_row.get("top10_var_pct") if feature_row else None,
        "full_cos": full.get("cos", np.nan) * 100.0 if full else np.nan,
        "full_best_white": full.get("best_whitening", np.nan) * 100.0 if full else np.nan,
        "full_ridge": full.get("ridge_trainval", np.nan) * 100.0 if full else np.nan,
        "full_best_white_minus_cos_pp": full.get("best_whitening_minus_cos_pp", np.nan) if full else np.nan,
        "full_ridge_minus_cos_pp": full.get("ridge_minus_cos_pp", np.nan) if full else np.nan,
        "mean_fewshot_best_white_minus_cos_pp": np.nan,
        "mean_fewshot_ridge_minus_cos_pp": np.nan,
        "n_stepb_groups": len(by_shot),
    }
    if finite:
        row["mean_fewshot_best_white_minus_cos_pp"] = float(np.mean([g["best_whitening_minus_cos_pp"] for g in finite]))
        row["mean_fewshot_ridge_minus_cos_pp"] = float(np.mean([g["ridge_minus_cos_pp"] for g in finite]))
    return row


def write_markdown(rows, out_path: Path):
    lines = ["# SDG Scope-Expansion Aggregate\n"]
    lines.append("| Dataset | Family | train/test/classes | tail Fisher@128 | full Cos | full best white | full ridge | white-Cos | few-shot white-Cos |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        size = "--" if r["n_train"] is None else f"{r['n_train']}/{r['n_test']}/{r['n_classes']}"
        lines.append(
            f"| {r['label']} | {r['family']} | {size} | "
            f"{fmt(r['tail128_fisher_pct'])} | {fmt(r['full_cos'])} | "
            f"{fmt(r['full_best_white'])} | {fmt(r['full_ridge'])} | "
            f"{fmt(r['full_best_white_minus_cos_pp'], 'pp')} | "
            f"{fmt(r['mean_fewshot_best_white_minus_cos_pp'], 'pp')} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def write_tex(rows, out_path: Path):
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Scope-expansion closed-set datasets under the same train-only evaluator protocol. The extra domains are used as boundary checks for the SDG audit rather than as a new leaderboard.}",
        "\\label{tab:scope_expansion}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Dataset & Family & Classes & Tail Fisher@128 & Full Cos & Full best white & Full ridge \\\\",
        "\\midrule",
    ]
    for r in rows:
        classes = "--" if r["n_classes"] is None else str(r["n_classes"])
        lines.append(
            f"{r['label']} & {r['family']} & {classes} & "
            f"{fmt(r['tail128_fisher_pct'])} & {fmt(r['full_cos'])} & "
            f"{fmt(r['full_best_white'])} ({fmt(r['full_best_white_minus_cos_pp'], 'pp')}) & "
            f"{fmt(r['full_ridge'])} ({fmt(r['full_ridge_minus_cos_pp'], 'pp')}) \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feature-root",
        default=str(FEATURE_ROOT),
        help="spectral_generalization.py output root holding <dataset>/result.json",
    )
    ap.add_argument("--stepb-root", default=str(OUTPUT_ROOT / "scope_expansion_stepb_v1"))
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fashion_mnist", "svhn", "gtsrb", "stl10", "pcam"])
    ap.add_argument("--out-dir", default=str(OUTPUT_ROOT / "scope_expansion"))
    ap.add_argument("--paper-dir", default=str(OUTPUT_ROOT / "scope_expansion"))
    args = ap.parse_args()

    feature_root = Path(args.feature_root)
    stepb_root = Path(args.stepb_root)
    groups = load_stepb_groups(stepb_root)
    rows = []
    for dataset in args.datasets:
        feature_row = load_feature_result(feature_root, dataset)
        if feature_row is None and not any(g["dataset"] == dataset for g in groups):
            continue
        rows.append(summarize_dataset(dataset, groups, feature_row))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scope_expansion_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_markdown(rows, out_dir / "scope_expansion_summary.md")
    write_tex(rows, Path(args.paper_dir) / "tables/table_scope_expansion.tex")
    print(f"[scope-aggregate] rows={len(rows)} -> {out_dir / 'scope_expansion_summary.md'}")


if __name__ == "__main__":
    main()
