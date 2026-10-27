#!/usr/bin/env python3
"""Aggregate backbone robustness results for the SDG-ED paper."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(os.environ.get("SDG_RESULTS_ROOT", RELEASE_ROOT / "artifacts/results"))
# DINOv2-B per-dataset result.json files are written by spectral_generalization.py
# (e.g. features/<dataset>/result.json); they are not bundled.
DINO_ROOT = Path(os.environ.get("SDG_FEATURE_ROOT", RELEASE_ROOT / "features"))
CLIP_ROOT = RESULTS_ROOT / "backbone/clip_vit_b32"
OUT_ROOT = Path(os.environ.get("SDG_OUTPUT_ROOT", RELEASE_ROOT / "outputs")) / "backbone_comparison"
DATASETS = ["cub200", "aircraft", "cars", "food101", "pets", "cifar100", "cifar10", "tiny_imagenet", "imagenet100"]
LABELS = {
    "cub200": "CUB-200",
    "aircraft": "Aircraft",
    "cars": "Cars",
    "food101": "Food-101",
    "pets": "Pets",
    "cifar100": "CIFAR-100",
    "cifar10": "CIFAR-10",
    "tiny_imagenet": "Tiny-ImageNet",
    "imagenet100": "ImageNet-100",
}


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path: Path):
    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text(json.dumps(obj, indent=2, default=float) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    shutil.copy2(tmp, path)
    tmp.unlink(missing_ok=True)


def save_text(text: str, path: Path):
    tmp = Path(tempfile.mktemp(suffix=".md"))
    tmp.write_text(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    shutil.copy2(tmp, path)
    tmp.unlink(missing_ok=True)


def summarize(path: Path):
    r = read_json(path)
    dim = r["pca_ablation"]["all"]["dim"]
    cls = r["classifier_matrix"]
    return {
        "dim": dim,
        "top10_var_pct": r["anisotropy"]["top10_var_pct"],
        "tail128_fisher_mass": r["fisher"]["tail128_mass"],
        "spearman_logeig_logfisher": r["fisher"]["spearman_logeig_logfisher"],
        "cosine": cls["cosine"],
        "mahalanobis": cls["mahalanobis"],
        "whitened_cosine": cls["whitened_cosine"],
        "mahal_minus_cos_pp": cls["mahal_minus_cos_pp"],
    }


def pct(x):
    return "-" if x is None else f"{x * 100:.2f}"


def pp(x):
    return "-" if x is None else f"{x:.2f}"


def build_rows(dino_root: Path, clip_root: Path):
    rows = []
    for dataset in DATASETS:
        dino_path = dino_root / dataset / "result.json"
        clip_path = clip_root / dataset / "result.json"
        dino = summarize(dino_path) if dino_path.exists() else None
        clip = summarize(clip_path) if clip_path.exists() else None
        valid_clip = clip is not None and clip["dim"] == 512
        rows.append({
            "dataset": dataset,
            "label": LABELS[dataset],
            "status": "done" if valid_clip else ("stale_or_wrong_dim" if clip else "missing"),
            "dino": dino,
            "clip": clip,
            "gap_delta_pp": None if dino is None or not valid_clip else clip["mahal_minus_cos_pp"] - dino["mahal_minus_cos_pp"],
        })
    return rows


def write_report(rows, out_dir: Path):
    lines = ["# CLIP ViT-B/32 vs DINOv2-B Backbone Robustness\n"]
    lines.append("| Dataset | Status | DINO dim | CLIP dim | DINO Cos | DINO Mah | DINO gap | CLIP Cos | CLIP Mah | CLIP gap | Δ gap |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        d = r["dino"]
        c = r["clip"]
        valid = r["status"] == "done"
        lines.append(
            f"| {r['label']} | {r['status']} | "
            f"{d['dim'] if d else '-'} | {c['dim'] if c else '-'} | "
            f"{pct(d['cosine']) if d else '-'} | {pct(d['mahalanobis']) if d else '-'} | {pp(d['mahal_minus_cos_pp']) if d else '-'} | "
            f"{pct(c['cosine']) if valid else '-'} | {pct(c['mahalanobis']) if valid else '-'} | {pp(c['mahal_minus_cos_pp']) if valid else '-'} | {pp(r['gap_delta_pp'])} |"
        )
    lines.append("")
    done = [r for r in rows if r["status"] == "done"]
    stale = [r["label"] for r in rows if r["status"] == "stale_or_wrong_dim"]
    missing = [r["label"] for r in rows if r["status"] == "missing"]
    lines.append(f"- valid CLIP rows: {len(done)}/{len(rows)}")
    if stale:
        lines.append(f"- stale/wrong-dim rows needing rerun: {', '.join(stale)}")
    if missing:
        lines.append(f"- missing rows: {', '.join(missing)}")
    lines.append("")
    save_text("\n".join(lines) + "\n", out_dir / "backbone_comparison.md")


def write_plot(rows, out_dir: Path):
    done = [r for r in rows if r["status"] == "done" and r["dino"] is not None]
    if not done:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([r["dino"]["mahal_minus_cos_pp"] for r in done])
    y = np.array([r["clip"]["mahal_minus_cos_pp"] for r in done])
    labels = [r["label"] for r in done]
    lo = min(x.min(), y.min()) - 2
    hi = max(x.max(), y.max()) + 2

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.scatter(x, y, s=70)
    ax.plot([lo, hi], [lo, hi], color="0.65", lw=1)
    for xi, yi, lab in zip(x, y, labels):
        ax.text(xi + 0.35, yi, lab, fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("DINOv2-B Mahal-Cos gap (pp)")
    ax.set_ylabel("CLIP ViT-B/32 Mahal-Cos gap (pp)")
    ax.set_title("SDG is backbone-sensitive but measurable")
    fig.tight_layout()
    tmp = Path(tempfile.mktemp(suffix=".png"))
    fig.savefig(tmp, dpi=200)
    target = out_dir / "clip_vs_dinov2_gap.png"
    target.unlink(missing_ok=True)
    shutil.copy2(tmp, target)
    tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino-root", default=str(DINO_ROOT))
    ap.add_argument("--clip-root", default=str(CLIP_ROOT))
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    rows = build_rows(Path(args.dino_root), Path(args.clip_root))
    save_json(rows, out_dir / "backbone_comparison.json")
    write_report(rows, out_dir)
    write_plot(rows, out_dir)
    print(f"[backbone-compare] wrote {out_dir / 'backbone_comparison.md'}")


if __name__ == "__main__":
    main()
