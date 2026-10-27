#!/usr/bin/env python3
"""Render SDG-ED manuscript figures and LaTeX tables from result artifacts."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.environ.get("SDG_ASSET_OUTPUT_ROOT", PAPER_ROOT / "outputs/paper_assets"))
FIG_DIR = ASSET_ROOT / "figures"
TABLE_DIR = ASSET_ROOT / "tables"
RESULTS_ROOT = Path(os.environ.get("SDG_RESULTS_ROOT", PAPER_ROOT / "artifacts/results"))

LABELS = {
    "aircraft": "Aircraft",
    "cub200": "CUB-200",
    "cars": "Cars",
    "food101": "Food-101",
    "pets": "Pets",
    "cifar100": "CIFAR-100",
    "cifar10": "CIFAR-10",
    "tiny_imagenet": "Tiny-ImageNet",
    "imagenet100": "ImageNet-100",
}

FULL_ORDER = [
    "aircraft",
    "cub200",
    "cars",
    "food101",
    "pets",
    "cifar100",
    "cifar10",
    "imagenet100",
    "tiny_imagenet",
]


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def metric_acc(row: dict, key: str) -> float:
    metrics = row["metrics"]
    if key == "cos":
        return metrics["cos"]
    if key == "mahal":
        if isinstance(metrics.get("mahal"), dict):
            return metrics["mahal"]["acc"]
        return metrics["mahal_best"]
    if key == "wcos":
        if "wcos" in metrics:
            return metrics["wcos"]
        return metrics["wcos_best"]
    if key == "fecam_diag":
        return metrics["fecam_diag"]
    if key == "logreg":
        value = metrics.get("logreg")
        if isinstance(value, dict):
            return value["acc"]
        return value
    if key == "mlp":
        value = metrics.get("mlp")
        if isinstance(value, dict):
            return value.get("acc")
        return value
    raise KeyError(key)


def method_dataframe() -> pd.DataFrame:
    rows = []
    for row in read_json(RESULTS_ROOT / "method_matrix_aggregate.json"):
        cos = metric_acc(row, "cos")
        mahal = metric_acc(row, "mahal")
        rows.append(
            {
                "dataset": row["dataset"],
                "label": LABELS[row["dataset"]],
                "shot": row["shot"],
                "shot_num": 999 if row["shot"] == "full" else int(row["shot"]),
                "seed": row["seed"],
                "cos": 100 * cos,
                "mahal": 100 * mahal,
                "gap": 100 * (mahal - cos),
                "wcos": 100 * metric_acc(row, "wcos"),
                "fecam_diag": 100 * metric_acc(row, "fecam_diag"),
                "logreg": 100 * metric_acc(row, "logreg"),
                "mlp": None if metric_acc(row, "mlp") is None else 100 * metric_acc(row, "mlp"),
                "sdgi_tail128": 100 * row["sdgi"]["tail128_fisher"],
                "top10_var": 100 * row["sdgi"]["top10_var"],
            }
        )
    return pd.DataFrame(rows)


def stepb_cached_dataframe() -> pd.DataFrame:
    rows = []
    path = RESULTS_ROOT / "stepb_cached_eval_v1/aggregate_groups.json"
    for row in read_json(path):
        shot = row["shot"]
        rows.append(
            {
                "dataset": row["dataset"],
                "label": LABELS[row["dataset"]],
                "shot": shot,
                "shot_num": 999 if shot == "full" else int(shot),
                "n": row["n"],
                "cos": 100 * row["cos"],
                "centered_cos": 100 * row["centered_cos"],
                "euclidean_ncm": 100 * row["euclidean_ncm"],
                "wcos": 100 * row["wcos_trainval"],
                "lda_trainval": 100 * row["lda_trainval"],
                "lda_oracle": 100 * row["lda_oracle"],
                "lda_gap": 100 * (row["lda_trainval"] - row["cos"]),
                "best_whitening": 100 * row["best_whitening"],
                "best_whitening_gap": row["best_whitening_minus_cos_pp"],
                "oracle_delta": row["oracle_minus_trainval_lda_pp"],
                "ridge": 100 * row["ridge_trainval"],
                "ridge_gap": row["ridge_minus_cos_pp"],
                "qda_diag": 100 * row["qda_diag_trainval"],
                "knn_cos": 100 * row["knn_cos_trainval"],
            }
        )
    return pd.DataFrame(rows)


def stepb_cached_records_dataframe() -> pd.DataFrame:
    rows = []
    for path in sorted((RESULTS_ROOT / "stepb_cached_eval_v1").glob("*/shot_*/result.json")):
        payload = read_json(path)
        for row in payload["records"]:
            metrics = row["metrics"]
            shot = row["shot"]
            cos = metrics["cos"]
            lda = metrics["lda_trainval"]["acc"]
            rows.append(
                {
                    "dataset": row["dataset"],
                    "label": LABELS[row["dataset"]],
                    "shot": shot,
                    "shot_num": 999 if shot == "full" else int(shot),
                    "seed": row["seed"],
                    "cos": 100 * cos,
                    "wcos": 100 * metrics["wcos_trainval"]["acc"],
                    "lda_trainval": 100 * lda,
                    "lda_gap": 100 * (lda - cos),
                }
            )
    return pd.DataFrame(rows)


def p0_full_dataframe() -> pd.DataFrame:
    payload = read_json(RESULTS_ROOT / "p0_mechanism_v2/summary_with_pred.json")
    rows = []
    for row in payload["rows"]:
        if row["shot"] != "full":
            continue
        rows.append(
            {
                "dataset": row["dataset"],
                "label": LABELS[row["dataset"]],
                "cos": 100 * row["cos"],
                "mahal": 100 * row["mahal_best"],
                "gap": 100 * (row["mahal_best"] - row["cos"]),
                "tail128_fisher": 100 * row["tail128_fisher"],
                "tail128_enrich": row["tail128_fisher_enrichment"],
                "top10_var": 100 * row["top10_var"],
                "spearman": row["sdgi_spearman_logeig_logfisher"],
            }
        )
    return pd.DataFrame(rows)


def spectral_scatter_dataframe(stepb: pd.DataFrame) -> pd.DataFrame:
    method_rows = read_json(RESULTS_ROOT / "method_matrix_aggregate.json")
    diagnostics = {}
    for row in method_rows:
        key = (row["dataset"], str(row["shot"]))
        diagnostics.setdefault(key, []).append(row)

    rows = []
    for _, row in stepb.iterrows():
        key = (row["dataset"], str(row["shot"]))
        ms = diagnostics[key]
        rows.append(
            {
                "dataset": row["dataset"],
                "label": row["label"],
                "shot": row["shot"],
                "shot_num": row["shot_num"],
                "best_whitening_gap": row["best_whitening_gap"],
                "lda_gap": row["lda_gap"],
                "tail128_fisher": 100 * float(np.mean([m["sdgi"]["tail128_fisher"] for m in ms])),
                "top10_var": 100 * float(np.mean([m["sdgi"]["top10_var"] for m in ms])),
            }
        )
    return pd.DataFrame(rows)


def supplement_dataframe() -> pd.DataFrame:
    path = RESULTS_ROOT / "supplement_stats_v1/aggregate_groups.json"
    rows = []
    for row in read_json(path):
        rows.append(
            {
                "dataset": row["dataset"],
                "label": LABELS[row["dataset"]],
                "shot": row["shot"],
                "shot_num": 999 if row["shot"] == "full" else int(row["shot"]),
                "n": row["n"],
                "cos": 100 * row["cos"],
                "mahal": 100 * row["mahal_best"],
                "gap": row["gap_best_pp"],
                "wcos": 100 * row["wcos_best"],
                "shrink": 100 * row["shared_shrink_best"],
                "tail128_fisher": 100 * row["tail128_fisher"],
                "tail128_enrich": row["tail128_enrichment"],
                "median_pc": row["fisher_median_pc"],
                "condition": row["condition_positive"],
                "effective_rank": row["effective_rank"],
            }
        )
    return pd.DataFrame(rows)


def p3_dataframe() -> pd.DataFrame:
    rows = []
    for row in read_json(RESULTS_ROOT / "p3_estimator_defense/aggregate.json"):
        metrics = row["metrics"]

        def best(value):
            if isinstance(value, dict):
                return max(value.values())
            return value

        rows.append(
            {
                "dataset": row["dataset"],
                "label": LABELS[row["dataset"]],
                "shot": row["shot"],
                "shot_num": 999 if row["shot"] == "full" else int(row["shot"]),
                "seed": row["seed"],
                "cos": 100 * metrics["cos"],
                "mahal": 100 * best(metrics["mahal_full"]),
                "shared_diag": 100 * metrics["shared_diag"],
                "shared_shrink": 100 * best(metrics["shared_shrink"]),
                "lowrank_diag": 100 * best(metrics["lowrank_diag"]),
                "wcos": 100 * best(metrics["wcos_eps"]),
                "fecam_diag": 100 * metrics["fecam_diag"],
                "fecam_shrink": 100 * best(metrics["fecam_shrink"]),
                "condition": row["covariance"]["condition_positive"],
                "effective_rank": row["covariance"]["effective_rank"],
            }
        )
    return pd.DataFrame(rows)


def backbone_dataframe() -> pd.DataFrame:
    rows = []
    comparison = RESULTS_ROOT / "backbone/clip_vit_b32/backbone_comparison.json"
    if comparison.exists():
        for row in read_json(comparison):
            if row["status"] != "done":
                continue
            for key, label in [("dino", "DINOv2-B"), ("clip", "CLIP-B/32")]:
                item = row[key]
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "label": row["label"],
                        "backbone": label,
                        "cos": 100 * item["cosine"],
                        "mahal": 100 * item["mahalanobis"],
                        "wcos": 100 * item["whitened_cosine"],
                        "gap": item["mahal_minus_cos_pp"],
                        "tail128_fisher": 100 * item["tail128_fisher_mass"],
                        "top10_var": 100 * item["top10_var_pct"],
                    }
                )

    hf_names = {
        "mae_vit_base": "MAE ViT-B",
        "siglip_b16": "SigLIP-B/16",
        "convnext_base": "ConvNeXt-B",
        "vit_sup_b16": "Sup. ViT-B/16",
    }
    for backbone, label in hf_names.items():
        root = RESULTS_ROOT / "backbone" / backbone
        for result_path in sorted(root.glob("*/result.json")):
            row = read_json(result_path)
            ds = row["dataset"]
            mat = row["classifier_matrix"]
            rows.append(
                {
                    "dataset": ds,
                    "label": LABELS.get(ds, ds),
                    "backbone": label,
                    "cos": 100 * mat["cosine"],
                    "mahal": 100 * mat["mahalanobis"],
                    "wcos": 100 * mat["whitened_cosine"],
                    "gap": mat["mahal_minus_cos_pp"],
                    "tail128_fisher": 100 * row["fisher"]["tail128_mass"],
                    "top10_var": 100 * row["anisotropy"]["top10_var_pct"],
                }
            )
    return pd.DataFrame(rows)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def mean_std(grouped: pd.DataFrame, col: str) -> tuple[float, float]:
    values = grouped[col].dropna().to_numpy()
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def fmt(x: float) -> str:
    return f"{x:.2f}"


def fmt_signed(x: float) -> str:
    return f"{x:+.2f}"


def fmt_pm(mean: float, std: float) -> str:
    if std == 0:
        return fmt(mean)
    return f"${mean:.2f}\\pm{std:.2f}$"


def fmt_ci(value: float, ci: dict, key: str) -> str:
    lo, hi = ci[key]
    return f"{value:.2f} [{lo:.2f},{hi:.2f}]"


def kendall_tau(x, y) -> tuple[float, int, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    concordant = 0
    discordant = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = concordant + discordant
    tau = 0.0 if total == 0 else (concordant - discordant) / total
    return float(tau), concordant, discordant


def fig0_schematic() -> None:
    rng = np.random.RandomState(7)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    colors = ["#4477aa", "#bb5566", "#55aa66"]
    left_centers = np.array([[1.8, 0.35], [2.0, -0.10], [1.65, -0.48]])
    right_centers = np.array([[-1.0, 0.75], [0.95, 0.65], [0.05, -1.0]])
    cov_left = np.array([[0.18, 0.08], [0.08, 0.05]])
    cov_right = np.array([[0.055, 0.0], [0.0, 0.055]])

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-2.4, 2.6)
        ax.set_ylim(-1.8, 1.8)
        ax.axis("off")

    for scale in [1.0, 1.45, 1.9]:
        axes[0].add_patch(Ellipse((0.95, -0.1), 3.7 * scale / 1.9, 0.7 * scale / 1.9, angle=21, fill=False, lw=1.0, edgecolor="#b8b8b8"))
    for angle in [-17, -5, 9]:
        t = np.deg2rad(angle)
        axes[0].plot([0, 2.45 * np.cos(t)], [0, 2.45 * np.sin(t)], color="#d0d0d0", lw=1.0)
    axes[0].plot(0, 0, marker="o", color="#888888", ms=3)
    for center, color in zip(left_centers, colors):
        pts = rng.multivariate_normal(center, cov_left, 42)
        axes[0].scatter(pts[:, 0], pts[:, 1], s=11, color=color, alpha=0.82, edgecolor="none")
        axes[0].scatter(center[0], center[1], s=44, color=color, edgecolor="white", linewidth=0.8, zorder=5)

    axes[1].add_patch(Circle((0, 0), 1.45, fill=False, lw=1.0, edgecolor="#b8b8b8"))
    axes[1].add_patch(Circle((0, 0), 0.95, fill=False, lw=0.9, edgecolor="#d0d0d0"))
    for slope, intercept in [(0.15, 0.0), (-1.1, 0.16), (1.0, -0.18)]:
        xs = np.array([-2.2, 2.2])
        axes[1].plot(xs, slope * xs + intercept, color="#b0b0b0", lw=1.0)
    for center, color in zip(right_centers, colors):
        pts = rng.multivariate_normal(center, cov_right, 42)
        axes[1].scatter(pts[:, 0], pts[:, 1], s=11, color=color, alpha=0.82, edgecolor="none")
        axes[1].scatter(center[0], center[1], s=44, color=color, edgecolor="white", linewidth=0.8, zorder=5)

    fig.subplots_adjust(wspace=0.04, left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(FIG_DIR / "fig0_schematic.pdf")
    fig.savefig(FIG_DIR / "fig0_schematic.png", dpi=220)
    plt.close(fig)


def table_fullshot(stepb: pd.DataFrame, method_df: pd.DataFrame) -> None:
    full = stepb[stepb["shot"] == "full"].set_index("dataset").loc[FULL_ORDER].reset_index()
    refs = method_df[method_df["shot"] == "full"].set_index("dataset")
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Full-shot frozen-feature evaluation with DINOv2-B under the train-only protocol. LDA-TV and WCos-TV use train-validated hyperparameters without test labels; $\\Delta_{\\mathrm{oracle}}$ is the remaining best-over-test LDA upper-bound gap. Ridge, LogReg, and MLP are reference probes.}",
        "\\label{tab:fullshot}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Dataset & Cos & WCos-TV & LDA-TV & SDG-TV & $\\Delta_{oracle}$ & Ridge & LogReg/MLP \\\\",
        "\\midrule",
    ]
    for _, row in full.iterrows():
        ref = refs.loc[row["dataset"]]
        mlp = "--" if pd.isna(ref["mlp"]) else fmt(ref["mlp"])
        lines.append(
            f"{row['label']} & {fmt(row['cos'])} & {fmt(row['wcos'])} & {fmt(row['lda_trainval'])} & {fmt(row['lda_gap'])} & "
            f"{fmt(row['oracle_delta'])} & {fmt(row['ridge'])} & {fmt(ref['logreg'])}/{mlp} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    write(TABLE_DIR / "table_fullshot.tex", "\n".join(lines))


def table_regime_boundary_examples(stepb: pd.DataFrame) -> None:
    points = [
        ("expanding", "cub200", 2),
        ("expanding", "cub200", 5),
        ("expanding", "cub200", 20),
        ("expanding", "aircraft", 5),
        ("expanding", "aircraft", 20),
        ("expanding", "cars", 5),
        ("expanding", "cars", 20),
        ("negative", "cifar100", 1),
        ("negative", "cifar100", 5),
        ("negative", "imagenet100", 5),
        ("negative", "tiny_imagenet", 5),
        ("negative", "tiny_imagenet", 999),
    ]
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\small",
        "\\caption{Selected train-only regime and boundary examples on DINOv2-B. Expanding rows show high-SDG fine-grained cases where shared covariance becomes useful with more support; negative rows show cases where LDA-TV trails raw cosine. Non-full rows report mean and standard deviation over 5 support-sampling seeds, while full-shot uses the official full training split once.}",
        "\\label{tab:regime_boundary_examples}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Regime & Dataset & Shot & Cos & WCos-TV & LDA-TV & SDG-TV \\\\",
        "\\midrule",
    ]
    for regime, dataset, shot in points:
        g = stepb[(stepb["dataset"] == dataset) & (stepb["shot_num"] == shot)]
        shot_label = "full" if shot == 999 else str(shot)
        vals = {col: mean_std(g, col) for col in ["cos", "wcos", "lda_trainval", "lda_gap"]}
        lines.append(
            f"{regime} & {LABELS[dataset]} & {shot_label} & {fmt_pm(*vals['cos'])} & "
            f"{fmt_pm(*vals['wcos'])} & {fmt_pm(*vals['lda_trainval'])} & {fmt_pm(*vals['lda_gap'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write(TABLE_DIR / "table_regime_boundary_examples.tex", "\n".join(lines))


def table_stepb_decomposition(stepb: pd.DataFrame) -> None:
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Full-shot estimator decomposition on DINOv2-B cached features. Entries are percentage-point changes relative to raw cosine NCM. Centering, Euclidean NCM, per-class diagonal covariance, and kNN do not reproduce the high fine-grained gains; shared whitening and ridge probes do.}",
        "\\label{tab:stepb_decomposition}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Dataset & $\\Delta$C-Cos & $\\Delta$Euc & $\\Delta$QDA diag & $\\Delta$kNN-cos & $\\Delta$WCos-TV & $\\Delta$LDA-TV & $\\Delta$Ridge \\\\",
        "\\midrule",
    ]
    full = stepb[stepb["shot_num"] == 999].set_index("dataset")
    for dataset in FULL_ORDER:
        row = full.loc[dataset]
        lines.append(
            f"{row['label']} & {fmt_signed(row['centered_cos'] - row['cos'])} & {fmt_signed(row['euclidean_ncm'] - row['cos'])} & "
            f"{fmt_signed(row['qda_diag'] - row['cos'])} & {fmt_signed(row['knn_cos'] - row['cos'])} & "
            f"{fmt_signed(row['wcos'] - row['cos'])} & {fmt_signed(row['lda_trainval'] - row['cos'])} & "
            f"{fmt_signed(row['ridge_gap'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    write(TABLE_DIR / "table_stepb_decomposition.tex", "\n".join(lines))


def table_predictive_protocol() -> None:
    rows = read_json(RESULTS_ROOT / "stepb_predictive_protocol_v1/predictive_results.json")
    keep = [
        ("target_best_whitening_minus_cos_pp", "best WCos/LDA - cosine"),
        ("target_ridge_minus_cos_pp", "ridge reference - cosine"),
    ]
    label_map = dict(keep)
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\begingroup",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Train-only predictive protocol. Rows report held-out prediction quality for accuracy-gap targets in percentage points; LOGO leaves out one dataset-shot group and LODO leaves out one dataset. Training-set spectral and covariance diagnostics are used without test labels for routing.}",
        "\\label{tab:predictive_protocol}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Target & Fold & Pearson $r$ [95\\% CI] & Spearman $\\rho$ [95\\% CI] & MAE (pp) & RMSE (pp) \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row["target"] not in label_map:
            continue
        fold = "LOGO" if row["fold"] == "leave_one_group" else "LODO"
        ci = row["ci95"]
        lines.append(
            f"{label_map[row['target']]} & {fold} & "
            f"{row['pearson']:.3f} [{ci['pearson'][0]:.3f},{ci['pearson'][1]:.3f}] & "
            f"{row['spearman']:.3f} [{ci['spearman'][0]:.3f},{ci['spearman'][1]:.3f}] & "
            f"{row['mae_pp']:.2f} & {row['rmse_pp']:.2f} \\\\"
        )
    lodo = next(r for r in rows if r["target"] == "target_best_whitening_minus_cos_pp" and r["fold"] == "leave_one_dataset")
    baseline = {r["policy"]: r for r in lodo["routing_baselines"]}
    route5 = next(r for r in lodo["routing"] if abs(r["threshold"] - 5.0) < 1e-9)
    route_ci = route5["ci95"]
    lines += [
        "\\midrule",
        "\\multicolumn{6}{p{0.98\\linewidth}}{Unit: 54 dataset-shot groups; non-full groups are 5-seed means.} \\\\",
        f"\\multicolumn{{6}}{{p{{0.98\\linewidth}}}}{{LODO 5pp router: acc {fmt_ci(route5['mean_chosen_acc'], route_ci, 'mean_chosen_acc')}, "
        f"regret {route5['routing_regret_pp']:.2f}pp [{route_ci['routing_regret_pp'][0]:.2f},{route_ci['routing_regret_pp'][1]:.2f}]; "
        f"always cosine regret {baseline['always cosine']['regret_pp']:.2f}pp, always WCos {baseline['always WCos']['regret_pp']:.2f}pp, always LDA {baseline['always LDA']['regret_pp']:.2f}pp.}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\endgroup",
        "\\end{table}",
        "",
    ]
    write(TABLE_DIR / "table_predictive_protocol.tex", "\n".join(lines))


def table_predictive_routing_baselines() -> None:
    rows = read_json(RESULTS_ROOT / "stepb_predictive_protocol_v1/predictive_results.json")
    lodo = next(r for r in rows if r["target"] == "target_best_whitening_minus_cos_pp" and r["fold"] == "leave_one_dataset")
    logo = next(r for r in rows if r["target"] == "target_best_whitening_minus_cos_pp" and r["fold"] == "leave_one_group")
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\small",
        "\\caption{Predictive routing baselines for the best-WCos/LDA-vs-cosine target. The learned router uses training-set diagnostics to decide when to add whitening-aware reporting; fixed baselines use the same evaluator for every group.}",
        "\\label{tab:predictive_routing}",
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Policy & Mean acc [95\\% CI] & Regret [95\\% CI] \\\\",
        "\\midrule",
    ]
    for fold_label, result in [("LOGO router, 5pp", logo), ("LODO router, 5pp", lodo)]:
        route5 = next(r for r in result["routing"] if abs(r["threshold"] - 5.0) < 1e-9)
        ci = route5["ci95"]
        lines.append(
            f"{fold_label} & {fmt_ci(route5['mean_chosen_acc'], ci, 'mean_chosen_acc')} & "
            f"{fmt_ci(route5['routing_regret_pp'], ci, 'routing_regret_pp')} \\\\"
        )
    for row in lodo["routing_baselines"]:
        ci = row["ci95"]
        lines.append(f"{row['policy']} & {fmt_ci(row['mean_acc'], ci, 'mean_acc')} & {fmt_ci(row['regret_pp'], ci, 'regret_pp')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write(TABLE_DIR / "table_appendix_predictive_routing.tex", "\n".join(lines))


def table_router_ablation() -> None:
    rows = read_json(RESULTS_ROOT / "stepb_predictive_protocol_v2/feature_ablation.json")
    keep = ["full", "spectral_only", "covariance_only", "variance_only", "tail_fisher_only", "condition_only", "no_covariance"]
    lines = [
        "\\begin{table*}[!htbp]",
        "\\centering",
        "\\small",
        "\\caption{Predictive-router feature ablation for the best-whitening-vs-cosine target under leave-one-dataset-out evaluation. The full row is the main-paper diagnostic set; variance-only is a weak baseline.}",
        "\\label{tab:router_ablation}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Feature set & Pearson & Spearman & MAE & RMSE & 5pp regret \\\\",
        "\\midrule",
    ]
    for name in keep:
        row = next(r for r in rows if r["feature_set"] == name and r["target"] == "target_best_whitening_minus_cos_pp" and r["fold"] == "leave_one_dataset")
        lines.append(
            f"{name.replace('_', ' ')} & {row['pearson']:.3f} & {row['spearman']:.3f} & "
            f"{row['mae_pp']:.2f} & {row['rmse_pp']:.2f} & {row['routing_regret_5pp']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write(TABLE_DIR / "table_router_ablation.tex", "\n".join(lines))


def table_heldout_router() -> None:
    path = RESULTS_ROOT / "heldout_router_v1/heldout_router_generalization.json"
    if not path.exists():
        return
    result = read_json(path)
    metrics = result["best_whitening_target"]["metrics"]
    routing = result["best_whitening_target"]["routing"]
    route_ci = routing["ci95"]
    label_map = {
        "caltech101": "Caltech-101",
        "dtd": "DTD",
        "eurosat": "EuroSAT",
        "flowers102": "Flowers-102",
    }
    lines = [
        "\\begin{table*}[!htbp]",
        "\\centering",
        "\\small",
        "\\caption{Held-out dataset validation of the predictive router. The router is fit only on the original nine-dataset suite and evaluated on four unseen datasets without refitting.}",
        "\\label{tab:heldout_router}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Dataset & Groups & Target mean (pp) & Pred. mean (pp) & MAE (pp) & Router regret (pp) \\\\",
        "\\midrule",
    ]
    for row in result["by_dataset"]:
        lines.append(
            f"{label_map.get(row['dataset'], row['dataset'])} & {row['n_groups']} & "
            f"{row['target_mean_pp']:.2f} & {row['pred_mean_pp']:.2f} & "
            f"{row['mae_pp']:.2f} & {row['router_regret_pp']:.2f} \\\\"
        )
    lines += [
        "\\midrule",
        f"All held-out & {result['n_heldout_groups']} & "
        f"\\multicolumn{{2}}{{r}}{{Pearson {metrics['pearson']:.3f}, Spearman {metrics['spearman']:.3f}}} & "
        f"{metrics['mae_pp']:.2f} & "
        f"{routing['routing_regret_pp']:.2f} [{route_ci['routing_regret_pp'][0]:.2f},{route_ci['routing_regret_pp'][1]:.2f}] \\\\",
        "\\multicolumn{6}{p{0.95\\linewidth}}{For comparison on these held-out groups, always-cosine regret is "
        + f"{next(r for r in result['routing_baselines'] if r['policy'] == 'always cosine')['regret_pp']:.2f}pp, "
        + f"always-WCos is {next(r for r in result['routing_baselines'] if r['policy'] == 'always WCos')['regret_pp']:.2f}pp, "
        + f"and always-LDA is {next(r for r in result['routing_baselines'] if r['policy'] == 'always LDA')['regret_pp']:.2f}pp.}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    write(TABLE_DIR / "table_heldout_router.tex", "\n".join(lines))


def table_hyperparameter_grids() -> None:
    lines = [
        "\\begin{table*}[!htbp]",
        "\\centering",
        "\\small",
        "\\caption{Hyperparameter grids for cached-feature evaluators. Tuned evaluators use the same inner training-validation split and are refit on the full support set before test evaluation; 1-shot uses fixed defaults because no support example is held out.}",
        "\\label{tab:hyperparameter_grids}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{p{0.22\\textwidth}p{0.47\\textwidth}p{0.22\\textwidth}}",
        "\\toprule",
        "Component & Grid / default & Selection protocol \\\\",
        "\\midrule",
        "LDA-TV $\\lambda$ & $\\{10^{-8},3\\!\\cdot\\!10^{-8},\\ldots,10^{-1}\\}$; 1-shot default $10^{-4}$ & inner validation \\\\",
        "WCos-TV $\\epsilon$ & $\\{10^{-8},3\\!\\cdot\\!10^{-8},\\ldots,10^{-1}\\}$; 1-shot default $10^{-4}$ & inner validation \\\\",
        "Ridge $\\alpha$ & $\\{10^{-4},3\\!\\cdot\\!10^{-4},\\ldots,10^{3}\\}$; 1-shot default $1$ & inner validation \\\\",
        "$k$NN-cos $k$ & $\\{1,3,5,10,20\\}$; 1-shot default $5$ & inner validation \\\\",
        "QDA-diag regularization & same logarithmic grid as LDA-TV & inner validation \\\\",
        "LogReg reference & standardized multinomial logistic regression, $C\\in\\{0.01,0.03,0.1,0.3,1,3,10\\}$ & reference probe only \\\\",
        "MLP reference & standardized 1-hidden-layer MLP, width 512, $\\alpha=10^{-4}$, lr $10^{-3}$, 80 iterations & reference probe only \\\\",
        "Estimator stress tests & shared-shrink $\\{0.1,0.25,0.5,0.75,0.9,1.0\\}$; low-rank $\\{16,64,128,256\\}$ & diagnostic appendix only \\\\",
        "Oracle diagnostic upper bound & best-over-test LDA grid, clearly labeled as oracle & not used for headline selection \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    write(TABLE_DIR / "table_hyperparameter_grids.tex", "\n".join(lines))


def pct_ci(metric: dict) -> str:
    return f"{100 * metric['acc']:.2f} [{100 * metric['ci95'][0]:.2f},{100 * metric['ci95'][1]:.2f}]"


def pp_ci(metric: dict) -> str:
    return f"{metric['mean_pp']:.2f} [{metric['ci95_pp'][0]:.2f},{metric['ci95_pp'][1]:.2f}]"


def table_bootstrap_ci() -> None:
    rows = read_json(RESULTS_ROOT / "bootstrap_ci_v1/fullshot_ci.json")
    lines = [
        "\\begin{table*}[!htbp]",
        "\\centering",
        "\\small",
        "\\caption{Full-shot test-example bootstrap 95\\% confidence intervals. Hyperparameters are the train-only selections used in the main tables; intervals quantify reporting uncertainty, not retuning uncertainty.}",
        "\\label{tab:bootstrap_ci}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Dataset & Cos & WCos-TV & LDA-TV & LDA-Cos & Ridge-Cos & QDA-Cos & kNN-Cos \\\\",
        "\\midrule",
    ]
    by_dataset = {r["dataset"]: r for r in rows}
    for dataset in FULL_ORDER:
        row = by_dataset[dataset]
        m = row["metrics"]
        lines.append(
            f"{LABELS[dataset]} & {pct_ci(m['cos'])} & {pct_ci(m['wcos_tv'])} & {pct_ci(m['lda_tv'])} & "
            f"{pp_ci(m['lda_tv_minus_cos_pp'])} & {pp_ci(m['ridge_minus_cos_pp'])} & "
            f"{pp_ci(m['qda_diag_minus_cos_pp'])} & {pp_ci(m['knn_cos_minus_cos_pp'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    write(TABLE_DIR / "table_bootstrap_ci.tex", "\n".join(lines))


def table_backbone_ranking(backbones: pd.DataFrame) -> None:
    backbone_order = ["DINOv2-B", "CLIP-B/32", "MAE ViT-B", "SigLIP-B/16", "ConvNeXt-B", "Sup. ViT-B/16"]
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Backbone ranking instability under raw cosine versus a fixed shared-whitening diagnostic. Regret is the fixed-whitening accuracy lost by selecting the cosine-best backbone instead of the diagnostic-best backbone; the diagnostic is not used as a universal ground-truth score.}",
        "\\label{tab:backbone_ranking}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllrllr}",
        "\\toprule",
        "Dataset & Cosine winner & Diagnostic winner & $\\tau$ & Flip & Cos-selected diagnostic & Regret \\\\",
        "\\midrule",
    ]
    total_conc = total_disc = flips = 0
    taus = []
    regrets = []
    full_rows = []
    for dataset in FULL_ORDER:
        sub = backbones[backbones["dataset"] == dataset].copy()
        sub["backbone"] = pd.Categorical(sub["backbone"], categories=backbone_order, ordered=True)
        sub = sub.sort_values("backbone")
        tau, conc, disc = kendall_tau(sub["cos"].to_numpy(), sub["mahal"].to_numpy())
        total_conc += conc
        total_disc += disc
        taus.append(tau)
        cos_row = sub.loc[sub["cos"].idxmax()]
        mah_row = sub.loc[sub["mahal"].idxmax()]
        flip = cos_row["backbone"] != mah_row["backbone"]
        flips += int(flip)
        regret = float(mah_row["mahal"] - cos_row["mahal"])
        regrets.append(regret)
        full_rows.append(
            {
                "dataset": dataset,
                "label": LABELS[dataset],
                "cos_winner": str(cos_row["backbone"]),
                "mahal_winner": str(mah_row["backbone"]),
                "tau": tau,
                "flip": bool(flip),
                "cos_selected_mahal": float(cos_row["mahal"]),
                "regret": regret,
            }
        )
        lines.append(
            f"{LABELS[dataset]} & {cos_row['backbone']} & {mah_row['backbone']} & "
            f"{tau:.2f} & {'yes' if flip else 'no'} & {cos_row['mahal']:.2f} & {regret:.2f} \\\\"
        )
    total_pairs = total_conc + total_disc
    lines += [
        "\\midrule",
        f"Summary & \\multicolumn{{2}}{{l}}{{top-1 flips {flips}/9}} & {np.mean(taus):.2f} & "
        f"pair flips {total_disc}/{total_pairs} & -- & {np.mean(regrets):.2f} \\\\",
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table*}",
        "",
    ]
    write(TABLE_DIR / "table_backbone_ranking.tex", "\n".join(lines))
    write(ASSET_ROOT / "ranking_instability.json", json.dumps({
        "rows": full_rows,
        "top1_flips": flips,
        "n_datasets": len(FULL_ORDER),
        "mean_kendall_tau": float(np.mean(taus)),
        "median_kendall_tau": float(np.median(taus)),
        "mean_cosine_selection_regret_pp": float(np.mean(regrets)),
        "max_cosine_selection_regret_pp": float(np.max(regrets)),
        "pairwise_discordant": int(total_disc),
        "pairwise_total": int(total_pairs),
    }, indent=2) + "\n")


def table_reference_probe_ranking(stepb: pd.DataFrame, method_df: pd.DataFrame) -> None:
    full = stepb[stepb["shot"] == "full"].set_index("dataset").loc[FULL_ORDER].reset_index()
    refs = method_df[method_df["shot"] == "full"].set_index("dataset")
    rows = []
    for _, row in full.iterrows():
        ref = refs.loc[row["dataset"]]
        rows.append(
            {
                "dataset": row["dataset"],
                "cos": row["cos"],
                "wcos": row["wcos"],
                "lda": row["lda_trainval"],
                "ridge": row["ridge"],
                "logreg": ref["logreg"],
            }
        )
    df = pd.DataFrame(rows)

    def rankdata(values):
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return ranks

    def pair_inversions(target):
        inv = 0
        total = 0
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                s = np.sign(df["cos"].iloc[i] - df["cos"].iloc[j]) * np.sign(df[target].iloc[i] - df[target].iloc[j])
                if s < 0:
                    inv += 1
                if s != 0:
                    total += 1
        return inv, total

    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\small",
        "\\caption{Reference-probe ranking check on full-shot DINOv2-B. Rankings are over the nine datasets, comparing raw-cosine ordering with each stronger evaluator.}",
        "\\label{tab:reference_probe_ranking}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Evaluator & Spearman with Cos & Pair inversions & Mean gain & Max gain \\\\",
        "\\midrule",
    ]
    for key, label in [("wcos", "WCos-TV"), ("lda", "LDA-TV"), ("ridge", "Ridge"), ("logreg", "LogReg")]:
        rho = float(np.corrcoef(rankdata(df["cos"].to_numpy()), rankdata(df[key].to_numpy()))[0, 1])
        inv, total = pair_inversions(key)
        gain = df[key] - df["cos"]
        lines.append(f"{label} & {rho:.3f} & {inv}/{total} & {gain.mean():.2f} & {gain.max():.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write(TABLE_DIR / "table_reference_probe_ranking.tex", "\n".join(lines))


def table_backbone(backbones: pd.DataFrame) -> None:
    backbone_order = ["DINOv2-B", "CLIP-B/32", "MAE ViT-B", "SigLIP-B/16", "ConvNeXt-B", "Sup. ViT-B/16"]
    mat = backbones.pivot_table(index="dataset", columns="backbone", values="gap", aggfunc="first")
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Backbone dependence of full-shot SDG under a fixed shared-whitening diagnostic. Gaps are diagnostic accuracy minus raw cosine in percentage points; the table is used to compare representation families rather than to tune the best LDA hyperparameter per backbone.}",
        "\\label{tab:backbone}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Dataset & DINOv2-B & CLIP-B/32 & MAE ViT-B & SigLIP-B/16 & ConvNeXt-B & Sup. ViT-B/16 \\\\",
        "\\midrule",
    ]
    for dataset in FULL_ORDER:
        vals = []
        for backbone in backbone_order:
            if dataset in mat.index and backbone in mat.columns and pd.notna(mat.loc[dataset, backbone]):
                vals.append(fmt(float(mat.loc[dataset, backbone])))
            else:
                vals.append("--")
        lines.append(f"{LABELS[dataset]} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    write(TABLE_DIR / "table_backbone.tex", "\n".join(lines))


def table_appendix_backbone_accuracy(backbones: pd.DataFrame) -> None:
    backbone_order = ["DINOv2-B", "CLIP-B/32", "MAE ViT-B", "SigLIP-B/16", "ConvNeXt-B", "Sup. ViT-B/16"]
    lines = [
        "\\begin{table*}[!htbp]",
        "\\centering",
        "\\caption{Full backbone accuracy matrix. Each cell is raw cosine / fixed shared-whitening diagnostic accuracy. This table supports the ranking-instability analysis in Table~\\ref{tab:backbone_ranking}.}",
        "\\label{tab:app_backbone_accuracy}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Dataset & DINOv2-B & CLIP-B/32 & MAE ViT-B & SigLIP-B/16 & ConvNeXt-B & Sup. ViT-B/16 \\\\",
        "\\midrule",
    ]
    for dataset in FULL_ORDER:
        sub = backbones[backbones["dataset"] == dataset].set_index("backbone")
        vals = []
        for backbone in backbone_order:
            row = sub.loc[backbone]
            vals.append(f"{row['cos']:.1f}/{row['mahal']:.1f}")
        lines.append(f"{LABELS[dataset]} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    write(TABLE_DIR / "table_appendix_backbone_accuracy.tex", "\n".join(lines))


def table_replay() -> None:
    payload = read_json(RESULTS_ROOT / "p2_synthetic_case/cub200/summary.json")
    matrix = payload["feature_classifier_matrix"]
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\small",
        "\\caption{CUB-200 synthetic replay case study. Prototype cosine remains high, but synthetic replay collapses under whitening-aware evaluation.}",
        "\\label{tab:replay}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Feature source & Cos & Mahalanobis & WCos & FeCAM \\\\",
        "\\midrule",
    ]
    for key, label in [("real", "Real"), ("gaussian", "Gaussian"), ("synthetic", "DiT synthetic"), ("wiener", "Wiener corrected")]:
        row = matrix[key]
        lines.append(
            f"{label} & {100*row['cosine']:.2f} & {100*row['mahalanobis']:.2f} & "
            f"{100*row['whitened_cosine']:.2f} & {100*row['fecam']:.2f} \\\\"
        )
    lines += [
        "\\midrule",
        f"Prototype cosine mean & \\multicolumn{{4}}{{r}}{{{payload['prototype_cosine_mean']:.4f}}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    write(TABLE_DIR / "table_replay.tex", "\n".join(lines))


def table_supplement_highshot(supp: pd.DataFrame) -> None:
    datasets = ["aircraft", "cub200", "cars", "food101", "cifar100", "imagenet100", "tiny_imagenet"]
    shots = [30, 100, 999]
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{High-shot robustness with dense regularization sweeps. Each cell reports diagnostic SDG in percentage points using the best-over-test regularized Mahalanobis upper bound; parenthesized values report tail128 Fisher enrichment. The shot count is capped by the available training examples per class, so 100-shot equals full-shot for several fine-grained datasets.}",
        "\\label{tab:supp_highshot}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Dataset & 30-shot SDG & 100-shot SDG & Full SDG & Full Cos & Full Mahalanobis & Full tail enrich \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        sub = supp[supp["dataset"] == dataset].set_index("shot_num")
        vals = []
        for shot in shots:
            row = sub.loc[shot]
            vals.append(f"{row['gap']:.2f} ({row['tail128_enrich']:.2f})")
        full = sub.loc[999]
        lines.append(
            f"{LABELS[dataset]} & {vals[0]} & {vals[1]} & {vals[2]} & "
            f"{full['cos']:.2f} & {full['mahal']:.2f} & {full['tail128_enrich']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    write(TABLE_DIR / "table_supplement_highshot.tex", "\n".join(lines))


def table_supplement_diagnostics() -> None:
    corr = read_json(RESULTS_ROOT / "supplement_stats_v1/diagnostic_correlations.json")
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Supplement diagnostic correlations across 36 dataset-shot groups from the high-shot sweep. Correlations are Pearson correlations with oracle SDG and should be read as post-hoc diagnostics until the non-oracle predictive protocol is run.}",
        "\\label{tab:supp_diagnostics}",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Diagnostic & Corr. with SDG \\\\",
        "\\midrule",
        f"Tail128 Fisher mass & {corr['corr_tail128_fisher_vs_gap']:.3f} \\\\",
        f"Tail128 Fisher enrichment & {corr['corr_tail128_enrichment_vs_gap']:.3f} \\\\",
        f"Effective rank & {corr['corr_effective_rank_vs_gap']:.3f} \\\\",
        f"$\\log_{10}$ covariance condition & {corr['corr_log_condition_vs_gap']:.3f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    write(TABLE_DIR / "table_supplement_diagnostics.tex", "\n".join(lines))


def table_appendix_fewshot_gap_matrix(stepb: pd.DataFrame) -> None:
    shots = [1, 2, 5, 10, 20, 999]
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\small",
        "\\caption{Complete DINOv2-B few-shot train-only protocol SDG matrix. Cells report train-only shared-covariance LDA minus raw cosine accuracy in percentage points; non-full shots average 5 support-sampling seeds, while full-shot uses the official full training split once.}",
        "\\label{tab:fewshot_gap_matrix}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Dataset & 1-shot & 2-shot & 5-shot & 10-shot & 20-shot & Full \\\\",
        "\\midrule",
    ]
    for dataset in FULL_ORDER:
        vals = []
        for shot in shots:
            g = stepb[(stepb["dataset"] == dataset) & (stepb["shot_num"] == shot)]
            vals.append(fmt_pm(*mean_std(g, "lda_gap")))
        lines.append(f"{LABELS[dataset]} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write(TABLE_DIR / "table_appendix_fewshot_gap_matrix.tex", "\n".join(lines))


def table_appendix_estimator_fullshot(p3: pd.DataFrame) -> None:
    full = p3[p3["shot"] == "full"].set_index("dataset").loc[FULL_ORDER].reset_index()
    lines = [
        "\\begin{table*}[!htbp]",
        "\\centering",
        "\\caption{Full-shot estimator taxonomy on DINOv2-B. Shared whitening methods differ sharply from FeCAM-style per-class/diagonal covariance variants on high-sensitivity datasets.}",
        "\\label{tab:app_estimator_fullshot}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrrrrr}",
        "\\toprule",
        "Dataset & Cos & Shared LDA & SDG & Shared diag & Shared shrink & Lowrank+diag & WCos & FeCAM diag & FeCAM shrink & Eff. rank \\\\",
        "\\midrule",
    ]
    for _, row in full.iterrows():
        gap = row["mahal"] - row["cos"]
        lines.append(
            f"{row['label']} & {fmt(row['cos'])} & {fmt(row['mahal'])} & {fmt(gap)} & "
            f"{fmt(row['shared_diag'])} & {fmt(row['shared_shrink'])} & {fmt(row['lowrank_diag'])} & "
            f"{fmt(row['wcos'])} & {fmt(row['fecam_diag'])} & {fmt(row['fecam_shrink'])} & "
            f"{row['effective_rank']:.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    write(TABLE_DIR / "table_appendix_estimator_fullshot.tex", "\n".join(lines))


def table_appendix_p0_spectral() -> None:
    df = p0_full_dataframe().set_index("dataset").loc[["aircraft", "cub200", "cars", "cifar100", "tiny_imagenet"]].reset_index()
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Full-shot spectral mechanism diagnostics. Cos and Mahalanobis are accuracies, and SDG is their difference in percentage points. Top10 var is the variance mass in the top 10 PCs; Tail128 Fisher is the Fisher-ratio mass outside the top 128 high-variance PCs; Tail128 enrich divides this tail Fisher mass by the corresponding tail variance mass. Spearman is the rank correlation between log eigenvalue and log Fisher ratio across PCs.}",
        "\\label{tab:p0_spectral}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Dataset & Cos & Mahalanobis & SDG & Top10 var & Tail128 Fisher & Tail128 enrich & Spearman \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['label']} & {fmt(row['cos'])} & {fmt(row['mahal'])} & {fmt(row['gap'])} & "
            f"{fmt(row['top10_var'])} & {fmt(row['tail128_fisher'])} & {row['tail128_enrich']:.2f} & "
            f"{row['spearman']:.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    write(TABLE_DIR / "table_appendix_p0_spectral.tex", "\n".join(lines))


def table_appendix_highshot_gap_matrix(supp: pd.DataFrame) -> None:
    shots = [30, 50, 100, 999]
    lines = [
        "\\begin{table*}[!htbp]",
        "\\centering",
        "\\caption{Complete high-shot supplement matrix. Cells report oracle SDG in percentage points from the best-over-test regularized Mahalanobis diagnostic; full denotes all available training examples.}",
        "\\label{tab:app_highshot_gap_matrix}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Dataset & 30-shot & 50-shot & 100-shot & Full & Tail enrich(full) & Cond.(full) & Eff. rank(full) & WCos(full) \\\\",
        "\\midrule",
    ]
    for dataset in FULL_ORDER:
        sub = supp[supp["dataset"] == dataset].set_index("shot_num")
        vals = [f"{sub.loc[shot]['gap']:.2f}" for shot in shots]
        full = sub.loc[999]
        lines.append(
            f"{LABELS[dataset]} & " + " & ".join(vals) + " & "
            f"{full['tail128_enrich']:.2f} & {full['condition']:.2e} & {full['effective_rank']:.1f} & {full['wcos']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    write(TABLE_DIR / "table_appendix_highshot_gap_matrix.tex", "\n".join(lines))


def table_synthetic_gaussian() -> None:
    rows = read_json(RESULTS_ROOT / "synthetic_gaussian_v4/summary.json")["rows"]
    bands = ["head", "mid", "tail", "mixed"]
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\small",
        "\\caption{Controlled Gaussian mechanism. The covariance spectrum is fixed; only the class-mean signal location changes. Fisher rank is the true Fisher-mass-weighted PC rank, where larger means more decision signal in lower-variance directions.}",
        "\\label{tab:synthetic_gaussian}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Signal PCs & Cos & WCos & LDA & LDA-Cos & Fisher rank \\\\",
        "\\midrule",
    ]
    for band in bands:
        sub = [r for r in rows if r["band"] == band]
        def ms(key, scale=100.0):
            vals = np.array([r[key] for r in sub], dtype=np.float64) * scale
            return f"${vals.mean():.2f}\\pm{vals.std(ddof=1):.2f}$"
        lines.append(
            f"{band} & {ms('cos')} & {ms('wcos')} & {ms('lda')} & {ms('sdg_lda_pp', scale=1.0)} & "
            f"{np.mean([r['true_fisher_rank_frac'] for r in sub]):.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write(TABLE_DIR / "table_synthetic_gaussian.tex", "\n".join(lines))


def fig_fullshot_sdg(stepb: pd.DataFrame) -> None:
    full = stepb[stepb["shot"] == "full"].set_index("dataset").loc[list(reversed(FULL_ORDER))].reset_index()
    colors = ["#b44d4d" if g < 0 else "#4477aa" for g in full["lda_gap"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.barh(full["label"], full["lda_gap"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Train-only LDA - Cosine accuracy (pp)")
    ax.set_title("Full-shot Train-Only Spectral Decision Gap (DINOv2-B)")
    ax.set_xlim(min(-3.5, full["lda_gap"].min() - 2.0), full["lda_gap"].max() + 3.5)
    for y, gap in enumerate(full["lda_gap"]):
        x = gap + 0.7 if gap >= 0 else 0.35
        ha = "left"
        ax.text(x, y, f"{gap:.1f}", va="center", ha=ha, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_fullshot_sdg.pdf")
    fig.savefig(FIG_DIR / "fig1_fullshot_sdg.png", dpi=200)
    plt.close(fig)


def fig_fewshot_curves(stepb: pd.DataFrame) -> None:
    shot_df = stepb[stepb["shot"] != "full"]
    groups = shot_df.groupby(["dataset", "shot_num"]).agg(gap=("lda_gap", "mean"), gap_std=("lda_gap", "std")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=False)
    panels = [
        (axes[0], ["cub200", "aircraft", "cars"], "High whitening-sensitivity datasets"),
        (axes[1], ["cifar100", "imagenet100", "tiny_imagenet"], "Boundary and negative datasets"),
    ]
    for ax, datasets, title in panels:
        for dataset in datasets:
            sub = groups[groups["dataset"] == dataset].sort_values("shot_num")
            ax.errorbar(sub["shot_num"], sub["gap"], yerr=sub["gap_std"], marker="o", linewidth=1.8, capsize=2, label=LABELS[dataset])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("shots per class")
        ax.set_ylabel("SDG (pp)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_fewshot_curves.pdf")
    fig.savefig(FIG_DIR / "fig2_fewshot_curves.png", dpi=200)
    plt.close(fig)


def fig_backbone(backbones: pd.DataFrame) -> None:
    backbone_order = ["DINOv2-B", "CLIP-B/32", "MAE ViT-B", "SigLIP-B/16", "ConvNeXt-B", "Sup. ViT-B/16"]
    mat = backbones.pivot_table(index="dataset", columns="backbone", values="gap", aggfunc="first")
    data = np.full((len(FULL_ORDER), len(backbone_order)), np.nan)
    for i, dataset in enumerate(FULL_ORDER):
        for j, backbone in enumerate(backbone_order):
            if dataset in mat.index and backbone in mat.columns and pd.notna(mat.loc[dataset, backbone]):
                data[i, j] = float(mat.loc[dataset, backbone])

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    im = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(backbone_order)))
    ax.set_xticklabels(backbone_order, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(FULL_ORDER)))
    ax.set_yticklabels([LABELS[d] for d in FULL_ORDER])
    ax.set_title("Backbone x Dataset SDG Heatmap")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=6.2)
            else:
                ax.text(j, i, "--", ha="center", va="center", fontsize=6.2, color="#666666")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Fixed shared-whitening diagnostic - Cosine (pp)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_backbone_gap.pdf")
    fig.savefig(FIG_DIR / "fig3_backbone_gap.png", dpi=200)
    plt.close(fig)


def fig_fisher_spectrum() -> None:
    datasets = [("cub200", "CUB-200", "+31.6pp LDA-TV gap"), ("tiny_imagenet", "Tiny-ImageNet", "-1.6pp LDA-TV gap")]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharey=True)
    for ax, (dataset, label, note) in zip(axes, datasets):
        path = RESULTS_ROOT / f"p0_mechanism_v2/{dataset}/shot_full/seed_0/spectrum_fisher.npz"
        payload = np.load(path)
        eigvals = payload["eigvals"].astype(np.float64)
        fisher_mass = payload["fisher_mass"].astype(np.float64)
        var_mass = eigvals / eigvals.sum()
        max_k = len(eigvals) // 2
        cutoffs = np.arange(1, max_k + 1)
        fisher_tail = np.array([fisher_mass[k:].sum() for k in cutoffs])
        var_tail = np.array([var_mass[k:].sum() for k in cutoffs])

        ax.plot(cutoffs, 100 * fisher_tail, color="#4477aa", lw=1.6, label="$M_K$: Fisher mass after top $K$")
        ax.plot(cutoffs, 100 * var_tail, color="#999999", lw=1.2, label="variance mass after top $K$")
        ax.axvline(128, color="#9a5a16", lw=1.0, ls="--")
        ax.text(128 + 5, 4, "$K=128$", fontsize=7, color="#9a5a16", va="bottom")
        m128 = fisher_tail[127]
        v128 = var_tail[127]
        ax.text(
            0.04,
            0.91,
            f"$M_{{128}}$={100*m128:.1f}%\n$E_{{128}}$={m128/(v128+1e-12):.2f}",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            bbox=dict(facecolor="white", edgecolor="#dddddd", boxstyle="round,pad=0.25"),
        )
        ax.set_title(f"{label}: {note}", fontsize=10)
        ax.set_xlabel("cutoff $K$ (drop top-$K$ high-variance PCs)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("remaining tail mass (%)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4b_fisher_spectrum.pdf")
    fig.savefig(FIG_DIR / "fig4b_fisher_spectrum.png", dpi=200)
    plt.close(fig)


def fig_spectral_scatter(stepb: pd.DataFrame) -> None:
    df = spectral_scatter_dataframe(stepb)
    shot_order = [1, 2, 5, 10, 20, 999]
    shot_labels = {1: "1", 2: "2", 5: "5", 10: "10", 20: "20", 999: "full"}
    colors = {1: "#8c8c8c", 2: "#4477aa", 5: "#66aa66", 10: "#aa7744", 20: "#bb5566", 999: "#7b3294"}

    def corr_title(x: pd.Series, y: pd.Series, title: str) -> str:
        corr = float(np.corrcoef(x, y)[0, 1])
        return f"{title} (r={corr:.2f})"

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for shot in shot_order:
        sub = df[df["shot_num"] == shot]
        for ax, xkey in [(axes[0], "tail128_fisher"), (axes[1], "top10_var")]:
            ax.scatter(
                sub[xkey],
                sub["best_whitening_gap"],
                s=32 if shot != 999 else 42,
                color=colors[shot],
                alpha=0.78,
                edgecolor="white",
                linewidth=0.35,
                label=shot_labels[shot] if ax is axes[0] else None,
            )
    axes[0].set_xlabel("tail128 Fisher mass (%)")
    axes[0].set_ylabel("Best whitening - Cosine (pp)")
    axes[0].set_title(corr_title(df["tail128_fisher"], df["best_whitening_gap"], "Tail decision signal"))
    axes[1].set_xlabel("top-10 variance (%)")
    axes[1].set_ylabel("Best whitening - Cosine (pp)")
    axes[1].set_title(corr_title(df["top10_var"], df["best_whitening_gap"], "Variance baseline"))
    label_full = {"aircraft", "cub200", "cars", "tiny_imagenet", "imagenet100"}
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(alpha=0.25)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax + 0.08 * (xmax - xmin))
        for _, row in df[(df["shot_num"] == 999) & (df["dataset"].isin(label_full))].iterrows():
            x = row["tail128_fisher"] if ax is axes[0] else row["top10_var"]
            ax.text(x + 0.5, row["best_whitening_gap"], row["label"].replace("-ImageNet", ""), fontsize=6.5)
    axes[0].legend(title="support shots", frameon=False, fontsize=7, title_fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_spectral_scatter.pdf")
    fig.savefig(FIG_DIR / "fig4_spectral_scatter.png", dpi=200)
    plt.close(fig)


def fig_replay_case() -> None:
    payload = read_json(RESULTS_ROOT / "p2_synthetic_case/cub200/summary.json")
    matrix = payload["feature_classifier_matrix"]
    partd = payload["causal"]
    sources = ["real", "gaussian", "synthetic", "wiener"]
    labels = ["Real", "Gaussian", "DiT", "Wiener"]
    cos = [100 * matrix[s]["cosine"] for s in sources]
    mahal = [100 * matrix[s]["mahalanobis"] for s in sources]
    x = np.arange(len(sources))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    axes[0].bar(x - 0.18, cos, width=0.36, label="Cosine", color="#999999")
    axes[0].bar(x + 0.18, mahal, width=0.36, label="Mahalanobis", color="#4477aa")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].set_title("Feature source x classifier")
    axes[0].legend(frameon=False, fontsize=8)
    interventions = [
        ("Real", matrix["real"]),
        ("Tail noise", partd["tail128_alpha5"]),
        ("Corrected DiT", partd["aggressive_correction"]),
    ]
    x2 = np.arange(len(interventions))
    axes[1].bar(x2 - 0.18, [100 * r["cosine"] for _, r in interventions], width=0.36, label="Cosine", color="#999999")
    axes[1].bar(x2 + 0.18, [100 * r["mahalanobis"] for _, r in interventions], width=0.36, label="Mahalanobis", color="#4477aa")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([name for name, _ in interventions], rotation=15, ha="right")
    axes[1].set_title("Causal spectral interventions")
    for ax in axes:
        ax.set_ylim(0, 90)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_replay_case.pdf")
    fig.savefig(FIG_DIR / "fig5_replay_case.png", dpi=200)
    plt.close(fig)


def fig_supplement_diagnostics(supp: pd.DataFrame) -> None:
    shot_order = [30, 50, 100, 999]
    shot_labels = {30: "30", 50: "50", 100: "100", 999: "full"}
    colors = {30: "#4477aa", 50: "#66aa66", 100: "#aa7744", 999: "#bb5566"}
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.3))

    for shot in shot_order:
        sub = supp[supp["shot_num"] == shot]
        axes[0].scatter(sub["tail128_enrich"], sub["gap"], s=38, color=colors[shot], label=shot_labels[shot], alpha=0.85)
        axes[1].scatter(np.log10(sub["condition"] + 1.0), sub["gap"], s=38, color=colors[shot], label=shot_labels[shot], alpha=0.85)

    label_datasets = {"aircraft", "cub200", "cars", "food101", "tiny_imagenet"}
    full = supp[(supp["shot_num"] == 999) & (supp["dataset"].isin(label_datasets))]
    for _, row in full.iterrows():
        label = row["label"].replace("-ImageNet", "")
        axes[0].text(row["tail128_enrich"] + 0.04, row["gap"], label, fontsize=6.5)
        axes[1].text(np.log10(row["condition"] + 1.0) + 0.025, row["gap"], label, fontsize=6.5)

    axes[0].set_xlabel("tail128 Fisher enrichment")
    axes[1].set_xlabel("log10 covariance condition")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("SDG (pp)")
        ax.grid(alpha=0.25)
    axes[0].set_title("Tail Fisher enrichment", fontsize=11)
    axes[1].set_title("Shared covariance conditioning", fontsize=11)
    axes[1].legend(title="shot", frameon=False, fontsize=7, title_fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_supplement_diagnostics.pdf")
    fig.savefig(FIG_DIR / "fig6_supplement_diagnostics.png", dpi=200)
    plt.close(fig)


def fig_synthetic_gaussian() -> None:
    src_pdf = RESULTS_ROOT / "synthetic_gaussian_v4/synthetic_gaussian_mechanism.pdf"
    src_png = RESULTS_ROOT / "synthetic_gaussian_v4/synthetic_gaussian_mechanism.png"
    if src_pdf.exists():
        shutil.copy2(src_pdf, FIG_DIR / "fig7_synthetic_gaussian.pdf")
    if src_png.exists():
        shutil.copy2(src_png, FIG_DIR / "fig7_synthetic_gaussian.png")


def manifest() -> None:
    lines = [
        "# Rendered SDG-ED Manuscript Assets",
        "",
        "Figures:",
    ]
    for p in sorted(FIG_DIR.glob("*.pdf")):
        lines.append(f"- `figures/{p.name}`")
    lines.append("")
    lines.append("Tables:")
    for p in sorted(TABLE_DIR.glob("*.tex")):
        lines.append(f"- `tables/{p.name}`")
    lines.append("")
    lines.append("Source artifacts are read from `artifacts/results` by default, or from `SDG_RESULTS_ROOT` when set.")
    write(ASSET_ROOT / "asset_manifest.md", "\n".join(lines) + "\n")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = method_dataframe()
    stepb = stepb_cached_dataframe()
    stepb_records = stepb_cached_records_dataframe()
    supp = supplement_dataframe()
    p3 = p3_dataframe()
    backbones = backbone_dataframe()
    table_fullshot(stepb, df)
    table_regime_boundary_examples(stepb_records)
    table_stepb_decomposition(stepb)
    table_predictive_protocol()
    table_predictive_routing_baselines()
    table_router_ablation()
    table_heldout_router()
    table_hyperparameter_grids()
    table_bootstrap_ci()
    table_backbone(backbones)
    table_backbone_ranking(backbones)
    table_reference_probe_ranking(stepb, df)
    table_appendix_backbone_accuracy(backbones)
    table_replay()
    table_supplement_highshot(supp)
    table_supplement_diagnostics()
    table_appendix_fewshot_gap_matrix(stepb)
    table_appendix_estimator_fullshot(p3)
    table_appendix_p0_spectral()
    table_appendix_highshot_gap_matrix(supp)
    table_synthetic_gaussian()
    fig_fullshot_sdg(stepb)
    fig0_schematic()
    fig_fewshot_curves(stepb)
    fig_backbone(backbones)
    fig_fisher_spectrum()
    fig_spectral_scatter(stepb)
    fig_replay_case()
    fig_supplement_diagnostics(supp)
    fig_synthetic_gaussian()
    manifest()
    print(f"Rendered assets to {ASSET_ROOT}")


if __name__ == "__main__":
    main()
