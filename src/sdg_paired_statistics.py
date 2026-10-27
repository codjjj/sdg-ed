#!/usr/bin/env python3
"""Paired statistics and target-definition audit for SDG-ED.

This analysis reads frozen seed-level evaluator outputs.  It adds paired
uncertainty for every dataset-shot cell and compares the original
best-of-(WCos, LDA) target with fixed WCos and fixed LDA targets.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from sdg_paths import OUTPUT_ROOT, RESULTS_ROOT
from sdg_audit_core import environment_manifest, write_json_new, write_text_new
from sdg_stepb_predictive_protocol import FEATURES, metric_bundle, predict_cv


DEFAULT_STEPB_ROOT = RESULTS_ROOT / "stepb_cached_eval_v1"
DEFAULT_MAIN_JOINED = RESULTS_ROOT / "stepb_predictive_protocol_reproduced/joined_records.json"
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT / "paired_statistics_audit"


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def load_seed_records(aggregate_records: str | None, stepb_root: Path) -> list[dict]:
    """Load seed-level evaluator records.

    Uses `aggregate_records.json` when given; otherwise concatenates the
    released per-group files `<stepb_root>/<dataset>/shot_<shot>/result.json`.
    """

    if aggregate_records:
        return read_json(Path(aggregate_records))
    paths = sorted(Path(stepb_root).glob("*/shot_*/result.json"))
    if not paths:
        raise FileNotFoundError(f"no */shot_*/result.json files under {stepb_root}")
    records = []
    for path in paths:
        records.extend(read_json(path)["records"])
    return records


def record_scores(record: dict) -> dict[str, float]:
    metrics = record["metrics"]
    return {
        "cos": float(metrics["cos"]),
        "wcos": float(metrics["wcos_trainval"]["acc"]),
        "lda": float(metrics["lda_trainval"]["acc"]),
    }


def exact_sign_flip_p(values: np.ndarray, alternative: str) -> float:
    """Exact paired randomization p-value under sign symmetry."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a nonempty finite vector")
    observed = float(values.mean())
    magnitudes = np.abs(values)
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(values))))
    null_means = (signs * magnitudes[None, :]).mean(axis=1)
    tolerance = 1e-12
    if alternative == "less":
        return float(np.mean(null_means <= observed + tolerance))
    if alternative == "greater":
        return float(np.mean(null_means >= observed - tolerance))
    if alternative == "two-sided":
        return float(np.mean(np.abs(null_means) >= abs(observed) - tolerance))
    raise ValueError(f"unknown alternative: {alternative}")


def paired_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("paired differences must be a nonempty finite vector")
    n = len(values)
    mean = float(values.mean())
    if n == 1:
        return {
            "n": 1,
            "mean_pp": mean,
            "sd_pp": None,
            "ci95_pp": None,
            "t_statistic": None,
            "p_t_less": None,
            "p_t_two_sided": None,
            "p_signflip_less": None,
            "p_signflip_two_sided": None,
        }

    sd = float(values.std(ddof=1))
    sem = sd / np.sqrt(n)
    if sd == 0.0:
        ci = [mean, mean]
        if mean < 0:
            t_statistic, p_less, p_two = float("-inf"), 0.0, 0.0
        elif mean > 0:
            t_statistic, p_less, p_two = float("inf"), 1.0, 0.0
        else:
            t_statistic, p_less, p_two = 0.0, 1.0, 1.0
    else:
        low, high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
        ci = [float(low), float(high)]
        less = stats.ttest_1samp(values, popmean=0.0, alternative="less")
        two = stats.ttest_1samp(values, popmean=0.0, alternative="two-sided")
        t_statistic = float(less.statistic)
        p_less = float(less.pvalue)
        p_two = float(two.pvalue)
    return {
        "n": n,
        "mean_pp": mean,
        "sd_pp": sd,
        "ci95_pp": ci,
        "t_statistic": t_statistic,
        "p_t_less": p_less,
        "p_t_two_sided": p_two,
        "p_signflip_less": exact_sign_flip_p(values, "less"),
        "p_signflip_two_sided": exact_sign_flip_p(values, "two-sided"),
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    for index in range(len(ranked) - 1, -1, -1):
        running = min(running, ranked[index] * len(ranked) / (index + 1))
        adjusted[index] = running
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out.tolist()


def add_bh_q_values(rows: list[dict], family_key: str, p_key: str, q_key: str) -> None:
    eligible = [
        row for row in rows if row[family_key][p_key] is not None
    ]
    adjusted = benjamini_hochberg([row[family_key][p_key] for row in eligible])
    for row in rows:
        row[family_key][q_key] = None
    for row, value in zip(eligible, adjusted):
        row[family_key][q_key] = float(value)


def build_group_statistics(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["dataset"], str(record["shot"]))].append(record)

    rows = []
    for (dataset, shot), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["seed"]))
        scores = [record_scores(record) for record in group]
        cos = np.asarray([row["cos"] for row in scores])
        wcos = np.asarray([row["wcos"] for row in scores])
        lda = np.asarray([row["lda"] for row in scores])
        best = np.maximum(wcos, lda)
        best_fixed = max(float(wcos.mean()), float(lda.mean()))
        rows.append(
            {
                "dataset": dataset,
                "shot": shot,
                "seeds": [int(record["seed"]) for record in group],
                "cos_mean_pct": float(100.0 * cos.mean()),
                "wcos_mean_pct": float(100.0 * wcos.mean()),
                "lda_mean_pct": float(100.0 * lda.mean()),
                "best_whitening_mean_pct": float(100.0 * best.mean()),
                "wcos_minus_cos": paired_summary(100.0 * (wcos - cos)),
                "lda_minus_cos": paired_summary(100.0 * (lda - cos)),
                "best_whitening_minus_cos": paired_summary(100.0 * (best - cos)),
                "target_decomposition": {
                    "max_target_pp": float(100.0 * (best.mean() - cos.mean())),
                    "fixed_wcos_target_pp": float(100.0 * (wcos.mean() - cos.mean())),
                    "fixed_lda_target_pp": float(100.0 * (lda.mean() - cos.mean())),
                    "best_fixed_family_target_pp": float(100.0 * (best_fixed - cos.mean())),
                    "oracle_family_selection_uplift_pp": float(
                        100.0 * (best.mean() - best_fixed)
                    ),
                    "max_over_wcos_uplift_pp": float(100.0 * (best.mean() - wcos.mean())),
                    "max_over_lda_uplift_pp": float(100.0 * (best.mean() - lda.mean())),
                },
            }
        )

    for family in ["wcos_minus_cos", "lda_minus_cos", "best_whitening_minus_cos"]:
        add_bh_q_values(rows, family, "p_t_less", "q_bh_t_less")
        add_bh_q_values(rows, family, "p_signflip_less", "q_bh_signflip_less")
    return rows


def validate_joined_alignment(group_rows: list[dict], joined_rows: list[dict], tolerance: float) -> None:
    joined = {(row["dataset"], str(row["shot"])): row for row in joined_rows}
    if set(joined) != {(row["dataset"], row["shot"]) for row in group_rows}:
        raise AssertionError("seed-level and joined group keys differ")
    for row in group_rows:
        key = (row["dataset"], row["shot"])
        reference = joined[key]
        checks = {
            "cos": row["cos_mean_pct"] / 100.0,
            "wcos_trainval": row["wcos_mean_pct"] / 100.0,
            "lda_trainval": row["lda_mean_pct"] / 100.0,
            "best_whitening": row["best_whitening_mean_pct"] / 100.0,
        }
        for field, expected in checks.items():
            if abs(float(reference[field]) - expected) > tolerance:
                raise AssertionError(
                    f"aggregate alignment mismatch at {key}/{field}: "
                    f"{reference[field]} vs {expected}"
                )


def router_target_rows(joined_rows: list[dict]) -> list[dict]:
    rows = []
    for source in joined_rows:
        row = dict(source)
        row["target_wcos_minus_cos_pp"] = 100.0 * (
            float(row["wcos_trainval"]) - float(row["cos"])
        )
        row["target_lda_minus_cos_pp"] = 100.0 * (
            float(row["lda_trainval"]) - float(row["cos"])
        )
        row["target_max_minus_cos_pp"] = 100.0 * (
            float(row["best_whitening"]) - float(row["cos"])
        )
        rows.append(row)
    return rows


def target_policy_metrics(rows: list[dict], predictions: np.ndarray, evaluator_key: str) -> dict:
    cos = np.asarray([float(row["cos"]) for row in rows])
    evaluator = np.asarray([float(row[evaluator_key]) for row in rows])
    oracle = np.maximum(cos, evaluator)
    route = predictions >= 5.0
    chosen = np.where(route, evaluator, cos)
    return {
        "threshold_pp": 5.0,
        "route_fraction": float(route.mean()),
        "router_mean_acc_pct": float(100.0 * chosen.mean()),
        "router_regret_pp": float(100.0 * (oracle - chosen).mean()),
        "always_cos_regret_pp": float(100.0 * (oracle - cos).mean()),
        "always_evaluator_regret_pp": float(100.0 * (oracle - evaluator).mean()),
    }


def target_sensitivity(joined_rows: list[dict]) -> list[dict]:
    rows = router_target_rows(joined_rows)
    specifications = [
        ("fixed_wcos", "target_wcos_minus_cos_pp", "wcos_trainval", True),
        ("fixed_lda", "target_lda_minus_cos_pp", "lda_trainval", True),
        ("best_of_wcos_lda", "target_max_minus_cos_pp", "best_whitening", False),
    ]
    results = []
    for name, target_key, evaluator_key, deployable_family in specifications:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            predictions, target = predict_cv(
                rows,
                target_key,
                lambda row: row["dataset"],
                features=FEATURES,
            )
        if not np.isfinite(predictions).all():
            raise FloatingPointError(f"non-finite LODO predictions for {name}")
        results.append(
            {
                "target": name,
                "target_key": target_key,
                "evaluator_key": evaluator_key,
                "family_fixed_before_test": deployable_family,
                "n_groups": len(rows),
                "target_mean_pp": float(target.mean()),
                "target_median_pp": float(np.median(target)),
                "target_positive_fraction": float(np.mean(target > 0.0)),
                "target_material_5pp_fraction": float(np.mean(target >= 5.0)),
                "lodo_metrics": metric_bundle(predictions, target),
                "routing": target_policy_metrics(rows, predictions, evaluator_key),
                "predictions": [
                    {
                        "dataset": row["dataset"],
                        "shot": str(row["shot"]),
                        "target_pp": float(actual),
                        "prediction_pp": float(predicted),
                    }
                    for row, actual, predicted in zip(rows, target, predictions)
                ],
            }
        )
    return results


def decision_facts(group_rows: list[dict], sensitivity: list[dict]) -> dict:
    finite = [row for row in group_rows if row["shot"] != "full"]
    uplifts = np.asarray(
        [row["target_decomposition"]["oracle_family_selection_uplift_pp"] for row in group_rows]
    )
    negative_lda = [row for row in finite if row["lda_minus_cos"]["mean_pp"] < 0.0]
    significant_lda = [
        row
        for row in negative_lda
        if row["lda_minus_cos"]["q_bh_t_less"] is not None
        and row["lda_minus_cos"]["q_bh_t_less"] < 0.05
    ]
    return {
        "n_dataset_shot_groups": len(group_rows),
        "n_nonfull_groups_with_five_seeds": len(finite),
        "n_negative_lda_minus_cos_groups": len(negative_lda),
        "n_negative_lda_groups_bh_q_lt_0_05": len(significant_lda),
        "negative_lda_groups_bh_q_lt_0_05": [
            f"{row['dataset']}/{row['shot']}" for row in significant_lda
        ],
        "oracle_family_selection_uplift_mean_pp": float(uplifts.mean()),
        "oracle_family_selection_uplift_median_pp": float(np.median(uplifts)),
        "oracle_family_selection_uplift_max_pp": float(uplifts.max()),
        "target_summary": {
            row["target"]: {
                "mean_pp": row["target_mean_pp"],
                "pearson": row["lodo_metrics"]["pearson"],
                "mae_pp": row["lodo_metrics"]["mae_pp"],
                "router_regret_pp": row["routing"]["router_regret_pp"],
                "always_evaluator_regret_pp": row["routing"]["always_evaluator_regret_pp"],
            }
            for row in sensitivity
        },
    }


def render_report(payload: dict) -> str:
    lines = ["# Paired statistics and target audit\n"]
    lines.append(
        "All evaluator scores are frozen seed-level outputs. Non-full cells use five paired seeds; "
        "full-shot cells have one run and therefore receive no seed-based interval."
    )
    lines.append("\n## Target-definition sensitivity\n")
    lines.append(
        "| target | family fixed pre-test | mean gap | LODO Pearson | MAE | router regret | always evaluator regret |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["target_sensitivity"]:
        lines.append(
            f"| {row['target']} | {str(row['family_fixed_before_test']).lower()} | "
            f"{row['target_mean_pp']:.2f} | {row['lodo_metrics']['pearson']:.3f} | "
            f"{row['lodo_metrics']['mae_pp']:.2f} | {row['routing']['router_regret_pp']:.2f} | "
            f"{row['routing']['always_evaluator_regret_pp']:.2f} |"
        )
    lines.append("\n## Negative paired LDA gaps\n")
    lines.append("| dataset | shot | mean ± SD | 95% t-CI | p<0 | BH q<0 | exact p<0 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["group_statistics"]:
        summary = row["lda_minus_cos"]
        if summary["mean_pp"] >= 0.0 or summary["n"] < 2:
            continue
        ci = summary["ci95_pp"]
        lines.append(
            f"| {row['dataset']} | {row['shot']} | {summary['mean_pp']:.2f} ± "
            f"{summary['sd_pp']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | "
            f"{summary['p_t_less']:.4g} | {summary['q_bh_t_less']:.4g} | "
            f"{summary['p_signflip_less']:.4g} |"
        )
    lines.append("\n## Decision facts\n")
    for key, value in payload["decision_facts"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def run(args) -> None:
    records = load_seed_records(args.aggregate_records, Path(args.stepb_root))
    joined = read_json(Path(args.main_joined))
    group_rows = build_group_statistics(records)
    validate_joined_alignment(group_rows, joined, args.point_tolerance)
    sensitivity = target_sensitivity(joined)
    payload = {
        "experiment": "sdg_paired_statistics",
        "post_submission": True,
        "frozen_submission_edited": False,
        "protocol": {
            "unit": "paired seed-level accuracy differences in percentage points",
            "nonfull_seeds": 5,
            "interval": "two-sided 95% Student-t confidence interval",
            "tests": [
                "one-sided paired one-sample t-test for mean gap < 0",
                "exact paired sign-flip randomization test",
            ],
            "multiplicity": "Benjamini-Hochberg within each evaluator-gap family",
            "router_fold": "leave-one-dataset-out",
            "router_features": FEATURES,
        },
        "environment": environment_manifest(args.commit),
        "group_statistics": group_rows,
        "target_sensitivity": sensitivity,
        "decision_facts": decision_facts(group_rows, sensitivity),
    }
    output_root = Path(args.output_root)
    write_json_new(output_root / "paired_statistics_audit.json", payload)
    write_text_new(output_root / "paired_statistics_audit.md", render_report(payload))
    print(f"[paired-statistics] complete output={output_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate-records",
        default=None,
        help="Optional aggregate_records.json; defaults to the per-group files under --stepb-root.",
    )
    parser.add_argument("--stepb-root", default=str(DEFAULT_STEPB_ROOT))
    parser.add_argument("--main-joined", default=str(DEFAULT_MAIN_JOINED))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--point-tolerance", type=float, default=5e-7)
    parser.add_argument("--commit", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
