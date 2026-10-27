#!/usr/bin/env python3
"""Eight-dataset out-of-domain stress test for the frozen SDG router.

The linear router is fit only on the original nine paper datasets.  This
script reuses the four original held-out datasets and adds four completed
scope-expansion datasets without fitting, standardizing, or selecting on any
of the eight evaluation datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sdg_audit_core import (
    environment_manifest,
    load_cache,
    write_json_new,
    write_text_new,
)
from sdg_paths import FEATURE_ROOT, OUTPUT_ROOT, RESULTS_ROOT, core_feature_dir
from sdg_stepb_predictive_protocol import FEATURES, fit_predict_linear, metric_bundle
from spectral_decision_gap_ed import sample_shot, sdgi_metrics


DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT / "router_ood8"
SCOPE_DATASETS = ["mnist", "fashion_mnist", "svhn", "gtsrb"]


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def shot_numeric(shot: str) -> float:
    return 999.0 if str(shot) == "full" else float(shot)


def covariance_diagnostics(features: np.ndarray) -> dict[str, float]:
    centered = features - features.mean(0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    threshold = max(float(eigenvalues.max()) * 1e-12, 1e-18)
    positive = eigenvalues[eigenvalues > threshold]
    if len(positive) == 0:
        condition = 0.0
        effective_rank = 0.0
    else:
        condition = float(positive.max() / positive.min())
        mass = positive / positive.sum()
        effective_rank = float(np.exp(-(mass * np.log(mass + 1e-12)).sum()))
    return {
        "log10_condition": float(np.log10(condition + 1.0)),
        "effective_rank": effective_rank,
    }


def diagnostics_for_training_group(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    shot: str,
    seeds: list[int],
) -> dict[str, float]:
    """Compute router inputs from support features and labels only."""

    rows = []
    for seed in seeds:
        features, labels = sample_shot(train_features, train_labels, shot, seed)
        row = sdgi_metrics(features, labels)
        row.update(covariance_diagnostics(features))
        row["log10_n_train"] = float(np.log10(max(1, len(labels))))
        row["shot_numeric"] = shot_numeric(shot)
        rows.append(row)
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def scope_rows(
    aggregate_groups: list[dict],
    feature_root: Path,
    datasets: list[str],
    seeds: list[int],
) -> list[dict]:
    selected = [
        row for row in aggregate_groups if row["dataset"] in set(datasets)
    ]
    found = sorted(set(row["dataset"] for row in selected))
    if found != sorted(datasets):
        raise ValueError(f"scope dataset mismatch: requested={sorted(datasets)} found={found}")

    rows = []
    for dataset in datasets:
        train_path = core_feature_dir(dataset, feature_root) / "train.pt"
        train_features, train_labels = load_cache(train_path)
        dataset_groups = sorted(
            [row for row in selected if row["dataset"] == dataset],
            key=lambda row: shot_numeric(str(row["shot"])),
        )
        for source in dataset_groups:
            shot = str(source["shot"])
            group_seeds = [0] if shot == "full" else seeds
            row = dict(source)
            row.update(
                diagnostics_for_training_group(
                    train_features,
                    train_labels,
                    shot,
                    group_seeds,
                )
            )
            row["target_best_whitening_minus_cos_pp"] = float(
                row["best_whitening_minus_cos_pp"]
            )
            row["target_ridge_minus_cos_pp"] = float(row["ridge_minus_cos_pp"])
            rows.append(row)
    return rows


def frozen_router_predict(
    training_rows: list[dict],
    evaluation_rows: list[dict],
    target_key: str,
    features: list[str] = FEATURES,
) -> np.ndarray:
    """Fit standardization and ridge weights on training rows only."""

    train_x = np.asarray([[row[key] for key in features] for row in training_rows])
    train_y = np.asarray([row[target_key] for row in training_rows])
    evaluation_x = np.asarray([[row[key] for key in features] for row in evaluation_rows])
    if not np.isfinite(train_x).all() or not np.isfinite(evaluation_x).all():
        raise FloatingPointError("non-finite router features")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        predictions = fit_predict_linear(train_x, train_y, evaluation_x)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("non-finite frozen-router predictions")
    return predictions


def validate_original_heldout_predictions(
    training_rows: list[dict],
    heldout_payload: dict,
    tolerance: float,
) -> list[dict]:
    rows = heldout_payload["heldout_records"]
    expected = np.asarray(heldout_payload["best_whitening_target"]["predictions"])
    actual = frozen_router_predict(
        training_rows,
        rows,
        "target_best_whitening_minus_cos_pp",
    )
    if not np.allclose(actual, expected, rtol=0.0, atol=tolerance):
        maximum = float(np.max(np.abs(actual - expected)))
        raise AssertionError(f"original held-out prediction mismatch: max_abs={maximum}")
    return rows


def point_metrics(rows: list[dict], predictions: np.ndarray) -> dict:
    target = np.asarray(
        [row["target_best_whitening_minus_cos_pp"] for row in rows],
        dtype=np.float64,
    )
    cos = np.asarray([row["cos"] for row in rows], dtype=np.float64)
    wcos = np.asarray([row["wcos_trainval"] for row in rows], dtype=np.float64)
    lda = np.asarray([row["lda_trainval"] for row in rows], dtype=np.float64)
    best = np.asarray([row["best_whitening"] for row in rows], dtype=np.float64)
    oracle = np.maximum(cos, best)
    route = predictions >= 5.0
    routed_oracle_family = np.where(route, best, cos)

    def policy(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_acc_pct": float(100.0 * values.mean()),
            "regret_pp": float(100.0 * (oracle - values).mean()),
        }

    return {
        "n_groups": len(rows),
        "n_datasets": len(set(row["dataset"] for row in rows)),
        "prediction": metric_bundle(predictions, target),
        "target_mean_pp": float(target.mean()),
        "target_material_5pp_fraction": float(np.mean(target >= 5.0)),
        "threshold_pp": 5.0,
        "route_fraction": float(route.mean()),
        "policies": {
            "router_oracle_family": policy(routed_oracle_family),
            "always_cos": policy(cos),
            "always_wcos": policy(wcos),
            "always_lda": policy(lda),
            "oracle_cos_or_whitening": policy(oracle),
        },
    }


def cluster_bootstrap(
    rows: list[dict],
    predictions: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    datasets = sorted(set(row["dataset"] for row in rows))
    indices_by_dataset = {
        dataset: np.asarray(
            [index for index, row in enumerate(rows) if row["dataset"] == dataset]
        )
        for dataset in datasets
    }
    rng = np.random.RandomState(seed)
    values: dict[str, list[float]] = {
        "pearson": [],
        "spearman": [],
        "mae_pp": [],
        "router_regret_pp": [],
        "always_wcos_regret_pp": [],
        "always_lda_regret_pp": [],
    }
    for _ in range(n_boot):
        sampled = rng.choice(datasets, size=len(datasets), replace=True)
        indices = np.concatenate([indices_by_dataset[name] for name in sampled])
        sampled_rows = [rows[int(index)] for index in indices]
        point = point_metrics(sampled_rows, predictions[indices])
        values["pearson"].append(point["prediction"]["pearson"])
        values["spearman"].append(point["prediction"]["spearman"])
        values["mae_pp"].append(point["prediction"]["mae_pp"])
        values["router_regret_pp"].append(
            point["policies"]["router_oracle_family"]["regret_pp"]
        )
        values["always_wcos_regret_pp"].append(
            point["policies"]["always_wcos"]["regret_pp"]
        )
        values["always_lda_regret_pp"].append(
            point["policies"]["always_lda"]["regret_pp"]
        )
    return {
        "unit": "dataset cluster",
        "n_clusters": len(datasets),
        "n_boot": n_boot,
        "seed": seed,
        "ci95": {
            key: [
                float(np.nanpercentile(sample, 2.5)),
                float(np.nanpercentile(sample, 97.5)),
            ]
            for key, sample in values.items()
        },
    }


def by_dataset(rows: list[dict], predictions: np.ndarray) -> list[dict]:
    summaries = []
    for dataset in sorted(set(row["dataset"] for row in rows)):
        indices = [
            index for index, row in enumerate(rows) if row["dataset"] == dataset
        ]
        subset = [rows[index] for index in indices]
        summaries.append(
            {
                "dataset": dataset,
                **point_metrics(subset, predictions[np.asarray(indices)]),
            }
        )
    return summaries


def summarize_suite(
    rows: list[dict],
    predictions: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    return {
        "point": point_metrics(rows, predictions),
        "cluster_bootstrap": cluster_bootstrap(rows, predictions, n_boot, seed),
        "by_dataset": by_dataset(rows, predictions),
        "records": [
            {
                **row,
                "router_prediction_pp": float(prediction),
            }
            for row, prediction in zip(rows, predictions)
        ],
    }


def render_report(payload: dict) -> str:
    lines = ["# Frozen-router 8-dataset OOD audit\n"]
    lines.append(
        "The router, feature standardization, and linear weights are fit only on the original "
        "nine paper datasets. No OOD dataset is used for fitting or model selection."
    )
    lines.append(
        "The router-oracle-family policy is retained only to match the frozen target definition; "
        "it is not presented as a deployable evaluator-family selector."
    )
    lines.append("\n| suite | datasets | groups | Pearson | MAE | 5pp route rate | router regret | WCos regret | LDA regret |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in ["original_heldout4", "scope_heldout4", "combined_heldout8"]:
        point = payload[name]["point"]
        policies = point["policies"]
        lines.append(
            f"| {name} | {point['n_datasets']} | {point['n_groups']} | "
            f"{point['prediction']['pearson']:.3f} | {point['prediction']['mae_pp']:.2f} | "
            f"{100.0 * point['route_fraction']:.1f}% | "
            f"{policies['router_oracle_family']['regret_pp']:.2f} | "
            f"{policies['always_wcos']['regret_pp']:.2f} | "
            f"{policies['always_lda']['regret_pp']:.2f} |"
        )
    lines.append("\n## Combined suite by dataset\n")
    lines.append("| dataset | groups | target mean | Pearson | MAE | router regret |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in payload["combined_heldout8"]["by_dataset"]:
        point = row
        lines.append(
            f"| {row['dataset']} | {point['n_groups']} | {point['target_mean_pp']:.2f} | "
            f"{point['prediction']['pearson']:.3f} | {point['prediction']['mae_pp']:.2f} | "
            f"{point['policies']['router_oracle_family']['regret_pp']:.2f} |"
        )
    lines.append("\n## Decision facts\n")
    for key, value in payload["decision_facts"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def run(args) -> None:
    training_rows = read_json(Path(args.main_joined))
    main_datasets = sorted(set(row["dataset"] for row in training_rows))
    heldout_payload = read_json(Path(args.original_heldout_router))
    original_rows = validate_original_heldout_predictions(
        training_rows,
        heldout_payload,
        args.prediction_tolerance,
    )
    scope_groups = read_json(Path(args.scope_aggregate_groups))
    scope = scope_rows(
        scope_groups,
        Path(args.scope_feature_root),
        args.scope_datasets,
        args.seeds,
    )
    evaluation_datasets = sorted(
        set(row["dataset"] for row in original_rows + scope)
    )
    overlap = sorted(set(main_datasets) & set(evaluation_datasets))
    if overlap:
        raise AssertionError(f"training/evaluation dataset overlap: {overlap}")

    original_predictions = frozen_router_predict(
        training_rows,
        original_rows,
        "target_best_whitening_minus_cos_pp",
    )
    scope_predictions = frozen_router_predict(
        training_rows,
        scope,
        "target_best_whitening_minus_cos_pp",
    )
    combined_rows = original_rows + scope
    combined_predictions = np.concatenate([original_predictions, scope_predictions])

    original = summarize_suite(
        original_rows,
        original_predictions,
        args.n_boot,
        args.seed,
    )
    scope_summary = summarize_suite(
        scope,
        scope_predictions,
        args.n_boot,
        args.seed + 1000,
    )
    combined = summarize_suite(
        combined_rows,
        combined_predictions,
        args.n_boot,
        args.seed + 2000,
    )
    combined_point = combined["point"]
    simple_regrets = {
        key: combined_point["policies"][key]["regret_pp"]
        for key in ["always_wcos", "always_lda"]
    }
    best_simple_name = min(simple_regrets, key=simple_regrets.get)
    decision = {
        "main_training_datasets": main_datasets,
        "evaluation_datasets": evaluation_datasets,
        "n_main_training_groups": len(training_rows),
        "n_ood_groups": len(combined_rows),
        "combined_pearson": combined_point["prediction"]["pearson"],
        "combined_pearson_ci95_dataset_cluster": combined["cluster_bootstrap"]["ci95"]["pearson"],
        "combined_mae_pp": combined_point["prediction"]["mae_pp"],
        "combined_router_oracle_family_regret_pp": combined_point["policies"][
            "router_oracle_family"
        ]["regret_pp"],
        "combined_best_fixed_simple_policy": best_simple_name,
        "combined_best_fixed_simple_regret_pp": simple_regrets[best_simple_name],
        "router_incremental_value_claimed": False,
    }
    payload = {
        "experiment": "sdg_router_ood8_eval",
        "post_submission": True,
        "frozen_submission_edited": False,
        "protocol": {
            "fit_scope": "original nine paper datasets only",
            "evaluation_scope": "eight unseen datasets, six shot regimes each",
            "router_features": FEATURES,
            "threshold_pp": 5.0,
            "scope_datasets": args.scope_datasets,
            "nonfull_seeds": args.seeds,
            "bootstrap_unit": "dataset",
            "n_boot": args.n_boot,
            "bootstrap_seed": args.seed,
            "test_features_or_labels_used_for_router_inputs": False,
        },
        "environment": environment_manifest(args.commit),
        "original_heldout4": original,
        "scope_heldout4": scope_summary,
        "combined_heldout8": combined,
        "decision_facts": decision,
    }
    output_root = Path(args.output_root)
    write_json_new(output_root / "router_ood8_eval.json", payload)
    write_text_new(output_root / "router_ood8_eval.md", render_report(payload))
    print(f"[router-ood8] complete output={output_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-joined",
        default=str(RESULTS_ROOT / "stepb_predictive_protocol_reproduced/joined_records.json"),
    )
    parser.add_argument(
        "--original-heldout-router",
        default=str(RESULTS_ROOT / "heldout_router_v1/heldout_router_generalization.json"),
    )
    parser.add_argument(
        "--scope-aggregate-groups",
        default=str(OUTPUT_ROOT / "scope_expansion_stepb_v1/aggregate_groups.json"),
        help="aggregate_groups.json from sdg_stepb_cached_eval.py --mode aggregate on the four scope datasets",
    )
    parser.add_argument(
        "--scope-feature-root",
        default=str(FEATURE_ROOT),
        help="DINOv2-B caches for the scope datasets, laid out as <root>/<dataset>/{train,test}.pt",
    )
    parser.add_argument("--scope-datasets", nargs="+", default=SCOPE_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--prediction-tolerance", type=float, default=1e-9)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--commit", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
