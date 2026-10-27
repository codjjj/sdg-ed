#!/usr/bin/env python3
"""Router uncertainty audit using only frozen SDG prediction artifacts.

This reporting pass adds dataset-cluster uncertainty, a no-1-shot
sensitivity check, and paired incremental-feature comparisons. It does not
refit an encoder or inspect test labels beyond the frozen evaluation targets.

All inputs default to the released files and are MD5-checked against the
frozen router outputs. In the paper's Tables 3-4, the LODO Spearman
interval and the LODO router mean-accuracy interval come from this audit
(`main_54_groups`); the other intervals come from
scripts/recompute/table34_dataset_cluster_ci.py.

    python src/sdg_router_uncertainty_audit.py --output outputs/router_uncertainty_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sdg_paths import RESULTS_ROOT
from sdg_stepb_predictive_protocol import (
    FEATURES,
    metric_bundle as frozen_metric_bundle,
    predict_cv,
    routing_arrays,
)


FEATURE_SETS = {
    "full": FEATURES,
    "covariance_only": [
        "log10_condition",
        "effective_rank",
        "log10_n_train",
        "shot_numeric",
    ],
    "no_tail_fisher": [
        "spearman_logeig_logfisher",
        "log10_condition",
        "effective_rank",
        "log10_n_train",
        "shot_numeric",
    ],
}
TARGET = "target_best_whitening_minus_cos_pp"
THRESHOLDS = [0.0, 2.0, 3.0, 5.0, 10.0]
FROZEN_INPUTS = {
    "joined_records": {
        "released_path": "stepb_predictive_protocol_reproduced/joined_records.json",
        "etag": "87AAE4EB203E5C12374E6AEF2A05450A",
    },
    "predictive_results": {
        "released_path": "stepb_predictive_protocol_v1/predictive_results.json",
        "etag": "9B739BBC578ABAF1F94F52B8B68F0116",
    },
    "feature_ablation": {
        "released_path": "stepb_predictive_protocol_v2/feature_ablation.json",
        "etag": "FD12B6219E79C5A3A795A4010C98C2D0",
    },
    "supplement_stats": {
        "released_path": None,  # supplement aggregate_records.json is not bundled
        "etag": "4D44E94E13CC91D1188D874BF30338FD",
    },
}


def read_json(path: Path):
    return json.loads(path.read_text())


def validate_md5(path: Path, expected: str) -> None:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().upper() != expected.upper():
        raise ValueError(f"input ETag/MD5 mismatch: {path}")


def average_rank(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with exact-tie handling."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    error = np.asarray(pred) - np.asarray(target)
    return {
        "pearson": pearson(pred, target),
        "spearman_tie_aware": pearson(average_rank(pred), average_rank(target)),
        "mae_pp": float(np.mean(np.abs(error))),
        "rmse_pp": float(np.sqrt(np.mean(error**2))),
    }


def predict_cv_checked(rows: list[dict], features: list[str]):
    # Apple Accelerate may emit floating-point warnings for finite matmul
    # results; the explicit finiteness gate preserves the semantic check.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        pred, target = predict_cv(
            rows, TARGET, lambda row: row["dataset"], features=features
        )
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise FloatingPointError("non-finite LODO prediction or target")
    return pred, target


def cluster_indices(labels: np.ndarray, draw: np.ndarray) -> np.ndarray:
    clusters = np.asarray(sorted(set(labels.tolist())), dtype=object)
    return np.concatenate([np.flatnonzero(labels == clusters[index]) for index in draw])


def percentile(values: list[float]) -> list[float]:
    return [
        float(np.nanpercentile(values, 2.5)),
        float(np.nanpercentile(values, 97.5)),
    ]


def cluster_metric_ci(
    pred: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    clusters = sorted(set(labels.tolist()))
    rng = np.random.RandomState(seed)
    samples = {key: [] for key in metrics(pred, target)}
    for _ in range(n_boot):
        draw = rng.randint(0, len(clusters), size=len(clusters))
        index = cluster_indices(labels, draw)
        row = metrics(pred[index], target[index])
        for key, value in row.items():
            samples[key].append(value)
    return {key: percentile(values) for key, values in samples.items()}


def routing_point(rows: list[dict], pred: np.ndarray, threshold: float) -> dict:
    chosen, oracle, route = routing_arrays(rows, pred, threshold)
    return {
        "threshold": threshold,
        "mean_chosen_acc": float(100.0 * chosen.mean()),
        "mean_oracle_acc": float(100.0 * oracle.mean()),
        "routing_regret_pp": float(100.0 * (oracle.mean() - chosen.mean())),
        "route_to_whitening_frac": float(route.mean()),
    }


def cluster_routing_ci(
    rows: list[dict],
    pred: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
) -> dict:
    chosen, oracle, route = routing_arrays(rows, pred, threshold)
    clusters = sorted(set(labels.tolist()))
    rng = np.random.RandomState(seed)
    samples = {
        "mean_chosen_acc": [],
        "mean_oracle_acc": [],
        "routing_regret_pp": [],
        "route_to_whitening_frac": [],
    }
    for _ in range(n_boot):
        draw = rng.randint(0, len(clusters), size=len(clusters))
        index = cluster_indices(labels, draw)
        chosen_mean = chosen[index].mean()
        oracle_mean = oracle[index].mean()
        samples["mean_chosen_acc"].append(100.0 * chosen_mean)
        samples["mean_oracle_acc"].append(100.0 * oracle_mean)
        samples["routing_regret_pp"].append(100.0 * (oracle_mean - chosen_mean))
        samples["route_to_whitening_frac"].append(route[index].mean())
    return {key: percentile(values) for key, values in samples.items()}


def routing_summary(
    rows: list[dict],
    pred: np.ndarray,
    labels: np.ndarray,
    n_boot: int,
    seed: int,
) -> list[dict]:
    output = []
    for offset, threshold in enumerate(THRESHOLDS):
        row = routing_point(rows, pred, threshold)
        row["dataset_cluster_ci95"] = cluster_routing_ci(
            rows,
            pred,
            labels,
            threshold,
            n_boot,
            seed + 101 * offset,
        )
        output.append(row)
    return output


def fold_rows(rows: list[dict], pred: np.ndarray) -> list[dict]:
    labels = np.asarray([row["dataset"] for row in rows], dtype=object)
    target = np.asarray([row[TARGET] for row in rows], dtype=np.float64)
    output = []
    for dataset in sorted(set(labels.tolist())):
        index = np.flatnonzero(labels == dataset)
        chosen, oracle, route = routing_arrays(
            [rows[i] for i in index], pred[index], threshold=5.0
        )
        output.append(
            {
                "dataset": dataset,
                "n_shot_groups": int(len(index)),
                "mean_target_pp": float(target[index].mean()),
                "mean_prediction_pp": float(pred[index].mean()),
                **metrics(pred[index], target[index]),
                "routing_regret_5pp": float(100.0 * (oracle.mean() - chosen.mean())),
                "route_to_whitening_frac_5pp": float(route.mean()),
            }
        )
    return output


def paired_incremental_ci(
    rows: list[dict],
    full_pred: np.ndarray,
    baseline_pred: np.ndarray,
    labels: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    target = np.asarray([row[TARGET] for row in rows], dtype=np.float64)
    full_chosen, oracle, _ = routing_arrays(rows, full_pred, threshold=5.0)
    base_chosen, _, _ = routing_arrays(rows, baseline_pred, threshold=5.0)
    full_error = full_pred - target
    base_error = baseline_pred - target
    point_deltas = {
        "mae_pp": float(
            np.mean(np.abs(full_error)) - np.mean(np.abs(base_error))
        ),
        "rmse_pp": float(
            np.sqrt(np.mean(full_error**2))
            - np.sqrt(np.mean(base_error**2))
        ),
        "routing_regret_5pp": float(
            100.0 * (base_chosen.mean() - full_chosen.mean())
        ),
    }
    clusters = sorted(set(labels.tolist()))
    rng = np.random.RandomState(seed)
    deltas = {"mae_pp": [], "rmse_pp": [], "routing_regret_5pp": []}
    for _ in range(n_boot):
        draw = rng.randint(0, len(clusters), size=len(clusters))
        index = cluster_indices(labels, draw)
        full_error = full_pred[index] - target[index]
        base_error = baseline_pred[index] - target[index]
        deltas["mae_pp"].append(
            float(np.mean(np.abs(full_error)) - np.mean(np.abs(base_error)))
        )
        deltas["rmse_pp"].append(
            float(
                np.sqrt(np.mean(full_error**2))
                - np.sqrt(np.mean(base_error**2))
            )
        )
        full_regret = 100.0 * (
            oracle[index].mean() - full_chosen[index].mean()
        )
        base_regret = 100.0 * (
            oracle[index].mean() - base_chosen[index].mean()
        )
        deltas["routing_regret_5pp"].append(float(full_regret - base_regret))
    return {
        key: {
            "point_full_minus_baseline": point_deltas[key],
            "bootstrap_mean_full_minus_baseline": float(np.mean(values)),
            "ci95": percentile(values),
            "bootstrap_fraction_favoring_full": float(
                np.mean(np.asarray(values) < 0.0)
            ),
        }
        for key, values in deltas.items()
    }


def find_lodo_prediction(predictive: list[dict]) -> dict:
    matches = [
        row
        for row in predictive
        if row["target"] == TARGET and row["fold"] == "leave_one_dataset"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one LODO prediction block, found {len(matches)}")
    return matches[0]


def ablation_reference(ablation: list[dict], feature_set: str) -> dict:
    matches = [
        row
        for row in ablation
        if row["target"] == TARGET
        and row["fold"] == "leave_one_dataset"
        and row["feature_set"] == feature_set
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one ablation row for {feature_set}")
    return matches[0]


def fullshot_tail_removal(supplement: list[dict]) -> dict:
    if len(supplement) != 90:
        raise ValueError(f"expected 90 supplement records, found {len(supplement)}")
    selected = [
        row
        for row in supplement
        if str(row["shot"]) == "full" and int(row["seed"]) == 0
    ]
    if len(selected) != 9:
        raise ValueError(f"expected nine full-shot records, found {len(selected)}")
    output = []
    for row in sorted(selected, key=lambda item: item["dataset"]):
        bands = {band["band"]: band for band in row["pca_bands"]}
        if "all" not in bands or "top128" not in bands:
            raise ValueError(f"missing PCA band for {row['dataset']}")
        full_gap = float(bands["all"]["gap_pp"])
        top_gap = float(bands["top128"]["gap_pp"])
        output.append(
            {
                "dataset": row["dataset"],
                "feature_dim": int(row["sdgi"]["dim"]),
                "all_pc_gap_pp": full_gap,
                "retain_top128_gap_pp": top_gap,
                "all_minus_top128_gap_pp": full_gap - top_gap,
            }
        )
    return {
        "scope": "DINOv2-B full-shot, seed 0",
        "intervention": "retain PCs 1--128; remove the M_128 tail after PC 128",
        "selection_warning": (
            "Mahalanobis regularization is best-over-test in each band; "
            "use only as a mechanism diagnostic, not a train-validated estimate"
        ),
        "rows": output,
    }


def audit(args: argparse.Namespace) -> dict:
    input_paths = {
        "joined_records": args.joined,
        "predictive_results": args.predictive,
        "feature_ablation": args.ablation,
        "supplement_stats": args.supplement,
    }
    for name, path in input_paths.items():
        if path is not None:
            validate_md5(path, FROZEN_INPUTS[name]["etag"])
    rows = read_json(args.joined)
    predictive = read_json(args.predictive)
    ablation = read_json(args.ablation)
    supplement = read_json(args.supplement) if args.supplement is not None else None
    if len(rows) != 54:
        raise ValueError(f"expected 54 dataset-shot rows, found {len(rows)}")
    keys = [(row["dataset"], str(row["shot"])) for row in rows]
    if len(set(keys)) != 54:
        raise ValueError("dataset-shot keys must be unique")
    labels = np.asarray([row["dataset"] for row in rows], dtype=object)
    if len(set(labels.tolist())) != 9:
        raise ValueError("expected nine dataset clusters")

    lodo = find_lodo_prediction(predictive)
    prediction_keys = [
        (row["dataset"], str(row["shot"])) for row in lodo["predictions"]
    ]
    if prediction_keys != keys:
        raise ValueError("prediction order does not match joined records")
    pred = np.asarray([row["pred_pp"] for row in lodo["predictions"]])
    target = np.asarray([row[TARGET] for row in rows], dtype=np.float64)
    if not np.allclose(target, [row["target_pp"] for row in lodo["predictions"]]):
        raise ValueError("prediction targets do not match joined records")

    recomputed = frozen_metric_bundle(pred, target)
    for key in ["pearson", "spearman", "mae_pp", "rmse_pp"]:
        if abs(recomputed[key] - lodo[key]) > 1e-12:
            raise AssertionError(f"frozen point metric mismatch: {key}")

    model_predictions = {}
    model_points = {}
    for name, features in FEATURE_SETS.items():
        model_pred, model_target = predict_cv_checked(rows, features)
        if not np.array_equal(model_target, target):
            raise AssertionError(f"target mismatch for {name}")
        reference = ablation_reference(ablation, name)
        point = frozen_metric_bundle(model_pred, target)
        for key in ["pearson", "spearman", "mae_pp", "rmse_pp"]:
            if abs(point[key] - reference[key]) > 1e-12:
                raise AssertionError(f"ablation point mismatch for {name}/{key}")
        route5 = routing_point(rows, model_pred, 5.0)["routing_regret_pp"]
        if abs(route5 - reference["routing_regret_5pp"]) > 1e-12:
            raise AssertionError(f"ablation routing mismatch for {name}")
        model_predictions[name] = model_pred
        model_points[name] = {
            **point,
            "routing_regret_5pp": route5,
        }

    if not np.allclose(model_predictions["full"], pred, atol=1e-12, rtol=0.0):
        raise AssertionError("full ablation predictions must match frozen LODO predictions")

    no_one_shot_rows = [row for row in rows if str(row["shot"]) != "1"]
    if len(no_one_shot_rows) != 45:
        raise AssertionError("excluding 1-shot must leave 45 rows")
    no_one_pred, no_one_target = predict_cv_checked(no_one_shot_rows, FEATURES)
    no_one_labels = np.asarray(
        [row["dataset"] for row in no_one_shot_rows], dtype=object
    )

    output = {
        "protocol": {
            "frozen_inputs": FROZEN_INPUTS,
            "bootstrap_unit": "dataset cluster; all shot groups retained together",
            "n_boot": args.n_boot,
            "seed": args.seed,
            "target": TARGET,
            "fold": "leave_one_dataset",
            "thresholds_pp": THRESHOLDS,
            "delta_sign": "full-minus-baseline; negative favors full",
        },
        "main_54_groups": {
            "n_datasets": 9,
            "n_groups": 54,
            "frozen_metrics": recomputed,
            "tie_aware_metrics": metrics(pred, target),
            "dataset_cluster_ci95": cluster_metric_ci(
                pred, target, labels, args.n_boot, args.seed
            ),
            "routing": routing_summary(
                rows, pred, labels, args.n_boot, args.seed + 1000
            ),
            "folds": fold_rows(rows, pred),
        },
        "exclude_1shot": {
            "n_datasets": 9,
            "n_groups": 45,
            "metrics": metrics(no_one_pred, no_one_target),
            "dataset_cluster_ci95": cluster_metric_ci(
                no_one_pred,
                no_one_target,
                no_one_labels,
                args.n_boot,
                args.seed + 2000,
            ),
            "routing": routing_summary(
                no_one_shot_rows,
                no_one_pred,
                no_one_labels,
                args.n_boot,
                args.seed + 3000,
            ),
        },
        "feature_incremental_value": {
            "point_metrics": model_points,
            "full_vs_covariance_only": paired_incremental_ci(
                rows,
                model_predictions["full"],
                model_predictions["covariance_only"],
                labels,
                args.n_boot,
                args.seed + 4000,
            ),
            "full_vs_no_tail_fisher": paired_incremental_ci(
                rows,
                model_predictions["full"],
                model_predictions["no_tail_fisher"],
                labels,
                args.n_boot,
                args.seed + 5000,
            ),
        },
        "real_data_tail_removal": (
            fullshot_tail_removal(supplement)
            if supplement is not None
            else {"skipped": "supplement aggregate_records.json not provided"}
        ),
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joined", type=Path, default=RESULTS_ROOT / FROZEN_INPUTS["joined_records"]["released_path"]
    )
    parser.add_argument(
        "--predictive", type=Path, default=RESULTS_ROOT / FROZEN_INPUTS["predictive_results"]["released_path"]
    )
    parser.add_argument(
        "--ablation", type=Path, default=RESULTS_ROOT / FROZEN_INPUTS["feature_ablation"]["released_path"]
    )
    parser.add_argument(
        "--supplement",
        type=Path,
        default=None,
        help="optional supplement aggregate_records.json (not bundled); enables the tail-removal section",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"[router-uncertainty] wrote {args.output}")


if __name__ == "__main__":
    main()
