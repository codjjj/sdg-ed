#!/usr/bin/env python3
"""Strict aggregation of the fine-grained unseen datasets for the frozen router.

Combines the four fine-grained candidates with the OOD8 records (OOD12). The
paper reports the OOD11 replay (sdg_router_ood11_replay.py), which excludes
iNaturalist-2021 mini because it produced no accepted result.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sdg_finegrained_manifest import canonical_json_bytes, sha256_file
from sdg_router_ood8_eval import frozen_router_predict
from sdg_stepb_predictive_protocol import FEATURES, metric_bundle


FINE_DATASETS = ("stanford_dogs", "nabirds", "inat2021_mini", "foodx251")
SHOTS = ("1", "2", "5", "10", "20", "full")
FINE_DATASET_SHOTS = {dataset: SHOTS for dataset in FINE_DATASETS}
UNAVAILABLE_FINE_CELLS = ()
THRESHOLD_PP = 5.0


def read_json(path: Path) -> dict | list:
    return json.loads(Path(path).read_text())


def write_json_new(path: Path, payload) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def validate_fine_results(result_paths: list[Path]) -> tuple[list[dict], list[dict]]:
    expected = {
        (dataset, shot)
        for dataset in FINE_DATASETS
        for shot in FINE_DATASET_SHOTS[dataset]
    }
    found = {}
    inputs = []
    for result_path in result_paths:
        result_path = Path(result_path)
        ready_path = result_path.parent / "RESULT_READY.json"
        if not ready_path.exists():
            raise FileNotFoundError(f"missing result-ready artifact: {ready_path}")
        ready = read_json(ready_path)
        if ready["result_sha256"] != sha256_file(result_path):
            raise ValueError(f"result-ready hash mismatch: {result_path}")
        payload = read_json(result_path)
        key = (payload["dataset"], str(payload["shot"]))
        if key in found:
            raise ValueError(f"duplicate fine-grained result: {key}")
        found[key] = payload
        for record in payload["records"]:
            for prediction in record["predictions"].values():
                path = result_path.parent / prediction["path"]
                if sha256_file(path) != prediction["sha256"]:
                    raise ValueError(f"prediction hash mismatch: {path}")
                values = np.load(path, allow_pickle=False)
                if values.ndim != 1 or len(values) != prediction["n_rows"]:
                    raise ValueError(f"prediction shape mismatch: {path}")
        inputs.append(
            {
                "dataset": key[0],
                "shot": key[1],
                "result_path": str(result_path),
                "result_sha256": sha256_file(result_path),
                "ready_path": str(ready_path),
                "ready_sha256": sha256_file(ready_path),
                "dataset_identity_sha256": payload["dataset_identity_sha256"],
                "feature_manifest_sha256": payload["feature_manifest_sha256"],
            }
        )
    if set(found) != expected:
        raise ValueError(
            f"fine-grained 24-cell identity mismatch: missing={sorted(expected - set(found))} "
            f"extra={sorted(set(found) - expected)}"
        )
    rows = [
        found[(dataset, shot)]["group"]
        for dataset in FINE_DATASETS
        for shot in FINE_DATASET_SHOTS[dataset]
    ]
    return rows, inputs


def normalize_row(source: dict, block: str) -> dict:
    row = dict(source)
    row.pop("router_prediction_pp", None)
    if "target_best_whitening_minus_cos_pp" not in row:
        row["target_best_whitening_minus_cos_pp"] = float(
            row["best_whitening_minus_cos_pp"]
        )
    required = {
        "dataset",
        "shot",
        "cos",
        "wcos_trainval",
        "lda_trainval",
        "best_whitening",
        "target_best_whitening_minus_cos_pp",
        *FEATURES,
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"row {row.get('dataset')}/{row.get('shot')} missing {missing}")
    row["block"] = block
    return row


def add_predictions(rows: list[dict], predictions: np.ndarray) -> list[dict]:
    output = []
    for source, prediction in zip(rows, predictions):
        row = dict(source)
        row["router_prediction_pp"] = float(prediction)
        target_positive = row["target_best_whitening_minus_cos_pp"] >= THRESHOLD_PP
        predicted_positive = prediction >= THRESHOLD_PP
        oracle = max(row["cos"], row["best_whitening"])
        chosen = row["best_whitening"] if predicted_positive else row["cos"]
        row["router_policy_regret_pp"] = float(100.0 * (oracle - chosen))
        row["target_material_5pp"] = bool(target_positive)
        row["router_material_5pp"] = bool(predicted_positive)
        if target_positive and predicted_positive:
            decision = "true_positive"
        elif target_positive:
            decision = "false_negative"
        elif predicted_positive:
            decision = "false_positive"
        else:
            decision = "true_negative"
        row["threshold_decision"] = decision
        output.append(row)
    return output


def policy_arrays(rows: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cos = np.asarray([row["cos"] for row in rows], dtype=np.float64)
    wcos = np.asarray([row["wcos_trainval"] for row in rows], dtype=np.float64)
    lda = np.asarray([row["lda_trainval"] for row in rows], dtype=np.float64)
    whitening = np.asarray([row["best_whitening"] for row in rows], dtype=np.float64)
    predictions = np.asarray([row["router_prediction_pp"] for row in rows])
    oracle = np.maximum(cos, whitening)
    policies = {
        "router_oracle_family": np.where(predictions >= THRESHOLD_PP, whitening, cos),
        "always_wcos": wcos,
        "always_lda": lda,
        "always_cos": cos,
    }
    return oracle, policies


def point_summary(rows: list[dict]) -> dict:
    predictions = np.asarray([row["router_prediction_pp"] for row in rows], dtype=np.float64)
    target = np.asarray(
        [row["target_best_whitening_minus_cos_pp"] for row in rows], dtype=np.float64
    )
    oracle, policies = policy_arrays(rows)
    return {
        "n_groups": len(rows),
        "n_datasets": len({row["dataset"] for row in rows}),
        "prediction": metric_bundle(predictions, target),
        "policy_regret_pp": {
            name: float(100.0 * np.mean(oracle - values))
            for name, values in policies.items()
        },
    }


def cluster_bootstrap(rows: list[dict], n_boot: int, seed: int) -> dict:
    datasets = sorted({row["dataset"] for row in rows})
    by_dataset = {
        dataset: [row for row in rows if row["dataset"] == dataset]
        for dataset in datasets
    }
    rng = np.random.RandomState(seed)
    samples = defaultdict(list)
    for _ in range(n_boot):
        selected = rng.choice(datasets, size=len(datasets), replace=True)
        bootstrap_rows = [row for dataset in selected for row in by_dataset[str(dataset)]]
        point = point_summary(bootstrap_rows)
        for name, value in point["prediction"].items():
            samples[f"prediction.{name}"].append(value)
        for name, value in point["policy_regret_pp"].items():
            samples[f"policy_regret_pp.{name}"].append(value)
    return {
        "unit": "dataset cluster with all preregistered available shots retained",
        "n_clusters": len(datasets),
        "n_boot": n_boot,
        "seed": seed,
        "ci95": {
            name: [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]
            for name, values in sorted(samples.items())
        },
    }


def confusion(rows: list[dict]) -> dict:
    counts = CounterLike(row["threshold_decision"] for row in rows)
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    tn = counts["true_negative"]
    return {
        "threshold_pp": THRESHOLD_PP,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
    }


class CounterLike(defaultdict):
    def __init__(self, values):
        super().__init__(int)
        for value in values:
            self[value] += 1


def fixed_shot_correlations(rows: list[dict]) -> list[dict]:
    output = []
    for shot in SHOTS:
        subset = [row for row in rows if str(row["shot"]) == shot]
        metrics = point_summary(subset)["prediction"] if len(subset) >= 3 else None
        output.append({"shot": shot, "n_datasets": len(subset), "prediction": metrics})
    return output


def within_dataset_correlations(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        output.append(
            {
                "dataset": dataset,
                "n_shots": len(subset),
                "prediction": point_summary(subset)["prediction"],
            }
        )
    return output


def shot_fixed_effect_residual(rows: list[dict]) -> dict:
    predictions = np.asarray([row["router_prediction_pp"] for row in rows], dtype=np.float64)
    targets = np.asarray(
        [row["target_best_whitening_minus_cos_pp"] for row in rows], dtype=np.float64
    )
    residual_prediction = predictions.copy()
    residual_target = targets.copy()
    for shot in SHOTS:
        indices = np.asarray(
            [index for index, row in enumerate(rows) if str(row["shot"]) == shot]
        )
        residual_prediction[indices] -= residual_prediction[indices].mean()
        residual_target[indices] -= residual_target[indices].mean()
    return metric_bundle(residual_prediction, residual_target)


def dataset_failures(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        failed = [row for row in subset if row["threshold_decision"] in {"false_positive", "false_negative"}]
        output.append(
            {
                "dataset": dataset,
                "confusion": confusion(subset),
                "failures": [
                    {
                        "shot": row["shot"],
                        "decision": row["threshold_decision"],
                        "predicted_gap_pp": row["router_prediction_pp"],
                        "true_gap_pp": row["target_best_whitening_minus_cos_pp"],
                        "router_regret_pp": row["router_policy_regret_pp"],
                    }
                    for row in failed
                ],
            }
        )
    return output


def lodo_influence(rows: list[dict]) -> list[dict]:
    full = point_summary(rows)
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] != dataset]
        point = point_summary(subset)
        output.append(
            {
                "removed_dataset": dataset,
                "n_groups": len(subset),
                "prediction": point["prediction"],
                "delta_from_full": {
                    name: float(point["prediction"][name] - full["prediction"][name])
                    for name in point["prediction"]
                },
                "policy_regret_pp": point["policy_regret_pp"],
                "policy_regret_delta_from_full": {
                    name: float(point["policy_regret_pp"][name] - full["policy_regret_pp"][name])
                    for name in point["policy_regret_pp"]
                },
            }
        )
    return output


def suite_summary(rows: list[dict], n_boot: int, seed: int) -> dict:
    return {
        "point": point_summary(rows),
        "dataset_cluster_bootstrap": cluster_bootstrap(rows, n_boot, seed),
        "fixed_shot_cross_dataset": fixed_shot_correlations(rows),
        "within_dataset_across_shot": within_dataset_correlations(rows),
        "shot_fixed_effect_residual": shot_fixed_effect_residual(rows),
        "confusion_5pp": confusion(rows),
        "leave_one_dataset_out_influence": lodo_influence(rows),
        "failures_by_dataset": dataset_failures(rows),
        "records": rows,
    }


def report(payload: dict) -> str:
    lines = ["# Fine-grained OOD4 and OOD12 frozen-router audit\n"]
    lines.append(
        "The router, feature standardization, linear weights, and the 5pp threshold are fixed "
        "on the nine core datasets; the new OOD datasets are never used for fitting, refitting, "
        "or tuning. Negative results and false positives/negatives are reported as observed."
    )
    lines.append("\n| suite | datasets | groups | Pearson [cluster 95% CI] | Spearman | MAE | RMSE | router regret [95% CI] | WCos | LDA | Cos |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("finegrained_ood4", "combined_ood12"):
        suite = payload["suites"][name]
        point = suite["point"]
        ci = suite["dataset_cluster_bootstrap"]["ci95"]
        pred = point["prediction"]
        regrets = point["policy_regret_pp"]
        lines.append(
            f"| {name} | {point['n_datasets']} | {point['n_groups']} | "
            f"{pred['pearson']:.3f} [{ci['prediction.pearson'][0]:.3f}, {ci['prediction.pearson'][1]:.3f}] | "
            f"{pred['spearman']:.3f} | {pred['mae_pp']:.2f} | {pred['rmse_pp']:.2f} | "
            f"{regrets['router_oracle_family']:.2f} [{ci['policy_regret_pp.router_oracle_family'][0]:.2f}, {ci['policy_regret_pp.router_oracle_family'][1]:.2f}] | "
            f"{regrets['always_wcos']:.2f} | {regrets['always_lda']:.2f} | {regrets['always_cos']:.2f} |"
        )
    lines.append("\n## Fine-grained OOD4 dataset x shot\n")
    lines.append("| dataset | shot | predicted gap | true gap | Cos | WCos | LDA | router regret | decision |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    by_key = {
        (row["dataset"], str(row["shot"])): row
        for row in payload["suites"]["finegrained_ood4"]["records"]
    }
    unavailable = {
        (row["dataset"], str(row["shot"])): row
        for row in payload["coverage"]["unavailable_cells"]
    }
    for dataset in FINE_DATASETS:
        for shot in SHOTS:
            key = (dataset, shot)
            if key in unavailable:
                lines.append(
                    f"| {dataset} | {shot} | n/a | n/a | n/a | n/a | n/a | n/a | unavailable |"
                )
                continue
            row = by_key[key]
            lines.append(
                f"| {row['dataset']} | {row['shot']} | {row['router_prediction_pp']:.2f} | "
                f"{row['target_best_whitening_minus_cos_pp']:.2f} | {100*row['cos']:.2f} | "
                f"{100*row['wcos_trainval']:.2f} | {100*row['lda_trainval']:.2f} | "
                f"{row['router_policy_regret_pp']:.2f} | {row['threshold_decision']} |"
            )
    return "\n".join(lines) + "\n"


def run(args) -> None:
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"aggregate output root must be new: {output_root}")
    training_path = Path(args.main_joined)
    ood8_path = Path(args.ood8_audit)
    training_rows = read_json(training_path)
    ood8 = read_json(ood8_path)
    fine_rows_raw, fine_inputs = validate_fine_results([Path(path) for path in args.results])
    original_rows = [normalize_row(row, "original_ood4") for row in ood8["original_heldout4"]["records"]]
    coarse_rows = [normalize_row(row, "coarse_ood4") for row in ood8["scope_heldout4"]["records"]]
    fine_rows = [normalize_row(row, "finegrained_ood4") for row in fine_rows_raw]
    all_rows = original_rows + coarse_rows + fine_rows
    if len(all_rows) != 72 or len({(row["dataset"], str(row["shot"])) for row in all_rows}) != 72:
        raise ValueError("OOD12 design must contain exactly 72 observed cells")
    if set(row["dataset"] for row in fine_rows) != set(FINE_DATASETS):
        raise ValueError("fine-grained OOD4 dataset identity mismatch")
    predictions = frozen_router_predict(
        training_rows,
        all_rows,
        "target_best_whitening_minus_cos_pp",
    )
    original_count = len(original_rows)
    coarse_count = len(coarse_rows)
    stored_ood8_predictions = np.asarray(
        [
            row["router_prediction_pp"]
            for row in ood8["original_heldout4"]["records"]
            + ood8["scope_heldout4"]["records"]
        ]
    )
    if not np.allclose(
        predictions[: original_count + coarse_count],
        stored_ood8_predictions,
        atol=args.prediction_tolerance,
        rtol=0.0,
    ):
        raise AssertionError("frozen-router OOD8 prediction replay mismatch")
    predicted_rows = add_predictions(all_rows, predictions)
    blocks = {
        block: [row for row in predicted_rows if row["block"] == block]
        for block in ("original_ood4", "coarse_ood4", "finegrained_ood4")
    }
    suites = {
        "finegrained_ood4": suite_summary(
            blocks["finegrained_ood4"], args.n_boot, args.bootstrap_seed
        ),
        "combined_ood12": suite_summary(
            predicted_rows, args.n_boot, args.bootstrap_seed + 10_000
        ),
    }
    payload = {
        "schema_version": 1,
        "kind": "sdg_caev_finegrained_ood4_strict_aggregate",
        "commit": args.commit,
        "protocol": {
            "router_fit_scope": "original nine paper datasets only",
            "router_features": FEATURES,
            "threshold_pp": THRESHOLD_PP,
            "new_dataset_fit_or_tuning": False,
            "bootstrap_unit": "dataset cluster",
            "bootstrap_replicates": args.n_boot,
            "bootstrap_seed": args.bootstrap_seed,
            "router_policy_note": "router_oracle_family retains the frozen target's oracle choice between train-only WCos and LDA after routing; it is not a deployable family selector",
            "nabirds_support_note": "for non-full regimes, each class contributes min(train_count, requested_k) examples without replacement",
            "incomplete_design": False,
            "missing_value_imputation": False,
        },
        "coverage": {
            "planned_fine_cells": 24,
            "observed_fine_cells": len(fine_rows),
            "planned_ood12_cells": 72,
            "observed_ood12_cells": len(all_rows),
            "unavailable_cells": list(UNAVAILABLE_FINE_CELLS),
        },
        "inputs": {
            "main_joined": {"path": str(training_path), "sha256": sha256_file(training_path)},
            "ood8_audit": {"path": str(ood8_path), "sha256": sha256_file(ood8_path)},
            "finegrained_results": fine_inputs,
        },
        "predefined_blocks": {
            block: suite_summary(rows, args.n_boot, args.bootstrap_seed + 20_000 + index * 1_000)
            for index, (block, rows) in enumerate(blocks.items())
        },
        "suites": suites,
        "frozen_ood8_prediction_replay": {
            "passed": True,
            "atol": args.prediction_tolerance,
            "n_records": original_count + coarse_count,
        },
    }
    output_root.mkdir(parents=True)
    write_json_new(output_root / "finegrained_ood4_aggregate.json", payload)
    write_text_new(output_root / "finegrained_ood4_aggregate.md", report(payload))
    write_json_new(
        output_root / "AGGREGATE_READY.json",
        {
            "kind": "sdg_caev_finegrained_ood4_aggregate_ready",
            "aggregate_sha256": sha256_file(output_root / "finegrained_ood4_aggregate.json"),
            "report_sha256": sha256_file(output_root / "finegrained_ood4_aggregate.md"),
            "fine_cells": len(fine_rows),
            "ood12_cells": len(predicted_rows),
            "planned_fine_cells": 24,
            "planned_ood12_cells": 72,
            "unavailable_cells": list(UNAVAILABLE_FINE_CELLS),
        },
    )
    print(f"[aggregate-ready] root={output_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-joined", required=True)
    parser.add_argument("--ood8-audit", required=True)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--prediction-tolerance", type=float, default=1e-9)
    parser.add_argument("--commit", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
