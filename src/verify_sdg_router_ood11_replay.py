#!/usr/bin/env python3
"""Independent, fail-closed verifier for the standalone OOD11 replay.

Full mode re-reads every input of the replay (per-seed prediction arrays, test
labels, result files). `--records-only` needs only the aggregate JSON and the
nine-dataset training records: it refits the frozen router, replays every
stored prediction, and recomputes all point metrics, dataset-cluster
bootstrap intervals, and the published OOD4/OOD8/OOD11 numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sdg_finegrained_manifest import canonical_json_bytes, sha256_file
from sdg_stepb_predictive_protocol import FEATURES, fit_predict_linear


SHOTS = ("1", "2", "5", "10", "20", "full")
FINE_DATASETS = ("stanford_dogs", "foodx251", "nabirds")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0.0):
        raise AssertionError(f"{label}: expected={expected} actual={actual}")


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = (left - left.mean()) / (left.std() + 1e-12)
    right = (right - right.mean()) / (right.std() + 1e-12)
    return float(np.mean(left * right))


def prediction_metrics(rows: list[dict]) -> dict:
    predicted = np.asarray([row["router_prediction_pp"] for row in rows])
    target = np.asarray([row["target_best_whitening_minus_cos_pp"] for row in rows])
    return {
        "pearson": correlation(predicted, target),
        "spearman": correlation(rankdata(predicted), rankdata(target)),
        "mae_pp": float(np.mean(np.abs(predicted - target))),
        "rmse_pp": float(np.sqrt(np.mean((predicted - target) ** 2))),
    }


def policy_regrets(rows: list[dict]) -> dict:
    cos = np.asarray([row["cos"] for row in rows])
    wcos = np.asarray([row["wcos_trainval"] for row in rows])
    lda = np.asarray([row["lda_trainval"] for row in rows])
    whitening = np.asarray([row["best_whitening"] for row in rows])
    predicted = np.asarray([row["router_prediction_pp"] for row in rows])
    oracle = np.maximum(cos, whitening)
    policies = {
        "router_oracle_family": np.where(predicted >= 5.0, whitening, cos),
        "always_wcos": wcos,
        "always_lda": lda,
        "always_cos": cos,
    }
    return {name: float(100.0 * np.mean(oracle - values)) for name, values in policies.items()}


def point(rows: list[dict]) -> dict:
    return {
        "n_groups": len(rows),
        "n_datasets": len({row["dataset"] for row in rows}),
        "prediction": prediction_metrics(rows),
        "policy_regret_pp": policy_regrets(rows),
    }


def independent_bootstrap(rows: list[dict], n_boot: int, seed: int) -> dict:
    datasets = sorted({row["dataset"] for row in rows})
    grouped = {dataset: [row for row in rows if row["dataset"] == dataset] for dataset in datasets}
    rng = np.random.RandomState(seed)
    samples = defaultdict(list)
    for _ in range(n_boot):
        selected = rng.choice(datasets, size=len(datasets), replace=True)
        sample = [row for dataset in selected for row in grouped[str(dataset)]]
        for name, value in prediction_metrics(sample).items():
            samples[f"prediction.{name}"].append(value)
        for name, value in policy_regrets(sample).items():
            samples[f"policy_regret_pp.{name}"].append(value)
    return {
        name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for name, values in sorted(samples.items())
    }


def residual_metrics(rows: list[dict]) -> dict:
    predicted = np.asarray([row["router_prediction_pp"] for row in rows], dtype=np.float64)
    target = np.asarray([row["target_best_whitening_minus_cos_pp"] for row in rows], dtype=np.float64)
    for shot in SHOTS:
        indices = np.asarray([index for index, row in enumerate(rows) if str(row["shot"]) == shot])
        predicted[indices] -= predicted[indices].mean()
        target[indices] -= target[indices].mean()
    return {
        "pearson": correlation(predicted, target),
        "spearman": correlation(rankdata(predicted), rankdata(target)),
        "mae_pp": float(np.mean(np.abs(predicted - target))),
        "rmse_pp": float(np.sqrt(np.mean((predicted - target) ** 2))),
    }


def confusion(rows: list[dict]) -> dict:
    values = defaultdict(int)
    for row in rows:
        target = row["target_best_whitening_minus_cos_pp"] >= 5.0
        predicted = row["router_prediction_pp"] >= 5.0
        values[(target, predicted)] += 1
    return {
        "true_positive": values[(True, True)],
        "false_positive": values[(False, True)],
        "false_negative": values[(True, False)],
        "true_negative": values[(False, False)],
    }


def compare_point(actual: dict, stored: dict, tolerance: float, label: str) -> None:
    if actual["n_groups"] != stored["n_groups"] or actual["n_datasets"] != stored["n_datasets"]:
        raise AssertionError(f"{label} coverage mismatch")
    for section in ("prediction", "policy_regret_pp"):
        for name, value in actual[section].items():
            close(value, stored[section][name], tolerance, f"{label}/{section}/{name}")


def validate_prediction_arrays(aggregate: dict, tolerance: float) -> dict:
    label_payloads = {}
    for dataset, entry in aggregate["inputs"]["test_label_features"].items():
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"test-label feature hash mismatch: {dataset}")
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        label_payloads[dataset] = payload["labels"].long().numpy()

    stored_rows = {
        (row["dataset"], str(row["shot"])): row
        for row in aggregate["suites"]["finegrained_ood3"]["records"]
    }
    records = arrays = 0
    identity_rows = []
    for entry in aggregate["inputs"]["finegrained_results"]:
        result_path = Path(entry["result_path"])
        ready_path = Path(entry["ready_path"])
        if sha256_file(result_path) != entry["result_sha256"]:
            raise ValueError(f"result input changed: {result_path}")
        if sha256_file(ready_path) != entry["ready_sha256"]:
            raise ValueError(f"ready input changed: {ready_path}")
        ready = read_json(ready_path)
        if ready.get("result_sha256") != entry["result_sha256"]:
            raise ValueError(f"ready/result binding mismatch: {result_path}")
        result = read_json(result_path)
        labels = label_payloads[result["dataset"]]
        group_values = defaultdict(list)
        declared = {
            (int(row["seed"]), row["method"]): row for row in entry["predictions"]
        }
        for record in result["records"]:
            values = {}
            for method, identity in record["predictions"].items():
                indexed = declared[(int(record["seed"]), method)]
                path = result_path.parent / identity["path"]
                digest = sha256_file(path)
                if digest != identity["sha256"] or digest != indexed["sha256"]:
                    raise ValueError(f"prediction input changed: {path}")
                predictions = np.load(path, allow_pickle=False)
                if predictions.ndim != 1 or len(predictions) != len(labels):
                    raise ValueError(f"prediction shape mismatch: {path}")
                accuracy = float(np.mean(predictions == labels))
                close(accuracy, record["metrics"][method], tolerance, f"{result['dataset']}/{result['shot']}/{record['seed']}/{method}")
                values[method] = accuracy
                group_values[method].append(accuracy)
                identity_rows.append((result["dataset"], str(result["shot"]), int(record["seed"]), method, digest, len(predictions)))
                arrays += 1
            best = max(values["wcos_trainval"], values["lda_trainval"])
            close(best, record["metrics"]["best_whitening"], tolerance, "record best whitening")
            group_values["best_whitening"].append(best)
            records += 1
        group = stored_rows[(result["dataset"], str(result["shot"]))]
        for method in ("cos", "wcos_trainval", "lda_trainval", "best_whitening"):
            close(float(np.mean(group_values[method])), group[method], tolerance, f"group/{result['dataset']}/{result['shot']}/{method}")
    if records != 78 or arrays != 234 or len(set(identity_rows)) != 234:
        raise ValueError(f"prediction replay coverage mismatch: records={records} arrays={arrays}")
    return {"seed_records": records, "prediction_arrays": arrays, "unique_identities": 234}


def validate_router(aggregate: dict, tolerance: float, main_joined: Path | None = None) -> dict:
    main_entry = aggregate["inputs"]["main_joined"]
    main_path = Path(main_joined) if main_joined else Path(main_entry["path"])
    if sha256_file(main_path) != main_entry["sha256"]:
        raise ValueError(f"main joined input changed: {main_path}")
    if main_joined is None:
        ood8_entry = aggregate["inputs"]["ood8_audit"]
        if sha256_file(Path(ood8_entry["path"])) != ood8_entry["sha256"]:
            raise ValueError("OOD8 input changed")
    training = read_json(main_path)
    rows = aggregate["suites"]["ood11"]["records"]
    train_x = np.asarray([[row[name] for name in FEATURES] for row in training])
    train_y = np.asarray([row["target_best_whitening_minus_cos_pp"] for row in training])
    test_x = np.asarray([[row[name] for name in FEATURES] for row in rows])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        predicted = fit_predict_linear(train_x, train_y, test_x)
    if not np.isfinite(predicted).all():
        raise FloatingPointError("non-finite independent router predictions")
    stored = np.asarray([row["router_prediction_pp"] for row in rows])
    if not np.allclose(predicted, stored, atol=tolerance, rtol=0.0):
        raise AssertionError("frozen router replay mismatch")
    return {"records": 66, "max_abs_error": float(np.max(np.abs(predicted - stored)))}


def validate_source_manifest(aggregate: dict) -> dict:
    entry = aggregate["inputs"]["source_manifest"]
    path = Path(entry["path"])
    if sha256_file(path) != entry["sha256"]:
        raise ValueError("source manifest changed")
    source = read_json(path)
    if set(source.get("accepted_task_ids", {})) != set(FINE_DATASETS):
        raise ValueError("accepted producer coverage mismatch")
    if source.get("excluded_datasets") != ["inat2021_mini"]:
        raise ValueError("source manifest iNaturalist exclusion mismatch")
    results = {
        (row["dataset"], str(row["shot"])): row
        for row in source.get("results", [])
    }
    expected = {(dataset, shot) for dataset in FINE_DATASETS for shot in SHOTS}
    if set(results) != expected:
        raise ValueError("source manifest result coverage mismatch")
    aggregate_results = {
        (row["dataset"], str(row["shot"])): row
        for row in aggregate["inputs"]["finegrained_results"]
    }
    for key in expected:
        if results[key]["result_sha256"] != aggregate_results[key]["result_sha256"]:
            raise ValueError(f"source/result hash mismatch: {key}")
        portable_identities = [
            {
                name: row[name]
                for name in ("seed", "method", "relative_path", "sha256", "n_rows")
            }
            for row in aggregate_results[key]["predictions"]
        ]
        identity_sha = hashlib.sha256(
            canonical_json_bytes(portable_identities)
        ).hexdigest()
        if results[key]["prediction_identity_sha256"] != identity_sha:
            raise ValueError(f"source/prediction identity mismatch: {key}")
    for dataset, local in aggregate["inputs"]["test_label_features"].items():
        if source["test_features"][dataset]["sha256"] != local["sha256"]:
            raise ValueError(f"source/test-feature hash mismatch: {dataset}")
    return {"results": 18, "accepted_tasks": 3, "excluded_datasets": 1}


def validate_statistics(aggregate: dict, tolerance: float) -> dict:
    for name, suite in {**aggregate["blocks"], **aggregate["suites"]}.items():
        rows = suite["records"]
        compare_point(point(rows), suite["point"], tolerance, name)
        ci = independent_bootstrap(rows, aggregate["protocol"]["bootstrap_replicates"], aggregate["protocol"]["bootstrap_seed"])
        for key, interval in ci.items():
            for index in (0, 1):
                close(interval[index], suite["dataset_cluster_bootstrap"]["ci95"][key][index], tolerance, f"{name}/{key}/ci{index}")
        residual = residual_metrics(rows)
        for key, value in residual.items():
            close(value, suite["shot_fixed_effect_residual"][key], tolerance, f"{name}/demeaned/{key}")
        for fixed in suite["fixed_shot_cross_dataset"]:
            subset = [row for row in rows if str(row["shot"]) == fixed["shot"]]
            compare_point(point(subset), fixed["point"], tolerance, f"{name}/shot/{fixed['shot']}")
    fine = aggregate["suites"]["finegrained_ood3"]["records"]
    if confusion(fine) != {key: aggregate["suites"]["finegrained_ood3"]["confusion_5pp"][key] for key in ("true_positive", "false_positive", "false_negative", "true_negative")}:
        raise AssertionError("fine-grained confusion mismatch")
    return {"blocks": 3, "suites": 3, "bootstrap_seed": aggregate["protocol"]["bootstrap_seed"]}


def _rounded(value: float, digits: int) -> float:
    return float(f"{value:.{digits}f}")


def validate_published_snapshot(aggregate: dict) -> dict:
    expected = {
        "original_ood4": (0.472, 1.150, 0.395, 0.931),
        "coarse_ood4": (0.914, 1.856, 1.014, 0.549),
        "finegrained_ood3": (0.500, 0.168, 2.264, 0.335),
        "ood8": (0.791, 1.503, 0.705, 0.740),
        "ood11": (0.523, 1.139, 1.130, 0.630),
    }
    sources = {**aggregate["blocks"], **aggregate["suites"]}
    for name, values in expected.items():
        point_value = sources[name]["point"]
        regrets = point_value["policy_regret_pp"]
        actual = (
            _rounded(point_value["prediction"]["pearson"], 3),
            _rounded(regrets["router_oracle_family"], 3),
            _rounded(regrets["always_wcos"], 3),
            _rounded(regrets["always_lda"], 3),
        )
        if actual != values:
            raise AssertionError(f"published block mismatch for {name}: {actual} != {values}")
    ood11 = aggregate["suites"]["ood11"]
    ci = ood11["dataset_cluster_bootstrap"]["ci95"]
    expected_ci = {
        "prediction.pearson": (0.285, 0.786),
        "policy_regret_pp.router_oracle_family": (0.390, 2.055),
        "policy_regret_pp.always_wcos": (0.555, 1.787),
        "policy_regret_pp.always_lda": (0.385, 0.907),
    }
    for key, expected_interval in expected_ci.items():
        actual = tuple(_rounded(value, 3) for value in ci[key])
        if actual != expected_interval:
            raise AssertionError(f"published CI mismatch for {key}: {actual}")
    fine = aggregate["suites"]["finegrained_ood3"]
    conf = fine["confusion_5pp"]
    if (conf["true_positive"], conf["false_positive"], conf["false_negative"], conf["true_negative"]) != (1, 14, 0, 3):
        raise AssertionError("published confusion mismatch")
    expected_mae = {"stanford_dogs": 10.06, "foodx251": 14.76, "nabirds": 12.64}
    for row in fine["within_dataset_across_shot"]:
        if _rounded(row["prediction"]["mae_pp"], 2) != expected_mae[row["dataset"]]:
            raise AssertionError(f"published MAE mismatch for {row['dataset']}")
    return {"published_blocks": 5, "published_ci": 4, "fine_confusion": [1, 14, 0, 3], "dataset_mae": expected_mae}


def run(args) -> None:
    root = Path(args.aggregate_root)
    aggregate_path = root / "router_ood11_replay.json"
    ready_path = root / "AGGREGATE_READY.json"
    aggregate = read_json(aggregate_path)
    if args.records_only:
        run_records_only(args, aggregate, aggregate_path)
        return
    ready = read_json(ready_path)
    if ready.get("aggregate_sha256") != sha256_file(aggregate_path):
        raise ValueError("aggregate-ready hash mismatch")
    if aggregate.get("coverage") != {
        "datasets": 11,
        "groups": 66,
        "finegrained_datasets": list(FINE_DATASETS),
        "finegrained_groups": 18,
        "excluded_datasets": ["inat2021_mini"],
    }:
        raise ValueError("OOD11 coverage identity mismatch")
    if aggregate["protocol"].get("bootstrap_unit") != "dataset cluster with all six shots retained" or aggregate["protocol"].get("bootstrap_seed") != 20260803:
        raise ValueError("bootstrap contract mismatch")
    if aggregate["protocol"].get("inat2021_mini") != "excluded: no accepted evaluation result":
        raise ValueError("iNaturalist exclusion gate failed")
    for name in ("published_audit", "published_paste_w2"):
        entry = aggregate["inputs"].get(name)
        if entry is None:
            continue
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"published reference changed: {name}")
    prediction_gate = validate_prediction_arrays(aggregate, args.tolerance)
    source_gate = (
        validate_source_manifest(aggregate)
        if "source_manifest" in aggregate["inputs"]
        else {"skipped": "no source manifest recorded"}
    )
    router_gate = validate_router(aggregate, args.tolerance)
    statistics_gate = validate_statistics(aggregate, args.tolerance)
    publication_gate = validate_published_snapshot(aggregate)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    payload = {
        "schema_version": 1,
        "kind": "sdg_caev_finegrained_ood11_independent_validation",
        "accepted": True,
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "ready_path": str(ready_path),
        "ready_sha256": sha256_file(ready_path),
        "prediction_replay": prediction_gate,
        "source_manifest_replay": source_gate,
        "router_replay": router_gate,
        "statistics_replay": statistics_gate,
        "published_snapshot_replay": publication_gate,
        "coverage": {"datasets": 11, "groups": 66, "finegrained_groups": 18},
        "inat2021_mini_included": False,
        "partial_results_used": False,
    }
    output.write_bytes(canonical_json_bytes(payload))
    print(f"[ood11-independent-validation] PASS {output}", flush=True)


def check_identity(aggregate: dict) -> None:
    if aggregate.get("coverage") != {
        "datasets": 11,
        "groups": 66,
        "finegrained_datasets": list(FINE_DATASETS),
        "finegrained_groups": 18,
        "excluded_datasets": ["inat2021_mini"],
    }:
        raise ValueError("OOD11 coverage identity mismatch")
    protocol = aggregate["protocol"]
    if (
        protocol.get("bootstrap_unit") != "dataset cluster with all six shots retained"
        or protocol.get("bootstrap_seed") != 20260803
    ):
        raise ValueError("bootstrap contract mismatch")


def run_records_only(args, aggregate: dict, aggregate_path: Path) -> None:
    """Verify the aggregate from its stored records and the core training records."""

    if not args.main_joined:
        raise SystemExit("--records-only requires --main-joined (core joined_records.json)")
    check_identity(aggregate)
    router_gate = validate_router(aggregate, args.tolerance, Path(args.main_joined))
    statistics_gate = validate_statistics(aggregate, args.tolerance)
    publication_gate = validate_published_snapshot(aggregate)
    payload = {
        "schema_version": 1,
        "kind": "sdg_ood11_records_only_validation",
        "accepted": True,
        "aggregate_sha256": sha256_file(aggregate_path),
        "router_replay": router_gate,
        "statistics_replay": statistics_gate,
        "published_snapshot_replay": publication_gate,
        "skipped": ["prediction-array replay", "source-manifest replay"],
    }
    text = canonical_json_bytes(payload)
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(text)
    print(text.decode("utf-8"), end="")
    print("[ood11-records-only-validation] PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--records-only",
        action="store_true",
        help="verify from the aggregate JSON and --main-joined only (no prediction arrays)",
    )
    parser.add_argument("--main-joined", default=None, help="core joined_records.json (records-only mode)")
    args = parser.parse_args()
    if not args.records_only and not args.output:
        parser.error("--output is required in full mode")
    run(args)


if __name__ == "__main__":
    main()
