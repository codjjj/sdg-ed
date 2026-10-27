#!/usr/bin/env python3
"""Read-only OOD11 replay from the accepted fine-grained producers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sdg_finegrained_aggregate import (
    SHOTS,
    add_predictions,
    cluster_bootstrap,
    confusion,
    dataset_failures,
    lodo_influence,
    normalize_row,
    point_summary,
    shot_fixed_effect_residual,
    within_dataset_correlations,
)
from sdg_finegrained_manifest import canonical_json_bytes, sha256_file
from sdg_router_ood8_eval import frozen_router_predict


FINE_DATASETS = ("stanford_dogs", "foodx251", "nabirds")
EXPECTED_FINE_KEYS = {
    (dataset, shot) for dataset in FINE_DATASETS for shot in SHOTS
}
THRESHOLD_PP = 5.0


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _prediction_identity(result_path: Path, record: dict, method: str) -> dict:
    identity = record["predictions"][method]
    relative = Path(identity["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"prediction path escapes result cell: {relative}")
    path = result_path.parent / relative
    if sha256_file(path) != identity["sha256"]:
        raise ValueError(f"prediction hash mismatch: {path}")
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1 or len(values) != identity["n_rows"]:
        raise ValueError(f"prediction shape mismatch: {path}")
    return {
        "method": method,
        "path": str(path),
        "relative_path": str(relative),
        "sha256": identity["sha256"],
        "n_rows": int(identity["n_rows"]),
    }


def _validate_support_contract(payload: dict) -> None:
    dataset = payload["dataset"]
    shot = str(payload["shot"])
    expected_seeds = [0] if shot == "full" else [0, 1, 2, 3, 4]
    seeds = [int(record["seed"]) for record in payload["records"]]
    if seeds != expected_seeds:
        raise ValueError(f"seed identity mismatch for {dataset}/{shot}: {seeds}")
    for record in payload["records"]:
        if record["n_train"] != record["support"]["n_indices"]:
            raise ValueError(f"support count mismatch for {dataset}/{shot}")
        if shot == "full":
            continue
        requested = int(shot)
        maximum = requested * int(record["n_classes"])
        if record["n_train"] > maximum:
            raise ValueError(f"support exceeds requested K for {dataset}/{shot}")
        if dataset != "nabirds" and record["n_train"] != maximum:
            raise ValueError(f"strict-K dataset was silently capped: {dataset}/{shot}")
    if dataset == "nabirds" and shot in {"5", "10", "20"}:
        if not any(
            record["n_train"] < int(shot) * int(record["n_classes"])
            for record in payload["records"]
        ):
            raise ValueError(f"NABirds capped-K evidence missing for shot={shot}")


def validate_fine_results(result_paths: list[Path]) -> tuple[list[dict], list[dict]]:
    found: dict[tuple[str, str], dict] = {}
    inputs = []
    dataset_identities: dict[str, tuple[str, str]] = {}
    for result_path in map(Path, result_paths):
        ready_path = result_path.parent / "RESULT_READY.json"
        if not ready_path.exists():
            raise FileNotFoundError(f"missing result-ready artifact: {ready_path}")
        ready = read_json(ready_path)
        result_sha = sha256_file(result_path)
        if ready.get("result_sha256") != result_sha:
            raise ValueError(f"result-ready hash mismatch: {result_path}")
        payload = read_json(result_path)
        key = (str(payload["dataset"]), str(payload["shot"]))
        if key not in EXPECTED_FINE_KEYS:
            raise ValueError(f"unexpected or excluded fine-grained cell: {key}")
        if key in found:
            raise ValueError(f"duplicate fine-grained result: {key}")
        if "inat2021_mini" in result_path.as_posix() or "inat2021_mini" in json.dumps(payload):
            raise ValueError("iNaturalist is excluded from the accepted OOD11 evidence")
        _validate_support_contract(payload)
        identity = (
            payload["dataset_identity_sha256"],
            payload["feature_manifest_sha256"],
        )
        prior = dataset_identities.setdefault(key[0], identity)
        if prior != identity:
            raise ValueError(f"dataset/feature identity drift for {key[0]}")
        prediction_inputs = []
        for record in payload["records"]:
            if set(record["predictions"]) != {"cos", "wcos_trainval", "lda_trainval"}:
                raise ValueError(f"prediction method identity mismatch for {key}")
            for method in sorted(record["predictions"]):
                prediction_inputs.append(
                    {"seed": int(record["seed"]), **_prediction_identity(result_path, record, method)}
                )
        found[key] = payload
        inputs.append(
            {
                "dataset": key[0],
                "shot": key[1],
                "result_path": str(result_path),
                "result_sha256": result_sha,
                "ready_path": str(ready_path),
                "ready_sha256": sha256_file(ready_path),
                "dataset_identity_sha256": identity[0],
                "feature_manifest_sha256": identity[1],
                "feature_root": payload["feature_root"],
                "predictions": prediction_inputs,
            }
        )
    if set(found) != EXPECTED_FINE_KEYS:
        raise ValueError(
            "fine-grained OOD3 18-cell identity mismatch: "
            f"missing={sorted(EXPECTED_FINE_KEYS - set(found))} "
            f"extra={sorted(set(found) - EXPECTED_FINE_KEYS)}"
        )
    rows = [found[(dataset, shot)]["group"] for dataset in FINE_DATASETS for shot in SHOTS]
    inputs.sort(key=lambda row: (FINE_DATASETS.index(row["dataset"]), SHOTS.index(row["shot"])))
    return rows, inputs


def compose_rows(ood8: dict, fine_rows_raw: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    original = [normalize_row(row, "original_ood4") for row in ood8["original_heldout4"]["records"]]
    coarse = [normalize_row(row, "coarse_ood4") for row in ood8["scope_heldout4"]["records"]]
    fine = [normalize_row(row, "finegrained_ood3") for row in fine_rows_raw]
    blocks = {"original_ood4": original, "coarse_ood4": coarse, "finegrained_ood3": fine}
    expected = {"original_ood4": (4, 24), "coarse_ood4": (4, 24), "finegrained_ood3": (3, 18)}
    for name, rows in blocks.items():
        n_datasets, n_groups = expected[name]
        if len(rows) != n_groups or len({row["dataset"] for row in rows}) != n_datasets:
            raise ValueError(f"{name} composition mismatch")
    all_rows = original + coarse + fine
    identities = {(row["dataset"], str(row["shot"])) for row in all_rows}
    if len(all_rows) != 66 or len(identities) != 66:
        raise ValueError("OOD11 must contain exactly 11 datasets and 66 unique groups")
    if len({row["dataset"] for row in all_rows}) != 11:
        raise ValueError("OOD11 dataset composition mismatch")
    if any(row["dataset"] == "inat2021_mini" for row in all_rows):
        raise ValueError("iNaturalist cannot enter OOD11")
    return all_rows, blocks


def fixed_shot_summary(rows: list[dict]) -> list[dict]:
    output = []
    for shot in SHOTS:
        subset = [row for row in rows if str(row["shot"]) == shot]
        output.append({"shot": shot, "point": point_summary(subset)})
    return output


def suite_summary(rows: list[dict], n_boot: int, seed: int) -> dict:
    return {
        "point": point_summary(rows),
        "dataset_cluster_bootstrap": cluster_bootstrap(rows, n_boot, seed),
        "fixed_shot_cross_dataset": fixed_shot_summary(rows),
        "within_dataset_across_shot": within_dataset_correlations(rows),
        "shot_fixed_effect_residual": shot_fixed_effect_residual(rows),
        "confusion_5pp": confusion(rows),
        "leave_one_dataset_out_influence": lodo_influence(rows),
        "failures_by_dataset": dataset_failures(rows),
        "records": rows,
    }


def _reference(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def parse_test_label_inputs(values: list[str]) -> dict[str, dict]:
    output = {}
    for value in values:
        dataset, separator, raw_path = value.partition("=")
        if not separator or dataset not in FINE_DATASETS or dataset in output:
            raise ValueError(f"invalid test-label mapping: {value}")
        path = Path(raw_path)
        output[dataset] = _reference(path)
    if set(output) != set(FINE_DATASETS):
        raise ValueError("test-label inputs must cover exactly Dogs, FoodX, and NABirds")
    return output


def report(payload: dict) -> str:
    lines = ["# OOD11 frozen-router replay\n"]
    lines.append(
        "The frozen router is fit only on the nine core datasets; Stanford Dogs, FoodX-251, "
        "and NABirds are never used for fitting or refitting."
    )
    lines.append(
        "iNaturalist-2021 mini produced no accepted result and is excluded; non-full NABirds "
        "groups use min(available, K) support images per class.\n"
    )
    lines.append("| block | datasets / groups | Pearson | router / WCos / LDA regret (pp) |")
    lines.append("|---|---:|---:|---:|")
    for name in ("original_ood4", "coarse_ood4", "finegrained_ood3"):
        point = payload["blocks"][name]["point"]
        reg = point["policy_regret_pp"]
        lines.append(
            f"| {name} | {point['n_datasets']} / {point['n_groups']} | "
            f"{point['prediction']['pearson']:.3f} | {reg['router_oracle_family']:.3f} / "
            f"{reg['always_wcos']:.3f} / {reg['always_lda']:.3f} |"
        )
    for name in ("ood8", "ood11"):
        point = payload["suites"][name]["point"]
        reg = point["policy_regret_pp"]
        lines.append(
            f"| {name} | {point['n_datasets']} / {point['n_groups']} | "
            f"{point['prediction']['pearson']:.3f} | {reg['router_oracle_family']:.3f} / "
            f"{reg['always_wcos']:.3f} / {reg['always_lda']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def run(args) -> None:
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"output root must be new: {output_root}")
    main_path = Path(args.main_joined)
    ood8_path = Path(args.ood8_audit)
    optional_references = {
        name: _reference(Path(value))
        for name, value in (
            ("published_audit", args.expected_audit),
            ("published_paste_w2", args.expected_paste),
            ("source_manifest", args.source_manifest),
        )
        if value
    }
    test_label_inputs = parse_test_label_inputs(args.test_labels)
    main_rows = read_json(main_path)
    ood8 = read_json(ood8_path)
    fine_raw, fine_inputs = validate_fine_results([Path(path) for path in args.results])
    all_rows, raw_blocks = compose_rows(ood8, fine_raw)
    predictions = frozen_router_predict(main_rows, all_rows, "target_best_whitening_minus_cos_pp")
    stored_ood8 = np.asarray([
        row["router_prediction_pp"]
        for row in ood8["original_heldout4"]["records"] + ood8["scope_heldout4"]["records"]
    ])
    if not np.allclose(predictions[:48], stored_ood8, atol=args.prediction_tolerance, rtol=0.0):
        raise AssertionError("frozen OOD8 router prediction replay mismatch")
    predicted = add_predictions(all_rows, predictions)
    blocks = {
        name: [row for row in predicted if row["block"] == name]
        for name in raw_blocks
    }
    ood8_rows = blocks["original_ood4"] + blocks["coarse_ood4"]
    fine_rows = blocks["finegrained_ood3"]
    payload = {
        "schema_version": 1,
        "kind": "sdg_caev_finegrained_ood11_standalone_replay",
        "commit": args.commit,
        "protocol": {
            "router_fit_scope": "original nine paper datasets only",
            "new_dataset_fit_or_tuning": False,
            "threshold_pp": THRESHOLD_PP,
            "bootstrap_unit": "dataset cluster with all six shots retained",
            "bootstrap_replicates": args.n_boot,
            "bootstrap_seed": args.bootstrap_seed,
            "nabirds_support": "min(available,K) per class for non-full; complete official train split for full",
            "inat2021_mini": "excluded: no accepted evaluation result",
        },
        "coverage": {
            "datasets": 11,
            "groups": 66,
            "finegrained_datasets": list(FINE_DATASETS),
            "finegrained_groups": 18,
            "excluded_datasets": ["inat2021_mini"],
        },
        "inputs": {
            "main_joined": _reference(main_path),
            "ood8_audit": _reference(ood8_path),
            "finegrained_results": fine_inputs,
            "test_label_features": test_label_inputs,
            **optional_references,
        },
        "frozen_ood8_prediction_replay": {
            "passed": True,
            "records": 48,
            "max_abs_error": float(np.max(np.abs(predictions[:48] - stored_ood8))),
            "atol": args.prediction_tolerance,
        },
        "blocks": {
            name: suite_summary(rows, args.n_boot, args.bootstrap_seed)
            for name, rows in blocks.items()
        },
        "suites": {
            "ood8": suite_summary(ood8_rows, args.n_boot, args.bootstrap_seed),
            "ood11": suite_summary(predicted, args.n_boot, args.bootstrap_seed),
            "finegrained_ood3": suite_summary(fine_rows, args.n_boot, args.bootstrap_seed),
        },
    }
    output_root.mkdir(parents=True)
    aggregate_path = output_root / "router_ood11_replay.json"
    report_path = output_root / "router_ood11_replay.md"
    write_new(aggregate_path, canonical_json_bytes(payload))
    write_new(report_path, report(payload).encode("utf-8"))
    write_new(
        output_root / "AGGREGATE_READY.json",
        canonical_json_bytes(
            {
                "kind": "sdg_caev_finegrained_ood11_aggregate_ready",
                "aggregate_sha256": sha256_file(aggregate_path),
                "report_sha256": sha256_file(report_path),
                "datasets": 11,
                "groups": 66,
                "finegrained_groups": 18,
                "inat2021_mini_included": False,
            }
        ),
    )
    print(f"[ood11-aggregate-ready] {output_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-joined", required=True)
    parser.add_argument("--ood8-audit", required=True)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--expected-audit", default=None, help="optional reference file whose hash is recorded")
    parser.add_argument("--expected-paste", default=None, help="optional reference file whose hash is recorded")
    parser.add_argument("--source-manifest", default=None, help="optional input manifest whose hash is recorded")
    parser.add_argument("--test-labels", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--prediction-tolerance", type=float, default=1e-10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
