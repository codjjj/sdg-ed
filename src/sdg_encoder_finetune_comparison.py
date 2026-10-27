#!/usr/bin/env python3
"""Independent comparison of full-encoder and last-block fine-tuning SDG artifacts.

The two runs carry experiment_id E14 (full encoder) and E11 (last block).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np


METHODS = ("wcos_tv", "lda_tv", "validation_whitening")
ALL_METHODS = ("cos",) + METHODS
CONTRASTS = ("full_minus_last", "frozen_control", "attenuation_did")
PROTOCOL_KEYS = (
    "backbone",
    "encoder_learning_rate",
    "head_learning_rate",
    "weight_decay",
    "max_epochs",
    "validation_fraction",
    "max_train_per_class",
    "batch_size",
    "eval_batch_size",
    "seeds",
    "regularization_grid",
    "regularization_mode",
    "test_bootstrap",
    "seed_bootstrap",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inferred_scope(payload: dict) -> str:
    scope = payload.get("fine_tune_scope")
    if scope:
        return str(scope)
    kind = str(payload.get("kind", ""))
    if "last_block" in kind:
        return "last_block"
    if "full_finetune" in kind:
        return "full_encoder"
    raise ValueError(f"cannot infer fine-tuning scope from kind={kind!r}")


def prediction_gap_pp(
    predictions: dict[str, np.ndarray], labels: np.ndarray, prefix: str, method: str
) -> float:
    left = np.asarray(predictions[f"{prefix}__{method}"])
    right = np.asarray(predictions[f"{prefix}__cos"])
    labels = np.asarray(labels)
    return float(100.0 * ((left == labels).mean() - (right == labels).mean()))


def class_macro_gap_pp(
    predictions: dict[str, np.ndarray], labels: np.ndarray, prefix: str, method: str
) -> float:
    left = np.asarray(predictions[f"{prefix}__{method}"])
    right = np.asarray(predictions[f"{prefix}__cos"])
    labels = np.asarray(labels)
    values = 100.0 * ((left == labels).astype(float) - (right == labels).astype(float))
    return float(np.mean([values[labels == label].mean() for label in np.unique(labels)]))


def _contrast_values(
    terms: list[tuple[np.ndarray, float]], labels: np.ndarray
) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    values = np.zeros(len(labels), dtype=np.float64)
    for predictions, coefficient in terms:
        predictions = np.asarray(predictions)
        if predictions.shape != labels.shape:
            raise ValueError("paired prediction axes differ")
        values += float(coefficient) * (predictions == labels)
    return values


def paired_contrast_bootstrap(
    terms: list[tuple[np.ndarray, float]],
    labels: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """Bootstrap a paired correctness contrast using its exact discrete support."""

    values = _contrast_values(terms, labels)
    unique, counts = np.unique(values, return_counts=True)
    rng = np.random.RandomState(seed)
    draws = rng.multinomial(
        len(values), counts.astype(np.float64) / len(values), size=n_boot
    )
    bootstrap = (
        100.0
        * np.sum(draws.astype(np.float64) * unique[None, :], axis=1)
        / len(values)
    )
    return {
        "point_pp": float(100.0 * values.mean()),
        "ci95_pp": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "n_test": int(len(values)),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def class_cluster_contrast_bootstrap(
    terms: list[tuple[np.ndarray, float]],
    labels: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    values = _contrast_values(terms, labels)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    class_means = np.asarray([values[labels == label].mean() for label in classes])
    rng = np.random.RandomState(seed)
    draws = rng.multinomial(
        len(classes), np.full(len(classes), 1.0 / len(classes)), size=n_boot
    )
    bootstrap = (
        100.0
        * np.sum(draws.astype(np.float64) * class_means[None, :], axis=1)
        / len(classes)
    )
    return {
        "point_pp": float(100.0 * class_means.mean()),
        "ci95_pp": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "n_classes": int(len(classes)),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def joint_direction(example: dict, class_cluster: dict) -> str:
    example_low, example_high = example["ci95_pp"]
    class_low, class_high = class_cluster["ci95_pp"]
    if example_low > 0 and class_low > 0:
        return "positive"
    if example_high < 0 and class_high < 0:
        return "negative"
    return "boundary"


def value_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values_pp": array.tolist(),
        "n_seeds": int(len(array)),
        "mean_pp": float(array.mean()),
        "std_pp": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "range_pp": [float(array.min()), float(array.max())],
        "positive_seeds": int(np.sum(array > 0)),
        "zero_seeds": int(np.sum(array == 0)),
        "negative_seeds": int(np.sum(array < 0)),
    }


def _assert_close(left: float, right: float, label: str) -> None:
    if not np.isclose(left, right, rtol=0.0, atol=1e-9):
        raise AssertionError(f"{label}: {left} != {right}")


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def load_scope_dataset(
    root: Path,
    dataset: str,
    *,
    expected_scope: str,
    expected_commit: str | None,
    expected_seeds: list[int],
) -> dict:
    dataset_root = root / dataset
    aggregate_path = dataset_root / "result.json"
    aggregate = _read_json(aggregate_path)
    if aggregate.get("dataset") != dataset:
        raise ValueError(f"aggregate dataset mismatch: {aggregate_path}")
    if inferred_scope(aggregate) != expected_scope:
        raise ValueError(f"aggregate scope mismatch: {aggregate_path}")
    if expected_commit and aggregate.get("environment", {}).get("git_commit") != expected_commit:
        raise ValueError(f"aggregate commit mismatch: {aggregate_path}")
    if [row.get("seed") for row in aggregate.get("seeds", [])] != expected_seeds:
        raise ValueError(f"aggregate seed order mismatch: {aggregate_path}")

    rows = {}
    source_files = {
        str(aggregate_path): {
            "bytes": aggregate_path.stat().st_size,
            "sha256": sha256_file(aggregate_path),
        }
    }
    reference_frozen = None
    for seed in expected_seeds:
        seed_root = dataset_root / f"seed_{seed}"
        seed_path = seed_root / "result.json"
        prediction_path = seed_root / "predictions.npz"
        row = _read_json(seed_path)
        if row.get("dataset") != dataset or row.get("seed") != seed:
            raise ValueError(f"seed identity mismatch: {seed_path}")
        if inferred_scope(row) != expected_scope:
            raise ValueError(f"seed scope mismatch: {seed_path}")
        if expected_commit and row.get("environment", {}).get("git_commit") != expected_commit:
            raise ValueError(f"seed commit mismatch: {seed_path}")
        if row.get("head_test_labels_used_for_selection") is not False:
            raise ValueError(f"head selection leakage gate failed: {seed_path}")
        if row.get("fine_tune_split", {}).get("test_labels_used") is not False:
            raise ValueError(f"split leakage gate failed: {seed_path}")
        if (
            row.get("finetuned_evaluator", {})
            .get("validation_split", {})
            .get("test_labels_used")
            is not False
        ):
            raise ValueError(f"evaluator leakage gate failed: {seed_path}")
        for stage in ("epoch_selection", "refit"):
            fraction = float(row[stage]["scope"]["trainable_fraction"])
            if expected_scope == "full_encoder" and fraction != 1.0:
                raise ValueError(f"full-encoder trainable fraction failed: {seed_path}")
            if expected_scope == "last_block" and not 0.0 < fraction < 1.0:
                raise ValueError(f"last-block trainable fraction failed: {seed_path}")
        refit = row["refit"]
        if (
            refit["frozen_parameter_sha256_before"]
            != refit["frozen_parameter_sha256_after"]
        ):
            raise ValueError(f"frozen parameter digest changed: {seed_path}")
        if (
            refit["trainable_parameter_sha256_before"]
            == refit["trainable_parameter_sha256_after"]
        ):
            raise ValueError(f"trainable parameter digest unchanged: {seed_path}")

        manifest = row["artifacts"]["predictions.npz"]
        if sha256_file(prediction_path) != manifest["sha256"]:
            raise ValueError(f"prediction SHA256 mismatch: {prediction_path}")
        with np.load(prediction_path, allow_pickle=False) as bundle:
            predictions = {name: np.asarray(bundle[name]) for name in bundle.files}
        expected_arrays = {"labels"} | {
            f"{prefix}__{method}"
            for prefix in ("frozen", "finetuned")
            for method in ALL_METHODS
        }
        if set(predictions) != expected_arrays or set(manifest["arrays"]) != expected_arrays:
            raise ValueError(f"prediction array manifest mismatch: {prediction_path}")
        labels = np.asarray(predictions["labels"])
        if labels.ndim != 1 or any(
            np.asarray(predictions[name]).shape != labels.shape
            for name in expected_arrays - {"labels"}
        ):
            raise ValueError(f"prediction axes mismatch: {prediction_path}")
        if len(labels) != int(aggregate["data"]["n_test"]):
            raise ValueError(f"test row count mismatch: {prediction_path}")
        if len(np.unique(labels)) != int(aggregate["data"]["n_classes"]):
            raise ValueError(f"class coverage mismatch: {prediction_path}")

        metrics = row["finetuned_evaluator"]
        for method in ALL_METHODS:
            observed = float(
                np.mean(predictions[f"finetuned__{method}"] == labels)
            )
            _assert_close(
                observed,
                float(metrics["accuracies"][method]),
                f"{dataset}/seed{seed}/{method} accuracy",
            )
        for method in METHODS:
            gap = prediction_gap_pp(predictions, labels, "finetuned", method)
            _assert_close(
                gap,
                float(metrics["gaps_pp"][method]),
                f"{dataset}/seed{seed}/{method} gap",
            )
            _assert_close(
                gap,
                float(row["paired_test_bootstrap"][method]["point_pp"]),
                f"{dataset}/seed{seed}/{method} paired point",
            )
            _assert_close(
                class_macro_gap_pp(predictions, labels, "finetuned", method),
                float(row["class_cluster_bootstrap"][method]["point_pp"]),
                f"{dataset}/seed{seed}/{method} class point",
            )

        frozen = {name: predictions[name] for name in predictions if name.startswith("frozen__")}
        if reference_frozen is None:
            reference_frozen = frozen
        elif any(
            not np.array_equal(reference_frozen[name], frozen[name])
            for name in reference_frozen
        ):
            raise ValueError(f"frozen predictions vary across seeds: {dataset_root}")
        if row != aggregate["seeds"][expected_seeds.index(seed)]:
            raise ValueError(f"aggregate/seed JSON identity mismatch: {seed_path}")
        rows[seed] = {"metadata": row, "labels": labels, "predictions": predictions}
        for path in (seed_path, prediction_path):
            source_files[str(path)] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    return {
        "aggregate": aggregate,
        "rows": rows,
        "source_files": source_files,
    }


def _terms_for(
    last_predictions: dict[str, np.ndarray],
    full_predictions: dict[str, np.ndarray],
    method: str,
    contrast: str,
) -> list[tuple[np.ndarray, float]]:
    full_gap = [
        (full_predictions[f"finetuned__{method}"], 1.0),
        (full_predictions["finetuned__cos"], -1.0),
    ]
    last_gap = [
        (last_predictions[f"finetuned__{method}"], 1.0),
        (last_predictions["finetuned__cos"], -1.0),
    ]
    full_frozen = [
        (full_predictions[f"frozen__{method}"], 1.0),
        (full_predictions["frozen__cos"], -1.0),
    ]
    last_frozen = [
        (last_predictions[f"frozen__{method}"], 1.0),
        (last_predictions["frozen__cos"], -1.0),
    ]
    if contrast == "full_minus_last":
        return full_gap + [(array, -coefficient) for array, coefficient in last_gap]
    if contrast == "frozen_control":
        return full_frozen + [
            (array, -coefficient) for array, coefficient in last_frozen
        ]
    if contrast == "attenuation_did":
        direct = full_gap + [(array, -coefficient) for array, coefficient in last_gap]
        control = full_frozen + [
            (array, -coefficient) for array, coefficient in last_frozen
        ]
        return direct + [(array, -coefficient) for array, coefficient in control]
    raise ValueError(contrast)


def compare_dataset(
    dataset: str,
    last: dict,
    full: dict,
    *,
    seeds: list[int],
    n_boot: int,
    bootstrap_seed: int,
) -> dict:
    for key in PROTOCOL_KEYS:
        if last["aggregate"]["protocol"].get(key) != full["aggregate"]["protocol"].get(key):
            raise ValueError(f"{dataset} protocol mismatch for {key}")
    for key in ("n_train", "n_test", "n_classes", "feature_dim"):
        if last["aggregate"]["data"].get(key) != full["aggregate"]["data"].get(key):
            raise ValueError(f"{dataset} data mismatch for {key}")

    seed_rows = []
    for seed in seeds:
        last_row = last["rows"][seed]
        full_row = full["rows"][seed]
        if last_row["metadata"]["effective_seed"] != full_row["metadata"]["effective_seed"]:
            raise ValueError(f"{dataset}/seed{seed} effective seed mismatch")
        if last_row["metadata"]["fine_tune_split"] != full_row["metadata"]["fine_tune_split"]:
            raise ValueError(f"{dataset}/seed{seed} train split mismatch")
        if not np.array_equal(last_row["labels"], full_row["labels"]):
            raise ValueError(f"{dataset}/seed{seed} test label axes mismatch")
        labels = full_row["labels"]
        method_rows = {}
        for method_index, method in enumerate(METHODS):
            contrasts = {}
            for contrast_index, contrast in enumerate(CONTRASTS):
                terms = _terms_for(
                    last_row["predictions"],
                    full_row["predictions"],
                    method,
                    contrast,
                )
                seed_base = (
                    bootstrap_seed
                    + seed * 100_003
                    + method_index * 10_007
                    + contrast_index * 1_009
                )
                example = paired_contrast_bootstrap(
                    terms, labels, n_boot=n_boot, seed=seed_base
                )
                cluster = class_cluster_contrast_bootstrap(
                    terms, labels, n_boot=n_boot, seed=seed_base + 503
                )
                contrasts[contrast] = {
                    "paired_example": example,
                    "class_cluster": cluster,
                    "joint_direction": joint_direction(example, cluster),
                }
            full_meta = full_row["metadata"]
            method_rows[method] = {
                "last_block_gap_pp": prediction_gap_pp(
                    last_row["predictions"], labels, "finetuned", method
                ),
                "full_encoder_gap_pp": prediction_gap_pp(
                    full_row["predictions"], labels, "finetuned", method
                ),
                "full_encoder_gap_direction": joint_direction(
                    full_meta["paired_test_bootstrap"][method],
                    full_meta["class_cluster_bootstrap"][method],
                ),
                "contrasts": contrasts,
            }
        frozen_agreement = {
            method: float(
                np.mean(
                    last_row["predictions"][f"frozen__{method}"]
                    == full_row["predictions"][f"frozen__{method}"]
                )
            )
            for method in ALL_METHODS
        }
        seed_rows.append(
            {
                "seed": seed,
                "effective_seed": full_row["metadata"]["effective_seed"],
                "selected_epoch": {
                    "last_block": last_row["metadata"]["epoch_selection"][
                        "selected_epoch"
                    ],
                    "full_encoder": full_row["metadata"]["epoch_selection"][
                        "selected_epoch"
                    ],
                },
                "selected_family": {
                    "last_block": last_row["metadata"]["finetuned_evaluator"][
                        "validation_whitening_family"
                    ],
                    "full_encoder": full_row["metadata"]["finetuned_evaluator"][
                        "validation_whitening_family"
                    ],
                },
                "frozen_prediction_agreement": frozen_agreement,
                "methods": method_rows,
            }
        )

    summaries = {}
    for method in METHODS:
        summaries[method] = {
            "last_block_gap_pp": value_summary(
                [row["methods"][method]["last_block_gap_pp"] for row in seed_rows]
            ),
            "full_encoder_gap_pp": value_summary(
                [row["methods"][method]["full_encoder_gap_pp"] for row in seed_rows]
            ),
            "full_minus_last_gap_pp": value_summary(
                [
                    row["methods"][method]["contrasts"]["full_minus_last"][
                        "paired_example"
                    ]["point_pp"]
                    for row in seed_rows
                ]
            ),
            "attenuation_did_pp": value_summary(
                [
                    row["methods"][method]["contrasts"]["attenuation_did"][
                        "paired_example"
                    ]["point_pp"]
                    for row in seed_rows
                ]
            ),
            "full_encoder_gap_directions": {
                direction: sum(
                    row["methods"][method]["full_encoder_gap_direction"] == direction
                    for row in seed_rows
                )
                for direction in ("positive", "negative", "boundary")
            },
            "full_minus_last_directions": {
                direction: sum(
                    row["methods"][method]["contrasts"]["full_minus_last"][
                        "joint_direction"
                    ]
                    == direction
                    for row in seed_rows
                )
                for direction in ("positive", "negative", "boundary")
            },
        }
    return {
        "dataset": dataset,
        "n_train": int(full["aggregate"]["data"]["n_train"]),
        "n_test": int(full["aggregate"]["data"]["n_test"]),
        "n_classes": int(full["aggregate"]["data"]["n_classes"]),
        "seeds": seed_rows,
        "summary": summaries,
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# E14 full-encoder versus E11 last-block comparison",
        "",
        "All values are percentage-point evaluator gaps. Seed summaries are descriptive "
        "means/ranges for three paired seeds, not confidence intervals.",
        "",
    ]
    for dataset in payload["datasets"]:
        lines.extend(
            [
                f"## {dataset['dataset']}",
                "",
                "| evaluator | E11 last-block | E14 full encoder | E14−E11 | attenuation DID | E14 gap P/N/B | E14−E11 P/N/B |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            summary = dataset["summary"][method]
            last = summary["last_block_gap_pp"]
            full = summary["full_encoder_gap_pp"]
            direct = summary["full_minus_last_gap_pp"]
            did = summary["attenuation_did_pp"]
            full_directions = summary["full_encoder_gap_directions"]
            direct_directions = summary["full_minus_last_directions"]
            lines.append(
                f"| {method} | {last['mean_pp']:+.3f} "
                f"[{last['range_pp'][0]:+.3f},{last['range_pp'][1]:+.3f}] | "
                f"{full['mean_pp']:+.3f} "
                f"[{full['range_pp'][0]:+.3f},{full['range_pp'][1]:+.3f}] | "
                f"{direct['mean_pp']:+.3f} | {did['mean_pp']:+.3f} | "
                f"{full_directions['positive']}/{full_directions['negative']}/"
                f"{full_directions['boundary']} | "
                f"{direct_directions['positive']}/{direct_directions['negative']}/"
                f"{direct_directions['boundary']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "- `full_minus_last` is the paired final-gap contrast.",
            "- `frozen_control` measures cross-worker numerical drift in the matched frozen baseline.",
            "- `attenuation_did` subtracts that frozen control from the final-gap contrast.",
            "- P/N/B requires both paired-example and class-cluster 95% intervals to have the same sign.",
            "- Results apply only to this DINOv2-B recipe, datasets, budget, and optimizer.",
            "",
        ]
    )
    return "\n".join(lines)


def write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(text)
        handle.flush()
    if path.read_text() != text:
        raise OSError(f"artifact readback mismatch: {path}")


def run(args: argparse.Namespace) -> dict:
    output_root = Path(args.output_root)
    result_path = output_root / "comparison.json"
    report_path = output_root / "comparison.md"
    if result_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")

    datasets = []
    sources = {}
    for dataset_index, dataset in enumerate(args.datasets):
        last = load_scope_dataset(
            Path(args.last_block_root),
            dataset,
            expected_scope="last_block",
            expected_commit=args.expected_last_block_commit,
            expected_seeds=args.seeds,
        )
        full = load_scope_dataset(
            Path(args.full_encoder_root),
            dataset,
            expected_scope="full_encoder",
            expected_commit=args.expected_full_encoder_commit,
            expected_seeds=args.seeds,
        )
        sources.update(last["source_files"])
        sources.update(full["source_files"])
        datasets.append(
            compare_dataset(
                dataset,
                last,
                full,
                seeds=args.seeds,
                n_boot=args.n_boot,
                bootstrap_seed=args.bootstrap_seed + dataset_index * 1_000_003,
            )
        )

    payload = {
        "schema_version": 1,
        "kind": "post_submission_full_encoder_vs_last_block_comparison",
        "artifact_gate": "PASS",
        "protocol": {
            "datasets": args.datasets,
            "seeds": args.seeds,
            "methods": list(METHODS),
            "contrasts": list(CONTRASTS),
            "n_boot": args.n_boot,
            "bootstrap_unit": ["paired test example", "test class cluster"],
            "direction_rule": (
                "positive/negative only when paired-example and class-cluster "
                "95% intervals share that strict sign; otherwise boundary"
            ),
        },
        "commits": {
            "last_block_experiment": args.expected_last_block_commit,
            "full_encoder_experiment": args.expected_full_encoder_commit,
            "verifier": args.commit,
        },
        "datasets": datasets,
        "source_files": sources,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    write_new(result_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    write_new(report_path, render_markdown(payload))
    print(f"[E14-VERIFY] PASS output={output_root}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--last-block-root",
        required=True,
        help="output root of sdg_encoder_finetune_audit.py --finetune-scope last_block",
    )
    parser.add_argument(
        "--full-encoder-root",
        required=True,
        help="output root of sdg_encoder_finetune_audit.py --finetune-scope full_encoder",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--expected-last-block-commit",
        default=None,
        help="optional: require this git commit in the last-block results",
    )
    parser.add_argument(
        "--expected-full-encoder-commit",
        default=None,
        help="optional: require this git commit in the full-encoder results",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["cub200", "aircraft", "cifar100"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026072414)
    parser.add_argument("--commit", default=None)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
