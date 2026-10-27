#!/usr/bin/env python3
"""Verify completed ImageNet-1K multi-backbone artifacts (Appendix J)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

import sdg_imagenet1k_controlled_multibackbone as audit


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_READY_MANIFEST_SHA256 = (
    "4c81ca72cd630dab24430582f7a9a133caa756bc33a3aff3eb5b4f058e727ce2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--expected-commit", default=None, help="optional: require this git commit in every result"
    )
    parser.add_argument("--verifier-commit", default=None)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--evidence-json-output", type=Path)
    parser.add_argument("--evidence-md-output", type=Path)
    parser.add_argument("--skip-feature-rehash", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def require_success(path: Path) -> None:
    if path.read_text(encoding="utf-8") != "ok\n":
        raise ValueError(f"invalid success marker: {path}")


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"invalid SHA256 for {label}: {value!r}")
    return value


def require_gap_stat(payload: object, label: str) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"expected gap object for {label}")
    point = payload.get("point_pp")
    interval = payload.get("ci95_pp")
    if not isinstance(point, (int, float)) or not math.isfinite(float(point)):
        raise ValueError(f"non-finite point estimate for {label}")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in interval)
        or float(interval[0]) > float(interval[1])
    ):
        raise ValueError(f"invalid confidence interval for {label}")
    if payload.get("n_boot") != 10_000:
        raise ValueError(f"unexpected bootstrap count for {label}")


def verify_feature_cache(
    output_root: Path,
    backbone: str,
    payload: dict,
    *,
    rehash: bool,
) -> dict:
    expected_dim = int(audit.BACKBONES[backbone]["expected_dim"])
    summary = {}
    for split, expected_rows in (
        ("train", audit.EXPECTED_TRAIN_ROWS),
        ("validation", audit.EXPECTED_VALIDATION_ROWS),
    ):
        metadata = payload["feature_metadata"][split]
        if metadata["rows"] != expected_rows or metadata["dim"] != expected_dim:
            raise ValueError(f"{backbone}/{split}: feature metadata shape mismatch")
        repeated = metadata.get("repeated_forward_max_abs")
        tolerance = metadata.get("repeated_forward_max_abs_tolerance")
        if (
            not isinstance(repeated, (int, float))
            or not isinstance(tolerance, (int, float))
            or float(repeated) > float(tolerance)
            or float(tolerance) != audit.REPEATED_FORWARD_MAX_ABS_TOL
        ):
            raise ValueError(f"{backbone}/{split}: repeated-forward gate failed")

        split_root = output_root / backbone / "feature_cache" / split
        feature_path = split_root / "features.npy"
        label_path = split_root / "labels.npy"
        features = np.load(feature_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        if features.shape != (expected_rows, expected_dim) or features.dtype != np.float32:
            raise ValueError(f"{backbone}/{split}: feature array contract failed")
        if labels.shape != (expected_rows,) or labels.dtype != np.int16:
            raise ValueError(f"{backbone}/{split}: label array contract failed")
        counts = np.bincount(
            np.asarray(labels, dtype=np.int64), minlength=audit.EXPECTED_CLASSES
        )
        if len(counts) != audit.EXPECTED_CLASSES or np.any(counts <= 0):
            raise ValueError(f"{backbone}/{split}: label coverage failed")
        if (
            int(counts.min()) != metadata["class_count_min"]
            or float(np.median(counts)) != metadata["class_count_median"]
            or int(counts.max()) != metadata["class_count_max"]
        ):
            raise ValueError(f"{backbone}/{split}: class-count metadata mismatch")

        files = {}
        for name, path in (("features.npy", feature_path), ("labels.npy", label_path)):
            recorded = payload["feature_cache"][split][name]
            if path.stat().st_size != recorded["bytes"]:
                raise ValueError(f"{backbone}/{split}/{name}: byte-size mismatch")
            expected_sha256 = require_sha256(
                recorded.get("sha256"), f"{backbone}/{split}/{name}"
            )
            metadata_sha256 = metadata[
                "feature_sha256" if name == "features.npy" else "label_sha256"
            ]
            if expected_sha256 != metadata_sha256:
                raise ValueError(f"{backbone}/{split}/{name}: metadata SHA256 mismatch")
            if rehash:
                actual_sha256 = audit.sha256_file(path)
                if actual_sha256 != expected_sha256:
                    raise ValueError(f"{backbone}/{split}/{name}: SHA256 mismatch")
            files[name] = {
                "bytes": int(path.stat().st_size),
                "sha256": expected_sha256,
                "rehash_verified": bool(rehash),
            }
        summary[split] = {
            "rows": expected_rows,
            "dim": expected_dim,
            "repeated_forward_max_abs": float(repeated),
            "files": files,
        }
    return summary


def verify_result_payload(payload: dict, backbone: str, expected_commit: str | None) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError(f"{backbone}: schema version mismatch")
    if payload.get("kind") != "sdg_imagenet1k_controlled_multibackbone_backbone_result":
        raise ValueError(f"{backbone}: result kind mismatch")
    commit_mismatch = expected_commit is not None and payload.get("git_commit") != expected_commit
    if commit_mismatch or payload.get("backbone") != backbone:
        raise ValueError(f"{backbone}: provenance mismatch")
    if payload.get("backbone_spec") != audit.BACKBONES[backbone]:
        raise ValueError(f"{backbone}: backbone specification mismatch")
    if (
        payload.get("source_ready_manifest_sha256")
        != EXPECTED_READY_MANIFEST_SHA256
        or payload.get("source_ready_manifest_total_bytes")
        != audit.EXPECTED_SOURCE_BYTES
    ):
        raise ValueError(f"{backbone}: source ready-manifest mismatch")
    source = payload.get("source", {})
    if source.get("repo") != audit.SOURCE_REPO or source.get("revision") != audit.SOURCE_REVISION:
        raise ValueError(f"{backbone}: source identity mismatch")
    for split, expected_rows, expected_shards in (
        ("train", audit.EXPECTED_TRAIN_ROWS, audit.EXPECTED_TRAIN_SHARDS),
        ("validation", audit.EXPECTED_VALIDATION_ROWS, audit.EXPECTED_VALIDATION_SHARDS),
    ):
        split_source = source.get("splits", {}).get(split, {})
        if (
            split_source.get("rows") != expected_rows
            or len(split_source.get("shards", [])) != expected_shards
        ):
            raise ValueError(f"{backbone}/{split}: source split mismatch")
    model_files = payload.get("model_identity", {}).get("files")
    if not isinstance(model_files, list) or not model_files:
        raise ValueError(f"{backbone}: model identity is empty")
    for index, model_file in enumerate(model_files):
        require_sha256(model_file.get("sha256"), f"{backbone}/model/{index}")
        if not isinstance(model_file.get("bytes"), int) or model_file["bytes"] <= 0:
            raise ValueError(f"{backbone}/model/{index}: invalid byte size")
    protocol = payload.get("protocol", {})
    if protocol.get("split_seeds") != list(audit.SPLIT_SEEDS):
        raise ValueError(f"{backbone}: split seeds mismatch")
    if protocol.get("regularization_grid") != list(audit.REG_GRID):
        raise ValueError(f"{backbone}: regularization grid mismatch")
    if protocol.get("n_boot") != 10_000:
        raise ValueError(f"{backbone}: bootstrap count mismatch")
    if protocol.get("exact_imagenet100_labels") != list(audit.IMAGENET100_LABELS):
        raise ValueError(f"{backbone}: exact-100 mapping mismatch")
    if set(payload.get("arms", {})) != set(audit.ARM_ORDER):
        raise ValueError(f"{backbone}: arm set mismatch")

    for arm in audit.ARM_ORDER:
        results = payload["arms"][arm]
        if len(results) != len(audit.SPLIT_SEEDS):
            raise ValueError(f"{backbone}/{arm}: seed count mismatch")
        expected_classes = len(audit.ARMS[arm]["labels"])
        for seed, result in zip(audit.SPLIT_SEEDS, results):
            label = f"{backbone}/{arm}/{seed}"
            if result.get("seed") != seed or result.get("n_classes") != expected_classes:
                raise ValueError(f"{label}: seed/class contract failed")
            if set(result.get("split_counts", {})) != {
                "inner",
                "selection",
                "refit_support",
                "test",
            } or not all(int(value) > 0 for value in result["split_counts"].values()):
                raise ValueError(f"{label}: split-count contract failed")
            if set(result.get("selection", {})) != {"wcos", "lda"}:
                raise ValueError(f"{label}: selection family set mismatch")
            if set(result.get("refit_actual_regularizers", {})) != {"wcos", "lda"}:
                raise ValueError(f"{label}: refit regularizer set mismatch")
            accuracies = result.get("test_accuracies", {})
            if set(accuracies) != {"cos", "wcos", "lda"} or not all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 1.0
                for value in accuracies.values()
            ):
                raise ValueError(f"{label}: accuracy contract failed")
            if set(result.get("prediction_sha256", {})) != {"cos", "wcos", "lda"}:
                raise ValueError(f"{label}: prediction hash set mismatch")
            for method, digest in result["prediction_sha256"].items():
                require_sha256(digest, f"{label}/{method}")
            for family in ("wcos", "lda"):
                require_gap_stat(result["fixed_evaluator_gaps"][family], f"{label}/{family}")

    contrasts = payload.get("paired_shared100_scale_contrasts")
    if not isinstance(contrasts, list) or len(contrasts) != len(audit.SPLIT_SEEDS):
        raise ValueError(f"{backbone}: contrast seed count mismatch")
    for seed, row in zip(audit.SPLIT_SEEDS, contrasts):
        if row.get("seed") != seed:
            raise ValueError(f"{backbone}/{seed}: contrast seed mismatch")
        for family in ("wcos", "lda"):
            require_gap_stat(
                row["fixed_evaluator_contrasts"][family],
                f"{backbone}/{seed}/shared100/{family}",
            )
        subgroup_contrasts = row.get("paired_subgroup_global_contrasts", {})
        if set(subgroup_contrasts) != {"dogs", "birds", "vehicles"}:
            raise ValueError(f"{backbone}/{seed}: subgroup contrast set mismatch")
        for subgroup, stats in subgroup_contrasts.items():
            for family in ("wcos", "lda"):
                require_gap_stat(
                    stats[family],
                    f"{backbone}/{seed}/{subgroup}-global/{family}",
                )


def stable_direction(stats: list[dict]) -> str:
    if len(stats) != len(audit.SPLIT_SEEDS):
        raise ValueError("stable direction requires every pre-registered seed")
    if all(float(stat["ci95_pp"][0]) > 0.0 for stat in stats):
        return "positive"
    if all(float(stat["ci95_pp"][1]) < 0.0 for stat in stats):
        return "negative"
    return "boundary"


def summarize_stats(stats: list[dict]) -> dict:
    points = [float(stat["point_pp"]) for stat in stats]
    return {
        "mean_point_pp": float(np.mean(points)),
        "seed_point_range_pp": [float(min(points)), float(max(points))],
        "stable_direction": stable_direction(stats),
        "per_seed": [
            {
                "seed": int(seed),
                "point_pp": point,
                "ci95_pp": [float(value) for value in stat["ci95_pp"]],
            }
            for seed, point, stat in zip(audit.SPLIT_SEEDS, points, stats)
        ],
    }


def cross_backbone_summary(rows: list[dict]) -> dict:
    if len(rows) != len(audit.BACKBONES):
        raise ValueError("cross-backbone decision requires every registered backbone")
    counts = {
        direction: sum(
            row["stable_direction"] == direction
            for row in rows
        )
        for direction in ("positive", "negative", "boundary")
    }
    if counts["positive"] >= 4 and counts["negative"] == 0:
        pattern = "positive"
    elif counts["negative"] >= 4 and counts["positive"] == 0:
        pattern = "negative"
    else:
        pattern = "representation-dependent"
    return {"counts": counts, "pattern": pattern}


def build_evidence_summary(payloads: list[dict]) -> dict:
    rows = []
    for payload in payloads:
        arms = {}
        for arm in audit.ARM_ORDER:
            arms[arm] = {}
            for family in ("wcos", "lda"):
                stats = [
                    result["fixed_evaluator_gaps"][family]
                    for result in payload["arms"][arm]
                ]
                arms[arm][family] = summarize_stats(stats)

        contrasts = {"shared100_scale": {}}
        for family in ("wcos", "lda"):
            stats = [
                result["fixed_evaluator_contrasts"][family]
                for result in payload["paired_shared100_scale_contrasts"]
            ]
            contrasts["shared100_scale"][family] = summarize_stats(stats)
        for subgroup in ("dogs", "birds", "vehicles"):
            key = f"{subgroup}_minus_global"
            contrasts[key] = {}
            for family in ("wcos", "lda"):
                stats = [
                    result["paired_subgroup_global_contrasts"][subgroup][family]
                    for result in payload["paired_shared100_scale_contrasts"]
                ]
                contrasts[key][family] = summarize_stats(stats)

        rows.append(
            {
                "backbone": payload["backbone"],
                "backbone_label": payload["backbone_label"],
                "arms": arms,
                "contrasts": contrasts,
            }
        )

    cross_backbone = {"arms": {}, "contrasts": {}}
    for arm in audit.ARM_ORDER:
        cross_backbone["arms"][arm] = {
            family: cross_backbone_summary(
                [row["arms"][arm][family] for row in rows]
            )
            for family in ("wcos", "lda")
        }
    for contrast in (
        "shared100_scale",
        "dogs_minus_global",
        "birds_minus_global",
        "vehicles_minus_global",
    ):
        cross_backbone["contrasts"][contrast] = {
            family: cross_backbone_summary(
                [row["contrasts"][contrast][family] for row in rows]
            )
            for family in ("wcos", "lda")
        }

    return {
        "schema_version": 1,
        "kind": "sdg_e13_rebuttal_evidence_summary",
        "source_repo": audit.SOURCE_REPO,
        "source_revision": audit.SOURCE_REVISION,
        "split_seeds": list(audit.SPLIT_SEEDS),
        "direction_rule": {
            "stable_positive": "all three class-bootstrap CI lower bounds > 0",
            "stable_negative": "all three class-bootstrap CI upper bounds < 0",
            "cross_backbone": (
                "at least 4/6 stable in one direction and no stable opposite"
            ),
        },
        "rows": rows,
        "cross_backbone": cross_backbone,
    }


def evidence_report(summary: dict) -> str:
    def cell(stat: dict) -> str:
        lo, hi = stat["seed_point_range_pp"]
        direction = {
            "positive": "P",
            "negative": "N",
            "boundary": "B",
        }[stat["stable_direction"]]
        return f"{stat['mean_point_pp']:+.2f} [{lo:+.2f},{hi:+.2f}] {direction}"

    lines = [
        "# ImageNet-1K multi-backbone evidence summary",
        "",
        "Every cell is three-seed mean pp [seed point range] and stable direction.",
        "The range is not a confidence interval. P/N require all three class-bootstrap",
        "CIs to exclude zero in the same direction; B denotes mixed/boundary.",
        "W/L denote WCos−Cos and LDA−Cos. Exact-100 is a within-source anchor,",
        "not a numeric replication of the frozen Table 1 because the source is a",
        "fixed resized public derivative.",
        "",
        "| backbone | exact-100 W/L | global-1K W/L | dogs W/L | birds W/L | vehicles W/L |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        cells = []
        for arm in audit.ARM_ORDER:
            cells.append(
                f"{cell(row['arms'][arm]['wcos'])} / "
                f"{cell(row['arms'][arm]['lda'])}"
            )
        lines.append(f"| {row['backbone_label']} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Paired contrasts",
            "",
            "| backbone | shared-100 scale DID W/L | dogs-global W/L | birds-global W/L | vehicles-global W/L |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        cells = []
        for contrast in (
            "shared100_scale",
            "dogs_minus_global",
            "birds_minus_global",
            "vehicles_minus_global",
        ):
            cells.append(
                f"{cell(row['contrasts'][contrast]['wcos'])} / "
                f"{cell(row['contrasts'][contrast]['lda'])}"
            )
        lines.append(f"| {row['backbone_label']} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Cross-backbone decisions",
            "",
            "| estimand | WCos P/N/B → decision | LDA P/N/B → decision |",
            "|---|---:|---:|",
        ]
    )
    estimands = [
        ("exact-100 gap", summary["cross_backbone"]["arms"]["exact_imagenet100"]),
        ("global-1K gap", summary["cross_backbone"]["arms"]["global_1k"]),
        ("dogs gap", summary["cross_backbone"]["arms"]["dogs"]),
        ("birds gap", summary["cross_backbone"]["arms"]["birds"]),
        ("vehicles gap", summary["cross_backbone"]["arms"]["vehicles"]),
        (
            "shared-100 scale DID",
            summary["cross_backbone"]["contrasts"]["shared100_scale"],
        ),
        (
            "dogs-global paired contrast",
            summary["cross_backbone"]["contrasts"]["dogs_minus_global"],
        ),
        (
            "birds-global paired contrast",
            summary["cross_backbone"]["contrasts"]["birds_minus_global"],
        ),
        (
            "vehicles-global paired contrast",
            summary["cross_backbone"]["contrasts"]["vehicles_minus_global"],
        ),
    ]
    for label, families in estimands:
        cells = []
        for family in ("wcos", "lda"):
            value = families[family]
            counts = value["counts"]
            cells.append(
                f"{counts['positive']}/{counts['negative']}/"
                f"{counts['boundary']} → {value['pattern']}"
            )
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines) + "\n"


def verify_output(
    output_root: Path,
    expected_commit: str,
    *,
    rehash_features: bool,
) -> dict:
    require_success(output_root / "_SUCCESS")
    source_gate = read_json(output_root / "_SOURCE_VERIFIED.json")
    if (
        source_gate.get("kind") != "sdg_imagenet1k_source_verification"
        or source_gate.get("repo_id") != audit.SOURCE_REPO
        or source_gate.get("revision") != audit.SOURCE_REVISION
        or source_gate.get("ready_manifest_sha256")
        != EXPECTED_READY_MANIFEST_SHA256
        or len(source_gate.get("files", [])) != audit.EXPECTED_SOURCE_FILES
        or sum(int(row["size"]) for row in source_gate["files"])
        != audit.EXPECTED_SOURCE_BYTES
    ):
        raise ValueError("source verification gate mismatch")

    payloads = []
    feature_summary = {}
    result_sha256 = {}
    for backbone in audit.BACKBONES:
        backbone_root = output_root / backbone
        require_success(backbone_root / "_SUCCESS")
        result_path = backbone_root / "result.json"
        payload = read_json(result_path)
        verify_result_payload(payload, backbone, expected_commit)
        if payload.get("source_verification") != source_gate:
            raise ValueError(f"{backbone}: source verification payload mismatch")
        feature_summary[backbone] = verify_feature_cache(
            output_root,
            backbone,
            payload,
            rehash=rehash_features,
        )
        result_sha256[backbone] = audit.sha256_file(result_path)
        payloads.append(payload)

    aggregate_path = output_root / "aggregate.json"
    aggregate = read_json(aggregate_path)
    expected_aggregate = audit.summarize_backbones(payloads)
    if aggregate != expected_aggregate:
        raise ValueError("aggregate.json does not equal independent recomputation")
    expected_report = audit.aggregate_report(expected_aggregate)
    if (output_root / "aggregate.md").read_text(encoding="utf-8") != expected_report:
        raise ValueError("aggregate.md does not equal independent recomputation")

    return {
        "schema_version": 1,
        "kind": "sdg_e13_artifact_verification",
        "artifact_gate": "PASS",
        "output_root": str(output_root),
        "expected_commit": expected_commit,
        "source_verification_sha256": audit.sha256_file(
            output_root / "_SOURCE_VERIFIED.json"
        ),
        "result_sha256": result_sha256,
        "aggregate_sha256": audit.sha256_file(aggregate_path),
        "aggregate_report_sha256": audit.sha256_file(output_root / "aggregate.md"),
        "feature_rehash_verified": bool(rehash_features),
        "feature_summary": feature_summary,
    }


def main() -> None:
    args = parse_args()
    if (args.evidence_json_output is None) != (args.evidence_md_output is None):
        raise ValueError("evidence JSON and Markdown outputs must be requested together")
    summary = verify_output(
        args.output_root,
        args.expected_commit,
        rehash_features=not args.skip_feature_rehash,
    )
    if args.evidence_json_output is not None:
        payloads = [
            read_json(args.output_root / backbone / "result.json")
            for backbone in audit.BACKBONES
        ]
        evidence = build_evidence_summary(payloads)
        args.evidence_json_output.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_json_output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        args.evidence_md_output.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_md_output.write_text(evidence_report(evidence), encoding="utf-8")
        summary["evidence_json_sha256"] = audit.sha256_file(
            args.evidence_json_output
        )
        summary["evidence_report_sha256"] = audit.sha256_file(
            args.evidence_md_output
        )
    summary["verifier_commit"] = args.verifier_commit
    text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.audit_output is not None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
