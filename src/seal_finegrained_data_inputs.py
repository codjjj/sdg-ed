#!/usr/bin/env python3
"""Seal exact data roots for the fine-grained unseen-dataset audit.

Binds each prepared dataset root (READY.json + manifest) and the code manifest
into one hash-identified bundle, so feature extraction and evaluation can
verify that they read exactly the frozen data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath

from sdg_finegrained_manifest import (
    DATASETS,
    canonical_json_bytes,
    load_and_verify_manifest,
    sha256_bytes,
    sha256_file,
)


EXPECTED_CONTRACTS = {
    "stanford_dogs": {"n_classes": 120, "splits": {"train": 12_000, "test": 8_580}},
    "nabirds": {"n_classes": 555, "splits": {"train": 23_929, "test": 24_633}},
    "inat2021_mini": {
        "n_classes": 10_000,
        "splits": {"train": 500_000, "test": 100_000},
    },
    "foodx251": {"n_classes": 251, "splits": {"train": 118_475, "test": 11_994}},
}


def ensure_safe(path: Path, label: str) -> None:
    resolved = Path(path).resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError(f"{label} must not be a filesystem root: {resolved}")


def write_new(path: Path, payload: bytes) -> None:
    ensure_safe(path, "output")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite data-input bundle: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def parse_bindings(values: list[str]) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"dataset binding must be DATASET=ROOT: {value}")
        dataset, root = value.split("=", 1)
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset binding: {dataset}")
        if dataset in bindings:
            raise ValueError(f"duplicate dataset binding: {dataset}")
        bindings[dataset] = Path(root)
    if set(bindings) != set(DATASETS):
        raise ValueError(
            f"data-input bundle requires exact datasets {sorted(DATASETS)}; "
            f"observed={sorted(bindings)}"
        )
    return bindings


def parse_dataset_binding(values: list[str]) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"dataset binding must be DATASET=ROOT: {value}")
        dataset, root = value.split("=", 1)
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset binding: {dataset}")
        if dataset in bindings:
            raise ValueError(f"duplicate dataset binding: {dataset}")
        bindings[dataset] = Path(root)
    if len(bindings) != 1:
        raise ValueError("dataset-scoped input bundle requires exactly one dataset")
    return bindings


def verify_dataset_root(dataset: str, root: Path) -> dict:
    ensure_safe(root, f"{dataset} root")
    ready_path = root / "READY.json"
    manifest_path = root / "manifest/dataset_manifest.json"
    if not ready_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"dataset readiness artifacts are missing: {root}")
    if (root / "INCOMPLETE.json").exists():
        raise ValueError(f"dataset root remains incomplete: {root}")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    manifest = load_and_verify_manifest(manifest_path)
    contract = EXPECTED_CONTRACTS[dataset]
    if ready.get("schema_version") != 1 or ready.get("kind") != "sdg_caev_finegrained_dataset_ready":
        raise ValueError(f"invalid READY schema for {dataset}")
    if ready.get("dataset") != dataset or manifest.get("dataset") != dataset:
        raise ValueError(f"dataset identity mismatch for {dataset}")
    if ready.get("fallback_used") is not False:
        raise ValueError(f"fallback dataset is not allowed for {dataset}")
    if ready.get("n_classes") != contract["n_classes"] or manifest.get("n_classes") != contract["n_classes"]:
        raise ValueError(f"class-count contract mismatch for {dataset}")
    if ready.get("splits") != contract["splits"] or manifest.get("splits") != contract["splits"]:
        raise ValueError(f"split-count contract mismatch for {dataset}")
    manifest_sha = sha256_file(manifest_path)
    if ready.get("dataset_manifest_sha256") != manifest_sha:
        raise ValueError(f"READY/manifest hash mismatch for {dataset}")
    if ready.get("dataset_identity_sha256") != manifest.get("dataset_identity_sha256"):
        raise ValueError(f"READY/manifest identity mismatch for {dataset}")
    if dataset == "inat2021_mini":
        split_definition = manifest.get("split_definition", {})
        if split_definition.get("n_species") != 10_000:
            raise ValueError("iNaturalist bundle must contain all 10000 species")
        if split_definition.get("class_subset", "missing") is not None:
            raise ValueError("iNaturalist class subset is forbidden")

    archive_entries = []
    for relative, expected in sorted(manifest.get("archive_sha256", {}).items()):
        normalized = PurePosixPath(relative)
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError(f"unsafe archive path in {dataset}: {relative}")
        archive_path = root / normalized
        actual = sha256_file(archive_path)
        if actual != expected:
            raise ValueError(f"archive hash mismatch for {dataset}: {relative}")
        archive_entries.append(
            {
                "path": relative,
                "sha256": actual,
                "bytes": archive_path.stat().st_size,
            }
        )
    if not archive_entries:
        raise ValueError(f"dataset has no bound archives: {dataset}")
    return {
        "dataset": dataset,
        "root": str(root),
        "ready_path": str(ready_path),
        "ready_sha256": sha256_file(ready_path),
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": manifest_sha,
        "dataset_identity_sha256": manifest["dataset_identity_sha256"],
        "n_classes": contract["n_classes"],
        "splits": contract["splits"],
        "archives": archive_entries,
    }


def build_data_input_bundle(bindings: dict[str, Path], commit: str) -> dict:
    if set(bindings) != set(DATASETS):
        raise ValueError("data-input bindings must cover the exact four datasets")
    entries = [verify_dataset_root(dataset, bindings[dataset]) for dataset in DATASETS]
    roots = [entry["root"] for entry in entries]
    if len(set(roots)) != len(roots):
        raise ValueError("dataset roots must be distinct")
    payload = {
        "schema_version": 1,
        "kind": "sdg_caev_finegrained_data_input_bundle",
        "commit": commit,
        "datasets": entries,
        "dataset_order": list(DATASETS),
        "inat2021_contract": {
            "all_10000_species": True,
            "class_subset": None,
            "train_per_class": 50,
            "validation_per_class": 10,
        },
    }
    payload["bundle_identity_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def build_dataset_input_bundle(bindings: dict[str, Path], commit: str) -> dict:
    if len(bindings) != 1:
        raise ValueError("dataset-scoped input bundle requires exactly one dataset")
    dataset = next(iter(bindings))
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset binding: {dataset}")
    payload = {
        "schema_version": 1,
        "kind": "sdg_caev_finegrained_dataset_input_bundle",
        "commit": commit,
        "datasets": [verify_dataset_root(dataset, bindings[dataset])],
        "dataset_order": [dataset],
        "scope": "single_dataset",
    }
    if dataset == "inat2021_mini":
        payload["inat2021_contract"] = {
            "all_10000_species": True,
            "class_subset": None,
            "train_per_class": 50,
            "validation_per_class": 10,
        }
    payload["bundle_identity_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def load_and_verify_data_input_bundle(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = payload.pop("bundle_identity_sha256", None)
    actual = sha256_bytes(canonical_json_bytes(payload))
    payload["bundle_identity_sha256"] = expected
    if expected != actual:
        raise ValueError("data-input bundle identity hash mismatch")
    if payload.get("schema_version") != 1 or payload.get("kind") not in {
        "sdg_caev_finegrained_data_input_bundle",
        "sdg_caev_finegrained_dataset_input_bundle",
    }:
        raise ValueError("unsupported data-input bundle schema")
    entries = payload.get("datasets")
    observed = [entry.get("dataset") for entry in entries] if isinstance(entries, list) else []
    if payload["kind"] == "sdg_caev_finegrained_data_input_bundle":
        if observed != list(DATASETS) or payload.get("dataset_order") != list(DATASETS):
            raise ValueError("joint data-input bundle dataset order mismatch")
        if payload.get("inat2021_contract") != {
            "all_10000_species": True,
            "class_subset": None,
            "train_per_class": 50,
            "validation_per_class": 10,
        }:
            raise ValueError("data-input bundle iNaturalist contract mismatch")
    else:
        if len(observed) != 1 or observed[0] not in DATASETS:
            raise ValueError("dataset-scoped input bundle must bind exactly one dataset")
        if payload.get("dataset_order") != observed or payload.get("scope") != "single_dataset":
            raise ValueError("dataset-scoped input bundle identity mismatch")
        if observed[0] == "inat2021_mini" and payload.get("inat2021_contract") != {
            "all_10000_species": True,
            "class_subset": None,
            "train_per_class": 50,
            "validation_per_class": 10,
        }:
            raise ValueError("dataset-scoped iNaturalist contract mismatch")
    return payload


def verify_dataset_binding(bundle_path: Path, dataset_root: Path) -> dict:
    bundle = load_and_verify_data_input_bundle(bundle_path)
    root_text = str(dataset_root)
    matches = [entry for entry in bundle["datasets"] if entry["root"] == root_text]
    if len(matches) != 1:
        raise ValueError(f"dataset root is not uniquely bound: {dataset_root}")
    entry = matches[0]
    ready_path = dataset_root / "READY.json"
    manifest_path = dataset_root / "manifest/dataset_manifest.json"
    if sha256_file(ready_path) != entry["ready_sha256"]:
        raise ValueError("bound READY hash changed")
    if sha256_file(manifest_path) != entry["dataset_manifest_sha256"]:
        raise ValueError("bound dataset manifest hash changed")
    return {
        **entry,
        "bundle_identity_sha256": bundle["bundle_identity_sha256"],
    }


def verify_code_manifest(path: Path, commit: str) -> str:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("git_commit") != commit:
        raise ValueError("code manifest identity mismatch")
    package_root = Path(__file__).resolve().parent.parent
    for relative, expected in sorted(manifest.get("files", {}).items()):
        candidate = package_root / relative
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"code bundle mismatch: {relative}")
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--scope", choices=("joint", "dataset"), default="joint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--commit", default=None)
    args = parser.parse_args()
    ensure_safe(args.output, "output")
    code_manifest_sha = verify_code_manifest(args.code_manifest, args.commit)
    if args.scope == "joint":
        payload = build_data_input_bundle(parse_bindings(args.dataset_root), args.commit)
    else:
        payload = build_dataset_input_bundle(parse_dataset_binding(args.dataset_root), args.commit)
    payload["code_manifest"] = str(args.code_manifest)
    payload["code_manifest_sha256"] = code_manifest_sha
    payload["bundle_identity_sha256"] = sha256_bytes(
        canonical_json_bytes({key: value for key, value in payload.items() if key != "bundle_identity_sha256"})
    )
    write_new(args.output, canonical_json_bytes(payload))
    write_new(args.output.with_suffix(args.output.suffix + ".sha256"), (sha256_file(args.output) + "\n").encode())
    print(f"[data-input-ready] {args.output}", flush=True)


if __name__ == "__main__":
    main()
