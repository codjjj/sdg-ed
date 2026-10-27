#!/usr/bin/env python3
"""Pre-result identity manifest for the fine-grained unseen datasets.

Part of the expanded router evaluation (Appendix C).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


DATASETS = ("stanford_dogs", "nabirds", "inat2021_mini", "foodx251")
NONFULL_SHOTS = ("1", "2", "5", "10", "20")
SHOTS = (*NONFULL_SHOTS, "full")
DATASET_NONFULL_SHOTS = {
    "stanford_dogs": NONFULL_SHOTS,
    "nabirds": NONFULL_SHOTS,
    "inat2021_mini": NONFULL_SHOTS,
    "foodx251": NONFULL_SHOTS,
}
NONFULL_SEEDS = (0, 1, 2, 3, 4)
SPLIT_ORDER = {"train": 0, "test": 1}
SCHEMA_VERSION = 1


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _source_sort_key(value: object) -> tuple[int, object]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def canonicalize_image_rows(raw_rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Assign contiguous labels and deterministic per-split row indices."""

    raw_rows = [dict(row) for row in raw_rows]
    if not raw_rows:
        raise ValueError("image manifest cannot be empty")
    required = {
        "split",
        "source_class_id",
        "class_name",
        "source_image_id",
        "archive_file",
        "archive_member",
    }
    for index, row in enumerate(raw_rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"image row {index} missing fields: {missing}")
        if row["split"] not in SPLIT_ORDER:
            raise ValueError(f"invalid split in row {index}: {row['split']}")
        for field in required - {"split"}:
            row[field] = str(row[field])
            if not row[field]:
                raise ValueError(f"empty {field} in image row {index}")

    names_by_source: dict[str, str] = {}
    for row in raw_rows:
        source_id = row["source_class_id"]
        previous = names_by_source.setdefault(source_id, row["class_name"])
        if previous != row["class_name"]:
            raise ValueError(f"inconsistent class name for source id {source_id}")
    source_ids = sorted(names_by_source, key=_source_sort_key)
    labels = {source_id: index for index, source_id in enumerate(source_ids)}
    classes = [
        {
            "label": labels[source_id],
            "source_class_id": source_id,
            "class_name": names_by_source[source_id],
        }
        for source_id in source_ids
    ]

    seen_image_ids: set[tuple[str, str]] = set()
    seen_members: set[tuple[str, str, str]] = set()
    for row in raw_rows:
        image_key = (row["split"], row["source_image_id"])
        member_key = (row["split"], row["archive_file"], row["archive_member"])
        if image_key in seen_image_ids:
            raise ValueError(f"duplicate source image id: {image_key}")
        if member_key in seen_members:
            raise ValueError(f"duplicate archive member: {member_key}")
        seen_image_ids.add(image_key)
        seen_members.add(member_key)
        row["label"] = labels[row["source_class_id"]]

    raw_rows.sort(
        key=lambda row: (
            SPLIT_ORDER[row["split"]],
            row["label"],
            row["source_image_id"],
            row["archive_file"],
            row["archive_member"],
        )
    )
    split_counters = Counter()
    for row in raw_rows:
        row["split_index"] = int(split_counters[row["split"]])
        split_counters[row["split"]] += 1

    counts = Counter((row["label"], row["split"]) for row in raw_rows)
    for cls in classes:
        label = cls["label"]
        cls["train_count"] = int(counts[(label, "train")])
        cls["test_count"] = int(counts[(label, "test")])
        if cls["train_count"] < 1:
            raise ValueError(f"class {cls['source_class_id']} has no train images")
        if cls["test_count"] < 1:
            raise ValueError(f"class {cls['source_class_id']} has no test images")
    return classes, raw_rows


def support_indices_for_labels(labels: np.ndarray, shot: str, seed: int) -> np.ndarray:
    """Return the exact indices used by ``spectral_decision_gap_ed.sample_shot``."""

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if str(shot) == "full":
        if int(seed) != 0:
            raise ValueError("full-shot support identity is frozen to seed 0")
        return np.arange(len(labels), dtype=np.int64)
    if str(shot) not in NONFULL_SHOTS:
        raise ValueError(f"unsupported shot: {shot}")
    if type(seed) is not int or seed not in NONFULL_SEEDS:
        raise ValueError(f"unsupported non-full seed: {seed}")
    k = int(shot)
    rng = np.random.RandomState(seed)
    indices: list[int] = []
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    unique, starts, counts = np.unique(
        sorted_labels, return_index=True, return_counts=True
    )
    for label, start, count in zip(unique.tolist(), starts.tolist(), counts.tolist()):
        where = order[start : start + count].copy()
        if len(where) < k:
            raise ValueError(f"label {label} has only {len(where)} rows for {k}-shot")
        rng.shuffle(where)
        indices.extend(where[:k].tolist())
    return np.asarray(sorted(indices), dtype=np.int64)


def support_index_grid(
    labels: np.ndarray,
    nonfull_shots: tuple[str, ...] = NONFULL_SHOTS,
    *,
    capped_per_class: bool = False,
) -> dict[tuple[str, int], np.ndarray]:
    """Build all frozen supports while shuffling each class once per seed."""

    labels = np.asarray(labels)
    if not nonfull_shots or any(shot not in NONFULL_SHOTS for shot in nonfull_shots):
        raise ValueError(f"invalid non-full shot contract: {nonfull_shots}")
    if tuple(shot for shot in NONFULL_SHOTS if shot in nonfull_shots) != nonfull_shots:
        raise ValueError(f"non-full shots must preserve frozen order: {nonfull_shots}")
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    unique, starts, counts = np.unique(
        sorted_labels, return_index=True, return_counts=True
    )
    groups = [order[start : start + count].copy() for start, count in zip(starts, counts)]
    largest_shot = max(int(shot) for shot in nonfull_shots)
    if not capped_per_class and any(len(group) < largest_shot for group in groups):
        raise ValueError(f"true {largest_shot}-shot is unavailable")
    grid: dict[tuple[str, int], np.ndarray] = {
        ("full", 0): np.arange(len(labels), dtype=np.int64)
    }
    for seed in NONFULL_SEEDS:
        rng = np.random.RandomState(seed)
        shuffled = []
        for group in groups:
            current = group.copy()
            rng.shuffle(current)
            shuffled.append(current)
        for shot in nonfull_shots:
            k = int(shot)
            grid[(shot, seed)] = np.asarray(
                sorted(
                    index
                    for group in shuffled
                    for index in group[: min(k, len(group))]
                ),
                dtype=np.int64,
            )
    return grid


def validate_inat2021_mini_counts(classes: list[dict]) -> None:
    if len(classes) != 10_000:
        raise ValueError(f"iNaturalist mini must contain exactly 10000 classes, got {len(classes)}")
    bad = [
        cls
        for cls in classes
        if int(cls["train_count"]) != 50 or int(cls["test_count"]) != 10
    ]
    if bad:
        first = bad[0]
        raise ValueError(
            "iNaturalist mini class count mismatch: "
            f"source_id={first['source_class_id']} train={first['train_count']} "
            f"validation={first['test_count']}"
        )


def _npy_bytes(array: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.int64), allow_pickle=False)
    return buffer.getvalue()


def write_manifest_bundle(
    output_dir: Path,
    *,
    dataset: str,
    raw_rows: Iterable[dict],
    source: dict,
    split_definition: dict,
    archive_hashes: dict[str, str],
) -> dict:
    """Create a fail-closed dataset/class/image/support identity bundle."""

    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"manifest output must be a new directory: {output_dir}")
    output_dir.mkdir(parents=True)

    classes, rows = canonicalize_image_rows(raw_rows)
    if dataset == "inat2021_mini":
        validate_inat2021_mini_counts(classes)

    classes_path = output_dir / "classes.json"
    images_path = output_dir / "images.jsonl"
    _write_new(classes_path, canonical_json_bytes(classes))
    image_bytes = b"".join(canonical_json_bytes(row) for row in rows)
    _write_new(images_path, image_bytes)

    train_rows = [row for row in rows if row["split"] == "train"]
    train_labels = np.asarray([row["label"] for row in train_rows], dtype=np.int64)
    supports = []
    nonfull_shots = DATASET_NONFULL_SHOTS[dataset]
    support_pairs = [(shot, seed) for shot in nonfull_shots for seed in NONFULL_SEEDS]
    support_pairs.append(("full", 0))
    capped_per_class = dataset == "nabirds"
    support_grid = support_index_grid(
        train_labels,
        nonfull_shots,
        capped_per_class=capped_per_class,
    )
    for shot, seed in support_pairs:
        indices = support_grid[(shot, seed)]
        support_counts = np.bincount(train_labels[indices], minlength=len(classes))
        relative = Path("supports") / f"shot_{shot}_seed_{seed}.npy"
        payload = _npy_bytes(indices)
        _write_new(output_dir / relative, payload)
        supports.append(
            {
                "shot": shot,
                "seed": seed,
                "path": relative.as_posix(),
                "sha256": sha256_bytes(payload),
                "n_indices": int(len(indices)),
                "class_support_count_min": int(support_counts.min()),
                "class_support_count_max": int(support_counts.max()),
                "classes_below_requested_k": (
                    int(np.sum(support_counts < int(shot))) if shot != "full" else 0
                ),
            }
        )

    identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": "sdg_caev_finegrained_pre_result_manifest",
        "dataset": dataset,
        "source": source,
        "split_definition": split_definition,
        "archive_sha256": dict(sorted(archive_hashes.items())),
        "n_classes": len(classes),
        "splits": {
            "train": len(train_rows),
            "test": sum(row["split"] == "test" for row in rows),
        },
        "classes": {
            "path": classes_path.name,
            "sha256": sha256_file(classes_path),
            "n_rows": len(classes),
        },
        "images": {
            "path": images_path.name,
            "sha256": sha256_file(images_path),
            "n_rows": len(rows),
        },
        "supports": supports,
        "support_contract": {
            "implementation": (
                "sdg_caev_finegrained_manifest.support_index_grid(capped_per_class=True)"
                if capped_per_class
                else "spectral_decision_gap_ed.sample_shot"
            ),
            "nonfull_shots": list(nonfull_shots),
            "sampling_rule": (
                "per class, sample min(train_count, requested_k) without replacement"
                if capped_per_class
                else "per class, sample exactly requested_k without replacement"
            ),
            "capped_per_class": capped_per_class,
            "unavailable_shots": {},
            "nonfull_seeds": list(NONFULL_SEEDS),
            "full_seed": 0,
            "class_iteration": "ascending contiguous label",
            "within_class_rng": "numpy.random.RandomState(seed).shuffle",
            "output_order": "ascending train split index",
        },
    }
    identity["dataset_identity_sha256"] = sha256_bytes(canonical_json_bytes(identity))
    manifest_path = output_dir / "dataset_manifest.json"
    _write_new(manifest_path, canonical_json_bytes(identity))
    manifest_sha = sha256_file(manifest_path)
    _write_new(output_dir / "dataset_manifest.sha256", (manifest_sha + "\n").encode())
    return identity


def load_and_verify_manifest(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text())
    expected_identity = payload.pop("dataset_identity_sha256")
    actual_identity = sha256_bytes(canonical_json_bytes(payload))
    payload["dataset_identity_sha256"] = expected_identity
    if actual_identity != expected_identity:
        raise ValueError("dataset identity hash mismatch")
    root = manifest_path.parent
    for key in ("classes", "images"):
        path = root / payload[key]["path"]
        if sha256_file(path) != payload[key]["sha256"]:
            raise ValueError(f"{key} file hash mismatch")
    for support in payload["supports"]:
        path = root / support["path"]
        if sha256_file(path) != support["sha256"]:
            raise ValueError(f"support file hash mismatch: {path}")
        indices = np.load(path, allow_pickle=False)
        if indices.dtype != np.int64 or indices.ndim != 1:
            raise ValueError(f"invalid support index array: {path}")
        if len(indices) != int(support["n_indices"]):
            raise ValueError(f"support length mismatch: {path}")
    return payload


def load_image_rows(manifest_path: Path) -> list[dict]:
    payload = load_and_verify_manifest(manifest_path)
    path = Path(manifest_path).parent / payload["images"]["path"]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != int(payload["images"]["n_rows"]):
        raise ValueError("image manifest row count mismatch")
    return rows
