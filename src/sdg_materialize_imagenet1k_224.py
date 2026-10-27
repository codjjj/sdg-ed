#!/usr/bin/env python3
"""Materialize the fixed public ImageNet-1K derivative used by the ImageNet study.

Downloads the parquet shards of the Hugging Face dataset mlnomad/imagenet-1k-224
at a pinned revision (1,281,167 train / 50,000 validation images, resized to
224px), verifies sizes and row counts against the Hub metadata, and copies the
verified tree to an empty `--target-root`. ImageNet terms of access apply.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


REPO_ID = "mlnomad/imagenet-1k-224"
REVISION = "d9e99f3b3b356abef801094154a0595729ac0818"
EXPECTED_TRAIN_SHARDS = 33
EXPECTED_VALIDATION_SHARDS = 2
EXPECTED_TRAIN_ROWS = 1_281_167
EXPECTED_VALIDATION_ROWS = 50_000

def _default_target_root() -> Path:
    from sdg_paths import DATA_ROOT

    return DATA_ROOT / "imagenet1k_224_hf_d9e99f3b"


DEFAULT_TARGET_ROOT = _default_target_root()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_files(root: Path) -> tuple[list[Path], list[Path]]:
    train = sorted((root / "data").glob("train-*.parquet"))
    validation = sorted((root / "data").glob("validation-*.parquet"))
    return train, validation


def count_rows(paths: list[Path]) -> int:
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def remote_metadata() -> dict[str, dict[str, Any]]:
    from huggingface_hub import HfApi

    info = HfApi().repo_info(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        files_metadata=True,
    )
    selected: dict[str, dict[str, Any]] = {}
    for sibling in info.siblings:
        name = sibling.rfilename
        if name not in {"README.md", ".gitattributes"} and not (
            name.startswith("data/") and name.endswith(".parquet")
        ):
            continue
        lfs = sibling.lfs or {}
        selected[name] = {
            "size": int(sibling.size),
            "sha256": lfs.get("sha256"),
        }
    return selected


def build_manifest(root: Path, remote: dict[str, dict[str, Any]]) -> dict[str, Any]:
    train, validation = parquet_files(root)
    if len(train) != EXPECTED_TRAIN_SHARDS:
        raise AssertionError(f"train shard count: {len(train)}")
    if len(validation) != EXPECTED_VALIDATION_SHARDS:
        raise AssertionError(f"validation shard count: {len(validation)}")
    train_rows = count_rows(train)
    validation_rows = count_rows(validation)
    if train_rows != EXPECTED_TRAIN_ROWS:
        raise AssertionError(f"train rows: {train_rows}")
    if validation_rows != EXPECTED_VALIDATION_ROWS:
        raise AssertionError(f"validation rows: {validation_rows}")

    expected_names = {".gitattributes", "README.md"} | {
        path.relative_to(root).as_posix() for path in train + validation
    }
    if set(remote) != expected_names:
        missing = sorted(expected_names - set(remote))
        extra = sorted(set(remote) - expected_names)
        raise AssertionError(f"remote manifest mismatch missing={missing} extra={extra}")

    files: list[dict[str, Any]] = []
    for name in sorted(expected_names):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != remote[name]["size"]:
            raise AssertionError(f"size mismatch {name}: {size} != {remote[name]['size']}")
        sha256 = sha256_file(path)
        expected_sha256 = remote[name]["sha256"]
        if expected_sha256 is not None and sha256 != expected_sha256:
            raise AssertionError(
                f"sha256 mismatch {name}: {sha256} != {expected_sha256}"
            )
        files.append({"path": name, "size": size, "sha256": sha256})

    return {
        "schema_version": 1,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "train_shards": len(train),
        "validation_shards": len(validation),
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "total_bytes": sum(row["size"] for row in files),
        "files": files,
    }


def ensure_safe_empty_target(target_root: Path) -> None:
    target = target_root.resolve()
    if target == Path(target.anchor):
        raise ValueError(f"target must not be a filesystem root: {target}")
    target.mkdir(parents=True, exist_ok=True)
    existing = [path.name for path in target.iterdir()]
    if existing:
        raise FileExistsError(f"target is not empty: {target}; entries={existing[:10]}")


def copy_verified_tree(
    staging_root: Path,
    target_root: Path,
    manifest: dict[str, Any],
) -> None:
    ensure_safe_empty_target(target_root)
    for row in manifest["files"]:
        relative = Path(row["path"])
        destination = target_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staging_root / relative, destination)
        if destination.stat().st_size != row["size"]:
            raise AssertionError(f"copied size mismatch: {relative}")

    marker = target_root / "_READY.json"
    fd, temporary = tempfile.mkstemp(prefix=".ready-", dir=target_root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, marker)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--print-spec", action="store_true")
    args = parser.parse_args()

    if args.print_spec:
        print(
            json.dumps(
                {
                    "repo_id": REPO_ID,
                    "revision": REVISION,
                    "target_root": str(args.target_root),
                    "train_shards": EXPECTED_TRAIN_SHARDS,
                    "validation_shards": EXPECTED_VALIDATION_SHARDS,
                    "train_rows": EXPECTED_TRAIN_ROWS,
                    "validation_rows": EXPECTED_VALIDATION_ROWS,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    from huggingface_hub import snapshot_download

    staging_root = Path(tempfile.mkdtemp(prefix="sdg-imagenet1k-224-"))
    try:
        print(f"[materialize] staging={staging_root}", flush=True)
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            local_dir=staging_root,
            allow_patterns=["README.md", ".gitattributes", "data/*.parquet"],
            max_workers=args.max_workers,
        )
        manifest = build_manifest(staging_root, remote_metadata())
        copy_verified_tree(staging_root, args.target_root, manifest)
        print(
            f"[materialize] complete target={args.target_root} "
            f"files={len(manifest['files'])} bytes={manifest['total_bytes']}",
            flush=True,
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    main()
