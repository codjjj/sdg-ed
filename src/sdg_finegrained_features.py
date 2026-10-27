#!/usr/bin/env python3
"""Manifest-driven DINOv2-B feature extraction for the fine-grained unseen datasets.

Extracts the final-layer CLS token of DINOv2 ViT-B/14 with registers (the core
suite's feature) for every image listed in a frozen dataset manifest written
by prepare_finegrained_heldout_data.py. `--model-path` must be a local
snapshot of facebook/dinov2-with-registers-base (its file tree is hashed into
the feature manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sdg_finegrained_manifest import (
    canonical_json_bytes,
    load_and_verify_manifest,
    load_image_rows,
    sha256_file,
)
from sdg_paths import MODEL_ROOT
from seal_finegrained_data_inputs import verify_dataset_binding


def model_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"model tree is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def make_transform():
    import torchvision.transforms as transforms

    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def cls_features(model_output) -> torch.Tensor:
    hidden = model_output.last_hidden_state
    if hidden.ndim != 3 or hidden.shape[1] < 1:
        raise ValueError(f"unexpected DINOv2 hidden-state shape: {tuple(hidden.shape)}")
    return hidden[:, 0, :]


def copy_and_verify_archives(dataset_root: Path, manifest: dict, scratch: Path) -> dict[str, Path]:
    copied = {}
    for relative, expected_hash in manifest["archive_sha256"].items():
        source = dataset_root / relative
        if sha256_file(source) != expected_hash:
            raise ValueError(f"source archive hash mismatch: {source}")
        destination = scratch / "archives" / Path(relative).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != expected_hash:
            raise ValueError(f"scratch archive hash mismatch: {destination}")
        copied[relative] = destination
    return copied


def safe_extract(archive: Path, destination: Path, required_members: set[str]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    destination_resolved = destination.resolve()
    found = set()
    with tarfile.open(archive, "r:*") as handle:
        for member in handle:
            normalized = member.name.lstrip("./")
            if normalized not in required_members:
                continue
            if not member.isfile():
                raise ValueError(f"required archive member is not a file: {member.name}")
            target = (destination / normalized).resolve()
            if destination_resolved not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot extract archive member: {member.name}")
            with target.open("wb") as output:
                shutil.copyfileobj(extracted, output, length=8 << 20)
            found.add(normalized)
    missing = sorted(required_members - found)
    if missing:
        raise ValueError(f"archive extraction missed {len(missing)} required members: {missing[:3]}")


def materialize_images(
    rows: list[dict],
    archives: dict[str, Path],
    scratch: Path,
) -> dict[tuple[str, str], Path]:
    members_by_archive: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        members_by_archive[row["archive_file"]].add(row["archive_member"])
    roots = {}
    for index, relative in enumerate(sorted(members_by_archive)):
        if relative not in archives:
            raise ValueError(f"manifest references unknown archive: {relative}")
        root = scratch / "extracted" / f"archive_{index:02d}"
        safe_extract(archives[relative], root, members_by_archive[relative])
        roots[relative] = root
    return {
        (relative, member): roots[relative] / member
        for relative, members in members_by_archive.items()
        for member in members
    }


class ManifestImageDataset(Dataset):
    def __init__(self, rows: list[dict], materialized: dict[tuple[str, str], Path], transform):
        self.rows = sorted(rows, key=lambda row: int(row["split_index"]))
        expected = list(range(len(self.rows)))
        actual = [int(row["split_index"]) for row in self.rows]
        if actual != expected:
            raise ValueError("split indices are not contiguous")
        self.materialized = materialized
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.materialized[(row["archive_file"], row["archive_member"])]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), int(row["split_index"])


def atomic_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def write_json_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def extract_split(
    split: str,
    rows: list[dict],
    materialized: dict[tuple[str, str], Path],
    model,
    transform,
    args,
    scratch: Path,
) -> tuple[Path, dict]:
    dataset = ManifestImageDataset(rows, materialized, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    feature_chunks = []
    label_chunks = []
    seen_indices = []
    for batch_index, (images, labels, indices) in enumerate(loader, start=1):
        images = images.to(args.device, non_blocking=True)
        with torch.inference_mode():
            pooled = cls_features(model(images)).detach().cpu().float()
        feature_chunks.append(pooled)
        label_chunks.append(labels.long())
        seen_indices.append(indices.long())
        if batch_index == 1 or batch_index % args.log_every == 0 or batch_index == len(loader):
            print(
                f"[features] split={split} batch={batch_index}/{len(loader)} "
                f"images={min(batch_index * args.batch_size, len(dataset))}/{len(dataset)}",
                flush=True,
            )
    features = torch.cat(feature_chunks)
    labels = torch.cat(label_chunks)
    indices = torch.cat(seen_indices)
    if not torch.equal(indices, torch.arange(len(dataset), dtype=torch.long)):
        raise AssertionError(f"{split} feature order differs from manifest")
    expected_labels = torch.tensor([int(row["label"]) for row in dataset.rows], dtype=torch.long)
    if not torch.equal(labels, expected_labels):
        raise AssertionError(f"{split} feature labels differ from manifest")
    if features.ndim != 2 or len(features) != len(dataset) or not torch.isfinite(features).all():
        raise ValueError(f"invalid {split} features: {tuple(features.shape)}")
    local_path = scratch / f"{split}.pt"
    torch.save(
        {
            "pooled": features,
            "labels": labels,
            "manifest_split_indices": indices,
        },
        local_path,
    )
    return local_path, {
        "n_rows": len(dataset),
        "feature_shape": list(features.shape),
        "labels_sha256": hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
        "split_indices_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
        "cache_sha256": sha256_file(local_path),
        "bytes": local_path.stat().st_size,
    }


def run(args) -> None:
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    model_path = Path(args.model_path)
    data_input_manifest = Path(args.data_input_manifest)
    if output_root.exists():
        raise FileExistsError(f"feature output root must be new: {output_root}")
    ready_path = dataset_root / "READY.json"
    if not ready_path.exists() or (dataset_root / "INCOMPLETE.json").exists():
        raise ValueError(f"dataset preparation is not complete: {dataset_root}")
    data_binding = verify_dataset_binding(data_input_manifest, dataset_root)
    ready = json.loads(ready_path.read_text())
    manifest_path = dataset_root / "manifest/dataset_manifest.json"
    manifest = load_and_verify_manifest(manifest_path)
    if ready["dataset_identity_sha256"] != manifest["dataset_identity_sha256"]:
        raise ValueError("READY/dataset manifest identity mismatch")
    if ready["dataset_manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("READY/dataset manifest file hash mismatch")
    rows = load_image_rows(manifest_path)

    scratch = Path(tempfile.mkdtemp(prefix=f"sdg_finegrained_features_{manifest['dataset']}_", dir=args.scratch_root))
    try:
        archives = copy_and_verify_archives(dataset_root, manifest, scratch)
        materialized = materialize_images(rows, archives, scratch)
        transform = make_transform()
        from transformers import AutoModel

        model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
        model = model.to(args.device).eval()
        split_results = {}
        local_paths = {}
        for split in ("train", "test"):
            split_rows = [row for row in rows if row["split"] == split]
            local_paths[split], split_results[split] = extract_split(
                split,
                split_rows,
                materialized,
                model,
                transform,
                args,
                scratch,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        output_root.mkdir(parents=True)
        for split in ("train", "test"):
            destination = output_root / f"{split}.pt"
            atomic_copy(local_paths[split], destination)
            if sha256_file(destination) != split_results[split]["cache_sha256"]:
                raise ValueError(f"published {split} cache hash mismatch")
        feature_manifest = {
            "schema_version": 1,
            "kind": "sdg_caev_finegrained_dinov2_feature_manifest",
            "dataset": manifest["dataset"],
            "commit": args.commit,
            "dataset_manifest": str(manifest_path),
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "dataset_identity_sha256": manifest["dataset_identity_sha256"],
            "data_input_manifest": str(data_input_manifest),
            "data_input_manifest_sha256": sha256_file(data_input_manifest),
            "data_input_bundle_identity_sha256": data_binding["bundle_identity_sha256"],
            "model_path": str(model_path),
            "model_tree_sha256": model_tree_sha256(model_path),
            "feature_token": "last_hidden_state[:,0,:]",
            "preprocessing": {
                "resize": 256,
                "interpolation": "torchvision bicubic",
                "center_crop": 224,
                "rgb": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "dtype": "torch.float32",
            "splits": split_results,
        }
        write_json_new(output_root / "feature_manifest.json", feature_manifest)
        write_json_new(
            output_root / "FEATURE_READY.json",
            {
                "kind": "sdg_caev_finegrained_feature_ready",
                "dataset": manifest["dataset"],
                "dataset_identity_sha256": manifest["dataset_identity_sha256"],
                "feature_manifest_sha256": sha256_file(output_root / "feature_manifest.json"),
                "train_cache_sha256": split_results["train"]["cache_sha256"],
                "test_cache_sha256": split_results["test"]["cache_sha256"],
            },
        )
        print(f"[feature-ready] dataset={manifest['dataset']} root={output_root}", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-input-manifest", required=True)
    parser.add_argument(
        "--model-path",
        default=str(MODEL_ROOT / "dinov2-with-registers-base"),
        help="local snapshot of facebook/dinov2-with-registers-base",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--scratch-root", default=None, help="temporary directory (default: system temp)")
    parser.add_argument("--commit", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
