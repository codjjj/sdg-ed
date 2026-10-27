#!/usr/bin/env python3
"""Download, validate, and freeze the fine-grained unseen datasets.

Covers Stanford Dogs, FoodX-251, NABirds, and iNaturalist-2021 mini (the last
was prepared but produced no accepted result and is excluded from the paper).
Every archive is size- and hash-checked, and the official split metadata is
frozen into a manifest before any feature is extracted. This program never
produces model features or evaluation results.

NABirds is distributed by the Cornell Lab of Ornithology under
non-commercial, no-redistribution terms and has no public download URL here:
download `nabirds.tar.gz` from the official NABirds page and pass its parent
directory layout through `--source-root` (see DATA_AND_FEATURES.md).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

from sdg_finegrained_manifest import (
    canonical_json_bytes,
    sha256_file,
    write_manifest_bundle,
)


FOODX_OFFICIAL_SPLIT_COUNTS = {"train": 118_475, "test": 11_994}

SOURCES = {
    "stanford_dogs": [
        {
            "name": "images.tar",
            "url": "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
            "md5": None,
            "bytes_expected": 793_579_520,
        },
        {
            "name": "lists.tar",
            "url": "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
            "md5": None,
            "bytes_expected": 481_280,
        },
    ],
    "nabirds": [
        {
            "name": "nabirds.tar.gz",
            # Obtain from the official NABirds release; pass it via --source-root.
            "url": None,
            "md5": None,
            "bytes_expected": 9_884_285_075,
            "sha256_expected": "cd20ef17a4ae110edcacaf964c4ad4794fc762ad760f237f826dbf30040f713d",
        }
    ],
    "inat2021_mini": [
        {
            "name": "train_mini.tar.gz",
            "url": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz",
            "md5": "db6ed8330e634445efc8fec83ae81442",
            "bytes_expected": 44_636_137_542,
        },
        {
            "name": "train_mini.json.tar.gz",
            "url": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz",
            "md5": "395a35be3651d86dc3b0d365b8ea5f92",
            "bytes_expected": 46_769_466,
        },
        {
            "name": "val.tar.gz",
            "url": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz",
            "md5": "f6f6e0e242e3d4c9569ba56400938afc",
            "bytes_expected": 8_931_661_582,
        },
        {
            "name": "val.json.tar.gz",
            "url": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz",
            "md5": "4d761e0f6a86cc63e8f7afc91f6a8f0b",
            "bytes_expected": 9_797_665,
        },
    ],
    "foodx251": [
        {
            "name": "annot.tar",
            "url": "https://food-x.s3.amazonaws.com/annot.tar",
            "md5": "0c632c543ceed0e70f0eb2db58eda3ab",
            "bytes_expected": 3_123_200,
        },
        {
            "name": "train.tar",
            "url": "https://food-x.s3.amazonaws.com/train.tar",
            "md5": "8e56440e365ee852dcb0953a9307e27f",
            "bytes_expected": 2_370_519_040,
        },
        {
            "name": "val.tar",
            "url": "https://food-x.s3.amazonaws.com/val.tar",
            "md5": "fa9a4c1eb929835a0fe68734f4868d3b",
            "bytes_expected": 241_346_560,
        },
    ],
}


def hash_file(path: Path, algorithm: str, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_safe_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"dataset output root must be new: {path}")


def verify_downloaded_source(source: dict, destination: Path) -> dict:
    actual_bytes = destination.stat().st_size
    expected_bytes = source.get("bytes_expected")
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError(
            f"size mismatch for {source['name']}: expected={expected_bytes} actual={actual_bytes}"
        )
    md5 = hash_file(destination, "md5")
    if source["md5"] is not None and md5 != source["md5"]:
        raise ValueError(
            f"MD5 mismatch for {source['name']}: expected={source['md5']} actual={md5}"
        )
    sha256 = sha256_file(destination)
    expected_sha256 = source.get("sha256_expected")
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {source['name']}: expected={expected_sha256} actual={sha256}"
        )
    return {
        **source,
        "bytes": actual_bytes,
        "md5_actual": md5,
        "sha256": sha256,
    }


def download(source: dict, destination: Path) -> dict:
    if not source.get("url"):
        raise FileNotFoundError(
            f"{source['name']} has no public download URL; download it from the official "
            "dataset page and rerun with --source-root <dir> containing <dataset>/<archive>"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-fL",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-delay",
            "10",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "300",
            "--max-time",
            "21600",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            source["url"],
        ],
        check=True,
    )
    return {
        **verify_downloaded_source(source, destination),
        "transport": "upstream_url",
        "transport_source": source["url"],
    }


def stage_project_source(
    source: dict,
    destination: Path,
    *,
    source_root: Path,
    dataset: str,
) -> dict:
    source_path = source_root / dataset / source["name"]
    if not source_path.is_file():
        raise FileNotFoundError(f"local source archive is missing: {source_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)
    return {
        **verify_downloaded_source(source, destination),
        "transport": "local_archive",
        "transport_source": f"{dataset}/{source['name']}",
    }


def regular_files(archive: Path) -> list[str]:
    with tarfile.open(archive, "r:*") as handle:
        return sorted(member.name.lstrip("./") for member in handle if member.isfile())


def tar_text_by_basename(archive: Path, basename: str) -> str:
    with tarfile.open(archive, "r:*") as handle:
        matches = [
            member
            for member in handle
            if member.isfile() and PurePosixPath(member.name).name == basename
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one {basename} in {archive}, found {len(matches)}")
        extracted = handle.extractfile(matches[0])
        if extracted is None:
            raise ValueError(f"cannot read {matches[0].name} from {archive}")
        return extracted.read().decode("utf-8-sig")


def tar_bytes_by_basename(archive: Path, basename: str) -> bytes:
    with tarfile.open(archive, "r:*") as handle:
        matches = [
            member
            for member in handle
            if member.isfile() and PurePosixPath(member.name).name == basename
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one {basename} in {archive}, found {len(matches)}")
        extracted = handle.extractfile(matches[0])
        if extracted is None:
            raise ValueError(f"cannot read {matches[0].name} from {archive}")
        return extracted.read()


def suffix_lookup(members: list[str], expected: list[str]) -> dict[str, str]:
    member_set = set(members)
    by_basename: dict[str, list[str]] = {}
    for member in members:
        by_basename.setdefault(PurePosixPath(member).name, []).append(member)
    resolved = {}
    for value in expected:
        normalized = value.lstrip("./")
        if normalized in member_set:
            resolved[value] = normalized
            continue
        suffix = [member for member in by_basename.get(PurePosixPath(normalized).name, []) if member.endswith(normalized)]
        if len(suffix) != 1:
            raise ValueError(f"archive member resolution failed for {value}: {suffix[:3]}")
        resolved[value] = suffix[0]
    return resolved


def _mat_strings(value) -> list[str]:
    import numpy as np

    def unwrap(item):
        while isinstance(item, np.ndarray):
            if item.size != 1:
                break
            item = item.reshape(-1)[0]
        return str(item)

    return [unwrap(item) for item in np.asarray(value).reshape(-1)]


def parse_stanford_dogs(local_raw: Path) -> tuple[list[dict], dict]:
    import numpy as np
    from scipy.io import loadmat

    lists = local_raw / "lists.tar"
    images = local_raw / "images.tar"
    members = regular_files(images)
    rows = []
    for split, basename in (("train", "train_list.mat"), ("test", "test_list.mat")):
        payload = loadmat(io.BytesIO(tar_bytes_by_basename(lists, basename)))
        paths = _mat_strings(payload["file_list"])
        labels = np.asarray(payload["labels"]).reshape(-1).astype(int).tolist()
        if len(paths) != len(labels):
            raise ValueError(f"Dogs {split} path/label count mismatch")
        resolved = suffix_lookup(members, paths)
        for path, source_label in zip(paths, labels):
            breed_dir = PurePosixPath(path).parts[0]
            class_name = breed_dir.split("-", 1)[-1].replace("_", " ")
            rows.append(
                {
                    "split": split,
                    "source_class_id": str(source_label),
                    "class_name": class_name,
                    "source_image_id": f"{split}:{path}",
                    "archive_file": "raw/images.tar",
                    "archive_member": resolved[path],
                }
            )
    if len(rows) != 20_580 or Counter(row["split"] for row in rows) != {"train": 12_000, "test": 8_580}:
        raise ValueError("Stanford Dogs official split count mismatch")
    if len(set(row["source_class_id"] for row in rows)) != 120:
        raise ValueError("Stanford Dogs must contain 120 classes")
    return rows, {"name": "official train_list.mat/test_list.mat", "train": 12_000, "test": 8_580}


def parse_key_value_lines(text: str) -> dict[str, str]:
    rows = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, value = line.split(maxsplit=1)
        if key in rows:
            raise ValueError(f"duplicate metadata key: {key}")
        rows[key] = value.strip()
    return rows


def source_id_sort_key(source_id: str) -> tuple[int, int | str]:
    if source_id.isdecimal():
        return (0, int(source_id))
    return (1, source_id)


def parse_nabirds(local_raw: Path) -> tuple[list[dict], dict]:
    archive = local_raw / "nabirds.tar.gz"
    image_paths = parse_key_value_lines(tar_text_by_basename(archive, "images.txt"))
    class_labels = parse_key_value_lines(tar_text_by_basename(archive, "image_class_labels.txt"))
    train_flags = parse_key_value_lines(tar_text_by_basename(archive, "train_test_split.txt"))
    class_names = parse_key_value_lines(tar_text_by_basename(archive, "classes.txt"))
    if set(image_paths) != set(class_labels) or set(image_paths) != set(train_flags):
        raise ValueError("NABirds metadata image id sets differ")
    members = regular_files(archive)
    resolved = suffix_lookup(members, [f"images/{path}" for path in image_paths.values()])
    rows = []
    for image_id in sorted(image_paths, key=source_id_sort_key):
        source_class = class_labels[image_id]
        image_path = image_paths[image_id]
        lookup_key = f"images/{image_path}"
        rows.append(
            {
                "split": "train" if train_flags[image_id] == "1" else "test",
                "source_class_id": source_class,
                "class_name": class_names[source_class],
                "source_image_id": image_id,
                "archive_file": "raw/nabirds.tar.gz",
                "archive_member": resolved[lookup_key],
            }
        )
    counts = Counter(row["split"] for row in rows)
    if len(rows) != 48_562 or counts != {"train": 23_929, "test": 24_633}:
        raise ValueError(f"NABirds official split count mismatch: {counts}")
    if len(set(row["source_class_id"] for row in rows)) != 555:
        raise ValueError("NABirds must contain 555 visual categories")
    return rows, {
        "name": "official train_test_split.txt",
        "train": 23_929,
        "test": 24_633,
        "mirror_role": "archive transport only; official metadata defines identity",
    }


def json_from_tar(archive: Path) -> dict:
    with tarfile.open(archive, "r:*") as handle:
        json_members = [member for member in handle if member.isfile() and member.name.endswith(".json")]
        if len(json_members) != 1:
            raise ValueError(f"expected one JSON member in {archive}, found {len(json_members)}")
        extracted = handle.extractfile(json_members[0])
        if extracted is None:
            raise ValueError(f"cannot read {json_members[0].name}")
        return json.load(extracted)


def parse_inat2021_mini(local_raw: Path) -> tuple[list[dict], dict]:
    rows = []
    category_payloads = []
    for split, image_archive, annotation_archive in (
        ("train", "train_mini.tar.gz", "train_mini.json.tar.gz"),
        ("test", "val.tar.gz", "val.json.tar.gz"),
    ):
        annotations = json_from_tar(local_raw / annotation_archive)
        category_payloads.append({int(row["id"]): row for row in annotations["categories"]})
        annotation_by_image = {}
        for row in annotations["annotations"]:
            image_id = int(row["image_id"])
            if image_id in annotation_by_image:
                raise ValueError(f"multiple iNaturalist labels for image {image_id}")
            annotation_by_image[image_id] = int(row["category_id"])
        images = {int(row["id"]): row for row in annotations["images"]}
        if set(images) != set(annotation_by_image):
            raise ValueError(f"iNaturalist {split} image/annotation id mismatch")
        members = set(regular_files(local_raw / image_archive))
        for image_id in sorted(images):
            image = images[image_id]
            file_name = str(image["file_name"]).lstrip("./")
            if file_name not in members:
                suffix = [member for member in members if member.endswith(file_name)]
                if len(suffix) != 1:
                    raise ValueError(f"iNaturalist archive member missing: {file_name}")
                file_name = suffix[0]
            category_id = annotation_by_image[image_id]
            category = category_payloads[-1][category_id]
            class_name = str(category.get("name") or category.get("scientific_name") or category_id)
            rows.append(
                {
                    "split": split,
                    "source_class_id": str(category_id),
                    "class_name": class_name,
                    "source_image_id": f"{split}:{image_id}",
                    "archive_file": f"raw/{image_archive}",
                    "archive_member": file_name,
                }
            )
    if set(category_payloads[0]) != set(category_payloads[1]):
        raise ValueError("iNaturalist train/validation category sets differ")
    counts = Counter(row["split"] for row in rows)
    if counts != {"train": 500_000, "test": 100_000}:
        raise ValueError(f"iNaturalist mini split count mismatch: {counts}")
    if len(category_payloads[0]) != 10_000:
        raise ValueError("iNaturalist mini must contain all 10000 species")
    per_class = Counter((row["source_class_id"], row["split"]) for row in rows)
    if any(per_class[(str(category), "train")] != 50 or per_class[(str(category), "test")] != 10 for category in category_payloads[0]):
        raise ValueError("iNaturalist mini must have exactly 50 train-mini and 10 validation images per species")
    return rows, {
        "name": "official train_mini/validation COCO annotations",
        "train": 500_000,
        "test": 100_000,
        "n_species": 10_000,
        "class_subset": None,
    }


def parse_foodx_classes(text: str) -> dict[str, str]:
    names = {}
    for index, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if "," in line and line.split(",", 1)[0].strip().isdigit():
            source_id, name = line.split(",", 1)
            names[source_id.strip()] = name.strip()
        elif line.split(maxsplit=1)[0].isdigit() and len(line.split(maxsplit=1)) == 2:
            source_id, name = line.split(maxsplit=1)
            names[source_id] = name.strip()
        else:
            names[str(index)] = line
    if len(names) != 251:
        raise ValueError(f"FoodX class_list must contain 251 classes, got {len(names)}")
    return names


def parse_foodx_csv(text: str) -> list[tuple[str, str]]:
    rows = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if len(row) < 2:
            raise ValueError(f"invalid FoodX annotation row: {row}")
        image, label = row[0].strip(), row[1].strip()
        if not label.isdigit():
            continue
        rows.append((image, label))
    return rows


def parse_foodx251(local_raw: Path) -> tuple[list[dict], dict]:
    annot = local_raw / "annot.tar"
    classes = parse_foodx_classes(tar_text_by_basename(annot, "class_list.txt"))
    rows = []
    expected_counts = FOODX_OFFICIAL_SPLIT_COUNTS
    for split, csv_name, archive_name in (
        ("train", "train_info.csv", "train.tar"),
        ("test", "val_info.csv", "val.tar"),
    ):
        annotations = parse_foodx_csv(tar_text_by_basename(annot, csv_name))
        members = regular_files(local_raw / archive_name)
        resolved = suffix_lookup(members, [image for image, _ in annotations])
        for image, source_label in annotations:
            if source_label not in classes:
                raise ValueError(f"FoodX unknown label {source_label}")
            rows.append(
                {
                    "split": split,
                    "source_class_id": source_label,
                    "class_name": classes[source_label],
                    "source_image_id": f"{split}:{image}",
                    "archive_file": f"raw/{archive_name}",
                    "archive_member": resolved[image],
                }
            )
        if len(annotations) != expected_counts[split]:
            raise ValueError(f"FoodX {split} count mismatch: {len(annotations)}")
    if len(set(row["source_class_id"] for row in rows)) != 251:
        raise ValueError("FoodX must contain 251 classes in train and validation")
    return rows, {
        "name": "official train_info.csv/val_info.csv",
        "train": expected_counts["train"],
        "test": expected_counts["test"],
    }


PARSERS = {
    "stanford_dogs": parse_stanford_dogs,
    "nabirds": parse_nabirds,
    "inat2021_mini": parse_inat2021_mini,
    "foodx251": parse_foodx251,
}


def write_json_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def run(args) -> None:
    dataset = args.dataset
    output_root = Path(args.output_root)
    ensure_safe_output(output_root)
    output_root.mkdir(parents=True)
    write_json_new(
        output_root / "INCOMPLETE.json",
        {"dataset": dataset, "kind": "sdg_caev_data_preparation_incomplete"},
    )
    staging = Path(tempfile.mkdtemp(prefix=f"sdg_finegrained_{dataset}_", dir=args.scratch_root))
    try:
        local_raw = staging / "raw"
        receipts = []
        source_root = Path(args.source_root) if args.source_root else None
        for source in SOURCES[dataset]:
            destination = local_raw / source["name"]
            if source_root is None:
                print(f"[download] dataset={dataset} file={source['name']}", flush=True)
                receipts.append(download(source, destination))
            else:
                print(f"[stage] dataset={dataset} file={source['name']}", flush=True)
                receipts.append(
                    stage_project_source(
                        source,
                        destination,
                        source_root=source_root,
                        dataset=dataset,
                    )
                )

        rows, split_definition = PARSERS[dataset](local_raw)
        raw_output = output_root / "raw"
        raw_output.mkdir()
        archive_hashes = {}
        for receipt in receipts:
            source_path = local_raw / receipt["name"]
            destination = raw_output / receipt["name"]
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copyfile(source_path, destination)
            copied_hash = sha256_file(destination)
            if copied_hash != receipt["sha256"]:
                raise ValueError(f"copied archive hash mismatch: {destination}")
            archive_hashes[f"raw/{receipt['name']}"] = copied_hash

        source_payload = {
            "dataset": dataset,
            "assets": receipts,
            "license_note": (
                "NABirds official archive; official non-commercial/no-redistribution terms apply."
                if dataset == "nabirds"
                else "official download endpoint"
            ),
            "model_results_existed_before_freeze": False,
        }
        manifest = write_manifest_bundle(
            output_root / "manifest",
            dataset=dataset,
            raw_rows=rows,
            source=source_payload,
            split_definition=split_definition,
            archive_hashes=archive_hashes,
        )
        ready = {
            "schema_version": 1,
            "kind": "sdg_caev_finegrained_dataset_ready",
            "dataset": dataset,
            "commit": args.commit,
            "dataset_identity_sha256": manifest["dataset_identity_sha256"],
            "dataset_manifest_sha256": sha256_file(output_root / "manifest/dataset_manifest.json"),
            "archive_sha256": archive_hashes,
            "splits": manifest["splits"],
            "n_classes": manifest["n_classes"],
            "fallback_used": False,
        }
        write_json_new(output_root / "READY.json", ready)
        (output_root / "INCOMPLETE.json").unlink()
        print(f"[ready] dataset={dataset} root={output_root}", flush=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scratch-root", default=None, help="temporary staging directory (default: system temp)")
    parser.add_argument(
        "--source-root",
        help="directory with pre-downloaded archives as <source-root>/<dataset>/<archive>",
    )
    parser.add_argument("--commit", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
