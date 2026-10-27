#!/usr/bin/env python3
"""Prepare additional closed-set datasets for the scope-expansion router check.

Downloads (torchvision or the Hugging Face Hub) or reads official archives,
then writes `{data_root}/{name}/hf_dataset/{train,test}` splits that
`src/spectral_generalization.py` loads with `load_from_disk`. The paper's
coarse unseen block uses MNIST, Fashion-MNIST, SVHN, and GTSRB; the script
also supports STL-10, PCam, and SUN397.

GTSRB expects the official archives under `{data_root}/gtsrb/gtsrb/`:
GTSRB-Training_fixed.zip, GTSRB_Final_Test_Images.zip, GTSRB_Final_Test_GT.zip.
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from datasets import ClassLabel, Dataset, Features, Image, load_from_disk
from PIL import Image as PILImage

from sdg_paths import DATA_ROOT

HF_HUB_REPOS = {"mnist": "ylecun/mnist", "fashion_mnist": "zalando-datasets/fashion_mnist"}


def save_split(rows, classes, out: Path, force: bool):
    done = out / "_SUCCESS"
    if done.exists() and not force:
        print(f"[skip] {out}: n={len(load_from_disk(str(out)))}", flush=True)
        return
    features = Features({"image": Image(), "label": ClassLabel(names=list(classes))})
    ds = Dataset.from_list(rows, features=features)
    tmp = Path(tempfile.mkdtemp(prefix="sdg_scope_arrow_"))
    ds.save_to_disk(str(tmp), max_shard_size="512MB")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(tmp, out)
    shutil.rmtree(tmp)
    done.write_text("ok\n")
    (out / "classes.txt").write_text("\n".join(classes) + "\n")
    print(f"[saved] {out}: n={len(ds)} classes={len(classes)}", flush=True)


def pil_to_png_bytes(image):
    if not isinstance(image, PILImage.Image):
        image = PILImage.fromarray(np.asarray(image))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def image_path_for(ds, idx):
    for attr in ["samples", "imgs", "_samples"]:
        rows = getattr(ds, attr, None)
        if rows is not None:
            return str(rows[idx][0])
    for attr in ["_image_files", "_image_paths", "images"]:
        rows = getattr(ds, attr, None)
        if rows is not None and len(rows) > idx and isinstance(rows[idx], (str, Path)):
            return str(rows[idx])
    return None


def row_for(ds, idx):
    image, label = ds[idx]
    label = int(label.item()) if hasattr(label, "item") else int(label)
    path = image_path_for(ds, idx)
    if path and Path(path).exists():
        return {"image": {"path": path, "bytes": None}, "label": label}
    return {"image": {"path": None, "bytes": pil_to_png_bytes(image)}, "label": label}


def rows_from_dataset(ds, indices=None, log_name="dataset"):
    if indices is None:
        indices = range(len(ds))
    rows = []
    indices = list(indices)
    for n, idx in enumerate(indices, start=1):
        rows.append(row_for(ds, int(idx)))
        if n == 1 or n % 10000 == 0 or n == len(indices):
            print(f"  [{log_name}] {n}/{len(indices)}", flush=True)
    return rows


def labels_for(ds):
    if hasattr(ds, "labels"):
        return np.asarray(ds.labels)
    if hasattr(ds, "targets"):
        return np.asarray(ds.targets)
    if hasattr(ds, "_labels"):
        return np.asarray(ds._labels)
    return np.asarray([int(ds[i][1]) for i in range(len(ds))])


def class_balanced_indices(ds, split, seed=1701, train_frac=0.7, max_train_per_class=None):
    labels = labels_for(ds)
    rng = np.random.RandomState(seed)
    selected = []
    for c in sorted(np.unique(labels).tolist()):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_train = int(round(len(idx) * train_frac))
        n_train = max(1, min(n_train, len(idx) - 1))
        if max_train_per_class is not None:
            n_train = min(n_train, max_train_per_class)
        chosen = idx[:n_train] if split == "train" else idx[n_train:]
        selected.extend(chosen.tolist())
    return sorted(selected)


def classes_for(name, ds=None):
    if ds is not None and hasattr(ds, "classes"):
        return [str(x) for x in ds.classes]
    if name in {"svhn", "mnist", "fashion_mnist", "stl10"}:
        if name == "fashion_mnist":
            return [
                "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
            ]
        return [str(i) for i in range(10)]
    if name == "pcam":
        return ["negative", "positive"]
    raise ValueError(f"cannot infer classes for {name}")


def make_torchvision_dataset(name, split, root):
    from torch.utils.data import ConcatDataset
    from torchvision import datasets

    if name == "svhn":
        return datasets.SVHN(root=root, split=("train" if split == "train" else "test"), download=True)
    if name == "stl10":
        return datasets.STL10(root=root, split=("train" if split == "train" else "test"), download=True)
    if name == "gtsrb":
        return datasets.GTSRB(root=root, split=("train" if split == "train" else "test"), download=True)
    if name == "pcam":
        if split == "train":
            return ConcatDataset([
                datasets.PCAM(root=root, split="train", download=True),
                datasets.PCAM(root=root, split="val", download=True),
            ])
        return datasets.PCAM(root=root, split="test", download=True)
    if name == "mnist":
        return datasets.MNIST(root=root, train=(split == "train"), download=True)
    if name == "fashion_mnist":
        return datasets.FashionMNIST(root=root, train=(split == "train"), download=True)
    if name == "sun397":
        # SUN397 has no official torchvision train/test split; use a deterministic
        # class-balanced split with a capped support set to keep evaluator sweeps sane.
        base = datasets.SUN397(root=root, download=True)
        indices = class_balanced_indices(base, split, train_frac=0.7, max_train_per_class=50)
        return base, indices
    raise ValueError(name)


def flatten_concat(ds):
    rows = []
    for child in ds.datasets:
        rows.extend(rows_from_dataset(child, log_name=type(child).__name__))
    return rows


def prepare_one(name, data_root: Path, force: bool):
    root = data_root / name
    print(f"\n[prepare] {name} root={root}", flush=True)
    if name in HF_HUB_REPOS:
        prepare_hf_hub(name, root, force)
        return
    if name == "gtsrb":
        prepare_gtsrb_archives(root, force)
        return
    class_source = None
    split_rows = {}
    for split in ["train", "test"]:
        out = root / "hf_dataset" / split
        if (out / "_SUCCESS").exists() and not force:
            print(f"[skip] {name}/{split}: {out}", flush=True)
            continue
        ds = make_torchvision_dataset(name, split, root)
        indices = None
        if isinstance(ds, tuple):
            ds, indices = ds
        if class_source is None:
            class_source = ds
        if hasattr(ds, "datasets"):
            rows = flatten_concat(ds)
        else:
            rows = rows_from_dataset(ds, indices, log_name=f"{name}/{split}")
        split_rows[split] = rows
        classes = classes_for(name, class_source)
        save_split(rows, classes, out, force)
    classes = classes_for(name, class_source)
    status = {
        "dataset": name,
        "classes": len(classes),
        "splits": {
            split: len(load_from_disk(str(root / "hf_dataset" / split)))
            for split in ["train", "test"]
            if (root / "hf_dataset" / split / "_SUCCESS").exists()
        },
    }
    (root / "hf_dataset" / "DATASET_STATUS.json").write_text(str(status) + "\n")
    print(f"[status] {name}: {status}", flush=True)


def prepare_gtsrb_archives(root: Path, force: bool):
    """Prepare GTSRB directly from official zips.

    The torchvision path extracts tens of thousands of tiny files, which is
    slow on network filesystems.  Reading the zip members directly and writing
    Arrow shards keeps the dataset preparation sequential and worker-friendly.
    """
    archive_root = root / "gtsrb"
    train_zip = archive_root / "GTSRB-Training_fixed.zip"
    test_zip = archive_root / "GTSRB_Final_Test_Images.zip"
    gt_zip = archive_root / "GTSRB_Final_Test_GT.zip"
    for path in [train_zip, test_zip, gt_zip]:
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")

    classes = [str(i) for i in range(43)]
    out_root = root / "hf_dataset"
    if (
        (out_root / "train" / "_SUCCESS").exists()
        and (out_root / "test" / "_SUCCESS").exists()
        and not force
    ):
        print(f"[skip] gtsrb: {out_root}", flush=True)
        return

    print(f"[gtsrb] reading {train_zip}", flush=True)
    train_rows = []
    with zipfile.ZipFile(train_zip) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".ppm"))
        for i, n in enumerate(names, start=1):
            label = int(Path(n).parts[2])
            train_rows.append({"image": {"path": n, "bytes": zf.read(n)}, "label": label})
            if i == 1 or i % 10000 == 0 or i == len(names):
                print(f"  [gtsrb/train] {i}/{len(names)}", flush=True)

    print(f"[gtsrb] reading {gt_zip}", flush=True)
    with zipfile.ZipFile(gt_zip) as zf:
        gt_rows = list(csv.DictReader(io.TextIOWrapper(zf.open("GT-final_test.csv")), delimiter=";"))
    test_labels = {row["Filename"]: int(row["ClassId"]) for row in gt_rows}

    print(f"[gtsrb] reading {test_zip}", flush=True)
    test_rows = []
    with zipfile.ZipFile(test_zip) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".ppm"))
        for i, n in enumerate(names, start=1):
            fname = Path(n).name
            test_rows.append({"image": {"path": n, "bytes": zf.read(n)}, "label": test_labels[fname]})
            if i == 1 or i % 10000 == 0 or i == len(names):
                print(f"  [gtsrb/test] {i}/{len(names)}", flush=True)

    save_split(train_rows, classes, out_root / "train", force)
    save_split(test_rows, classes, out_root / "test", force)
    status = {
        "dataset": "gtsrb",
        "source": "official_zip_direct",
        "classes": len(classes),
        "splits": {"train": len(train_rows), "test": len(test_rows)},
    }
    (out_root / "DATASET_STATUS.json").write_text(str(status) + "\n")
    print(f"[status] gtsrb: {status}", flush=True)


def prepare_hf_hub(name, root: Path, force: bool):
    """Load MNIST / Fashion-MNIST (official train/test splits) from the HF Hub."""
    from datasets import load_dataset

    split_map = {"train": "train", "test": "test"}
    for out_split, hf_split in split_map.items():
        out = root / "hf_dataset" / out_split
        if (out / "_SUCCESS").exists() and not force:
            print(f"[skip] {name}/{out_split}: {out}", flush=True)
            continue
        print(f"[hf-hub] {HF_HUB_REPOS[name]}/{hf_split}", flush=True)
        ds = load_dataset(HF_HUB_REPOS[name], split=hf_split)
        tmp = Path(tempfile.mkdtemp(prefix=f"sdg_scope_{name}_{out_split}_"))
        ds.save_to_disk(str(tmp), max_shard_size="512MB")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(tmp, out)
        shutil.rmtree(tmp)
        (out / "_SUCCESS").write_text("ok\n")
        print(f"[saved] {out}: n={len(ds)}", flush=True)
    status = {
        "dataset": name,
        "source": "hf_hub",
        "splits": {
            split: len(load_from_disk(str(root / "hf_dataset" / split)))
            for split in ["train", "test"]
        },
    }
    (root / "hf_dataset" / "DATASET_STATUS.json").write_text(str(status) + "\n")
    print(f"[status] {name}: {status}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--datasets", nargs="+", default=["svhn", "stl10", "gtsrb", "pcam", "mnist", "fashion_mnist"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    for name in args.datasets:
        prepare_one(name, data_root, args.force)


if __name__ == "__main__":
    main()
