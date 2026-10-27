#!/usr/bin/env python3
"""Prepare the four original held-out router datasets from public archives.

Turns the downloaded archives into HF Arrow datasets with embedded image bytes
under `<data-root>/<dataset>/hf_dataset/{train,test}`, which
spectral_generalization.py reads with `load_from_disk`. Expected archives:

    <data-root>/eurosat/archives/EuroSAT_RGB.zip
    <data-root>/caltech101/archives/caltech-101.zip
    <data-root>/flowers102/archives/102flowers.tgz (+ imagelabels.mat, setid.mat)
    <data-root>/dtd/archives/dtd-r1.0.1.tar.gz

EuroSAT uses a class-balanced 80/20 split and Caltech-101 a 70/30 split with
at most 30 training images per class (both seed 1701, BACKGROUND_Google
removed). Flowers-102 trains on the official train+val images and tests on the
official test images; DTD does the same with partition 1 (train1+val1 / test1).
"""

from __future__ import annotations

import argparse
import io
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Image, Value

from sdg_paths import DATA_ROOT

EUROSAT_CLASSES = {
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
}


def save_split(rows, classes, out):
    out = Path(out)
    done = out / "_SUCCESS"
    if done.exists():
        print(f"[prepare] exists: {out}", flush=True)
        return
    tmp = Path(tempfile.mkdtemp(prefix="hf_arrow_"))
    features = Features({"image": Image(decode=True), "label": Value("int64")})
    ds = Dataset.from_list(rows, features=features)
    ds.save_to_disk(str(tmp))
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tmp, out)
    shutil.rmtree(tmp)
    done.write_text("ok\n")
    (out / "classes.txt").write_text("\n".join(classes) + "\n")
    print(f"[prepare] saved {out}: n={len(ds)} classes={len(classes)}", flush=True)


def split_rows_by_class(items, train_frac, max_train_per_class=None, seed=1701):
    rng = np.random.RandomState(seed)
    train, test = [], []
    by_class = {}
    for item in items:
        by_class.setdefault(item["label"], []).append(item)
    for label in sorted(by_class):
        rows = by_class[label]
        rows = sorted(rows, key=lambda r: r["path"])
        order = np.arange(len(rows))
        rng.shuffle(order)
        n = len(rows)
        n_train = int(round(n * train_frac))
        n_train = max(1, min(n_train, n - 1))
        if max_train_per_class is not None:
            n_train = min(n_train, max_train_per_class)
        train_ids = set(order[:n_train].tolist())
        for i, row in enumerate(rows):
            out = {"image": {"bytes": row["bytes"], "path": row["path"]}, "label": int(label)}
            if i in train_ids:
                train.append(out)
            else:
                test.append(out)
    return train, test


def prepare_eurosat(root: Path):
    archive = root / "eurosat" / "archives" / "EuroSAT_RGB.zip"
    out_root = root / "eurosat" / "hf_dataset"
    if (out_root / "train" / "_SUCCESS").exists() and (out_root / "test" / "_SUCCESS").exists():
        print(f"[prepare] EuroSAT HF Arrow ready: {out_root}", flush=True)
        return
    if not archive.exists():
        raise FileNotFoundError(f"missing {archive}")

    print(f"[prepare] reading EuroSAT archive {archive}", flush=True)
    classes = []
    raw = []
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        classes = sorted({Path(n).parts[-2] for n in names})
        if len(set(classes) & EUROSAT_CLASSES) < 8:
            raise RuntimeError(f"unexpected EuroSAT classes: {classes}")
        class_to_label = {c: i for i, c in enumerate(classes)}
        for n in names:
            cls = Path(n).parts[-2]
            raw.append({"path": n, "label": class_to_label[cls], "bytes": zf.read(n)})
    train, test = split_rows_by_class(raw, train_frac=0.8, seed=1701)
    save_split(train, classes, out_root / "train")
    save_split(test, classes, out_root / "test")


def prepare_caltech101(root: Path):
    archive = root / "caltech101" / "archives" / "caltech-101.zip"
    out_root = root / "caltech101" / "hf_dataset"
    if (out_root / "train" / "_SUCCESS").exists() and (out_root / "test" / "_SUCCESS").exists():
        print(f"[prepare] Caltech101 HF Arrow ready: {out_root}", flush=True)
        return
    if not archive.exists():
        raise FileNotFoundError(f"missing {archive}")

    print(f"[prepare] reading Caltech101 archive {archive}", flush=True)
    raw = []
    with zipfile.ZipFile(archive) as zf:
        tar_name = next(n for n in zf.namelist() if n.endswith("101_ObjectCategories.tar.gz"))
        tar_bytes = io.BytesIO(zf.read(tar_name))
    with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tf:
        members = [
            m for m in tf.getmembers()
            if m.isfile() and m.name.lower().endswith((".jpg", ".jpeg", ".png"))
            and "BACKGROUND_Google" not in m.name
        ]
        classes = sorted({Path(m.name).parts[-2] for m in members})
        class_to_label = {c: i for i, c in enumerate(classes)}
        for m in members:
            cls = Path(m.name).parts[-2]
            f = tf.extractfile(m)
            raw.append({"path": m.name, "label": class_to_label[cls], "bytes": f.read()})
    train, test = split_rows_by_class(raw, train_frac=0.7, max_train_per_class=30, seed=1701)
    save_split(train, classes, out_root / "train")
    save_split(test, classes, out_root / "test")


def prepare_flowers102(root: Path):
    archive = root / "flowers102" / "archives" / "102flowers.tgz"
    label_mat = root / "flowers102" / "archives" / "imagelabels.mat"
    split_mat = root / "flowers102" / "archives" / "setid.mat"
    out_root = root / "flowers102" / "hf_dataset"
    if (out_root / "train" / "_SUCCESS").exists() and (out_root / "test" / "_SUCCESS").exists():
        print(f"[prepare] Flowers102 HF Arrow ready: {out_root}", flush=True)
        return
    for path in [archive, label_mat, split_mat]:
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")

    from scipy.io import loadmat

    labels = loadmat(label_mat, squeeze_me=True)["labels"].astype(int) - 1
    split_ids = loadmat(split_mat, squeeze_me=True)
    train_ids = set(np.asarray(split_ids["trnid"]).astype(int).tolist())
    train_ids.update(np.asarray(split_ids["valid"]).astype(int).tolist())
    test_ids = set(np.asarray(split_ids["tstid"]).astype(int).tolist())
    classes = [f"flower_{i:03d}" for i in range(102)]

    train, test = [], []
    print(f"[prepare] reading Flowers102 archive {archive}", flush=True)
    with tarfile.open(archive, mode="r:gz") as tf:
        for m in tf.getmembers():
            if not (m.isfile() and m.name.lower().endswith((".jpg", ".jpeg", ".png"))):
                continue
            stem = Path(m.name).stem
            image_id = int(stem.split("_")[-1])
            label = int(labels[image_id - 1])
            row = {"image": {"bytes": tf.extractfile(m).read(), "path": m.name}, "label": label}
            if image_id in train_ids:
                train.append(row)
            elif image_id in test_ids:
                test.append(row)
    save_split(train, classes, out_root / "train")
    save_split(test, classes, out_root / "test")


def prepare_dtd(root: Path):
    archive = root / "dtd" / "archives" / "dtd-r1.0.1.tar.gz"
    out_root = root / "dtd" / "hf_dataset"
    if (out_root / "train" / "_SUCCESS").exists() and (out_root / "test" / "_SUCCESS").exists():
        print(f"[prepare] DTD HF Arrow ready: {out_root}", flush=True)
        return
    if not archive.exists():
        raise FileNotFoundError(f"missing {archive}")

    print(f"[prepare] reading DTD archive {archive}", flush=True)
    with tarfile.open(archive, mode="r:gz") as tf:
        members = {m.name: m for m in tf.getmembers() if m.isfile()}

        def read_label_file(name):
            key = next(k for k in members if k.endswith(f"labels/{name}"))
            data = tf.extractfile(members[key]).read().decode("utf-8").splitlines()
            return [line.strip() for line in data if line.strip()]

        train_rel = set(read_label_file("train1.txt"))
        train_rel.update(read_label_file("val1.txt"))
        test_rel = set(read_label_file("test1.txt"))
        classes = sorted({p.split("/", 1)[0] for p in train_rel | test_rel})
        class_to_label = {c: i for i, c in enumerate(classes)}

        train, test = [], []
        for name, member in members.items():
            marker = "/images/"
            if marker not in name or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            rel = name.split(marker, 1)[1]
            cls = rel.split("/", 1)[0]
            row = {"image": {"bytes": tf.extractfile(member).read(), "path": name}, "label": class_to_label[cls]}
            if rel in train_rel:
                train.append(row)
            elif rel in test_rel:
                test.append(row)
    save_split(train, classes, out_root / "train")
    save_split(test, classes, out_root / "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--datasets", nargs="+", default=["eurosat", "caltech101"])
    args = ap.parse_args()
    root = Path(args.data_root)
    for name in args.datasets:
        if name == "eurosat":
            prepare_eurosat(root)
        elif name == "caltech101":
            prepare_caltech101(root)
        elif name == "flowers102":
            prepare_flowers102(root)
        elif name == "dtd":
            prepare_dtd(root)
        else:
            raise ValueError(name)


if __name__ == "__main__":
    main()
