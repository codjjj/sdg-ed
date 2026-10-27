#!/usr/bin/env python3
"""Spectral Bottleneck generalization experiments.

This script extends the CUB/CIFAR observations in two directions:
1. P0 mechanism checks on cached features: per-PC Fisher ratio and PCA band ablations.
2. P1 multi-dataset checks: extract DINOv2 features, evaluate Cosine/Mahalanobis/WCos,
   run the same Fisher/PCA diagnostics, and run stats-only CIL.

The implementation is intentionally single-file and research-style so each
job can run independently for one dataset.

Feature extraction (`--mode dataset`) writes `<out-dir>/<dataset>/features/
{train,test}.pt` with one pooled vector per image: for `--backbone dinov2`,
the final-layer CLS token of DINOv2 ViT-B/14 with registers after resize-256
(bicubic), center-crop-224, and ImageNet normalization. Datasets are read from
`<data-root>/<dataset>/hf_dataset/{train,test}` when present (see the
prepare_*_data.py scripts), otherwise from the Hugging Face Hub or torchvision.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


from sdg_paths import DATA_ROOT, FEATURE_ROOT, MODEL_ROOT


DEFAULT_OUT = FEATURE_ROOT
DEFAULT_DATA = DATA_ROOT
# Local snapshot of facebook/dinov2-with-registers-base; a Hub repo id also works.
MODEL_PATH = MODEL_ROOT / "dinov2-with-registers-base"
CLIP_MODEL_PATH = MODEL_ROOT / "clip-vit-base-patch32"


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(suffix=".json"))
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    shutil.copy2(tmp, path)
    tmp.unlink(missing_ok=True)


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(suffix=".npz"))
    np.savez(tmp, **arrays)
    shutil.copy2(tmp, path)
    tmp.unlink(missing_ok=True)


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(suffix=".pt"))
    torch.save(obj, tmp)
    shutil.copy2(tmp, path)
    tmp.unlink(missing_ok=True)


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_cache(path: Path):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    feat = cache["pooled"].float()
    lab = cache["labels"].long()
    return feat, lab


def ncm_accuracy(train_feat, train_lab, test_feat, test_lab, classes):
    protos = torch.stack([train_feat[train_lab == c].mean(0) for c in classes])
    protos = F.normalize(protos, dim=1)
    test = F.normalize(test_feat, dim=1)
    pred = (test @ protos.T).argmax(1)
    pred_lab = torch.tensor([classes[int(i)] for i in pred])
    return float((pred_lab == test_lab).float().mean())


def mahalanobis_accuracy(train_feat, train_lab, test_feat, test_lab, classes, reg=1e-4):
    x = as_numpy(train_feat).astype(np.float64)
    y = as_numpy(train_lab)
    xt = as_numpy(test_feat).astype(np.float64)
    yt = as_numpy(test_lab)
    d = x.shape[1]
    scatter = np.zeros((d, d), dtype=np.float64)
    n_total = 0
    protos = []
    for c in classes:
        xc = x[y == c]
        mu = xc.mean(0)
        protos.append(mu)
        centered = xc - mu
        scatter += centered.T @ centered
        n_total += len(xc)
    denom = max(1, n_total - len(classes))
    sigma = scatter / denom + reg * np.eye(d)
    si = np.linalg.inv(sigma)
    protos = np.stack(protos)
    proto_scores = protos @ si
    bias = -0.5 * np.sum(protos * proto_scores, axis=1)
    scores = xt @ proto_scores.T + bias[None, :]
    pred = scores.argmax(1)
    pred_lab = np.array([classes[int(i)] for i in pred])
    return float((pred_lab == yt).mean())


def whitened_cosine_accuracy(train_feat, train_lab, test_feat, test_lab, classes, eps=1e-6):
    x = as_numpy(train_feat).astype(np.float64)
    xt = as_numpy(test_feat).astype(np.float64)
    mu = x.mean(0)
    xc = x - mu
    cov = (xc.T @ xc) / max(1, len(xc) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    w = vecs @ np.diag(1.0 / np.sqrt(vals + eps)) @ vecs.T
    xw = (x - mu) @ w.T
    xtw = (xt - mu) @ w.T
    return ncm_accuracy(
        torch.from_numpy(xw).float(),
        train_lab,
        torch.from_numpy(xtw).float(),
        test_lab,
        classes,
    )


def covariance_pca(train_feat):
    x = as_numpy(train_feat).astype(np.float64)
    mu = x.mean(0)
    xc = x - mu
    cov = (xc.T @ xc) / max(1, len(xc) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return mu, vals, vecs


def rankdata(a):
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def spearman(a, b):
    ra, rb = rankdata(np.asarray(a)), rankdata(np.asarray(b))
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float(np.mean(ra * rb))


def fisher_by_pc(train_feat, train_lab, mean, pcs):
    x = as_numpy(train_feat).astype(np.float64)
    y = as_numpy(train_lab)
    z = (x - mean) @ pcs
    classes = sorted(np.unique(y).tolist())
    global_mean = z.mean(0)
    between = np.zeros(z.shape[1], dtype=np.float64)
    within = np.zeros(z.shape[1], dtype=np.float64)
    n_total = 0
    for c in classes:
        zc = z[y == c]
        mc = zc.mean(0)
        between += len(zc) * (mc - global_mean) ** 2
        within += ((zc - mc) ** 2).sum(0)
        n_total += len(zc)
    between /= max(1, len(classes) - 1)
    within /= max(1, n_total - len(classes))
    fisher = between / (within + 1e-12)
    mass = fisher / (fisher.sum() + 1e-12)
    ranks = np.arange(len(fisher))
    return {
        "fisher": fisher,
        "fisher_mass": mass,
        "tail128_mass": float(mass[128:].sum()) if len(mass) > 128 else 0.0,
        "tail256_mass": float(mass[256:].sum()) if len(mass) > 256 else 0.0,
        "top64_mass": float(mass[:64].sum()),
        "top128_mass": float(mass[:128].sum()),
        "fisher_weighted_rank": float((ranks * mass).sum()),
        "spearman_logeig_logfisher": None,
    }


def pca_selections(dim):
    base = {
        "all": np.arange(dim),
        "top16": np.arange(min(16, dim)),
        "top32": np.arange(min(32, dim)),
        "top64": np.arange(min(64, dim)),
        "top128": np.arange(min(128, dim)),
        "top256": np.arange(min(256, dim)),
        "top512": np.arange(min(512, dim)),
        "bottom64": np.arange(max(0, dim - 64), dim),
        "bottom128": np.arange(max(0, dim - 128), dim),
        "drop_bottom64": np.arange(max(1, dim - 64)),
        "drop_bottom128": np.arange(max(1, dim - 128)),
        "drop_top16": np.arange(min(16, dim), dim),
        "drop_top64": np.arange(min(64, dim), dim),
    }
    if dim >= 256:
        base["band_64_128"] = np.arange(64, 128)
        base["band_128_256"] = np.arange(128, 256)
        base["band_256_end"] = np.arange(256, dim)
    return {k: v for k, v in base.items() if len(v) > 0}


def pca_ablation(train_feat, train_lab, test_feat, test_lab, classes, mean, pcs):
    x = as_numpy(train_feat).astype(np.float64)
    xt = as_numpy(test_feat).astype(np.float64)
    z = (x - mean) @ pcs
    zt = (xt - mean) @ pcs
    results = {}
    for name, idx in pca_selections(z.shape[1]).items():
        tr = torch.from_numpy(z[:, idx]).float()
        te = torch.from_numpy(zt[:, idx]).float()
        results[name] = {
            "dim": int(len(idx)),
            "cosine": ncm_accuracy(tr, train_lab, te, test_lab, classes),
            "mahalanobis": mahalanobis_accuracy(tr, train_lab, te, test_lab, classes),
        }
        print(
            f"    {name:<16} d={len(idx):>3d} "
            f"Cos={results[name]['cosine']*100:>6.2f}% "
            f"Mah={results[name]['mahalanobis']*100:>6.2f}%"
        )
    return results


def stats_only_cil(train_feat, train_lab, test_feat, test_lab, n_tasks=10, seed=42):
    classes = sorted(torch.unique(train_lab).tolist())
    rng = np.random.RandomState(seed)
    shuffled = classes[:]
    rng.shuffle(shuffled)
    chunks = np.array_split(np.array(shuffled), min(n_tasks, len(classes)))
    seen = []
    per = []
    for ti, chunk in enumerate(chunks):
        seen += [int(c) for c in chunk.tolist()]
        train_mask = torch.zeros(len(train_lab), dtype=torch.bool)
        test_mask = torch.zeros(len(test_lab), dtype=torch.bool)
        for c in seen:
            train_mask |= train_lab == c
            test_mask |= test_lab == c
        trf, trl = train_feat[train_mask], train_lab[train_mask]
        tef, tel = test_feat[test_mask], test_lab[test_mask]
        cos = ncm_accuracy(trf, trl, tef, tel, seen)
        mah = mahalanobis_accuracy(trf, trl, tef, tel, seen)
        per.append({"task": ti, "n_classes": len(seen), "stats_cos": cos, "stats_mah": mah})
        print(f"    CIL {ti}: {len(seen)} cls  Cos={cos*100:.2f}%  Mah={mah*100:.2f}%")
    out = {"per_session": per}
    for m in ["stats_cos", "stats_mah"]:
        vals = [x[m] for x in per]
        out[m + "_final"] = float(vals[-1])
        out[m + "_avg"] = float(np.mean(vals))
        out[m + "_forgetting"] = float(vals[0] - vals[-1])
    return out


def analyze_features(dataset_name, train_feat, train_lab, test_feat, test_lab, out_dir, run_cil=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(torch.unique(train_lab).tolist())
    print(f"[analyze] {dataset_name}: train={tuple(train_feat.shape)} test={tuple(test_feat.shape)} classes={len(classes)}")
    mean, eigvals, pcs = covariance_pca(train_feat)
    top10 = float(eigvals[:10].sum() / eigvals.sum())
    ratio10_tail = float(eigvals[9] / max(eigvals[-10], 1e-18))
    cond = float(eigvals[0] / max(eigvals[-1], 1e-18))
    fisher = fisher_by_pc(train_feat, train_lab, mean, pcs)
    fisher["spearman_logeig_logfisher"] = spearman(np.log(eigvals + 1e-18), np.log(fisher["fisher"] + 1e-18))

    print("[matrix]")
    cos = ncm_accuracy(train_feat, train_lab, test_feat, test_lab, classes)
    mah = mahalanobis_accuracy(train_feat, train_lab, test_feat, test_lab, classes)
    wcos = whitened_cosine_accuracy(train_feat, train_lab, test_feat, test_lab, classes)
    print(f"  Cos={cos*100:.2f}% Mah={mah*100:.2f}% WCos={wcos*100:.2f}% gap={((mah-cos)*100):.2f}pp")

    print("[pca ablation]")
    ablation = pca_ablation(train_feat, train_lab, test_feat, test_lab, classes, mean, pcs)
    cil = stats_only_cil(train_feat, train_lab, test_feat, test_lab) if run_cil else None

    result = {
        "dataset": dataset_name,
        "n_train": int(len(train_lab)),
        "n_test": int(len(test_lab)),
        "n_classes": int(len(classes)),
        "anisotropy": {
            "top10_var_pct": top10,
            "eig10_tail10_ratio": ratio10_tail,
            "condition_number": cond,
        },
        "fisher": {
            "tail128_mass": fisher["tail128_mass"],
            "tail256_mass": fisher["tail256_mass"],
            "top64_mass": fisher["top64_mass"],
            "top128_mass": fisher["top128_mass"],
            "fisher_weighted_rank": fisher["fisher_weighted_rank"],
            "spearman_logeig_logfisher": fisher["spearman_logeig_logfisher"],
        },
        "classifier_matrix": {
            "cosine": cos,
            "mahalanobis": mah,
            "whitened_cosine": wcos,
            "mahal_minus_cos_pp": (mah - cos) * 100.0,
        },
        "pca_ablation": ablation,
        "cil": cil,
    }
    save_json(result, out_dir / "result.json")
    save_npz(
        out_dir / "fisher_pca.npz",
        eigenvalues=eigvals,
        fisher=fisher["fisher"],
        fisher_mass=fisher["fisher_mass"],
    )
    write_report(result, out_dir / "exp_result.md")
    return result


def write_report(result, path: Path):
    lines = []
    lines.append(f"# Spectral Generalization Result: {result['dataset']}\n")
    lines.append("## Summary\n")
    lines.append(f"- train/test/classes: {result['n_train']} / {result['n_test']} / {result['n_classes']}")
    an = result["anisotropy"]
    mat = result["classifier_matrix"]
    fs = result["fisher"]
    lines.append(f"- top10 variance: {an['top10_var_pct']*100:.2f}%")
    lines.append(f"- eig10/tail10 ratio: {an['eig10_tail10_ratio']:.2f}")
    lines.append(f"- Fisher tail128 mass: {fs['tail128_mass']*100:.2f}%")
    lines.append(f"- Spearman(log eig, log fisher): {fs['spearman_logeig_logfisher']:.4f}")
    lines.append(f"- Cosine: {mat['cosine']*100:.2f}%")
    lines.append(f"- Mahalanobis: {mat['mahalanobis']*100:.2f}%")
    lines.append(f"- WCos: {mat['whitened_cosine']*100:.2f}%")
    lines.append(f"- Mahal-Cos gap: {mat['mahal_minus_cos_pp']:.2f}pp\n")

    lines.append("## PCA Ablation\n")
    lines.append("| selection | dim | Cos | Mahal | Gap |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, r in result["pca_ablation"].items():
        gap = (r["mahalanobis"] - r["cosine"]) * 100.0
        lines.append(f"| {name} | {r['dim']} | {r['cosine']*100:.2f} | {r['mahalanobis']*100:.2f} | {gap:.2f} |")
    if result.get("cil"):
        lines.append("\n## Stats-only CIL\n")
        lines.append("| method | final | average | forgetting |")
        lines.append("|---|---:|---:|---:|")
        cil = result["cil"]
        for m in ["stats_cos", "stats_mah"]:
            lines.append(
                f"| {m} | {cil[m + '_final']*100:.2f} | "
                f"{cil[m + '_avg']*100:.2f} | {cil[m + '_forgetting']*100:.2f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(suffix=".md"))
    tmp.write_text("\n".join(lines) + "\n")
    shutil.copy2(tmp, path)
    tmp.unlink(missing_ok=True)


def load_cached_dataset(name, args):
    if args.backbone != "dinov2":
        return None
    if name == "cub200" and args.cub_cache:
        root = Path(args.cub_cache)
        tr = torch.load(root / "train.pt", map_location="cpu", weights_only=False)
        te = torch.load(root / "test.pt", map_location="cpu", weights_only=False)
        return tr["pooled"].float(), tr["labels"].long(), te["pooled"].float(), te["labels"].long()
    if name == "cifar100" and args.cifar100_cache:
        root = Path(args.cifar100_cache)
        tr = torch.load(root / "train.pt", map_location="cpu", weights_only=False)
        te = torch.load(root / "test.pt", map_location="cpu", weights_only=False)
        return tr["pooled"].float(), tr["labels"].long(), te["pooled"].float(), te["labels"].long()
    return None


def make_transform(args=None):
    import torchvision.transforms as T
    resize_size = int(getattr(args, "resize_size", 256))
    crop_size = int(getattr(args, "crop_size", 224))
    crop_mode = getattr(args, "crop_mode", "center")
    if crop_mode == "center":
        spatial = [
            T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(crop_size),
        ]
    elif crop_mode == "square":
        spatial = [
            T.Resize((crop_size, crop_size), interpolation=T.InterpolationMode.BICUBIC),
        ]
    else:
        raise ValueError(f"unknown crop_mode={crop_mode}")
    return T.Compose(spatial + [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_clip_transform():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]),
    ])


def _size_pair(value, default=224):
    if isinstance(value, dict):
        if "height" in value and "width" in value:
            return int(value["height"]), int(value["width"])
        if "shortest_edge" in value:
            v = int(value["shortest_edge"])
            return v, v
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if value is not None:
        v = int(value)
        return v, v
    return default, default


def make_hf_vision_transform(model_path):
    import torchvision.transforms as T
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(str(model_path))
    crop_h, crop_w = _size_pair(getattr(processor, "crop_size", None), default=224)
    resize_h, resize_w = _size_pair(getattr(processor, "size", None), default=max(crop_h, crop_w))
    if resize_h < crop_h or resize_w < crop_w:
        resize_h, resize_w = crop_h, crop_w
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    return T.Compose([
        T.Resize((resize_h, resize_w), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop((crop_h, crop_w)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def hf_repo_for(name):
    return {
        "cifar10": "cifar10",
        "food101": "food101",
        "pets": "timm/oxford-iiit-pet",
    }.get(name)


def hf_image_key_for(name):
    return "img" if name == "cifar10" else "image"


def hf_split_for(name, split):
    if name == "food101" and split == "test":
        return "validation"
    return split


def target_to_int(y):
    if isinstance(y, (tuple, list)):
        y = y[0]
    if isinstance(y, torch.Tensor):
        y = int(y.item())
    return int(y)


class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def dataset_labels(ds):
    if hasattr(ds, "_labels"):
        return list(ds._labels)
    if hasattr(ds, "targets"):
        return list(ds.targets)
    if hasattr(ds, "y"):
        return list(ds.y)
    if hasattr(ds, "samples"):
        return [int(s[1]) for s in ds.samples]
    raise ValueError(f"cannot infer labels for {type(ds)}")


def class_balanced_split(ds, split, seed=0, train_frac=0.8, max_train_per_class=None):
    labels = np.asarray(dataset_labels(ds))
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for c in sorted(np.unique(labels).tolist()):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_frac))
        n_train = max(1, min(n_train, n - 1))
        if max_train_per_class is not None:
            n_train = min(n_train, max_train_per_class)
        train_idx.extend(idx[:n_train].tolist())
        test_idx.extend(idx[n_train:].tolist())
    indices = train_idx if split == "train" else test_idx
    return IndexedSubset(ds, sorted(indices))


def prepare_tiny_imagenet(root: Path):
    from torchvision.datasets.utils import download_url, extract_archive
    root.mkdir(parents=True, exist_ok=True)
    base = root / "tiny-imagenet-200"
    if not base.exists():
        zip_name = "tiny-imagenet-200.zip"
        if not (root / zip_name).exists():
            print("[download] tiny-imagenet-200")
            download_url("http://cs231n.stanford.edu/tiny-imagenet-200.zip", str(root), filename=zip_name)
        extract_archive(str(root / zip_name), str(root))
    val_by_class = base / "val_by_class"
    if not val_by_class.exists():
        val_by_class.mkdir()
        ann = base / "val/val_annotations.txt"
        for line in ann.read_text().splitlines():
            fname, cls = line.split("\t")[:2]
            cls_dir = val_by_class / cls
            cls_dir.mkdir(exist_ok=True)
            src = base / "val/images" / fname
            dst = cls_dir / fname
            if not dst.exists():
                os.symlink(src, dst)
    return base / "train", val_by_class


class HFDataset(Dataset):
    def __init__(self, hf_ds, transform, image_key):
        self.hf_ds = hf_ds
        self.transform = transform
        self.image_key = image_key

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        row = self.hf_ds[idx]
        image = row[self.image_key].convert("RGB")
        return self.transform(image), int(row["label"])


def make_hf_dataset(name, split, transform):
    """Load a dataset split from the Hugging Face Hub when a mapping exists."""
    repo = hf_repo_for(name)
    if repo is None:
        return None
    from datasets import load_dataset

    hf_split = hf_split_for(name, split)
    image_key = hf_image_key_for(name)
    print(f"[dataset] {name}/{split}: load_dataset repo={repo} split={hf_split}")
    hf_ds = load_dataset(repo, split=hf_split)
    return HFDataset(hf_ds, transform, image_key)


def make_dataset(name, split, root, transform, args):
    prepared = Path(root) / "hf_dataset" / split
    if prepared.exists():
        from datasets import load_from_disk
        print(f"[dataset] {name}/{split}: load_from_disk {prepared}")
        return HFDataset(load_from_disk(str(prepared)), transform, hf_image_key_for(name))
    if name == "cub200":
        return make_cub200_imagefolder(split, transform, root)
    if not args.torchvision_only:
        hf_ds = make_hf_dataset(name, split, transform)
        if hf_ds is not None:
            return hf_ds
    return make_torchvision_dataset(name, split, root, transform)


def make_cub200_imagefolder(split, transform, root):
    """CUB-200-2011 as an ImageFolder: <data-root>/cub200/{train,test}/<class>/<image>."""
    from torchvision import datasets
    data_dir = Path(root) / ("train" if split == "train" else "test")
    if not data_dir.exists():
        raise FileNotFoundError(f"CUB-200 ImageFolder not found at {data_dir}")
    print(f"[dataset] cub200/{split}: ImageFolder {data_dir}")
    return datasets.ImageFolder(data_dir, transform=transform)


def make_torchvision_dataset(name, split, root, transform):
    from torchvision import datasets
    root = Path(root)
    download = True
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=(split == "train"), transform=transform, download=download)
    if name == "cifar100":
        return datasets.CIFAR100(root=root, train=(split == "train"), transform=transform, download=download)
    if name == "food101":
        return datasets.Food101(root=root, split=("train" if split == "train" else "test"), transform=transform, download=download)
    if name == "aircraft":
        sp = "trainval" if split == "train" else "test"
        return datasets.FGVCAircraft(root=root, split=sp, annotation_level="variant", transform=transform, download=download)
    if name == "cars":
        return datasets.StanfordCars(root=root, split=("train" if split == "train" else "test"), transform=transform, download=download)
    if name == "pets":
        sp = "trainval" if split == "train" else "test"
        return datasets.OxfordIIITPet(root=root, split=sp, target_types="category", transform=transform, download=download)
    if name == "tiny_imagenet":
        train_dir, test_dir = prepare_tiny_imagenet(root / "tiny-imagenet")
        data_dir = train_dir if split == "train" else test_dir
        return datasets.ImageFolder(data_dir, transform=transform)
    if name == "imagenet100":
        data_dir = root / "imagenet100" / ("train" if split == "train" else "val")
        if not data_dir.exists():
            raise FileNotFoundError(f"ImageNet-100 not found at {data_dir}; place ImageFolder data there first.")
        return datasets.ImageFolder(data_dir, transform=transform)
    if name == "flowers102":
        if split == "train":
            return ConcatDataset([
                datasets.Flowers102(root=root, split="train", transform=transform, download=download),
                datasets.Flowers102(root=root, split="val", transform=transform, download=download),
            ])
        return datasets.Flowers102(root=root, split="test", transform=transform, download=download)
    if name == "dtd":
        if split == "train":
            return ConcatDataset([
                datasets.DTD(root=root, split="train", partition=1, transform=transform, download=download),
                datasets.DTD(root=root, split="val", partition=1, transform=transform, download=download),
            ])
        return datasets.DTD(root=root, split="test", partition=1, transform=transform, download=download)
    if name == "eurosat":
        local = root / "eurosat" / "2750"
        if not local.exists():
            raise FileNotFoundError(f"EuroSAT not found at {local}; run src/prepare_heldout_data.py first.")
        ds = datasets.EuroSAT(root=root, transform=transform, download=False)
        return class_balanced_split(ds, split, seed=1701, train_frac=0.8)
    if name == "caltech101":
        local = root / "caltech101" / "101_ObjectCategories"
        if not local.exists():
            raise FileNotFoundError(f"Caltech101 not found at {local}; run src/prepare_heldout_data.py first.")
        ds = datasets.Caltech101(root=root, target_type="category", transform=transform, download=False)
        return class_balanced_split(ds, split, seed=1701, train_frac=0.7, max_train_per_class=30)
    if name == "svhn":
        return datasets.SVHN(root=root, split=("train" if split == "train" else "test"), transform=transform, download=True)
    if name == "stl10":
        return datasets.STL10(root=root, split=("train" if split == "train" else "test"), transform=transform, download=True)
    if name == "gtsrb":
        return datasets.GTSRB(root=root, split=("train" if split == "train" else "test"), transform=transform, download=True)
    if name == "pcam":
        if split == "train":
            return ConcatDataset([
                datasets.PCAM(root=root, split="train", transform=transform, download=True),
                datasets.PCAM(root=root, split="val", transform=transform, download=True),
            ])
        return datasets.PCAM(root=root, split="test", transform=transform, download=True)
    if name == "mnist":
        return datasets.MNIST(root=root, train=(split == "train"), transform=transform, download=True)
    if name == "fashion_mnist":
        return datasets.FashionMNIST(root=root, train=(split == "train"), transform=transform, download=True)
    if name == "sun397":
        ds = datasets.SUN397(root=root, transform=transform, download=download)
        return class_balanced_split(ds, split, seed=1701, train_frac=0.7, max_train_per_class=50)
    raise ValueError(f"unknown dataset: {name}")


def extract_features_for_dataset(name, args):
    out_dir = Path(args.out_dir) / name
    feat_dir = out_dir / "features"
    train_path, test_path = feat_dir / "train.pt", feat_dir / "test.pt"
    if train_path.exists() and test_path.exists() and not args.force_extract:
        print(f"[skip] cached features for {name}")
        trf, trl = load_cache(train_path)
        tef, tel = load_cache(test_path)
        return trf, trl, tef, tel

    if args.backbone == "clip":
        transform = make_clip_transform()
    elif args.backbone == "hf_vision":
        transform = make_hf_vision_transform(args.model_path)
    else:
        transform = make_transform(args)
    root = Path(args.data_root) / name
    print(f"[dataset] {name} root={root}")
    train_set = make_dataset(name, "train", root, transform, args)
    test_set = make_dataset(name, "test", root, transform, args)

    device = args.device
    model_path = Path(args.model_path)
    if args.backbone == "dinov2":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True).to(device).eval()
    elif args.backbone == "clip":
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained(str(model_path)).to(device).eval()
    elif args.backbone == "hf_vision":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True).to(device).eval()
    else:
        raise ValueError(args.backbone)

    for split, ds, path in [("train", train_set, train_path), ("test", test_set, test_path)]:
        print(f"[extract] {name}/{split}: n={len(ds)}")
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        feats, labels = [], []
        for bi, (images, target) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                if args.backbone == "dinov2":
                    out = model(images)
                    pooled = out.last_hidden_state[:, 0, :].detach().cpu().float()
                elif args.backbone == "clip":
                    pooled = model.get_image_features(pixel_values=images).detach().cpu().float()
                else:
                    if hasattr(model, "get_image_features"):
                        pooled = model.get_image_features(pixel_values=images).detach().cpu().float()
                    else:
                        out = model(pixel_values=images)
                        pooled = hf_vision_pooled(out).detach().cpu().float()
            feats.append(pooled)
            labels.append(torch.tensor([target_to_int(t) for t in target]).long())
            if bi == 1 or bi % args.log_every == 0 or bi == len(loader):
                seen = min(bi * args.batch_size, len(ds))
                print(f"  [{name}/{split}] batch {bi}/{len(loader)} images={seen}/{len(ds)}")
        cache = {"pooled": torch.cat(feats), "labels": torch.cat(labels)}
        atomic_torch_save(cache, path)
        print(f"  saved {path}: {tuple(cache['pooled'].shape)}")

    trf, trl = load_cache(train_path)
    tef, tel = load_cache(test_path)
    return trf, trl, tef, tel


def hf_vision_pooled(out):
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "vision_model_output") and out.vision_model_output is not None:
        vout = out.vision_model_output
        if hasattr(vout, "last_hidden_state") and vout.last_hidden_state is not None:
            h = vout.last_hidden_state
            return h[:, 0, :] if h.ndim == 3 else h.mean(dim=(-1, -2))
        if hasattr(vout, "pooler_output") and vout.pooler_output is not None:
            return vout.pooler_output
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        h = out.last_hidden_state
        if h.ndim == 4:
            return h.mean(dim=(-1, -2))
        if h.ndim == 3:
            return h[:, 0, :]
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    h = out.last_hidden_state
    if h.ndim == 4:
        return h.mean(dim=(-1, -2))
    if h.ndim == 3:
        return h[:, 0, :]
    raise ValueError(f"cannot pool output with shape {tuple(h.shape)}")


def run_dataset(name, args):
    cached = load_cached_dataset(name, args) if not args.force_extract else None
    if cached is not None:
        trf, trl, tef, tel = cached
    else:
        trf, trl, tef, tel = extract_features_for_dataset(name, args)
    return analyze_features(name, trf, trl, tef, tel, Path(args.out_dir) / name, run_cil=not args.no_cil)


def prepare_hf_datasets(args):
    from datasets import load_dataset

    for name in args.datasets:
        repo = hf_repo_for(name)
        if repo is None:
            print(f"[skip] no Hugging Face Hub mapping for {name}")
            continue
        for split in ["train", "test"]:
            out = Path(args.data_root) / name / "hf_dataset" / split
            done = out / "_SUCCESS"
            if done.exists() and not args.force_prepare:
                print(f"[skip] prepared {name}/{split}: {out}")
                continue
            hf_split = hf_split_for(name, split)
            tmp = Path(tempfile.mkdtemp(prefix=f"spectral_{name}_{split}_"))
            print(f"[prepare] {name}/{split}: repo={repo} split={hf_split}")
            ds = load_dataset(repo, split=hf_split)
            ds.save_to_disk(str(tmp))
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                shutil.rmtree(out)
            shutil.copytree(tmp, out)
            shutil.rmtree(tmp)
            done.write_text("ok\n")
            print(f"[prepare] saved {name}/{split}: n={len(ds)} -> {out}")


def run_p0(args):
    results = {}
    for name in args.datasets:
        print("\n" + "=" * 80)
        print(f"P0: {name}")
        print("=" * 80)
        results[name] = run_dataset(name, args)
    save_json({k: summarize(v) for k, v in results.items()}, Path(args.out_dir) / "p0_summary.json")


def summarize(result):
    return {
        "dataset": result["dataset"],
        "n_classes": result["n_classes"],
        "top10_var_pct": result["anisotropy"]["top10_var_pct"],
        "eig10_tail10_ratio": result["anisotropy"]["eig10_tail10_ratio"],
        "tail128_fisher_mass": result["fisher"]["tail128_mass"],
        "spearman_logeig_logfisher": result["fisher"]["spearman_logeig_logfisher"],
        "cosine": result["classifier_matrix"]["cosine"],
        "mahalanobis": result["classifier_matrix"]["mahalanobis"],
        "whitened_cosine": result["classifier_matrix"]["whitened_cosine"],
        "mahal_minus_cos_pp": result["classifier_matrix"]["mahal_minus_cos_pp"],
    }


def aggregate(args):
    out_dir = Path(args.out_dir)
    rows = []
    for p in sorted(out_dir.glob("*/result.json")):
        with open(p) as f:
            rows.append(summarize(json.load(f)))
    lines = ["# Spectral Generalization Aggregate\n"]
    lines.append("| dataset | classes | top10 var | tail128 Fisher | rho(logeig,logfisher) | Cos | Mahal | WCos | Gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['n_classes']} | {r['top10_var_pct']*100:.2f} | "
            f"{r['tail128_fisher_mass']*100:.2f} | {r['spearman_logeig_logfisher']:.3f} | "
            f"{r['cosine']*100:.2f} | {r['mahalanobis']*100:.2f} | "
            f"{r['whitened_cosine']*100:.2f} | {r['mahal_minus_cos_pp']:.2f} |"
        )
    (out_dir / "aggregate.md").write_text("\n".join(lines) + "\n")
    save_json(rows, out_dir / "aggregate.json")
    print(f"[aggregate] wrote {out_dir / 'aggregate.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["p0", "dataset", "aggregate", "prepare_hf"], required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--datasets", nargs="+", default=["cub200", "cifar100"])
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--data-root", default=str(DEFAULT_DATA))
    ap.add_argument(
        "--cub-cache",
        default=None,
        help="optional existing CUB-200 cache dir with train.pt/test.pt (e.g. the core patch-mean cache)",
    )
    ap.add_argument(
        "--cifar100-cache",
        default=None,
        help="optional existing CIFAR-100 cache dir with train.pt/test.pt",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", choices=["dinov2", "clip", "hf_vision"], default="dinov2")
    ap.add_argument("--model-path", default=str(MODEL_PATH))
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resize-size", type=int, default=256)
    ap.add_argument("--crop-size", type=int, default=224)
    ap.add_argument("--crop-mode", choices=["center", "square"], default="center")
    ap.add_argument("--torchvision-only", action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--force-prepare", action="store_true")
    ap.add_argument("--no-cil", action="store_true")
    args = ap.parse_args()

    if args.mode == "p0":
        run_p0(args)
    elif args.mode == "dataset":
        if not args.dataset:
            raise ValueError("--dataset is required for mode=dataset")
        run_dataset(args.dataset, args)
    elif args.mode == "prepare_hf":
        prepare_hf_datasets(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
