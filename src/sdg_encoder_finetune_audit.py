#!/usr/bin/env python3
"""Audit of SDG after fine-tuning the DINOv2-B encoder from raw images (Appendix K).

With `--finetune-scope last_block` (experiment_id E11 in the outputs) the audit
updates only the last transformer block and final normalization layer of
DINOv2-B using raw training images; `--finetune-scope full_encoder` (E14)
updates the full encoder under the same protocol.  A temporary linear head
selects the epoch budget on a stratified train-only validation split.  The
model is then reinitialized and retrained on the selected train budget, the
head is discarded, and Cos/WCos/LDA are compared on the resulting CLS
features.  Test labels are used only for final metrics and paired uncertainty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sdg_paths import DATA_ROOT, MODEL_ROOT
from sdg_audit_core import (
    REG_GRID,
    accuracy,
    cosine_predictions,
    environment_manifest,
    paired_bootstrap_difference,
    select_then_refit_evaluator,
    write_json_new,
    write_text_new,
)


DATASET_LABELS = {
    "cub200": "CUB-200",
    "aircraft": "FGVC Aircraft",
    "cifar100": "CIFAR-100",
}
METHODS = ("cos", "wcos_tv", "lda_tv", "validation_whitening")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def stratified_train_validation_indices(
    labels: np.ndarray,
    *,
    seed: int,
    validation_fraction: float,
    max_total_per_class: int | None,
) -> dict[str, np.ndarray]:
    """Return disjoint fit/validation indices without consulting test data."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("invalid labels or validation fraction")
    rng = np.random.RandomState(seed)
    fit, validation, selected, ignored = [], [], [], []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        indices = indices[rng.permutation(len(indices))]
        if max_total_per_class is not None and len(indices) > max_total_per_class:
            ignored.extend(indices[max_total_per_class:].tolist())
            indices = indices[:max_total_per_class]
        if len(indices) < 2:
            raise ValueError(f"class {label} has fewer than two selected examples")
        n_validation = int(round(len(indices) * validation_fraction))
        n_validation = max(1, min(n_validation, len(indices) - 1))
        validation.extend(indices[:n_validation].tolist())
        fit.extend(indices[n_validation:].tolist())
        selected.extend(indices.tolist())
    result = {
        "fit": np.asarray(sorted(fit), dtype=np.int64),
        "validation": np.asarray(sorted(validation), dtype=np.int64),
        "selected": np.asarray(sorted(selected), dtype=np.int64),
        "ignored": np.asarray(sorted(ignored), dtype=np.int64),
    }
    if set(result["fit"].tolist()) & set(result["validation"].tolist()):
        raise AssertionError("fit and validation indices overlap")
    if set(result["fit"].tolist()) | set(result["validation"].tolist()) != set(
        result["selected"].tolist()
    ):
        raise AssertionError("selected indices do not equal fit union validation")
    for key in ("fit", "validation"):
        if set(np.unique(labels[result[key]]).tolist()) != set(np.unique(labels).tolist()):
            raise AssertionError(f"{key} split does not cover every class")
    return result


class CUBRawDataset(Dataset):
    def __init__(self, root: Path, split: str):
        base = root
        if not (base / "images.txt").exists():
            base = root / "CUB_200_2011"
        required = [
            base / "images.txt",
            base / "train_test_split.txt",
            base / "image_class_labels.txt",
        ]
        if not all(path.exists() for path in required):
            raise FileNotFoundError(f"invalid CUB root: {root}")
        images = {
            int(line.split(" ", 1)[0]): line.split(" ", 1)[1]
            for line in required[0].read_text().splitlines()
        }
        is_train = {
            int(line.split(" ", 1)[0]): int(line.split(" ", 1)[1])
            for line in required[1].read_text().splitlines()
        }
        labels = {
            int(line.split(" ", 1)[0]): int(line.split(" ", 1)[1]) - 1
            for line in required[2].read_text().splitlines()
        }
        wanted = 1 if split == "train" else 0
        self.items = [
            (base / "images" / images[index], labels[index])
            for index in sorted(images)
            if is_train[index] == wanted
        ]
        self.labels = np.asarray([label for _, label in self.items], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        return Image.open(path).convert("RGB"), int(label)


class HFRawDataset(Dataset):
    def __init__(self, path: Path):
        from datasets import load_from_disk

        if not path.exists():
            raise FileNotFoundError(path)
        self.dataset = load_from_disk(str(path))
        self.image_key = "image" if "image" in self.dataset.column_names else "img"
        self.labels = np.asarray(self.dataset["label"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[int(index)]
        return row[self.image_key].convert("RGB"), int(row["label"])


class TransformedSubset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: Iterable[int],
        transform,
        label_mapping: dict[int, int],
    ):
        self.dataset = dataset
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.transform = transform
        self.label_mapping = label_mapping

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.dataset[int(self.indices[index])]
        return self.transform(image), self.label_mapping[int(label)]


def load_raw_dataset(
    dataset: str,
    split: str,
    *,
    data_root: Path,
    cub_root: Path,
) -> Dataset:
    if dataset == "cub200":
        return CUBRawDataset(cub_root, split)
    return HFRawDataset(data_root / dataset / "hf_dataset" / split)


def _size_pair(value, default: int) -> tuple[int, int]:
    if isinstance(value, dict):
        if "height" in value and "width" in value:
            return int(value["height"]), int(value["width"])
        if "shortest_edge" in value:
            edge = int(value["shortest_edge"])
            return edge, edge
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if value is not None:
        edge = int(value)
        return edge, edge
    return default, default


def build_transforms(model_path: Path):
    import torchvision.transforms as T
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(str(model_path))
    crop_h, crop_w = _size_pair(getattr(processor, "crop_size", None), 224)
    resize_h, resize_w = _size_pair(
        getattr(processor, "size", None), max(crop_h, crop_w)
    )
    resize_h, resize_w = max(resize_h, crop_h), max(resize_w, crop_w)
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    train = T.Compose(
        [
            T.RandomResizedCrop(
                (crop_h, crop_w),
                scale=(0.70, 1.0),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    evaluation = T.Compose(
        [
            T.Resize((resize_h, resize_w), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop((crop_h, crop_w)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return train, evaluation


def load_dinov2(model_path: Path, device: torch.device) -> nn.Module:
    from transformers import AutoModel

    return AutoModel.from_pretrained(
        str(model_path), trust_remote_code=True
    ).to(device)


def _last_transformer_block(model: nn.Module) -> nn.Module:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise ValueError("model has no encoder")
    layers = getattr(encoder, "layer", None)
    if layers is None:
        layers = getattr(encoder, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("cannot locate transformer layers")
    return layers[-1]


def _trainable_scope_summary(model: nn.Module, updated_modules: list[str]) -> dict:
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names:
        raise AssertionError("fine-tuning scope is empty")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "trainable_names": trainable_names,
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_fraction": float(trainable / total),
        "updated_modules": updated_modules,
    }


def configure_last_block_finetuning(model: nn.Module) -> dict:
    """Freeze the encoder except its final block and output normalization."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = [_last_transformer_block(model)]
    for name in ("layernorm", "norm", "post_layernorm"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module) and module not in modules:
            modules.append(module)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return _trainable_scope_summary(
        model,
        ["last_transformer_block", "final_output_normalization"],
    )


def configure_full_encoder_finetuning(model: nn.Module) -> dict:
    """Make every encoder parameter trainable."""

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return _trainable_scope_summary(model, ["full_encoder"])


def configure_finetuning(model: nn.Module, scope: str) -> dict:
    if scope == "last_block":
        return configure_last_block_finetuning(model)
    if scope == "full_encoder":
        return configure_full_encoder_finetuning(model)
    raise ValueError(f"unsupported fine-tuning scope: {scope}")


def finetune_experiment_metadata(scope: str) -> dict:
    if scope == "last_block":
        return {
            "experiment_id": "E11",
            "kind": "post_submission_true_encoder_last_block_finetune_audit",
            "seed_kind": "post_submission_true_encoder_last_block_finetune_seed_audit",
            "updated_modules": [
                "last transformer block",
                "final output normalization",
            ],
            "report_description": (
                "DINOv2-B last-block + final-layernorm fine-tuning on raw images; "
                "the temporary classification head is discarded before evaluator audit."
            ),
            "scope": (
                "Raw-image DINOv2-B fine-tuning of the final transformer block and "
                "final normalization layer; temporary classifier head discarded before "
                "Cos/WCos/LDA evaluation. This is a stress test, not a universal claim "
                "about arbitrary end-to-end training."
            ),
        }
    if scope == "full_encoder":
        return {
            "experiment_id": "E14",
            "kind": "post_submission_true_encoder_full_finetune_audit",
            "seed_kind": "post_submission_true_encoder_full_finetune_seed_audit",
            "updated_modules": ["full encoder"],
            "report_description": (
                "DINOv2-B full-encoder fine-tuning on raw images; the temporary "
                "classification head is discarded before evaluator audit."
            ),
            "scope": (
                "Raw-image DINOv2-B full-encoder fine-tuning under the fixed E11 "
                "data, budget, optimizer, epoch-selection, and evaluator protocol; "
                "temporary classifier head discarded before Cos/WCos/LDA evaluation. "
                "This is one controlled full-encoder stress test, not a universal "
                "claim about arbitrary training recipes."
            ),
        }
    raise ValueError(f"unsupported fine-tuning scope: {scope}")


def forward_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(pixel_values=images)
    hidden = output.last_hidden_state
    if hidden.ndim != 3 or hidden.shape[1] < 1:
        raise ValueError(f"invalid model output shape: {tuple(hidden.shape)}")
    return hidden[:, 0, :]


def parameter_digest(model: nn.Module, *, trainable: bool) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if bool(parameter.requires_grad) != bool(trainable):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_loader(
    dataset: Dataset,
    indices: Iterable[int],
    transform,
    label_mapping: dict[int, int],
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TransformedSubset(dataset, indices, transform, label_mapping),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )


def _autocast(enabled: bool):
    return torch.cuda.amp.autocast(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    *,
    amp: bool,
    gradient_clip: float,
) -> dict:
    model.train()
    head.train()
    loss_sum = 0.0
    correct = 0
    seen = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(amp):
            logits = head(forward_features(model, images))
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ],
            gradient_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach().cpu()) * len(labels)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        seen += len(labels)
    return {
        "loss": loss_sum / max(seen, 1),
        "accuracy": correct / max(seen, 1),
        "n": seen,
    }


@torch.no_grad()
def head_accuracy(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
) -> float:
    model.eval()
    head.eval()
    correct = 0
    seen = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast(amp):
            predictions = head(forward_features(model, images)).argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        seen += len(labels)
    return correct / max(seen, 1)


def build_optimizer(
    model: nn.Module,
    head: nn.Module,
    *,
    encoder_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    encoder_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    return torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": encoder_learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": list(head.parameters()),
                "lr": head_learning_rate,
                "weight_decay": weight_decay,
            },
        ]
    )


def select_epoch(
    *,
    model_path: Path,
    train_dataset: Dataset,
    split: dict[str, np.ndarray],
    train_transform,
    evaluation_transform,
    label_mapping: dict[int, int],
    hidden_size: int,
    args: argparse.Namespace,
    effective_seed: int,
    device: torch.device,
) -> dict:
    seed_everything(effective_seed)
    model = load_dinov2(model_path, device)
    scope = configure_finetuning(model, args.finetune_scope)
    head = nn.Linear(hidden_size, len(label_mapping)).to(device)
    optimizer = build_optimizer(
        model,
        head,
        encoder_learning_rate=args.encoder_learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
    )
    amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    train_loader = make_loader(
        train_dataset,
        split["fit"],
        train_transform,
        label_mapping,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=effective_seed,
    )
    validation_loader = make_loader(
        train_dataset,
        split["validation"],
        evaluation_transform,
        label_mapping,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        seed=effective_seed + 1,
    )
    history = []
    best_epoch = None
    best_accuracy = -math.inf
    for epoch in range(1, args.max_epochs + 1):
        train_row = train_one_epoch(
            model,
            head,
            train_loader,
            optimizer,
            scaler,
            device,
            amp=amp,
            gradient_clip=args.gradient_clip,
        )
        validation_accuracy = head_accuracy(
            model, head, validation_loader, device, amp=amp
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_row["loss"],
                "train_accuracy": train_row["accuracy"],
                "validation_head_accuracy": validation_accuracy,
            }
        )
        print(
            f"[epoch-select] epoch={epoch}/{args.max_epochs} "
            f"loss={train_row['loss']:.5f} "
            f"train={train_row['accuracy']:.4f} val={validation_accuracy:.4f}",
            flush=True,
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
    if best_epoch is None:
        raise AssertionError("epoch selection failed")
    del model, head, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "selected_epoch": int(best_epoch),
        "selected_validation_head_accuracy": float(best_accuracy),
        "history": history,
        "selection_uses": "stratified training validation only",
        "scope": scope,
    }


def fit_selected_budget(
    *,
    model_path: Path,
    train_dataset: Dataset,
    selected_indices: np.ndarray,
    train_transform,
    label_mapping: dict[int, int],
    hidden_size: int,
    selected_epoch: int,
    args: argparse.Namespace,
    effective_seed: int,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, dict]:
    seed_everything(effective_seed)
    model = load_dinov2(model_path, device)
    scope = configure_finetuning(model, args.finetune_scope)
    head = nn.Linear(hidden_size, len(label_mapping)).to(device)
    optimizer = build_optimizer(
        model,
        head,
        encoder_learning_rate=args.encoder_learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
    )
    amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loader = make_loader(
        train_dataset,
        selected_indices,
        train_transform,
        label_mapping,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=effective_seed,
    )
    frozen_before = parameter_digest(model, trainable=False)
    trainable_before = parameter_digest(model, trainable=True)
    history = []
    for epoch in range(1, selected_epoch + 1):
        row = train_one_epoch(
            model,
            head,
            loader,
            optimizer,
            scaler,
            device,
            amp=amp,
            gradient_clip=args.gradient_clip,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": row["loss"],
                "train_accuracy": row["accuracy"],
            }
        )
        print(
            f"[refit] epoch={epoch}/{selected_epoch} "
            f"loss={row['loss']:.5f} train={row['accuracy']:.4f}",
            flush=True,
        )
    frozen_after = parameter_digest(model, trainable=False)
    trainable_after = parameter_digest(model, trainable=True)
    if frozen_before != frozen_after:
        raise AssertionError("a frozen encoder parameter changed during fine-tuning")
    if trainable_before == trainable_after:
        raise AssertionError("trainable encoder parameters did not change")
    return model, head, {
        "history": history,
        "scope": scope,
        "frozen_parameter_sha256_before": frozen_before,
        "frozen_parameter_sha256_after": frozen_after,
        "trainable_parameter_sha256_before": trainable_before,
        "trainable_parameter_sha256_after": trainable_after,
    }


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataset: Dataset,
    transform,
    label_mapping: dict[int, int],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(
        dataset,
        np.arange(len(dataset)),
        transform,
        label_mapping,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        seed=0,
    )
    model.eval()
    features, labels = [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        with _autocast(amp):
            batch = forward_features(model, images)
        features.append(batch.float().cpu().numpy())
        labels.append(target.numpy().astype(np.int64))
    output = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    targets = np.concatenate(labels, axis=0)
    if output.ndim != 2 or len(output) != len(targets) or not np.isfinite(output).all():
        raise ValueError("invalid extracted features")
    return output, targets


def evaluate_geometry(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    validation_seed: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    split = stratified_train_validation_indices(
        train_labels,
        seed=validation_seed,
        validation_fraction=0.2,
        max_total_per_class=None,
    )
    classes = sorted(set(train_labels.tolist()) & set(test_labels.tolist()))
    inner_features = np.asarray(train_features[split["fit"]], dtype=np.float64)
    inner_labels = train_labels[split["fit"]]
    validation_features = np.asarray(
        train_features[split["validation"]], dtype=np.float64
    )
    validation_labels = train_labels[split["validation"]]
    train64 = np.asarray(train_features, dtype=np.float64)
    test64 = np.asarray(test_features, dtype=np.float64)
    cosine = cosine_predictions(train64, train_labels, test64, classes)
    wcos = select_then_refit_evaluator(
        "wcos",
        train64,
        train_labels,
        inner_features,
        inner_labels,
        validation_features,
        validation_labels,
        test64,
        classes,
        REG_GRID,
        regularization_mode="absolute",
    )
    lda = select_then_refit_evaluator(
        "lda",
        train64,
        train_labels,
        inner_features,
        inner_labels,
        validation_features,
        validation_labels,
        test64,
        classes,
        REG_GRID,
        regularization_mode="absolute",
    )
    if max(wcos["validation_scores"]) >= max(lda["validation_scores"]):
        selected_family = "wcos"
        selected_predictions = wcos["predictions"]
    else:
        selected_family = "lda"
        selected_predictions = lda["predictions"]
    predictions = {
        "cos": cosine,
        "wcos_tv": wcos["predictions"],
        "lda_tv": lda["predictions"],
        "validation_whitening": selected_predictions,
    }
    result = {
        "validation_whitening_family": selected_family,
        "selection": {
            "wcos_tv": {key: value for key, value in wcos.items() if key != "predictions"},
            "lda_tv": {key: value for key, value in lda.items() if key != "predictions"},
        },
        "accuracies": {
            method: accuracy(prediction, test_labels)
            for method, prediction in predictions.items()
        },
        "gaps_pp": {
            method: 100.0
            * (
                accuracy(predictions[method], test_labels)
                - accuracy(predictions["cos"], test_labels)
            )
            for method in METHODS
            if method != "cos"
        },
        "validation_split": {
            "seed": validation_seed,
            "n_fit": int(len(split["fit"])),
            "n_validation": int(len(split["validation"])),
            "test_labels_used": False,
        },
    }
    return result, predictions


def paired_gap_change_bootstrap(
    adapted_whitening: np.ndarray,
    adapted_cosine: np.ndarray,
    frozen_whitening: np.ndarray,
    frozen_cosine: np.ndarray,
    labels: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    arrays = [
        np.asarray(adapted_whitening),
        np.asarray(adapted_cosine),
        np.asarray(frozen_whitening),
        np.asarray(frozen_cosine),
        np.asarray(labels),
    ]
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("paired arrays have different lengths")
    adapted_delta = (
        (arrays[0] == arrays[4]).astype(np.float64)
        - (arrays[1] == arrays[4]).astype(np.float64)
    )
    frozen_delta = (
        (arrays[2] == arrays[4]).astype(np.float64)
        - (arrays[3] == arrays[4]).astype(np.float64)
    )
    contrast = 100.0 * (adapted_delta - frozen_delta)
    rng = np.random.RandomState(seed)
    bootstrap = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sample = rng.randint(0, len(labels), size=len(labels))
        bootstrap[index] = contrast[sample].mean()
    return {
        "point_pp": float(contrast.mean()),
        "ci95_pp": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "n_test": int(len(labels)),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def class_cluster_gap_bootstrap(
    left_predictions: np.ndarray,
    right_predictions: np.ndarray,
    labels: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    labels = np.asarray(labels)
    delta = 100.0 * (
        (np.asarray(left_predictions) == labels).astype(np.float64)
        - (np.asarray(right_predictions) == labels).astype(np.float64)
    )
    classes = np.unique(labels)
    class_means = np.asarray([delta[labels == label].mean() for label in classes])
    rng = np.random.RandomState(seed)
    bootstrap = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        bootstrap[index] = class_means[
            rng.randint(0, len(classes), size=len(classes))
        ].mean()
    return {
        "point_pp": float(class_means.mean()),
        "ci95_pp": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "n_classes": int(len(classes)),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def seed_summary(values: list[float], *, seed: int, n_boot: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)
    bootstrap = np.asarray(
        [
            array[rng.randint(0, len(array), size=len(array))].mean()
            for _ in range(n_boot)
        ]
    )
    return {
        "values_pp": array.tolist(),
        "n_seeds": int(len(array)),
        "mean_pp": float(array.mean()),
        "std_pp": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "bootstrap_ci95_pp": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "positive_seeds": int(np.sum(array > 0)),
        "zero_seeds": int(np.sum(array == 0)),
        "negative_seeds": int(np.sum(array < 0)),
    }


def _write_npz_new(path: Path, arrays: dict[str, np.ndarray]) -> dict:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mktemp(suffix=".npz"))
    np.savez(temporary, **arrays)
    shutil.copy2(temporary, path)
    temporary.unlink(missing_ok=True)
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "arrays": sorted(arrays),
    }


def _path_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def dataset_report(payload: dict) -> str:
    lines = [
        (
            f"# {payload['experiment_id']} true encoder fine-tuning: "
            f"{payload['dataset_label']}"
        ),
        "",
        payload["report_description"],
        "",
        "| seed | epoch | head test | Cos | WCos-TV | LDA-TV | WVal | WVal−Cos | Δ gap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["seeds"]:
        metrics = row["finetuned_evaluator"]["accuracies"]
        gaps = row["finetuned_evaluator"]["gaps_pp"]
        lines.append(
            f"| {row['seed']} | {row['epoch_selection']['selected_epoch']} | "
            f"{100*row['head_test_accuracy']:.2f} | {100*metrics['cos']:.2f} | "
            f"{100*metrics['wcos_tv']:.2f} | {100*metrics['lda_tv']:.2f} | "
            f"{100*metrics['validation_whitening']:.2f} | "
            f"{gaps['validation_whitening']:+.2f} | "
            f"{row['gap_change_bootstrap']['validation_whitening']['point_pp']:+.2f} |"
        )
    summary = payload["seed_summary"]["validation_whitening"]
    lines.extend(
        [
            "",
            (
                f"Fine-tuned WVal−Cos mean gap: "
                f"{summary['finetuned_gap_pp']['mean_pp']:+.3f} pp "
                f"(seed bootstrap CI {summary['finetuned_gap_pp']['bootstrap_ci95_pp']})."
            ),
            (
                f"Mean change from matched frozen gap: "
                f"{summary['gap_change_pp']['mean_pp']:+.3f} pp "
                f"(seed bootstrap CI {summary['gap_change_pp']['bootstrap_ci95_pp']})."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_dataset(args: argparse.Namespace) -> dict:
    if args.dataset not in DATASET_LABELS:
        raise ValueError(args.dataset)
    experiment = finetune_experiment_metadata(args.finetune_scope)
    output_root = Path(args.output_root)
    dataset_root = output_root / args.dataset
    summary_path = dataset_root / "result.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    model_path = Path(args.model_path)
    data_root = Path(args.data_root)
    cub_root = Path(args.cub_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(f"{experiment['experiment_id']} production audit requires CUDA")
    train_dataset = load_raw_dataset(
        args.dataset, "train", data_root=data_root, cub_root=cub_root
    )
    test_dataset = load_raw_dataset(
        args.dataset, "test", data_root=data_root, cub_root=cub_root
    )
    train_labels = np.asarray(train_dataset.labels, dtype=np.int64)
    test_labels_original = np.asarray(test_dataset.labels, dtype=np.int64)
    classes = sorted(set(train_labels.tolist()) & set(test_labels_original.tolist()))
    if set(train_labels.tolist()) != set(test_labels_original.tolist()):
        raise ValueError("train/test class sets do not match")
    label_mapping = {label: index for index, label in enumerate(classes)}
    train_transform, evaluation_transform = build_transforms(model_path)

    seed_everything(args.base_seed)
    frozen_model = load_dinov2(model_path, device)
    frozen_model.eval()
    hidden_size = int(getattr(frozen_model.config, "hidden_size"))
    frozen_train, frozen_train_labels = extract_features(
        frozen_model,
        train_dataset,
        evaluation_transform,
        label_mapping,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=device,
        amp=bool(args.amp and device.type == "cuda"),
    )
    frozen_test, frozen_test_labels = extract_features(
        frozen_model,
        test_dataset,
        evaluation_transform,
        label_mapping,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=device,
        amp=bool(args.amp and device.type == "cuda"),
    )
    del frozen_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    frozen_result, frozen_predictions = evaluate_geometry(
        frozen_train,
        frozen_train_labels,
        frozen_test,
        frozen_test_labels,
        validation_seed=args.evaluator_validation_seed,
    )

    seed_rows = []
    for seed in args.seeds:
        seed_root = dataset_root / f"seed_{seed}"
        seed_path = seed_root / "result.json"
        if seed_path.exists():
            row = json.loads(seed_path.read_text())
            if row.get("seed") != seed or row.get("dataset") != args.dataset:
                raise ValueError(f"invalid resume artifact: {seed_path}")
            existing_scope = row.get("fine_tune_scope")
            if (
                existing_scope is None
                and row.get("kind")
                == "post_submission_true_encoder_last_block_finetune_seed_audit"
            ):
                existing_scope = "last_block"
            if existing_scope != args.finetune_scope:
                raise ValueError(
                    f"fine-tuning scope mismatch in resume artifact: {seed_path}"
                )
            seed_rows.append(row)
            print(f"[resume] verified existing seed artifact {seed_path}", flush=True)
            continue
        effective_seed = args.base_seed + 1000003 * seed
        split = stratified_train_validation_indices(
            train_labels,
            seed=effective_seed,
            validation_fraction=args.validation_fraction,
            max_total_per_class=args.max_train_per_class,
        )
        selection = select_epoch(
            model_path=model_path,
            train_dataset=train_dataset,
            split=split,
            train_transform=train_transform,
            evaluation_transform=evaluation_transform,
            label_mapping=label_mapping,
            hidden_size=hidden_size,
            args=args,
            effective_seed=effective_seed,
            device=device,
        )
        model, head, refit = fit_selected_budget(
            model_path=model_path,
            train_dataset=train_dataset,
            selected_indices=split["selected"],
            train_transform=train_transform,
            label_mapping=label_mapping,
            hidden_size=hidden_size,
            selected_epoch=selection["selected_epoch"],
            args=args,
            effective_seed=effective_seed,
            device=device,
        )
        evaluation_train, evaluation_train_labels = extract_features(
            model,
            train_dataset,
            evaluation_transform,
            label_mapping,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            device=device,
            amp=bool(args.amp and device.type == "cuda"),
        )
        evaluation_test, evaluation_test_labels = extract_features(
            model,
            test_dataset,
            evaluation_transform,
            label_mapping,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            device=device,
            amp=bool(args.amp and device.type == "cuda"),
        )
        if not np.array_equal(evaluation_train_labels, frozen_train_labels):
            raise AssertionError("fine-tuned and frozen train example axes differ")
        if not np.array_equal(evaluation_test_labels, frozen_test_labels):
            raise AssertionError("fine-tuned and frozen test example axes differ")
        test_loader = make_loader(
            test_dataset,
            np.arange(len(test_dataset)),
            evaluation_transform,
            label_mapping,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            seed=effective_seed,
        )
        test_head_accuracy = head_accuracy(
            model,
            head,
            test_loader,
            device,
            amp=bool(args.amp and device.type == "cuda"),
        )
        finetuned_result, finetuned_predictions = evaluate_geometry(
            evaluation_train,
            evaluation_train_labels,
            evaluation_test,
            evaluation_test_labels,
            validation_seed=args.evaluator_validation_seed,
        )
        paired = {}
        gap_change = {}
        class_cluster = {}
        for method in METHODS:
            if method == "cos":
                continue
            method_index = METHODS.index(method)
            paired[method] = paired_bootstrap_difference(
                finetuned_predictions[method],
                finetuned_predictions["cos"],
                evaluation_test_labels,
                n_boot=args.test_bootstrap,
                seed=args.bootstrap_seed + seed * 1009 + method_index * 7919,
            )
            gap_change[method] = paired_gap_change_bootstrap(
                finetuned_predictions[method],
                finetuned_predictions["cos"],
                frozen_predictions[method],
                frozen_predictions["cos"],
                evaluation_test_labels,
                n_boot=args.test_bootstrap,
                seed=args.bootstrap_seed + seed * 2029 + method_index * 9176,
            )
            class_cluster[method] = class_cluster_gap_bootstrap(
                finetuned_predictions[method],
                finetuned_predictions["cos"],
                evaluation_test_labels,
                n_boot=args.test_bootstrap,
                seed=args.bootstrap_seed + seed * 4099 + method_index * 3253,
            )
        prediction_artifact = _write_npz_new(
            seed_root / "predictions.npz",
            {
                "labels": evaluation_test_labels.astype(np.int16),
                **{
                    f"frozen__{method}": prediction.astype(np.int16)
                    for method, prediction in frozen_predictions.items()
                },
                **{
                    f"finetuned__{method}": prediction.astype(np.int16)
                    for method, prediction in finetuned_predictions.items()
                },
            },
        )
        row = {
            "schema_version": 1,
            "kind": experiment["seed_kind"],
            "dataset": args.dataset,
            "dataset_label": DATASET_LABELS[args.dataset],
            "fine_tune_scope": args.finetune_scope,
            "seed": int(seed),
            "effective_seed": int(effective_seed),
            "epoch_selection": selection,
            "refit": refit,
            "head_test_accuracy": float(test_head_accuracy),
            "head_test_labels_used_for_selection": False,
            "finetuned_evaluator": finetuned_result,
            "paired_test_bootstrap": paired,
            "gap_change_bootstrap": gap_change,
            "class_cluster_bootstrap": class_cluster,
            "fine_tune_split": {
                "n_fit": int(len(split["fit"])),
                "n_validation": int(len(split["validation"])),
                "n_selected": int(len(split["selected"])),
                "n_ignored_by_budget": int(len(split["ignored"])),
                "validation_fraction": args.validation_fraction,
                "max_total_per_class": args.max_train_per_class,
                "test_labels_used": False,
            },
            "artifacts": {"predictions.npz": prediction_artifact},
            "environment": environment_manifest(args.commit),
        }
        write_json_new(seed_path, row)
        seed_rows.append(row)
        del model, head, evaluation_train, evaluation_test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summaries = {}
    for method in METHODS:
        if method == "cos":
            continue
        finetuned_gaps = [
            row["finetuned_evaluator"]["gaps_pp"][method] for row in seed_rows
        ]
        frozen_gap = float(frozen_result["gaps_pp"][method])
        gap_changes = [value - frozen_gap for value in finetuned_gaps]
        summaries[method] = {
            "frozen_gap_pp": frozen_gap,
            "finetuned_gap_pp": seed_summary(
                finetuned_gaps,
                seed=args.bootstrap_seed + METHODS.index(method),
                n_boot=args.seed_bootstrap,
            ),
            "gap_change_pp": seed_summary(
                gap_changes,
                seed=args.bootstrap_seed + 10000 + METHODS.index(method),
                n_boot=args.seed_bootstrap,
            ),
        }
    payload = {
        "schema_version": 1,
        "kind": experiment["kind"],
        "experiment_id": experiment["experiment_id"],
        "dataset": args.dataset,
        "dataset_label": DATASET_LABELS[args.dataset],
        "fine_tune_scope": args.finetune_scope,
        "scope": experiment["scope"],
        "report_description": experiment["report_description"],
        "protocol": {
            "backbone": "DINOv2-B with registers",
            "updated_modules": experiment["updated_modules"],
            "encoder_learning_rate": args.encoder_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "validation_fraction": args.validation_fraction,
            "max_train_per_class": args.max_train_per_class,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "seeds": args.seeds,
            "regularization_grid": REG_GRID,
            "regularization_mode": "absolute",
            "test_bootstrap": args.test_bootstrap,
            "seed_bootstrap": args.seed_bootstrap,
        },
        "selection_guards": {
            "epoch_selection": "stratified original-train validation only",
            "refit": "fresh pretrained initialization on selected original-train budget",
            "evaluator_selection": "stratified original-train validation only",
            "test_labels": "final head metric, evaluator metrics, and uncertainty only",
            "classification_head": "discarded before evaluator audit",
        },
        "data": {
            "n_train": int(len(train_dataset)),
            "n_test": int(len(test_dataset)),
            "n_classes": int(len(classes)),
            "feature_dim": int(hidden_size),
            "model_config": _path_identity(model_path / "config.json"),
            "model_weights": _path_identity(model_path / "model.safetensors"),
            "train_source": str(
                cub_root
                if args.dataset == "cub200"
                else data_root / args.dataset / "hf_dataset" / "train"
            ),
            "test_source": str(
                cub_root
                if args.dataset == "cub200"
                else data_root / args.dataset / "hf_dataset" / "test"
            ),
        },
        "frozen_baseline": frozen_result,
        "seeds": seed_rows,
        "seed_summary": summaries,
        "environment": environment_manifest(args.commit),
    }
    write_json_new(summary_path, payload)
    write_text_new(dataset_root / "report.md", dataset_report(payload))
    print(
        f"[{experiment['experiment_id']}] complete "
        f"dataset={args.dataset} output={dataset_root}",
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_LABELS))
    parser.add_argument(
        "--model-path",
        default=str(MODEL_ROOT / "dinov2-with-registers-base"),
        help="local snapshot of facebook/dinov2-with-registers-base",
    )
    parser.add_argument(
        "--data-root",
        default=str(DATA_ROOT),
        help="prepared datasets as <data-root>/<dataset>/hf_dataset/{train,test}",
    )
    parser.add_argument(
        "--cub-root",
        default=str(DATA_ROOT / "CUB_200_2011"),
        help="official CUB_200_2011 directory (images.txt, train_test_split.txt, images/)",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--finetune-scope",
        choices=["last_block", "full_encoder"],
        default="last_block",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--base-seed", type=int, default=20260724)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-per-class", type=int, default=100)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--evaluator-validation-seed", type=int, default=1701)
    parser.add_argument("--test-bootstrap", type=int, default=4000)
    parser.add_argument("--seed-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026072401)
    parser.add_argument("--commit")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dataset(args)


if __name__ == "__main__":
    main()
