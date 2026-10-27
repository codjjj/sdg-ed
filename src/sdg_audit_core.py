#!/usr/bin/env python3
"""Shared, train-only evaluation primitives for the additional SDG analyses.

This module is intentionally separate from the core evaluation pipeline.  It
only reads cached features and exposes prediction-level functions so every
additional analysis can retain paired uncertainty and explicit model-
selection provenance.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


REG_GRID = [
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
]
RIDGE_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def write_json_new(path: Path, payload) -> None:
    """Atomically create a JSON artifact and refuse accidental overwrite."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(suffix=".json"))
    tmp.write_text(json.dumps(payload, indent=2, default=float, allow_nan=False) + "\n")
    shutil.copy2(tmp, path)
    tmp.unlink(missing_ok=True)


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mktemp(suffix=path.suffix))
    tmp.write_text(text)
    shutil.copy2(tmp, path)
    tmp.unlink(missing_ok=True)


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = payload["pooled"].float().numpy().astype(np.float64)
    labels = payload["labels"].long().numpy()
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError(f"invalid feature cache shapes at {path}: {features.shape}, {labels.shape}")
    if not np.isfinite(features).all():
        raise FloatingPointError(f"non-finite cached features at {path}")
    return features, labels


def row_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def classes_from(train_labels: np.ndarray, eval_labels: np.ndarray) -> list[int]:
    return sorted(set(train_labels.tolist()) & set(eval_labels.tolist()))


def cosine_predictions(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    classes: Iterable[int],
    *,
    normalize_before_mean: bool = False,
) -> np.ndarray:
    """Cosine NCM with explicit prototype-construction semantics."""

    classes = list(classes)
    source = row_normalize(train_features) if normalize_before_mean else np.asarray(train_features, dtype=np.float64)
    prototypes = np.stack([source[train_labels == cls].mean(0) for cls in classes])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = row_normalize(eval_features) @ row_normalize(prototypes).T
    if not np.isfinite(scores).all():
        raise FloatingPointError("non-finite cosine scores")
    indices = np.argmax(scores, axis=1)
    return np.asarray([classes[int(index)] for index in indices])


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(predictions) == np.asarray(labels)))


def covariance_scale(covariance: np.ndarray) -> float:
    scale = float(np.trace(covariance) / covariance.shape[0])
    return max(scale, 1e-12)


def sorted_eigh(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.asarray(covariance, dtype=np.float64)
    if not np.isfinite(covariance).all():
        raise FloatingPointError("non-finite covariance before eigendecomposition")
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    return np.maximum(values[order], 0.0), vectors[:, order]


def pooled_within_stats(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    classes: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    classes = list(classes)
    dim = train_features.shape[1]
    scatter = np.zeros((dim, dim), dtype=np.float64)
    means = []
    n_total = 0
    for cls in classes:
        subset = train_features[train_labels == cls]
        mean = subset.mean(0)
        means.append(mean)
        centered = subset - mean
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scatter += centered.T @ centered
        n_total += len(subset)
    if not np.isfinite(scatter).all():
        raise FloatingPointError("non-finite pooled within-class scatter")
    covariance = scatter / max(1, n_total - len(classes))
    return np.stack(means), covariance


def total_covariance_stats(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(0)
    centered = train_features - mean
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        covariance = centered.T @ centered / max(1, len(centered) - 1)
    if not np.isfinite(covariance).all():
        raise FloatingPointError("non-finite total covariance")
    return mean, covariance


def resolved_regularizers(candidates: Iterable[float], covariance: np.ndarray, mode: str) -> list[float]:
    base = [float(value) for value in candidates]
    if mode == "absolute":
        return base
    if mode == "relative_trace":
        scale = covariance_scale(covariance)
        return [value * scale for value in base]
    raise ValueError(f"unknown regularization mode: {mode}")


def lda_predictions_from_stats(
    means: np.ndarray,
    covariance: np.ndarray,
    eval_features: np.ndarray,
    classes: Iterable[int],
    regularizer: float,
) -> np.ndarray:
    values, vectors = sorted_eigh(covariance)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        inverse = (vectors * (1.0 / (values + regularizer))) @ vectors.T
        weights = means @ inverse
    bias = -0.5 * np.sum(means * weights, axis=1)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = eval_features @ weights.T + bias[None, :]
    if not np.isfinite(weights).all() or not np.isfinite(scores).all():
        raise FloatingPointError("non-finite LDA weights or scores")
    indices = np.argmax(scores, axis=1)
    classes = list(classes)
    return np.asarray([classes[int(index)] for index in indices])


def lda_grid_predictions(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    classes: Iterable[int],
    candidates: Iterable[float],
    *,
    regularization_mode: str,
) -> tuple[list[np.ndarray], list[float]]:
    means, covariance = pooled_within_stats(train_features, train_labels, classes)
    values, vectors = sorted_eigh(covariance)
    regs = resolved_regularizers(candidates, covariance, regularization_mode)
    predictions = []
    classes = list(classes)
    for regularizer in regs:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            inverse = (vectors * (1.0 / (values + regularizer))) @ vectors.T
            weights = means @ inverse
        bias = -0.5 * np.sum(means * weights, axis=1)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = eval_features @ weights.T + bias[None, :]
        if not np.isfinite(weights).all() or not np.isfinite(scores).all():
            raise FloatingPointError("non-finite LDA grid weights or scores")
        indices = np.argmax(scores, axis=1)
        predictions.append(np.asarray([classes[int(index)] for index in indices]))
    return predictions, regs


def wcos_grid_predictions(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    classes: Iterable[int],
    candidates: Iterable[float],
    *,
    regularization_mode: str,
) -> tuple[list[np.ndarray], list[float]]:
    mean, covariance = total_covariance_stats(train_features)
    values, vectors = sorted_eigh(covariance)
    epsilons = resolved_regularizers(candidates, covariance, regularization_mode)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        train_pc = (train_features - mean) @ vectors
        eval_pc = (eval_features - mean) @ vectors
    if not np.isfinite(train_pc).all() or not np.isfinite(eval_pc).all():
        raise FloatingPointError("non-finite PCA projections")
    predictions = []
    for epsilon in epsilons:
        scale = 1.0 / np.sqrt(values + epsilon)
        predictions.append(
            cosine_predictions(
                train_pc * scale[None, :],
                train_labels,
                eval_pc * scale[None, :],
                classes,
            )
        )
    return predictions, epsilons


def select_index_from_scores(scores: Iterable[float]) -> int:
    values = np.asarray(list(scores), dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("selection scores must be a nonempty finite vector")
    return int(np.flatnonzero(values == values.max())[0])


def select_then_refit_evaluator(
    family: str,
    all_features: np.ndarray,
    all_labels: np.ndarray,
    inner_features: np.ndarray,
    inner_labels: np.ndarray,
    validation_features: np.ndarray | None,
    validation_labels: np.ndarray | None,
    test_features: np.ndarray,
    classes: Iterable[int],
    candidates: Iterable[float] = REG_GRID,
    *,
    regularization_mode: str = "absolute",
    default_candidate: float = 1e-4,
) -> dict:
    """Select a candidate on support validation and refit on all support."""

    candidates = [float(value) for value in candidates]
    if validation_features is None:
        selected_base = float(default_candidate)
        validation_scores: list[float] = []
        selection = "fixed_default_no_validation"
    else:
        if family == "lda":
            val_predictions, _ = lda_grid_predictions(
                inner_features,
                inner_labels,
                validation_features,
                classes,
                candidates,
                regularization_mode=regularization_mode,
            )
        elif family == "wcos":
            val_predictions, _ = wcos_grid_predictions(
                inner_features,
                inner_labels,
                validation_features,
                classes,
                candidates,
                regularization_mode=regularization_mode,
            )
        else:
            raise ValueError(f"unsupported evaluator family: {family}")
        validation_scores = [accuracy(prediction, validation_labels) for prediction in val_predictions]
        selected_base = candidates[select_index_from_scores(validation_scores)]
        selection = "support_validation"

    if family == "lda":
        test_predictions, actual = lda_grid_predictions(
            all_features,
            all_labels,
            test_features,
            classes,
            [selected_base],
            regularization_mode=regularization_mode,
        )
    else:
        test_predictions, actual = wcos_grid_predictions(
            all_features,
            all_labels,
            test_features,
            classes,
            [selected_base],
            regularization_mode=regularization_mode,
        )
    return {
        "predictions": test_predictions[0],
        "selected_base": selected_base,
        "actual_regularizer_after_refit": float(actual[0]),
        "validation_scores": validation_scores,
        "selection": selection,
        "regularization_mode": regularization_mode,
    }


def ridge_trainval_predictions(
    all_features: np.ndarray,
    all_labels: np.ndarray,
    inner_features: np.ndarray,
    inner_labels: np.ndarray,
    validation_features: np.ndarray | None,
    validation_labels: np.ndarray | None,
    test_features: np.ndarray,
    candidates: Iterable[float] = RIDGE_GRID,
    *,
    default_candidate: float = 1.0,
) -> dict:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    candidates = [float(value) for value in candidates]

    def fit_predict(x_train, y_train, x_eval, alpha):
        classifier = make_pipeline(StandardScaler(), RidgeClassifier(alpha=alpha))
        classifier.fit(x_train, y_train)
        return classifier.predict(x_eval)

    if validation_features is None:
        selected = float(default_candidate)
        validation_scores = []
        selection = "fixed_default_no_validation"
    else:
        val_predictions = [
            fit_predict(inner_features, inner_labels, validation_features, candidate) for candidate in candidates
        ]
        validation_scores = [accuracy(prediction, validation_labels) for prediction in val_predictions]
        selected = candidates[select_index_from_scores(validation_scores)]
        selection = "support_validation"
    predictions = fit_predict(all_features, all_labels, test_features, selected)
    return {
        "predictions": np.asarray(predictions),
        "selected_base": selected,
        "validation_scores": validation_scores,
        "selection": selection,
    }


def paired_bootstrap_difference(
    left_predictions: np.ndarray,
    right_predictions: np.ndarray,
    labels: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    left_correct = (np.asarray(left_predictions) == labels).astype(np.float64)
    right_correct = (np.asarray(right_predictions) == labels).astype(np.float64)
    delta = 100.0 * (left_correct - right_correct)
    rng = np.random.RandomState(seed)
    values = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sample = rng.randint(0, len(labels), size=len(labels))
        values[index] = float(delta[sample].mean())
    return {
        "point_pp": float(delta.mean()),
        "ci95_pp": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "n_test": int(len(labels)),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def environment_manifest(commit: str | None = None) -> dict:
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = None
    try:
        import torch
    except ImportError:  # JSON-only analyses do not need torch
        torch = None
    return {
        "git_commit": commit or os.environ.get("SDG_GIT_COMMIT") or "unavailable",
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "torch": torch.__version__ if torch is not None else None,
        "sklearn": sklearn_version,
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch is not None and torch.cuda.is_available()
            else None
        ),
    }
