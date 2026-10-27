import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sdg_encoder_finetune_comparison import (
    class_cluster_contrast_bootstrap,
    compare_dataset,
    inferred_scope,
    joint_direction,
    load_scope_dataset,
    paired_contrast_bootstrap,
)


def test_paired_contrast_bootstrap_has_exact_point_and_is_deterministic():
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    left = np.asarray([0, 1, 1, 1, 2, 0])
    right = np.asarray([1, 0, 0, 1, 2, 2])
    terms = [(left, 1.0), (right, -1.0)]
    first = paired_contrast_bootstrap(terms, labels, n_boot=500, seed=7)
    second = paired_contrast_bootstrap(terms, labels, n_boot=500, seed=7)
    expected = 100.0 * (
        np.mean(left == labels) - np.mean(right == labels)
    )
    assert first == second
    assert first["point_pp"] == pytest.approx(expected)
    assert first["n_test"] == 6


def test_class_cluster_bootstrap_uses_macro_classes_and_joint_rule_is_strict():
    labels = np.asarray([0, 0, 0, 1])
    left = np.asarray([0, 0, 1, 1])
    right = np.asarray([1, 1, 0, 0])
    result = class_cluster_contrast_bootstrap(
        [(left, 1.0), (right, -1.0)], labels, n_boot=500, seed=11
    )
    class_zero = np.mean(
        (left[:3] == labels[:3]).astype(float)
        - (right[:3] == labels[:3]).astype(float)
    )
    class_one = float(left[3] == labels[3]) - float(right[3] == labels[3])
    assert result["point_pp"] == pytest.approx(100.0 * (class_zero + class_one) / 2)
    assert (
        joint_direction(
            {"ci95_pp": [0.1, 2.0]}, {"ci95_pp": [-0.1, 3.0]}
        )
        == "boundary"
    )
    assert (
        joint_direction(
            {"ci95_pp": [0.1, 2.0]}, {"ci95_pp": [0.2, 3.0]}
        )
        == "positive"
    )


def _write_scope_fixture(
    root: Path, *, scope: str, commit: str, tamper_hash: bool = False
) -> None:
    dataset_root = root / "toy"
    seed_root = dataset_root / "seed_0"
    seed_root.mkdir(parents=True)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int16)
    arrays = {
        "labels": labels,
        "frozen__cos": np.asarray([0, 1, 1, 0], dtype=np.int16),
        "frozen__wcos_tv": np.asarray([0, 0, 1, 0], dtype=np.int16),
        "frozen__lda_tv": np.asarray([0, 0, 1, 1], dtype=np.int16),
        "frozen__validation_whitening": np.asarray([0, 0, 1, 1], dtype=np.int16),
        "finetuned__cos": np.asarray([0, 1, 1, 1], dtype=np.int16),
        "finetuned__wcos_tv": np.asarray([0, 0, 1, 1], dtype=np.int16),
        "finetuned__lda_tv": np.asarray([0, 0, 1, 0], dtype=np.int16),
        "finetuned__validation_whitening": np.asarray([0, 0, 1, 1], dtype=np.int16),
    }
    prediction_path = seed_root / "predictions.npz"
    np.savez(prediction_path, **arrays)
    prediction_hash = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    if tamper_hash:
        prediction_hash = "0" * 64

    def gap(method):
        return 100.0 * (
            np.mean(arrays[f"finetuned__{method}"] == labels)
            - np.mean(arrays["finetuned__cos"] == labels)
        )

    def class_gap(method):
        delta = 100.0 * (
            (arrays[f"finetuned__{method}"] == labels).astype(float)
            - (arrays["finetuned__cos"] == labels).astype(float)
        )
        return np.mean([delta[labels == cls].mean() for cls in np.unique(labels)])

    kind_scope = "last_block" if scope == "last_block" else "full"
    fraction = 0.1 if scope == "last_block" else 1.0
    row = {
        "schema_version": 1,
        "kind": f"post_submission_true_encoder_{kind_scope}_finetune_seed_audit",
        "dataset": "toy",
        "seed": 0,
        "effective_seed": 13,
        "epoch_selection": {
            "selected_epoch": 1,
            "selection_uses": "stratified training validation only",
            "scope": {"trainable_fraction": fraction},
        },
        "refit": {
            "scope": {"trainable_fraction": fraction},
            "frozen_parameter_sha256_before": "a",
            "frozen_parameter_sha256_after": "a",
            "trainable_parameter_sha256_before": "b",
            "trainable_parameter_sha256_after": "c",
        },
        "head_test_labels_used_for_selection": False,
        "fine_tune_split": {
            "n_fit": 2,
            "n_validation": 2,
            "n_selected": 4,
            "n_ignored_by_budget": 0,
            "validation_fraction": 0.5,
            "max_total_per_class": 2,
            "test_labels_used": False,
        },
        "finetuned_evaluator": {
            "validation_whitening_family": "lda",
            "validation_split": {"test_labels_used": False},
            "accuracies": {
                method: float(np.mean(arrays[f"finetuned__{method}"] == labels))
                for method in ("cos", "wcos_tv", "lda_tv", "validation_whitening")
            },
            "gaps_pp": {
                method: gap(method)
                for method in ("wcos_tv", "lda_tv", "validation_whitening")
            },
        },
        "paired_test_bootstrap": {
            method: {"point_pp": gap(method), "ci95_pp": [-1.0, 1.0]}
            for method in ("wcos_tv", "lda_tv", "validation_whitening")
        },
        "class_cluster_bootstrap": {
            method: {"point_pp": class_gap(method), "ci95_pp": [-1.0, 1.0]}
            for method in ("wcos_tv", "lda_tv", "validation_whitening")
        },
        "artifacts": {
            "predictions.npz": {
                "sha256": prediction_hash,
                "arrays": sorted(arrays),
            }
        },
        "environment": {"git_commit": commit},
    }
    (seed_root / "result.json").write_text(json.dumps(row))
    aggregate = {
        "schema_version": 1,
        "kind": f"post_submission_true_encoder_{kind_scope}_finetune_audit",
        "dataset": "toy",
        "protocol": {},
        "data": {"n_train": 4, "n_test": 4, "n_classes": 2, "feature_dim": 3},
        "seeds": [row],
        "environment": {"git_commit": commit},
    }
    (dataset_root / "result.json").write_text(json.dumps(aggregate))


def test_scope_loader_accepts_legacy_last_block_and_rejects_bad_prediction_hash(
    tmp_path,
):
    valid = tmp_path / "valid"
    _write_scope_fixture(valid, scope="last_block", commit="legacy")
    loaded = load_scope_dataset(
        valid,
        "toy",
        expected_scope="last_block",
        expected_commit="legacy",
        expected_seeds=[0],
    )
    assert inferred_scope(loaded["aggregate"]) == "last_block"

    invalid = tmp_path / "invalid"
    _write_scope_fixture(
        invalid, scope="full_encoder", commit="full", tamper_hash=True
    )
    with pytest.raises(ValueError, match="prediction SHA256 mismatch"):
        load_scope_dataset(
            invalid,
            "toy",
            expected_scope="full_encoder",
            expected_commit="full",
            expected_seeds=[0],
        )


def test_compare_dataset_recomputes_paired_direct_and_did_and_rejects_label_drift(
    tmp_path,
):
    last_root = tmp_path / "last"
    full_root = tmp_path / "full"
    _write_scope_fixture(last_root, scope="last_block", commit="last")
    _write_scope_fixture(full_root, scope="full_encoder", commit="full")
    last = load_scope_dataset(
        last_root,
        "toy",
        expected_scope="last_block",
        expected_commit="last",
        expected_seeds=[0],
    )
    full = load_scope_dataset(
        full_root,
        "toy",
        expected_scope="full_encoder",
        expected_commit="full",
        expected_seeds=[0],
    )

    full_predictions = full["rows"][0]["predictions"]
    full_predictions["finetuned__wcos_tv"] = full_predictions[
        "finetuned__cos"
    ].copy()
    comparison = compare_dataset(
        "toy",
        last,
        full,
        seeds=[0],
        n_boot=500,
        bootstrap_seed=17,
    )
    wcos = comparison["seeds"][0]["methods"]["wcos_tv"]
    assert wcos["last_block_gap_pp"] == pytest.approx(25.0)
    assert wcos["full_encoder_gap_pp"] == pytest.approx(0.0)
    assert wcos["contrasts"]["full_minus_last"]["paired_example"][
        "point_pp"
    ] == pytest.approx(-25.0)
    assert wcos["contrasts"]["frozen_control"]["paired_example"][
        "point_pp"
    ] == pytest.approx(0.0)
    assert wcos["contrasts"]["attenuation_did"]["paired_example"][
        "point_pp"
    ] == pytest.approx(-25.0)

    full["rows"][0]["labels"] = full["rows"][0]["labels"][::-1]
    with pytest.raises(ValueError, match="test label axes mismatch"):
        compare_dataset(
            "toy",
            last,
            full,
            seeds=[0],
            n_boot=50,
            bootstrap_seed=17,
        )
