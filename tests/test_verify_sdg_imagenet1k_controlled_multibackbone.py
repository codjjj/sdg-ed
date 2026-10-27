from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import sdg_imagenet1k_controlled_multibackbone as audit
import verify_sdg_imagenet1k_controlled_multibackbone as verifier


DIGEST = "a" * 64


def gap_stat(point: float = 1.0) -> dict:
    return {
        "point_pp": point,
        "ci95_pp": [point - 0.5, point + 0.5],
        "n_boot": 10_000,
    }


def result_payload(expected_commit: str) -> dict:
    arms = {}
    for arm in audit.ARM_ORDER:
        arms[arm] = [
            {
                "seed": seed,
                "n_classes": len(audit.ARMS[arm]["labels"]),
                "split_counts": {
                    "inner": 100,
                    "selection": 20,
                    "refit_support": 120,
                    "test": 10,
                },
                "selection": {"wcos": {}, "lda": {}},
                "refit_actual_regularizers": {"wcos": 1.0, "lda": 1.0},
                "test_accuracies": {"cos": 0.5, "wcos": 0.6, "lda": 0.7},
                "prediction_sha256": {
                    "cos": DIGEST,
                    "wcos": DIGEST,
                    "lda": DIGEST,
                },
                "fixed_evaluator_gaps": {
                    "wcos": gap_stat(),
                    "lda": gap_stat(),
                },
            }
            for seed in audit.SPLIT_SEEDS
        ]
    contrasts = []
    for seed in audit.SPLIT_SEEDS:
        contrasts.append(
            {
                "seed": seed,
                "fixed_evaluator_contrasts": {
                    "wcos": gap_stat(),
                    "lda": gap_stat(),
                },
                "paired_subgroup_global_contrasts": {
                    subgroup: {
                        "wcos": gap_stat(),
                        "lda": gap_stat(),
                    }
                    for subgroup in ("dogs", "birds", "vehicles")
                },
            }
        )
    return {
        "schema_version": 1,
        "kind": "sdg_imagenet1k_controlled_multibackbone_backbone_result",
        "git_commit": expected_commit,
        "backbone": "dinov2",
        "backbone_spec": audit.BACKBONES["dinov2"],
        "source_ready_manifest_sha256": verifier.EXPECTED_READY_MANIFEST_SHA256,
        "source_ready_manifest_total_bytes": audit.EXPECTED_SOURCE_BYTES,
        "source": {
            "repo": audit.SOURCE_REPO,
            "revision": audit.SOURCE_REVISION,
            "splits": {
                "train": {
                    "rows": audit.EXPECTED_TRAIN_ROWS,
                    "shards": [{}] * audit.EXPECTED_TRAIN_SHARDS,
                },
                "validation": {
                    "rows": audit.EXPECTED_VALIDATION_ROWS,
                    "shards": [{}] * audit.EXPECTED_VALIDATION_SHARDS,
                },
            },
        },
        "model_identity": {
            "files": [{"bytes": 1, "sha256": DIGEST}],
        },
        "protocol": {
            "split_seeds": list(audit.SPLIT_SEEDS),
            "regularization_grid": list(audit.REG_GRID),
            "n_boot": 10_000,
            "exact_imagenet100_labels": list(audit.IMAGENET100_LABELS),
        },
        "arms": arms,
        "paired_shared100_scale_contrasts": contrasts,
    }


def test_result_verifier_requires_all_paired_contrasts():
    commit = "f" * 40
    payload = result_payload(commit)
    verifier.verify_result_payload(payload, "dinov2", commit)

    missing = deepcopy(payload)
    del missing["paired_shared100_scale_contrasts"][0][
        "paired_subgroup_global_contrasts"
    ]["vehicles"]
    with pytest.raises(ValueError, match="subgroup contrast set"):
        verifier.verify_result_payload(missing, "dinov2", commit)


def test_gap_stat_verifier_rejects_invalid_interval_or_bootstrap_count():
    verifier.require_gap_stat(gap_stat(), "valid")
    reversed_interval = gap_stat()
    reversed_interval["ci95_pp"] = [2.0, 1.0]
    with pytest.raises(ValueError, match="confidence interval"):
        verifier.require_gap_stat(reversed_interval, "reversed")
    wrong_bootstrap = gap_stat()
    wrong_bootstrap["n_boot"] = 999
    with pytest.raises(ValueError, match="bootstrap count"):
        verifier.require_gap_stat(wrong_bootstrap, "wrong-bootstrap")


def test_stability_and_cross_backbone_rules_are_pre_registered():
    assert verifier.stable_direction([gap_stat(1.0)] * 3) == "positive"
    assert verifier.stable_direction([gap_stat(-1.0)] * 3) == "negative"
    assert verifier.stable_direction([gap_stat(0.0)] * 3) == "boundary"
    with pytest.raises(ValueError, match="every pre-registered seed"):
        verifier.stable_direction([gap_stat(1.0)] * 2)

    four_positive = [
        {"stable_direction": "positive"},
    ] * 4 + [{"stable_direction": "boundary"}] * 2
    assert verifier.cross_backbone_summary(four_positive)["pattern"] == "positive"

    positive_with_opposite = four_positive[:4] + [
        {"stable_direction": "negative"},
        {"stable_direction": "boundary"},
    ]
    assert (
        verifier.cross_backbone_summary(positive_with_opposite)["pattern"]
        == "representation-dependent"
    )
    with pytest.raises(ValueError, match="every registered backbone"):
        verifier.cross_backbone_summary(four_positive[:5])


def test_evidence_summary_reports_every_backbone_arm_and_contrast():
    commit = "f" * 40
    payloads = []
    for backbone, spec in audit.BACKBONES.items():
        payload = result_payload(commit)
        payload["backbone"] = backbone
        payload["backbone_label"] = spec["label"]
        payload["backbone_spec"] = spec
        payloads.append(payload)

    summary = verifier.build_evidence_summary(payloads)
    assert [row["backbone"] for row in summary["rows"]] == list(audit.BACKBONES)
    assert set(summary["cross_backbone"]["arms"]) == set(audit.ARM_ORDER)
    assert set(summary["cross_backbone"]["contrasts"]) == {
        "shared100_scale",
        "dogs_minus_global",
        "birds_minus_global",
        "vehicles_minus_global",
    }
    assert (
        summary["cross_backbone"]["arms"]["global_1k"]["wcos"]["pattern"]
        == "positive"
    )

    report = verifier.evidence_report(summary)
    assert "The range is not a confidence interval" in report
    assert "shared-100 scale DID" in report
    for spec in audit.BACKBONES.values():
        assert spec["label"] in report


def test_feature_verifier_rehashes_copied_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(audit, "EXPECTED_TRAIN_ROWS", 4)
    monkeypatch.setattr(audit, "EXPECTED_VALIDATION_ROWS", 2)
    monkeypatch.setattr(audit, "EXPECTED_CLASSES", 2)
    monkeypatch.setitem(audit.BACKBONES["dinov2"], "expected_dim", 3)

    root = tmp_path / "output"
    metadata = {}
    cache = {}
    for split, rows, labels in (
        ("train", 4, np.asarray([0, 0, 1, 1], dtype=np.int16)),
        ("validation", 2, np.asarray([0, 1], dtype=np.int16)),
    ):
        split_root = root / "dinov2" / "feature_cache" / split
        split_root.mkdir(parents=True)
        features = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
        feature_path = split_root / "features.npy"
        label_path = split_root / "labels.npy"
        np.save(feature_path, features)
        np.save(label_path, labels)
        counts = np.bincount(labels.astype(np.int64), minlength=2)
        metadata[split] = {
            "rows": rows,
            "dim": 3,
            "class_count_min": int(counts.min()),
            "class_count_median": float(np.median(counts)),
            "class_count_max": int(counts.max()),
            "repeated_forward_max_abs": 0.0,
            "repeated_forward_max_abs_tolerance": audit.REPEATED_FORWARD_MAX_ABS_TOL,
            "feature_sha256": audit.sha256_file(feature_path),
            "label_sha256": audit.sha256_file(label_path),
        }
        cache[split] = {
            "features.npy": {
                "bytes": feature_path.stat().st_size,
                "sha256": audit.sha256_file(feature_path),
            },
            "labels.npy": {
                "bytes": label_path.stat().st_size,
                "sha256": audit.sha256_file(label_path),
            },
        }

    payload = {"feature_metadata": metadata, "feature_cache": cache}
    summary = verifier.verify_feature_cache(
        root, "dinov2", payload, rehash=True
    )
    assert summary["train"]["files"]["features.npy"]["rehash_verified"] is True

    train_feature_path = (
        root / "dinov2" / "feature_cache" / "train" / "features.npy"
    )
    with train_feature_path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verifier.verify_feature_cache(root, "dinov2", payload, rehash=True)
