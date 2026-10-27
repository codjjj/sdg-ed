import json

import numpy as np
import pytest

from sdg_finegrained_manifest import (
    canonicalize_image_rows,
    load_and_verify_manifest,
    support_indices_for_labels,
    support_index_grid,
    validate_inat2021_mini_counts,
    write_manifest_bundle,
)
from spectral_decision_gap_ed import sample_shot


def _rows(n_classes=3, n_train=20, n_test=2):
    rows = []
    for source_id in reversed(range(n_classes)):
        for split, count in [("train", n_train), ("test", n_test)]:
            for image in reversed(range(count)):
                name = f"{source_id}_{image:03d}.jpg"
                rows.append(
                    {
                        "split": split,
                        "source_class_id": str(source_id),
                        "class_name": f"class {source_id}",
                        "source_image_id": f"{split}:{name}",
                        "archive_file": f"raw/{split}.tar",
                        "archive_member": name,
                    }
                )
    return rows


def test_canonical_rows_are_order_invariant_and_labels_are_contiguous():
    classes_a, rows_a = canonicalize_image_rows(_rows())
    classes_b, rows_b = canonicalize_image_rows(list(reversed(_rows())))
    assert classes_a == classes_b
    assert rows_a == rows_b
    assert [row["source_class_id"] for row in classes_a] == ["0", "1", "2"]
    assert [row["label"] for row in classes_a] == [0, 1, 2]


@pytest.mark.parametrize("shot", ["1", "2", "5", "10", "20", "full"])
def test_support_indices_match_frozen_sample_shot(shot):
    labels = np.repeat(np.arange(4), 23)
    seeds = [0] if shot == "full" else [0, 1, 2, 3, 4]
    features = np.arange(len(labels), dtype=np.int64)[:, None]
    for seed in seeds:
        expected_features, expected_labels = sample_shot(features, labels, shot, seed)
        indices = support_indices_for_labels(labels, shot, seed)
        assert np.array_equal(features[indices], expected_features)
        assert np.array_equal(labels[indices], expected_labels)


def test_true_twenty_shot_is_fail_closed():
    labels = np.repeat(np.arange(3), 19)
    with pytest.raises(ValueError, match="only 19 rows for 20-shot"):
        support_indices_for_labels(labels, "20", 0)


def test_nabirds_manifest_records_capped_fixed_shots(tmp_path):
    root = tmp_path / "nabirds"
    manifest = write_manifest_bundle(
        root,
        dataset="nabirds",
        raw_rows=_rows(n_train=17),
        source={"name": "NABirds fixture"},
        split_definition={"name": "official fixture"},
        archive_hashes={"raw/nabirds.tar.gz": "0" * 64},
    )
    assert manifest["support_contract"]["nonfull_shots"] == ["1", "2", "5", "10", "20"]
    assert manifest["support_contract"]["capped_per_class"] is True
    assert manifest["support_contract"]["unavailable_shots"] == {}
    assert {row["shot"] for row in manifest["supports"]} == {
        "1",
        "2",
        "5",
        "10",
        "20",
        "full",
    }
    twenty = [row for row in manifest["supports"] if row["shot"] == "20"]
    assert {row["class_support_count_min"] for row in twenty} == {17}
    assert {row["class_support_count_max"] for row in twenty} == {17}
    assert {row["classes_below_requested_k"] for row in twenty} == {3}


def test_capped_support_grid_uses_min_available_and_is_nested():
    labels = np.asarray([0] * 4 + [1] * 7 + [2] * 23)
    grid = support_index_grid(labels, capped_per_class=True)
    for seed in [0, 1, 2, 3, 4]:
        previous = set()
        for shot in ["1", "2", "5", "10", "20"]:
            indices = grid[(shot, seed)]
            counts = np.bincount(labels[indices], minlength=3).tolist()
            assert counts == [min(4, int(shot)), min(7, int(shot)), min(23, int(shot))]
            assert previous <= set(indices.tolist())
            previous = set(indices.tolist())


def test_strict_support_grid_still_rejects_insufficient_classes():
    labels = np.asarray([0] * 4 + [1] * 20)
    with pytest.raises(ValueError, match="true 20-shot is unavailable"):
        support_index_grid(labels)


def test_other_datasets_still_require_true_twenty_shot(tmp_path):
    with pytest.raises(ValueError, match="true 20-shot"):
        write_manifest_bundle(
            tmp_path / "dogs",
            dataset="stanford_dogs",
            raw_rows=_rows(n_train=17),
            source={"name": "Dogs fixture"},
            split_definition={"name": "official fixture"},
            archive_hashes={"raw/images.tar": "0" * 64},
        )


def test_batched_support_grid_is_byte_identical_to_frozen_sampling():
    labels = np.repeat(np.arange(7), 23)
    grid = support_index_grid(labels)
    for shot in ["1", "2", "5", "10", "20"]:
        for seed in [0, 1, 2, 3, 4]:
            expected = support_indices_for_labels(labels, shot, seed)
            assert np.array_equal(grid[(shot, seed)], expected)
    assert np.array_equal(grid[("full", 0)], np.arange(len(labels)))


def test_inat_requires_all_ten_thousand_classes_and_exact_counts():
    classes = [
        {
            "source_class_id": str(index),
            "train_count": 50,
            "test_count": 10,
        }
        for index in range(10_000)
    ]
    validate_inat2021_mini_counts(classes)
    with pytest.raises(ValueError, match="exactly 10000"):
        validate_inat2021_mini_counts(classes[:-1])
    classes[123]["test_count"] = 9
    with pytest.raises(ValueError, match="count mismatch"):
        validate_inat2021_mini_counts(classes)


def test_manifest_bundle_detects_tampering(tmp_path):
    root = tmp_path / "manifest"
    manifest = write_manifest_bundle(
        root,
        dataset="stanford_dogs",
        raw_rows=_rows(),
        source={"name": "synthetic fixture"},
        split_definition={"name": "fixture"},
        archive_hashes={"raw/train.tar": "0" * 64, "raw/test.tar": "1" * 64},
    )
    loaded = load_and_verify_manifest(root / "dataset_manifest.json")
    assert loaded["dataset_identity_sha256"] == manifest["dataset_identity_sha256"]
    images = root / loaded["images"]["path"]
    images.write_text(images.read_text() + json.dumps({"tampered": True}) + "\n")
    with pytest.raises(ValueError, match="images file hash mismatch"):
        load_and_verify_manifest(root / "dataset_manifest.json")
