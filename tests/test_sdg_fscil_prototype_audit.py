from pathlib import Path

import numpy as np
import pytest
import torch

from sdg_fscil_prototype_audit import (
    BASE_CLASSES,
    METHODS,
    NUM_CLASSES,
    SESSIONS,
    SHOT,
    _load_frozen_slim,
    _load_source_split,
    all_class_scores,
    audit_index_files,
    evaluate_score_sessions,
    fit_base_geometries,
    label_from_index_line,
    official_support_indices,
    path_key,
    resampled_support_indices,
    seed_paired_summary,
    support_arrays,
)


INDEX_ROOT = Path(__file__).resolve().parents[1] / "splits" / "fscil_cub200"


def _official_rows():
    paths = []
    labels = []
    for session in range(1, SESSIONS + 1):
        lines = [
            line.strip()
            for line in (INDEX_ROOT / f"session_{session}.txt").read_text().splitlines()
            if line.strip()
        ]
        paths.extend(lines)
        labels.extend(label_from_index_line(line) for line in lines)
    return paths, np.asarray(labels, dtype=np.int64)


def test_official_index_is_exact_100_plus_10x10x5_protocol():
    audit = audit_index_files(INDEX_ROOT)
    assert audit["base_classes"] == 100
    assert audit["way"] == 10
    assert audit["sessions"] == 11
    assert audit["shot"] == 5
    assert audit["class_order"] == list(range(200))
    assert audit["files"][0]["rows"] == 3000
    assert [row["rows"] for row in audit["files"][1:]] == [50] * 10
    assert audit["unique_support_paths"] == 3500


def test_official_support_mapping_preserves_exact_image_identities():
    paths, labels = _official_rows()
    support, order, manifest = official_support_indices(paths, labels, INDEX_ROOT)
    assert order == list(range(NUM_CLASSES))
    assert manifest["kind"] == "official_index"
    assert all(len(support[label]) == 30 for label in range(BASE_CLASSES))
    assert all(len(support[label]) == SHOT for label in range(BASE_CLASSES, NUM_CLASSES))
    selected = np.concatenate([support[label] for label in range(NUM_CLASSES)])
    assert len(selected) == len(set(selected.tolist())) == 3500
    assert np.array_equal(labels[selected], np.repeat(np.arange(200), [30] * 100 + [5] * 100))


def test_path_and_label_parsing_ignore_machine_specific_roots():
    relative = (
        "CUB_200_2011/images/101.White_Pelican/"
        "White_Pelican_0081_96148.jpg"
    )
    rooted = f"/different/machine/root/{relative}"
    legacy_cache = (
        "data/SLDC-full/cub200/train/100_101_White_Pelican/"
        "White_Pelican_0081_96148.jpg"
    )
    assert path_key(relative) == path_key(rooted)
    assert path_key(relative) == path_key(legacy_cache)
    assert path_key(relative) == "101.White_Pelican/White_Pelican_0081_96148.jpg"
    assert label_from_index_line(relative) == 100


def test_official_support_mapping_accepts_legacy_cache_class_directories():
    paths, labels = _official_rows()
    cache_paths = []
    for path, label in zip(paths, labels.tolist()):
        source = Path(path)
        class_name = source.parent.name.split(".", 1)[1]
        cache_paths.append(
            "data/SLDC-full/cub200/train/"
            f"{label:03d}_{label + 1:03d}_{class_name}/{source.name}"
        )
    support, order, manifest = official_support_indices(
        cache_paths, labels, INDEX_ROOT
    )
    assert order == list(range(NUM_CLASSES))
    assert manifest["kind"] == "official_index"
    assert all(len(support[label]) == 30 for label in range(BASE_CLASSES))
    assert all(len(support[label]) == SHOT for label in range(BASE_CLASSES, NUM_CLASSES))
    selected = np.concatenate([support[label] for label in range(NUM_CLASSES)])
    assert len(selected) == len(set(selected.tolist())) == 3500
    assert np.array_equal(labels[selected], np.repeat(np.arange(200), [30] * 100 + [5] * 100))


def test_path_key_rejects_inconsistent_legacy_cache_class_ids():
    with pytest.raises(ValueError, match="inconsistent"):
        path_key(
            "data/SLDC-full/cub200/train/099_101_White_Pelican/"
            "White_Pelican_0081_96148.jpg"
        )


def test_mmap_source_loader_selects_pooled_labels_and_paths_without_semantic_drift(tmp_path):
    source_path = tmp_path / "source.pt"
    pooled = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    labels = torch.tensor([0, 0, 1, 1])
    paths = [f"/root/{index:03d}.Class/image_{index}.jpg" for index in range(4)]
    torch.save(
        {
            "latents": torch.randn(4, 32, 8, 8),
            "pooled": pooled,
            "labels": labels,
            "paths": paths,
        },
        source_path,
    )
    loaded = _load_source_split(source_path)
    assert set(loaded) == {"pooled", "labels", "paths"}
    torch.testing.assert_close(loaded["pooled"], pooled)
    torch.testing.assert_close(loaded["labels"], labels)
    assert loaded["paths"] == paths

    frozen_path = tmp_path / "frozen.pt"
    torch.save({"pooled": pooled, "labels": labels}, frozen_path)
    frozen = _load_frozen_slim(frozen_path)
    assert set(frozen) == {"pooled", "labels"}
    torch.testing.assert_close(frozen["pooled"], pooled)
    torch.testing.assert_close(frozen["labels"], labels)


def test_resampled_support_is_deterministic_exact_five_and_base_frozen():
    paths = []
    labels = []
    base_support = {}
    for label in range(NUM_CLASSES):
        count = 30 if label < BASE_CLASSES else 12
        start = len(paths)
        paths.extend(
            f"CUB_200_2011/images/{label + 1:03d}.Class_{label}/image_{label}_{row}.jpg"
            for row in range(count)
        )
        labels.extend([label] * count)
        base_support[label] = np.arange(
            start,
            start + (30 if label < BASE_CLASSES else SHOT),
            dtype=np.int64,
        )
    labels = np.asarray(labels, dtype=np.int64)
    first, order_first, manifest_first = resampled_support_indices(
        paths, labels, base_support, seed=7
    )
    repeat, order_repeat, manifest_repeat = resampled_support_indices(
        paths, labels, base_support, seed=7
    )
    second, order_second, _ = resampled_support_indices(paths, labels, base_support, seed=8)
    assert order_first == order_repeat
    assert manifest_first == manifest_repeat
    assert sorted(order_first) == list(range(NUM_CLASSES))
    assert order_first[:BASE_CLASSES] == list(range(BASE_CLASSES))
    assert all(np.array_equal(first[label], base_support[label]) for label in range(BASE_CLASSES))
    assert all(len(first[label]) == SHOT for label in range(BASE_CLASSES, NUM_CLASSES))
    assert all(np.array_equal(first[label], repeat[label]) for label in range(NUM_CLASSES))
    assert order_first != order_second
    assert any(
        not np.array_equal(first[label], second[label])
        for label in range(BASE_CLASSES, NUM_CLASSES)
    )


def test_session_scoring_excludes_future_classes_and_future_test_rows():
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    scores = np.asarray(
        [
            [5.0, 1.0, 2.0, 200.0],
            [1.0, 5.0, 2.0, 200.0],
            [1.0, 2.0, 5.0, 200.0],
            [1.0, 2.0, 3.0, 5.0],
        ]
    )
    rows, predictions = evaluate_score_sessions(
        scores,
        labels,
        [0, 1, 2, 3],
        base_classes=2,
        way=1,
    )
    assert len(rows) == 3
    assert rows[0]["num_eval"] == 2
    assert rows[0]["all_acc"] == 1.0
    assert predictions[0].tolist() == [0, 1, -1, -1]
    assert rows[1]["num_eval"] == 3
    assert predictions[1].tolist() == [0, 1, 2, -1]
    assert rows[2]["num_eval"] == 4
    assert predictions[2].tolist() == [3, 3, 3, 3]


def test_base_geometry_cannot_change_when_future_support_changes():
    rng = np.random.RandomState(11)
    base_features = np.concatenate(
        [rng.normal(loc=label / 100.0, scale=0.2, size=(3, 6)) for label in range(100)]
    )
    base_labels = np.repeat(np.arange(100), 3)
    future_features = rng.normal(size=(100, 6))
    future_labels = np.arange(100, 200)
    features = np.concatenate([base_features, future_features])
    labels = np.concatenate([base_labels, future_labels])
    base_rows = np.arange(len(base_features), dtype=np.int64)
    first = fit_base_geometries(features, labels, base_rows, validation_seed=0)
    changed = features.copy()
    changed[len(base_features) :] = 1e6 * rng.normal(size=future_features.shape)
    second = fit_base_geometries(changed, labels, base_rows, validation_seed=0)
    assert first["base_rows_sha256"] == second["base_rows_sha256"]
    assert first["lda"]["selected_regularizer"] == second["lda"]["selected_regularizer"]
    assert first["wcos"]["selected_regularizer"] == second["wcos"]["selected_regularizer"]
    np.testing.assert_array_equal(first["lda"]["inverse"], second["lda"]["inverse"])
    np.testing.assert_array_equal(first["wcos"]["mean"], second["wcos"]["mean"])
    np.testing.assert_array_equal(first["wcos"]["vectors"], second["wcos"]["vectors"])
    np.testing.assert_array_equal(first["wcos"]["scale"], second["wcos"]["scale"])


def test_all_geometry_heads_share_the_same_support_rows_and_class_set():
    rng = np.random.RandomState(19)
    train_features = np.concatenate(
        [rng.normal(loc=label / 50.0, scale=0.1, size=(3, 5)) for label in range(200)]
    )
    train_labels = np.repeat(np.arange(200), 3)
    base_rows = np.flatnonzero(train_labels < 100)
    geometry = fit_base_geometries(train_features, train_labels, base_rows, validation_seed=3)
    test_features = rng.normal(size=(17, 5))
    scores = all_class_scores(train_features, train_labels, test_features, geometry)
    assert set(scores) == set(METHODS)
    assert all(value.shape == (17, 200) for value in scores.values())
    assert all(np.isfinite(value).all() for value in scores.values())


def test_support_array_gate_rejects_cross_class_row_mixup():
    features = np.arange(200 * 2, dtype=np.float64).reshape(200, 2)
    labels = np.arange(200, dtype=np.int64)
    support = {label: np.asarray([label], dtype=np.int64) for label in range(200)}
    selected_features, selected_labels = support_arrays(features, labels, support)
    assert np.array_equal(selected_features, features)
    assert np.array_equal(selected_labels, labels)
    support[17] = np.asarray([18], dtype=np.int64)
    with pytest.raises(ValueError, match="align"):
        support_arrays(features, labels, support)


def test_paired_seed_summary_preserves_signs_and_raw_values():
    summary = seed_paired_summary([1.0, 2.0, 3.0, -1.0, 0.0], seed=5, n_boot=200)
    assert summary["n_seeds"] == 5
    assert summary["positive_seeds"] == 3
    assert summary["zero_seeds"] == 1
    assert summary["negative_seeds"] == 1
    assert summary["values_pp"] == [1.0, 2.0, 3.0, -1.0, 0.0]
