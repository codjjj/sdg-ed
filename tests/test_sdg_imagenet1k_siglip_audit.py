import numpy as np
import torch

import sdg_imagenet1k_siglip_audit as audit
from sdg_imagenet1k_siglip_audit import (
    REG_GRID,
    SEMANTIC_GROUPS,
    add_statistics,
    covariance_bundle,
    family_parameters,
    local_label_map,
    predict_cosine,
    predict_family,
    select_base_regularizer,
    stable_within_class_split,
    sufficient_statistics,
)
from sdg_audit_core import (
    cosine_predictions,
    select_then_refit_evaluator,
)


def _fixture():
    rng = np.random.RandomState(17)
    dim = 5
    classes = np.array([0, 1, 2], dtype=np.int64)
    means = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
        ]
    )
    features = []
    labels = []
    original = []
    for cls in classes:
        values = means[cls] + 0.15 * rng.randn(40, dim)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        features.append(values.astype(np.float32))
        labels.extend([cls] * len(values))
        original.extend((10_000 * cls + np.arange(len(values))).tolist())
    return (
        np.concatenate(features),
        np.asarray(labels, dtype=np.int64),
        np.asarray(original, dtype=np.int64),
        classes,
    )


def test_semantic_group_definitions_are_frozen_wordnet_descendants():
    assert len(SEMANTIC_GROUPS["global_1k"]["labels"]) == 1000
    assert len(SEMANTIC_GROUPS["dogs"]["labels"]) == 118
    assert len(SEMANTIC_GROUPS["birds"]["labels"]) == 59
    assert len(SEMANTIC_GROUPS["vehicles"]["labels"]) == 67
    assert set(SEMANTIC_GROUPS["dogs"]["labels"]).isdisjoint(
        SEMANTIC_GROUPS["birds"]["labels"]
    )
    assert set(SEMANTIC_GROUPS["dogs"]["labels"]).isdisjoint(
        SEMANTIC_GROUPS["vehicles"]["labels"]
    )
    assert set(SEMANTIC_GROUPS["birds"]["labels"]).isdisjoint(
        SEMANTIC_GROUPS["vehicles"]["labels"]
    )


def test_stable_split_is_order_invariant_and_class_stratified():
    _, labels, original, _ = _fixture()
    first = stable_within_class_split(labels, original, seed=20260724)
    permutation = np.random.RandomState(3).permutation(len(labels))
    second = stable_within_class_split(
        labels[permutation], original[permutation], seed=20260724
    )
    second_by_original = dict(zip(original[permutation].tolist(), second.tolist()))
    assert np.array_equal(
        first,
        np.asarray([second_by_original[value] for value in original], dtype=np.uint8),
    )
    for cls in range(3):
        values = first[labels == cls]
        assert np.sum(values == 0) == 32
        assert np.sum(values == 1) == 4
        assert np.sum(values == 2) == 4


def test_torch_evaluators_match_frozen_float64_primitives():
    features, labels, original, classes = _fixture()
    assignment = stable_within_class_split(labels, original, seed=20260724)
    mapping = local_label_map(classes)
    local = mapping[labels]
    inner = np.flatnonzero(assignment == 0)
    validation = np.flatnonzero(assignment == 1)
    test = np.flatnonzero(assignment == 2)
    device = torch.device("cpu")

    inner_stats = sufficient_statistics(
        features, local, inner, len(classes), device=device, batch_size=17
    )
    validation_stats = sufficient_statistics(
        features, local, validation, len(classes), device=device, batch_size=17
    )
    inner_bundle = covariance_bundle(inner_stats)
    refit_bundle = covariance_bundle(add_statistics(inner_stats, validation_stats))

    ours_cos = predict_cosine(
        features,
        test,
        refit_bundle["means"],
        classes,
        device=device,
        batch_size=7,
        score_dtype=torch.float64,
    )
    expected_cos = cosine_predictions(
        features[np.concatenate([inner, validation])].astype(np.float64),
        labels[np.concatenate([inner, validation])],
        features[test].astype(np.float64),
        classes,
    )
    assert np.array_equal(ours_cos, expected_cos)

    for family in ("wcos", "lda"):
        selection = select_base_regularizer(
            family,
            features,
            validation,
            labels[validation],
            family_parameters(family, inner_bundle),
            classes,
            device=device,
            batch_size=7,
            score_dtype=torch.float64,
        )
        ours, _ = predict_family(
            family,
            features,
            test,
            family_parameters(family, refit_bundle),
            selection["selected_base"],
            classes,
            device=device,
            batch_size=7,
            score_dtype=torch.float64,
        )
        expected = select_then_refit_evaluator(
            family,
            features[np.concatenate([inner, validation])].astype(np.float64),
            labels[np.concatenate([inner, validation])],
            features[inner].astype(np.float64),
            labels[inner],
            features[validation].astype(np.float64),
            labels[validation],
            features[test].astype(np.float64),
            classes,
            REG_GRID,
            regularization_mode="relative_trace",
        )
        assert selection["selected_base"] == expected["selected_base"]
        assert np.array_equal(ours, expected["predictions"])

        production_selection = select_base_regularizer(
            family,
            features,
            validation,
            labels[validation],
            family_parameters(family, inner_bundle),
            classes,
            device=device,
            batch_size=7,
            score_dtype=torch.float32,
        )
        production_predictions, _ = predict_family(
            family,
            features,
            test,
            family_parameters(family, refit_bundle),
            production_selection["selected_base"],
            classes,
            device=device,
            batch_size=7,
            score_dtype=torch.float32,
        )
        assert production_selection["selected_base"] == expected["selected_base"]
        assert np.array_equal(production_predictions, expected["predictions"])


def test_write_new_does_not_require_cross_filesystem_rename(tmp_path, monkeypatch):
    target = tmp_path / "mounted-output" / "result.json"

    def fail_if_called(*_args, **_kwargs):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(audit.os, "replace", fail_if_called)
    audit.write_new(target, '{"status": "ok"}\n')

    assert target.read_text() == '{"status": "ok"}\n'
