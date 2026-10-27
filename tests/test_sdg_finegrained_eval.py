from types import SimpleNamespace

import numpy as np
import torch

from sdg_finegrained_eval import (
    chunked_linear_predictions,
    cosine_predictions,
    evaluate_seed,
    train_val_indices,
)
from sdg_stepb_cached_eval import train_val_split
from spectral_decision_gap_ed import ncm_cosine


def _fixture(seed=7, n_classes=4, per_class=8, dim=6):
    rng = np.random.RandomState(seed)
    labels = np.repeat(np.arange(n_classes), per_class)
    centers = rng.normal(size=(n_classes, dim))
    features = centers[labels] + 0.2 * rng.normal(size=(len(labels), dim))
    test_labels = np.repeat(np.arange(n_classes), 5)
    test = centers[test_labels] + 0.2 * rng.normal(size=(len(test_labels), dim))
    return features, labels, test, test_labels


def test_train_validation_indices_match_frozen_stepb_split():
    features, labels, _, _ = _fixture()
    train_idx, validation_idx, metadata = train_val_indices(labels, seed=3)
    xtr, ytr, xval, yval, expected = train_val_split(features, labels, seed=3)
    assert metadata == expected
    assert np.array_equal(features[train_idx], xtr)
    assert np.array_equal(labels[train_idx], ytr)
    assert np.array_equal(features[validation_idx], xval)
    assert np.array_equal(labels[validation_idx], yval)


def test_chunked_cosine_matches_frozen_numpy_ncm():
    train, labels, test, test_labels = _fixture()
    old = ncm_cosine(train, labels, test, test_labels, list(range(4)))
    prediction, audit = cosine_predictions(
        torch.tensor(train, dtype=torch.float64),
        torch.tensor(labels),
        torch.tensor(test, dtype=torch.float32),
        4,
        query_chunk=3,
    )
    assert np.mean(prediction == test_labels) == old
    assert audit["max_score_matrix_shape"][0] <= 3
    assert audit["full_score_matrix_materialized"] is False


def test_chunked_linear_argmax_never_uses_full_query_matrix():
    query = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    weights = torch.arange(55, dtype=torch.float64).reshape(11, 5)
    prediction, audit = chunked_linear_predictions(query, weights, None, query_chunk=2)
    expected = torch.argmax(query.double() @ weights.T, dim=1).numpy()
    assert np.array_equal(prediction, expected)
    assert audit["max_score_matrix_shape"] == [2, 11]


def test_full_evaluator_matches_frozen_methods_on_small_fixture():
    from sdg_stepb_cached_eval import run_one

    train, labels, test, test_labels = _fixture(per_class=6)
    args = SimpleNamespace(
        train_cache=None,
        test_cache=None,
        val_frac=0.2,
        max_val_per_class=50,
        lda_regs=[1e-4, 1e-3],
        wcos_eps=[1e-4, 1e-3],
        ridge_alphas=[1.0],
        knn_ks=[1],
        default_lda_reg=1e-4,
        default_wcos_eps=1e-4,
        default_ridge_alpha=1.0,
        default_knn_k=1,
    )
    import sdg_stepb_cached_eval

    original_loader = sdg_stepb_cached_eval.load_features_for_args
    sdg_stepb_cached_eval.load_features_for_args = lambda dataset, ignored: (
        train,
        labels,
        test,
        test_labels,
    )
    try:
        frozen = run_one("fixture", "5", 2, args)
    finally:
        sdg_stepb_cached_eval.load_features_for_args = original_loader
    support = []
    rng = np.random.RandomState(2)
    for label in range(4):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        support.extend(indices[:5].tolist())
    record, _ = evaluate_seed(
        torch.tensor(train, dtype=torch.float32),
        torch.tensor(labels),
        torch.tensor(test, dtype=torch.float32),
        torch.tensor(test_labels),
        np.asarray(sorted(support), dtype=np.int64),
        shot="5",
        seed=2,
        device="cpu",
        query_chunk=3,
        lda_regs=[1e-4, 1e-3],
        wcos_eps=[1e-4, 1e-3],
    )
    assert abs(record["metrics"]["cos"] - frozen["metrics"]["cos"]) < 1e-12
    assert abs(record["metrics"]["wcos_trainval"] - frozen["metrics"]["wcos_trainval"]["acc"]) < 1e-12
    assert abs(record["metrics"]["lda_trainval"] - frozen["metrics"]["lda_trainval"]["acc"]) < 1e-12
    assert record["selection"]["wcos"]["selected"] == frozen["metrics"]["wcos_trainval"]["selected"]
    assert record["selection"]["lda"]["selected"] == frozen["metrics"]["lda_trainval"]["selected"]

