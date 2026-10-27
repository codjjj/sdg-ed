import numpy as np

from src.sdg_audit_core import (
    accuracy,
    cosine_predictions,
    pooled_within_stats,
    resolved_regularizers,
    select_index_from_scores,
    select_then_refit_evaluator,
)


def test_unit_mean_cosine_is_invariant_to_positive_per_example_rescaling():
    train = np.array([[1.0, 0.0], [0.1, 1.0], [-1.0, 0.0], [-0.1, -1.0]])
    labels = np.array([0, 0, 1, 1])
    query = np.array([[0.4, 1.0], [-0.4, -1.0]])
    classes = [0, 1]
    scales = np.array([[100.0], [0.2], [3.0], [50.0]])
    original = cosine_predictions(train, labels, query, classes, normalize_before_mean=True)
    rescaled = cosine_predictions(train * scales, labels, query, classes, normalize_before_mean=True)
    assert np.array_equal(original, rescaled)


def test_raw_mean_and_unit_mean_expose_different_prototype_semantics():
    train = np.array(
        [
            [0.90209199, 0.88162679],
            [-0.03679425, -0.02871736],
            [-0.04533866, -0.06140054],
            [-0.04715724, 0.05391478],
            [-2.15076146, -1.84869870],
            [-0.04473134, 0.02776047],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    query = np.array([[-0.51080514, -1.18063218]])
    classes = [0, 1]
    raw = cosine_predictions(train, labels, query, classes, normalize_before_mean=False)
    unit = cosine_predictions(train, labels, query, classes, normalize_before_mean=True)
    assert raw.tolist() != unit.tolist()


def test_relative_regularization_scales_with_global_feature_rescaling():
    rng = np.random.RandomState(4)
    train = rng.randn(24, 5)
    labels = np.repeat([0, 1, 2], 8)
    _, covariance = pooled_within_stats(train, labels, [0, 1, 2])
    base = resolved_regularizers([1e-3, 1e-1], covariance, "relative_trace")
    scale = 7.0
    _, scaled_covariance = pooled_within_stats(train * scale, labels, [0, 1, 2])
    scaled = resolved_regularizers([1e-3, 1e-1], scaled_covariance, "relative_trace")
    assert np.allclose(np.asarray(scaled), np.asarray(base) * scale**2)


def test_relative_lda_predictions_are_invariant_to_global_feature_rescaling():
    rng = np.random.RandomState(7)
    centers = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    labels = np.repeat(np.arange(3), 12)
    train = np.vstack([center + 0.25 * rng.randn(12, 3) for center in centers])
    test = np.vstack([center + 0.25 * rng.randn(5, 3) for center in centers])
    test_labels = np.repeat(np.arange(3), 5)
    inner = np.r_[0:9, 12:21, 24:33]
    validation = np.r_[9:12, 21:24, 33:36]
    result = select_then_refit_evaluator(
        "lda",
        train,
        labels,
        train[inner],
        labels[inner],
        train[validation],
        labels[validation],
        test,
        [0, 1, 2],
        [1e-4, 1e-2, 1.0],
        regularization_mode="relative_trace",
    )
    factor = 13.0
    scaled = select_then_refit_evaluator(
        "lda",
        train * factor,
        labels,
        train[inner] * factor,
        labels[inner],
        train[validation] * factor,
        labels[validation],
        test * factor,
        [0, 1, 2],
        [1e-4, 1e-2, 1.0],
        regularization_mode="relative_trace",
    )
    assert np.array_equal(result["predictions"], scaled["predictions"])
    assert accuracy(result["predictions"], test_labels) == accuracy(scaled["predictions"], test_labels)


def test_selection_tie_breaks_to_first_candidate_without_test_information():
    assert select_index_from_scores([0.5, 0.7, 0.7, 0.6]) == 1
