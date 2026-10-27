import numpy as np
import pytest
import torch

from sdg_lightweight_adaptation_audit import (
    LowRankResidualAdapter,
    dataset_for_rank,
    dataset_or_idle_for_rank,
    evaluate_geometry,
    evaluate_matched_representation_pair,
    paired_gap_change_bootstrap,
    resolve_evaluator_validation_seeds,
    seed_summary,
    select_training_epoch,
    train_full_support,
    transform_features,
)


def _toy(seed=0):
    rng = np.random.RandomState(seed)
    train_features = np.concatenate(
        [rng.normal(loc=label * 1.5, scale=0.3, size=(8, 6)) for label in range(3)]
    ).astype(np.float32)
    train_labels = np.repeat(np.arange(3), 8)
    test_features = np.concatenate(
        [rng.normal(loc=label * 1.5, scale=0.3, size=(5, 6)) for label in range(3)]
    ).astype(np.float32)
    test_labels = np.repeat(np.arange(3), 5)
    return train_features, train_labels, test_features, test_labels


def test_dataset_matrix_rank_mapping_is_complete_and_fail_closed():
    assert [dataset_for_rank(rank) for rank in range(9)] == [
        "aircraft",
        "cub200",
        "cars",
        "food101",
        "pets",
        "cifar100",
        "cifar10",
        "imagenet100",
        "tiny_imagenet",
    ]
    with pytest.raises(ValueError, match="outside"):
        dataset_for_rank(-1)
    with pytest.raises(ValueError, match="outside"):
        dataset_for_rank(9)
    assert dataset_or_idle_for_rank(8) == "tiny_imagenet"
    assert dataset_or_idle_for_rank(9) is None
    assert dataset_or_idle_for_rank(15) is None
    with pytest.raises(ValueError, match="nonnegative"):
        dataset_or_idle_for_rank(-1)


def test_zero_initialized_adapter_is_exact_identity():
    torch.manual_seed(3)
    adapter = LowRankResidualAdapter(7, rank=3, residual_scale=1.0)
    features = torch.randn(11, 7)
    torch.testing.assert_close(adapter(features), features, rtol=0.0, atol=0.0)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 42


def test_head_only_full_training_keeps_representation_exactly_frozen():
    train, labels, test, _ = _toy()
    adapter, _, _ = train_full_support(
        train,
        labels,
        rank=2,
        residual_scale=1.0,
        residual_penalty=0.0,
        epochs=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        batch_size=8,
        seed=5,
        device=torch.device("cpu"),
        train_adapter=False,
    )
    transformed, change = transform_features(
        adapter, test, device=torch.device("cpu"), batch_size=4
    )
    np.testing.assert_array_equal(transformed, test)
    assert change["global_l2_ratio"] == 0.0
    assert change["max_row_l2_ratio"] == 0.0


def test_epoch_selection_is_deterministic_and_uses_only_supplied_support_split():
    train, labels, _, _ = _toy(1)
    inner = np.concatenate([train[labels == label][:6] for label in range(3)])
    inner_labels = np.repeat(np.arange(3), 6)
    validation = np.concatenate([train[labels == label][6:] for label in range(3)])
    validation_labels = np.repeat(np.arange(3), 2)
    kwargs = dict(
        rank=2,
        residual_scale=0.5,
        residual_penalty=1e-3,
        max_epochs=3,
        learning_rate=1e-2,
        weight_decay=1e-4,
        batch_size=8,
        seed=17,
        device=torch.device("cpu"),
        train_adapter=True,
    )
    first = select_training_epoch(
        inner, inner_labels, validation, validation_labels, **kwargs
    )
    second = select_training_epoch(
        inner, inner_labels, validation, validation_labels, **kwargs
    )
    assert first == second
    assert first["selection_uses"] == "inner_support_validation_only"
    assert 1 <= first["selected_epoch"] <= 3


def test_evaluator_selection_and_predictions_do_not_read_test_label_order():
    train, train_labels, test, test_labels = _toy(2)
    first, first_predictions = evaluate_geometry(
        train, train_labels, test, test_labels, validation_seed=0
    )
    permuted_labels = np.roll(test_labels, 1)
    second, second_predictions = evaluate_geometry(
        train, train_labels, test, permuted_labels, validation_seed=0
    )
    assert first["selection"] == second["selection"]
    assert first["validation_whitening_family"] == second["validation_whitening_family"]
    assert first["cosine_validation_accuracy"] == second["cosine_validation_accuracy"]
    for method in first_predictions:
        np.testing.assert_array_equal(first_predictions[method], second_predictions[method])


def test_evaluator_seed_schedule_is_one_to_one_and_rejects_pseudoreplication():
    adapter_seeds = [0, 1, 2, 3, 4]
    resolved, is_multiseed = resolve_evaluator_validation_seeds(
        adapter_seeds,
        evaluator_validation_seed=7,
        evaluator_validation_seeds=[101, 102, 103, 104, 105],
    )
    assert resolved == [101, 102, 103, 104, 105]
    assert is_multiseed is True

    fixed, is_multiseed = resolve_evaluator_validation_seeds(
        adapter_seeds,
        evaluator_validation_seed=7,
        evaluator_validation_seeds=None,
    )
    assert fixed == [7, 7, 7, 7, 7]
    assert is_multiseed is False

    with pytest.raises(ValueError, match="exactly one value"):
        resolve_evaluator_validation_seeds(
            adapter_seeds,
            evaluator_validation_seed=7,
            evaluator_validation_seeds=[101, 102],
        )
    with pytest.raises(ValueError, match="must be unique"):
        resolve_evaluator_validation_seeds(
            adapter_seeds,
            evaluator_validation_seed=7,
            evaluator_validation_seeds=[101, 101, 102, 103, 104],
        )


def test_matched_pair_uses_the_same_evaluator_split_for_both_representations():
    train, train_labels, test, test_labels = _toy(4)
    adapted_train = train.copy()
    adapted_test = test.copy()
    adapted_train[:, 0] *= 1.7
    adapted_test[:, 0] *= 1.7

    frozen, frozen_predictions, adapted, adapted_predictions = (
        evaluate_matched_representation_pair(
            train,
            train_labels,
            test,
            test_labels,
            adapted_train,
            adapted_test,
            validation_seed=101,
        )
    )
    assert frozen["validation_split"] == adapted["validation_split"]
    assert frozen["validation_split"]["seed"] == 101
    assert set(frozen_predictions) == set(adapted_predictions)
    assert all(
        prediction.shape == test_labels.shape
        for prediction in [*frozen_predictions.values(), *adapted_predictions.values()]
    )


def test_adapted_and_frozen_heads_share_identical_test_example_axis():
    train, labels, test, test_labels = _toy(3)
    adapter, _, _ = train_full_support(
        train,
        labels,
        rank=2,
        residual_scale=0.5,
        residual_penalty=1e-3,
        epochs=2,
        learning_rate=1e-2,
        weight_decay=1e-4,
        batch_size=8,
        seed=23,
        device=torch.device("cpu"),
        train_adapter=True,
    )
    adapted_train, _ = transform_features(
        adapter, train, device=torch.device("cpu"), batch_size=8
    )
    adapted_test, _ = transform_features(
        adapter, test, device=torch.device("cpu"), batch_size=8
    )
    _, frozen_predictions = evaluate_geometry(
        train, labels, test, test_labels, validation_seed=1
    )
    _, adapted_predictions = evaluate_geometry(
        adapted_train, labels, adapted_test, test_labels, validation_seed=1
    )
    assert set(frozen_predictions) == set(adapted_predictions)
    assert all(value.shape == test_labels.shape for value in frozen_predictions.values())
    assert all(value.shape == test_labels.shape for value in adapted_predictions.values())


def test_gap_change_bootstrap_matches_exact_per_example_contrast():
    labels = np.asarray([0, 1, 0, 1])
    adapted_whitening = np.asarray([0, 1, 0, 1])
    adapted_cosine = np.asarray([0, 0, 0, 0])
    frozen_whitening = np.asarray([0, 1, 1, 1])
    frozen_cosine = np.asarray([0, 0, 1, 1])
    result = paired_gap_change_bootstrap(
        adapted_whitening,
        adapted_cosine,
        frozen_whitening,
        frozen_cosine,
        labels,
        n_boot=200,
        seed=9,
    )
    adapted_gap = 100 * (
        np.mean(adapted_whitening == labels) - np.mean(adapted_cosine == labels)
    )
    frozen_gap = 100 * (
        np.mean(frozen_whitening == labels) - np.mean(frozen_cosine == labels)
    )
    assert result["point_pp"] == adapted_gap - frozen_gap
    assert result["n_test"] == 4


def test_seed_summary_retains_paired_values_and_sign_counts():
    result = seed_summary([2.0, 1.0, 0.0, -1.0, 3.0], seed=11, n_boot=200)
    assert result["n_seeds"] == 5
    assert result["positive_seeds"] == 3
    assert result["zero_seeds"] == 1
    assert result["negative_seeds"] == 1
    assert result["values_pp"] == [2.0, 1.0, 0.0, -1.0, 3.0]
