from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from sdg_encoder_finetune_audit import (
    configure_finetuning,
    configure_full_encoder_finetuning,
    configure_last_block_finetuning,
    evaluate_geometry,
    forward_features,
    paired_gap_change_bootstrap,
    stratified_train_validation_indices,
)


class _DummyBlock(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear = nn.Linear(dimension, dimension)

    def forward(self, hidden):
        return torch.tanh(self.linear(hidden))


class _DummyEncoder(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.layer = nn.ModuleList([_DummyBlock(dimension), _DummyBlock(dimension)])


class _DummyVisionModel(nn.Module):
    def __init__(self, dimension=5):
        super().__init__()
        self.embedding = nn.Linear(3, dimension)
        self.encoder = _DummyEncoder(dimension)
        self.layernorm = nn.LayerNorm(dimension)

    def forward(self, pixel_values):
        hidden = self.embedding(pixel_values).unsqueeze(1)
        for block in self.encoder.layer:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=self.layernorm(hidden))


def test_stratified_split_is_deterministic_disjoint_and_budgeted():
    labels = np.repeat(np.arange(3), 8)
    first = stratified_train_validation_indices(
        labels, seed=7, validation_fraction=0.25, max_total_per_class=6
    )
    second = stratified_train_validation_indices(
        labels, seed=7, validation_fraction=0.25, max_total_per_class=6
    )
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert len(first["selected"]) == 18
    assert len(first["ignored"]) == 6
    assert set(first["fit"]) & set(first["validation"]) == set()
    assert set(np.unique(labels[first["fit"]])) == {0, 1, 2}
    assert set(np.unique(labels[first["validation"]])) == {0, 1, 2}


def test_only_last_block_and_final_norm_update_and_features_change():
    torch.manual_seed(3)
    model = _DummyVisionModel()
    scope = configure_last_block_finetuning(model)
    assert scope["trainable_fraction"] < 1.0
    assert not model.embedding.weight.requires_grad
    assert not model.encoder.layer[0].linear.weight.requires_grad
    assert model.encoder.layer[-1].linear.weight.requires_grad
    assert model.layernorm.weight.requires_grad

    images = torch.randn(12, 3)
    labels = torch.arange(12) % 3
    before_frozen = model.encoder.layer[0].linear.weight.detach().clone()
    before_features = forward_features(model, images).detach().clone()
    head = nn.Linear(before_features.shape[1], 3)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
        + list(head.parameters()),
        lr=0.1,
    )
    loss = nn.CrossEntropyLoss()(head(forward_features(model, images)), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.testing.assert_close(
        model.encoder.layer[0].linear.weight, before_frozen, rtol=0.0, atol=0.0
    )
    after_features = forward_features(model, images).detach()
    assert not torch.equal(before_features, after_features)


def test_full_encoder_scope_updates_early_and_late_parameters():
    torch.manual_seed(4)
    model = _DummyVisionModel()
    scope = configure_full_encoder_finetuning(model)
    assert scope["trainable_fraction"] == 1.0
    assert scope["updated_modules"] == ["full_encoder"]
    assert all(parameter.requires_grad for parameter in model.parameters())

    images = torch.randn(12, 3)
    labels = torch.arange(12) % 3
    before_embedding = model.embedding.weight.detach().clone()
    before_early = model.encoder.layer[0].linear.weight.detach().clone()
    before_late = model.encoder.layer[-1].linear.weight.detach().clone()
    head = nn.Linear(model.layernorm.normalized_shape[0], 3)
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(head.parameters()), lr=0.1
    )
    loss = nn.CrossEntropyLoss()(head(forward_features(model, images)), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.equal(model.embedding.weight, before_embedding)
    assert not torch.equal(model.encoder.layer[0].linear.weight, before_early)
    assert not torch.equal(model.encoder.layer[-1].linear.weight, before_late)


def test_scope_dispatch_preserves_last_block_default_and_rejects_unknown_scope():
    model = _DummyVisionModel()
    scope = configure_finetuning(model, "last_block")
    assert scope["updated_modules"] == [
        "last_transformer_block",
        "final_output_normalization",
    ]
    assert not model.embedding.weight.requires_grad
    assert model.encoder.layer[-1].linear.weight.requires_grad
    with pytest.raises(ValueError, match="unsupported fine-tuning scope"):
        configure_finetuning(_DummyVisionModel(), "unknown")


def _toy_features(seed=0):
    rng = np.random.RandomState(seed)
    train = np.concatenate(
        [rng.normal(label, 0.25, size=(8, 6)) for label in range(3)]
    ).astype(np.float32)
    train_labels = np.repeat(np.arange(3), 8)
    test = np.concatenate(
        [rng.normal(label, 0.25, size=(5, 6)) for label in range(3)]
    ).astype(np.float32)
    test_labels = np.repeat(np.arange(3), 5)
    return train, train_labels, test, test_labels


def test_evaluator_selection_does_not_read_test_labels():
    train, train_labels, test, test_labels = _toy_features(5)
    first, first_predictions = evaluate_geometry(
        train, train_labels, test, test_labels, validation_seed=17
    )
    second, second_predictions = evaluate_geometry(
        train, train_labels, test, np.roll(test_labels, 2), validation_seed=17
    )
    assert first["selection"] == second["selection"]
    assert (
        first["validation_whitening_family"]
        == second["validation_whitening_family"]
    )
    assert first["validation_split"]["test_labels_used"] is False
    for method in first_predictions:
        np.testing.assert_array_equal(
            first_predictions[method], second_predictions[method]
        )


def test_gap_change_bootstrap_matches_exact_paired_contrast():
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
        seed=11,
    )
    adapted_gap = 100 * (
        np.mean(adapted_whitening == labels) - np.mean(adapted_cosine == labels)
    )
    frozen_gap = 100 * (
        np.mean(frozen_whitening == labels) - np.mean(frozen_cosine == labels)
    )
    assert result["point_pp"] == adapted_gap - frozen_gap
    assert result["n_test"] == 4
