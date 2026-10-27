from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from sdg_simplecil_fulldata_evaluators import (
    SPLIT_SALT,
    _cosine_predictions,
    _total_stats,
    _within_stats,
    fixed_train_validation_split,
    select_regularizer,
    validate_protocol,
)


PROTOCOL = Path(__file__).resolve().parents[1] / "configs" / "simplecil_fulldata_evaluators_protocol.json"
PROTOCOL_SHA = "9472fb3cf2529dcac7ae9103dc092cbefc3132fc2a778ee19af3e045cee25c52"


def test_frozen_protocol_hash_and_history_memory_assumption():
    payload = validate_protocol(PROTOCOL, PROTOCOL_SHA)
    assert payload["schedule"]["sensitivity_seeds"] == list(range(20))
    assert payload["memory_assumption"] == {
        "retains_historical_train_features": True,
        "retains_only_sufficient_statistics": False,
        "declared_reason": (
            "evaluator-only diagnostic reuses the immutable 5994x768 canonical feature cache; "
            "it is not an exemplar-free or memory-matched SimpleCIL system claim"
        ),
    }


def test_protocol_hash_drift_fails_closed(tmp_path):
    changed = tmp_path / "protocol.json"
    changed.write_bytes(PROTOCOL.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_protocol(changed, PROTOCOL_SHA)


def test_train_only_split_is_disjoint_complete_and_path_stable():
    paths = [f"class{label}/image{index}.jpg" for label in range(3) for index in range(5)]
    labels = np.repeat(np.arange(3), 5)
    split = fixed_train_validation_split(paths, labels)
    assert len(split["inner_rows"]) == 12
    assert len(split["validation_rows"]) == 3
    assert not set(split["inner_rows"]) & set(split["validation_rows"])
    assert sorted(np.concatenate([split["inner_rows"], split["validation_rows"]]).tolist()) == list(
        range(15)
    )
    selected_paths = {paths[index] for index in split["validation_rows"]}
    expected = set()
    for label in range(3):
        candidates = list(range(label * 5, (label + 1) * 5))
        ranked = sorted(
            candidates,
            key=lambda row: hashlib.sha256(
                f"{SPLIT_SALT}\0{label}\0{paths[row]}".encode()
            ).digest(),
        )
        expected.add(paths[ranked[0]])
    assert selected_paths == expected


def test_covariance_paths_match_direct_formulas():
    features = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=torch.float64
    )
    labels = torch.tensor([0, 0, 1, 1])
    mean, total = _total_stats(features)
    centered = features - features.mean(0)
    assert torch.allclose(mean, torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert torch.allclose(total, centered.T @ centered / 3)
    means, within = _within_stats(features, labels, [0, 1])
    assert torch.allclose(means, torch.tensor([[1.0, 0.0], [1.0, 2.0]], dtype=torch.float64))
    expected_scatter = torch.tensor([[4.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(within, expected_scatter / 2)


def test_predictions_are_restricted_to_supplied_seen_classes():
    train = torch.eye(4, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 3])
    query = torch.stack([train[0], train[2]])
    prediction = _cosine_predictions(train[[0, 2]], labels[[0, 2]], query, [0, 2])
    assert prediction.tolist() == [0, 2]
    assert set(prediction.tolist()) <= {0, 2}


def test_regularizer_tie_uses_first_candidate_without_test_labels():
    inner = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float64
    )
    labels = torch.tensor([0, 0, 1, 1])
    validation = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    validation_labels = torch.tensor([0, 1])
    candidates = [1e-8, 1e-4, 0.1]
    selected, scores = select_regularizer(
        "wcos", inner, labels, validation, validation_labels, [0, 1], candidates
    )
    assert scores == [1.0, 1.0, 1.0]
    assert selected == candidates[0]


def test_split_rejects_duplicate_path_identity():
    with pytest.raises(ValueError, match="unique aligned"):
        fixed_train_validation_split(["same", "same"], np.asarray([0, 0]))
