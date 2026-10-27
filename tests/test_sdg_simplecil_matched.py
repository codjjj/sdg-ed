from __future__ import annotations

import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import sdg_simplecil_matched as module

from sdg_simplecil_matched import (
    MODEL_REPO,
    MODEL_REVISION,
    evaluate_simplecil_seed,
    materialize_model,
    mean_sample_sd,
    select_cls_token,
    simplecil_task_classes,
)


def test_cls_token_is_first_token_for_every_batch_member():
    hidden = torch.arange(3 * 5 * 7).reshape(3, 5, 7)
    selected = select_cls_token(hidden)
    assert selected.shape == (3, 7)
    assert torch.equal(selected, hidden[:, 0])


def test_cls_token_rejects_non_sequence_tensor():
    with pytest.raises(ValueError, match="hidden state"):
        select_cls_token(torch.zeros(3, 7))


def test_task_partition_matches_upstream_shuffle_then_interleave():
    expected = list(range(200))
    random.seed(2)
    random.shuffle(expected)
    tasks = simplecil_task_classes(2)
    assert tasks == [expected[index::10] for index in range(10)]
    assert all(len(task) == 20 for task in tasks)


def test_simplecil_cosine_uses_all_seen_classes_and_full_class_means():
    features = np.eye(200, dtype=np.float64)
    labels = np.arange(200, dtype=np.int64)
    metrics, predictions = evaluate_simplecil_seed(features, labels, features, labels, seed=0)
    assert metrics["average_accuracy"] == 1.0
    assert metrics["final_accuracy"] == 1.0
    assert len(predictions) == 10
    first_seen = set(metrics["task_classes"][0])
    assert set(np.flatnonzero(predictions[0] >= 0)) == first_seen
    assert np.array_equal(predictions[-1], labels.astype(np.int16))


def test_sample_sd_is_ddof_one():
    result = mean_sample_sd([1.0, 2.0, 4.0])
    assert result["mean"] == pytest.approx(7 / 3)
    assert result["sample_sd"] == pytest.approx(np.std([1.0, 2.0, 4.0], ddof=1))


def test_existing_model_root_requires_dinov2_base_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "dinov2", "hidden_size": 768}), encoding="utf-8"
    )
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    manifest = materialize_model(tmp_path, MODEL_REPO, MODEL_REVISION)
    assert manifest["repo_id"] == MODEL_REPO
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["files"]["model.safetensors"]["bytes"] == 7


def test_existing_model_root_rejects_dino_v1(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "vit", "hidden_size": 768}), encoding="utf-8"
    )
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="unexpected DINOv2-base"):
        materialize_model(tmp_path, MODEL_REPO, MODEL_REVISION)


def test_model_materialization_downloads_pinned_snapshot(tmp_path, monkeypatch):
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs["revision"], kwargs["local_dir"]))
        root = Path(kwargs["local_dir"])
        (root / "config.json").write_text(
            json.dumps({"model_type": "dinov2", "hidden_size": 768}), encoding="utf-8"
        )
        (root / "preprocessor_config.json").write_text("{}", encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"fixture")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download)
    )
    manifest = materialize_model(tmp_path, MODEL_REPO, MODEL_REVISION)
    assert calls == [(MODEL_REPO, MODEL_REVISION, tmp_path)]
    assert manifest["revision"] == MODEL_REVISION
    assert set(manifest["files"]) == {"config.json", "preprocessor_config.json", "model.safetensors"}


def test_model_materialization_skips_download_when_snapshot_exists(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "dinov2", "hidden_size": 768}), encoding="utf-8"
    )
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fixture")

    def snapshot_download(*args, **kwargs):
        raise AssertionError("download must not run for a complete local snapshot")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download)
    )
    manifest = materialize_model(tmp_path, MODEL_REPO, MODEL_REVISION)
    assert manifest["repo_id"] == MODEL_REPO
