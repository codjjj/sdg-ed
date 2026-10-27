from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import sdg_imagenet1k_controlled_multibackbone as audit


def test_exact_imagenet100_mapping_is_frozen_and_unambiguous():
    labels = audit.IMAGENET100_LABELS
    assert len(labels) == 100
    assert len(set(labels)) == 100
    assert min(labels) >= 0
    assert max(labels) < 1000
    assert labels[63] == 494
    assert labels[81] == 593
    assert 577 not in labels
    assert 594 not in labels


def test_train_selection_split_is_class_balanced_and_seeded():
    labels = np.repeat(np.arange(1000, dtype=np.int64), 60)
    first = audit.stable_train_validation_mask(labels, 20260724)
    repeated = audit.stable_train_validation_mask(labels, 20260724)
    second_seed = audit.stable_train_validation_mask(labels, 20260725)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, second_seed)
    for cls in range(1000):
        assert np.sum(first[labels == cls]) == 12


def test_train_selection_split_uses_fullshot_cap():
    labels = np.repeat(np.arange(1000, dtype=np.int64), 1300)
    selected = audit.stable_train_validation_mask(labels, 20260724)
    assert selected.sum() == 50_000
    assert np.all(
        np.bincount(labels[selected], minlength=1000) == np.full(1000, 50)
    )


def test_parquet_row_and_label_order_are_preserved(tmp_path):
    image = Image.new("RGB", (4, 4), color=(10, 20, 30))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    records = [
        {"bytes": buffer.getvalue(), "path": None},
        {"bytes": buffer.getvalue(), "path": None},
    ]
    table = pa.table(
        {
            "image": pa.array(
                records,
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
            ),
            "label": pa.array([17, 3], type=pa.int64()),
        }
    )
    path = tmp_path / "train-00000-of-00001.parquet"
    pq.write_table(table, path)
    batches = list(audit.iter_parquet_batches([path], batch_size=1))
    assert [int(labels[0]) for _, labels in batches] == [17, 3]
    decoded = audit.decode_image(batches[0][0][0], lambda value: np.asarray(value))
    assert decoded.shape == (4, 4, 3)


class _FakeMAE:
    def __call__(self, pixel_values):
        batch = len(pixel_values)
        values = torch.arange(batch * 197 * 5, dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=values.reshape(batch, 197, 5))


def test_mae_pooling_requires_no_mask_full_token_contract():
    pixels = torch.zeros(2, 3, 224, 224)
    pooled = audit.pool_features("mae_vit_base", _FakeMAE(), pixels)
    assert pooled.shape == (2, 5)
    expected = _FakeMAE()(pixels).last_hidden_state[:, 0, :]
    assert torch.equal(pooled, expected)


class _FakeMaskedMAE:
    def __call__(self, pixel_values):
        return SimpleNamespace(last_hidden_state=torch.zeros(len(pixel_values), 50, 5))


def test_mae_pooling_rejects_default_random_mask_contract():
    pixels = torch.zeros(2, 3, 224, 224)
    try:
        audit.pool_features("mae_vit_base", _FakeMaskedMAE(), pixels)
    except AssertionError as error:
        assert "MAE no-mask contract failed" in str(error)
    else:
        raise AssertionError("masked MAE output was not rejected")


def test_repeated_forward_gate_accepts_float_noise_but_rejects_semantic_change():
    first = np.ones((2, 4), dtype=np.float32)
    numerical_noise = first.copy()
    numerical_noise[0, 0] += 2.5e-6
    assert audit.repeated_forward_max_abs(first, numerical_noise) < 1e-5

    changed = first.copy()
    changed[0, 0] += 1e-3
    with pytest.raises(AssertionError, match="repeatability tolerance"):
        audit.repeated_forward_max_abs(first, changed)


def test_shared100_contrast_is_a_paired_difference_in_differences():
    truth = np.repeat(np.asarray(audit.IMAGENET100_LABELS, dtype=np.int64), 2)
    global_truth = np.concatenate([truth, np.asarray([0, 1], dtype=np.int64)])
    exact_indices = np.arange(len(truth))
    exact_cos = truth.copy()
    exact_wcos = truth.copy()
    exact_lda = truth.copy()
    global_cos = global_truth.copy()
    global_wcos = global_truth.copy()
    global_lda = global_truth.copy()
    # On one shared query, WCos fixes a global-1K error; exact-100 is unchanged.
    global_cos[0] = -1
    payload = audit.paired_shared100_scale_contrast(
        {
            "indices": exact_indices,
            "truth": truth,
            "cos": exact_cos,
            "wcos": exact_wcos,
            "lda": exact_lda,
        },
        {
            "indices": np.arange(len(global_truth)),
            "truth": global_truth,
            "cos": global_cos,
            "wcos": global_wcos,
            "lda": global_lda,
        },
        n_boot=200,
        seed=7,
    )
    assert payload["wcos"]["point_pp"] == 0.5
    assert payload["lda"]["point_pp"] == 0.5


def test_subgroup_global_contrast_is_paired_on_the_same_queries():
    global_truth = np.asarray([0, 10, 10, 20, 20, 30], dtype=np.int64)
    global_predictions = global_truth.copy()
    subgroup_indices = np.asarray([1, 2, 3, 4])
    subgroup_truth = global_truth[subgroup_indices]
    subgroup_cos = subgroup_truth.copy()
    subgroup_cos[0] = -1
    payload = audit.paired_subgroup_global_gap_contrast(
        {
            "indices": subgroup_indices,
            "truth": subgroup_truth,
            "cos": subgroup_cos,
            "wcos": subgroup_truth.copy(),
            "lda": subgroup_cos.copy(),
        },
        {
            "indices": np.arange(len(global_truth)),
            "truth": global_truth,
            "cos": global_predictions,
            "wcos": global_predictions.copy(),
            "lda": global_predictions.copy(),
        },
        np.asarray([10, 20], dtype=np.int64),
        n_boot=200,
        seed=7,
    )
    assert payload["wcos"]["point_pp"] == 25.0
    assert payload["lda"]["point_pp"] == 0.0
    assert payload["wcos"]["bootstrap_unit"] == "subgroup ImageNet class"


def test_aggregate_retains_scale_and_subgroup_contrasts():
    arm_result = {
        "test_accuracies": {"cos": 0.5, "wcos": 0.6, "lda": 0.7},
        "fixed_evaluator_gaps": {
            "wcos": {"point_pp": 10.0},
            "lda": {"point_pp": 20.0},
        },
    }
    contrast = {
        "fixed_evaluator_contrasts": {"wcos": {"point_pp": 1.0}},
        "paired_subgroup_global_contrasts": {
            "dogs": {"wcos": {"point_pp": 2.0}},
            "birds": {"wcos": {"point_pp": 3.0}},
            "vehicles": {"wcos": {"point_pp": 4.0}},
        },
    }
    aggregate = audit.summarize_backbones(
        [
            {
                "backbone": "dinov2",
                "backbone_label": "DINOv2-B/14-reg",
                "arms": {
                    arm: [arm_result, arm_result, arm_result]
                    for arm in audit.ARM_ORDER
                },
                "paired_shared100_scale_contrasts": [
                    contrast,
                    contrast,
                    contrast,
                ],
            }
        ]
    )
    assert len(aggregate["rows"]) == 3
    assert aggregate["rows"][0]["paired_shared100_scale_contrast"]["wcos"][
        "point_pp"
    ] == 1.0
    assert aggregate["rows"][0]["paired_subgroup_global_contrasts"]["dogs"][
        "wcos"
    ]["point_pp"] == 2.0


def test_matrix_rank_assignment_covers_each_backbone_once():
    names = tuple(audit.BACKBONES)
    assert len(names) == 6
    assert set(names) == {
        "dinov2",
        "clip_vit_b32",
        "mae_vit_base",
        "siglip_b16",
        "convnext_base",
        "vit_sup_b16",
    }


def test_model_smoke_covers_all_backbones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_make(backbone, device):
        del device
        return (
            torch.nn.Identity(),
            lambda image: torch.zeros(3, image.height, image.width),
            {"backbone": backbone},
        )

    def fake_pool(backbone, model, pixels):
        del model
        dim = audit.BACKBONES[backbone]["expected_dim"]
        return torch.full((len(pixels), dim), float(dim))

    monkeypatch.setattr(audit, "make_model_and_transform", fake_make)
    monkeypatch.setattr(audit, "pool_features", fake_pool)
    args = SimpleNamespace(
        output_root=tmp_path / "smoke",
        device="cpu",
        commit="smoke-commit",
    )
    result = audit.run_model_smoke(args)
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert [row["backbone"] for row in payload["rows"]] == list(audit.BACKBONES)
    assert (args.output_root / "_SUCCESS").read_text(encoding="utf-8") == "ok\n"


def test_source_ready_manifest_rejects_partial_or_wrong_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(audit, "EXPECTED_SOURCE_BYTES", audit.EXPECTED_SOURCE_FILES)
    rows = []
    relatives = [".gitattributes", "README.md"]
    relatives.extend(
        f"data/train-{index:05d}-of-{audit.EXPECTED_TRAIN_SHARDS:05d}.parquet"
        for index in range(audit.EXPECTED_TRAIN_SHARDS)
    )
    relatives.extend(
        f"data/validation-{index:05d}-of-{audit.EXPECTED_VALIDATION_SHARDS:05d}.parquet"
        for index in range(audit.EXPECTED_VALIDATION_SHARDS)
    )
    for relative in relatives:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        rows.append({"path": relative, "size": 1, "sha256": "unused"})
    payload = {
        "repo_id": audit.SOURCE_REPO,
        "revision": audit.SOURCE_REVISION,
        "train_shards": audit.EXPECTED_TRAIN_SHARDS,
        "validation_shards": audit.EXPECTED_VALIDATION_SHARDS,
        "train_rows": audit.EXPECTED_TRAIN_ROWS,
        "validation_rows": audit.EXPECTED_VALIDATION_ROWS,
        "total_bytes": audit.EXPECTED_SOURCE_FILES,
        "files": rows,
    }
    marker = tmp_path / "_READY.json"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    assert audit.validate_source_ready_manifest(tmp_path)["revision"] == audit.SOURCE_REVISION

    payload["revision"] = "wrong"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        audit.validate_source_ready_manifest(tmp_path)


def test_source_content_gate_rejects_same_size_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(audit, "EXPECTED_SOURCE_BYTES", audit.EXPECTED_SOURCE_FILES)
    relatives = [".gitattributes", "README.md"]
    relatives.extend(
        f"data/train-{index:05d}-of-{audit.EXPECTED_TRAIN_SHARDS:05d}.parquet"
        for index in range(audit.EXPECTED_TRAIN_SHARDS)
    )
    relatives.extend(
        f"data/validation-{index:05d}-of-{audit.EXPECTED_VALIDATION_SHARDS:05d}.parquet"
        for index in range(audit.EXPECTED_VALIDATION_SHARDS)
    )
    rows = []
    for relative in relatives:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        rows.append(
            {
                "path": relative,
                "size": 1,
                "sha256": audit.sha256_file(path),
            }
        )
    ready = {
        "repo_id": audit.SOURCE_REPO,
        "revision": audit.SOURCE_REVISION,
        "train_shards": audit.EXPECTED_TRAIN_SHARDS,
        "validation_shards": audit.EXPECTED_VALIDATION_SHARDS,
        "train_rows": audit.EXPECTED_TRAIN_ROWS,
        "validation_rows": audit.EXPECTED_VALIDATION_ROWS,
        "total_bytes": audit.EXPECTED_SOURCE_FILES,
        "files": rows,
    }
    (tmp_path / "_READY.json").write_text(json.dumps(ready), encoding="utf-8")
    verified = audit.verify_source_content(tmp_path, ready)
    assert len(verified["files"]) == audit.EXPECTED_SOURCE_FILES
    output_root = tmp_path / "output"
    rank0 = audit.distributed_source_gate(
        tmp_path, output_root, rank=0, wait_seconds=1
    )
    rank1 = audit.distributed_source_gate(
        tmp_path, output_root, rank=1, wait_seconds=1
    )
    assert rank0 == rank1
    assert (output_root / "_SOURCE_VERIFIED.json").is_file()

    (tmp_path / relatives[-1]).write_bytes(b"y")
    with pytest.raises(ValueError, match="source sha256 mismatch"):
        audit.verify_source_content(tmp_path, ready)
