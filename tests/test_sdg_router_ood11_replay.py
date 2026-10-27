import hashlib
import json

import numpy as np
import pytest

from sdg_router_ood11_replay import (
    FINE_DATASETS,
    SHOTS,
    compose_rows,
    parse_test_label_inputs,
    validate_fine_results,
)
from sdg_stepb_predictive_protocol import FEATURES


def _row(dataset, shot):
    row = {
        "dataset": dataset,
        "shot": shot,
        "cos": 0.5,
        "wcos_trainval": 0.51,
        "lda_trainval": 0.52,
        "best_whitening": 0.52,
        "target_best_whitening_minus_cos_pp": 2.0,
    }
    row.update({feature: 1.0 for feature in FEATURES})
    return row


def _ood8():
    datasets = [f"original_{index}" for index in range(4)]
    original = [_row(dataset, shot) for dataset in datasets for shot in SHOTS]
    datasets = [f"coarse_{index}" for index in range(4)]
    coarse = [_row(dataset, shot) for dataset in datasets for shot in SHOTS]
    return {
        "original_heldout4": {"records": original},
        "scope_heldout4": {"records": coarse},
    }


def _fine_rows():
    return [_row(dataset, shot) for dataset in FINE_DATASETS for shot in SHOTS]


def test_compose_rows_locks_11_datasets_and_66_groups():
    rows, blocks = compose_rows(_ood8(), _fine_rows())
    assert len(rows) == 66
    assert len({row["dataset"] for row in rows}) == 11
    assert {name: len(values) for name, values in blocks.items()} == {
        "original_ood4": 24,
        "coarse_ood4": 24,
        "finegrained_ood3": 18,
    }


def test_compose_rows_rejects_inaturalist_even_if_total_count_is_unchanged():
    fine = _fine_rows()
    fine[0] = {**fine[0], "dataset": "inat2021_mini"}
    with pytest.raises(ValueError, match="iNaturalist|composition"):
        compose_rows(_ood8(), fine)


def _write_grid(root, uncapped_nabirds=False):
    paths = []
    for dataset in FINE_DATASETS:
        for shot in SHOTS:
            cell = root / dataset / shot
            predictions = cell / "predictions"
            predictions.mkdir(parents=True)
            seeds = [0] if shot == "full" else list(range(5))
            records = []
            for seed in seeds:
                n_classes = 3
                if shot == "full":
                    n_train = 30
                else:
                    n_train = int(shot) * n_classes
                    if dataset == "nabirds" and shot in {"5", "10", "20"} and not uncapped_nabirds:
                        n_train -= 1
                identities = {}
                for method in ("cos", "wcos_trainval", "lda_trainval"):
                    path = predictions / f"seed_{seed}_{method}.npy"
                    np.save(path, np.asarray([0, 1, 2], dtype=np.int64))
                    identities[method] = {
                        "path": f"predictions/{path.name}",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "n_rows": 3,
                    }
                records.append(
                    {
                        "seed": seed,
                        "n_classes": n_classes,
                        "n_train": n_train,
                        "support": {"n_indices": n_train},
                        "predictions": identities,
                    }
                )
            result = cell / "result.json"
            result.write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "shot": shot,
                        "dataset_identity_sha256": dataset[0] * 64,
                        "feature_manifest_sha256": dataset[-1] * 64,
                        "feature_root": f"/features/{dataset}",
                        "records": records,
                        "group": {"dataset": dataset, "shot": shot},
                    }
                )
                + "\n"
            )
            digest = hashlib.sha256(result.read_bytes()).hexdigest()
            (cell / "RESULT_READY.json").write_text(json.dumps({"result_sha256": digest}) + "\n")
            paths.append(result)
    return paths


def test_result_grid_locks_capped_k_and_prediction_hashes(tmp_path):
    paths = _write_grid(tmp_path)
    rows, entries = validate_fine_results(paths)
    assert len(rows) == len(entries) == 18
    assert sum(len(entry["predictions"]) for entry in entries) == 234
    prediction = tmp_path / "nabirds" / "5" / "predictions" / "seed_0_cos.npy"
    prediction.write_bytes(b"changed")
    with pytest.raises(ValueError, match="prediction hash mismatch"):
        validate_fine_results(paths)


def test_result_grid_rejects_uncapped_nabirds_claim(tmp_path):
    paths = _write_grid(tmp_path, uncapped_nabirds=True)
    with pytest.raises(ValueError, match="capped-K evidence missing"):
        validate_fine_results(paths)


def test_test_label_mapping_requires_exact_three_datasets(tmp_path):
    paths = []
    for dataset in FINE_DATASETS:
        path = tmp_path / f"{dataset}.pt"
        path.write_bytes(dataset.encode())
        paths.append(f"{dataset}={path}")
    assert set(parse_test_label_inputs(paths)) == set(FINE_DATASETS)
    with pytest.raises(ValueError, match="exactly"):
        parse_test_label_inputs(paths[:-1])
