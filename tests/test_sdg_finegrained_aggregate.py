import hashlib
import json

import numpy as np
import pytest

from sdg_finegrained_aggregate import (
    FINE_DATASETS,
    FINE_DATASET_SHOTS,
    add_predictions,
    cluster_bootstrap,
    confusion,
    point_summary,
    shot_fixed_effect_residual,
    validate_fine_results,
)


def _write_result(root, dataset, shot):
    cell = root / dataset / shot
    cell.mkdir(parents=True)
    result = cell / "result.json"
    result.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "shot": shot,
                "dataset_identity_sha256": "d" * 64,
                "feature_manifest_sha256": "f" * 64,
                "records": [],
                "group": {"dataset": dataset, "shot": shot},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(result.read_bytes()).hexdigest()
    (cell / "RESULT_READY.json").write_text(
        json.dumps({"result_sha256": digest}) + "\n", encoding="utf-8"
    )
    return result


def _available_result_paths(tmp_path):
    return [
        _write_result(tmp_path, dataset, shot)
        for dataset in FINE_DATASETS
        for shot in FINE_DATASET_SHOTS[dataset]
    ]


def test_fine_result_identity_requires_complete_capped_nabirds_grid(tmp_path):
    paths = _available_result_paths(tmp_path)
    rows, inputs = validate_fine_results(paths)
    assert len(rows) == len(inputs) == 24
    observed_nabirds = {
        row["shot"] for row in rows if row["dataset"] == "nabirds"
    }
    assert observed_nabirds == {"1", "2", "5", "10", "20", "full"}


def test_fine_result_identity_rejects_any_other_missing_cell(tmp_path):
    paths = _available_result_paths(tmp_path)
    with pytest.raises(ValueError, match="24-cell identity mismatch"):
        validate_fine_results(paths[:-1])


def test_fine_result_identity_rejects_duplicate_nabirds_cell(tmp_path):
    paths = _available_result_paths(tmp_path)
    duplicate_root = tmp_path / "duplicate"
    paths.append(_write_result(duplicate_root, "nabirds", "20"))
    with pytest.raises(ValueError, match="duplicate fine-grained result"):
        validate_fine_results(paths)


def _rows(n_datasets=4):
    rows = []
    shots = ["1", "2", "5", "10", "20", "full"]
    for dataset in range(n_datasets):
        for shot_index, shot in enumerate(shots):
            cos = 0.5 + 0.01 * dataset
            gap = float(shot_index - 2)
            rows.append(
                {
                    "dataset": f"d{dataset}",
                    "shot": shot,
                    "cos": cos,
                    "wcos_trainval": cos + gap / 100.0,
                    "lda_trainval": cos + (gap - 1) / 100.0,
                    "best_whitening": cos + gap / 100.0,
                    "target_best_whitening_minus_cos_pp": gap,
                }
            )
    predictions = np.asarray(
        [row["target_best_whitening_minus_cos_pp"] for row in rows]
    )
    return add_predictions(rows, predictions)


def test_policy_summary_and_cluster_bootstrap_include_all_required_policies():
    rows = _rows()
    point = point_summary(rows)
    assert point["prediction"]["mae_pp"] == 0.0
    assert set(point["policy_regret_pp"]) == {
        "router_oracle_family",
        "always_wcos",
        "always_lda",
        "always_cos",
    }
    bootstrap = cluster_bootstrap(rows, n_boot=20, seed=3)
    for policy in point["policy_regret_pp"]:
        assert f"policy_regret_pp.{policy}" in bootstrap["ci95"]


def test_confusion_matrix_is_exact():
    rows = _rows()
    result = confusion(rows)
    assert result["false_positive"] == 0
    assert result["false_negative"] == 0
    assert result["true_positive"] + result["true_negative"] == len(rows)


def test_shot_fixed_effect_removes_shot_only_signal():
    rows = _rows()
    metrics = shot_fixed_effect_residual(rows)
    assert abs(metrics["pearson"]) < 1e-8
    assert abs(metrics["rmse_pp"]) < 1e-12
