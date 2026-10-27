import hashlib
import json

import numpy as np
import pytest

from sdg_finegrained_manifest import canonical_json_bytes

from verify_sdg_router_ood11_replay import (
    SHOTS,
    confusion,
    independent_bootstrap,
    policy_regrets,
    prediction_metrics,
    residual_metrics,
    validate_source_manifest,
)


def _rows():
    rows = []
    for dataset in ("a", "b", "c"):
        for index, shot in enumerate(SHOTS):
            target = float(index - 2)
            predicted = target + {"a": -0.5, "b": 0.0, "c": 0.5}[dataset]
            rows.append(
                {
                    "dataset": dataset,
                    "shot": shot,
                    "router_prediction_pp": predicted,
                    "target_best_whitening_minus_cos_pp": target,
                    "cos": 0.5,
                    "wcos_trainval": 0.5 + target / 100.0,
                    "lda_trainval": 0.49 + target / 100.0,
                    "best_whitening": 0.5 + target / 100.0,
                }
            )
    return rows


def test_independent_metrics_cover_regret_and_demeaned_correlation():
    rows = _rows()
    assert prediction_metrics(rows)["mae_pp"] > 0
    assert set(policy_regrets(rows)) == {
        "router_oracle_family",
        "always_wcos",
        "always_lda",
        "always_cos",
    }
    assert set(residual_metrics(rows)) == {"pearson", "spearman", "mae_pp", "rmse_pp"}


def test_dataset_cluster_bootstrap_is_seeded_and_retains_clusters():
    rows = _rows()
    first = independent_bootstrap(rows, n_boot=200, seed=20260803)
    second = independent_bootstrap(rows, n_boot=200, seed=20260803)
    other = independent_bootstrap(rows, n_boot=200, seed=20260804)
    assert first == second
    assert first != other


def test_confusion_preserves_all_negative_and_positive_cells():
    rows = _rows()
    counts = confusion(rows)
    assert sum(counts.values()) == len(rows)
    assert set(counts) == {
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    }


def test_nonrank_metrics_are_independent_of_row_order():
    rows = _rows()
    reverse = list(reversed(rows))
    # The frozen protocol uses ordinal ranks without tie averaging, so the
    # exact dataset/shot order is part of the Spearman replay contract.
    for name in ("pearson", "mae_pp", "rmse_pp"):
        assert np.isclose(prediction_metrics(rows)[name], prediction_metrics(reverse)[name])


def test_source_manifest_rejects_inaturalist_or_incomplete_coverage(tmp_path):
    prediction_identity = [
        {"seed": 0, "method": "cos", "path": "/tmp/cos.npy", "relative_path": "predictions/cos.npy", "sha256": "c" * 64, "n_rows": 3}
    ]
    portable_identity = [
        {name: row[name] for name in ("seed", "method", "relative_path", "sha256", "n_rows")}
        for row in prediction_identity
    ]
    identity_sha = hashlib.sha256(canonical_json_bytes(portable_identity)).hexdigest()
    results = [
        {"dataset": dataset, "shot": shot, "result_sha256": "a" * 64, "prediction_identity_sha256": identity_sha}
        for dataset in ("stanford_dogs", "foodx251", "nabirds")
        for shot in SHOTS
    ]
    source = {
        "accepted_task_ids": {
            "stanford_dogs": "producer-dogs",
            "foodx251": "producer-foodx",
            "nabirds": "producer-nabirds",
        },
        "excluded_datasets": ["inat2021_mini"],
        "results": results,
        "test_features": {
            dataset: {"sha256": "b" * 64}
            for dataset in ("stanford_dogs", "foodx251", "nabirds")
        },
    }
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(source) + "\n")
    aggregate = {
        "inputs": {
            "source_manifest": {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            "finegrained_results": [
                {**row, "predictions": prediction_identity}
                for row in results
            ],
            "test_label_features": {
                dataset: {"sha256": "b" * 64}
                for dataset in ("stanford_dogs", "foodx251", "nabirds")
            },
        }
    }
    assert validate_source_manifest(aggregate)["results"] == 18
    source["excluded_datasets"] = []
    path.write_text(json.dumps(source) + "\n")
    aggregate["inputs"]["source_manifest"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="iNaturalist exclusion"):
        validate_source_manifest(aggregate)
