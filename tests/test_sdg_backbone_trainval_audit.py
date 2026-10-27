import numpy as np

from sdg_audit_core import cosine_predictions
from sdg_backbone_trainval_audit import (
    METHODS,
    frozen_cosine_predictions,
    paired_ranking_bootstrap,
)


def _correct():
    base = np.array(
        [
            [True, True, False, False, True, False],
            [True, False, False, False, True, False],
        ],
        dtype=bool,
    )
    return {method: base.copy() for method in METHODS}


def test_identical_predictions_have_zero_ranking_regret():
    result = paired_ranking_bootstrap(_correct(), ["a", "b"], n_boot=100, seed=7)
    for row in result.values():
        assert row["observed_regret_pp"] == 0.0
        assert row["bootstrap_regret_ci95_pp"] == [0.0, 0.0]


def test_bootstrap_is_seed_deterministic():
    correct = _correct()
    correct["validation_whitening_absolute"][1] = np.array([True] * 6)
    left = paired_ranking_bootstrap(correct, ["a", "b"], n_boot=50, seed=11)
    right = paired_ranking_bootstrap(correct, ["a", "b"], n_boot=50, seed=11)
    assert left == right


def test_regret_is_nonnegative_for_every_replicate():
    correct = _correct()
    correct["validation_whitening_absolute"][1] = np.array([True] * 6)
    result = paired_ranking_bootstrap(correct, ["a", "b"], n_boot=100, seed=3)
    samples = result["validation_whitening_absolute"]["bootstrap_samples"]["regret_pp"]
    assert min(samples) >= 0.0


def test_cosine_baseline_uses_submitted_torch_float32_semantics():
    train = np.array(
        [
            [0.5576744079589844, 0.09914659708738327],
            [0.6379261612892151, 0.7031106948852539],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1])
    query = np.array([[1.6565039001558184, 0.9156441971318662]], dtype=np.float64)

    submitted = frozen_cosine_predictions(train, labels, query, [0, 1])
    float64_reimplementation = cosine_predictions(train, labels, query, [0, 1])

    assert submitted.tolist() == [0]
    assert float64_reimplementation.tolist() == [1]
