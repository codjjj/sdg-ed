import numpy as np

from sdg_router_ood8_eval import (
    covariance_diagnostics,
    diagnostics_for_training_group,
    frozen_router_predict,
)
from sdg_stepb_predictive_protocol import FEATURES


def _router_row(offset, target):
    row = {feature: float(offset + index) for index, feature in enumerate(FEATURES)}
    row["target"] = float(target)
    return row


def test_frozen_router_prediction_is_invariant_to_other_evaluation_rows():
    training = [_router_row(index, 2.0 * index - 1.0) for index in range(12)]
    first = [_router_row(20, 0.0)]
    extra = [_router_row(1000, 0.0)]
    prediction_alone = frozen_router_predict(training, first, "target")
    prediction_with_extra = frozen_router_predict(training, first + extra, "target")
    assert np.allclose(prediction_alone, prediction_with_extra[:1])


def test_training_group_diagnostics_are_deterministic_and_finite():
    rng = np.random.RandomState(7)
    labels = np.repeat(np.arange(3), 6)
    features = rng.normal(size=(len(labels), 16))
    first = diagnostics_for_training_group(features, labels, "2", [0, 1])
    second = diagnostics_for_training_group(features, labels, "2", [0, 1])
    assert first == second
    assert set(FEATURES).issubset(first)
    assert np.isfinite([first[key] for key in FEATURES]).all()


def test_covariance_diagnostics_handle_rank_deficiency():
    features = np.repeat(np.asarray([[1.0, 2.0, 3.0]]), repeats=5, axis=0)
    diagnostics = covariance_diagnostics(features)
    assert diagnostics["log10_condition"] == 0.0
    assert diagnostics["effective_rank"] == 0.0
