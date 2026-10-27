import numpy as np

from sdg_paired_statistics import (
    benjamini_hochberg,
    build_group_statistics,
    exact_sign_flip_p,
    paired_summary,
)


def _record(seed, cos, wcos, lda):
    return {
        "dataset": "toy",
        "shot": "5",
        "seed": seed,
        "metrics": {
            "cos": cos,
            "wcos_trainval": {"acc": wcos},
            "lda_trainval": {"acc": lda},
        },
    }


def test_exact_sign_flip_all_negative_has_minimum_one_sided_p():
    values = np.asarray([-1.0, -2.0, -3.0, -4.0, -5.0])
    assert np.isclose(exact_sign_flip_p(values, "less"), 1.0 / 32.0)
    assert np.isclose(exact_sign_flip_p(values, "two-sided"), 2.0 / 32.0)


def test_paired_summary_reports_negative_interval_and_tests():
    summary = paired_summary(np.asarray([-4.0, -3.5, -5.0, -4.5, -4.2]))
    assert summary["mean_pp"] < 0.0
    assert summary["ci95_pp"][1] < 0.0
    assert summary["p_t_less"] < 0.01
    assert np.isclose(summary["p_signflip_less"], 1.0 / 32.0)


def test_best_of_target_separates_oracle_family_selection_uplift():
    records = [
        _record(0, 0.50, 0.60, 0.55),
        _record(1, 0.50, 0.55, 0.65),
        _record(2, 0.50, 0.60, 0.55),
        _record(3, 0.50, 0.55, 0.65),
        _record(4, 0.50, 0.60, 0.55),
    ]
    row = build_group_statistics(records)[0]
    decomposition = row["target_decomposition"]
    assert decomposition["max_target_pp"] >= decomposition["best_fixed_family_target_pp"]
    assert decomposition["oracle_family_selection_uplift_pp"] > 0.0
    assert np.isclose(
        decomposition["max_target_pp"],
        decomposition["best_fixed_family_target_pp"]
        + decomposition["oracle_family_selection_uplift_pp"],
    )


def test_bh_adjustment_is_monotone_in_ranked_p_values():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])
    order = np.argsort([0.01, 0.04, 0.03, 0.002])
    ranked = np.asarray(adjusted)[order]
    assert np.all(ranked[:-1] <= ranked[1:])
    assert np.all(np.asarray(adjusted) >= np.asarray([0.01, 0.04, 0.03, 0.002]))
