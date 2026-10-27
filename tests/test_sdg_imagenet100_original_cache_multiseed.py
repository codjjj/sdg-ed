import math

import pytest

from sdg_backbone_bootstrap_ci import BACKBONES
from sdg_imagenet100_original_cache_multiseed import (
    METHODS,
    SELECTION_SEEDS,
    summarize_cells,
)


def _cells():
    cells = []
    for backbone_index, backbone in enumerate(BACKBONES):
        for seed_index, seed in enumerate(SELECTION_SEEDS):
            cosine = 0.5 + backbone_index / 100.0
            cells.append(
                {
                    "backbone": backbone,
                    "selection_seed": seed,
                    "accuracies": {
                        "cos": cosine,
                        "wcos_relative": cosine + (seed_index + 1) / 100.0,
                        "lda_relative": cosine - (seed_index + 1) / 200.0,
                    },
                }
            )
    return cells


def test_summary_uses_percentage_point_gaps_and_sample_sd():
    rows = summarize_cells(_cells())
    assert [row["backbone"] for row in rows] == list(BACKBONES)
    for row in rows:
        wcos = row[f"{METHODS[0]}_minus_cosine_pp"]
        lda = row[f"{METHODS[1]}_minus_cosine_pp"]
        assert math.isclose(wcos["mean"], 2.0)
        assert math.isclose(wcos["sample_sd"], 1.0)
        assert math.isclose(lda["mean"], -1.0)
        assert math.isclose(lda["sample_sd"], 0.5)


def test_summary_rejects_an_incomplete_matrix():
    with pytest.raises(ValueError, match="incomplete cell matrix"):
        summarize_cells(_cells()[:-1])


def test_summary_rejects_duplicate_cells():
    cells = _cells()
    with pytest.raises(ValueError, match="duplicate"):
        summarize_cells(cells + [cells[0]])
