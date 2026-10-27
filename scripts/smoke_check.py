#!/usr/bin/env python3
"""Small release sanity check.

This script verifies that the bundled result artifacts needed by the manuscript
renderer and the recomputation scripts are present, that release files do not
contain project-specific absolute paths or infrastructure names, and that a few
headline numbers match the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = [
    "method_matrix_aggregate.json",
    "p0_mechanism_v2/summary_with_pred.json",
    "p3_estimator_defense/aggregate.json",
    "stepb_cached_eval_v1/aggregate_groups.json",
    "stepb_predictive_protocol_v1/predictive_results.json",
    "stepb_predictive_protocol_v2/feature_ablation.json",
    "supplement_stats_v1/aggregate_groups.json",
    "bootstrap_ci_v1/fullshot_ci.json",
    "heldout_router_v1/heldout_router_generalization.json",
    "p2_synthetic_case/cub200/summary.json",
    "synthetic_gaussian_v4/summary.json",
    "backbone/clip_vit_b32/backbone_comparison.json",
    "p0_mechanism_v2/cub200/shot_full/seed_0/spectrum_fisher.npz",
    "p0_mechanism_v2/tiny_imagenet/shot_full/seed_0/spectrum_fisher.npz",
    "stepb_predictive_protocol_reproduced/joined_records.json",
    "stepb_predictive_protocol_reproduced/predictive_results.json",
    "router_ood11_replay/router_ood11_replay.json",
    "recompute_reference/table34_dataset_cluster_ci.json",
    "recompute_reference/fixed_targets_cluster_ci.json",
    "recompute_reference/paired_statistics.json",
    "recompute_reference/ood8_per_dataset.json",
]

REQUIRED_ROOT = [
    "README.md",
    "DATA_AND_FEATURES.md",
    "artifact_reproducibility_checklist.md",
    "LICENSE",
    "NOTICE",
    "artifacts/README.md",
    "artifacts/RESULTS_MANIFEST.md",
    "configs/simplecil_fulldata_evaluators_protocol.json",
    "splits/fscil_cub200/session_1.txt",
    "splits/fscil_cub200/session_11.txt",
]

# Keep this list encoded so the release package itself does not contain the
# private strings it is meant to detect.
FORBIDDEN_CODEPOINTS = [
    [47, 100, 97, 116, 97, 47, 111, 115, 115, 95, 98, 117, 99, 107, 101, 116, 95, 48],
    [111, 115, 115, 95, 98, 117, 99, 107, 101, 116],
    [47, 109, 110, 116, 47, 119, 111, 114, 107, 115, 112, 97, 99, 101],
    [100, 106, 47, 114, 97, 101],
    [99, 104, 101, 110, 103, 98, 97, 105],
    [112, 97, 105, 108, 105, 116, 97, 111],
    [78, 101, 98, 117, 108, 97],
    [111, 112, 101, 110, 108, 109],
    [105, 100, 101, 97, 108, 97, 98],
    [116, 109, 117, 120, 45, 98, 114, 105, 100, 103, 101],
    [65, 108, 105, 98, 97, 98, 97],
    [97, 108, 105, 98, 97, 98, 97],
    [65, 117, 116, 104, 111, 114, 105, 122, 97, 116, 105, 111, 110],
    [66, 101, 97, 114, 101, 114],
    [111, 115, 115, 58, 47, 47],
    [76, 84, 65, 73],
    [108, 112, 97, 105],
    [108, 105, 120, 105, 97, 110, 103],
    [97, 108, 105, 121, 117, 110, 99, 115],
    [111, 100, 112, 115, 95, 112, 114, 111, 106, 101, 99, 116],
    [100, 105, 110, 103, 106, 105, 97, 110],
    [114, 101, 99, 111, 109, 109, 101, 110, 100, 95, 103, 117, 108],
    [99, 111, 109, 112, 97, 110, 121, 45, 109, 97, 99],
    [110, 101, 98, 117, 108, 97],
]

FORBIDDEN = ["".join(chr(c) for c in codepoints) for codepoints in FORBIDDEN_CODEPOINTS]


def assert_close(name: str, value: float, expected: float, tol: float = 0.02) -> None:
    if abs(value - expected) > tol:
        raise SystemExit(f"{name}: expected {expected:.2f}, got {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    results = root / "artifacts/results"

    missing_root = [p for p in REQUIRED_ROOT if not (root / p).exists()]
    if missing_root:
        raise SystemExit("Missing required release files:\n" + "\n".join(missing_root))

    missing = [p for p in REQUIRED if not (results / p).exists()]
    if missing:
        raise SystemExit("Missing required artifacts:\n" + "\n".join(missing))

    bad = []
    skipped_dirs = {".git", "outputs", "features", "data", "models", "__pycache__"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix in {".pdf", ".png", ".zip", ".aux", ".bbl", ".blg", ".log", ".pyc"}:
            continue
        if skipped_dirs.intersection(path.relative_to(root).parts):
            continue
        text = path.read_text(errors="ignore")
        for forbidden in FORBIDDEN:
            if forbidden in text:
                bad.append(f"{path.relative_to(root)}: {forbidden}")
    if bad:
        raise SystemExit("Found private/internal strings:\n" + "\n".join(bad))

    stepb = json.loads((results / "stepb_cached_eval_v1/aggregate_groups.json").read_text())
    by_key = {(row["dataset"], str(row["shot"])): row for row in stepb}
    for dataset, cos, lda, gap in [
        ("aircraft", 42.33, 79.96, 37.62),
        ("cub200", 48.90, 80.46, 31.57),
        ("cars", 78.07, 90.16, 12.09),
    ]:
        row = by_key[(dataset, "full")]
        assert_close(f"{dataset} cosine", 100.0 * row["cos"], cos)
        assert_close(f"{dataset} LDA-TV", 100.0 * row["lda_trainval"], lda)
        assert_close(f"{dataset} LDA-TV gap", 100.0 * (row["lda_trainval"] - row["cos"]), gap)

    predictive = json.loads((results / "stepb_predictive_protocol_v1/predictive_results.json").read_text())
    lodo = [
        row
        for row in predictive
        if row["target"] == "target_best_whitening_minus_cos_pp" and row["fold"] == "leave_one_dataset"
    ][0]
    assert_close("LODO Pearson", lodo["pearson"], 0.752, tol=0.001)
    route5 = [row for row in lodo["routing"] if row["threshold"] == 5.0][0]
    assert_close("LODO 5pp routing regret", route5["routing_regret_pp"], 0.56, tol=0.01)
    always_cos = [row for row in lodo["routing_baselines"] if row["policy"] == "always cosine"][0]
    assert_close("always-cosine routing regret", always_cos["regret_pp"], 6.71, tol=0.01)

    # Camera-ready additions: dataset-cluster interval of the LODO router regret
    # (Table 4) and the OOD8 block regrets from the bundled replay (Appendix C).
    cluster = json.loads((results / "recompute_reference/table34_dataset_cluster_ci.json").read_text())
    lodo_policy = [row for row in cluster["table4"] if row["policy"] == "LODO router, 5pp"][0]
    assert_close("LODO regret CI low", lodo_policy["regret_ci"][0], 0.19, tol=0.006)
    assert_close("LODO regret CI high", lodo_policy["regret_ci"][1], 1.16, tol=0.006)
    replay = json.loads((results / "router_ood11_replay/router_ood11_replay.json").read_text())
    regrets = replay["blocks"]["coarse_ood4"]["point"]["policy_regret_pp"]
    assert_close("coarse block router regret", regrets["router_oracle_family"], 1.856, tol=0.0006)
    assert_close("coarse block always-WCos regret", regrets["always_wcos"], 1.014, tol=0.0006)
    assert_close("OOD11 Pearson", replay["suites"]["ood11"]["point"]["prediction"]["pearson"], 0.523, tol=0.0006)

    print("release smoke check passed")


if __name__ == "__main__":
    main()
