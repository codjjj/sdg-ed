#!/usr/bin/env bash
# Recompute the camera-ready additions that need only the bundled result files
# (CPU, a few minutes): paired statistics (Appendix I), fixed targets and
# dataset-cluster intervals (Appendix C, Tables 3-4), spectral-diagnostic
# correlations without CUB-200 (Figure 3, Appendix F), and the expanded router
# tables on 8 and 11 unseen datasets (Appendix C).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS="${SDG_RESULTS_ROOT:-$ROOT/artifacts/results}"
OUT="${SDG_OUTPUT_ROOT:-$ROOT/outputs}/new_analyses"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "== Appendix I: paired support-regime statistics"
"$PYTHON_BIN" "$ROOT/scripts/recompute/paired_statistics.py" \
  --stepb-root "$RESULTS/stepb_cached_eval_v1" --output "$OUT/paired_statistics.json"

echo "== Appendix I/C: full statistics audit (sign-flip tests, fixed targets)"
if [ -e "$OUT/paired_statistics_audit/paired_statistics_audit.json" ]; then
  echo "exists, skipping: $OUT/paired_statistics_audit"
else
  "$PYTHON_BIN" "$ROOT/src/sdg_paired_statistics.py" \
    --stepb-root "$RESULTS/stepb_cached_eval_v1" \
    --main-joined "$RESULTS/stepb_predictive_protocol_reproduced/joined_records.json" \
    --output-root "$OUT/paired_statistics_audit"
fi

echo "== Appendix C: gap targets and fixed policies with dataset-cluster CIs"
"$PYTHON_BIN" "$ROOT/scripts/recompute/fixed_targets_cluster_ci.py" \
  --joined "$RESULTS/stepb_predictive_protocol_reproduced/joined_records.json" \
  --output "$OUT/fixed_targets_cluster_ci.json"

echo "== Tables 3-4: dataset-cluster bootstrap intervals"
"$PYTHON_BIN" "$ROOT/scripts/recompute/table34_dataset_cluster_ci.py" \
  --protocol-dir "$RESULTS/stepb_predictive_protocol_reproduced" \
  --output "$OUT/table34_dataset_cluster_ci.json"

echo "== Tables 3-4: LODO Spearman and router mean-accuracy intervals (router uncertainty audit)"
if [ -e "$OUT/router_uncertainty_audit.json" ]; then
  echo "exists, skipping: $OUT/router_uncertainty_audit.json"
else
  "$PYTHON_BIN" "$ROOT/src/sdg_router_uncertainty_audit.py" --output "$OUT/router_uncertainty_audit.json"
fi

echo "== Figure 3 / Appendix F: diagnostics with and without CUB-200"
"$PYTHON_BIN" "$ROOT/scripts/recompute/spectral_diagnostics.py" \
  --results-root "$RESULTS" --output "$OUT/spectral_diagnostics.json"
"$PYTHON_BIN" "$ROOT/scripts/recompute/highshot_diagnostics.py" \
  --input "$RESULTS/supplement_stats_v1/aggregate_groups.json" \
  --output "$OUT/highshot_diagnostics.json" > /dev/null

echo "== Appendix C: expanded router check (per-dataset rows)"
for block in original_ood4 coarse_ood4 finegrained_ood3; do
  "$PYTHON_BIN" "$ROOT/scripts/recompute/ood8_per_dataset.py" \
    --replay "$RESULTS/router_ood11_replay/router_ood11_replay.json" --block "$block"
done

echo "== Appendix C: verify the OOD11 replay from its records"
"$PYTHON_BIN" "$ROOT/src/verify_sdg_router_ood11_replay.py" --records-only \
  --aggregate-root "$RESULTS/router_ood11_replay" \
  --main-joined "$RESULTS/stepb_predictive_protocol_reproduced/joined_records.json" \
  > "$OUT/ood11_records_only_validation.log"
tail -n 1 "$OUT/ood11_records_only_validation.log"

echo "Outputs written under $OUT"
