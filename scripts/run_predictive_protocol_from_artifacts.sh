#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export SDG_RESULTS_ROOT="${SDG_RESULTS_ROOT:-$ROOT/artifacts/results}"
export SDG_OUTPUT_ROOT="${SDG_OUTPUT_ROOT:-$ROOT/outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" "$ROOT/src/sdg_stepb_predictive_protocol.py" \
  --results-root "$SDG_RESULTS_ROOT" \
  --out-root "$SDG_OUTPUT_ROOT/predictive_protocol" \
  --stepb-root "$SDG_RESULTS_ROOT/stepb_cached_eval_v1" \
  --exp-name stepb_predictive_protocol_reproduced

"$PYTHON_BIN" "$ROOT/src/sdg_stepb_predictive_protocol.py" \
  --results-root "$SDG_RESULTS_ROOT" \
  --out-root "$SDG_OUTPUT_ROOT/predictive_protocol" \
  --stepb-root "$SDG_RESULTS_ROOT/stepb_cached_eval_v1" \
  --exp-name stepb_predictive_protocol_reproduced_ablation \
  --ablation

echo "Wrote reproduced predictive protocol outputs under $SDG_OUTPUT_ROOT/predictive_protocol"
