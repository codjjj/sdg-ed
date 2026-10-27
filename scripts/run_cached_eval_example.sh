#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export SDG_FEATURE_ROOT="${SDG_FEATURE_ROOT:-$ROOT/features}"
export SDG_OUTPUT_ROOT="${SDG_OUTPUT_ROOT:-$ROOT/outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET="${1:-cub200}"
SHOT="${2:-full}"
SEED="${3:-0}"

"$PYTHON_BIN" "$ROOT/src/sdg_stepb_cached_eval.py" \
  --mode run \
  --dataset "$DATASET" \
  --shot "$SHOT" \
  --seeds "$SEED" \
  --out-dir "$SDG_OUTPUT_ROOT"

echo "Finished cached-feature evaluator example:"
echo "  dataset=$DATASET shot=$SHOT seed=$SEED"
echo "  output=$SDG_OUTPUT_ROOT/stepb_cached_eval_v1/$DATASET/shot_$SHOT/result.json"
