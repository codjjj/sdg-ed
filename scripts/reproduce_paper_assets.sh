#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export SDG_RESULTS_ROOT="${SDG_RESULTS_ROOT:-$ROOT/artifacts/results}"
export SDG_ASSET_OUTPUT_ROOT="${SDG_ASSET_OUTPUT_ROOT:-$ROOT/outputs/paper_assets}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" "$ROOT/scripts/render_assets.py"

echo "Rendered paper figures and tables from: $SDG_RESULTS_ROOT"
echo "Outputs:"
echo "  $SDG_ASSET_OUTPUT_ROOT/figures/*.pdf"
echo "  $SDG_ASSET_OUTPUT_ROOT/tables/*.tex"
