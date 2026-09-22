#!/usr/bin/env bash
# Execute all baseline phases inside one already-owned GPU borrowing block.
set -euo pipefail
cd "$(dirname "$0")/.."
SHINGI_RUN_ROOT="${1:?Usage: investigate.sh OUTPUT_ROOT MODEL.gguf}"
SHINGI_MODEL_FILE="${2:?Supply the pinned native GGUF}"
test ! -e "$SHINGI_RUN_ROOT" || { echo 'Output directory already exists.' >&2; exit 1; }
mkdir -p "$SHINGI_RUN_ROOT"
uv run --locked python scripts/evaluate_native.py --model "$SHINGI_MODEL_FILE" --phase canary --output "$SHINGI_RUN_ROOT/canary"
uv run --locked python scripts/evaluate_native.py --model "$SHINGI_MODEL_FILE" --phase calibration --output "$SHINGI_RUN_ROOT/calibration"
uv run --locked python scripts/fit_calibration.py --run "$SHINGI_RUN_ROOT/calibration" --output "$SHINGI_RUN_ROOT/calibration.json"
uv run --locked python scripts/evaluate_native.py --model "$SHINGI_MODEL_FILE" --phase test --calibration "$SHINGI_RUN_ROOT/calibration.json" --output "$SHINGI_RUN_ROOT/test"
uv run --locked python scripts/evaluate_native.py --model "$SHINGI_MODEL_FILE" --phase shuffle --calibration "$SHINGI_RUN_ROOT/calibration.json" --output "$SHINGI_RUN_ROOT/shuffle"
uv run --locked python scripts/evaluate_native.py --model "$SHINGI_MODEL_FILE" --phase context --calibration "$SHINGI_RUN_ROOT/calibration.json" --output "$SHINGI_RUN_ROOT/context"
