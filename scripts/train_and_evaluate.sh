#!/usr/bin/env bash
# Run only after arranging GPU availability; never changes system services.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -ne 6 ]; then
    echo 'usage: train_and_evaluate.sh MODEL.gguf PRISM_ROOT CANARY.json DATA_DIR TRAIN_OUTPUT EVAL_OUTPUT' >&2
    exit 2
fi
uv run --locked --extra training python scripts/train_decision_adapter.py \
    --model "$1" --prism "$2" --canary "$3" --data "$4" --output "$5" --max-seconds 21600
uv run --locked --extra training python scripts/evaluate_training.py \
    --model "$1" --training "$5" --data "$4" --output "$6"
