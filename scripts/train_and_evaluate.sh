#!/usr/bin/env bash
# Invoke through LegoLM's process-owned Smarty GPU service block.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${CUDA_VISIBLE_DEVICES:-}" != GPU-a71210ca-e14a-755a-88bb-77f53a2102f6 ]; then
    echo 'Run inside the authorized 6000 block.' >&2
    exit 2
fi
SHINGI_BASE=/home/francip/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots/6ed5e12bf84b7a63069882c91dd9e9218647d17b/Ternary-Bonsai-2-27B-PQ2_0.gguf
SHINGI_PRISM=/home/francip/src/models.server/.engines/llama-prism
.venv/bin/python scripts/train_decision_adapter.py \
    --model "$SHINGI_BASE" --prism "$SHINGI_PRISM" \
    --canary artifacts/training-v1/canary-03/parity.json \
    --output artifacts/training-v1/run-02 --max-seconds 21600
.venv/bin/python scripts/evaluate_training.py \
    --model "$SHINGI_BASE" --training artifacts/training-v1/run-02 \
    --output artifacts/training-v1/evaluation-02
