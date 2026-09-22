#!/usr/bin/env bash
# This launcher never borrows memory by stopping services. Obtain a new GPU
# claim and check user availability before invoking it. GPU policy is inherited.
set -euo pipefail
cd "$(dirname "$0")/.."
test "$(hostname -s)" = smarty || { echo 'Run this on Smarty.' >&2; exit 1; }
test -z "$(git status --porcelain)" || { echo 'Commit source changes before a run.' >&2; exit 1; }
# --until-free 0 leaves services running. NativeReadout separately requires
# 30 GiB free before load and 10 GiB free before every prefill.
exec ../legolm/scripts/smarty_gpu_block.sh --until-free 0 uv run --locked python scripts/evaluate_native.py "$@"
