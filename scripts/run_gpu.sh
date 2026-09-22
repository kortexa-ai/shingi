#!/usr/bin/env bash
# The operator owns GPU availability. This launcher never changes services.
set -euo pipefail
cd "$(dirname "$0")/.."
test -z "$(git status --porcelain)" || { echo 'Commit source changes before a run.' >&2; exit 1; }
# NativeReadout checks the selected UUID and free-memory profile itself.
exec uv run --locked python scripts/evaluate_native.py "$@"
