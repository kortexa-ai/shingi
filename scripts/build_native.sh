#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SHINGI_PRISM_ROOT="${SHINGI_PRISM_ROOT:-$PWD/artifacts/prism}"
EXPECTED_REVISION=d8f26eec76da6d09bb708bcba51ef64b8cd868a3
test "$(git -C "$SHINGI_PRISM_ROOT" rev-parse HEAD)" = "$EXPECTED_REVISION" || {
    echo 'Prism runtime revision mismatch; revalidate before using another build.' >&2
    exit 1
}
mkdir -p artifacts/bin
test -z "$(git -C "$SHINGI_PRISM_ROOT" status --porcelain --untracked-files=no)" || {
    echo 'Prism source has local changes; use a clean pinned checkout.' >&2
    exit 1
}
c++ -std=c++17 -O2 -Wall -Wextra src/native/readout.cpp \
    -I"$SHINGI_PRISM_ROOT/include" -I"$SHINGI_PRISM_ROOT/ggml/include" \
    -I"$SHINGI_PRISM_ROOT/vendor" -L"$SHINGI_PRISM_ROOT/build/bin" \
    -Wl,-rpath,"$SHINGI_PRISM_ROOT/build/bin" \
    -lllama -lggml -lggml-base -o artifacts/bin/readout
