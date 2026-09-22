#!/usr/bin/env bash
# Build the pinned public runtime. No model is loaded and no service is changed.
set -euo pipefail
cd "$(dirname "$0")/.."
SHINGI_PRISM_ROOT="${SHINGI_PRISM_ROOT:-$PWD/artifacts/prism}"
export SHINGI_PRISM_ROOT
SHINGI_PRISM_REVISION=d8f26eec76da6d09bb708bcba51ef64b8cd868a3
command -v cmake >/dev/null
command -v nvcc >/dev/null || { echo 'Install the CUDA toolkit and put nvcc on PATH.' >&2; exit 1; }
if [ ! -e "$SHINGI_PRISM_ROOT" ]; then
    git clone --no-checkout https://github.com/PrismML-Eng/llama.cpp.git "$SHINGI_PRISM_ROOT"
    git -C "$SHINGI_PRISM_ROOT" checkout --detach "$SHINGI_PRISM_REVISION"
fi
test "$(git -C "$SHINGI_PRISM_ROOT" rev-parse HEAD)" = "$SHINGI_PRISM_REVISION" || {
    echo 'Existing Prism checkout has a different revision. Choose a new SHINGI_PRISM_ROOT.' >&2; exit 1;
}
test -z "$(git -C "$SHINGI_PRISM_ROOT" status --porcelain --untracked-files=no)" || {
    echo 'Existing Prism checkout has local changes.' >&2; exit 1;
}
cmake -S "$SHINGI_PRISM_ROOT" -B "$SHINGI_PRISM_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES='89;120' -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build "$SHINGI_PRISM_ROOT/build" --target llama --parallel "${SHINGI_BUILD_JOBS:-4}"
bash scripts/build_native.sh
