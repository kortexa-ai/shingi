# CUDA installation

The release target is Linux x86-64 with an RTX 4090 or RTX PRO 6000 Blackwell.
Use Python 3.11 or later, `uv`, Git, CMake, a C++17 compiler, NVIDIA drivers,
the Hugging Face CLI (`hf`),
and a CUDA toolkit supporting both `sm_89` and `sm_120` (CUDA 12.8 or later).
The release validation records the exact tested toolkit and driver.

```bash
uv sync --locked
export PATH=/usr/local/cuda/bin:$PATH
bash scripts/setup_cuda.sh
```

Setup clones the public [Prism runtime](https://github.com/PrismML-Eng/llama.cpp)
at `d8f26eec76da6d09bb708bcba51ef64b8cd868a3` into `artifacts/prism` and
builds CUDA libraries and `artifacts/bin/readout`. It does not load a model or
stop any service. Four compiler jobs are the default; set `SHINGI_BUILD_JOBS`
if necessary. `SHINGI_PRISM_ROOT` can name another clean checkout of the exact
pinned revision. Stock llama.cpp is not a substitute for this PQ2_0 runtime.

Download the unchanged base using the public Hugging Face CLI:

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-PQ2_0.gguf LICENSE NOTICE.txt \
  --revision 6ed5e12bf84b7a63069882c91dd9e9218647d17b \
  --local-dir artifacts/base
```

The Shingi release bundle supplies `adapter.gguf`, `calibration.json`, and its
manifest. The base is downloaded separately. Put that bundle in
`artifacts/shingi-v0.1`, then select a GPU and start the local API:

```bash
nvidia-smi --query-gpu=name,uuid,memory.free --format=csv
export CUDA_VISIBLE_DEVICES=GPU-YOUR-FULL-UUID
uv run --locked shingi \
  --model artifacts/base/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --adapter artifacts/shingi-v0.1/adapter.gguf \
  --calibration artifacts/shingi-v0.1/calibration.json
```

Replace the UUID with one complete value printed by `nvidia-smi`. UUID selection
avoids ambiguous device indices. The API listens on `127.0.0.1:8765` only. Model
and adapter checksums are verified against calibration before loading. Omit
both adapter and calibration to explore the uncalibrated base; its API identity
will differ from Shingi v0.1. The repository's `release/calibration.json` is for
the selected adapter only.

```bash
curl -s http://127.0.0.1:8765/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"shingi","state":"Please replace my damaged parcel.","questions":{"route":{"type":"choice","instructions":"Select the handling team.","criteria":{"returns":"Damaged goods and replacements","billing":"Charges and refunds","shipping":"Delivery tracking"}}}}'
```

`/health`, `/v1/models`, and `/v1/version` provide health, model identity,
adapter hash, and calibration parameters. The [API contract](api-contract.md)
documents the TypeSafe-compatible primitives and limits.

## Memory and operating limits

The runtime accepts one GPU and 512–16,384 context tokens. It checks available
memory before loading and before each prefill. For GPUs with up to 32 GiB VRAM,
the gates are 14 GiB free before loading and 4 GiB remaining during inference.
Larger GPUs use 30 GiB and 10 GiB. These conservative gates are not a model-size
claim. CUDA v1 rejects devices with less than 20 GiB usable VRAM.

An operator must arrange GPU availability. No Shingi command stops production
services or needs a service manager, a particular hostname, or a sibling
repository. An oversized prompt returns an error instead of truncating. Native
inference is serialized; this release is designed for local single requests.

If setup fails, check the CUDA compiler version and pinned Prism revision.
Do not mix shared libraries from different llama.cpp builds. If startup reports
insufficient memory, stop only workloads you own or choose another GPU. Training
expands the frozen base and requires at least 80 GiB free; the compact inference
requirements do not apply to training.
