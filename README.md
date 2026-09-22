# Shingi — 審議

Local decision models: choices, probabilities, and scores from candidate logits.

Shingi (Japanese for deliberation) adapts **Bonsai 2 27B** for request-defined
decisions. Give it context, a question, and possible answers. It reads every
candidate's logit directly, without generating an explanation or parsing prose.

The CUDA v0.1 candidate combines the original compact ternary base with a
**132 MiB decision adapter**. In the [first training experiment](results/training-v1/REPORT.md),
calibrated accuracy improved from **70.93% to 74.80%** across 1,500 fresh decisions.
Reordering choices still changed **17%** of answers. This is an experimental
model with measured limitations, not a general reliability guarantee. The
release package and fresh CUDA measurements are being prepared; no public
checkpoint has been uploaded.

## Run locally

Target hardware: **RTX 4090** and **RTX PRO 6000 Blackwell**, Linux x86-64.
Follow [CUDA installation](docs/cuda.md) to build the pinned public Prism runtime,
download the unchanged base, and load the Shingi adapter and calibration.

```bash
uv sync --locked
bash scripts/setup_cuda.sh
nvidia-smi -L
export CUDA_VISIBLE_DEVICES=GPU-YOUR-FULL-UUID
uv run --locked shingi \
  --model artifacts/base/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --adapter artifacts/shingi-v0.1/adapter.gguf \
  --calibration artifacts/shingi-v0.1/calibration.json
```

Use a complete UUID printed by `nvidia-smi`. The API listens on
`http://127.0.0.1:8765`. Shingi does not manage services or require any Kortexa
infrastructure. The setup builds from public source; model files stay outside Git.

## Interface

`POST /v1/systemone` accepts text or JSON state and named questions:

| Primitive | Result |
|---|---|
| Choice | Selected key and probabilities for every supplied choice |
| Noul | Probability that the supplied statement is true |
| Score | Expected ordinal score and probabilities for every level |

The [API contract](docs/api-contract.md) documents TypeSafe SDK compatibility,
model identity, calibration, error responses, and confidence semantics. Up to
52 choices use one forward pass; 53–255 use approximate chunk-and-anchor scoring
and need multiple passes. Questions run sequentially. Context is capped at
16,384 tokens, with an explicit error for oversized inputs.

## Evidence and scope

- [Unchanged-base investigation](results/baseline-v1/REPORT.md): RTX 4090 feasibility
  and live SDK checks, 1,500 decisions, option-order and synthetic context tests.
- [First adapter experiment](results/training-v1/REPORT.md): matched base/adapter
  quality, calibration, source holdouts, native export agreement, RTX PRO 6000
  inference and training resource measurements.
- [Evaluation protocol](docs/evaluation-protocol.md): data separation, exact
  candidate logits, failure accounting, and attributed third-party comparisons.

The adapter improves aggregate quality in the first experiment. Emotion
classification regresses, and accuracy gains on sources excluded from training
are uncertain. The original ternary scales and transforms are preserved. The
adapter is separate FP32 LoRA weights; this is not a merged all-ternary model.

No hosted Jev benchmark is run by this project. Historical Jev comparisons are
explicitly labeled reanalyses of attributed public predictions. Percentages on
different datasets are not a controlled comparison with OpenJev or other models.

Text and JSON are supported; screenshot inputs are not. A 24 GB M3/M4 Mac is a
future deployment target. CUDA results do not establish Mac memory or speed.
Metal/MLX work and alternative weight formats follow quality validation.

## Development

```bash
uv sync --locked
uv run --locked pytest -q
```

The training path uses `uv sync --locked --extra training`, a separate numerical
and backward canary, frozen ternary base weights expanded for training, and
native GGUF adapter export. It needs 96 GB class GPU capacity; inference memory
measurements are not training requirements. Research scripts accept explicit
paths and never change system services. Keep training, development, calibration,
and locked evaluation data separate. Retain checkpoints and reproducible records.

## License and attribution

Repository code is [Apache-2.0](LICENSE). Upstream model, runtime, and dataset
licenses remain separate. See the [Bonsai base](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf),
[Prism runtime](https://github.com/PrismML-Eng/llama.cpp), and
[OpenJev](https://huggingface.co/openjev/openjev), whose public decision interface
and methods informed this independent project. Release notices and a source-by-source
data review accompany the publication package.
