# Shingi — 審議

General-purpose decision models: choices, scores, and probabilities in one forward pass.

**Shingi** takes its name from the Japanese word for deliberation. This repository is a research workspace, initially exploring decision readouts from **Bonsai 2 27B**. It does not yet contain a trained Shingi checkpoint or validated quality results.

## Research direction

The intended interface accepts a question, context, and request-defined choices. Candidate token logits provide a decision distribution; yes/no probabilities and ordinal scores are related interfaces to investigate. The aim is broad usefulness across domains before any task-specific tuning.

The first question is whether a Bonsai-based model can make reliable decisions this way. Establish baseline quality, calibration, sensitivity to choice order, and performance on held-out domains before deciding whether training is warranted. Experiment goals and acceptance thresholds will be recorded separately before runs begin.

Quality comes before release formats. Preserve the native ternary representation and its scales and transforms as the reference. MLX packaging and alternative GPU execution formats are later work; storing ternary values in FP4 does not by itself establish numerical equivalence or a speedup.

## Workspace

- Source: this Git repository, synchronized between Snappy and Smarty through Git.
- Compute: Smarty's RTX PRO 6000, subject to the machine's GPU and service procedures.
- Model artifacts: `/home/francip/src/shingi/artifacts` on Smarty, excluded from Git.
- Research records: commit compact methods, configurations, and result summaries; keep weights, caches, and bulk run output out of Git.

## Prototype

The first prototype exposes `POST /v1/systemone` and `GET /v1/models`. It implements Choice, Noul, and Score over native Bonsai candidate logits. See [API compatibility and limits](docs/api-contract.md). Protocol tests use a deterministic backend; they do not establish Bonsai quality.

```bash
uv sync --locked
uv run pytest -q
uv run python scripts/prepare_benchmark.py
```

The benchmark preparation downloads a pinned public JevBench revision into ignored `artifacts/`. It fixes 1,500 test and 350 calibration records, checks ID and state-content overlap, and selects 200 choice-order pairs before inference. Upstream dataset licenses remain separate. Saved hosted Jev predictions permit a same-revision ID join, but do not include independent input hashes.

On Smarty, `bash scripts/build_native.sh` builds the small readout executable against the existing pinned Prism runtime. Building uses no GPU. Run a GPU canary only after checking availability and following the [investigation procedure](planning/work-units/shingi-2.md). A model-quality verdict requires the measured run; no Shingi checkpoint is released yet.

## References and licensing

- [Bonsai 2 27B GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) and [MLX](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit): starting model and native execution references.
- [OpenJev](https://huggingface.co/openjev/openjev): a reference for the decision-model interface and published evaluation methodology. Shingi is an independent project.

Code in this repository is licensed under [Apache-2.0](LICENSE). Upstream models, datasets, and other third-party artifacts retain their own licenses and notices; this repository's license does not relicense them.
