# Shingi contributor instructions

Follow applicable parent workspace instructions when present. Keep changes
focused, reproducible, and covered by checks appropriate to their risk.

## Research discipline

Shingi investigates general-purpose decision models based on Bonsai 2 27B.
Choices and domains are request-defined. Record each experiment's objective,
acceptance thresholds, and resource budget in its owning issue before running.
Keep training, development, calibration, and locked evaluation data distinct.
Record dataset provenance, overlap checks, model revisions, prompt templates,
candidate token IDs, and inference settings with results.

Read every candidate logit explicitly; truncated top-k output is insufficient.
Preserve the upstream ternary scales and transforms when establishing numerical
agreement. Report measured accuracy, probability quality, option-order effects,
single-request latency, and total runtime memory. Never infer reliability or a
speedup from a working wrapper or a smaller file.

## GPU operation and artifacts

An operator must authorize shared GPU use and arrange availability. Pin every
job to one full GPU UUID. Start with a batch-one canary and retain deliberate
memory headroom. CUDA v1 caps context at 16,384 tokens. Its inference gates are
14 GiB free before load and 4 GiB during inference on GPUs up to 32 GiB, or
30/10 GiB on larger GPUs. Expanded-base training requires separate memory checks.

Product commands must not manage host services, assume a particular hostname,
embed a machine's GPU UUID, or import another project's private tools. Any
operator-specific downtime/restoration wrapper belongs outside the distributed
project. Record the initial service set and restore exactly that set after an
authorized block, including failures.

Use an in-project `.venv` and `uv`. Keep credentials, environments, downloaded
data, checkpoints, and bulk logs out of Git. Preserve checksums and the artifacts
needed to reproduce reported results. Do not commit model weights or use Git LFS
for them. Synchronize repository source through Git.

## Release discipline

Validate quality before preparing a release. Preserve upstream licenses and
notices; check model, dataset, runtime, and comparator permissions separately.
Model cards must identify the actual changes, evaluation splits, measured
results, and limitations. A separate adapter is not a merged ternary checkpoint.

Published Jev comparisons use attributed third-party public information. Do not
benchmark hosted Jev. Follow `docs/evaluation-protocol.md` and distinguish third-
party reported results from our reanalysis of public predictions.

The local deployment targets are RTX 4090-class GPUs and, in future work,
M3/M4 Macs with 24 GB unified memory. Mac feasibility must include load and
inference peaks, context cache, temporary buffers, and operating-system headroom.
Do not infer Mac performance from CUDA runs, weight-file size, or a larger Mac.
