# Compact local inference targets

Owning issue: https://github.com/kortexa-ai/shingi/issues/3

Franci clarified the product target after the first baseline: comfortable local decision inference on RTX 4090-class GPUs and M3/M4 Macs with 24 GB unified memory. Quality, decision stability, single-request latency, and memory use take priority over concurrent throughput. Snappy's 64 GB is development capacity, not the required deployment size.

This work records the priorities in the README and project agent guidance. Comfortable Mac use must include runtime allocations, loading peaks, context cache, temporary buffers, and headroom for macOS and applications. Metal/MLX optimization and actual 24 GB Mac validation follow quality work; this clarification does not establish those results.

Scope is documentation only. Review the wording against the user's constraints, check the diff, commit and push, and synchronize Smarty through Git. No training, conversion, GPU experiment, or service change belongs to this work unit. The measured baseline remains in [its original report](../../results/baseline-v1/REPORT.md).
