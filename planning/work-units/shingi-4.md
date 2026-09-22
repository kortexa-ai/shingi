# First Shingi training experiment

Owning issue: https://github.com/kortexa-ai/shingi/issues/4

## Question and gates

Can a compact adapter improve general decision quality and option-order stability while preserving the pinned ternary Bonsai 2 27B base? Labels remain request-defined. The first experiment must separate training, development, calibration, and a fresh locked test set. Exclude all records and state hashes used in the baseline experiment from training. Hold out whole sources where possible.

Establish a faithful trainable representation before a long run. Preserve the original scales and Hadamard transforms; never replace Bonsai with its Qwen donor. Check zero-adapter logits against native Prism inference, then measure a batch-one forward/backward canary. Prefer frozen base weights and a small adapter. Record any approximation and reject a conversion that fails parity. Select a checkpoint using development data only.

Compare the selected checkpoint with the matched unchanged base on the fresh test set, paired option permutations, probability metrics, runtime memory, and native deployment. Preserve failed experiments. A small readout-only experiment, if used, must be identified as such and cannot establish backbone fine-tuning quality.

## Resource boundary

Franci authorized the RTX PRO 6000 after the cinference handoff, including necessary production downtime. Use only UUID `GPU-a71210ca-e14a-755a-88bb-77f53a2102f6`. The handoff transferred no existing downtime obligation. Record and health-check the live service set before changes. Use a process-owned restore path and restore only the services this experiment stopped. Leave the 4090 and Miso alone.

Run one GPU job at a time, at most eight GPU hours for the first experiment. Keep at least 10 GiB free after the first backward and throughout the run; cap the allocator at 0.88. Bound context and start with batch one. Check current disk space before conversion, stream weights, and keep the original checkpoint immutable. Bulk weights and checkpoints stay in ignored Smarty artifacts. Source moves through Git.

## Delivery

Keep reproducible code, pinned revisions, data exclusion evidence, parity and memory measurements, adapter checksums, evaluation results, and a clear continue/stop verdict. Quality validation comes before MLX or other release formats. No Hugging Face upload, hosted Jev call, or paid inference is authorized by this experiment. Any published Jev comparison uses attributed public third-party information.

## Experiment record

The 2026-09-22 experiment used a frozen, faithfully expanded Bonsai base and 34.6 million MLP LoRA parameters. Development NLL selected update 128; training stopped after update 401. Native export parity passed before the fresh comparison. Calibrated test accuracy improved by 3.87 points, but option-order flips remained 17%. The [experiment report](../../results/training-v1/REPORT.md) records methods, paired uncertainty, per-source regressions, artifact checksums, and deployment limits. This supports another controlled quality experiment, not a model release.
