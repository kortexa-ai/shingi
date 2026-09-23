# Non-share-alike decision adaptation

Fresh rank-8, alpha-16 LoRA on the unchanged Bonsai 2 27B base. The selected checkpoint is update **559**, chosen by minimum uncalibrated development NLL. The previous adapter, optimizer and calibration were not used for fitting.

## Data and selection

The prepared pool has 2,284 records from six sources: Banking77, CLINC150, MMLU, HelpSteer2 helpfulness, Measuring Hate Speech and Civil Comments. Their publisher-stated terms are CC BY, MIT or CC0. ARC and BoolQ are excluded from all fitting stages. The source-card revisions, transformation revision, licenses and exact file hashes are in provenance.json.

Two deterministic training permutations produced 4,466 usable prompts; 102 prompts over 1,024 tokens were excluded without truncation. The selected checkpoint saw 4,466 prompts from 2,233 distinct records and 1,073,860 input tokens. The prepared pool reuses 378 permitted earlier training records. All published validation/test states and prior project evaluation records were excluded from training. New development and calibration data exclude every earlier project split by ID and state hash.

Checkpoint selection uses 192 development records. Calibration uses a separate 180 records. The Choice readout sorts keys before inference and was fixed before the run. No fresh release-test record was used to fit, select or calibrate. Upstream pretraining overlap remains unknown.

| Update | Development accuracy | Uncalibrated NLL |
|---:|---:|---:|
| 0 | 65.62% | 1.123039 |
| 128 | 73.44% | 0.700159 |
| 256 | 73.44% | 0.679470 |
| 384 | 74.48% | 0.655932 |
| 512 | 76.56% | 0.641571 |
| 559 | 75.52% | 0.626760 |

The run completed 559 updates with zero skipped optimizer steps. Stop condition: finished. The fixed budget was one prepared epoch, at most 640 updates or 21,600 training seconds, with checks every 128 updates and early stopping after two checks without improvement.

Batch one, accumulation eight, AdamW with peak learning rate 5e-5, 32 warmup updates, cosine decay, gradient clipping at 1.0, and 1,024-token training context were used. Loss was candidate cross entropy against human vote distributions where available and the gold label otherwise. All 64 layers receive gate/up/down MLP adapters, with 34,603,008 trainable parameters. The native export is a separate FP32 GGUF adapter; base scales and transforms are unchanged.

## Native export and resources

The synthetic numerical/backward canary passed before training. Native Q8-KV development predictions were then checked against the selected differentiable checkpoint; these are measured agreement, not exact equivalence.

| Model | Native development accuracy | NLL | Torch/native argmax agreement | Mean / max TVD |
|---|---:|---:|---:|---:|
| base | 65.10% | 1.123221 | 99.48% | 0.004062 / 0.022601 |
| adapter | 75.52% | 0.627184 | 100.00% | 0.001770 / 0.015671 |

Training used the RTX PRO 6000. Minimum logged free memory was 33.19 GiB; peak PyTorch reservation was 61.06 GiB. An independent CPU build overlapped part of training, so elapsed training time is not a controlled speed benchmark. Training memory does not describe compact native inference.

Training source: `c3b2cc5cb4e4fe241e9d2d002f74699ac6e02174`. Selected native adapter SHA-256: `6f4c2b7434b5ef6e0bb83b6f9c196de00c850a79b9380175e06a1df2ae1b7dd2`. Safetensors and native checkpoints, tokenized inputs, complete candidate traces, canary evidence and raw logs are retained in the authoritative artifact store. Raw datasets and prompts are not redistributed.

The separate CUDA report contains the locked fresh test, matched old/new adapter comparison, runtime measurements and release gates.
