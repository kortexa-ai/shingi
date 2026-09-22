# First Shingi adapter experiment

Date: 2026-09-22. Tracking: [Shingi #4](https://github.com/kortexa-ai/shingi/issues/4).

## Verdict

**Continue research. The compact adapter works and improves overall test quality, but it is not ready for release.** On the same 1,500 fresh decisions, calibrated accuracy rises from **70.93% to 74.80%**. Probability and score metrics also improve. The adapter adds **132.03 MiB** to the immutable ternary base and runs in about **8.47 GiB** of sampled GPU allocation on the RTX PRO 6000.

Option-order answer flips remain **34/200, or 17%**. Emotion classification regresses, and the seven-source holdout gain is uncertain. The next experiment should focus on order consistency and these regressions before format conversion or publication. A new comparison needs fresh locked evaluation data; this test has now been inspected.

## Fresh native test

Both models use identical records, native runtime, 16K context, Q8 KV cache, and decision prompts. Each model fits its own calibration on the separate 240-record set. Neither model has a failed test prediction.

| Metric | Unchanged Bonsai | Shingi adapter |
|---|---:|---:|
| Raw correct / 1,500 | 1,065 (71.00%) | 1,120 (74.67%) |
| Calibrated correct / 1,500 | 1,064 (70.93%) | 1,122 (74.80%) |
| Calibrated accuracy, Wilson 95% interval | 68.58–73.18% | 72.54–76.93% |
| NLL, 1,500 records | 0.89583 | 0.76396 |
| Multiclass Brier, 1,500 records | 0.40928 | 0.35073 |
| Binary Brier, 400 Noul records | 0.10114 | 0.06643 |
| Top-label ECE | 0.03340 | 0.01306 |
| Expected score MAE, 300 records | 0.87758 | 0.71916 |
| Normalized score RPS, 300 records | 0.18570 | 0.13723 |
| TVD to human labels, 400 records | 0.40541 | 0.34594 |

All rows after accuracy use calibrated probabilities; lower is better. Raw and per-source probability metrics are retained in [experiment.json](experiment.json).

The paired accuracy change is **+3.87 percentage points**: 130 cases improve and 72 regress. A descriptive paired bootstrap gives a 95% interval of **+2.00 to +5.73 points** (20,000 resamples, fixed seed; no multiplicity correction).

On the seven sources held out from adapter training, development, and calibration, accuracy rises from **484/700 (69.14%) to 498/700 (71.14%)**. The paired change is +2.00 points, with a bootstrap interval of **−0.57 to +4.57 points**. This sample does not establish a reliable accuracy gain across unseen sources. Holdout NLL improves from 0.97536 to 0.91845; holdout ECE worsens from 0.02313 to 0.03121.

| Source, 100 test records each | Held out from adaptation | Base correct | Adapter correct |
|---|:---:|---:|---:|
| Banking77 | | 74 | 76 |
| CLINC150 | | 84 | 84 |
| LEDGAR | Yes | 71 | 72 |
| GoEmotions | Yes | 35 | 31 |
| MMLU | | 77 | 78 |
| ARC Challenge | | 96 | 95 |
| MNLI | Yes | 86 | 92 |
| SST-5 | Yes | 54 | 52 |
| HelpSteer2 helpfulness | | 33 | 41 |
| Measuring Hate Speech | | 46 | 69 |
| BoolQ | | 83 | 87 |
| FEVER Evidence | Yes | 87 | 93 |
| Civil Comments | | 87 | 94 |
| SMS Spam | Yes | 97 | 97 |
| ChaosNLI | Yes | 54 | 61 |

The largest gain is on Measuring Hate Speech. This is a source used in training. GoEmotions loses four correct answers and its NLL worsens from 2.424 to 2.722. SST-5 loses two exact-label answers, but its expected score MAE improves from 0.762 to 0.548. Per-source samples are small; these are diagnostics, not proof of consistent gains.

## Order sensitivity and runtime

| Metric | Unchanged Bonsai | Shingi adapter |
|---|---:|---:|
| Answer flips, all 200 pairs | 34 (17.0%) | 34 (17.0%) |
| Flips with at most 52 options, 133 pairs | 18 | 21 |
| Flips with more than 52 options, 67 pairs | 16 | 13 |
| Mean order-pair probability TVD | 0.23279 | 0.15956 |
| Median local test latency | 73.0 ms | 78.9 ms |
| p95 local test latency | 558.2 ms | 603.7 ms |
| Maximum sampled GPU allocation delta | 8,465 MiB | 8,675 MiB |
| Added native adapter file | — | 138,438,400 bytes |

Probability shifts from option order are smaller, but answer stability has not improved. The direct-readout subset gets worse on this small sample. The adapter adds about 8% latency in this one sequential run. These are single-request wall times, including readout safety checks; they are not generation tokens/second or a concurrent throughput benchmark. Test latency excludes model loading. Both models were tested while the six authorized 6000 production services were stopped; the 4090 remained in production.

## Model and method

The base is the pinned **Prism Ternary Bonsai 2 27B PQ2_0 GGUF**, not the original Qwen donor. Its SHA-256 is `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`. Native inference uses Prism revision `d8f26eec76da6d09bb708bcba51ef64b8cd868a3`.

The training loader expands the ternary matrices into frozen FP16 storage. It preserves the original block scales, signed Hadamard transforms, tensor mapping, and recurrent head ordering. FP32 activation and backward math uses temporary expanded matrices. Two earlier FP16 backward canaries produced non-finite gradients; neither made an optimizer update. The FP32 canary passed. [Canary evidence](canaries.json) records these attempts.

Rank-8 LoRA adapters cover the gate, up, and down MLP projections in all 64 layers. Alpha is 16. There are **34,603,008 trainable parameters**. Only these parameters receive updates. The exported model uses the original ternary base plus an FP32 adapter. It is not a merged, all-ternary checkpoint. No release quantization was performed.

Training uses batch size 1, gradient accumulation 8, AdamW, peak learning rate `5e-5`, 32 warmup updates, cosine decay, gradient clipping at 1, and a 1,024-token limit. The loss is candidate cross entropy against human vote shares when available, otherwise the gold label. Labels and choices remain request-defined.

## Data and selection

Data comes from [Praveenrajus / uspraveen's public JevBench](https://huggingface.co/datasets/Praveenrajus/jev-bench/tree/002ad22de8db2df5e0eb898b3da8072dbd4af4de), revision `002ad22de8db2df5e0eb898b3da8072dbd4af4de`. The experiment manifest SHA-256 is `6fadc5da66abf7d192fe5c30ac5c24860fe318bd77e18d5bdd21ea881e8130c0`.

Training uses eight sources: Banking77, CLINC150, MMLU, ARC Challenge, HelpSteer2 helpfulness, Measuring Hate Speech, BoolQ, and Civil Comments. The seven source holdouts are ChaosNLI, FEVER Evidence, GoEmotions, LEDGAR, MNLI, SMS Spam, and SST-5. These holdouts appear in neither adapter training nor development or calibration.

The initial training pool contains 3,084 unique records and 6,168 prompts with paired randomized option orders. Of these, 6,040 prompts fit the token bound; 128 were excluded without truncation. Training excludes all published validation/test IDs and state hashes. The prior baseline's records are also excluded from the new development, calibration, and test slices. Upstream model pretraining overlap is unknown.

Development has 256 records, calibration has 240, and the fresh test has 1,500, with 100 per source. The order test has 200 paired choice prompts. The fresh test differs from the earlier 4090 baseline, so its accuracy must not be compared directly with that report's 69.07%.

The selection rule was fixed before training: lowest uncalibrated development NLL, including the unchanged base. After updates 256 and 384 failed to improve on update 128, training received a graceful stop request. It finished update 401, saved and evaluated that checkpoint, then selected update 128. This was an adaptive development-based early stop within a maximum budget. Calibration and test results did not influence it. All checkpoints are retained.

| Update | Development correct / 256 | Development NLL |
|---:|---:|---:|
| 0 | 181 | 0.79099 |
| **128, selected** | **212** | **0.50944** |
| 256 | 216 | 0.51117 |
| 384 | 215 | 0.54340 |
| 401 | 215 | 0.54959 |

Training processed 3,208 prompts in 401 updates, with no skipped optimizer steps. The selected checkpoint saw only the first **1,024 prompts: 939 distinct records and 227,768 input tokens**, from all eight training sources. The full training process, including loading and development evaluations, took 7,344 seconds. Peak PyTorch reserved memory was 61.09 GiB; sampled free memory stayed at or above 33.16 GiB.

## Deployment parity

The synthetic forward/backward/export canary passed before training. The selected adapter then passed a separate check on all 256 development records with the deployment settings: 16K context and Q8 KV cache.

| Model | Native vs training argmax agreement | Mean total variation distance | Maximum total variation distance |
|---|---:|---:|---:|
| Unchanged base | 255/256 | 0.002962 | 0.017471 |
| Selected adapter | 256/256 | 0.002037 | 0.028450 |

These checks establish close measured agreement on these prompts, not exact equivalence for all inputs. Test inference uses the native exported adapter, not the expanded training model.

The selected adapter also passes **12/12 synthetic context probes** at approximately 512, 4,096, 8,192, and 15,000 tokens, with the planted fact at the beginning, middle, and end. Training prompts were at most 1,024 tokens. This check detects a simple retrieval regression; it does not establish real-document comprehension. [Context evidence](context.json) records exact token counts, predictions, hashes, and source revision `fe1af14`.

## Service restoration and validation

The process-owned wrapper restored all six services that the training block stopped: ComfyUI, the base image service, Bonsai, LFM2.5-VL, HY-MT2, and vision. All six health endpoints returned HTTP 200 in an independent check. The context check then ran beside production without stopping any service; all six endpoints passed again afterward. The [restoration record](restoration.json) includes timestamps and GPU/process snapshots. The 4090 service PIDs, including Miso, remained unchanged.

The 6000 returned to 50,763 MiB used and 46,518 MiB free. The 4090 remained at 21,925 MiB used and 659 MiB free. All 44 CPU tests passed on Snappy and Smarty. Native compilation, the numerical/backward canary, the complete fresh comparison, and the context check passed. No Shingi service was installed and production continues to use its original Bonsai model.

## Artifacts and reproduction

Training and fresh evaluation ran from source revision `d518847`. Bulk artifacts remain under `/home/francip/src/shingi/artifacts` on Smarty:

- `decision-v2/`: pinned data, exclusions, prepared prompts, tokenization, and manifests.
- `training-v1/run-02/`: training log, all development predictions, five checkpoints in safetensors and GGUF, and the immutable selection receipt.
- `training-v1/evaluation-02/`: raw base/adapter predictions, logits, calibration, order pairs, and metrics.
- `training-v1/run-02.log`: the process-owned GPU block, evaluation, and restoration log.

Selected native adapter: `training-v1/run-02/adapter-0128.gguf`, SHA-256 `d7ea6bf61f6ef5fe26bd82ca1ece4686c20d29a1937834a18239629bb6db31d4`.

Selected safetensors: `training-v1/run-02/adapter-0128.safetensors`, SHA-256 `7538d66c1b926d75d787a814cc49c568be2f34645ffa5bd514229a9229ccda02`.

The implementation is in `scripts/prepare_training_data.py`, `scripts/prepare_training_tokens.py`, `scripts/training_canary.py`, `scripts/train_decision_adapter.py`, and `scripts/evaluate_training.py`. Use fresh output directories for new runs. The frozen selection receipt precedes calibration/test inference. The reporting script checks artifact hashes and derives paired comparisons from saved predictions.

To reproduce the compact statistical record from the retained artifacts on Smarty:

```bash
.venv/bin/python scripts/summarize_training.py \
  --training artifacts/training-v1/run-02 \
  --evaluation artifacts/training-v1/evaluation-02 \
  --output artifacts/training-v1/recomputed-experiment.json
```

One startup attempt before run 02 failed because the CPU-only tokenizer path still required a visible GPU. The corrected path hides CUDA and prepares tokens before the training block. That attempt made no training updates, and its wrapper restored services. Its log remains in `training-v1/run-01.log`.

## Limits

- One seed and one small adapter experiment do not establish general decision reliability. Training uses eight benchmark sources, although choice labels remain request-defined.
- Source holdouts test transfer beyond adapter-training sources, not absence from base pretraining.
- The test aggregate weights each source equally. It is not a measured distribution of local user workloads.
- More than 52 choices still use the approximate chunk-and-anchor readout.
- Sampled GPU memory is not a continuous peak measurement. Latency here is local single-request latency on the RTX PRO 6000, not the 4090 or a Mac.
- The adapter has not been tested on a 24 GB M3/M4 Mac. Metal/MLX optimization and release packaging remain later work.
- Native readout with the adapter was tested. The existing live SDK/API checks apply to the unchanged base; adapter serving is not yet wired into that API.
- No hosted Jev requests were made. This report contains no new Jev performance measurement or direct Jev quality claim.
- No checkpoint was uploaded to Hugging Face. Model and dataset permissions still require a release review.
