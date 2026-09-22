# Shingi: first native Bonsai investigation

Run date: 2026-09-22 UTC. [Owning issue](https://github.com/kortexa-ai/shingi/issues/2).

The approach works as a local decision API. Native ternary Bonsai 2 27B fits on the RTX 4090 and returns complete Choice, Noul, and Score distributions. The baseline is useful enough to justify further research. It is not ready for a claim of reliable general-purpose decision quality or a new trained model release.

The main quality concern is option order: 31 of 200 answers changed when the same choices were reordered. No weights were trained or converted. The only fitted values are three global calibration parameters.

## Measured quality

We froze 1,500 test records, with 100 records from each of 15 sources across 12 domains. We used a separate 350-record validation set for calibration. There are no shared record IDs or canonical state hashes between these sets. These are held out from our fitting, but upstream pretraining contamination is unknown. This study does not establish transfer to wholly unseen domains.

All 1,500 test requests produced valid results. The calibrated model got 1,036 correct: **69.07%**, with a descriptive Wilson 95% interval of **66.68–71.35%**. This interval does not account for dataset selection, label ambiguity, or contamination. The aggregate gives equal weight to each source and is not an estimate of every user's task mix.

| Metric | Raw logits, no calibration | Validation-fitted calibration |
| --- | ---: | ---: |
| Accuracy, 1,500 records | 68.73% | 69.07% |
| Negative log likelihood, lower is better | 1.0598 | 0.9352 |
| Multiclass Brier score, lower is better | 0.4404 | 0.4106 |
| Top-label ECE, ten bins | 0.1073 | 0.0376 |
| Binary Brier score, 400 Noul records | 0.0846 | 0.0747 |
| Expected score MAE, 300 Score records | 0.8085 | 0.8482 |
| Normalized ranked probability score, 300 records | 0.1886 | 0.1801 |
| Distance to human vote shares, 400 records | 0.4389 | 0.4081 |

Calibration improved most probability metrics but worsened expected score MAE. It is not a universal quality improvement. Raw results replay the same saved logits with no fitted adjustments; they are not a second GPU run or a test-set parameter search. Calibration used a fixed grid and validation NLL only: temperature 1.5, Noul temperature multiplier 0.4333333333, and Noul bias -0.5. The effective Noul temperature is 0.65. Choice and Score rankings do not change under this temperature scaling; the five extra correct answers come from Noul.

API `confidence` is a distribution statistic. ECE above uses the largest class probability, not API confidence. Neither is a guarantee for an individual decision.

## Results by source and public Jev evidence

The Jev column below is **our reanalysis of third-party public predictions**, published by **Praveenrajus / uspraveen in JevBench**, for `jev-1.13.0`. We made no hosted Jev requests. [Pinned public prediction file](https://huggingface.co/datasets/Praveenrajus/jev-bench/blob/002ad22de8db2df5e0eb898b3da8072dbd4af4de/results/jev-1.13.0/test_predictions.jsonl).

The join uses the same 1,500 record IDs at the same dataset revision. The prediction file has no independent input hashes, so we cannot prove that Jev received identical payloads. We renormalized 41 rounded distributions. The file's rounded zero probabilities also limit NLL comparisons. We make no hosted latency comparison or claim of a controlled head-to-head experiment.

Each entry below is correct answers out of 100. Score accuracy uses the most probable ordinal label; expected-score distance is reported separately above.

| Source | Shingi, measured locally | Jev, reanalysis of public predictions |
| --- | ---: | ---: |
| Banking77 | 64 | 77 |
| CLINC150 | 88 | 91 |
| LEDGAR | 56 | 71 |
| GoEmotions | 21 | 23 |
| MMLU | 76 | 89 |
| ARC Challenge | 95 | 98 |
| MNLI | 90 | 91 |
| SST5 | 49 | 50 |
| HelpSteer2 helpfulness | 37 | 36 |
| Measuring Hate Speech | 43 | 50 |
| BoolQ | 87 | 94 |
| FEVER evidence | 99 | 100 |
| Civil Comments | 80 | 70 |
| SMS Spam | 94 | 94 |
| ChaosNLI | 57 | 60 |
| Total, out of 1,500 | **1,036 (69.07%)** | **1,094 (72.93%)** |

The descriptive gap is 3.87 percentage points on this selected slice. This is not OpenJev's published 10,000-question benchmark. We did not measure OpenJev or Laya in this investigation, so these results do not rank Shingi against either one. Subjective labels and human disagreement also limit the meaning of hard-label accuracy.

## Option order and context

We selected the 200 order pairs before inference. Reordering changed **15.5%** of decisions; the Wilson 95% interval is 11.1–21.2%. Accuracy on these pairs changed from 132/200 to 138/200. A better aggregate score after shuffling does not remove the instability.

| Choice count | Pairs | Changed decisions | Mean label-aligned probability distance |
| --- | ---: | ---: | ---: |
| At most 52 | 127 | 18 (14.2%) | 0.1857 |
| More than 52 | 73 | 13 (17.8%) | 0.3047 |

The effect exists within a single readout. Chunking is not its only cause. Above 52 options, the baseline compares chunk winners to align probability scales. This adds approximation and extra forward passes. The API supports up to 255 choices, but support does not establish reliable quality at that limit.

All **12 synthetic context probes** passed: targets of 512, 4,096, 8,192, and 15,000 tokens, with a planted fact at the beginning, middle, and end. These tests use repeated filler. They establish fact retrieval in those prompts, not real-document comprehension or resistance to conflicting instructions. Context is capped at 16,384 tokens; oversized inputs fail explicitly.

## Hardware and API

The run used Smarty's RTX 4090 with native PQ2_0 weights, batch one, flash attention, and Q8 KV cache. All model layers were offloaded. At recorded checkpoints, the largest allocation increase was **8,253 MiB (8.06 GiB)**. At least **7,572 MiB (7.39 GiB)** remained free while ASR and Miso stayed loaded. These are sampled observations, not continuous peak measurements. The fallback LM and TTS were stopped for each bounded block and restored with successful health checks. The 6000 was not used.

For test decisions, median evaluation latency was **117 ms**, and p95 was **712 ms**. This includes Python and GPU memory safety queries. Summed native prefill time per decision had a median of **65 ms** and p95 of **537 ms**. These are batch-one measurements with varied prompt lengths and choice counts; they do not establish HTTP throughput or concurrent serving capacity. The 15K context probes took about 5.5 seconds each.

The real TypeSafe Python SDK passed against the calibrated native localhost API. Checks covered all three primitives in one request, exact-key choices with 52, 53, and 255 candidates, finite normalized probabilities, and rejection of 256 choices with HTTP 422. All 39 CPU tests passed on Snappy and Smarty. The native C++ build passed. [Saved live API evidence](live-api.json).

Compatibility remains bounded: text/JSON only, sequential questions, no hosted authentication or billing behavior, and no image projector. This is a research implementation of the decision API. See [the API contract](../../docs/api-contract.md).

## Recommendation

Continue toward a general-purpose model, with a separate training investigation before release packaging. First test label permutations and decision prompts on development data. Use those tests to distinguish readout bias from model limitations. Treat this baseline test set as exposed for future design decisions, and freeze fresh evaluation records before the next comparison.

If training proceeds, use a broad mixture of domains, request-defined choices, randomized label order, and ordinal and human-disagreement targets. Hold out whole domains as well as examples. The aim is a common decision behavior, not one task-specific classifier. A native ternary training and export path must first pass forward-parity and round-trip checks; this inference result does not prove that path.

Keep these weights as the immutable reference. Do not describe this wrapper and calibration file as a newly trained checkpoint. MLX packaging and FP8/NVFP4 experiments remain later work, after quality and conversion parity checks. No model was uploaded to Hugging Face.

## Reproduction and evidence

- Measured inference source: `9d4c76547ac540d0734b3fe9dfe8ab18f432d3ff`.
- API smoke and report source: `9a80aa11c36a50a20a87ec603f996f8269e1f78a`.
- Model: `prism-ml/Ternary-Bonsai-2-27B-gguf`, revision `6ed5e12bf84b7a63069882c91dd9e9218647d17b`, file `Ternary-Bonsai-2-27B-PQ2_0.gguf`.
- Model SHA-256: `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`.
- Prism runtime: `d8f26eec76da6d09bb708bcba51ef64b8cd868a3`; native executable SHA-256: `fb65bdaebd78549d66b35c8913d24f3f0945ace3e0ec625ed3fb1a3faa8f9ac6`.
- Dataset: `Praveenrajus/jev-bench`, revision `002ad22de8db2df5e0eb898b3da8072dbd4af4de`. [Selection manifest](data-manifest.json).
- [Full compact metrics and phase provenance](native-investigation.json), [public Jev prediction reanalysis](jev-third-party-reanalysis.json), and [service restoration records](resource-blocks.json).

Raw per-example artifacts remain at `/home/francip/src/shingi/artifacts/baseline-v1-4090`. The API smoke is at `artifacts/live-api-v1`. Run `scripts/investigate.sh` inside an authorized GPU block for fresh inference. Run `scripts/summarize_investigation.py --run artifacts/baseline-v1-4090 --output NEW_FILE.json` to reproduce the summary from saved logits. It needs no GPU. Run outputs are never overwritten.
