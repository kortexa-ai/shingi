# Frozen Shingi v0.2 external evaluation

Issue: https://github.com/kortexa-ai/shingi/issues/7

This run measures the released v0.2 adapter, with its existing calibration and prompt. No weights, prompts or calibration parameters were selected on these datasets. All 8,122 planned questions returned valid predictions. The three benchmark suites remain evaluation-only.

## Accuracy

| Benchmark | Correct / planned | Question micro | State/group macro | Jev-Omni reported macro / micro |
|---|---:|---:|---:|---:|
| This/That | 3694/7305 | 50.57% | 48.31% | — |
| DecisionBench Medium | 233/293 | 79.52% | 79.21% | 87.57% / 86.01% |
| DecisionBench Hard | 193/293 | 65.87% | 61.24% | — |
| JevBench public subset | 198/231 | 85.71% | 83.85% | 86.15% / 87.45% |

Failures count as incorrect; none occurred. Choice and Score accuracy use the most likely class. Noul uses P(yes) ≥ 0.5. Probability scores use every valid answer. Wilson intervals are in the JSON and assume independent questions; shared states and paraphrases limit that interpretation.

The This/That headline is question-micro accuracy against its supplied labels. Its state-macro column is a supplementary equal-state average, not an upstream headline. DecisionBench uses 80 scenarios per subset. JevBench uses 195 groups: a non-null group joins paraphrases; each null-group task is independent.

[Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni/tree/c050d51354147985d13286cf4acf90f562f2c631) supplies the two comparison claims. They are third-party reported results, not our measurements of that model. Its full training corpus, frozen evaluation manifest, prediction files and exact scoring code are not published in that repository. We cannot independently confirm training overlap, option order or its Score correctness rule. The numerical gaps are descriptive, not paired significance tests. No hosted Jev was called.

The public JevBench tasks are 231 of the full 534-task benchmark. These are public-subset accuracy results, not its official full-suite composite score or rank. This is fstandhartinger/jevbench, distinct from the Praveenrajus/jev-bench source collection used in earlier Shingi work.

DecisionBench has 13 Choice questions per subset with more than 52 options, up to 120. These use Shingi's existing approximate chunk-and-anchor readout, with every candidate retained. This/That has at most eight options and public JevBench at most six. The run saved and verified 9,154 native traces for 9,022 main and diagnostic answers.

## Ordinal scoring sensitivity

The predeclared supplementary diagnostic rounds the expected Score index with floor(E[index] + 0.5), while leaving Choice and Noul unchanged. It does not replace the primary argmax metric.

| DecisionBench subset | Rounded-EV correct | Rounded-EV micro | Rounded-EV macro |
|---|---:|---:|---:|
| medium | 239/293 | 81.57% | 80.50% |
| hard | 191/293 | 65.19% | 59.55% |

## Probability quality and latency

ECE uses ten equal-width bins of top-class probability, not the API confidence field. NLL uses natural logarithms and a 1e-12 reporting floor. Multiclass Brier is the sum of squared errors. These are sequential batch-one latencies, including the native wrapper and its memory check. They do not support a speed comparison with the Jev-Omni card.

| Benchmark | NLL ↓ | Brier ↓ | ECE ↓ | Median / p95 ms |
|---|---:|---:|---:|---:|
| This/That | 1.0001 | 0.5824 | 0.0818 | 115.8 / 269.4 |
| DecisionBench Medium | 0.5455 | 0.2936 | 0.0389 | 744.3 / 2474.7 |
| DecisionBench Hard | 0.9450 | 0.4854 | 0.1272 | 611.7 / 1844.6 |
| JevBench public | 0.3447 | 0.1880 | 0.0347 | 75.9 / 921.3 |

## This/That spatial results

Binary This/That questions remain Choice questions, as supplied. Option keys are their zero-based original indices; the unchanged option text is the description. State and question are passed unchanged. Labels, analytic distributions, exposure flags and fingerprints never enter the prompt.

| Family | Correct / n | Sample-label accuracy |
|---|---:|---:|
| blocked_pair | 294/500 | 58.80% |
| cell_kind | 167/498 | 33.53% |
| distance_band | 161/500 | 32.20% |
| first_move | 171/500 | 34.20% |
| goal_bearing | 428/496 | 86.29% |
| move_legal | 380/500 | 76.00% |
| move_two_step | 331/500 | 66.20% |
| only_way | 97/415 | 23.37% |
| onward | 297/500 | 59.40% |
| open_count | 146/400 | 36.50% |
| plan_survives | 153/500 | 30.60% |
| reachable_within | 256/500 | 51.20% |
| snake_food | 279/496 | 56.25% |
| snake_safe | 271/500 | 54.20% |
| stochastic | 263/500 | 52.60% |

The 6,805 deterministic questions score 50.42%. The 500 stochastic questions have sampled labels and an analytic target distribution. Their expected accuracy under that distribution is 51.01%, versus a Bayes ceiling of 85.88%. Their mean TVD is 0.3462, KL divergence 0.3231, cross entropy 0.7025, and expected multiclass Brier 0.5094. Sample-label accuracy alone is insufficient for those questions.

For context, uniform random selection over each question's supplied options has 34.76% expected accuracy across the full This/That set. This is an analytic baseline, not another model run.

The source flag seen_in_training describes the dataset authors’ training-family exposure, not Shingi’s. The corresponding separate results are retained in summary.json.

## JevBench tiers and decision types

| Tier | Correct / n | Micro accuracy |
|---|---:|---:|
| easy | 48/48 | 100.00% |
| hard | 81/111 | 72.97% |
| original | 69/72 | 95.83% |

| DecisionBench subset | Type | Correct / n | Accuracy |
|---|---|---:|---:|
| hard | choice | 58/94 | 61.70% |
| hard | noul | 94/128 | 73.44% |
| hard | score | 41/71 | 57.75% |
| medium | choice | 84/94 | 89.36% |
| medium | noul | 103/128 | 80.47% |
| medium | score | 46/71 | 64.79% |

## Option order

Each suite has 100 preselected Choice pairs. The released engine sorts stable option keys. Reversing dictionary insertion order tests that contract. The separate raw-order diagnostic bypasses this sort, with the same weights, calibration and options. It is not the released configuration. This does not test renaming or renumbering option identities.

| Suite | Released flips / pairs | Raw-order diagnostic flips / pairs | Raw-order mean TVD |
|---|---:|---:|---:|
| decisionbench | 0/100 | 18/100 | 0.1539 |
| jevbench | 0/100 | 9/100 | 0.0721 |
| this-that | 0/100 | 28/100 | 0.1348 |

## Identity, resources and verification

- Model: `shingi-bonsai-2-27b-v0.2`; inference source `beecfe837f4c03ec368a6d1513a6fb8a61304ed3`. Full base, adapter, calibration, binary and data hashes are in summary.json and provenance.json.
- RTX PRO 6000 Blackwell, one full GPU UUID, batch one, 16,384-token context, Q8 KV. Production services remained running. No 4090 use or hosted-model requests.
- Wall time: 2122.2 seconds, including load, canary, all suites and order diagnostics. Start: 2026-09-23T06:32:22.056224+00:00; finish: 2026-09-23T07:07:44.278364+00:00.
- Device memory was sampled every 100 ms. Values include co-resident services; the baseline delta is not a continuous or process-isolated peak. Native RSS and an in-run kernel RSS high-water snapshot are also retained. CUDA results do not establish Mac feasibility.

| Phase | Peak device MiB | Peak delta MiB | Minimum free MiB |
|---|---:|---:|---:|
| load | 57858 | 7095 | 39423 |
| warmup | 59406 | 8643 | 37875 |
| mixed | 59434 | 8671 | 37847 |

The sampled peak device-allocation delta is 8.47 GiB. During inference, the native process's kernel RSS high-water mark was 7.06 GiB; the Python evaluator's was 74.96 MiB at that observation. These per-process lifetime peaks need not coincide. The host snapshot does not include unrelated services or the system filesystem cache.

All 9,022 main and diagnostic answers were reconstructed from their saved logits with exact answer equality. All prompt hashes and candidate token identities matched; raw reconstructed prompts are retained. The maximum input length was 8,031 tokens. The upstream JevBench scorer and an independent DecisionBench calculation agree with the reported counts and macro scores. All 19 downloaded source files match their frozen checksums.

76 CPU tests passed on Mac and Linux. The batch-one native canary passed. The operator record confirms that no service was stopped, all six observed 6000 service endpoints remained healthy, and the evaluation child exited successfully. A final check found no remaining readout process, all nine GPU-service endpoints returned HTTP 200, and GPU allocations returned to their recorded baselines. Validation details and artifact hashes are in validation.json.

## Data provenance and limits

- [This/That](https://huggingface.co/datasets/limberc/this-that-spatial-bench/tree/423c49f4ad2b8d200b38e845fdf046ba6b37d46b): MIT, 7,305 synthetic spatial questions.
- [DecisionBench](https://huggingface.co/datasets/akhilaaa3/decision-bench/tree/19334fec40b54b693a63e1ffd91636651d39e847): Apache-2.0, 80 scenarios / 293 questions in each subset. Synthetic writer-intended answer keys; Hard includes judgment calls.
- [JevBench](https://github.com/fstandhartinger/jevbench/tree/f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84): MIT public original/easy/hard tasks, 72/48/111.
- No exact or normalized state matches against 11 retained Shingi fitting/development/calibration/test files. The check normalizes Unicode, case and whitespace within string leaves. It does not detect semantic paraphrases, different structural wrappers or base-model pretraining exposure.
- This run evaluates only current Shingi. It does not measure a v0.1-to-v0.2 regression on these new suites. The earlier matched release comparison is separate.
- For the distinct openjev/openjev model, the inspected public release contains weights, inference code and reported benchmark results; we did not find its training corpus or exact 10,000-question evaluation set. Do not confuse it with the separate Zefan-Cai/Open-Jev project.

## Reproduction

Use scripts/prepare_external_benchmarks.py with the prior record files listed in provenance.json. The prepared manifest fixes every source revision, checksum, row mapping, overlap check and order-pair ID. Use scripts/evaluate_external_benchmarks.py with current v0.2 artifacts, the pinned native binary, and the exact manifest SHA256. Select an authorized GPU externally; product code never manages services.

Bulk inputs, predictions, complete logits, token IDs, raw reconstructed prompts, memory samples and operator records are in ignored artifacts/external-v1 on the evaluation machine, with a local artifact copy. Compact results are in [summary.json](summary.json), [provenance.json](provenance.json), and [validation.json](validation.json).
