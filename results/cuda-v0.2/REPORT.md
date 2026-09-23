# Shingi v0.2 CUDA evaluation

Fresh test: **1124/1,500 (74.93%)** Shingi versus **1002/1,500 (66.80%)** unchanged Bonsai on the RTX PRO 6000.

The paired change is +8.13 percentage points, with a descriptive paired-bootstrap 95% interval [+6.00, +10.27]. The 9-source holdout change is +6.11 points [+3.67, +8.56].

![Accuracy by source](figures/accuracy.png)

| Metric, calibrated | Unchanged Bonsai | Shingi v0.2 |
|---|---:|---:|
| NLL (n=1500) | 0.95619 | 0.73016 |
| Multiclass Brier (n=1500) | 0.43559 | 0.32850 |
| Binary Brier (n=400) | 0.10848 | 0.04877 |
| Expected-score MAE (n=300) | 0.80599 | 0.56875 |
| Ordinal RPS (n=300) | 0.17671 | 0.10517 |
| Distance to human labels (n=400) | 0.40500 | 0.32314 |
| Top-label ECE (n=1,500) | 0.05147 | 0.04734 |

![Calibration and score quality](figures/calibration.png)

| GPU, same 1,500 inputs | Unchanged Bonsai correct | Shingi correct |
|---|---:|---:|
| 6000 | 1002/1,500 (66.80%) | 1124/1,500 (74.93%) |
| 4090 | 1004/1,500 (66.93%) | 1125/1,500 (75.00%) |

| Cross-GPU comparison | Argmax agreement | Mean / maximum probability TVD |
|---|---:|---:|
| base | 99.5333% | 0.0048461 / 0.0617394 |
| adapter | 99.7333% | 0.0028019 / 0.3467724 |

Both models pass the predeclared 99% agreement / 0.01 mean-TVD gates, but this is not exact numerical equivalence. Different final adaptive prompts occur in 3 base and 3 adapter requests after a chunk winner changes. Every static chunk prompt and candidate token ID matches across devices; every adaptive trace is verified independently by replay. Small hardware-dependent logit changes can cause a large individual probability shift in this approximate method; the maximum TVD is reported above and its record ID is in the JSON.

Choice keys are sorted before inference. Map-order stability is an interface property, not learned invariance. The additional input-order diagnostic varies the underlying prompt order.

| GPU | Map-order flips, base / adapter | Input-order diagnostic flips, base / adapter | Adapter context probes |
|---|---:|---:|---:|
| 6000 | 0/200 / 0/200 | 26/200 / 14/200 | 12/12 |
| 4090 | 0/200 / 0/200 | 26/200 / 14/200 | 12/12 |

| GPU | Mixed median / p95 | Short HTTP median / p95 | Model initialization |
|---|---:|---:|---:|
| 6000 | 81.1 / 612.3 ms | 51.0 / 58.0 ms | 1.17 s |
| 4090 | 107.5 / 755.3 ms | 70.5 / 73.8 ms | 9.11 s |

## Comparison with the frozen v0.1 adapter

Both adapters use the same fresh test. Each retains its own frozen calibration. Raw probability comparisons are also recorded in the JSON.

| GPU | v0.1 accuracy | New accuracy | Paired change, 95% interval |
|---|---:|---:|---:|
| 6000 | 74.47% | 74.93% | +0.47 pp [-0.93, +1.87] |
| 4090 | 74.40% | 75.00% | +0.60 pp [-0.80, +2.00] |

### Measured regressions

The following losses are retained even when the aggregate improves. Source slices contain 100 examples. Small changes can be sampling noise; these are descriptive measurements.

| GPU | Scope | Metric | Previous | New |
|---|---|---|---:|---:|
| 6000 | overall | ece_top_label | 0.02697 | 0.04734 |
| 6000 | banking77 | ece_top_label | 0.05990 | 0.08226 |
| 6000 | boolq | nll | 0.23052 | 0.32196 |
| 6000 | boolq | brier_multiclass | 0.14044 | 0.15556 |
| 6000 | boolq | brier_binary | 0.07022 | 0.07778 |
| 6000 | boolq | ece_top_label | 0.05725 | 0.06719 |
| 6000 | chaosnli | accuracy_failures_incorrect | 0.60000 | 0.57000 |
| 6000 | fever_evidence | nll | 0.15015 | 0.17628 |
| 6000 | go_emotions | human_tvd | 0.67764 | 0.67846 |
| 6000 | helpsteer2_helpfulness | accuracy_failures_incorrect | 0.48000 | 0.45000 |
| 6000 | helpsteer2_helpfulness | ece_top_label | 0.12104 | 0.12334 |
| 6000 | ledgar | nll | 1.12368 | 1.12537 |
| 6000 | sms_spam | accuracy_failures_incorrect | 0.99000 | 0.96000 |
| 6000 | sms_spam | brier_multiclass | 0.05283 | 0.05776 |
| 6000 | sms_spam | brier_binary | 0.02642 | 0.02888 |
| 4090 | overall | ece_top_label | 0.02717 | 0.04564 |
| 4090 | banking77 | ece_top_label | 0.05062 | 0.08572 |
| 4090 | boolq | nll | 0.23105 | 0.32666 |
| 4090 | boolq | brier_multiclass | 0.14113 | 0.15488 |
| 4090 | boolq | brier_binary | 0.07056 | 0.07744 |
| 4090 | boolq | ece_top_label | 0.05721 | 0.07422 |
| 4090 | chaosnli | accuracy_failures_incorrect | 0.60000 | 0.58000 |
| 4090 | fever_evidence | nll | 0.15006 | 0.17766 |
| 4090 | go_emotions | human_tvd | 0.67757 | 0.67825 |
| 4090 | helpsteer2_helpfulness | accuracy_failures_incorrect | 0.48000 | 0.45000 |
| 4090 | helpsteer2_helpfulness | ece_top_label | 0.09464 | 0.12303 |
| 4090 | ledgar | nll | 1.12220 | 1.12559 |
| 4090 | ledgar | ece_top_label | 0.08144 | 0.08272 |
| 4090 | mmlu | ece_top_label | 0.07102 | 0.08043 |
| 4090 | sms_spam | accuracy_failures_incorrect | 0.99000 | 0.96000 |
| 4090 | sms_spam | brier_multiclass | 0.05317 | 0.05874 |
| 4090 | sms_spam | brier_binary | 0.02658 | 0.02937 |

| GPU | Mixed median, previous / new | Mixed p95, previous / new |
|---|---:|---:|
| 6000 | 81.50 / 81.12 ms | 613.75 / 612.26 ms |
| 4090 | 107.17 / 107.48 ms | 755.57 / 755.29 ms |

![Latency curves](figures/latency.png)

![Hardware measurements](figures/hardware.png)

## Method and limits

- 100 fresh records from each of 15 sources; 10,708 prior IDs and 10,708 state hashes excluded. The aggregate weights sources equally and is not a user-workload population estimate.
- Frozen adapter and readout, with separate 180-record calibration. No release-test tuning. Upstream pretraining overlap is unknown.
- Choice keys are sorted before inference. Checkpoint selection and calibration preceded this locked test.
- Paired intervals use 20,000 bootstrap draws, seed 20260922; descriptive, without multiplicity correction. Source slices have only 100 records.
- Mixed latency: 300 fixed records, 20/source, repeated three times after ten warmups. Synthetic curves: ten repetitions per case after warmup. Short HTTP: 30 localhost SDK requests after three warmups; one three-choice question.
- Latency percentiles use sorted values at index round((n−1) × p). With ten repeats, the sample p95 is the maximum. Curves are workload measurements, not population tail-latency guarantees.
- Wall times include all forward passes, tokenization and memory checks; loading is separate. Model initialization times exclude artifact checksum hashing. OS filesystem cache is uncontrolled. Device memory is sampled every 100 ms; a brief peak can be missed. Host RSS is sampled only after native readiness, so it does not establish a host load-time peak. Other resident processes are recorded in private operator evidence.
- Synthetic context probes test simple planted-fact retrieval through about 15K tokens. They do not establish real-document comprehension.
- Text/JSON only; sequential serving. More than 52 choices use approximate multi-pass chunk-and-anchor scoring. Labels and wording can still affect decisions.
- No Mac result, hosted Jev request, or hosted latency comparison is included.

Exact metrics, GPU/driver information, file hashes, source revisions, calibration and data identifiers are in `summary.json`. `accuracy.csv` contains both devices' per-source results. Raw predictions and traces are retained separately.
