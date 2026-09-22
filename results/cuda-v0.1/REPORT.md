# Shingi CUDA v0.1 evaluation

Fresh test: **1108/1,500 (73.87%)** Shingi versus **1023/1,500 (68.20%)** unchanged Bonsai on the RTX PRO 6000.

The paired change is +5.67 percentage points, with a descriptive paired-bootstrap 95% interval [+3.87, +7.47]. The seven-source holdout change is +2.57 points [+0.43, +4.71].

![Accuracy by source](figures/accuracy.png)

| Metric, calibrated | Unchanged Bonsai | Shingi v0.1 |
|---|---:|---:|
| NLL (n=1500) | 0.94049 | 0.78908 |
| Multiclass Brier (n=1500) | 0.41873 | 0.34730 |
| Binary Brier (n=400) | 0.10440 | 0.06267 |
| Expected-score MAE (n=300) | 0.82214 | 0.66682 |
| Ordinal RPS (n=300) | 0.17538 | 0.12277 |
| Distance to human labels (n=400) | 0.41491 | 0.34046 |
| Top-label ECE (n=1,500) | 0.04636 | 0.04009 |

![Calibration and score quality](figures/calibration.png)

| GPU, same 1,500 inputs | Unchanged Bonsai correct | Shingi correct |
|---|---:|---:|
| 6000 | 1023/1,500 (68.20%) | 1108/1,500 (73.87%) |
| 4090 | 1020/1,500 (68.00%) | 1106/1,500 (73.73%) |

| Cross-GPU comparison | Argmax agreement | Mean / maximum probability TVD |
|---|---:|---:|
| base | 99.4667% | 0.0051497 / 0.4540853 |
| adapter | 99.7333% | 0.0037206 / 0.5718285 |

Both models pass the predeclared 99% agreement / 0.01 mean-TVD gates, but this is not exact numerical equivalence. Different final adaptive prompts occur in 6 base and 6 adapter requests after a chunk winner changes. Every static chunk prompt and candidate token ID matches across devices; every adaptive trace is verified independently by replay. Small hardware-dependent logit changes can cause a large individual probability shift in this approximate method; the maximum TVD is reported above and its record ID is in the JSON.

Choice keys are sorted before inference. Map-order stability is an interface property, not learned invariance. The additional input-order diagnostic varies the underlying prompt order.

| GPU | Map-order flips, base / adapter | Input-order diagnostic flips, base / adapter | Adapter context probes |
|---|---:|---:|---:|
| 6000 | 0/200 / 0/200 | 40/200 / 28/200 | 12/12 |
| 4090 | 0/200 / 0/200 | 39/200 / 28/200 | 12/12 |

| GPU | Mixed median / p95 | Short HTTP median / p95 | Model initialization |
|---|---:|---:|---:|
| 6000 | 80.1 / 604.6 ms | 51.3 / 53.4 ms | 1.24 s |
| 4090 | 107.0 / 757.2 ms | 70.7 / 75.6 ms | 9.09 s |

![Latency curves](figures/latency.png)

![Hardware measurements](figures/hardware.png)

## Method and limits

- 100 fresh records from each of 15 sources; 6,930 prior IDs and state hashes excluded. The aggregate weights sources equally and is not a user-workload population estimate.
- Existing update-128 adapter; frozen development-selected readout and separate 240-record calibration. No release-test tuning. Upstream pretraining overlap is unknown.
- Choice sorting was selected with a predeclared development gate. Adapter development accuracy changed from 212/256 to 209/256; NLL from 0.509250 to 0.520816. It avoids caller map-order effects at this measured development tradeoff.
- Paired intervals use 20,000 bootstrap draws, seed 20260922; descriptive, without multiplicity correction. Source slices have only 100 records.
- Mixed latency: 300 fixed records, 20/source, repeated three times after ten warmups. Synthetic curves: ten repetitions per case after warmup. Short HTTP: 30 localhost SDK requests after three warmups; one three-choice question.
- Latency percentiles use sorted values at index round((n−1) × p). With ten repeats, the sample p95 is the maximum. Curves are workload measurements, not population tail-latency guarantees.
- Wall times include all forward passes, tokenization and memory checks; loading is separate. Model initialization times exclude artifact checksum hashing. OS filesystem cache is uncontrolled. Device memory is sampled every 100 ms; a brief peak can be missed. Host RSS is sampled only after native readiness, so it does not establish a host load-time peak. Other resident processes are recorded in private operator evidence.
- Synthetic context probes test simple planted-fact retrieval through about 15K tokens. They do not establish real-document comprehension.
- Text/JSON only; sequential serving. More than 52 choices use approximate multi-pass chunk-and-anchor scoring. Labels and wording can still affect decisions.
- No Mac result, hosted Jev request, or hosted latency comparison is included.

Exact metrics, GPU/driver information, file hashes, source revisions, calibration and data identifiers are in `summary.json`. `accuracy.csv` contains both devices' per-source results. Raw predictions and traces are retained separately.
