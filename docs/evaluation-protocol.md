# Evaluation and publication protocol

## Jev comparisons

Franci requires any published comparison with hosted Jev to use attributed third-party public information. Do not run hosted Jev benchmarks or present a Jev result as our own hosted-model measurement. This is a project constraint, not a legal interpretation of TypeSafe's terms.

Keep these evidence classes distinct:

- **Our Shingi measurements:** results obtained from the local Bonsai readout, with source revision, model revision, dataset split, hardware, calibration, and run settings.
- **Third-party reported Jev results:** numbers quoted or summarized from a linked public report. Name the author, report, dataset, model version, and relevant limitations.
- **Our reanalysis of third-party Jev predictions:** derived statistics from a public prediction file. Identify both the original publisher and our selection or normalization steps. These are not our measurements of hosted Jev and must not be described that way.

The initial source is [JevBench by Praveenrajus / uspraveen](https://huggingface.co/datasets/Praveenrajus/jev-bench/tree/002ad22de8db2df5e0eb898b3da8072dbd4af4de), with [published Jev 1.13.0 predictions](https://huggingface.co/datasets/Praveenrajus/jev-bench/blob/002ad22de8db2df5e0eb898b3da8072dbd4af4de/results/jev-1.13.0/test_predictions.jsonl). The file carries record IDs, but no input hashes. A join to the same-revision dataset supports provenance; it does not independently prove that the hosted model received identical payloads. Its latency reflects the publisher's network and execution conditions and cannot support a local speed comparison.

The TypeSafe SDK tests call localhost with a deterministic backend. They test API compatibility, not Jev performance.

## Frozen baseline

The selection in `scripts/prepare_benchmark.py` is fixed before Shingi inference: 100 unique test records per source for 15 sources, 25 validation records per source except ChaosNLI, and 200 deterministic choice-order pairs. Test and calibration records have no shared IDs or canonical state hashes. Contamination in the upstream model's pretraining data is unknown.

The 15 sources cover 12 domains and all three primitives. They include subjective labels and human disagreement. Report per-source results and score distances; an aggregate accuracy alone is insufficient. The 1,500-record slice is not OpenJev's published 10,000-question benchmark, so percentages from those sets are not directly comparable.

Use calibration data only to fit calibration. Freeze prompts, readout settings, and fitted parameters before scoring the locked test set. Do not select configurations from test accuracy. If implementation errors require a rerun, retain the failed run, state what changed, and explain the evaluation exposure.

## Metrics

- Accuracy includes missing and failed predictions as incorrect, with a Wilson interval.
- Probability metrics report their valid denominator. NLL uses a `1e-12` floor for numerical reporting only.
- Report multiclass Brier, plus the conventional binary Brier for Noul.
- Noul accuracy thresholds at `P(yes) >= 0.5`, including exact ties.
- Top-label ECE uses ten equal-width bins of maximum class probability. API `confidence` is a separate statistic.
- Score uses zero-based expected-value MAE and normalized ranked probability score.
- Where human vote shares exist, report total variation distance to their normalized distribution.
- For rounded third-party predictions, explicitly count renormalized distributions. Never fill missing options or silently replace malformed results.
- Report option-order answer flips and probability changes with label identities aligned.

Keep raw prompts, labels, logits, per-example predictions, and traces in ignored artifacts. Commit compact reports and manifests sufficient to identify and reproduce the run. Review provenance wording before any public release.
