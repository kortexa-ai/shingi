# CUDA v0.2 release preparation

The publication unit is a native GGUF decision adapter plus its calibration,
model card, checksums, and attribution. Users obtain the unchanged Bonsai base
separately. Source code and the freshly trained v0.2 adapter are Apache-2.0.
The historical v0.1 adapter retains CC BY-SA 4.0. The software license does
not override upstream model or dataset terms.

## License and provenance review

The base's pinned model card identifies Apache-2.0 and its Qwen3.8-27B ancestry.
Preserve both the [base LICENSE](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/6ed5e12bf84b7a63069882c91dd9e9218647d17b/LICENSE)
and [NOTICE.txt](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/6ed5e12bf84b7a63069882c91dd9e9218647d17b/NOTICE.txt)
when distributing the base. The release adapter keeps that base unchanged.
The public Prism runtime is MIT licensed. Shingi's normal installation compiles
it from source; any binary distribution must include its notices.

The [OpenJev card](https://huggingface.co/openjev/openjev/blob/5ec9e5fd2f80a6fff386779b1e5ac7e389971889/README.md)
licenses `helper/` and `serve/` under Apache-2.0, separately from its CC BY-NC
model weights. Shingi includes no OpenJev weights. The
[TypeSafe adapter](https://github.com/typesafe-ai/system-one-adapter-python/blob/adffc2eab300a4fa3c0e92252d4ffd6ceaa53700/LICENSE)
is MIT licensed. Its permission notice is retained in Shingi's `NOTICE`.

JevBench is a collection with mixed source licenses. The exact transformed data
revision is `002ad22de8db2df5e0eb898b3da8072dbd4af4de`. The six v0.2 fitting
sources were checked against their primary dataset cards on 2026-09-23:

| Source | Publisher's stated license | Primary source |
|---|---|---|
| Banking77 | CC BY 4.0 | [PolyAI](https://huggingface.co/datasets/PolyAI/banking77) |
| CLINC150 | CC BY 3.0 | [CLINC](https://huggingface.co/datasets/clinc/clinc_oos) |
| MMLU | MIT | [CAIS](https://huggingface.co/datasets/cais/mmlu) |
| HelpSteer2 | CC BY 4.0 | [NVIDIA](https://huggingface.co/datasets/nvidia/HelpSteer2) |
| Measuring Hate Speech | CC BY 4.0 | [UC Berkeley D-Lab](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) |
| Civil Comments | CC0 1.0 | [Google](https://huggingface.co/datasets/google/civil_comments) |

These cards establish stated source terms, not an assertion that a mirror owns
all underlying rights. The benchmark transformations and Shingi's adaptation
are attributed in `NOTICE`. No raw dataset, prompt corpus, or human labels are
redistributed in the model bundle.

The first adapter also trained on ARC Challenge (CC BY-SA 4.0) and BoolQ
(CC BY-SA 3.0). Its preserved CC BY-SA 4.0 package is a conservative release
choice, not a claim that all training automatically transfers dataset licenses
to weights. The v0.2 adapter starts from fresh zero-output LoRA weights on the
unchanged base, with no old adapter, optimizer state, or fitted calibration.
ARC and BoolQ are excluded from training, checkpoint selection and calibration.
Changing the recipe does not relicense the historical weights.

Nine sources are evaluation-only: ARC Challenge, BoolQ, LEDGAR, GoEmotions,
MNLI, SST-5, FEVER Evidence, SMS Spam, and ChaosNLI. Their terms vary. In particular, the
JevBench manifest records research-use or unspecified terms for some sources;
FEVER's primary card also lists GPL-3.0 alongside CC BY-SA 3.0, and the SMS mirror
labels its license unknown. The bundle includes aggregate measurements and data
identifiers, not those datasets or source text. These sources are excluded from
v0.2 training, development selection, and calibration. Reusers who
redownload datasets must follow their respective source terms.

## Measurement contract

### Non-share-alike retraining profile

The `apache-v2` preparation profile restricts training, development selection,
and calibration to Banking77, CLINC150, MMLU, HelpSteer2 helpfulness, Measuring
Hate Speech, and Civil Comments. The primary publisher card revisions and
hashes are in [`results/training-v2/provenance.json`](../results/training-v2/provenance.json).
ARC and BoolQ remain evaluation-only for this profile. The first adapter's
CC BY-SA packaging and measurements remain historical records; changing the
training recipe does not change that adapter's license.

Start a fresh LoRA on the unchanged base. Never initialize from the first
adapter or reuse its fitted calibration. Permitted prior training records may
recur, but every published validation/test state and prior project evaluation
record is excluded from training. New development and calibration splits also
exclude prior training records. The profile checks source membership and split
overlap before fitting. Tokenization excludes prompts over 1,024 tokens without
truncation. The default legacy preparation profile is retained for historical
reproduction and must not be used for an Apache-targeted run.

```bash
uv run --locked python scripts/prepare_training_data.py \
  --profile apache-v2 --baseline artifacts/benchmark-v1 \
  --previous-training artifacts/decision-v2 \
  --previous-release artifacts/release-v0.1/data \
  --output artifacts/apache-v2/data
uv run --locked python scripts/prepare_training_tokens.py \
  --model /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --data artifacts/apache-v2/data
```

Run the numerical/backward canary before the fresh trainer on an authorized GPU.
The trainer selects minimum uncalibrated development NLL, including the base,
and records a selection receipt. The `evaluate_release.py --phase calibration`
path requires that receipt, checks the audited data identity, and uses the
fixed canonical Choice readout. Its data contains no test split.

Prepare the new test only after weights and calibration are frozen. Supply
`prepare_release_data.py` with the new fitting directory, every prior dataset
directory through `--prior-data`, a new fixed `--seed`, the selected
`--adapter`, and the frozen `--protocol`. The Apache profile requires all prior
exclusion hashes and a matching weight/calibration identity. A frozen earlier adapter can be
evaluated on that same test with `--historical-comparator` and its own frozen
protocol. It must never influence fitting or selection. The report compares
both raw and calibrated probabilities and retains per-source and latency
regressions. Calibration differences are part of the packaged-model comparison,
not evidence of a weight-only change.

The v0.2 run selected update 559. Its immutable training and native-export
evidence is in [`results/training-v2/REPORT.md`](../results/training-v2/REPORT.md).
After native calibration has produced `development-01/protocol.json`, prepare
the frozen test with:

```bash
uv run --locked python scripts/prepare_release_data.py \
  --baseline artifacts/benchmark-v1 \
  --training-data artifacts/apache-v2/data \
  --prior-data artifacts/decision-v2 \
  --prior-data artifacts/release-v0.1/data \
  --seed shingi-apache-v02-release-20260923 \
  --adapter artifacts/apache-v2/run-01/adapter-0559.gguf \
  --protocol artifacts/apache-v2/development-01/protocol.json \
  --output artifacts/apache-v2/release-data
```

### Locked evaluation

Freeze adapter, decision prompt, choice-order method and calibration before
release test inference. Exclude every prior experiment split by both record ID
and canonical state hash. Report the fresh 1,500-record result separately from
the first training experiment. Test data is never used to select a checkpoint
or fit calibration.

Report both CUDA cards using the same candidate and inputs. Keep batch-one
request wall times, native prefill timing, process initialization, and sampled
memory separate. Repeated mixed-workload measurements use 20 records per source,
three repeats, deterministic ordering, and ten warmup requests. Synthetic
context and option-count curves use ten measured repetitions after warmup.
No result is a hosted Jev latency comparison or a generation-token benchmark.

A deterministic ordering of Choice keys, if selected by development checks,
makes request-map reorderings identical before model inference. Report this as
an interface property. Preserve a separate diagnostic with model input order
varied; zero API order flips must not be described as learned model invariance.

## Publication audit

Before publication, verify the complete bundle inventory and every checksum;
check a fresh installation and the real SDK on both cards; inspect charts;
verify source/weight identities and reported metric denominators; inspect Git
history for credentials, bulk data, model weights, and private runtime imports;
and verify restoration of every borrowed production service.

The source/history audit is recorded in
[`results/cuda-v0.2/validation.json`](../results/cuda-v0.2/validation.json).
Check any secret-scanner findings against the actual artifacts; reproducibility
hashes can trigger detectors. Keep redacted scan evidence outside the
distribution and do not add broad allowlists.

Historical reports contain original machine paths and GPU identities as
provenance. Those records are not dependencies of the public runtime. Product
source, scripts, and tests must remain independent of local service managers,
private hosts, fixed GPU identities, and sibling repositories.

## Reproduce the release evidence

Use the preserved `release-data`, `quality-6000-01`, `quality-4090-01`,
`speed-6000-01`, `speed-4090-01`, `api-6000-01`, and `api-4090-01`
directories beneath `artifacts/apache-v2`. `release-data` contains the frozen manifest and
inputs; raw prediction files include every candidate's logits and token IDs.
They are retained for audit and are not included in the public bundle.

```bash
uv sync --locked --extra reporting
uv run --locked --extra reporting python scripts/package_release.py report \
  --data artifacts/apache-v2/release-data \
  --protocol artifacts/apache-v2/development-01/protocol.json \
  --evaluation-6000 artifacts/apache-v2/quality-6000-01 \
  --evaluation-4090 artifacts/apache-v2/quality-4090-01 \
  --speed-6000 artifacts/apache-v2/speed-6000-01 \
  --speed-4090 artifacts/apache-v2/speed-4090-01 \
  --api-6000 artifacts/apache-v2/api-6000-01 \
  --api-4090 artifacts/apache-v2/api-4090-01 \
  --previous-evaluation-6000 artifacts/apache-v2/previous-quality-6000-01 \
  --previous-evaluation-4090 artifacts/apache-v2/previous-quality-4090-01 \
  --previous-speed-6000 artifacts/apache-v2/previous-speed-6000-01 \
  --previous-speed-4090 artifacts/apache-v2/previous-speed-4090-01 \
  --previous-protocol artifacts/apache-v2/previous-protocol.json \
  --output artifacts/apache-v2/report
uv run --locked --extra reporting python scripts/package_release.py bundle \
  --adapter artifacts/apache-v2/run-01/adapter-0559.gguf \
  --report artifacts/apache-v2/report \
  --output artifacts/apache-v2/hf-bundle
uv run --locked python scripts/package_release.py verify \
  artifacts/apache-v2/hf-bundle
```

Use new output directories for each attempt. Report generation verifies the
frozen protocol, dataset hashes, complete prediction inventories, measured
source revisions, context/order checks, memory floor, and cross-GPU agreement.
It recomputes quality metrics from the raw predictions. Cross-GPU comparison
checks identical static prompts and candidate IDs, then independently replays
each device's adaptive chunk-winner path. Saved answers and recomputed metrics
allow only 1e-12 numerical replay tolerance across Python/libm implementations;
reported GPU distributions are the original saved outputs. Different adaptive
paths and maximum probability shifts remain explicit in the report. Bundle assembly requires
a clean committed checkout, checks the selected adapter, and downloads exact
checksum-pinned license texts. The bundle contains one adapter, calibration,
protocol, model card, notices/licenses, aggregate evidence, and figures. Its
manifest binds each file by byte count and SHA-256. Verification rejects missing,
extra, corrupted, and externally linked files.

The initial public v0.2 bundle also includes the frozen external results under
`evaluation/external/`. Assembly checks their model, calibration, data and audit
hashes against the released artifacts. The model card includes the supplied
Shingi illustration. Source code remains on GitHub; it is not duplicated in the
Hub model bundle. External benchmarks remain evaluation-only and do not change
the frozen release protocol.

The v0.1 evidence and bundle remain separate. To reproduce that historical
package, use its source revision `262a004c849ffcd59bbe7dc3c0455c2de5023e8b`
and preserved artifacts; current release constants identify the new adapter.

## Publication procedure

The intended Hub repository is `kortexa-ai/shingi-bonsai-2-27b-v0.2`. Review the
verified bundle's README and manifest, then upload exactly that directory. Do
not upload the surrounding artifact root. The unchanged base is obtained from
Prism's pinned Hub revision separately. The source repository is
`kortexa-ai/shingi`; its owner can change visibility in GitHub repository
settings after reviewing the source/history audit and release evidence.

The model card is authored for the final published locations. Before those
repositories are public, its download example requires access and the Hub
repository must exist. After publication, verify the Hub file inventory and
checksums, capture its immutable revision, and use that revision in deployment
instructions. Test a download from an unauthenticated environment. Tag the
matching source commit and retain the exact bundle manifest alongside the
release record.
