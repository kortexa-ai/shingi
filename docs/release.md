# CUDA v0.1 publication preparation

The publication unit is a native GGUF decision adapter plus its calibration,
model card, checksums, and attribution. Users obtain the unchanged Bonsai base
separately. Source code is Apache-2.0. The final bundle must state its weight
license explicitly; the software license does not override dataset terms.

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
revision is `002ad22de8db2df5e0eb898b3da8072dbd4af4de`. The eight training sources
were checked against their primary dataset cards on 2026-09-22:

| Source | Publisher's stated license | Primary source |
|---|---|---|
| Banking77 | CC BY 4.0 | [PolyAI](https://huggingface.co/datasets/PolyAI/banking77) |
| CLINC150 | CC BY 3.0 | [CLINC](https://huggingface.co/datasets/clinc/clinc_oos) |
| MMLU | MIT | [CAIS](https://huggingface.co/datasets/cais/mmlu) |
| ARC Challenge | CC BY-SA 4.0 | [Allen AI](https://huggingface.co/datasets/allenai/ai2_arc) |
| HelpSteer2 | CC BY 4.0 | [NVIDIA](https://huggingface.co/datasets/nvidia/HelpSteer2) |
| Measuring Hate Speech | CC BY 4.0 | [UC Berkeley D-Lab](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) |
| BoolQ | CC BY-SA 3.0 | [Google](https://huggingface.co/datasets/google/boolq) |
| Civil Comments | CC0 1.0 | [Google](https://huggingface.co/datasets/google/civil_comments) |

These cards establish stated source terms, not an assertion that a mirror owns
all underlying rights. The benchmark transformations and Shingi's adaptation
are attributed in `NOTICE`. No raw dataset, prompt corpus, or human labels are
redistributed in the model bundle.

ARC and BoolQ have share-alike terms. The first adapter's proposed packaging uses
CC BY-SA 4.0 for the separate adapter, while retaining Apache-2.0 for the software
and unchanged base. This is a conservative release choice, not a claim that
training automatically subjects all resulting weights to share-alike. CC BY-SA
3.0 permits adaptations under a later version with the same license elements;
see [section 4(b)](https://creativecommons.org/licenses/by-sa/3.0/legalcode.en).
Creative Commons discusses the conditional application of license terms to AI
training in its [AI FAQ](https://creativecommons.org/faq/#artificial-intelligence-and-cc-licenses).

Seven additional sources are evaluation-only: LEDGAR, GoEmotions, MNLI, SST-5,
FEVER Evidence, SMS Spam, and ChaosNLI. Their terms vary. In particular, the
JevBench manifest records research-use or unspecified terms for some sources;
FEVER's primary card also lists GPL-3.0 alongside CC BY-SA 3.0, and the SMS mirror
labels its license unknown. The bundle includes aggregate measurements and data
identifiers, not those datasets or source text. These sources are excluded from the current candidate's adapter training,
development selection, and calibration. Reusers who
redownload datasets must follow their respective source terms.

## Measurement contract

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

The initial history audit at source `8fa8548` scanned 22 commits and 119 unique
blobs. Gitleaks found one generic-key match: the recorded SHA-256 of
`artifacts/decision-v2/tokenized-manifest.json` in the training report. It is an
artifact digest, not a credential. The reviewed finding is retained in the
ignored audit evidence; no broad secret-scanner allowlist is used.

Historical reports contain original machine paths and GPU identities as
provenance. Those records are not dependencies of the public runtime. Product
source, scripts, and tests must remain independent of local service managers,
private hosts, fixed GPU identities, and sibling repositories.
