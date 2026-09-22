# Initial Shingi investigation

Owning issue: https://github.com/kortexa-ai/shingi/issues/2

## Research question

Can native ternary Bonsai 2 27B support a general-purpose decision model with the TypeSafe Jev and OpenJev request and response interface? Measure the untuned first-position readout before deciding whether training is justified. A working API is separate from evidence of decision quality.

## Method and deliverables

1. Pin the API references and implement Choice, Noul, and Score with complete candidate logits, request-defined labels, and no generated JSON. Verify against the real TypeSafe SDK and boundary cases.
2. Run a batch-1 native GGUF canary on Smarty's RTX PRO 6000. Use the existing Prism build and cached Bonsai weights. Bound context and leave at least 10 GiB free. Use the GPU block wrapper; no services need to stop if measured memory permits. Its `--until-free 0` mode records the baseline without service stops; the Shingi launcher separately requires sufficient free memory.
3. Freeze deterministic development/calibration and test samples from a pinned benchmark revision. Check IDs and content for overlap. Preserve source provenance and per-example results outside Git. Evaluate at least 1,000 test decisions across eight domains, at least 200 option-order pairs, calibration, latency, and a context-length canary.
4. Report measured results, uncertainty, SDK compatibility, and limitations. Use saved hosted-model predictions only when record IDs and content match. Do not equate results from different benchmark slices or claim proprietary-model equivalence.

## Resource and release boundaries

Use one model job at a time on the explicitly authorized GPU, within an eight GPU-hour block. All model and bulk data artifacts remain ignored in Smarty's checkout. Source moves through Git. Production downtime is authorized if required; any affected service needs a claimed surface, a recorded baseline, process-owned restoration, and health verification. No paid inference or publication is part of this investigation.

Franci subsequently authorized the RTX 4090 while using the 6000 for separate experiments. The 4090 exception uses `scripts/borrow_4090.py`: require 14 GiB free before model load, retain at least 4 GiB headroom, cap context at 16K, and stop only as many of the recorded fallback LM, TTS, and Qwen ASR services as necessary. Miso stays running. The 6000 service state is outside this block. The wrapper owns restoration and enforces an eight-hour total timeout around the canary, calibration, test, shuffle, and synthetic-context phases.

Native scales and transforms remain the reference. Conversion, fine-tuning, and release formats require evidence from the baseline; the first API prototype is not a new model checkpoint.

Published Jev comparisons must use clearly attributed third-party public information. Do not benchmark hosted Jev. Distinguish a publisher's reported number from our reanalysis of its public predictions, and distinguish both from our Shingi measurements. Follow [the evaluation and publication protocol](../../docs/evaluation-protocol.md).
