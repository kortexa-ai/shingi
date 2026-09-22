# API contract and CUDA v1 limits

Reference date: 2026-09-22.

The interface is `POST /v1/systemone`, with `model`, `state`, and a map of named `questions`. It returns the actual Shingi model ID, matching `answers`, and token `usage`. Question IDs are routing keys; the model does not see them. All questions see the same state and are evaluated independently. The initial native backend serializes inference; it makes no claim of Jev's parallel latency.

| Primitive | Input | Output |
| --- | --- | --- |
| Choice | 1–255 named criteria, with string, object, array, or null descriptions | Highest-probability `choice`, full `probabilities`, `confidence` |
| Noul | Instructions and optional `true` / `false` criteria | `noul`, the probability of yes; no confidence field |
| Score | 2–10 ordered descriptions | Expected zero-based `score`, `legend`, full `probabilities`, `confidence` |

State and instructions accept strings, objects, and arrays. The API rejects malformed requests with HTTP 422. Native backend failure returns 503. Output probabilities retain full floating-point precision and sum to one within floating-point tolerance. `output_tokens` is zero: the native backend reads logits without generating a token.

`shingi`, `shingi-latest`, and the active model ID select the local model. The selected adapter is `shingi-bonsai-2-27b-v0.1`; the unchanged base is `shingi-bonsai-2-27b-baseline`. A custom weight pair reports `shingi-custom`. A different explicit model ID is rejected. For client migration, `jev-latest` and `openjev` are accepted aliases too; the response always names Shingi. This does not route to or impersonate either upstream provider.

## Readout and confidence

CUDA v0.1 sorts Choice keys lexicographically before assigning candidate letters
and chunks. Reordering the same JSON map therefore gives an identical prompt.
Score criteria retain their ordinal list order; Noul remains yes/no. This is an
interface guarantee, not learned resistance to all prompt or label changes.
The readout identifier is `bonsai-sorted-choice-v2`. Historical results used
input-map ordering (`bonsai-letter-v1`), and their calibration files are not
silently applied to this release.

The first 52 choices map to unique single-token ASCII letters. The native runtime checks tokenization and reads every candidate logit directly from the final prompt position. It does not use truncated top-k log probabilities. It resets the model context between calls. Prompts use the Bonsai chat template with thinking disabled, checked against the installed Prism runtime's `/apply-template` endpoint.

For more than 52 options, Shingi uses the same mathematical approximation as OpenJev: read chunks, compare their winners, and use each winner to place its chunk on a shared probability scale. Every option remains in the final distribution. This requires multiple passes and is sensitive to chunk context; it is not an exact global softmax. Tests cover 52, 53, and 255 options. Quality at these sizes requires separate measurement.

For K choices, confidence is `(K * max_probability - 1) / (K - 1)`, with confidence 1 for a singleton. Score confidence measures the mean distance from the modal level, normalized by a uniform distribution's mean distance from its center, then clipped below at zero. These follow the public TypeSafe adapter formulas. Confidence is a distribution statistic, not an empirical probability of correctness.

Without `--calibration`, either weight configuration uses temperature 1 with no Noul bias adjustment. The model ID identifies the weights; `/v1/version` identifies the active calibration. Release measurements require the supplied calibration file. OpenJev's fitted calibration constants do not transfer automatically to Bonsai. The first investigation fitted calibration on 350 separate validation records. Pass a JSON file with `--calibration`; `/v1/version` reports the active parameters and file hash. Calibration must match both the base and adapter hashes. The trained release uses its own 240-record calibration, with temperature 1.25, additional Noul temperature divisor 1.2, and Noul bias -1.0; pass `--adapter` with `release/calibration.json`. See the [release measurements and tradeoffs](../results/cuda-v0.1/REPORT.md).

## Explicit limits

- Text and JSON only. OpenJev screenshot inputs are rejected; this backend does not load Bonsai's vision projector.
- Default context: 16,384 tokens. Oversized prompts fail rather than truncate. CUDA v1 rejects larger configured contexts.
- Up to 256 questions per request in this prototype, processed sequentially.
- Localhost only, with no authentication or hosted billing/rate-limit behavior. This is a research service.
- CPU contract tests use a deterministic backend. They establish serialization and math correctness, not model quality.

## Primary references

- [TypeSafe HTTP API](https://docs.typesafe.ai/api), [Choice](https://docs.typesafe.ai/primitives/choice), [Score](https://docs.typesafe.ai/primitives/score), and [Noul](https://docs.typesafe.ai/primitives/noul).
- [TypeSafe confidence formulas, revision adffc2e](https://github.com/typesafe-ai/system-one-adapter-python/blob/adffc2eab300a4fa3c0e92252d4ffd6ceaa53700/src/system_one_adapter/_utils/confidence_metrics.py).
- [OpenJev helper, revision 5ec9e5f](https://huggingface.co/openjev/openjev/blob/5ec9e5fd2f80a6fff386779b1e5ac7e389971889/helper/shim.py), Apache-2.0. Its letter prompt and chunk readout inform this independently implemented baseline. No OpenJev weights are included.
- [Bonsai GGUF, revision 6ed5e12](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/6ed5e12bf84b7a63069882c91dd9e9218647d17b), using native PQ2_0 weights and the public Prism runtime at `d8f26eec76da6d09bb708bcba51ef64b8cd868a3`.
