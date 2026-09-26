# Data v3.1

Issue: https://github.com/kortexa-ai/shingi/issues/11. Program:
https://github.com/kortexa-ai/shingi/issues/9. This plan lists only the changes
from [data v3](data-v3.md); everything not named here is unchanged. Results
belong in `results/data-v3.1/`.

## Motivation

The stage 1 teacher gained on its trained sources but not as a general judge
([stage 1 report](../results/stage1/REPORT.md), Interpretation). The flip analysis
found two systematic shifts: lenient pass/fail judging, which says yes to
responses with substantive errors, and conservative ordinal scores on holistic
severity, urgency and fault scales. It also found losses on numeric and date
items. The likely causes are the HelpSteer share, where a response with a small
error still scores 3 of 4, and the rubric family's cumulative "highest level
whose requirements are all met" rule. The calibration split had 73 yes/no
records with 20.5% gold yes, mostly from Civil Comments, so its fitted bias
followed that base rate (Calibration section).

## Synthetic families

Three new families join the five v3 families in the private
`kortexa-ai/shingi-synthetic` dataset. Exact counts are pending; the synthetic
share stays about 35% of training tokens (32–38% exact), and the new families
take their share from the existing ones.

| Family | Purpose |
|---|---|
| `synth_judge` | Strict pass/fail judging: a request and a response with planted errors and computed labels. The response passes only when every planted error is absent. |
| `synth_severity` | Holistic ordinal scales: severity, urgency or fault from weighted, counted factors, not cumulative criteria. |
| `synth_arithmetic` | Numeric and date arithmetic under stated rules: refunds, caps, prorating, deadlines and grace periods, with near-miss options. |

The rubric family also gains non-cumulative scales. Held-out subfamilies remain
for transfer evaluation. `SYNTHETIC_V31` in `src/shingi/sources_v3.py` lists
the families; the v3.1 build refuses to run until their counts are set.

## GSM8K judging

`gsm8k_judge` converts GSM8K (MIT, `openai/gsm8k` at `740312ad`) into yes/no
records. GSM8K has no validation split, so its train split is the training pool
and its test split is the evaluation pool.

- **State.** The problem and a response: the rationale with the calculator
  markup (`<<48/2=24>>`) removed and the final line shown as `Answer: <n>`.
- **Question.** Does the response fully and correctly solve the problem? Any
  arithmetic error, unsupported step or wrong final answer means no.
- **Variants.** A seeded hash of the row picks one variant per row, half of
  them gold:
  - `gold` (yes): the upstream rationale.
  - `final_wrong` (no): a nearby wrong final answer, also in the last step when
    that step produces the final.
  - `step_wrong` (no): one intermediate result changed without propagation, so
    the derivation is inconsistent.
  - `step_wrong_propagated` (no): one intermediate result changed and every
    later step recomputed, so the final answer is wrong too.
- **Skips.** A row is skipped, with the reason recorded, when its variant
  cannot be applied cleanly: no intermediate step, a changed value that also
  appears in the problem or earlier in the rationale, a propagated result that
  becomes fractional or negative, or a final answer the change does not reach.
  Wrong values are never fractional where GSM8K's are whole, so number format
  does not reveal the label.

## Natural mix

| Source | v3 | v3.1 |
|---|---:|---:|
| HelpSteer (five attributes) | 1,500 | 500 |
| HelpSteer2 helpfulness | 1,000 | 700 |
| HelpSteer2 verbosity | 500 | 300 |
| HelpSteer2 correctness, coherence, complexity | 500 each | 300 each |
| GSM8K judging | — | 1,500 |
| **Natural total** | **19,500** | **18,900** |

The other natural sources keep their v3 counts. The HelpSteer reduction is an
experiment; the build manifest records the exact counts.

## Calibration pool

The calibration split selects yes/no records separately from the others:

- every natural and synthetic source with yes/no records in its calibration pool
  takes part;
- each source gives equal gold yes and no: the smaller of 40 per label and the
  available records of either label;
- the build fails unless there are at least 150 yes/no records, gold yes is
  between 40% and 60%, and at least three sources contribute.

Yes/no records are chosen before development and calibration records, because
Civil Comments validation has only 39 positives. Other calibration records keep
the v3 rule (50 per natural source, the synthetic calibration files whole).
Development keeps the v3 selection: balancing it would spend those positives
and drop synthetic development records. Calibration stays temperature-only.

## Protocol

- The v3 locked test is reused as the matched comparison set for v3.1. It is not
  used for selection. v3.1 builds its own splits, disjoint from every earlier
  split, including all of v3.
- JevBench and DecisionBench informed this design through the stage 1 flip
  analysis. For v3.1 they are development-informed, and reports must say so.
- This/That remains not fully out-of-distribution.
- Generators are designed from task descriptions, never from JevBench,
  DecisionBench or This/That items. The 13-gram contamination check against
  every evaluation set stays a hard gate.

## Reproduction

`prepare_data_v3.py --profile v3` still reproduces v3. For v3.1, the earlier
artifacts and the v3 dataset are both prior roots:

```bash
uv run --locked python scripts/verify_licenses.py --output results/data-v3.1/licenses.json
uv run --locked --extra data python scripts/prepare_data_v3.py --profile v3.1 \
  --output ~/data/datasets/shingi/v3.1 --cache ~/data/datasets/shingi/cache \
  --synthetic <v3.1 synthetic data> --synthetic-revision <rev> \
  --prior artifacts ~/data/datasets/shingi/v3 \
  --external artifacts/external-v1/data-01/records.jsonl
```

The v3.1 training manifests keep the `v3-pilot` and `v3-full` profile names and
add `data_version: v3.1`. The trainer's source allowlist must read
`data_version` (through `sources_v3.PROFILES`) before v3.1 training; until then
it rejects the new sources, so a v3.1 run cannot start by mistake.
