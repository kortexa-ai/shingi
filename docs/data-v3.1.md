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
`kortexa-ai/shingi-synthetic` dataset (generator v1.1 at revision `2a46d571`).
The old families' files there are byte-identical to v3's. The synthetic share
stays about 35% of training tokens (32–38% exact), and the new families take
their share from the existing ones.

| Family | Purpose |
|---|---|
| `synth_judge` | Strict pass/fail judging: a request and a response with planted errors and computed labels. The response passes only when every planted error is absent. |
| `synth_severity` | Holistic ordinal scales: severity, urgency or fault from weighted, counted factors, not cumulative criteria. |
| `synth_arithmetic` | Numeric and date arithmetic under stated rules: refunds, caps, prorating, deadlines and grace periods, with near-miss options. |

The rubric family is unchanged from v3; non-cumulative scales come from
`synth_severity`. Held-out subfamilies remain for transfer evaluation.

Training counts (`SYNTHETIC_V31` in `src/shingi/sources_v3.py`):

| Family | v3 | Proposal | v3.1 |
|---|---:|---:|---:|
| `synth_judge` | — | 1,500 | 1,150 |
| `synth_severity` | — | 1,000 | 800 |
| `synth_arithmetic` | — | 1,000 | 800 |
| `synth_policy` | 1,750 | 1,300 | 1,000 |
| `synth_routing` | 1,150 | 900 | 700 |
| `synth_taxonomy` | 1,150 | 700 | 550 |
| `synth_state` | 1,450 | 1,300 | 1,000 |
| `synth_rubric` | 600 | 400 | 300 |
| **Synthetic total** | **6,100** | **8,100** | **6,300** |

Exact tokenization of an oversized probe build put the proposal at 41% of Bonsai
tokens. Every family was scaled by 0.78 and rounded to 50 records. The frozen
build measures 35.1% synthetic tokens in the full profile and 35.0% in the pilot.
The new families average 260–405 tokens per record, against 300–1,150 for the
old ones, so 35% of tokens is now 25.0% of records (v3: 23.8%).

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
the v3 rule (50 per natural source, the synthetic calibration files whole),
except for the HelpSteer2 exceptions under Evaluation splits.
Development keeps the v3 selection: balancing it would spend those positives
and drop synthetic development records. Calibration stays temperature-only.

## Evaluation splits

v3.1 draws a fresh locked test, out-of-distribution, development and calibration
selection from the natural pools that remain after every earlier record is
excluded by ID and state hash, including all of v3. The v3.1 locked test and
out-of-distribution split are the matched comparison sets: v0.2, stage 1 and
v3.1 are all evaluated on them. They are not used for selection.

Three natural pools cannot give the v3 per-source counts after these exclusions
(`EVAL_COUNTS_V31`):

| Source | Split | v3 | v3.1 | Why |
|---|---|---:|---:|---|
| HelpSteer2 helpfulness, verbosity | locked test | 200 each | 100 each | They rate the same responses and share 200 free test states. |
| HelpSteer2 helpfulness, verbosity | calibration | 50 each | 25 each | They share 151 free validation states; development takes 100. |
| SMS Spam | out-of-distribution | 200 | 180 | 187 free test records remain. |

Synthetic evaluation files are generator-owned, and the transfer subfamilies are
held out by design. The old families' development, calibration, transfer and
long-context evaluation files are byte-identical to v3's, so v3.1 reuses them:
synthetic evaluation selection excludes earlier training records and this
build's own selections, but not earlier evaluation records. The yes/no balance
applies to synthetic calibration records too, so five old-family yes/no records
in v3's calibration split are not in v3.1's.

Training still excludes every evaluation state of v3 and v3.1, natural and
synthetic, including synthetic evaluation records that v3.1 did not select.

## Protocol

- The v3.1 locked test is new; the v3 locked test is not reused.
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
REV=2a46d5719326cda7fd376793444283e67b987a8a
hf download kortexa-ai/shingi-synthetic --repo-type dataset --revision $REV \
  --local-dir ~/data/datasets/shingi/synthetic-2a46d571
uv run --locked python scripts/verify_licenses.py --output results/data-v3.1/licenses.json
PRIOR=$(ls -d artifacts/*/ | grep -v '^artifacts/stage2/$' | sed 's|/$||')
uv run --locked --extra data python scripts/prepare_data_v3.py --profile v3.1 \
  --output ~/data/datasets/shingi/v3.1 --cache ~/data/datasets/shingi/cache \
  --synthetic ~/data/datasets/shingi/synthetic-2a46d571/data/v1.1 --synthetic-revision $REV \
  --prior $PRIOR ~/data/datasets/shingi/v3 \
  --external artifacts/external-v1/data-01/records.jsonl
uv run --locked python scripts/prepare_training_tokens.py --model <base.gguf> \
  --data ~/data/datasets/shingi/v3.1/full --max-tokens 2048
```

The prior roots are every `artifacts/` directory except `stage2/`. Stage 2
holds teacher outputs for v3 training and development records, which the v3
root already covers. Its files are not named as training files, so the build
would classify them as evaluation files, and any v3 pilot training record they
identify would leave v3.1 training.

The v3.1 training manifests keep the `v3-pilot` and `v3-full` profile names and
add `data_version: v3.1`. The trainer reads its fitting allowlist from
`sources_v3.PROFILES[data_version]` (default `v3`) and rejects any other
source or data version. It seeds v3.1 runs with 20260925 (v3: 20260924) and
records `data_version` in `run.json`. Tokenization applies the 32–38% synthetic
token gate to v3.1 manifests and records `data_version` in its manifest.
