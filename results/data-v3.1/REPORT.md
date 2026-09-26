# Data v3.1 preparation

Issue: https://github.com/kortexa-ai/shingi/issues/11. Plan:
[docs/data-v3.1.md](../../docs/data-v3.1.md). No model was trained or evaluated
for this report. The GPU was not used.

## License verification

`licenses.json` repeats the v3 audit with GSM8K added.

- **Cleared for fitting (13):** the twelve v3 fitting sources and GSM8K
  (`openai/gsm8k` at `740312ad`, card license `mit`).
- **Evaluation only (11)** and **excluded (8):** unchanged from v3.
- Every dataset card that v3 also audited has the same SHA-256 as in
  `results/data-v3/licenses.json`.

These are publisher-stated terms, not legal conclusions.

## GSM8K judging conversion

`gsm8k_judge_pool` ran on the CPU over both pinned GSM8K files with the build
seed `shingi-data-v3-20260924`. The training pool is GSM8K train; the evaluation
pool is GSM8K test.

| Pool | Rows | Records | Skipped | Gold | Final wrong | Step wrong | Step wrong, propagated |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 7,473 | 6,919 | 554 (7.4%) | 54.5% | 18.1% | 16.0% | 11.4% |
| Evaluation (test) | 1,319 | 1,212 | 107 (8.1%) | 52.1% | 19.2% | 17.0% | 11.6% |

Gold is above 50% because only perturbed variants are skipped: in the train
pool 437 skips are `step_wrong_propagated`, 116 are `step_wrong` and one is
`gold`.

Skip reasons:

| Reason | Train | Test |
|---|---:|---:|
| No intermediate calculator step | 135 | 29 |
| Propagated result would be fractional | 93 | 18 |
| Changed value is ambiguous (repeated earlier or in an unaffected step) | 74 | 10 |
| Propagated result has more than two decimals | 72 | 11 |
| Change does not reach the final answer | 49 | 11 |
| Changed value appears in the problem (propagated) | 44 | 6 |
| Step value appears in the problem | 31 | 5 |
| No calculator markup | 31 | 7 |
| Every intermediate equals the final answer | 8 | 2 |
| Propagated result would be negative | 8 | 2 |
| Other (propagation check, unparsed step, unevaluable expression) | 6 | 4 |
| Negative final answer | 3 | 2 |

Rendered lengths, in characters:

| Pool | Response median | Response p95 | Response max | Training prompt median | Training prompt max |
|---|---:|---:|---:|---:|---:|
| Train | 223 | 503 | 1,140 | 977 | 2,203 |
| Test | 232 | 504 | 954 | 994 | 2,068 |

- No response contains calculator markup or the `####` marker.
- Median response length differs by at most 17 characters between variants, and
  every wrong final answer is a whole number, like the gold answers.
- Every training prompt is far below the 7,600-character limit.

**Known limits.**
- Rows with a single calculator step can only be gold or `final_wrong`.
- Labels trust the upstream gold rationale. GSM8K has a small number of known
  upstream errors, which would make a few gold records wrong.

## Build

Pending. The v3.1 synthetic families are still being generated, and the build
refuses to run until their counts are set in `SYNTHETIC_V31`.
