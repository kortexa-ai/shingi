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

## Synthetic data v1.1 (private)

`kortexa-ai/shingi-synthetic` at revision `2a46d571`, generator v1.1.0, code
commit `07118176`, seed 20260924. Its manifest reports independent checker
agreement on 20,750 of 20,750 records and a passing QA build.

- **Old families.** All 23 files of policy, routing, taxonomy, state, rubric and
  long context have the same SHA-256 as in v1 (revision `279e4ff8`) and as
  recorded in the v1.1 manifest.
- **New families:**

| Family | Train | Development | Calibration | Transfer | Yes/no share of train |
|---|---:|---:|---:|---:|---:|
| `judge` | 3,000 | 100 | 200 | 300 | 79% |
| `severity` | 2,000 | 50 | 50 | 200 | 0% (score) |
| `arithmetic` | 2,000 | 50 | 50 | 200 | 20% (rest choice) |

## Build

Built on the training host on the CPU at source commit `4370424`, into
`~/data/datasets/shingi/v3.1`, with `--profile v3.1`, synthetic revision
`2a46d571`, and the external records file for the 13-gram check. The prior roots
were `~/data/datasets/shingi/v3` and every `artifacts/` directory except
`stage2/` (see the plan, Reproduction).

| Artifact | SHA-256 |
|---|---|
| Root manifest | `1e3ebc747608e7010e6f83e9ae203e41892851ab174391c6fcffa5efa9d8d4af` |
| Locked test | `efe878e9ed70bc46bd15ddb2c28d762d28ee6c281ceb783818f5b78c88ff7a84` |
| Out-of-distribution | `6963b2f03b102cd9cc49e38776d91fc9c4827ba9a2c798f9bb9a4e11802259de` |
| Training records | `51dcde4fa9cc2aa7ec72aa1fc02ba47e2ead0fc8fcdafeb7b29c249bfa36e7ad` |
| Full profile manifest | `058ea3b3e3bc5ada7c528e159007bc183c4e2ed8735c6b8231d47db18c42c873` |
| Pilot profile manifest | `f178447b0f3c914519f598f8cfc8bec5dda279fbbb3e381c962bf40213d73947` |

| Split | Records |
|---|---:|
| Training | 25,200 (18,900 natural, 6,300 synthetic) |
| Pilot | 8,398 |
| Locked test | 3,200 (17 sources) |
| Out-of-distribution | 2,180 (11 sources) |
| Development | 1,300 (850 natural, 450 synthetic) |
| Calibration | 1,276 (820 natural, 456 synthetic) |
| Synthetic transfer | 1,700 (8 families) |
| Long context | 300 training / 50 development / 200 evaluation |

Evaluation per source: 200 in the locked test and 200 out of distribution,
except HelpSteer2 helpfulness and verbosity (100 test each) and SMS Spam (180).
Development has 50 per source and family, and 100 for `judge`. Calibration has
50 per natural source, except Civil Comments (40, all yes/no), GSM8K (80, all
yes/no) and HelpSteer2 helpfulness and verbosity (25 each).

Training records by source (full; the pilot is the first third of each):

| Source | Full | Pilot | Exact tokens (full) | Mean tokens |
|---|---:|---:|---:|---:|
| GoEmotions | 3,000 | 1,000 | 601,540 | 201 |
| LEDGAR | 2,500 | 833 | 674,761 | 270 |
| Banking77 | 1,500 | 500 | 339,347 | 226 |
| CLINC150 | 1,500 | 500 | 280,360 | 187 |
| MASSIVE | 1,500 | 500 | 300,866 | 201 |
| Measuring Hate Speech | 1,500 | 500 | 225,426 | 150 |
| GSM8K judging | 1,500 | 500 | 397,575 | 265 |
| CommonsenseQA | 1,230 | 410 | 113,998 | 93 |
| Civil Comments | 1,000 | 333 | 133,948 | 134 |
| WinoGrande | 1,000 | 333 | 84,643 | 85 |
| HelpSteer2 helpfulness | 700 | 233 | 481,261 | 688 |
| HelpSteer | 500 | 167 | 440,780 | 882 |
| HelpSteer2 verbosity | 300 | 100 | 202,376 | 677 |
| HelpSteer2 correctness | 300 | 100 | 198,757 | 663 |
| HelpSteer2 coherence | 300 | 100 | 201,082 | 670 |
| HelpSteer2 complexity | 300 | 100 | 188,024 | 627 |
| MMLU | 270 | 90 | 43,605 | 162 |
| `synth_judge` | 1,150 | 383 | 301,355 | 262 |
| `synth_policy` | 1,000 | 333 | 445,013 | 445 |
| `synth_state` | 1,000 | 333 | 295,948 | 296 |
| `synth_severity` | 800 | 267 | 325,045 | 406 |
| `synth_arithmetic` | 800 | 267 | 234,155 | 293 |
| `synth_routing` | 700 | 233 | 330,143 | 472 |
| `synth_taxonomy` | 550 | 183 | 630,864 | 1,147 |
| `synth_rubric` | 300 | 100 | 97,394 | 325 |

The HelpSteer2 verbosity token total covers 299 records; one is excluded as too
long (below).

Exact native tokenization, with a 2,048-token limit and no truncation:

| Profile | Prompts | Excluded as too long | Natural tokens | Synthetic tokens | Synthetic share |
|---|---:|---:|---:|---:|---:|
| Pilot | 8,398 | 0 | 1,646,222 | 884,535 | 34.95% |
| Full | 25,199 | 1 (HelpSteer2 verbosity, 2,609 tokens) | 4,908,349 | 2,659,917 | 35.15% |

Sizing: a probe build with every synthetic family oversized was tokenized
exactly. Synthetic selection is a deterministic prefix, so the shares of smaller
counts follow from it. The proposal (8,100 synthetic records) came to 40.95%.
Scaled by 0.78 and rounded to 50s (6,300 records), the prediction was 35.15%
full and 34.95% pilot, and the frozen build measured exactly that.

**Isolation.**
- Every pair of splits, including long context, has zero shared state hashes,
  and no record ID appears in two splits.
- Natural evaluation records against every v3 split file, by ID and state hash:

| v3.1 split | Natural records | ID in v3 | State in v3 |
|---|---:|---:|---:|
| Locked test | 3,200 | 0 | 0 |
| Out-of-distribution | 2,180 | 0 | 0 |
| Development | 850 | 0 | 0 |
| Calibration | 820 | 0 | 0 |

- Old-family synthetic evaluation records against v3:
  - Development (250), transfer (1,000), long-context development (50) and
    evaluation (200): the same IDs, and every record is identical.
  - Calibration: 245 of v3's 250, each identical. The yes/no balance left out
    five yes/no records (one policy, four state).
- No training record has an ID or state from any v3 evaluation split. 17,230
  permitted earlier training records were reused.

**Contamination.** Synthetic 13-gram contamination is 0 in every family,
including long context, against the natural evaluation text and all external
benchmark states. Natural training records sharing at least one 13-gram with
evaluation text:

| Source | Records |
|---|---:|
| LEDGAR | 251 |
| HelpSteer | 57 |
| HelpSteer2 helpfulness | 41 |
| HelpSteer2 correctness | 22 |
| HelpSteer2 coherence | 21 |
| HelpSteer2 verbosity | 14 |
| MMLU | 14 |
| HelpSteer2 complexity | 11 |
| Banking77 | 1 |
| Other sources, including GSM8K judging | 0 |

**Calibration yes/no pool.** 220 records, 110 gold yes (50.0%), six sources; the
gate needs 150 records, 40–60% gold yes and three sources.

| Source | Records | Gold yes | Available yes | Available no |
|---|---:|---:|---:|---:|
| GSM8K judging | 80 | 40 | 527 | 485 |
| `synth_judge` | 80 | 40 | 70 | 97 |
| Civil Comments | 40 | 20 | 20 | 231 |
| `synth_state` | 14 | 7 | 7 | 11 |
| `synth_policy` | 4 | 2 | 2 | 3 |
| `synth_arithmetic` | 2 | 1 | 3 | 1 |

Civil Comments has 20 free positives left in its validation pool after earlier
splits.

**Known limits.**
- The HelpSteer2 helpfulness and verbosity test sets are half the size of the
  other sources' sets, so their per-source intervals are wider.
- Old-family synthetic evaluation sets are the same as v3's, so stage 1 has
  already been evaluated on them. They were not used for selection.
