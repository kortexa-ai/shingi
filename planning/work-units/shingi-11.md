# Data v3.1: natural data, GSM8K judging and the calibration pool

Owning issue: https://github.com/kortexa-ai/shingi/issues/11
Program: https://github.com/kortexa-ai/shingi/issues/9

Prepare data v3.1 without a GPU: natural data, the synthetic counts, the frozen
splits and their tokenization. The plan is in
[docs/data-v3.1.md](../../docs/data-v3.1.md); build facts are in
[results/data-v3.1/REPORT.md](../../results/data-v3.1/REPORT.md).

## Decisions

- **Profiles.** `FIT` and `SYNTHETIC` keep the frozen v3 values. v3.1 adds
  `FIT_V31` and `SYNTHETIC_V31`, and `prepare_data_v3.py --profile` selects one
  (default `v3`). The v3 path produces the same selection and manifests as before.
- **Training profile names.** v3.1 manifests keep `v3-pilot` and `v3-full`, so the
  trainer applies the v3 recipe, and add `data_version: v3.1`. The trainer's
  allowlist still reads only the v3 sources, so it rejects v3.1 data until it
  uses `PROFILES[data_version]`. That fails closed.
- **Seed.** v3.1 reuses the v3 seed, so unchanged natural sources select largely
  the same training records and the comparison with v3 stays matched.
- **GSM8K pools.** GSM8K train is the training pool and GSM8K test is the
  evaluation pool; GSM8K has no validation split.
- **GSM8K variants.** Half gold by a seeded row hash; the rest split over
  `final_wrong`, `step_wrong` and `step_wrong_propagated`. Propagation recomputes
  later steps from their calculator expressions with exact fractions. Skipped
  rows are not replaced, so gold is 52–55% of records. Wrong values stay whole
  numbers and keep thousands separators, so format does not reveal the label.
  Five rows with negative finals are skipped.
- **Calibration balance.** Each yes/no source gives min(40, available yes,
  available no) records of each label, so every source is exactly balanced.
  The build fails below 150 records, outside 40–60% gold yes, or with fewer than
  three sources. Yes/no calibration records are selected before development,
  because Civil Comments validation has only 39 positives in 500.
- **Development.** Unchanged. Balancing it would spend the scarce Civil Comments
  positives and drop synthetic development records.
- **Build gate.** The v3.1 build refuses to run while any `SYNTHETIC_V31` count is
  unset.
- **Fresh natural evaluation.** v3.1 draws its own locked test, OOD, development
  and calibration records; the v3 locked test is not reused. v0.2, stage 1 and
  v3.1 are compared on the v3.1 test and OOD splits.
- **Exhausted pools.** HelpSteer2 helpfulness and verbosity share 200 free test
  and 151 free validation states, and SMS Spam has 187 free test records. v3.1
  takes 100 test and 25 calibration records from each HelpSteer2 source and 180
  OOD records from SMS Spam (`EVAL_COUNTS_V31`). Other sources keep v3's counts.
- **Synthetic evaluation reuse.** Synthetic development, calibration, transfer
  and long-context evaluation selection excludes earlier training records and
  this build's own selections, not earlier evaluation records. The old families'
  files are byte-identical to v3, so an exclusion would empty them. Training still
  excludes every v3 and v3.1 evaluation state, and in v3.1 also every synthetic
  evaluation record that was not selected. The v3 profile is unchanged.
- **Synthetic counts.** The proposal came to 40.95% of exact tokens on a probe
  build; every family was scaled by 0.78 and rounded to 50s: judge 1,150,
  severity 800, arithmetic 800, policy 1,000, routing 700, taxonomy 550, state
  1,000, rubric 300 (6,300 records).
- **Prior roots.** Every `artifacts/` directory except `stage2/`, plus the v3
  dataset. Stage 2 holds teacher outputs for v3 records that the v3 root already
  covers, its file names would classify them as evaluation files, and a stage 2
  GPU block was writing there during the build.
- **Trainer.** The allowlist comes from `PROFILES[data_version]` (default `v3`);
  unknown data versions are rejected. v3.1 runs use torch seed 20260925. `run.json`
  and the token manifest record `data_version`.

## Validation

- `results/data-v3.1/licenses.json` clears all 13 fitting sources, including
  GSM8K (MIT).
- GSM8K conversion statistics are in `results/data-v3.1/REPORT.md`.
- Unit tests cover markup stripping, each variant's exact change, labels,
  determinism, skip reasons, the v3.1 mix and its build refusal, and the
  balanced calibration selection and gate.
- A tiny-fixture build test shows that v3.1 draws fresh natural evaluation,
  reuses synthetic evaluation files and keeps unselected synthetic evaluation
  states out of training, and that the v3 profile still excludes them. Trainer
  tests cover the v3 and v3.1 allowlists. 126 tests pass.
- Synthetic v1.1: all 23 old-family files match v1 by SHA-256.
- Frozen build at `~/data/datasets/shingi/v3.1` (source `4370424`): 25,200
  training records, test 3,200, OOD 2,180, development 1,300, calibration 1,276,
  transfer 1,700. Root manifest SHA-256 `1e3ebc74…d8d4af`.
- Exact synthetic token share: 35.15% full (25,199 prompts, one excluded as too
  long), 34.95% pilot (8,398).
- Zero split overlap; zero synthetic 13-gram contamination; no v3.1 natural
  evaluation record shares an ID or state with any v3 split; old-family synthetic
  development and transfer equal v3's, calibration is 245 of v3's 250.
- Calibration yes/no: 220 records, 50.0% gold yes, six sources.
