# Data v3.1: natural data, GSM8K judging and the calibration pool

Owning issue: https://github.com/kortexa-ai/shingi/issues/11
Program: https://github.com/kortexa-ai/shingi/issues/9

Prepare the natural-data side of v3.1 without a GPU. The plan is in
[docs/data-v3.1.md](../../docs/data-v3.1.md). The synthetic families are a
separate part of this issue.

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
  unset (TODO(#11)).

## Validation

- `results/data-v3.1/licenses.json` clears all 13 fitting sources, including
  GSM8K (MIT).
- GSM8K conversion statistics are in `results/data-v3.1/REPORT.md`.
- Unit tests cover markup stripping, each variant's exact change, labels,
  determinism, skip reasons, the v3.1 mix and its build refusal, and the
  balanced calibration selection and gate.
