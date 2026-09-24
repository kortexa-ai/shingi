# Data v3 and two-stage training preparation

Owning issue: https://github.com/kortexa-ai/shingi/issues/10
Program: https://github.com/kortexa-ai/shingi/issues/9

Prepare the data and code for v0.3 without training or using a GPU. The locked
plan is in [docs/data-v3.md](../../docs/data-v3.md).

## Decisions

- **Mix.** 25,600 training records, with one option order per record: 19,500
  natural and 6,100 synthetic. The synthetic target is about 35% of exact
  training tokens, which is about 24% of records, because synthetic prompts are
  longer and tokenize more densely. Franci chose tokens over records on
  2026-09-24. Tokenization enforces a 32–38% synthetic token share. A character
  estimate understated the share: 34.9% estimated, 39.1% exact for 7,350 records.
- **Licenses.** Fitting uses only CC BY, CC0, MIT and Apache-2.0 sources.
  Share-alike, unspecified and research-use sources are evaluation-only.
  HellaSwag, WANLI, UltraFeedback, OpenBookQA, PIQA, AG News, SuperGLUE and Yelp
  are excluded.
- **Holdouts.** GoEmotions and LEDGAR move from holdouts into training.
  StrategyQA and PAWS stay out-of-distribution by choice, although they are
  permissive.
- **MMLU.** jev-bench has only 285 MMLU training rows, so MMLU contributes 270
  records and CommonsenseQA takes the rest of its allocation.
- **CommonsenseQA keys.** Choice keys are answer texts, so displayed option letters
  never conflict with upstream letters.
- **HelpSteer2 evaluation.** jev-bench already uses 1,000 of the 1,038 HelpSteer2
  validation states. The converted attributes evaluate on a fixed quarter of the
  training states that jev-bench never used.
- **Synthetic data.** The synthetic generators and data are proprietary to Kortexa
  and live in the private `kortexa-ai/shingi-synthetic` Hugging Face dataset. Only
  record counts, checksums and quality summaries appear here.
- **Dataset root.** Datasets live under `~/data/datasets/shingi/` on the training
  host. `~/data` is `/mnt/data` on smarty.
- **Stage 2 domain.** Native LoRA reads the original activation. The stored-domain
  merge delta for Hadamard-folded tensors is therefore `(alpha / rank) · B · H(A)`.

## Validation

- `results/data-v3/licenses.json` clears every fitting source.
- Unit tests cover:
  - converters, selection and prior-split classification;
  - the 30K/35% mix and the v3 source allowlist;
  - PQ2 scale codecs, including byte-exact unit factors;
  - the Hadamard merge identity;
  - the scale-factor gradient.
- The merge check on the v0.2 adapter and the data build are recorded in
  `results/data-v3/REPORT.md`.
