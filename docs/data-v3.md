# Data v3 and the two-stage ternary model

Program issue: https://github.com/kortexa-ai/shingi/issues/9. This document fixes
the data, evaluation, training stages and gates for Shingi v0.3. Results and live
status belong in the issues and `results/`, not here.

## Goals

1. Expand training data from 2,233 records to about 27,000, weighted toward
   measured weaknesses.
2. Keep a large locked evaluation corpus, including out-of-distribution sources,
   held-out synthetic families and external suites.
3. Train a stage 1 LoRA teacher on the RTX PRO 6000. Then produce a stage 2
   model: one plain PQ2_0 GGUF with no separate adapter.
4. Prepare long-context data now. Train a long-context variant only after the 16K
   model passes its gates.

v0.2 weaknesses on its fresh test (100 records per source):

| Area | v0.2 accuracy |
|---|---|
| Emotion (GoEmotions) | 27% |
| Helpfulness (HelpSteer2) | 45% |
| Sentiment (SST-5) | 49% |
| Human-disagreement NLI (ChaosNLI) | 57% |
| Contracts (LEDGAR) | 71% |
| Banking intents (Banking77) | 72% |
| This/That spatial questions | 50.6% |
| DecisionBench Hard | 65.9% |

## License policy

`scripts/verify_licenses.py` fetches every dataset card at a pinned revision. It
records the stated license and the card SHA-256, plus pinned evidence documents
where a card is silent. The frozen record is `results/data-v3/licenses.json`.

- **Fit** only on explicit CC BY, CC0, MIT or Apache-2.0 sources.
- **Evaluate only** on share-alike, unspecified or research-use sources. Their raw
  text, prompts and per-item traces are never redistributed; only IDs, hashes and
  scores are published.
- **Exclude entirely** sources with restrictive or disputed terms, or a
  contamination risk.

| Source | License evidence | Role |
|---|---|---|
| Banking77, CLINC150, Measuring Hate Speech, HelpSteer2, HelpSteer, MASSIVE | CC BY 4.0 / 3.0 | fit |
| LEDGAR (LexGLUE) | CC BY 4.0, stated for LexGLUE as a whole | fit (moved from holdout) |
| GoEmotions | Apache-2.0 | fit (moved from holdout) |
| Civil Comments | CC0 | fit |
| MMLU, CommonsenseQA | MIT | fit |
| WinoGrande | CC BY, per upstream README; the Hub card is silent | fit |
| MNLI, ChaosNLI, BoolQ, FEVER, ARC, STS-B | share-alike or mixed | evaluation only |
| SST-5, SMS Spam | unspecified / unknown | evaluation only |
| PAWS, StrategyQA | permissive | evaluation only, by design |
| HellaSwag | wikiHow DMCA notice against upstream, 2026-09-14 | excluded |
| WANLI | generated from MNLI seed examples | excluded (would contaminate NLI evaluation) |
| UltraFeedback | labels from GPT-4 | excluded |
| OpenBookQA, PIQA, AG News, SuperGLUE | unknown or mixed | excluded |
| Yelp | restrictive dataset license | excluded, including from evaluation |

We may review the share-alike sources for fitting again after stage 1 results.

## Training data: 26,850 records, one option order each

### Natural sources (19,500 records, about 65% of tokens)

| Source | Records | Origin |
|---|---:|---|
| GoEmotions | 3,000 | jev-bench |
| HelpSteer2: helpfulness | 1,000 | jev-bench |
| HelpSteer2: verbosity | 500 | jev-bench |
| HelpSteer2: correctness, coherence, complexity | 500 each | converted |
| HelpSteer (five attributes, one per record) | 1,500 | converted |
| LEDGAR | 2,500 | jev-bench |
| MASSIVE | 1,500 | jev-bench |
| Banking77 | 1,500 | jev-bench |
| CLINC150 | 1,500 | jev-bench |
| Measuring Hate Speech | 1,500 | jev-bench |
| Civil Comments | 1,000 | jev-bench |
| MMLU | 270 | jev-bench, limited by its 285-row train pool after exclusions |
| CommonsenseQA | 1,230 | converted |
| WinoGrande | 1,000 | converted |

The jev-bench sources use the pinned transformation already used by v0.1 and
v0.2. `src/shingi/sources_v3.py` converts the others from pinned upstream files
into the same record schema:
- **CommonsenseQA:** answer texts are the choice keys, so displayed letters never
  disagree with upstream letters.
- **HelpSteer2:** jev-bench already uses almost all of its validation split. The
  converted attributes therefore evaluate on a fixed quarter of the HelpSteer2
  training states that jev-bench never used.

### Synthetic families (7,350 records, about 35% of tokens)

Code generates the synthetic data and labels in the private
`kortexa-ai/shingi-synthetic` Hugging Face dataset. The generator code, design
details and data are proprietary and never enter this repository.

| Family | Records | Target |
|---|---:|---|
| Policy application, including "not enough information" | 2,100 | rule reasoning |
| Priority routing with conflicting rules | 1,400 | DecisionBench-style decisions |
| Long-option taxonomies (20–52 options) | 1,400 | many-class choices |
| Grid, graph, schedule and ledger state | 1,750 | This/That spatial weakness |
| Rubric scoring | 700 | ordinal calibration |

This/That is not fully out-of-distribution for its spatial question families after
this training, and reports must say so.

The synthetic target is about 35% of training **tokens**. Synthetic prompts are
longer than natural ones (taxonomy averages about 970 tokens, natural sources
about 330), so 35% of tokens is about 27% of records. The first plan used 35% of
records, which estimated at 43% of tokens. Every family was scaled by the same
factor (0.70), so the family mix is unchanged. The generated pools keep the
original sizes, and the build selects a deterministic subset. Tokenization fails
unless the exact synthetic token share is between 32% and 38%.

The pilot is a fixed one-third prefix of each source's deterministic selection
order (about 8,950 records) with the same mix.

### Record filters

- Training prompts are limited to 7,600 characters before exact tokenization.
- Exact tokenization then excludes any prompt over 2,048 tokens without
  truncation.
- Evaluation states are limited to 40,000 characters, inside the 16K context.
- Choices with more than 52 options use the existing training subset sampler.

## Evaluation (never used for fitting)

| Split | Size | Content |
|---|---:|---|
| Locked in-distribution test | 200 × 16 sources | Test pools of every fitting source. Converted sources use their validation or held-out pools. |
| Out-of-distribution | 200 × 11 sources | MNLI, ChaosNLI, SST-5, SMS Spam, BoolQ, FEVER, ARC, PAWS, STS-B, StrategyQA closed and grounded |
| Synthetic transfer | 200 × 5 families | Held-out subfamilies only (domains, rule structures) |
| Development | 50 per source and family | Checkpoint selection only |
| Calibration | 50 per source and family | Calibration fitting only |
| External | 8,122 | This/That, DecisionBench Medium/Hard, public JevBench, unchanged from external-v1 |

v0.2 is re-run on the new locked test, so every comparison is matched.

## Isolation and contamination

- **Earlier splits.** Every earlier project split file is read from the prior
  artifact roots; raw downloads are skipped. New evaluation splits exclude those
  records by ID and state hash.
- **Training exclusions.** Training may reuse earlier training records. It excludes
  every earlier evaluation state and every public validation or test state,
  selected or not.
- **Split overlap.** Every pair of splits, including the long-context splits, has
  zero shared state hashes. The build fails otherwise.
- **HelpSteer prompts.** HelpSteer pairs one prompt with several responses. A
  training response is excluded when its prompt appears in any evaluation
  record.
- **Near duplicates.** A natural training record is excluded when more than 20%
  of its 13-grams appear in evaluation text. This removes reworded duplicates and
  keeps shared boilerplate, such as contract clauses and instruction templates.
- **13-gram overlap.** Synthetic training, synthetic long-context training and
  every natural evaluation text are checked for overlapping 13-grams, including
  the external benchmark states. Any synthetic hit fails the build. Natural
  overlap is reported.
- **Pretraining contamination** of natural sources remains unknown.

## Synthetic quality validation

The private generator build fails unless all of these hold:

1. **Correct by construction.** Code computes every label from a stored rule tree
   and facts. An independently written checker recomputes 100% of labels and must
   agree on every one.
2. **Property tests.**
   - Changing a decisive fact changes the label.
   - Changing a distractor never changes the label.
   - Exactly one option is correct, with unique keys and descriptions.
   - "Not enough information" is correct exactly when a required fact is missing.
   - Output is deterministic for a fixed seed.
3. **Shortcut probes.** Heuristics that see only the options, only the state or
   only the question must stay within 10 points of chance or the majority class.
   Gold positions are uniform.
4. **Diversity.** Exact-duplicate states: 0. Near duplicates (5-gram Jaccard at or
   above 0.95): under 1%. No template exceeds 10% of a family.
5. **Spot audit.** Rendered samples from every family are reviewed by eye before
   training.
6. **Difficulty screen.** Base Bonsai runs on a development sample at the first
   GPU window. We drop families near 100% and audit families near chance.
7. **Ablation.** Stage 1 compares pilot runs with and without each family when a
   family's value is unclear. A family stays only if it helps without natural-text
   regressions.

## Stage 1: LoRA teacher

The recipe is v0.2's: rank 8, alpha 16, all 64 MLP layers, AdamW with peak
learning rate 5e-5, and loss against human distributions where available. The
changes are:
- the `v3-pilot` and `v3-full` profiles;
- a 2,048-token context with its own canary (`training_canary.py --context 2048`);
- development evaluation every 256 updates, with early stopping after two checks
  without improvement.

Run order on an operator-authorized 6000 window:
1. Canary.
2. Tokenization (`prepare_training_tokens.py --max-tokens 2048`).
3. Pilot, with at most 12 GPU-hours.
4. Full run, only if pilot development accuracy is at least 2 points above v0.2
   and development NLL was still improving. At most 48 GPU-hours.
5. Calibration on the calibration split.
6. The locked test and the other evaluation sets.

Stage 1 gates, matched against v0.2:

| Measure | Required |
|---|---|
| In-distribution accuracy | at least +3.0 points; paired 95% interval lower bound above 0 |
| Out-of-distribution aggregate | paired interval lower bound at or above −1.5 points |
| DecisionBench Medium, DecisionBench Hard, public JevBench | at most −2.0 points each |
| Calibrated NLL, top-label ECE | not worse |
| Any source with a paired interval entirely below 0 | reported, and needs explicit acceptance |

## Stage 2: ternary model

The pinned base stores ternary codes with an FP16 scale per 128 weights and
signed Hadamard folding. The native LoRA reads the original activation, and
folded weights read the transformed one. The stored-domain merge delta is
therefore `(alpha / rank) · B · H(A)` for folded tensors and
`(alpha / rank) · B · A` otherwise.

1. **Merge check (CPU).** `scripts/merge_check.py` measures, for every adapted
   tensor:
   - the delta's size relative to the weights;
   - code flips under a naive re-ternarized merge;
   - how much of the delta the naive merge and the best scale-only projection
     realize.
2. **Scale-only distillation (GPU).**
   - `scripts/train_scales.py teacher` records the native teacher logits.
   - `train` learns one factor per block for every PQ2_0 matmul, about 210M
     factors. It trains against 0.7 × teacher distribution + 0.3 × label target,
     with a (factor − 1)² penalty, and selects on development NLL.
   - `src/shingi/pq2_scales.py` exports a plain PQ2_0 GGUF by changing only
     scale bytes. Unit factors reproduce the base exactly.
   - Budget: at most 24 GPU-hours.
3. **QAT, now the expected path.** On the v0.2 adapter, the merge check found:
   - the delta is 0.11–0.44% of the weight norm;
   - a naive merge flips no codes and loses the whole adapter;
   - the best scale-only projection reaches cosine 0.0885 in every tensor. That is
     1/√128, the value expected for a delta unrelated to the block-scale
     direction.

   Scale-only tuning therefore cannot imitate the LoRA in weight space. It can
   succeed only by finding a different solution in function space. It stays as
   the cheap first attempt because it keeps the file format and costs little. QAT
   is the planned follow-up:
   - latent weights initialized from the decoded ternary values;
   - ternary rounding with a straight-through estimator in forward passes;
   - teacher distillation;
   - MLP layer groups trained in turn to fit memory.

   QAT needs its own design and budget in the program issue.

Stage 2 gates, against the stage 1 teacher:

| Measure | Required |
|---|---|
| Locked-test accuracy | paired interval lower bound at or above −2.0 points; point difference at most 1.0 point |
| Calibrated NLL | at most 5% worse |
| File | same PQ2_0 size as the base, excluding metadata; loads in the unchanged runtime without an adapter |
| Single-request latency | not worse than base plus adapter |

## Long-context variant (later)

The data is prepared now in `longctx/`:

| Split | Records | Size |
|---|---:|---|
| Training | 300 | 4K–16K tokens, decisive fact at a random depth |
| Development | 50 | same distribution |
| Evaluation grid | 200 | about 16K, 32K, 64K, 128K and 250K tokens × depths 0/25/50/75/100% × 8 items |

The runtime work comes first:
- lift the 16,384-token cap;
- save and restore prefix state across questions (the Gated DeltaNet layers need a
  snapshot rather than KV truncation);
- memory gates that depend on context size;
- latency and peak memory measured at each size.

Estimated Q8 KV cache is 8.5 GiB at 262K tokens, from 16 full-attention layers
with 4 KV heads of dimension 256. That is estimated at about 17–19 GiB total, so
it probably fits a 4090 but not a 24 GB Mac. Evaluate the base at each length
before training. If the base degrades, advertise a lower validated limit.

## Reproduction

The builds run on the host with the earlier artifacts. The dataset root is
`~/data/datasets/shingi/`.

```bash
uv sync --locked --extra data --extra training
uv run --locked python scripts/verify_licenses.py
hf download kortexa-ai/shingi-synthetic --repo-type dataset --revision <rev> \
  --local-dir ~/data/datasets/shingi/synthetic
uv run --locked --extra data python scripts/prepare_data_v3.py \
  --output ~/data/datasets/shingi/v3 --cache ~/data/datasets/shingi/cache \
  --synthetic ~/data/datasets/shingi/synthetic/data/v1 --synthetic-revision <rev> \
  --prior artifacts --external artifacts/external-v1/data-01/records.jsonl
uv run --locked python scripts/prepare_training_tokens.py --model <base.gguf> \
  --data ~/data/datasets/shingi/v3/pilot --max-tokens 2048
```
