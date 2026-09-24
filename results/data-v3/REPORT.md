# Data v3 preparation

Issue: https://github.com/kortexa-ai/shingi/issues/10. Plan:
[docs/data-v3.md](../../docs/data-v3.md). No model was trained or evaluated for
this report. The GPU was not used.

## License verification

`licenses.json` records every dataset card at a pinned revision, with its stated
license and SHA-256, plus pinned evidence documents.

- **Cleared for fitting (12):** Banking77, CLINC150, MMLU, HelpSteer2,
  Measuring Hate Speech, Civil Comments, GoEmotions, LEDGAR, MASSIVE, HelpSteer,
  CommonsenseQA and WinoGrande.
  - WinoGrande has no license field on its Hub card. It is cleared by the pinned
    upstream README statement "The dataset is licensed under CC-BY".
- **Evaluation only (11):** MNLI, ChaosNLI, SST-5, SMS Spam, BoolQ, FEVER, ARC,
  PAWS, STS-B, and StrategyQA closed and grounded.
- **Excluded (8):** HellaSwag, WANLI, UltraFeedback, OpenBookQA, PIQA, AG News,
  Yelp and SuperGLUE.
  - HellaSwag's exclusion rests on the 2026-09-14 wikiHow DMCA notice, which names
    `rowanz/hellaswag`. That repository now returns HTTP 451 on GitHub.

These are publisher-stated terms, not legal conclusions.

## Stage 2 merge check on the v0.2 adapter

`scripts/merge_check.py` ran on the CPU against the released v0.2 adapter
(alpha 16, rank 8), covering all 192 adapted MLP tensors.

| Tensor | Delta / weight norm | Naive merge: code flips | Naive merge: delta realized | Scale-only projection: cosine |
|---|---:|---:|---:|---:|
| ffn_gate | 0.21–0.42% | 0 | none | 0.0885 |
| ffn_up | 0.19–0.44% | 0 | none | 0.0884 |
| ffn_down | 0.11–0.25% | 0 | none | 0.0884 |

- **Naive merge.** Rounding the merged weights back to the existing ternary grid
  changes no code, so the merged model equals the base.
- **Scale-only projection.** The best per-block scale change reaches cosine 1/√128
  with the delta. That is the value expected when the delta has no relation to a
  block's single scale direction.

Scale-only tuning therefore cannot copy this adapter in weight space. It may
still find a different solution in function space, which only a GPU run can show.
QAT is the expected stage 2 path. The full per-tensor JSON stays with the
artifacts on the training host.

## Synthetic data (private)

The proprietary generators and data are in the private
`kortexa-ai/shingi-synthetic` Hugging Face dataset, at revision
`279e4ff851da820db3eabf89bcf204783ea302e5`. That revision contains generator
v1.0.0, code commit `38153558`, and seed 20260924.

**Counts.** The five families have exactly the planned train, development,
calibration and transfer counts. Long context has 300 training, 50 development
and 200 grid evaluation records.

**Checks.**
- The independent label checker agrees on 12,550 of 12,550 records.
- 20 property tests pass, and the QA build reports 0 violations.
- Exact-duplicate states: 0. Near duplicates (5-gram Jaccard at or above 0.95):
  0.0% in every family.
- No template exceeds 7.5% of a family's training split.
- Every option-only, state-only and question-only probe stays within 10 points of
  its baseline. Routing's option-overlap probe was first +7.6 points: queues
  missing from the rule table were never correct. After a fix it is +0.4.
- Hand spot audit: one random training record from each of the five families was
  checked by hand, and every gold label is correct.

**Known limits.**
- Wording comes from finite templates, so held-out subfamilies test new domains
  and resolution strategies rather than new writing styles.
- Policies use only conjunctive conditions and withhold at most one fact.
- Long-context filler is generic office text.

## Build validation

`scripts/prepare_data_v3.py` was run end to end twice.

**Run 1: training host, earlier artifacts, stand-in synthetic files.** This
run used the first mix of 30,000 records, 35% of them synthetic.
- Counts: 30,000 training records, of which 35.0% synthetic; a 10,000-record
  pilot; a locked test of 3,200; out-of-distribution 2,200; development and
  calibration 1,050 each; transfer 1,000; long-context splits 300/50/200.
- Every pair of splits has zero shared state hashes.
- 903 permitted earlier training records were reused.
- No earlier evaluation record entered training.

**Run 2: real synthetic data, without earlier artifacts.**
- Synthetic 13-gram contamination is 0 in every family, including long context.
  The check covers the locked test, out-of-distribution, development and
  calibration text, and all 8,122 external benchmark states (This/That,
  DecisionBench, JevBench).

**Remaining natural overlap.** After the prompt-group and 20% near-duplicate
exclusions, some natural training records still share at least one 13-gram with
evaluation text:

| Source | Records |
|---|---:|
| LEDGAR (contract boilerplate) | about 270 |
| HelpSteer (instruction templates) | about 150 |
| HelpSteer2 attributes | about 30–70 each |
| MMLU | about 15 |
| Other sources | 0–1 |

**Token share.** The first mix, 35% synthetic by records, came to about 43% of
training tokens by character estimate. On 2026-09-24 the target became 35% of
tokens, and every synthetic family was scaled by 0.70. A rebuild with the real
synthetic data measured:

| Profile | Records | Synthetic share of records | Synthetic share of tokens (estimate) | Total prompt tokens |
|---|---:|---:|---:|---:|
| Full | 26,850 | 27.4% | 34.9% | about 10.0M |
| Pilot | 8,950 | 27.4% | 34.9% | about 3.3M |

Tokenization on the training host checks the exact share and must find it
between 32% and 38%.

## Frozen build

Built on the training host at source commit `2bbb35a`, with synthetic revision
`279e4ff8`, into `~/data/datasets/shingi/v3`:

| Artifact | SHA-256 |
|---|---|
| Root manifest | `86849abf96ff17db1e8073b52e6e6ddca6cd1103fd583dc9170391f6f5068204` |
| Locked test | `b40bb5bb7b6e4b8d9e6cf77ecbffe85416db2f79f1681f0564b4a95a42c3daca` |
| Training records | `48a0a42a90dc13a99c3e7946a6d1fe74e737bb614ccbda80d60fa1d97854c675` |

| Split | Records |
|---|---:|
| Training | 25,600 (19,500 natural, 6,100 synthetic) |
| Pilot | 8,532 |
| Locked test | 3,200 |
| Out-of-distribution | 2,200 |
| Development | 1,050 |
| Calibration | 1,050 |
| Synthetic transfer | 1,000 |
| Long context | 300 training / 50 development / 200 evaluation |

- Every pair of splits has zero shared state hashes.
- 903 permitted earlier training records were reused.
- No earlier evaluation record entered training.

Exact native tokenization, with a 2,048-token limit and no truncation:

| Profile | Prompts | Excluded as too long | Natural tokens | Synthetic tokens | Synthetic share |
|---|---:|---:|---:|---:|---:|
| Pilot | 8,532 | 0 | 2,043,994 | 1,087,833 | 34.7% |
| Full | 25,597 | 3 | 6,110,342 | 3,247,408 | 34.7% |

The mix was sized twice. The first cut (7,350 synthetic records) estimated 34.9%
by characters, but measured 39.1% with the exact tokenizer, because synthetic
text packs 2.8–3.9 characters into a token against 4.25 for natural text. The
final sizes come from the exact token counts. The data is ready for the stage 1
canary and pilot.
