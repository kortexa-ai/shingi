# Stage 1 LoRA teacher against v0.2

Issue: https://github.com/kortexa-ai/shingi/issues/9. Plan:
[docs/data-v3.md](../../docs/data-v3.md). This report compares the stage 1
candidate (`full-01`, update 2048) with the released v0.2 adapter on matched
frozen data v3 splits and on the frozen external-v1 inventory. Accuracy counts
failures as incorrect; none occurred. Intervals are paired percentile bootstrap
95% intervals of candidate minus v0.2 accuracy, in points. Compact results are in
[summary.json](summary.json), [flips.json](flips.json) and
[calibration.json](calibration.json).

## Training

| Measure | Pilot (`pilot-01`) | Full (`full-01`) |
|---|---:|---:|
| Profile | v3-pilot | v3-full |
| Prepared prompts | 8,532 | 25,597 (3 excluded for length) |
| Planned / completed updates | 1,067 / 1,067 | 3,200 / 2,560 |
| Stop reason | finished | two development evaluations without improvement |
| Elapsed | 29,410 s (8.2 h) | 66,269 s (18.4 h) |
| Selected update | 1024 | 2048 |
| Selected development NLL / accuracy | 0.6666 / 74.38% | 0.6181 / 76.29% |

The recipe is rank 8, alpha 16, learning rate 5e-5, gradient accumulation 8 and a
2,048-token context. Selection used the lowest uncalibrated development NLL; test
and calibration data were not read during training. The selected safetensors
adapter SHA-256 is `753baa5a…66de2f`; its native GGUF is `d22a1075…8e1c38a4e3a`.
Full hashes are in summary.json.

Full-run development curve (1,050 records, uncalibrated):

| Update | Dev NLL | Dev accuracy |
|---:|---:|---:|
| 0 | 1.2961 | 56.57% |
| 256 | 0.7464 | 71.81% |
| 512 | 0.7397 | 73.43% |
| 768 | 0.7014 | 74.19% |
| 1024 | 0.6996 | 73.14% |
| 1280 | 0.6430 | 75.81% |
| 1536 | 0.6516 | 75.90% |
| 1792 | 0.6273 | 76.10% |
| **2048** | **0.6181** | **76.29%** |
| 2304 | 0.6357 | 76.00% |
| 2560 | 0.6345 | 76.48% |

## Matched frozen splits

The candidate ran with the bias fit (`full-2048-calibration.json`, below); v0.2
ran with its released calibration. The temperature-only column replays the
candidate's recorded logits with the identity calibration. Both calibrations use
choice/score temperature 1.0, so only yes/no answers change in the replay.

| Split | n | v0.2 | Candidate | Paired difference | Temperature-only replay |
|---|---:|---:|---:|---:|---:|
| Locked test | 3,200 | 65.50% | 70.91% | +5.4 [+4.2, +6.7] | +5.3 [+4.1, +6.6] |
| Out-of-distribution | 2,200 | 80.23% | 79.50% | −0.7 [−2.2, +0.7] | +0.0 [−1.3, +1.4] |
| Synthetic transfer | 1,000 | 62.90% | 85.80% | +22.9 [+19.8, +26.0] | +22.8 [+19.7, +25.9] |

| Split | v0.2 NLL | Candidate NLL (bias fit / temperature-only) | v0.2 ECE | Candidate ECE (bias fit / temperature-only) |
|---|---:|---:|---:|---:|
| Locked test | 1.0008 | 0.7930 / 0.7932 | 0.0350 | 0.0415 / 0.0404 |
| Out-of-distribution | 0.4940 | 0.4758 / 0.4629 | 0.0509 | 0.0251 / 0.0183 |
| Synthetic transfer | 0.9940 | 0.3502 / 0.3525 | 0.0289 | 0.0446 / 0.0466 |

Yes/no questions correct (v0.2 / bias fit / temperature-only): test 179 / 182 /
180 of 200; OOD 1,064 / 1,049 / 1,066 of 1,200; transfer 26 / 37 / 36 of 41.

### Per source

Correct of 200 per source. Differences are with the bias fit unless marked.

| Locked test source | v0.2 | Candidate | Paired difference |
|---|---:|---:|---:|
| banking77 | 153 | 155 | +1.0 [−2.5, +4.5] |
| civil_comments | 179 | 182 | +1.5 [−0.5, +4.0] |
| clinc150 | 174 | 174 | +0.0 [−3.5, +3.0] |
| commonsense_qa | 161 | 163 | +1.0 [−3.5, +5.5] |
| go_emotions | 56 | 108 | +26.0 [+18.5, +33.5] |
| helpsteer | 69 | 91 | +11.0 [+3.5, +18.5] |
| helpsteer2_coherence | 135 | 140 | +2.5 [+0.0, +5.5] |
| helpsteer2_complexity | 93 | 133 | +20.0 [+12.5, +27.5] |
| helpsteer2_correctness | 97 | 96 | −0.5 [−5.5, +4.5] |
| helpsteer2_helpfulness | 93 | 93 | +0.0 [−4.0, +4.0] |
| helpsteer2_verbosity | 107 | 126 | +9.5 [+3.0, +16.0] |
| ledgar | 137 | 157 | +10.0 [+5.5, +15.0] |
| massive | 159 | 161 | +1.0 [−2.0, +4.0] |
| measuring_hate_speech | 172 | 178 | +3.0 [+1.0, +5.5] |
| mmlu | 161 | 157 | −2.0 [−6.0, +2.0] |
| winogrande | 150 | 155 | +2.5 [−2.5, +8.0] |

Under the temperature-only replay only civil_comments changes: +0.5 [−1.5, +2.5].

| OOD source | v0.2 | Candidate | Bias fit | Temperature-only replay |
|---|---:|---:|---:|---:|
| arc_challenge | 186 | 190 | +2.0 [+0.5, +4.0] | +2.0 [+0.5, +4.0] |
| boolq | 187 | 183 | −2.0 [−5.5, +1.0] | +1.0 [−1.5, +3.5] |
| chaosnli | 121 | 133 | +6.0 [+1.5, +11.0] | +6.0 [+1.5, +11.0] |
| fever_evidence | 192 | 191 | −0.5 [−2.0, +1.0] | −0.5 [−2.5, +1.0] |
| mnli | 176 | 173 | −1.5 [−5.5, +2.5] | −1.5 [−5.5, +2.5] |
| paws | 161 | 167 | +3.0 [−1.5, +7.5] | +1.0 [−2.5, +4.5] |
| sms_spam | 196 | 195 | −0.5 [−3.5, +2.5] | +0.0 [−2.0, +2.0] |
| sst5 | 109 | 104 | −2.5 [−10.5, +6.0] | −2.5 [−10.5, +6.0] |
| strategyqa_closed | 143 | 127 | −8.0 [−15.0, −1.0] | −2.0 [−7.5, +3.5] |
| strategyqa_grounded | 185 | 186 | +0.5 [−3.5, +5.0] | +1.5 [−2.0, +5.5] |
| stsb | 109 | 100 | −4.5 [−10.5, +2.5] | −4.5 [−10.5, +2.5] |

| Transfer family | v0.2 | Candidate | Paired difference |
|---|---:|---:|---:|
| synth_policy | 105 | 170 | +32.5 [+24.5, +40.5] |
| synth_routing | 125 | 184 | +29.5 [+22.5, +37.0] |
| synth_rubric | 128 | 195 | +33.5 [+27.0, +40.0] |
| synth_state | 98 | 122 | +12.0 [+4.5, +20.0] |
| synth_taxonomy | 173 | 187 | +7.0 [+3.0, +11.5] |

## External suites

The candidate ran on the frozen external-v1 inventory with the temperature-only
parameters (identity). v0.2 figures are from the external-v1 run with its
released calibration. The bias-fit column replays the candidate's recorded
yes/no logits with the bias fit.

| Suite / subset | n | v0.2 | Candidate | Paired difference | Bias-fit replay |
|---|---:|---:|---:|---:|---:|
| DecisionBench medium | 293 | 233 (79.52%) | 241 (82.25%) | +2.7 [−1.4, +6.8] | +4.1 [−0.3, +8.5] |
| DecisionBench hard | 293 | 193 (65.87%) | 181 (61.77%) | −4.1 [−8.2, +0.0] | −4.8 [−9.6, −0.3] |
| JevBench public | 231 | 198 (85.71%) | 191 (82.68%) | −3.0 [−6.9, +0.9] | −1.7 [−5.2, +1.7] |
| JevBench easy | 48 | 48 | 48 | +0.0 | — |
| JevBench hard | 111 | 81 (72.97%) | 75 (67.57%) | −5.4 [−12.6, +0.9] | — |
| JevBench original | 72 | 69 (95.83%) | 68 (94.44%) | −1.4 [−8.3, +4.2] | — |
| This/That | 7,305 | 3,694 (50.57%) | 4,787 (65.53%) | +15.0 [+13.5, +16.4] | — |

Questions that share a DecisionBench scenario or a JevBench group are not
independent, so these intervals are narrower than a cluster bootstrap would give.

## Calibration

Both fits use hard-label NLL on the frozen data v3 calibration split only
(1,050 records: 977 choice/score, 73 yes/no), with canonical replay of the
candidate's uncalibrated evaluation and the standard temperature grid.

| Fit | Temperature | Yes/no temperature | Yes/no bias | Choice/score NLL before → after | Yes/no NLL before → after |
|---|---:|---:|---:|---:|---:|
| Bias fit (`full-2048-calibration.json`) | 1.0 | 1.25 | −0.5 | 0.6922 → 0.6922 | 0.3116 → 0.3027 |
| Temperature-only ([calibration.json](calibration.json)) | 1.0 | 1.0 | 0 (fixed) | 0.6922 → 0.6922 | 0.3116 → 0.3116 |

The temperature-only fit selects the identity. The external run used a
separately written identity file (`55e0acda…`) with the same parameters.

The 73 yes/no calibration records come from three sources:

| Source | Yes/no records | Gold yes |
|---|---:|---:|
| civil_comments | 50 | 6 |
| synth_state | 18 | 7 |
| synth_policy | 5 | 2 |
| **Total** | **73** | **15 (20.5%)** |

The negative bias fit follows this pool's 20.5% gold-yes share, which is set
mostly by civil_comments.

## Gate status

| Gate | Required | Bias fit | Temperature-only |
|---|---|---|---|
| Locked-test accuracy | ≥ +3.0, lower bound > 0 | pass, +5.4 [+4.2, +6.7] | pass, +5.3 [+4.1, +6.6] (replay) |
| Locked-test NLL | not worse | pass, 0.7930 vs 1.0008 | pass, 0.7932 (replay) |
| Locked-test ECE | not worse | fail, 0.0415 vs 0.0350 | fail, 0.0404 (replay) |
| Test sources entirely below 0 | reported | none | none (replay) |
| OOD aggregate | lower bound ≥ −1.5 | fail, −0.7 [−2.2, +0.7] | pass, +0.0 [−1.3, +1.4] (replay) |
| OOD sources entirely below 0 | reported | strategyqa_closed | none (replay) |
| DecisionBench medium | ≥ −2.0 | pass, +4.1 (replay) | pass, +2.7 |
| DecisionBench hard | ≥ −2.0 | fail, −4.8 (replay) | fail, −4.1 |
| JevBench public | ≥ −2.0 | pass, −1.7 (replay) | fail, −3.0 |

"Replay" marks results recomputed on the CPU from recorded logits with
`shingi.calibration.replay`, not new inference. Matched splits were run with
the bias fit, and the external suites with the temperature-only calibration.

## Flip analysis

Four-way split per item: both correct, v0.2 only, candidate only, both wrong.
Accuracy does not change under temperature-only calibration. Probabilities
below are as recorded: v0.2 with its released calibration (choice temperature
1.5, effective yes/no temperature 0.5) and the candidate with identity. Full
per-family and per-primitive tables are in flips.json.

| Scope | n | Both | v0.2 only | Candidate only | Both wrong |
|---|---:|---:|---:|---:|---:|
| DecisionBench medium | 293 | 218 | 15 | 23 | 37 |
| DecisionBench hard | 293 | 168 | 25 | 13 | 87 |
| JevBench easy | 48 | 48 | 0 | 0 | 0 |
| JevBench hard | 111 | 70 | 11 | 5 | 25 |
| JevBench original | 72 | 66 | 3 | 2 | 1 |
| This/That | 7,305 | 2,723 | 971 | 2,064 | 1,547 |

### By primitive

| Scope | Choice v0.2 → candidate | Yes/no | Score |
|---|---:|---:|---:|
| DecisionBench medium | 84 → 85 / 94 | 103 → 104 / 128 | 46 → 52 / 71 |
| DecisionBench hard | 58 → 55 / 94 | 94 → 87 / 128, −5.5 [−10.9, −0.8] | 41 → 39 / 71 |
| JevBench hard | 49 → 45 / 67 | 30 → 28 / 38 | 2 → 2 / 6 |
| JevBench original | 36 → 36 / 36 | 21 → 20 / 24 | 12 → 12 / 12 |

### Families with net changes

DecisionBench hard, net −12: losses spread over 14 families, the largest being
web_reading_qa (14 → 11 of 17), code_ci (4 → 2 of 5), customer_support
(15 → 13 of 19), feedback_survey_coding (5 → 3 of 7) and incident_containment
(3 → 1 of 4). Seven families gain one or two, led by document_enrichment
(5 → 7 of 12). No family interval excludes 0 on the loss side.

JevBench hard, net −6: judge_hard (13 → 11 of 17), temporal_numeric
(5 → 3 of 15), adversarial, ambiguous, long_policy and multi_hop (−1 each);
probability and tradeoff gain one each.

### Confidence of the flips

Mean (quartiles) probability. "Chosen" is the probability on the answer the
losing model chose.

| Scope | v0.2-only n | Candidate on its chosen | Candidate on gold | v0.2 on gold | Candidate-only n | v0.2 on its chosen | v0.2 on gold | Candidate on gold | Both correct: mean change on gold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DecisionBench medium | 15 | 0.61 (0.58/0.63/0.71) | 0.27 | 0.69 (0.49/0.80/0.89) | 23 | 0.56 (0.44/0.54/0.63) | 0.26 | 0.68 (0.55/0.73/0.83) | +0.054 |
| DecisionBench hard | 25 | 0.53 (0.43/0.55/0.64) | 0.29 | 0.57 (0.48/0.53/0.66) | 13 | 0.57 (0.47/0.50/0.67) | 0.28 | 0.62 (0.52/0.63/0.71) | −0.053 |
| JevBench hard | 11 | 0.56 (0.49/0.59/0.64) | 0.30 | 0.70 (0.51/0.78/0.91) | 5 | 0.68 (0.68/0.76/0.78) | 0.23 | 0.73 (0.62/0.67/0.75) | +0.021 |
| JevBench original | 3 | 0.72 | 0.28 | 0.83 | 2 | 0.59 | 0.41 | 0.72 | +0.007 |
| This/That | 971 | 0.64 (0.53/0.64/0.76) | 0.26 | 0.59 (0.48/0.56/0.66) | 2,064 | 0.56 (0.47/0.54/0.62) | 0.28 | 0.73 (0.60/0.75/0.87) | +0.185 |

### Option count and state length

Option count includes yes/no questions as two options. State length is in
characters of the string state, or of its canonical JSON when structured.

| Subset | Bucket | n | v0.2 | Candidate | Paired difference |
|---|---|---:|---:|---:|---:|
| DecisionBench hard | 2 options | 143 | 102 | 97 | −3.5 [−8.4, +1.4] |
| DecisionBench hard | 3–4 | 47 | 29 | 28 | −2.1 [−12.8, +8.5] |
| DecisionBench hard | 5–8 | 67 | 39 | 33 | −9.0 [−20.9, +1.5] |
| DecisionBench hard | 9–24 | 19 | 10 | 10 | +0.0 |
| DecisionBench hard | 25–52 | 4 | 2 | 2 | +0.0 |
| DecisionBench hard | >52 | 13 | 11 | 11 | +0.0 |
| DecisionBench hard | 500–2k chars | 10 | 8 | 8 | +0.0 |
| DecisionBench hard | 2k–8k chars | 213 | 134 | 126 | −3.8 [−8.9, +0.9] |
| DecisionBench hard | >8k chars | 70 | 51 | 47 | −5.7 [−12.9, +1.4] |
| DecisionBench medium | 5–8 options | 67 | 45 | 52 | +10.4 [+0.0, +20.9] |
| JevBench hard | 2 options | 38 | 30 | 28 | −5.3 [−18.4, +7.9] |
| JevBench hard | 3–4 | 52 | 37 | 36 | −1.9 [−11.5, +7.7] |
| JevBench hard | 5–8 | 21 | 14 | 11 | −14.3 [−28.6, +0.0] |
| JevBench hard | <500 chars | 6 | 3 | 2 | −16.7 |
| JevBench hard | 500–2k chars | 56 | 45 | 43 | −3.6 [−10.7, +3.6] |
| JevBench hard | 2k–8k chars | 13 | 7 | 6 | −7.7 [−30.8, +15.4] |
| JevBench hard | >8k chars | 36 | 26 | 24 | −5.6 [−22.2, +11.1] |

### Group consistency

Share of groups with at least two questions where every question is correct.

| Groups | Count | v0.2 all correct | Candidate all correct | Both | v0.2 only | Candidate only |
|---|---:|---:|---:|---:|---:|---:|
| JevBench paraphrase groups (all in original) | 36 | 33 (91.7%) | 33 (91.7%) | 31 | 2 | 2 |
| DecisionBench medium scenarios | 66 | 27 (40.9%) | 31 (47.0%) | 24 | 3 | 7 |
| DecisionBench hard scenarios | 66 | 9 (13.6%) | 8 (12.1%) | 6 | 3 | 2 |

JevBench hard and easy have no group with two or more questions. JevBench group
consistency is unchanged.

### This/That families

| Family | n | v0.2 | Candidate | Paired difference |
|---|---:|---:|---:|---:|
| blocked_pair | 500 | 58.8% | 69.2% | +10.4 [+4.6, +16.2] |
| cell_kind | 498 | 33.5% | 43.8% | +10.2 [+4.4, +16.1] |
| distance_band | 500 | 32.2% | 37.0% | +4.8 [−0.8, +10.6] |
| first_move | 500 | 34.2% | 38.6% | +4.4 [+0.0, +8.8] |
| goal_bearing | 496 | 86.3% | 99.2% | +12.9 [+9.9, +15.9] |
| move_legal | 500 | 76.0% | 84.4% | +8.4 [+4.8, +12.2] |
| move_two_step | 500 | 66.2% | 80.0% | +13.8 [+8.8, +18.6] |
| only_way | 415 | 23.4% | 65.8% | +42.4 [+35.7, +48.9] |
| onward | 500 | 59.4% | 81.0% | +21.6 [+16.4, +26.8] |
| open_count | 400 | 36.5% | 33.5% | −3.0 [−9.0, +3.0] |
| plan_survives | 500 | 30.6% | 57.2% | +26.6 [+21.4, +31.8] |
| reachable_within | 500 | 51.2% | 50.8% | −0.4 [−8.8, +8.0] |
| snake_food | 496 | 56.2% | 73.0% | +16.7 [+11.5, +22.0] |
| snake_safe | 500 | 54.2% | 89.0% | +34.8 [+30.4, +39.4] |
| stochastic | 500 | 52.6% | 74.4% | +21.8 [+16.0, +27.4] |

11 of 15 families improve by more than 5 points; none regresses by more than 5.

### Option order

100 preselected Choice pairs per suite. The released canonical order sorts
option keys; the raw-order diagnostic bypasses the sort.

| Suite | Canonical flips v0.2 / candidate | Raw-order flips v0.2 / candidate | Raw-order mean TVD v0.2 / candidate | Raw-order max TVD v0.2 / candidate |
|---|---:|---:|---:|---:|
| DecisionBench | 0 / 0 | 18 / 10 | 0.154 / 0.107 | 0.685 / 0.968 |
| JevBench | 0 / 0 | 9 / 6 | 0.072 / 0.040 | 0.521 / 0.258 |
| This/That | 0 / 0 | 28 / 19 | 0.135 / 0.113 | 0.594 / 0.513 |

## Limitations

- The temperature-only calibration policy was chosen after the OOD results of
  the bias fit were seen. Its gate results are therefore not a clean held-out
  test of that policy.
- The external subsets have 231–293 questions. Their intervals are wide and
  treat questions as independent, although questions share scenarios and groups.
- This/That is not fully out-of-distribution for its spatial families after
  training on the synthetic state family.
- Synthetic transfer uses held-out subfamilies from the same generator style as
  the training data, so its gain overstates transfer to other sources.
- The flip probabilities compare two different calibrations. v0.2's released
  calibration flattens choices and sharpens yes/no answers, so probability
  differences are not all model differences.
- Family, bucket and primitive tables have small cells and many comparisons. No
  multiplicity correction is applied.
- Temperature-only results on the matched splits and bias-fit results on the
  external suites are CPU replays of recorded logits, not separate runs.

## Interpretation

**What the training clearly did.** It taught the targeted tasks. The locked test
gains are large where the data was added (GoEmotions +26, HelpSteer2 complexity
+20, HelpSteer +11, LEDGAR +10) and no source regresses. The synthetic state
family transferred to a different benchmark: This/That gains 15 points, with the
largest gains in formats the generator never produced (snake, stochastic, plan
survival). Raw option-order flips fall on every external suite, so the one
random order per record made the model more order-stable. The candidate's raw
probabilities need no calibration: the fitted temperatures are 1.0.

**What it did not do.** It did not make Shingi a better general judge. The
out-of-distribution sources are flat, and the two judgment suites move against
the candidate: DecisionBench hard −4.1 points and public JevBench −3.0. Both
intervals reach zero, so neither is a proven regression, but the item-level
flips show two systematic shifts rather than pure noise:

- **Lenient pass/fail judging.** Four of the fourteen JevBench losses are
  "does the response fully and correctly satisfy the request" items where the
  candidate now says yes to responses with substantive errors; v0.2 was confident
  on three of them. One gain is the same item type where leniency happened to be
  right. Two DecisionBench-hard yes/no losses judge an agent's or a flow's
  handling as adequate when it was not. The plausible cause is the 4,000
  HelpSteer records, where a response with a small error still scores 3 of 4;
  strict any-error judging is a different skill and no training source teaches
  it.
- **Conservative ordinal scores.** On DecisionBench, nine of the ten hard-subset
  ordinal losses are under-scores, and seven of the eight ordinal gains are also
  lower predictions than v0.2 that happened to be right. The same downward shift
  produces most of the medium-subset gain (+6 on score items) and most of the
  hard-subset loss. On the locked test the candidate's HelpSteer scores are
  centred (expected score minus gold within ±0.07), so the shift appears on
  holistic severity, urgency and fault judgments, not on rubric-like quality
  scales. The plausible cause is the rubric family, which teaches "the highest
  level whose requirements are all met": a conservative rule that does not fit
  holistic scales.

Two smaller patterns: numeric and temporal items lose three on JevBench and
gain none, plus two procedural filtering items on DecisionBench hard; and long
multi-hop documents churn both ways (five losses, three gains, including one
0.16 → 0.99 correction), which reads as changed behaviour with no net direction.
One DecisionBench loss is a Spanish-language item; training is English only.

**Calibration.** The bias fit is not trustworthy: it saw 73 yes/no records,
50 of them Civil Comments with a 12% yes-rate, and produced a −0.5 bias that
helped JevBench (where the candidate over-answers yes) and hurt StrategyQA
(where it under-answers yes). No global bias can serve request-defined decisions
with different base rates, so temperature-only is the right policy, and the
calibration split should balance yes/no sources and gold rates in the next data
version. Under temperature-only calibration the out-of-distribution gate passes
and no source regresses. The ECE miss (0.040 against 0.035) is small for a
ten-bin statistic on 3,200 items, and NLL, Brier and out-of-distribution ECE
all improve.

**Assessment.** The adapter is a clear improvement on the trained task families
and on spatial state reasoning, and a wash to slightly negative on general
judgment. It is not the teacher to distil into a ternary model yet: distillation
would preserve the leniency and the conservative scoring. The data, not the
recipe, is the limiting factor. The next data version should add a strict
pass/fail judging family with planted errors and computed labels, diversify the
rubric family beyond cumulative criteria toward holistic severity and urgency
scales, add numeric and date arithmetic under stated rules, and reconsider the
HelpSteer share. Stage 2 method work (scale-only tuning, then QAT) can proceed
on this teacher to measure how much of a LoRA a ternary file can hold; that
measurement does not depend on which teacher is final.

## Reproduction

```bash
uv run --locked --extra data python scripts/compare_stage1.py --data ~/data/datasets/shingi/v3 \
  --eval artifacts/data-v3-stage1/eval --output artifacts/data-v3-stage1/eval/summary.json
uv run --locked --extra data python scripts/fit_calibration.py --data ~/data/datasets/shingi/v3 \
  --split calibration --run artifacts/data-v3-stage1/eval/full-2048-calibration-raw \
  --output results/stage1/calibration.json --temperature-only
uv run --locked --extra data python scripts/flip_analysis.py \
  --records artifacts/external-v1/data-01/records.jsonl --old artifacts/external-v1/run-01 \
  --new artifacts/data-v3-stage1/external-full-2048 --old-name v0.2 --new-name stage1 \
  --output results/stage1/flips.json --dump-dir artifacts/data-v3-stage1/flips
```

summary.json copies `artifacts/data-v3-stage1/eval/summary.json` and adds the
external, training, calibration and replay sections. The flip dumps under
`artifacts/data-v3-stage1/flips/` hold full evaluation text for private review
and are not redistributed.
