# Program v0.3: expanded data and a self-contained ternary model

Owning issue: https://github.com/kortexa-ai/shingi/issues/9
Data preparation: https://github.com/kortexa-ai/shingi/issues/10

The plan is in [docs/data-v3.md](../../docs/data-v3.md). Stage 1 results are in
[results/stage1/REPORT.md](../../results/stage1/REPORT.md).

## Decisions

- **GPU authority.** Franci hands over the RTX PRO 6000 for a block; that
  includes stopping and restoring its six production services. The 4090 is never
  touched. Every block records the initial service set and restores exactly it.
- **Model path.** Scripts pin the base to snapshot `6ed5e12`. A second Hugging
  Face snapshot of the same blob appeared on 2026-09-24 and broke a path glob.
- **Pilot gate.** Passed on 2026-09-25: the pilot reached 74.6% on the v3
  development set against 67.4% for v0.2 (matched native evaluation). The
  "still improving" condition was borderline and accepted.
- **Checkpoint selection.** Minimum development NLL over all 1,050 records.
  For the full run this chose update 2,048, which is also the minimum over the
  800 natural records alone, so selection was not pulled by synthetic data.
- **Calibration policy.** Temperature only; the yes/no bias stays 0. The bias fit
  used a 73-record yes/no pool (50 Civil Comments, 20% gold-yes) and produced
  −0.5, which helped JevBench yes/no items and hurt StrategyQA. A global prior
  shift is wrong for request-defined decisions whose base rates differ. The
  temperature-only fit selects the identity. This policy was chosen after the
  out-of-distribution result was seen; the report discloses that.
- **ECE gate.** Franci accepted the locked-test ECE miss (0.040 against 0.035)
  on 2026-09-26; accuracy is the program's priority, and NLL, Brier and
  out-of-distribution ECE improve.
- **Parallel work.** Data v3.1 (#11, CPU) and stage 2 method work on the stage 1
  teacher (GPU) proceed in parallel. The stage 2 measurement is method
  de-risking: it does not depend on which teacher is final.
- **Stage 2 path.** The merge check shows a naive merge loses the whole adapter
  and a scale-only projection reaches cosine 0.088; QAT is the expected path.
  The published F16 Bonsai file is dequantized ternary and is not used.

- **FP16 scale resolution.** PQ2_0 scales are FP16 (about 0.05–0.1% relative
  steps), so scale-only training rounds the effective scale to FP16 in the
  forward pass with a straight-through gradient; the trained model then equals
  the exported file. Canary block 01 found this: one step at lr 2e-4 changed
  190M factors in FP32 and zero scale bytes on export. Since each stored weight
  is code × scale with code in {−1, 0, 1}, fp16(w × f) = code × fp16(scale × f).
- **Data v3.1 evaluation.** v3.1 draws a fresh locked test and OOD split; v0.2,
  stage 1 and v3.1 are all evaluated on them. Old synthetic families are
  byte-identical to v1, so their evaluation records equal v3's.

## Validation facts

- Full run: 25,597 prompts, stopped at update 2,560 after two checks without
  improvement; selected update 2,048 (native adapter SHA-256 `d22a1075…8a4e3a`).
- Matched locked test: +5.3 points [+4.1, +6.6] under temperature-only
  calibration; no source regresses. Out-of-distribution: +0.0 [−1.3, +1.4].
  Held-out synthetic families: +22.9.
- External: This/That +15.0 [+13.5, +16.4]; DecisionBench medium +2.7
  [−1.4, +6.8]; DecisionBench hard −4.1 [−8.2, 0.0]; public JevBench −3.0
  [−6.9, +0.9]. Raw input-order flips fall on every suite (18→10, 9→6, 28→19
  of 100 pairs).
- Item-level flips: four of fourteen JevBench losses are strict pass/fail
  judging items answered leniently; nine of ten DecisionBench-hard ordinal
  losses are under-scores; long-document items churn both ways; numeric and
  temporal items lose three, gain none.
- Gates not met as measured: DecisionBench hard, public JevBench, and
  locked-test ECE (0.040 against 0.035, with better NLL and Brier).
- Stage 2 canary block 01: 401 PQ2_0 tensors and 199,987,200 factors wrapped;
  1,186 sampled rows match stored bytes; unit-factor parity RMSE 8.8e-7; one
  2,048-token step 13.8 s; peak reserved 58.1 GiB with 36.1 GiB free.
- Stage 2 teacher: full-2048 adapter, 8,532 pilot prompts; teacher on v3
  development 76.3%, NLL 0.617.
- Stage 2 scale-only run (block 02, pilot profile, FP16 straight-through
  forward): 857 updates in 41,618 s, stopped at the 39,600 s training cap with
  development NLL still falling (0.763 → 0.708 → 0.684 → 0.678). Export
  `shingi-ternary.gguf`, 7,206,168,928 bytes, SHA-256 `44e7e634…5914e`.
- Ternary file on the locked test: 69.8% / NLL 0.829 / ECE 0.030; teacher 70.8%
  / 0.793 / 0.040; v0.2 65.5% / 1.001 / 0.035. Versus teacher −1.03 points
  [−1.97, −0.09]; versus v0.2 +4.3 [+3.1, +5.5] with no source regression. The
  native development score (73.4% / 0.678) equals the training-time score of the
  exported checkpoint. Stage 2 gates: interval and NLL pass; the 1.0-point
  difference misses by 0.03.

## Incidents

- After block 01 the first evaluation launch failed on the snapshot glob and went
  unnoticed for about 3.5 hours; the claim lease lapsed.
- The candidate external run was reported finished while its order diagnostic
  was still running; the GPU claim was released early and re-acquired.
