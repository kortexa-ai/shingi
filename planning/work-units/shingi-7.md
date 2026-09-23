# Frozen external evaluation

Owning issue: https://github.com/kortexa-ai/shingi/issues/7

Evaluate Shingi v0.2 without changing weights, calibration, prompts or native
readout. The inventory is This/That (7,305), DecisionBench Medium and Hard
(293 each), and JevBench public original/easy/hard (72/48/111).

Preparation pins upstream revisions, hashes every download, maps only state and
question into model inputs, checks prior Shingi state overlap and freezes order
diagnostics. The runner checks release identities and the prepared manifest
before it loads the model. Its output contains every prediction and candidate
logit, input/prompt hashes, errors, latency and sampled memory.

The acceptance test is complete, reproducible measurement, with no accuracy
threshold. Do not use this evaluation to select a new configuration. Compare
Jev-Omni only through attributed public results. Its full training mixture and
exact evaluation recipe are unavailable. Do not claim an official JevBench
leaderboard score from the public subset.
