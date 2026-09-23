# Apache-targeted retraining protocol

Owning issue: https://github.com/kortexa-ai/shingi/issues/6

Use a fresh LoRA on the unchanged Bonsai base. The `apache-v2` preparation
profile admits only the six sources recorded in
`results/training-v2/provenance.json`. Earlier permitted training examples
may recur; all published validation/test states and prior project evaluation
records remain excluded from training. New development and calibration data
exclude every earlier project split. Preserve all previous weights and reports.

The fitting path uses the fixed canonical Choice readout, two deterministic
training permutations, rank 8, alpha 16, batch one, accumulation eight, 1024-token
training context, and at most 640 updates or 21600 seconds. Select minimum
uncalibrated development NLL including the base. Stop after two development
checks without improvement. Require NLL improvement and no more than a
2-percentage-point development accuracy loss before evaluating a candidate.

Freeze new weights and calibration before fresh test inference. Compare the
base, the frozen previous adapter, and the new adapter on identical fresh
records. The prior adapter is an evaluation-only comparator. Report regressions
against it explicitly. Test results must not select or tune the new adapter.
The issue records numerical, quality, memory, runtime and restoration gates.

Bulk artifacts belong in the ignored `artifacts/apache-v2` directory on the
operator's research machine. Tracked source moves through Git. GPU operations
require a current exact resource claim, a live baseline, an independent
restoration wrapper, and final health checks. Distribution follows measured
quality and an explicit Apache-2.0 weight license with source attribution.
