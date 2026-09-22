# CUDA v1 release preparation

Owning issue: https://github.com/kortexa-ai/shingi/issues/5

Prepare a portable CUDA release using the frozen update-128 Bonsai decision
adapter. Publication and repository visibility are the final, separate step.

## Delivery sequence

1. Remove machine and service dependencies from the public build, runtime and
   research scripts. Add checksum-bound adapter serving and regression tests.
2. Build the pinned public Prism runtime in an independent checkout. Freeze a
   fresh evaluation slice excluding every prior experiment split, and retain
   the existing base and adapter calibration without test-driven fitting.
3. Validate the API, accuracy, option order, context, latency and memory on both
   CUDA targets. Preserve raw evidence and process-owned service restoration.
4. Produce reproducible charts, the model card, third-party notices, artifact
   manifests, installation instructions and the publication audit.

Acceptance thresholds and resource budgets are declared in the issue before
GPU execution. Any changed model or decision method requires a new development
and locked evaluation plan. Historical reports remain evidence of their exact
revisions and are not silently relabeled as release measurements.

Code and compact evidence move through Git. Checkpoints, raw datasets and
machine-specific service wrappers stay outside the public distribution.
