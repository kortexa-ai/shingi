# Shingi repository bootstrap

Owning issue: https://github.com/kortexa-ai/shingi/issues/1

## Purpose

Establish a private research repository after the selection of the Shingi name. The intended project is a general-purpose decision model based initially on Bonsai 2 27B, with quality validation preceding training and release-format decisions.

## Decisions

- Repository: `kortexa-ai/shingi`, private, default branch `main`.
- Code license: Apache-2.0; third-party artifact licenses remain separate.
- Source checkouts: `/Users/francip/src/shingi` on Snappy and `/home/francip/src/shingi` on Smarty.
- Authoritative project artifacts: `/home/francip/src/shingi/artifacts`, excluded from Git.
- This work establishes documentation and exclusions only. Experiment goals, acceptance thresholds, implementation, GPU jobs, service changes, weight downloads, and model publication are outside the bootstrap.

## Validation

Review the five-file tracked inventory and run `git diff --check`. Use `git check-ignore` to verify representative checkpoint, secret, environment, dataset-cache, and run-output paths are excluded. Confirm that README and this planning note remain trackable. After the initial push and Smarty clone, verify both worktrees are clean and their HEAD matches `origin/main`.

The owning issue records the resulting commit and verification evidence. No runtime tests are required for this documentation-only scaffold.
