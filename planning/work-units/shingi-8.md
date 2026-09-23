# Initial public v0.2 release

Owning issue: https://github.com/kortexa-ai/shingi/issues/8

Publish the already tested Apache-2.0 adapter with its frozen calibration,
original release evaluation and subsequent external benchmark evidence.
Keep measured regressions and limits explicit in the model card. No training,
calibration changes, new GPU measurements or service changes belong to this unit.

Bundle assembly checks external evidence against the weight/calibration identity
and audit hashes. Include the supplied Shingi illustration. Keep source code on
GitHub without duplicating it in the model bundle. Verify the exact package before
upload, capture the immutable Hub revision, compare its complete inventory and
checksums, and download without authentication. Preserve the original bundle.

Publication receipts, immutable revisions and validation outcomes belong in the
owning issue. Tag the matching source commit; keep model weights outside Git.
