# Shingi agent instructions

Follow the parent workspace instructions in `../AGENTS.md` when present, including canonical user guidance, issue tracking, work claims, and repository workflow.

## Research scope

Shingi investigates general-purpose decision models, initially using Bonsai 2 27B. Keep choices and domains request-defined. Record the current experiment's goal, acceptance thresholds, and resource budget in its owning issue before running it. The bootstrap is not an instruction to start training or stop services.

Keep training, calibration, development, and locked evaluation data distinct. Record dataset provenance, overlap checks, model revisions, prompt templates, candidate token IDs, and decoding settings with results. Retrieve every candidate's logit explicitly; a truncated top-k response is not sufficient. Preserve original scales and transforms when establishing conversion parity. Do not claim quality, numerical equivalence, or speedups without measurements.

## Machines and artifacts

Before GPU work, read `../legolm/SMARTY_6000_GUIDE.md` on Smarty. Pin each experiment to its authorized GPU UUID. The RTX PRO 6000 is the normal research GPU; use the RTX 4090 only with Franci's explicit authorization for that block. The initial investigation received that exception while Franci uses the 6000. Follow the documented service-control process, record the initially running services, and restore that set when the authorized run ends. The 4090 profile requires 14 GiB free before load, at least 4 GiB headroom, and at most 16K context; the 6000 retains its 30/10 GiB gates. Never stop or restore another experiment's 6000 services from a 4090 block.

Use `/home/francip/src/shingi/artifacts` on Smarty for authoritative project checkpoints and bulk artifacts. This project uses its own artifact paths, not LegoLM's. Record revisions and checksums, verify copies before cleanup, and preserve checkpoints needed to reproduce reported results. Never commit weights or use Git LFS for them.

Synchronize source through Git. Use an isolated `.venv` and `uv` for Python work once implementation begins. Keep credentials, local environments, downloaded datasets, and bulk run output out of Git; commit concise, reproducible research records separately.

## Release discipline

Validate decision quality before preparing distribution formats. Preserve the upstream model's license and notices, and check dataset and comparator permissions separately. A working inference wrapper alone is not a newly trained checkpoint. Public model cards must identify the actual changes, limitations, evaluation split, and measured results. Published Jev comparisons must use attributed third-party public information; do not benchmark hosted Jev. Follow `docs/evaluation-protocol.md` and distinguish reported results from our reanalysis of public predictions.
