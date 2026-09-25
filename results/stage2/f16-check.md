# Published F16 Bonsai versus the PQ2_0 base

Issue: https://github.com/kortexa-ai/shingi/issues/9

**Question:** could `Ternary-Bonsai-2-27B-F16.gguf` (53.8 GB) give stage 2 QAT
continuous weights from before ternarization to start from? It comes from
`prism-ml/Ternary-Bonsai-2-27B-gguf`, revision `6ed5e12`, the same revision as
the pinned base.

**Method:** `scripts/inspect_f16.py` read the F16 GGUF header and 64 rows each of
four tensors through HTTP range requests, then compared them with the decoded
PQ2_0 blocks of the pinned base (SHA-256 `3907dc16…`).

| Tensor | Blocks | Distinct values per 128-block | Code agreement | Residual to scale × code |
|---|---:|---:|---:|---:|
| `blk.0.ffn_gate.weight` | 2,560 | 3 | 100% | 0 |
| `blk.3.attn_q.weight` | 2,560 | 3 | 100% | 0 |
| `blk.31.ffn_down.weight` | 8,704 | 3 | 100% | 0 |
| `blk.62.ffn_up.weight` | 2,560 | 3 | 100% | 0 |

The F16 file has the same `prism.hadamard.*` folding metadata as the PQ2_0 base,
so the two files were compared in the same stored domain.

**Result:** the F16 file is the ternary model dequantized to F16. It contains no
information from before ternarization, so it offers QAT nothing that decoding
PQ2_0 does not. Stage 2 QAT initializes its latent weights from the decoded
ternary values, and this repository does not use the F16 file. Truly continuous
starting weights would need a separate release from the model publisher.
