"""CPU check: how much of a trained LoRA survives a ternary PQ2_0 merge.

For every adapted MLP tensor, map the LoRA delta into the stored weight domain
(native LoRA reads the original activation; Hadamard-folded weights read the
transformed one), then measure the naive re-ternarized merge and the best
scale-only projection. Weight-space agreement is a screening measure only;
decision quality still requires native evaluation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from shingi.pq2_scales import merge_statistics, split
from shingi.ternary_training import BASE_SHA256


def fwht_rows(a, signs, block):
    """Signed normalized Hadamard of each row, matching ternary_training.hadamard."""
    y = (a * signs).reshape(-1, block).astype(np.float64)
    stride = 1
    while stride < block:
        z = y.reshape(-1, block // (2 * stride), 2, stride)
        y = np.stack((z[:, :, 0] + z[:, :, 1], z[:, :, 0] - z[:, :, 1]), axis=2).reshape(-1, block)
        stride *= 2
    return (y / np.sqrt(block)).reshape(a.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True, help="native GGUF LoRA adapter")
    parser.add_argument("--prism", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("preserve existing output")
    with args.model.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != BASE_SHA256:
            raise SystemExit("base checksum mismatch")
    sys.path.insert(0, str(args.prism / "gguf-py"))
    from gguf import GGUFReader
    base = GGUFReader(str(args.model))
    fields = {k: v.contents() for k, v in base.fields.items() if k.startswith("prism.")}
    folded = set(fields["prism.hadamard.weight_names"])
    block = int(fields["prism.hadamard.block_size"])
    signs, offset = {}, 0
    for width in fields["prism.hadamard.sign_widths"]:
        signs[width] = np.asarray(fields["prism.hadamard.sign_values"][offset:offset + width], dtype=np.float64)
        offset += width
    adapter = GGUFReader(str(args.adapter))
    alpha = float(adapter.fields["adapter.lora.alpha"].contents())
    lora = {t.name: np.asarray(t.data, dtype=np.float32).reshape([int(n) for n in t.shape[::-1]]) for t in adapter.tensors}
    tensors = {t.name: t for t in base.tensors}
    results, totals = {}, defaultdict(list)
    for name in sorted({n.rsplit(".", 1)[0] for n in lora}):
        a, b = lora[name + ".lora_a"], lora[name + ".lora_b"]
        tensor = tensors[name]
        width, rows = (int(n) for n in tensor.shape[:2])
        scales, codes = split(tensor.data, rows, width)
        a_stored = fwht_rows(a.astype(np.float64), signs[width], block) if name in folded else a.astype(np.float64)
        delta = (alpha / a.shape[0]) * (b.astype(np.float64) @ a_stored)
        if delta.shape != (rows, width):
            raise ValueError(f"{name}: delta {delta.shape} != {(rows, width)}")
        results[name] = merge_statistics(scales, codes, delta)
        results[name]["hadamard_folded"] = name in folded
        kind = name.split(".")[2]
        for key in ("delta_to_weight_norm", "code_flip_fraction"):
            totals[kind + "/" + key].append(results[name][key])
        for method in ("naive", "scale_only"):
            totals[kind + "/" + method + "_cosine"].append(results[name][method]["cosine"])
            totals[kind + "/" + method + "_relative_error"].append(results[name][method]["relative_error"])
        print(json.dumps({name: results[name]}), flush=True)
    summary = {key: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for key, v in sorted(totals.items())}
    with args.adapter.open("rb") as stream:
        adapter_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"base_sha256": BASE_SHA256, "adapter_sha256": adapter_sha, "alpha": alpha,
                                       "tensors": len(results), "summary": summary, "per_tensor": results}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
