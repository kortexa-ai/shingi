"""Compare the published Bonsai F16 GGUF with the pinned PQ2_0 base, without a full download.

Reads the remote F16 header and a few rows of chosen tensors through HTTP range
requests, maps them into the stored PQ2 domain (signed Hadamard for folded
tensors when the F16 file is not itself folded), and reports whether the F16
values are dequantized ternary blocks or continuous weights that round to the
stored codes. The answer decides whether QAT can start from them.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import urllib.request

import numpy as np

from merge_check import fwht_rows
from shingi.pq2_scales import BLOCK, BLOCK_BYTES, split

REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
REVISION = "6ed5e12bf84b7a63069882c91dd9e9218647d17b"
F16_NAME = "Ternary-Bonsai-2-27B-F16.gguf"
SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Remote:
    def __init__(self, url):
        self.url = url

    def read(self, start, size):
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + size - 1}",
                                                            "User-Agent": "shingi-f16-check"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        if len(data) != size:
            raise IOError(f"short range read at {start}: {len(data)} of {size}")
        return data


class Cursor:
    def __init__(self, remote, chunk=16 << 20):
        self.remote, self.chunk, self.base, self.buffer, self.pos = remote, chunk, 0, b"", 0

    def take(self, size):
        while self.pos + size > self.base + len(self.buffer):
            self.buffer += self.remote.read(self.base + len(self.buffer), self.chunk)
        out = self.buffer[self.pos - self.base:self.pos - self.base + size]
        self.pos += size
        return out

    def unpack(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.unpack("<Q")).decode("utf-8", "replace")

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind == 9:
            item, count = self.unpack("<I"), self.unpack("<Q")
            values = [self.value(item) for _ in range(count)]
            return values if count <= 64 or item != 8 else f"<{count} strings>"
        return self.unpack(SCALARS[kind])


def header(remote):
    """Metadata, tensor table and absolute data start of a remote GGUF."""
    c = Cursor(remote)
    if c.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version, tensors, kvs = c.unpack("<I"), c.unpack("<Q"), c.unpack("<Q")
    meta = {}
    for _ in range(kvs):
        key = c.string()
        meta[key] = c.value(c.unpack("<I"))
    table = {}
    for _ in range(tensors):
        name = c.string()
        dims = [c.unpack("<Q") for _ in range(c.unpack("<I"))]
        table[name] = {"dims": dims, "type": c.unpack("<I"), "offset": c.unpack("<Q")}
    alignment = int(meta.get("general.alignment", 32))
    return version, meta, table, -(-c.pos // alignment) * alignment


def compare(f16_rows, scales, codes):
    """Relationship between F16 rows and stored ternary blocks with their scales."""
    rows, nb, _ = codes.shape
    w = f16_rows.reshape(rows, nb, BLOCK).astype(np.float64)
    s = scales[..., None].astype(np.float64)
    q = s * codes
    distinct = [len(np.unique(np.round(block, 6))) for block in w.reshape(-1, BLOCK)]
    rounded = np.clip(np.rint(np.divide(w, s, out=np.zeros_like(w), where=s != 0)), -1, 1)
    return {"blocks": rows * nb,
            "median_distinct_values_per_block": float(np.median(distinct)),
            "max_distinct_values_per_block": int(np.max(distinct)),
            "code_agreement_after_rounding": float((rounded == codes).mean()),
            "relative_residual_to_ternary": float(np.linalg.norm(w - q) / max(np.linalg.norm(w), 1e-300)),
            "cosine_with_ternary": float((w * q).sum() / max(np.sqrt((w * w).sum() * (q * q).sum()), 1e-300)),
            "exactly_dequantized": bool(np.allclose(w, q, rtol=0, atol=1e-3 * float(np.abs(q).max())))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="pinned PQ2_0 base")
    p.add_argument("--prism", type=Path, required=True)
    p.add_argument("--tensors", nargs="+", default=["blk.0.ffn_gate.weight", "blk.31.ffn_down.weight",
                                                    "blk.3.attn_q.weight", "blk.62.ffn_up.weight"])
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.prism / "gguf-py"))
    from gguf import GGUFReader
    base = GGUFReader(str(a.model))
    base_fields = {k: v.contents() for k, v in base.fields.items() if k.startswith("prism.")}
    folded = set(base_fields["prism.hadamard.weight_names"])
    block, signs, offset = int(base_fields["prism.hadamard.block_size"]), {}, 0
    for width in base_fields["prism.hadamard.sign_widths"]:
        signs[width] = np.asarray(base_fields["prism.hadamard.sign_values"][offset:offset + width], dtype=np.float64)
        offset += width
    remote = Remote(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{F16_NAME}")
    version, meta, table, data_start = header(remote)
    f16_folded = any(k.startswith("prism.hadamard") for k in meta)
    types = sorted({t["type"] for t in table.values()})
    base_tensors = {t.name: t for t in base.tensors}
    result = {"f16_repository": REPO, "revision": REVISION, "gguf_version": version, "tensor_types": types,
              "f16_metadata_prism_keys": sorted(k for k in meta if k.startswith("prism.")),
              "f16_hadamard_folded": f16_folded, "general_file_type": meta.get("general.file_type"),
              "tensors": {}}
    for name in a.tensors:
        info, tensor = table[name], base_tensors[name]
        width, rows = info["dims"][0], info["dims"][1]
        if info["type"] != 1 or tensor.tensor_type.name != "PQ2_0":
            raise ValueError(f"{name}: expected F16 against PQ2_0")
        n = min(a.rows, rows)
        raw = remote.read(data_start + info["offset"], n * width * 2)
        f16 = np.frombuffer(raw, dtype="<f2").astype(np.float64).reshape(n, width)
        scales, codes = split(np.asarray(tensor.data).reshape(-1)[:n * width // BLOCK * BLOCK_BYTES], n, width)
        direct = compare(f16, scales, codes)
        entry = {"rows_compared": n, "width": width, "stored_folded": name in folded, "as_stored": direct}
        if name in folded and not f16_folded:
            entry["after_hadamard"] = compare(fwht_rows(f16, signs[width], block), scales, codes)
        result["tensors"][name] = entry
        print(json.dumps({name: entry}), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
