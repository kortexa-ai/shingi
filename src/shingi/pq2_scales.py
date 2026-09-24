"""PQ2_0 block scales: analysis, trainable scaling and byte-exact export.

A PQ2_0 row stores, per 128 weights, one FP16 scale followed by 32 bytes of
two-bit ternary codes. Stage 2 keeps every code and changes only scales, so the
exported file is still a plain PQ2_0 model with the base's size and layout.
"""
from pathlib import Path
import shutil
import sys

import numpy as np

BLOCK = 128
BLOCK_BYTES = 34


def blocks(data, rows, width):
    """View raw PQ2_0 bytes as (rows, width/128, 34) blocks."""
    if width % BLOCK:
        raise ValueError("PQ2_0 width must be a multiple of 128")
    return np.asarray(data, dtype=np.uint8).reshape(rows, width // BLOCK, BLOCK_BYTES)


def split(data, rows, width):
    """Return FP32 scales (rows, nb) and int8 ternary codes (rows, nb, 128)."""
    b = blocks(data, rows, width)
    scales = b[..., :2].copy().view("<f2")[..., 0].astype(np.float32)
    codes = ((b[..., 2:, None] >> np.arange(0, 8, 2, dtype=np.uint8)) & 3).reshape(rows, width // BLOCK, BLOCK)
    if np.any(codes == 3) or not np.isfinite(scales).all():
        raise ValueError("not finite ternary PQ2 weights")
    return scales, codes.astype(np.int8) - 1


def scaled_bytes(data, rows, width, factors):
    """Copy of PQ2_0 bytes with every block scale multiplied by its factor.

    Codes are untouched. A factor of exactly 1 reproduces the original bytes.
    """
    factors = np.asarray(factors, dtype=np.float32)
    if factors.shape != (rows, width // BLOCK) or not np.isfinite(factors).all():
        raise ValueError("factors must be finite with shape (rows, width/128)")
    out = blocks(data, rows, width).copy()
    old = out[..., :2].copy().view("<f2")[..., 0]
    new = (old.astype(np.float32) * factors).astype(np.float16)
    new = np.where(factors == 1, old, new)
    if not np.isfinite(new).all():
        raise ValueError("scaled PQ2 block overflowed FP16")
    out[..., :2] = new[..., None].view(np.uint8).reshape(rows, width // BLOCK, 2)
    return out.reshape(-1)


def merge_statistics(weight_scales, codes, delta):
    """How much of a dense weight delta survives each ternary projection.

    weight_scales: (rows, nb); codes: (rows, nb, 128) in {-1,0,1}; delta: (rows, width).
    Returns relative sizes and the fraction of the delta each projection realizes.
    """
    rows, nb, _ = codes.shape
    d = delta.reshape(rows, nb, BLOCK).astype(np.float64)
    s = weight_scales[..., None].astype(np.float64)
    w = s * codes
    target = w + d
    # Naive merge: keep scales, round the merged weight back to ternary codes.
    rounded = np.clip(np.rint(np.divide(target, s, out=np.zeros_like(target), where=s != 0)), -1, 1)
    naive = s * rounded - w
    # Scale-only: best per-block scale for the unchanged codes.
    norm = (w * w).sum(-1, keepdims=True)
    factor = np.divide((target * w).sum(-1, keepdims=True), norm, out=np.ones_like(norm), where=norm > 0)
    scale_only = factor * w - w
    dd = (d * d).sum()

    def realized(r):
        return {"cosine": float((r * d).sum() / np.sqrt(max((r * r).sum() * dd, 1e-300))),
                "relative_error": float(np.sqrt(((r - d) ** 2).sum() / max(dd, 1e-300)))}

    return {"delta_to_weight_norm": float(np.sqrt(dd / max((w * w).sum(), 1e-300))),
            "code_flip_fraction": float((rounded != codes).mean()),
            "naive": realized(naive), "scale_only": realized(scale_only),
            "scale_factor_abs_deviation_mean": float(np.abs(factor - 1).mean())}


def export_scaled(base, output, factors, prism_root):
    """Write a PQ2_0 GGUF whose named tensors have scaled block scales.

    factors maps GGUF tensor name to (rows, width/128) factors in stored row order.
    Every other byte is copied unchanged.
    """
    sys.path.insert(0, str(Path(prism_root) / "gguf-py"))
    from gguf import GGUFReader
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    tensors = {t.name: t for t in GGUFReader(str(base)).tensors}
    missing = set(factors) - set(tensors)
    if missing:
        raise KeyError(f"unknown tensors {sorted(missing)}")
    shutil.copyfile(base, output)
    image = np.memmap(output, dtype=np.uint8, mode="r+")
    try:
        for name, value in factors.items():
            tensor = tensors[name]
            if tensor.tensor_type.name != "PQ2_0":
                raise ValueError(f"{name} is not PQ2_0")
            width, rows = (int(n) for n in tensor.shape[:2])
            start = int(tensor.data_offset)
            size = rows * width // BLOCK * BLOCK_BYTES
            image[start:start + size] = scaled_bytes(image[start:start + size], rows, width, value)
        image.flush()
    finally:
        del image
    return output
