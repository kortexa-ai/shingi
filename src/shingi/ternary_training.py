"""Frozen Bonsai GGUF weights in a differentiable Qwen3.5 implementation.

Prism's pinned loader and converter define tensor names, GDN head ordering,
zero-centred norms, and the signed Hadamard contract. No donor weights are used.
PQ2 scales are expanded exactly into FP16, not requantized. The base remains
immutable; only explicitly attached LoRA parameters receive gradients.
"""
from pathlib import Path
import sys
import numpy as np

PRISM_REVISION = "d8f26eec76da6d09bb708bcba51ef64b8cd868a3"
BASE_SHA256 = "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"
MAPPING = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight", "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight", "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight", "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight", "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight", "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight", "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight", "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias", "ssm_conv1d.weight": "linear_attn.conv1d.weight",
}
GLOBAL = {"output.weight": "lm_head.weight", "output_norm.weight": "model.norm.weight",
          "token_embd.weight": "model.embed_tokens.weight"}


def decode_pq2(data, rows, width):
    """Reference codec: FP16 scale then 32 low-bit-first bytes per 128 values."""
    blocks = np.asarray(data, dtype=np.uint8).reshape(rows, width // 128, 34)
    scales = blocks[..., :2].copy().view('<f2').reshape(rows, width // 128, 1)
    codes = ((blocks[..., 2:, None] >> np.arange(0, 8, 2, dtype=np.uint8)) & 3).reshape(rows, width // 128, 128)
    if np.any(codes == 3) or not np.isfinite(scales).all():
        raise ValueError("not finite ternary PQ2 weights")
    return ((codes.astype(np.int8) - 1) * scales).reshape(rows, width).astype(np.float16)


def grouped_v_permutation(nk, nv, dim):
    return np.arange(nv * dim).reshape(nv // nk, nk, dim).transpose(1, 0, 2).reshape(-1)


def reorder_rows(a, stem, nk, nv, hk, hv):
    if nk == nv:
        return a
    if stem in ("attn_qkv.weight", "ssm_conv1d.weight"):
        qk = 2 * nk * hk
        return np.concatenate([a[:qk], a[qk:][grouped_v_permutation(nk, nv, hv)]])
    if stem == "attn_gate.weight":
        return a[grouped_v_permutation(nk, nv, hv)]
    if stem in ("ssm_alpha.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias"):
        return a[grouped_v_permutation(nk, nv, 1)]
    return a


def hadamard(x, signs, block=1024, inverse=False):
    import torch
    shape, dtype = x.shape, x.dtype
    if shape[-1] % block or block & (block - 1):
        raise ValueError("invalid Hadamard block")
    y = x.float()
    if not inverse:
        y = y * signs
    y = y.reshape(-1, block)
    stride = 1
    while stride < block:
        z = y.reshape(-1, block // (2 * stride), 2, stride)
        a, b = z.unbind(dim=2)
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, block)
        stride *= 2
    y = (y / block**0.5).reshape(shape)
    if inverse:
        y = y * signs
    return y.to(dtype)


def load_bonsai(path, prism_root, *, device="cuda", dtype=None):
    import subprocess
    import torch
    from torch import nn
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    if subprocess.check_output(["git", "-C", str(prism_root), "rev-parse", "HEAD"], text=True).strip() != PRISM_REVISION:
        raise ValueError("Prism revision mismatch")
    sys.path.insert(0, str(Path(prism_root) / "gguf-py"))
    from gguf import GGUFReader
    from gguf.quants import dequantize
    reader = GGUFReader(str(path))
    f = {k: v.contents() for k, v in reader.fields.items() if not k.startswith("tokenizer.")}
    if f.get("general.architecture") != "qwen35" or f.get("prism.hadamard.version") != 1 or not f.get("prism.hadamard.gdn_v_grouped"):
        raise ValueError("unsupported Bonsai contract")
    g = lambda k: f["qwen35." + k]
    nk, nv = int(g("ssm.group_count")), int(g("ssm.time_step_rank"))
    hk, hv = int(g("ssm.state_size")), int(g("ssm.inner_size")) // nv
    config = Qwen3_5TextConfig(
        hidden_size=int(g("embedding_length")), intermediate_size=int(g("feed_forward_length")),
        num_hidden_layers=int(g("block_count")), num_attention_heads=int(g("attention.head_count")),
        num_key_value_heads=int(g("attention.head_count_kv")), head_dim=int(g("attention.key_length")),
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")), vocab_size=248320,
        max_position_embeddings=int(g("context_length")), linear_num_value_heads=nv,
        linear_num_key_heads=nk, linear_value_head_dim=hv, linear_key_head_dim=hk,
        linear_conv_kernel_dim=int(g("ssm.conv_kernel")), full_attention_interval=int(g("full_attention_interval")),
        tie_word_embeddings=False, use_cache=False, attn_implementation="sdpa",
        rope_parameters={"rope_type": "default", "rope_theta": float(g("rope.freq_base")),
                         "partial_rotary_factor": int(g("rope.dimension_count")) / int(g("attention.key_length")),
                         "mrope_section": list(g("rope.dimension_sections"))[:3], "mrope_interleaved": True})
    dtype = dtype or torch.float16
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    expected = dict(model.named_parameters())
    assigned = set()
    folded = set(f["prism.hadamard.weight_names"])
    inverse = set(f["prism.hadamard.inverse_weight_names"])
    block = int(f["prism.hadamard.block_size"])
    signs, offset = {}, 0
    for width in f["prism.hadamard.sign_widths"]:
        raw = f["prism.hadamard.sign_values"][offset:offset + width]
        if len(raw) != width or any(x not in (-1, 1) for x in raw):
            raise ValueError("invalid Hadamard signs")
        signs[width] = torch.tensor(raw, dtype=torch.float32, device=device)
        offset += width

    class Folded(nn.Module):
        def __init__(self, weight, sign, embedding):
            super().__init__()
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.register_buffer("signs", sign)
            self.embedding = embedding
        def forward(self, x):
            if self.embedding:
                return hadamard(torch.nn.functional.embedding(x, self.weight), self.signs, block, inverse=True)
            return torch.nn.functional.linear(hadamard(x, self.signs, block), self.weight)

    for tensor in reader.tensors:
        name = tensor.name
        stem = name
        if name in GLOBAL:
            target = GLOBAL[name]
        else:
            _, layer, stem = name.split(".", 2)
            target = f"model.layers.{layer}." + MAPPING[stem]
        if target not in expected or target in assigned:
            raise ValueError("unexpected tensor " + target)
        shape = tuple(int(n) for n in tensor.shape[::-1])
        if tensor.tensor_type.name == "PQ2_0":
            # Expand directly to its destination, bounded to 256 CPU rows.
            weight = torch.empty(shape, dtype=dtype, device=device)
            perm = reorder_rows(np.arange(shape[0]), stem, nk, nv, hk, hv)
            inverse_perm = np.argsort(perm)
            for start in range(0, shape[0], 256):
                stop = min(start + 256, shape[0])
                data = decode_pq2(tensor.data[start:stop], stop-start, shape[1])
                idx = torch.as_tensor(inverse_perm[start:stop], device=device)
                weight.index_copy_(0, idx, torch.from_numpy(data).to(device=device, dtype=dtype))
        else:
            a = dequantize(tensor.data, tensor.tensor_type).reshape(shape)
            a = reorder_rows(a, stem, nk, nv, hk, hv)
            if stem == "ssm_a":
                if not (a < 0).all():
                    raise ValueError("invalid recurrent A")
                a = np.log(-a)
            if stem == "ssm_conv1d.weight":
                a = a[:, None, :]
            if stem.endswith("norm.weight") and stem != "ssm_norm.weight":
                a = a - 1.0
            # Norms/recurrent parameters stay FP32; linear projections use activation dtype.
            td = dtype if a.ndim >= 2 else torch.float32
            weight = torch.from_numpy(np.array(a, copy=True)).to(device=device, dtype=td)
        if tuple(weight.shape) != tuple(expected[target].shape):
            raise ValueError(f"shape mismatch {name}: {weight.shape} != {expected[target].shape}")
        parent_name, _, attr = target.rpartition(".")
        parent = model.get_submodule(parent_name)
        if name in folded or name in inverse:
            grand_name, _, module_name = parent_name.rpartition(".")
            model.get_submodule(grand_name).add_module(module_name, Folded(weight, signs[shape[-1]], name in inverse))
        else:
            setattr(parent, attr, nn.Parameter(weight, requires_grad=False))
        assigned.add(target)
        if len(assigned) % 100 == 0:
            print(f"Loaded {len(assigned)}/{len(expected)} Bonsai tensors", flush=True)
    if assigned != set(expected):
        raise ValueError("missing weights " + str(set(expected) - assigned))
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config).to(device)
    if any(t.is_meta for t in model.parameters()) or any(t.is_meta for t in model.buffers()):
        raise ValueError("unmaterialized model tensors")
    model.requires_grad_(False)
    model.eval()
    return model, {"tensors": len(assigned), "dtype": str(dtype), "hadamard_block": block,
                   "config": config.to_dict(), "prism_revision": PRISM_REVISION}


def attach_lora(model, rank=8, alpha=16, last_layers=64):
    import torch
    from torch import nn
    class Adapter(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            out_size, in_size = base.weight.shape
            self.a = nn.Parameter(torch.empty(rank, in_size, device=base.weight.device, dtype=torch.float32))
            self.b = nn.Parameter(torch.zeros(out_size, rank, device=base.weight.device, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.a, a=5**0.5)
            self.scale = alpha / rank
        def forward(self, x):
            delta = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.a), self.b)
            return self.base(x) + (delta * self.scale).to(x.dtype)
    for layer in model.model.layers[-last_layers:]:
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(layer.mlp, name, Adapter(getattr(layer.mlp, name)))
    return [p for p in model.parameters() if p.requires_grad]
