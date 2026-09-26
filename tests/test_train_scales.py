from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shingi.pq2_scales import scaled_bytes, split
from shingi.ternary_training import decode_pq2
from train_scales import changed_scales, check_wrapped, differing_bytes, native_parity, target_map, unit_parity

FIELDS = {'qwen35.ssm.group_count': 2, 'qwen35.ssm.time_step_rank': 4, 'qwen35.ssm.state_size': 2,
          'qwen35.ssm.inner_size': 12, 'prism.hadamard.inverse_weight_names': ['token_embd.weight']}
QKV = 2 * 2 * 2 + 4 * 3  # q and k heads, then grouped v heads
TENSORS = [('blk.0.attn_qkv.weight', 'PQ2_0', QKV), ('blk.0.ffn_down.weight', 'PQ2_0', 8),
           ('output.weight', 'PQ2_0', 5), ('token_embd.weight', 'PQ2_0', 5), ('blk.0.attn_norm.weight', 'F32', 8)]


def test_target_map_names_paths_and_permutations():
    targets = target_map(TENSORS, FIELDS)
    assert {k: v[0] for k, v in targets.items()} == {
        'blk.0.attn_qkv.weight': 'model.layers.0.linear_attn.in_proj_qkv',
        'blk.0.ffn_down.weight': 'model.layers.0.mlp.down_proj', 'output.weight': 'lm_head'}
    perm = targets['blk.0.attn_qkv.weight'][1]
    assert sorted(perm) == list(range(QKV)) and not np.array_equal(perm, np.arange(QKV))
    np.testing.assert_array_equal(perm[:8], np.arange(8))  # q/k rows keep their order
    np.testing.assert_array_equal(targets['blk.0.ffn_down.weight'][1], np.arange(8))


def test_stored_factors_invert_the_loader_permutation():
    torch = pytest.importorskip('torch')
    from train_scales import stored_factors
    perm = target_map(TENSORS, FIELDS)['blk.0.attn_qkv.weight'][1]
    stored = np.arange(QKV * 2, dtype=np.float32).reshape(QKV, 2)
    loaded = np.empty_like(stored)
    loaded[np.argsort(perm)] = stored  # load_bonsai: index_copy_(0, argsort(perm)[rows], stored rows)
    module = SimpleNamespace(factors=torch.tensor(loaded))
    np.testing.assert_array_equal(stored_factors({'t': (module, perm)})['t'], stored)


def test_check_wrapped_requires_each_target_once_with_stored_shape():
    targets = target_map(TENSORS, FIELDS)
    expected = [n for n, kind, _ in TENSORS if kind == 'PQ2_0' and n != 'token_embd.weight']
    paths = [p for p, _ in targets.values()]
    shapes = {n: (r, 256) for n, _, r in TENSORS if n in expected}
    factors = {n: np.ones((r, 2)) for n, (r, _) in shapes.items()}
    assert check_wrapped(expected, targets, paths, shapes, factors) == {'scaled_tensors': 3, 'factors': 2 * (QKV + 13)}
    with pytest.raises(ValueError, match='differ'):
        check_wrapped(expected + ['blk.1.ffn_up.weight'], targets, paths, shapes, factors)
    with pytest.raises(ValueError, match='exactly once'):
        check_wrapped(expected, targets, paths + ['lm_head.inner'], shapes, factors)
    with pytest.raises(ValueError, match='exactly once'):
        check_wrapped(expected, targets, paths[1:], shapes, factors)
    with pytest.raises(ValueError, match='stored shape'):
        check_wrapped(expected, targets, paths, shapes, {**factors, 'output.weight': np.ones((2, 5))})


def test_parity_gates():
    reference = [[1., 2., 3.], [0., -1.]]
    exact = unit_parity(reference, reference)
    assert exact['passed'] and exact['max_abs_difference'] == 0
    assert unit_parity(reference, [[1.0005, 2., 3.], [0., -1.]])['passed']
    assert not unit_parity(reference, [[1.1, 2., 3.], [0., -1.]])['passed']
    assert not unit_parity(reference, [[3., 2., 1.], [0., -1.]])['passed']
    assert native_parity(reference, [[1.05, 2., 3.], [0., -1.02]])['passed']
    assert not native_parity(reference, [[3., 2., 1.], [0., -1.]])['passed']


def test_differing_bytes(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.write_bytes(bytes(range(200)))
    b.write_bytes(bytes(range(100)) + bytes([0]) + bytes(range(101, 199)) + bytes([7]))
    assert differing_bytes(a, a, chunk=64) == 0 and differing_bytes(a, b, chunk=64) == 2
    b.write_bytes(bytes(10))
    with pytest.raises(ValueError):
        differing_bytes(a, b, chunk=64)


def test_chunked_scaled_matmul_matches_single_chunk():
    torch = pytest.importorskip('torch')
    from train_scales import scaled_matmul
    torch.manual_seed(1)
    x = torch.randn(1, 3, 256)
    weight = torch.randn(5, 256, dtype=torch.float16)
    results = []
    for chunk in (1 << 28, 512):  # one chunk, then two rows per chunk
        factors = (1 + .1 * torch.randn(5, 2, generator=torch.Generator().manual_seed(2))).requires_grad_(True)
        xi = x.clone().requires_grad_(True)
        y = scaled_matmul(chunk).apply(xi, weight, factors)
        y.square().sum().backward()
        results.append((y.detach(), xi.grad, factors.grad))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-3)  # FP32 summation order differs across chunks


def test_distill_loss_is_stationary_at_the_teacher():
    torch = pytest.importorskip('torch')
    from train_scales import distill_loss
    teacher = torch.tensor([.5, -1., 2.])
    logits = teacher.clone().requires_grad_(True)
    distill_loss(logits, teacher, torch.tensor([0., 1., 0.]), 1.).backward()
    torch.testing.assert_close(logits.grad, torch.zeros(3))


def ternary(rows=4, width=256, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 3, size=(rows, width), dtype=np.uint8)
    bits = np.sum(codes.reshape(rows, width // 128, 32, 4) << np.arange(0, 8, 2, dtype=np.uint8), axis=-1).astype(np.uint8)
    scales = rng.uniform(.001, .05, size=(rows, width // 128)).astype('<f2')
    return np.concatenate((scales[..., None].view(np.uint8), bits), axis=-1).reshape(-1)


def test_fp16_scaled_weight_equals_exported_ternary_weight():
    torch = pytest.importorskip('torch')
    from train_scales import effective
    data = ternary(seed=3)
    factors = (1 + .003 * np.random.default_rng(4).standard_normal((4, 2))).astype(np.float32)
    scales, codes = split(data, 4, 256)
    rounded = effective(torch.from_numpy(decode_pq2(data, 4, 256)), torch.from_numpy(factors)).numpy()
    np.testing.assert_array_equal(rounded, (codes * (scales * factors).astype(np.float16)[..., None]).reshape(4, 256))
    np.testing.assert_array_equal(rounded, decode_pq2(scaled_bytes(data, 4, 256, factors), 4, 256).astype(np.float32))


def test_changed_scales_counts_fp16_visible_blocks():
    data = ternary()
    assert changed_scales(data, 4, 256, np.ones((4, 2))) == 0
    assert changed_scales(data, 4, 256, np.full((4, 2), 1 + 1e-5)) == 0  # below FP16 resolution
    factors = np.ones((4, 2))
    factors[1, 0] = factors[3, 1] = 1.02
    assert changed_scales(data, 4, 256, factors) == 2
