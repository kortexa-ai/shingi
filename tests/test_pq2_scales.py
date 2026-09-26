from pathlib import Path
import sys

import numpy as np
import pytest

from shingi.pq2_scales import merge_statistics, scaled_bytes, split
from shingi.ternary_training import decode_pq2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from merge_check import fwht_rows


def packed(rows=3, width=256, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 3, size=(rows, width), dtype=np.uint8)
    bits = np.sum(codes.reshape(rows, width // 128, 32, 4) << np.arange(0, 8, 2, dtype=np.uint8), axis=-1).astype(np.uint8)
    scales = rng.uniform(.01, .05, size=(rows, width // 128)).astype('<f2')
    data = np.concatenate((scales[..., None].view(np.uint8).reshape(rows, width // 128, 2), bits), axis=-1)
    return data.reshape(-1), codes.astype(np.int8) - 1, scales


def test_split_matches_reference_decoder():
    data, codes, scales = packed()
    got_scales, got_codes = split(data, 3, 256)
    np.testing.assert_array_equal(got_codes.reshape(3, 256), codes)
    np.testing.assert_array_equal(got_scales, scales.astype(np.float32))
    np.testing.assert_array_equal(decode_pq2(data, 3, 256),
                                  (got_codes * got_scales[..., None]).reshape(3, 256).astype(np.float16))


def test_unit_factors_are_byte_exact_and_codes_never_change():
    data, _, _ = packed()
    np.testing.assert_array_equal(scaled_bytes(data, 3, 256, np.ones((3, 2))), data)
    factors = np.array([[2, 1], [.5, 1], [1, 3]], dtype=np.float32)
    out = scaled_bytes(data, 3, 256, factors)
    before, after = split(data, 3, 256), split(out, 3, 256)
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_allclose(after[0], (before[0] * factors).astype(np.float16).astype(np.float32))


def test_scaled_bytes_rejects_bad_factors():
    data, _, _ = packed()
    with pytest.raises(ValueError):
        scaled_bytes(data, 3, 256, np.ones((3, 3)))
    with pytest.raises(ValueError):
        scaled_bytes(data, 3, 256, np.full((3, 2), np.inf))
    with pytest.raises(ValueError, match='overflow'):
        scaled_bytes(data, 3, 256, np.full((3, 2), 1e7))


def test_merge_statistics_separates_scale_and_code_changes():
    data, _, _ = packed(seed=1)
    scales, codes = split(data, 3, 256)
    weight = (codes * scales[..., None]).reshape(3, 256)
    # A pure 10% block-scale change is fully realized by scale-only tuning.
    stats = merge_statistics(scales, codes, .1 * weight)
    assert stats['scale_only']['cosine'] > .999 and stats['scale_only']['relative_error'] < 1e-6
    # A tiny dense change disappears under naive re-ternarization.
    small = 1e-4 * np.random.default_rng(2).standard_normal((3, 256))
    stats = merge_statistics(scales, codes, small)
    assert stats['code_flip_fraction'] == 0 and stats['naive']['relative_error'] == pytest.approx(1.0)


def test_row_hadamard_matches_training_transform():
    torch = pytest.importorskip('torch')
    from shingi.ternary_training import hadamard
    rng = np.random.default_rng(3)
    a = rng.standard_normal((4, 2048))
    signs = rng.choice([-1., 1.], size=2048)
    expected = hadamard(torch.tensor(a, dtype=torch.float64), torch.tensor(signs), 1024).numpy()
    np.testing.assert_allclose(fwht_rows(a, signs, 1024), expected, atol=1e-5)  # training transform runs in FP32
    # Stored-domain merge: B A x == B H(A) H(x) for the orthogonal signed transform.
    x = rng.standard_normal(2048)
    hx = hadamard(torch.tensor(x[None]), torch.tensor(signs), 1024).numpy()[0]
    np.testing.assert_allclose(a @ x, fwht_rows(a, signs, 1024) @ hx, atol=1e-4)


def test_scaled_matmul_uses_rounded_forward_and_straight_through_factor_gradient():
    torch = pytest.importorskip('torch')
    from train_scales import scaled_matmul
    torch.manual_seed(0)
    x = torch.randn(2, 5, 256, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(3, 256, dtype=torch.float16)
    factors = (1 + .01 * torch.randn(3, 2)).requires_grad_(True)
    y = scaled_matmul().apply(x, weight, factors)
    y.square().sum().backward()
    rounded = (weight.float() * factors.detach().repeat_interleave(128, 1)).half().float()
    torch.testing.assert_close(y, torch.nn.functional.linear(x.detach(), rounded))
    g = 2 * y.detach().reshape(-1, 3)
    flat = x.detach().reshape(-1, 256)
    torch.testing.assert_close(x.grad, (g @ rounded).reshape(x.shape))
    expected = ((g.T @ flat) * weight.float()).reshape(3, 2, 128).sum(-1)  # unrounded straight-through
    assert expected.abs().min() > 0
    torch.testing.assert_close(factors.grad, expected, rtol=1e-4, atol=1e-3)
