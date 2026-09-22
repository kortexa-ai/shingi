import numpy as np
from shingi.ternary_training import decode_pq2, grouped_v_permutation, reorder_rows


def test_pq2_scale_sign_and_bit_order():
    codes = np.resize(np.array([0, 1, 2, 1], dtype=np.uint8), 256).reshape(2, 128)
    packed = np.sum(codes.reshape(2, 32, 4) << np.arange(0, 8, 2, dtype=np.uint8), axis=-1).astype(np.uint8)
    scales = np.array([0.03125, 0.046875], dtype='<f2')
    data = np.concatenate((scales.view(np.uint8).reshape(2, 2), packed), axis=1)
    got = decode_pq2(data, 1, 256).reshape(2, 128)
    np.testing.assert_array_equal(got, (codes.astype(np.int8) - 1) * scales[:, None])


def test_gdn_tiled_to_grouped_rows():
    # Native tiled v heads: k0v0, k1v0, k0v1, k1v1, k0v2, k1v2.
    np.testing.assert_array_equal(grouped_v_permutation(2, 6, 1), [0, 2, 4, 1, 3, 5])
    a = np.arange(10)
    np.testing.assert_array_equal(reorder_rows(a, 'attn_qkv.weight', 2, 6, 1, 1), [0,1,2,3,4,6,8,5,7,9])
