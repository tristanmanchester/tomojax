"""The optional challenge executor preserves physical fractional translations."""

from bench.nanotomo_motion_cuda import shift_complex_objects_cuda
import jax
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(17, 23), (18, 24)])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_cuda_complex_motion_has_correct_sign_dtype_and_independent_view_shifts(shape, dtype):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    v, u = np.indices(shape)
    shifts = np.array([[0.0, 0.0], [1.3, -2.7], [-0.25, 0.6]])
    source = np.broadcast_to(
        np.exp(2j * np.pi * (3 * u / shape[1] + 2 * v / shape[0])), (3, *shape)
    ).astype(dtype)
    expected = np.exp(
        2j
        * np.pi
        * (
            3 * (u[None] - shifts[:, 0, None, None]) / shape[1]
            + 2 * (v[None] - shifts[:, 1, None, None]) / shape[0]
        )
    )
    actual = shift_complex_objects_cuda(source, shifts)
    assert actual.dtype == source.dtype
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=2e-6 if dtype == np.complex64 else 2e-13
    )
