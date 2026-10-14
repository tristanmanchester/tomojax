"""FBP filter dtype requests retain precision before coefficients reach JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.recon.filters import fft_filter_rows, rfft_filter_array


@pytest.mark.parametrize("filter_name", ["ramp", "hann", "shepp-logan"])
@pytest.mark.parametrize("representation", ["jax", "numpy", "dtype", "string"])
@pytest.mark.parametrize("bits", [32, 64])
def test_fbp_filter_preserves_requested_coefficient_precision(filter_name, representation, bits):
    """Typed scalar classes and dtype values denote the same precision."""
    name = f"float{bits}"
    dtype = {
        "jax": getattr(jnp, name),
        "numpy": getattr(np, name),
        "dtype": np.dtype(name),
        "string": name,
    }[representation]
    nu, du, n_fft = 31, 0.7, 64
    offsets = np.minimum(np.arange(n_fft), n_fft - np.arange(n_fft))
    impulse = np.zeros(n_fft)
    impulse[0] = 0.25 / du
    odd = offsets % 2 != 0
    impulse[odd] = -1 / (np.pi**2 * offsets[odd] ** 2 * du)
    spectrum = np.fft.fft(impulse).real[: n_fft // 2 + 1]
    frequency = 2 * np.arange(n_fft // 2 + 1) / n_fft
    if filter_name == "hann":
        spectrum *= (1 + np.cos(np.pi * frequency)) / 2
    elif filter_name == "shepp-logan":
        spectrum *= np.sinc(frequency / 2)
    with jax.enable_x64():
        actual = np.asarray(rfft_filter_array(filter_name, nu, du, dtype))
    assert actual.dtype == np.dtype(name)
    np.testing.assert_allclose(actual, spectrum, rtol=2e-7 if bits == 32 else 2e-15, atol=1e-16)


@pytest.mark.parametrize(
    "representation", [jnp.float64, np.float64, np.dtype("float64"), "float64"]
)
def test_fp64_ramp_filter_matches_independent_dense_convolution(representation):
    """The whole real-signal output, not only its declared dtype, retains FP64."""
    nu, du = 31, 0.7
    indices = np.arange(nu)
    rows = 0.4 + 0.1 * np.sin(indices / 4)[None, :]
    distance = indices[:, None] - indices[None, :]
    kernel = np.zeros((nu, nu))
    kernel[distance == 0] = 0.25 / du
    odd = distance % 2 != 0
    kernel[odd] = -1 / (np.pi**2 * distance[odd] ** 2 * du)
    with jax.enable_x64():
        actual = np.asarray(
            fft_filter_rows(jnp.asarray(rows), rfft_filter_array("ramp", nu, du, representation))
        )
    np.testing.assert_allclose(actual, rows @ kernel.T, rtol=2e-13, atol=2e-15)
