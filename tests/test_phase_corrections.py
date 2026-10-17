"""Complex transmission to air-referenced phase, without periodic boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.corrections import remove_phase_ramp, unwrap_phase


@pytest.mark.parametrize("shape", [(31, 42), (32, 41)])
def test_unwrap_recovers_nonperiodic_wrapped_phase(shape):
    v, u = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing="ij")
    phase = np.stack((13 * u + 7 * v + 2 * np.sin(3 * u - v), -9 * u + 3 * v**2)).astype(np.float32)
    amplitude = 1 + 0.2 * np.cos(u + v)
    expected = phase - phase.mean(axis=(-2, -1), keepdims=True)
    actual = unwrap_phase(jnp.asarray(amplitude * np.exp(1j * phase), jnp.complex64))
    np.testing.assert_allclose(actual, expected, atol=5e-5)
    np.testing.assert_allclose(unwrap_phase(jnp.angle(jnp.exp(1j * phase))), expected, atol=5e-5)


def test_air_reference_preserves_sample_and_handles_per_view_masks():
    v, u = np.meshgrid(np.linspace(-1, 1, 43), np.linspace(-1, 1, 47), indexing="ij")
    sample = -18 * np.maximum(1 - 4 * (u**2 + v**2), 0) ** 2
    truth = np.stack((sample, sample * 0.7)).astype(np.float32)
    plane = np.array([3, -5])[:, None, None] + np.array([7, -9])[:, None, None] * u + 2 * v
    air = np.broadcast_to(abs(u) > 0.65, truth.shape).copy()
    air[0, :5] = False
    phase = unwrap_phase(jnp.asarray(np.exp(1j * (truth + plane)), jnp.complex64))
    corrected = remove_phase_ramp(phase, air_mask=jnp.asarray(air))
    np.testing.assert_allclose(corrected, truth, atol=5e-5)
    # A shared air mask broadcasts to all views and remains composable under jit.
    corrected = jax.jit(lambda p: remove_phase_ramp(p, air_mask=abs(u) > 0.65))(phase)
    np.testing.assert_allclose(corrected, truth, atol=5e-5)
