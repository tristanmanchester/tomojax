import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.corrections._ring_parity import _plan, regularize
from tomojax.corrections._rings import _fill  # check-public-imports: allow-private


@pytest.mark.parametrize("size", [63, 64])
def test_local_variance_matches_dense_reflection_covariance(size: int) -> None:
    edges, _, filters, weights, *_ = _plan(size)
    powers = np.linspace(0.2, 2, len(edges) - 1)
    spectrum = np.r_[powers[0], np.repeat(powers, np.diff(edges))]
    eye = np.eye(size)
    even = (eye + eye[::-1]) / 2
    covariance = np.fft.irfft(np.fft.rfft(eye, axis=0) * spectrum[:, None], n=size, axis=0)
    np.testing.assert_allclose(filters.sum(0), 1, atol=1e-14)
    for k in [0, 2, len(filters) - 1]:
        operator = np.fft.irfft(np.fft.rfft(eye, axis=0) * filters[k, :, None], n=size, axis=0)
        exact = np.diag(operator @ even @ covariance @ even @ operator.T)
        np.testing.assert_allclose(powers @ weights[k], exact, atol=1e-12)


@pytest.mark.parametrize(("size", "axis"), [(127, 63.0), (128, 63.5), (128, 66.5)])
def test_parity_prior_preserves_odd_fields_and_units(size: int, axis: float) -> None:
    u = np.arange(size) - axis
    odd = jnp.asarray(np.sin(u / 3), jnp.float32)
    np.testing.assert_array_equal(regularize(odd, axis, "significance"), odd)
    rng = np.random.default_rng(8647)
    profile = jnp.asarray(
        0.02 * rng.normal(size=size) + 0.1 * np.exp(-((u / 12) ** 2)), jnp.float32
    )
    fixed = np.asarray(regularize(profile, axis, "significance"))
    for scale in [0.01, 100]:
        rescaled = np.asarray(regularize(profile * scale, axis, "significance")) / scale
        assert np.linalg.norm(rescaled - fixed) < 2e-6 * np.linalg.norm(fixed)


def test_stuck_column_fit_does_not_invent_variation_in_constant_views() -> None:
    profile = np.random.default_rng(7543).uniform(0, 3, 256).astype(np.float32)
    values = jnp.asarray(np.broadcast_to(profile, (180, 256)))
    np.testing.assert_array_equal(_fill(values), values)
