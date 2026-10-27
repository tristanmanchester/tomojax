import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.corrections._ring_parity import _plan, regularize
from tomojax.corrections._rings import (  # check-public-imports: allow-private
    _feasible_response,
    _invert,
)


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


@pytest.mark.parametrize("beta", [0.0, 0.1, 1.0, 10.0, 100.0])
def test_response_inverse_including_negative_attenuation(beta: float) -> None:
    truth = jnp.linspace(-3.0, 10.0, 101)[:, None]
    alpha, slope = jnp.array([0.2]), jnp.array([beta])
    observed = truth + alpha - slope * jnp.expm1(-truth)
    recovered = _invert(observed, alpha, slope)
    np.testing.assert_allclose(recovered, truth, atol=2e-5, rtol=2e-6)


def test_feasible_negative_response_is_unchanged() -> None:
    slope = jnp.array([-2.0, -0.9, -0.5, -0.1])
    truth = jnp.log(-2 * slope) + jnp.linspace(0.1, 4.0, 101)[:, None]
    alpha = jnp.array([0.0, 0.2, -0.1, 0.0])
    observed = truth + alpha - slope * jnp.expm1(-truth)
    fitted_alpha, fitted_slope = _feasible_response(observed, truth, alpha, slope)
    np.testing.assert_array_equal(fitted_alpha, alpha)
    np.testing.assert_array_equal(fitted_slope, slope)
    np.testing.assert_allclose(_invert(observed, alpha, slope), truth, atol=2e-6)


def test_regularised_response_is_refit_before_inversion() -> None:
    # Similar to a coarse real-scan fit: the original law has no real root
    # for its lowest observations, despite all inputs being finite.
    observed = jnp.linspace(0.08, 1.45, 121)[:, None]
    alpha, slope = jnp.array([0.342]), jnp.array([-0.64])
    assert float(observed[0, 0] - alpha[0]) < float(jnp.log(-slope[0]) + slope[0] + 1)
    fitted_alpha, fitted_slope = _feasible_response(observed, observed, alpha, slope)
    recovered = _invert(observed, fitted_alpha, fitted_slope)
    assert np.isfinite(recovered).all()
    derivative = 1 + fitted_slope * jnp.exp(-recovered)
    assert float(derivative.min()) >= 0.5 - 1e-6
    mapped = recovered + fitted_alpha - fitted_slope * jnp.expm1(-recovered)
    np.testing.assert_allclose(mapped, observed, atol=3e-7, rtol=2e-6)
    perturbed = _invert(observed + 1e-4, fitted_alpha, fitted_slope)
    assert float(jnp.abs(perturbed - recovered).max()) <= 2.01e-4


def test_feasibility_boundary_refit_is_continuous() -> None:
    slope, alpha = jnp.array([-2.0]), jnp.array([0.0])
    lower = jnp.log(-2 * slope) + slope + 0.5
    q = lower + jnp.linspace(-1e-4, 1.0, 21)[:, None]
    a, b = _feasible_response(q, q, alpha, slope)
    assert float(jnp.abs(a - alpha).max() + jnp.abs(b - slope).max()) < 1e-3
    p = _invert(q, a, b)
    np.testing.assert_allclose(p + a - b * jnp.expm1(-p), q, atol=5e-7)
