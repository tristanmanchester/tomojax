import numpy as np
import pytest

from tomojax.data.contrast import (
    absorption_to_transmission,
    flat_dark_to_absorption,
    transmission_to_absorption,
)


def test_transmission_absorption_roundtrip_numpy():
    transmission = np.array([[1.0, 0.5, 0.1]], dtype=np.float32)
    absorption = transmission_to_absorption(transmission)
    rebuilt = absorption_to_transmission(absorption)
    np.testing.assert_allclose(
        rebuilt, np.clip(transmission, 1e-6, None), rtol=1e-6, atol=1e-6
    )


def test_flat_dark_to_absorption_matches_manual():
    rng = np.random.default_rng(0)
    projections = rng.uniform(0.2, 1.0, size=(4, 8, 6)).astype(np.float32)
    flats = rng.uniform(0.9, 1.1, size=(6, 8, 6)).astype(np.float32)
    darks = rng.uniform(0.0, 0.05, size=(3, 8, 6)).astype(np.float32)

    absorption = flat_dark_to_absorption(projections, flats, darks, min_intensity=1e-5)

    flat_avg = flats.mean(axis=0)
    dark_avg = darks.mean(axis=0)
    norm = (projections - dark_avg) / np.maximum(flat_avg - dark_avg, 1e-5)
    norm = np.maximum(norm, 1e-5)
    expected = -np.log(norm)

    np.testing.assert_allclose(absorption, expected, rtol=1e-6, atol=1e-6)


def test_transmission_absorption_with_jax():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    transmission = jnp.array([0.8, 0.5, 0.2], dtype=jnp.float32)
    absorption = transmission_to_absorption(transmission)
    rebuilt = absorption_to_transmission(absorption)
    assert isinstance(absorption, jax.Array)
    assert isinstance(rebuilt, jax.Array)
    np.testing.assert_allclose(
        np.asarray(rebuilt),
        np.clip(np.asarray(transmission), 1e-6, None),
        rtol=1e-6,
        atol=1e-6,
    )
