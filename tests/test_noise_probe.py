"""Noise draws are independent of batching and respect the data objective."""

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.recon._noise_probe import gaussian_probe  # check-public-imports: allow-private


def test_probe_global_views_and_objective_weight_scale():
    indices = jnp.arange(5)
    probe = gaussian_probe(indices, (3, 4))
    np.testing.assert_array_equal(probe[2:], gaussian_probe(indices[2:], (3, 4)))
    np.testing.assert_allclose(np.asarray(probe).mean((-2, -1)), 0, atol=1e-7)
    weights = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4)
    weighted = gaussian_probe(indices, (3, 4), weights=weights)
    np.testing.assert_allclose(weighted.sum((-2, -1)), 0, atol=1e-5)
    np.testing.assert_allclose(
        gaussian_probe(indices, (3, 4), weights=weights * 3), weighted * 3, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_array_equal(weighted[:, 0, 0], 0)


def test_probe_rms_matches_dense_transpose_with_shifted_tail_and_support():
    from tomojax import Detector, Grid
    from tomojax.recon._noise_probe import probe_gradient_rms  # check-public-imports: allow-private

    grid, detector = Grid(2, 2, 1, 1, 1, 1), Detector(3, 2, 1, 1)
    poses = jnp.zeros((5, 4, 4))
    matrix = jnp.arange(120, dtype=jnp.float32).reshape(5, 6, 4) / 120
    support = jnp.asarray([[[1, 0], [1, 0.5]]])

    def backproject(images, start, selected, total):
        blocks = jax.lax.dynamic_slice_in_dim(matrix, start, selected.shape[0])
        return total + jnp.einsum("biv,bi->v", blocks, images.reshape(-1, 6)).reshape(1, 2, 2)

    weights = jnp.arange(30, dtype=jnp.float32).reshape(5, 2, 3)
    expected = (
        matrix.reshape(30, 4).T @ gaussian_probe(jnp.arange(5), (2, 3), weights=weights).ravel()
    ).reshape(1, 2, 2) * support
    actual = probe_gradient_rms(
        poses,
        grid,
        detector,
        backproject,
        batch_size=3,
        support=support,
        read_weights=lambda start, size: jax.lax.dynamic_slice_in_dim(weights, start, size),
    )
    np.testing.assert_allclose(actual, jnp.sqrt(jnp.mean(expected**2)), rtol=2e-5)
