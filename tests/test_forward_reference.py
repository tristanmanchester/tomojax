from __future__ import annotations

# pyright: reportAny=false, reportUnknownMemberType=false
import jax.numpy as jnp
import numpy as np

from tomojax.forward import (
    masked_whitened_residual,
    project_parallel_reference,
    pseudo_huber_loss,
    pseudo_huber_weights,
    residual_loss,
)
from tomojax.geometry import GeometryState


def test_project_parallel_reference_shape_and_zero_pose_values() -> None:
    volume = jnp.ones((4, 4, 4), dtype=jnp.float32)
    geometry = GeometryState.zeros(3)

    projections = project_parallel_reference(volume, geometry)

    assert projections.shape == (3, 4, 4)
    np.testing.assert_allclose(np.asarray(projections), 4.0)


def test_project_parallel_reference_applies_detector_shift() -> None:
    volume = jnp.zeros((4, 4, 4), dtype=jnp.float32)
    volume = volume.at[0, :, 1].set(1.0)
    geometry = GeometryState.zeros(1)
    shifted_pose = geometry.pose.with_updates(dx_px=np.array([1.0], dtype=np.float64))
    shifted_geometry = GeometryState(setup=geometry.setup, pose=shifted_pose)

    base = project_parallel_reference(volume, geometry)
    shifted = project_parallel_reference(volume, shifted_geometry)

    np.testing.assert_allclose(np.asarray(shifted[0]), np.roll(np.asarray(base[0]), 1, axis=1))


def test_masked_whitened_residual_zeros_invalid_pixels() -> None:
    predicted = jnp.array([[2.0, 4.0], [6.0, 8.0]], dtype=jnp.float32)
    observed = jnp.ones((2, 2), dtype=jnp.float32)
    mask = jnp.array([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)

    residual = masked_whitened_residual(predicted, observed, mask=mask, sigma=2.0)

    np.testing.assert_allclose(np.asarray(residual), [[0.5, 0.0], [2.5, 0.0]])


def test_pseudo_huber_loss_is_quadratic_near_zero_and_robust_for_large_values() -> None:
    residual = jnp.array([0.01, 10.0], dtype=jnp.float32)
    loss = pseudo_huber_loss(residual, delta=1.0)

    np.testing.assert_allclose(float(loss[0]), 0.5 * 0.01**2, rtol=2e-3)
    assert float(loss[1]) < 0.5 * 10.0**2


def test_residual_loss_reports_valid_count_and_downweights_outliers() -> None:
    predicted = jnp.array([0.0, 10.0, 2.0], dtype=jnp.float32)
    observed = jnp.zeros((3,), dtype=jnp.float32)
    mask = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)

    result = residual_loss(predicted, observed, mask=mask, delta=1.0)

    assert float(result.valid_count) == 2.0
    assert float(result.loss) > 0.0
    assert float(result.weights[1]) < float(result.weights[0])
    assert float(pseudo_huber_weights(jnp.asarray([0.0]), delta=1.0)[0]) == 1.0
