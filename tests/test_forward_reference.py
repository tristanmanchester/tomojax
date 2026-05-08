from __future__ import annotations

from typing import cast

import jax

# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry import Detector, Grid, LaminographyGeometry
from tomojax.forward import (
    PROJECTION_OPERATOR,
    core_projection_geometry_from_state,
    masked_whitened_residual,
    project_parallel_reference,
    project_parallel_reference_arrays,
    pseudo_huber_loss,
    pseudo_huber_weights,
    residual_loss,
    robust_residual_scale,
)
from tomojax.geometry import (
    CORE_X_AXIS,
    CORE_Y_AXIS,
    CORE_Z_AXIS,
    DET_U_VOLUME_AXIS,
    DET_V_VOLUME_AXIS,
    TOMO_AXIS,
    TOMO_ROTATION_PLANE_AXES,
    AcquisitionParameters,
    GeometryState,
)


def test_project_parallel_reference_shape_and_zero_pose_values() -> None:
    volume = jnp.ones((4, 4, 4), dtype=jnp.float32)
    geometry = GeometryState.zeros(3)

    projections = project_parallel_reference(volume, geometry)

    assert projections.shape == (3, 4, 4)
    np.testing.assert_allclose(np.asarray(projections), 4.0)


def test_project_parallel_reference_applies_detector_shift() -> None:
    volume = jnp.zeros((5, 5, 5), dtype=jnp.float32)
    volume = volume.at[2, :, 2].set(1.0)
    geometry = GeometryState.zeros(1)
    shifted_pose = geometry.pose.with_updates(dx_px=np.array([1.0], dtype=np.float64))
    shifted_geometry = GeometryState(setup=geometry.setup, pose=shifted_pose)

    base = project_parallel_reference(volume, geometry)
    shifted = project_parallel_reference(volume, shifted_geometry)

    assert float(shifted[0, 2, 1]) == float(base[0, 2, 2])
    assert float(shifted[0, 2, 2]) == 0.0


def test_core_volume_axis_constants_match_projector_convention() -> None:
    assert CORE_X_AXIS == 0
    assert CORE_Y_AXIS == 1
    assert CORE_Z_AXIS == 2
    assert DET_U_VOLUME_AXIS == CORE_X_AXIS
    assert DET_V_VOLUME_AXIS == CORE_Z_AXIS
    assert TOMO_ROTATION_PLANE_AXES == (CORE_X_AXIS, CORE_Y_AXIS)
    assert TOMO_AXIS == CORE_Z_AXIS


def test_core_volume_axis_translations_match_detector_axes() -> None:
    base_volume = jnp.zeros((5, 5, 5), dtype=jnp.float32).at[2, :, 2].set(1.0)
    geometry = GeometryState.zeros(1)
    det_u_minus = GeometryState(
        setup=geometry.setup,
        pose=geometry.pose.with_updates(dx_px=np.array([-1.0], dtype=np.float64)),
    )

    axis0_shifted = jnp.zeros((5, 5, 5), dtype=jnp.float32).at[3, :, 2].set(1.0)
    axis1_shifted = jnp.zeros((5, 5, 5), dtype=jnp.float32).at[2, 3, 2].set(1.0)
    axis2_shifted = jnp.zeros((5, 5, 5), dtype=jnp.float32).at[2, :, 3].set(1.0)

    base = project_parallel_reference(base_volume, geometry)
    det_u = project_parallel_reference(base_volume, det_u_minus)
    shifted0 = project_parallel_reference(axis0_shifted, geometry)
    shifted1 = project_parallel_reference(axis1_shifted, geometry)
    shifted2 = project_parallel_reference(axis2_shifted, geometry)

    assert _projection_centroid(base[0]) == (2.0, 2.0)
    assert _projection_centroid(det_u[0]) == (2.0, 3.0)
    assert _projection_centroid(shifted0[0]) == (2.0, 3.0)
    assert _projection_centroid(shifted1[0]) == (2.0, 2.0)
    assert _projection_centroid(shifted2[0]) == (3.0, 2.0)
    np.testing.assert_allclose(np.asarray(shifted0), np.asarray(det_u), atol=1e-6)


def test_project_parallel_reference_applies_fractional_detector_shift() -> None:
    volume = jnp.zeros((5, 5, 5), dtype=jnp.float32)
    volume = volume.at[2, :, 2].set(1.0)

    shifted = project_parallel_reference_arrays(
        volume,
        theta_rad=jnp.asarray([0.0], dtype=jnp.float32),
        dx_px=jnp.asarray([0.5], dtype=jnp.float32),
        dz_px=jnp.asarray([0.0], dtype=jnp.float32),
    )

    row = np.asarray(shifted[0, 2])
    np.testing.assert_allclose(row, [0.0, 2.5, 2.5, 0.0, 0.0], atol=1e-6)


def test_project_parallel_reference_arrays_is_differentiable_for_dx() -> None:
    volume = jnp.zeros((5, 5, 5), dtype=jnp.float32)
    volume = volume.at[2, :, 2].set(1.0)

    def pixel_value(dx_px: jax.Array) -> jax.Array:
        projected = project_parallel_reference_arrays(
            volume,
            theta_rad=jnp.asarray([0.0], dtype=jnp.float32),
            dx_px=jnp.asarray([dx_px], dtype=jnp.float32),
            dz_px=jnp.asarray([0.0], dtype=jnp.float32),
        )
        return projected[0, 2, 2]

    gradient = jax.grad(pixel_value)(jnp.asarray(0.25, dtype=jnp.float32))

    assert float(gradient) < 0.0


def test_core_projection_geometry_records_nominal_theta_and_shift() -> None:
    geometry = GeometryState.zeros(2)
    pose = geometry.pose.with_updates(
        theta_nominal_rad=np.array([0.0, np.pi / 2.0], dtype=np.float64),
        dx_px=np.array([1.5, -2.0], dtype=np.float64),
        dz_px=np.array([0.25, -0.5], dtype=np.float64),
    )
    geometry = GeometryState(setup=geometry.setup, pose=pose)

    core = core_projection_geometry_from_state((8, 8, 8), geometry)

    assert core.operator == PROJECTION_OPERATOR
    np.testing.assert_allclose(np.asarray(core.t_all[0, :3, 3]), [-1.5, 0.0, -0.25])
    np.testing.assert_allclose(np.asarray(core.t_all[1, :3, 3]), [2.0, 0.0, 0.5])
    np.testing.assert_allclose(
        np.asarray(core.t_all[1, :2, :2]), [[0.0, -1.0], [1.0, 0.0]], atol=1e-6
    )


def test_project_parallel_reference_applies_detector_roll() -> None:
    volume = jnp.zeros((7, 7, 7), dtype=jnp.float32)
    volume = volume.at[2, :, 4].set(1.0)
    volume = volume.at[5, :, 1].set(0.5)
    geometry = GeometryState.zeros(1)
    rolled_setup = geometry.setup.replace_parameter(
        "detector_roll_rad",
        geometry.setup.detector_roll_rad.with_value(0.20),
    )
    rolled_geometry = GeometryState(setup=rolled_setup, pose=geometry.pose)

    base = project_parallel_reference(volume, geometry)
    rolled = project_parallel_reference(volume, rolled_geometry)

    assert float(jnp.linalg.norm(rolled - base)) > 0.0
    assert rolled.shape == base.shape
    np.testing.assert_allclose(
        np.asarray(rolled),
        np.asarray(
            project_parallel_reference_arrays(
                volume,
                theta_rad=jnp.asarray([0.0], dtype=jnp.float32),
                dx_px=jnp.asarray([0.0], dtype=jnp.float32),
                dz_px=jnp.asarray([0.0], dtype=jnp.float32),
                detector_roll_rad=jnp.asarray(0.20, dtype=jnp.float32),
            )
        ),
        atol=1e-6,
    )


def test_project_parallel_reference_applies_axis_tilt() -> None:
    volume = jnp.zeros((7, 7, 7), dtype=jnp.float32)
    volume = volume.at[2, :, 4].set(1.0)
    volume = volume.at[5, :, 1].set(0.5)
    geometry = GeometryState.zeros(2)
    tilted_setup = geometry.setup.replace_parameter(
        "axis_rot_x_rad",
        geometry.setup.axis_rot_x_rad.with_value(0.18),
    )
    tilted_setup = tilted_setup.replace_parameter(
        "axis_rot_y_rad",
        geometry.setup.axis_rot_y_rad.with_value(-0.12),
    )
    tilted_geometry = GeometryState(setup=tilted_setup, pose=geometry.pose)

    base = project_parallel_reference(volume, geometry)
    tilted = project_parallel_reference(volume, tilted_geometry)

    assert float(jnp.linalg.norm(tilted - base)) > 0.0
    assert tilted.shape == base.shape
    core = core_projection_geometry_from_state((7, 7, 7), tilted_geometry)
    assert float(jnp.linalg.norm(core.t_all[:, :3, :3] - jnp.eye(3))) > 0.0


def test_core_projection_geometry_applies_alpha_beta_residual_pose() -> None:
    geometry = GeometryState.zeros(1)
    pose = geometry.pose.with_updates(
        alpha_rad=np.array([0.20], dtype=np.float64),
        beta_rad=np.array([-0.10], dtype=np.float64),
    )
    geometry = GeometryState(setup=geometry.setup, pose=pose)

    core = core_projection_geometry_from_state((7, 7, 7), geometry)

    ca, sa = np.cos(0.20), np.sin(0.20)
    cb, sb = np.cos(-0.10), np.sin(-0.10)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    np.testing.assert_allclose(np.asarray(core.t_all[0, :3, :3]), ry @ rx, atol=1e-6)
    provenance = core.provenance()
    assert np.isclose(float(cast("float", provenance["alpha_rad_max_abs"])), 0.20)
    assert np.isclose(float(cast("float", provenance["beta_rad_max_abs"])), 0.10)


def test_core_projection_geometry_matches_core_laminography_pose_convention() -> None:
    geometry = GeometryState.zeros(2)
    geometry = GeometryState(
        setup=geometry.setup,
        pose=geometry.pose.with_updates(
            theta_nominal_rad=np.array([0.0, np.pi / 3.0], dtype=np.float64),
        ),
        acquisition=AcquisitionParameters.parallel_laminography(
            tilt_rad=float(np.deg2rad(30.0)),
            tilt_about="x",
        ),
    )

    core = core_projection_geometry_from_state((7, 7, 7), geometry)
    reference = LaminographyGeometry(
        grid=Grid(nx=7, ny=7, nz=7, vx=1.0, vy=1.0, vz=1.0),
        detector=Detector(nu=7, nv=7, du=1.0, dv=1.0),
        thetas_deg=[0.0, 60.0],
        tilt_deg=30.0,
        tilt_about="x",
    )

    np.testing.assert_allclose(
        np.asarray(core.t_all[:, :3, :3]),
        np.asarray([reference.pose_for_view(0), reference.pose_for_view(1)])[:, :3, :3],
        atol=1e-6,
    )
    provenance = core.provenance()
    assert provenance["acquisition_model"] == "parallel_laminography"
    assert np.isclose(float(cast("float", provenance["laminography_tilt_rad"])), np.deg2rad(30.0))


def test_project_parallel_reference_applies_alpha_beta_pose() -> None:
    volume = jnp.zeros((7, 7, 7), dtype=jnp.float32)
    volume = volume.at[1, :, 5].set(1.0)
    volume = volume.at[5, :, 2].set(0.6)
    geometry = GeometryState.zeros(1)
    tilted_pose = geometry.pose.with_updates(
        alpha_rad=np.array([0.18], dtype=np.float64),
        beta_rad=np.array([-0.12], dtype=np.float64),
    )
    tilted_geometry = GeometryState(setup=geometry.setup, pose=tilted_pose)

    base = project_parallel_reference(volume, geometry)
    tilted = project_parallel_reference(volume, tilted_geometry)

    assert float(jnp.linalg.norm(tilted - base)) > 0.0
    np.testing.assert_allclose(
        np.asarray(tilted),
        np.asarray(
            project_parallel_reference_arrays(
                volume,
                theta_rad=jnp.asarray([0.0], dtype=jnp.float32),
                dx_px=jnp.asarray([0.0], dtype=jnp.float32),
                dz_px=jnp.asarray([0.0], dtype=jnp.float32),
                alpha_rad=jnp.asarray([0.18], dtype=jnp.float32),
                beta_rad=jnp.asarray([-0.12], dtype=jnp.float32),
            )
        ),
        atol=1e-6,
    )


def test_project_parallel_reference_changes_smoothly_with_theta() -> None:
    volume = jnp.zeros((5, 5, 5), dtype=jnp.float32)
    volume = volume.at[1, 2, 1].set(1.0)
    volume = volume.at[3, 1, 3].set(0.5)

    base = project_parallel_reference_arrays(
        volume,
        theta_rad=jnp.asarray([0.0], dtype=jnp.float32),
        dx_px=jnp.asarray([0.0], dtype=jnp.float32),
        dz_px=jnp.asarray([0.0], dtype=jnp.float32),
    )
    tilted = project_parallel_reference_arrays(
        volume,
        theta_rad=jnp.asarray([0.2], dtype=jnp.float32),
        dx_px=jnp.asarray([0.0], dtype=jnp.float32),
        dz_px=jnp.asarray([0.0], dtype=jnp.float32),
    )

    assert float(jnp.linalg.norm(tilted - base)) > 0.0


def test_project_parallel_reference_arrays_is_differentiable_for_theta() -> None:
    volume = jnp.zeros((5, 5, 5), dtype=jnp.float32)
    volume = volume.at[1, 2, 1].set(1.0)
    volume = volume.at[3, 1, 3].set(0.5)

    def pixel_value(theta_rad: jax.Array) -> jax.Array:
        projected = project_parallel_reference_arrays(
            volume,
            theta_rad=jnp.asarray([theta_rad], dtype=jnp.float32),
            dx_px=jnp.asarray([0.0], dtype=jnp.float32),
            dz_px=jnp.asarray([0.0], dtype=jnp.float32),
        )
        return projected[0, 1, 1]

    gradient = jax.grad(pixel_value)(jnp.asarray(0.2, dtype=jnp.float32))

    assert jnp.isfinite(gradient)
    assert abs(float(gradient)) > 1e-6


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


def test_residual_loss_l2_mode_uses_plain_squared_residual() -> None:
    predicted = jnp.array([2.0, 10.0, 4.0], dtype=jnp.float32)
    observed = jnp.zeros((3,), dtype=jnp.float32)
    mask = jnp.array([1.0, 0.0, 1.0], dtype=jnp.float32)

    result = residual_loss(predicted, observed, mask=mask, mode="l2")

    assert float(result.valid_count) == 2.0
    np.testing.assert_allclose(float(result.loss), 5.0)
    np.testing.assert_allclose(np.asarray(result.weights), np.ones((3,), dtype=np.float32))


def test_robust_residual_scale_uses_masked_mad() -> None:
    residual = jnp.array([0.0, 1.0, 2.0, 100.0], dtype=jnp.float32)
    mask = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)

    scale = robust_residual_scale(residual, mask=mask)

    np.testing.assert_allclose(float(scale), 1.4826, rtol=1e-6)


def _projection_centroid(projection: jax.Array) -> tuple[float, float]:
    image = jnp.asarray(projection, dtype=jnp.float32)
    mass = jnp.sum(image)
    rows = jnp.arange(image.shape[0], dtype=jnp.float32)
    cols = jnp.arange(image.shape[1], dtype=jnp.float32)
    row_center = jnp.sum(image * rows[:, None]) / mass
    col_center = jnp.sum(image * cols[None, :]) / mass
    return (float(row_center), float(col_center))
