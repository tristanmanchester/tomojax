from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.calibration.axis_geometry import (
    axis_pose_stack,
    axis_unit_from_rotations,
    default_active_axis_dofs,
    nominal_axis_unit_from_inputs,
)
from tomojax.align.geometry_blocks import GeometryCalibrationState
from tomojax.align.pipeline import AlignConfig
from tomojax.core.geometry import Detector, Grid, LaminographyGeometry, RotationAxisGeometry


def test_laminography_default_axis_dof_matches_tilt_plane():
    geometry_inputs = {
        "geometry_type": "lamino",
        "tilt_deg": 30.0,
        "tilt_about": "x",
    }

    assert default_active_axis_dofs(geometry_inputs) == ("axis_rot_x_deg",)


def test_align_config_tilt_alias_resolves_against_laminography_tilt_plane():
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0)
    geometry = LaminographyGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=[0.0, 45.0],
        tilt_deg=30.0,
        tilt_about="z",
    )
    cfg = AlignConfig(geometry_dofs=("tilt_deg",))

    state = GeometryCalibrationState.from_geometry(
        geometry,
        active_geometry_dofs=cfg.geometry_dofs,
    )

    assert cfg.geometry_dofs == ("tilt_deg",)
    assert state.active_geometry_dofs == ("axis_rot_y_deg",)


def test_axis_rotation_x_adjusts_laminography_tilt_angle():
    nominal = nominal_axis_unit_from_inputs(
        {"geometry_type": "lamino", "tilt_deg": 30.0, "tilt_about": "x"}
    )

    axis = axis_unit_from_rotations(
        nominal,
        axis_rot_x_deg=4.4,
        axis_rot_y_deg=0.0,
    )

    expected = np.asarray([0.0, np.sin(np.deg2rad(34.4)), np.cos(np.deg2rad(34.4))])
    np.testing.assert_allclose(np.asarray(axis), expected, atol=1e-6)


def test_axis_pose_stack_is_jax_differentiable():
    nominal = jnp.asarray([0.0, np.sin(np.deg2rad(30.0)), np.cos(np.deg2rad(30.0))])
    thetas = jnp.asarray([0.0, 45.0], dtype=jnp.float32)

    def summed_pose(delta_x: jnp.ndarray) -> jnp.ndarray:
        axis = axis_unit_from_rotations(nominal, axis_rot_x_deg=delta_x, axis_rot_y_deg=0.0)
        return jnp.sum(axis_pose_stack(thetas, axis))

    value, tangent = jax.jvp(summed_pose, (jnp.asarray(1.0),), (jnp.asarray(0.1),))

    assert jnp.isfinite(value)
    assert jnp.isfinite(tangent)


def test_parallel_axis_pose_stack_has_local_sensitivity_at_nominal_axis():
    nominal = jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32)
    thetas = jnp.asarray([0.0, 45.0, 90.0], dtype=jnp.float32)

    def flattened_pose(values: jnp.ndarray) -> jnp.ndarray:
        axis = axis_unit_from_rotations(
            nominal,
            axis_rot_x_deg=values[0],
            axis_rot_y_deg=values[1],
        )
        return axis_pose_stack(thetas, axis).reshape(-1)

    jacobian = jax.jacfwd(flattened_pose)(jnp.asarray([0.0, 0.0], dtype=jnp.float32))

    assert float(jnp.linalg.norm(jacobian)) > 1e-4


def test_rotation_axis_geometry_matches_lamino_for_nominal_axis():
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0)
    thetas = [0.0, 30.0]
    lamino = LaminographyGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=thetas,
        tilt_deg=30.0,
        tilt_about="x",
    )
    arbitrary_axis = RotationAxisGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=thetas,
        axis_unit_lab=lamino._axis_unit(),
    )

    for i in range(len(thetas)):
        np.testing.assert_allclose(
            np.asarray(arbitrary_axis.pose_for_view(i)),
            np.asarray(lamino.pose_for_view(i)),
            atol=1e-7,
        )


def test_rotation_axis_geometry_rejects_zero_axis():
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0)

    with pytest.raises(ValueError, match="axis_unit_lab"):
        RotationAxisGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=[0.0],
            axis_unit_lab=(0.0, 0.0, 0.0),
        )
