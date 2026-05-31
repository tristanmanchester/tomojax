from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

# check-public-imports: allow-private
from tomojax.core.projector import _build_detector_grid, forward_project_view_T
from tomojax.forward import (
    ProjectionArrayGeometryInput,
    project_parallel_reference_from_input,
)

# check-public-imports: allow-private
from tomojax.forward._projector import core_projection_geometry_from_input
from tomojax.geometry import Detector, Grid


def test_detector_grid_is_flattened_centered_and_read_only() -> None:
    detector = Detector(nu=3, nv=2, du=2.0, dv=4.0, det_center=(10.0, -5.0))

    x_grid, z_grid = _build_detector_grid(detector)

    assert x_grid.shape == (6,)
    assert z_grid.shape == (6,)
    np.testing.assert_allclose(x_grid, [8.0, 10.0, 12.0, 8.0, 10.0, 12.0])
    np.testing.assert_allclose(z_grid, [-7.0, -7.0, -7.0, -3.0, -3.0, -3.0])
    assert not x_grid.flags.writeable
    assert not z_grid.flags.writeable


def test_forward_project_view_rejects_invalid_traversal_controls() -> None:
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)
    pose = jnp.eye(4, dtype=jnp.float32)
    volume = jnp.ones((2, 2, 2), dtype=jnp.float32)

    with pytest.raises(ValueError, match="step_size must be finite and > 0"):
        forward_project_view_T(pose, grid, detector, volume, step_size=0.0)

    with pytest.raises(ValueError, match="n_steps must be a positive integer"):
        forward_project_view_T(pose, grid, detector, volume, n_steps=True)


def test_forward_project_view_rejects_mismatched_detector_grid() -> None:
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)
    pose = jnp.eye(4, dtype=jnp.float32)
    volume = jnp.ones((2, 2, 2), dtype=jnp.float32)
    det_grid = (
        jnp.zeros((3,), dtype=jnp.float32),
        jnp.zeros((4,), dtype=jnp.float32),
    )

    with pytest.raises(ValueError, match=r"det_grid\[0\].*expected .*4"):
        forward_project_view_T(pose, grid, detector, volume, det_grid=det_grid)


def test_forward_adapter_rejects_mismatched_pose_arrays() -> None:
    volume = jnp.ones((2, 2, 2), dtype=jnp.float32)

    with pytest.raises(ValueError, match="theta_rad, dx_px, and dz_px must have matching"):
        project_parallel_reference_from_input(
            volume,
            ProjectionArrayGeometryInput(
                theta_rad=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
                dx_px=jnp.asarray([0.0], dtype=jnp.float32),
                dz_px=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
            ),
        )

    with pytest.raises(ValueError, match="alpha_rad must be scalar or match theta_rad shape"):
        project_parallel_reference_from_input(
            volume,
            ProjectionArrayGeometryInput(
                theta_rad=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
                dx_px=jnp.zeros((2,), dtype=jnp.float32),
                dz_px=jnp.zeros((2,), dtype=jnp.float32),
                alpha_rad=jnp.asarray([0.0], dtype=jnp.float32),
            ),
        )


def test_core_projection_geometry_uses_explicit_detector_shape() -> None:
    core = core_projection_geometry_from_input(
        (2, 3, 4),
        ProjectionArrayGeometryInput(
            theta_rad=jnp.asarray([0.0], dtype=jnp.float32),
            dx_px=jnp.asarray([0.0], dtype=jnp.float32),
            dz_px=jnp.asarray([0.0], dtype=jnp.float32),
            detector_shape=(5, 6),
        ),
        checkpoint_projector=False,
        projector_unroll=2,
        n_steps=7,
    )

    assert core.grid == Grid(nx=2, ny=3, nz=4, vx=1.0, vy=1.0, vz=1.0)
    assert core.detector == Detector(nu=6, nv=5, du=1.0, dv=1.0)
    assert core.t_all.shape == (1, 4, 4)
    assert core.det_grid[0].shape == (30,)
    assert core.det_grid[1].shape == (30,)
    assert core.provenance()["n_steps"] == 7
    assert core.provenance()["checkpoint_projector"] is False
    assert core.provenance()["projector_unroll"] == 2
