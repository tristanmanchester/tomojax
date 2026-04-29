from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.pallas_projector import (
    PallasProjectorUnsupported,
    forward_project_view_T_pallas,
)
from tomojax.core.projector import forward_project_view_T, get_detector_grid_device


def _pose(theta_deg: float = 0.0, *, grid: Grid, detector: Detector) -> jnp.ndarray:
    geom = ParallelGeometry(grid=grid, detector=detector, thetas_deg=[theta_deg])
    return jnp.asarray(geom.pose_for_view(0), dtype=jnp.float32)


def _assert_matches_jax(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    det_grid=None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    oracle = forward_project_view_T(
        T,
        grid,
        detector,
        volume,
        step_size=step_size,
        n_steps=n_steps,
        det_grid=det_grid,
    )
    candidate = forward_project_view_T_pallas(
        T,
        grid,
        detector,
        volume,
        step_size=step_size,
        n_steps=n_steps,
        det_grid=det_grid,
        interpret=True,
    )
    assert candidate.shape == oracle.shape == (detector.nv, detector.nu)
    np.testing.assert_allclose(np.asarray(candidate), np.asarray(oracle), atol=atol, rtol=rtol)


def test_pallas_forward_project_uniform_volume_returns_path_length() -> None:
    grid = Grid(nx=16, ny=16, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=16, nv=16, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.ones((16, 16, 16), dtype=jnp.float32)

    projected = forward_project_view_T_pallas(T, grid, detector, volume, interpret=True)

    assert projected.shape == (16, 16)
    np.testing.assert_allclose(np.asarray(projected), 16.0, atol=1e-4, rtol=1e-4)


def test_pallas_forward_project_localized_center_voxel_matches_jax() -> None:
    grid = Grid(nx=5, ny=5, nz=5, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=9, nv=1, du=0.25, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.zeros((5, 5, 5), dtype=jnp.float32).at[2, 2, 2].set(1.0)

    _assert_matches_jax(T, grid, detector, volume)


def test_pallas_forward_project_localized_voxel_with_grid_center_matches_jax() -> None:
    grid = Grid(
        nx=5,
        ny=5,
        nz=5,
        vx=1.0,
        vy=1.0,
        vz=1.0,
        vol_center=(1.0, 0.0, 0.0),
    )
    detector = Detector(nu=9, nv=1, du=0.25, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.zeros((5, 5, 5), dtype=jnp.float32).at[2, 2, 2].set(1.0)

    _assert_matches_jax(T, grid, detector, volume)


def test_pallas_forward_project_non_cubic_rotated_case_matches_jax() -> None:
    grid = Grid(nx=64, ny=16, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=3, nv=3, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(90.0, grid=grid, detector=detector)
    volume = jnp.ones((64, 16, 16), dtype=jnp.float32)

    _assert_matches_jax(T, grid, detector, volume, atol=1e-3, rtol=1e-4)


def test_pallas_forward_project_handles_detector_tile_remainder() -> None:
    grid = Grid(nx=16, ny=16, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=18, nv=17, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(17.0, grid=grid, detector=detector)
    rng = np.random.default_rng(0)
    volume = jnp.asarray(np.abs(rng.normal(size=(16, 16, 16))).astype(np.float32))

    _assert_matches_jax(T, grid, detector, volume)


def test_pallas_forward_project_explicit_traversal_controls_match_jax() -> None:
    grid = Grid(nx=12, ny=10, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=7, nv=5, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(31.0, grid=grid, detector=detector)
    volume = jnp.arange(12 * 10 * 8, dtype=jnp.float32).reshape((12, 10, 8)) / 1000.0

    _assert_matches_jax(
        T,
        grid,
        detector,
        volume,
        step_size=0.5,
        n_steps=64,
        atol=1e-3,
        rtol=1e-4,
    )


def test_pallas_forward_project_accepts_canonical_detector_grid() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.ones((8, 8, 8), dtype=jnp.float32)
    det_grid = get_detector_grid_device(detector)

    _assert_matches_jax(T, grid, detector, volume, det_grid=det_grid)


@pytest.mark.parametrize("gather_dtype", ["bf16", "fp16"])
def test_pallas_forward_project_rejects_unsupported_gather_dtype(gather_dtype: str) -> None:
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=4, nv=4, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.ones((4, 4, 4), dtype=jnp.float32)

    with pytest.raises(PallasProjectorUnsupported, match="fp32 only"):
        forward_project_view_T_pallas(
            T,
            grid,
            detector,
            volume,
            gather_dtype=gather_dtype,
            interpret=True,
        )


def test_pallas_forward_project_rejects_noncanonical_detector_grid() -> None:
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=4, nv=4, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.ones((4, 4, 4), dtype=jnp.float32)
    Xr, Zr = get_detector_grid_device(detector)
    shifted_grid = (Xr + jnp.float32(0.125), Zr)

    with pytest.raises(PallasProjectorUnsupported, match="get_detector_grid_device"):
        forward_project_view_T_pallas(
            T,
            grid,
            detector,
            volume,
            det_grid=shifted_grid,
            interpret=True,
        )
