import sys
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tomojax.core.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.projector import (
    backproject_view,
    backproject_view_T,
    forward_project_view,
    get_detector_grid_device,
    sum_backproject_views_T,
)
from tomojax.core.operators import adjoint_test_once


if sys.version_info < (3, 8):
    pytest.skip("Requires Python 3.8+ for package code", allow_module_level=True)


def make_aligned_case(nx=16, ny=16, nz=16):
    grid = Grid(nx=nx, ny=ny, nz=nz, vx=1.0, vy=1.0, vz=1.0)
    # Detector aligned and covering FOV exactly
    det = Detector(nu=nx, nv=nz, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    thetas = [0.0]
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
    # Volume of ones -> line integral equals ny * vy
    vol = jnp.ones((nx, ny, nz), dtype=jnp.float32)
    return grid, det, geom, vol


def test_forward_project_uniform_volume_returns_path_length():
    grid, det, geom, vol = make_aligned_case(16, 16, 16)
    proj = forward_project_view(geom, grid, det, vol, view_index=0)
    expected = grid.ny * grid.vy
    assert np.allclose(np.asarray(proj), expected, atol=1e-4)


def test_forward_project_rejects_bad_volume_shape_with_expected_and_actual():
    grid, det, geom, _ = make_aligned_case(8, 8, 8)
    bad_vol = jnp.zeros((grid.nx, grid.ny), dtype=jnp.float32)

    with pytest.raises(
        ValueError,
        match=r"expected .*=\(8, 8, 8\).*actual \(8, 8\).*Likely fix",
    ):
        forward_project_view(geom, grid, det, bad_vol, view_index=0)


def test_backproject_rejects_bad_detector_image_shape_with_expected_and_actual():
    grid, det, _, _ = make_aligned_case(8, 8, 8)
    bad_image = jnp.zeros((det.nv + 1, det.nu), dtype=jnp.float32)

    with pytest.raises(
        ValueError,
        match=r"expected .*=\(8, 8\).*actual \(9, 8\).*Likely fix",
    ):
        backproject_view_T(jnp.eye(4, dtype=jnp.float32), grid, det, bad_image)


@pytest.mark.parametrize("gather_dtype", ["fp32", "bf16", "fp16"])
def test_adjoint_small_case(gather_dtype: str):
    grid, det, geom, vol = make_aligned_case(8, 8, 8)
    y_like = jnp.ones((det.nv, det.nu), dtype=jnp.float32)
    rel = adjoint_test_once(
        geom,
        grid,
        det,
        vol,
        y_like,
        view_index=0,
        gather_dtype=gather_dtype,
    )
    assert rel < 5e-3


def test_forward_project_non_cubic_rotated_volume_uses_full_ray_extent():
    grid = Grid(nx=64, ny=16, nz=16, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=1, nv=1, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=[0.0, 90.0])
    vol = jnp.ones((64, 16, 16), dtype=jnp.float32)

    proj_0 = np.asarray(forward_project_view(geom, grid, det, vol, view_index=0))
    proj_90 = np.asarray(forward_project_view(geom, grid, det, vol, view_index=1))

    assert proj_0.shape == (1, 1)
    assert proj_90.shape == (1, 1)
    assert proj_0[0, 0] == pytest.approx(16.0, abs=1e-4)
    assert proj_90[0, 0] == pytest.approx(64.0, abs=1e-3)


def test_forward_project_localized_voxel_uses_center_indexed_origin() -> None:
    grid = Grid(nx=5, ny=5, nz=5, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=9, nv=1, du=0.25, dv=1.0, det_center=(0.0, 0.0))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=[0.0])
    vol = jnp.zeros((5, 5, 5), dtype=jnp.float32).at[2, 2, 2].set(1.0)

    proj = np.asarray(forward_project_view(geom, grid, det, vol, view_index=0))[0]
    u = (np.arange(det.nu, dtype=np.float32) - (det.nu / 2.0 - 0.5)) * det.du

    # The default centred grid places voxel (2, 2, 2) at x = z = 0.0.
    assert u[np.argmax(proj)] == pytest.approx(0.0, abs=1e-6)
    assert proj[np.argmax(proj)] == pytest.approx(1.0, abs=1e-6)


def _vjp_backproject(
    geom,
    grid: Grid,
    det: Detector,
    image: jnp.ndarray,
    *,
    view_index: int = 0,
    gather_dtype: str = "fp32",
) -> jnp.ndarray:
    zero_vol = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)

    def fwd(vol):
        return forward_project_view(
            geom,
            grid,
            det,
            vol,
            view_index=view_index,
            gather_dtype=gather_dtype,
        ).ravel()

    _, vjp = jax.vjp(fwd, zero_vol)
    return vjp(image.ravel().astype(jnp.float32))[0]


@pytest.mark.parametrize("gather_dtype", ["fp32", "bf16", "fp16"])
def test_backproject_matches_vjp_parallel(gather_dtype: str):
    grid = Grid(nx=6, ny=7, nz=5, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=6, nv=5, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=[0.0])
    image = jax.random.normal(jax.random.PRNGKey(0), (det.nv, det.nu), dtype=jnp.float32)

    explicit = backproject_view(
        geom,
        grid,
        det,
        image,
        view_index=0,
        gather_dtype=gather_dtype,
    )
    oracle = _vjp_backproject(
        geom,
        grid,
        det,
        image,
        view_index=0,
        gather_dtype=gather_dtype,
    )

    assert np.allclose(np.asarray(explicit), np.asarray(oracle), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("gather_dtype", ["fp32", "bf16", "fp16"])
def test_backproject_matches_vjp_lamino(gather_dtype: str):
    grid = Grid(nx=6, ny=6, nz=5, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=6, nv=5, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    geom = LaminographyGeometry(
        grid=grid,
        detector=det,
        thetas_deg=[17.5],
        tilt_deg=30.0,
        tilt_about="x",
    )
    image = jax.random.normal(jax.random.PRNGKey(1), (det.nv, det.nu), dtype=jnp.float32)

    explicit = backproject_view(
        geom,
        grid,
        det,
        image,
        view_index=0,
        gather_dtype=gather_dtype,
    )
    oracle = _vjp_backproject(
        geom,
        grid,
        det,
        image,
        view_index=0,
        gather_dtype=gather_dtype,
    )

    assert np.allclose(np.asarray(explicit), np.asarray(oracle), atol=1e-5, rtol=1e-5)


def test_sum_backproject_views_matches_per_view_sum():
    grid = Grid(nx=6, ny=7, nz=5, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=6, nv=5, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=[0.0, 35.0])
    images = jax.random.normal(jax.random.PRNGKey(2), (2, det.nv, det.nu), dtype=jnp.float32)
    poses = stack_view_poses(geom, images.shape[0])
    det_grid = get_detector_grid_device(det)

    summed = sum_backproject_views_T(poses, grid, det, images, det_grid=det_grid)
    expected = jnp.stack(
        [
            backproject_view_T(poses[i], grid, det, images[i], det_grid=det_grid)
            for i in range(images.shape[0])
        ],
        axis=0,
    ).sum(axis=0)

    assert np.allclose(np.asarray(summed), np.asarray(expected), atol=1e-5, rtol=1e-5)
