from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from tomojax.core.geometry.base import Grid, Detector
from tomojax.cli.align import _resolve_recon_grid_and_mask
from tomojax.recon.fbp import _fft_filter_rows
from tomojax.recon.filters import get_filter_np
from tomojax.utils.fov import (
    compute_roi,
    grid_from_detector_fov,
    grid_from_detector_fov_cube,
    grid_from_detector_fov_slices,
)


def _full_fft_filter_rows(rows: np.ndarray, du: float, filter_name: str) -> np.ndarray:
    nu = rows.shape[-1]
    H = get_filter_np(filter_name, nu, du).astype(rows.dtype, copy=False)
    F = np.fft.fft(rows, axis=-1)
    out = np.fft.ifft(F * H, axis=-1)
    return out.real.astype(rows.dtype, copy=False)


def test_fbp_rfft_filter_matches_full_fft_reference() -> None:
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(7, 33)).astype(np.float32)

    for filter_name in ("ramp", "hann", "shepp-logan"):
        got = np.asarray(_fft_filter_rows(jnp.asarray(rows), du=1.0, filter_name=filter_name))
        want = _full_fft_filter_rows(rows, du=1.0, filter_name=filter_name)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_compute_roi_and_bbox_crop_limit_y_axis() -> None:
    grid = Grid(nx=10, ny=10, nz=10, vx=1.0, vy=10.0, vz=1.0)
    detector = Detector(nu=11, nv=101, du=1.0, dv=1.0)

    info = compute_roi(grid, detector)
    assert info.r_u == 5.0
    assert info.nx_roi == 10
    assert info.ny_roi == 2
    assert info.nz_roi == 10

    cropped = grid_from_detector_fov(grid, detector)
    assert (cropped.nx, cropped.ny, cropped.nz) == (10, 2, 10)

def test_lamino_bbox_keeps_full_y_extent() -> None:
    grid = Grid(nx=10, ny=10, nz=10, vx=1.0, vy=10.0, vz=1.0)
    detector = Detector(nu=11, nv=101, du=1.0, dv=1.0)

    info = compute_roi(grid, detector, crop_y_to_u=False)
    assert info.nx_roi == 10
    assert info.ny_roi == 10
    assert info.nz_roi == 10

    cropped = grid_from_detector_fov(grid, detector, crop_y_to_u=False)
    assert (cropped.nx, cropped.ny, cropped.nz) == (10, 10, 10)


def test_grid_from_detector_fov_slices_preserves_preferred_parity() -> None:
    grid = Grid(nx=201, ny=200, nz=100, vx=1.0, vy=1.2, vz=1.0)
    detector = Detector(nu=100, nv=100, du=1.0, dv=1.0)

    result = grid_from_detector_fov_slices(grid, detector)
    assert (result.nx, result.ny, result.nz) == (81, 81, 100)
    assert result.nx % 2 == grid.nx % 2


def test_grid_from_detector_fov_cube_avoids_min_parity_leakage() -> None:
    grid = Grid(nx=200, ny=201, nz=198, vx=1.0, vy=1.2, vz=1.0)
    detector = Detector(nu=80, nv=80, du=1.0, dv=1.0)

    result = grid_from_detector_fov_cube(grid, detector)
    assert (result.nx, result.ny, result.nz) == (64, 64, 64)
    assert result.nx % 2 == grid.nx % 2
    assert result.nz % 2 == grid.nz % 2

def test_align_auto_roi_uses_bbox_for_lamino() -> None:
    grid = Grid(nx=10, ny=10, nz=20, vx=1.0, vy=10.0, vz=1.0)
    detector = Detector(nu=11, nv=15, du=1.0, dv=1.0)

    recon_grid, apply_cyl_mask = _resolve_recon_grid_and_mask(
        grid,
        detector,
        is_parallel=False,
        roi_mode="auto",
        grid_override=None,
    )

    assert (recon_grid.nx, recon_grid.ny, recon_grid.nz) == (10, 10, 15)
    assert not apply_cyl_mask
