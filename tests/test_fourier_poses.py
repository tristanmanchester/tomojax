"""Translation poses in Fourier inversion, using independent analytic data."""

import jax
import numpy as np
import pytest

import tomojax as tj
from tomojax.geometry import grid_volume_origin
from tomojax.recon import FourierConfig


@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("nx", [32, 33])
@pytest.mark.parametrize("angle_dtype", [np.float32, np.float64])
def test_fourier_translation_poses_recover_analytic_gaussian(backend, nx, angle_dtype):
    if backend == "cupy" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid = tj.Grid(nx, 27, 9, 1.4, 1.2, 0.8, vol_center=(0.4, -0.3, 0.1))
    detector = tj.Detector(65, 31, 1.0, 0.7, center=(-2.3, -0.4))
    angles = np.arange(-270, 90, 3, dtype=angle_dtype)
    theta = np.deg2rad(angles.astype(np.float64))
    poses = np.zeros((len(angles), 6), dtype=np.float32)
    poses[:, 3] = 3.3 * np.sin(2.7 * theta)
    poses[:, 4] = 2.1 * np.cos(1.3 * theta)
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) + detector.center[0]
    v = (np.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv + detector.center[1]
    sigma = 3.1
    centre_u = 1.2 * np.cos(theta) + 2.0 * np.sin(theta)
    data = (
        np.sqrt(2 * np.pi)
        * sigma
        * np.exp(
            -0.5
            * (
                (u[None, None, :] - centre_u[:, None, None] - poses[:, 3, None, None]) ** 2
                + (v[None, :, None] - 0.7 - poses[:, 4, None, None]) ** 2
            )
            / sigma**2
        )
    ).astype(np.float32)
    scan = tj.Scan(data, tj.ParallelGeometry(grid, detector, angles)).with_poses(poses)
    result = tj.reconstruct(
        scan, "fourier", config=FourierConfig(backend=backend, slices_per_batch=4)
    )
    origin = grid_volume_origin(grid)
    z, y, x = np.meshgrid(
        origin[2] + np.arange(grid.nz) * grid.vz,
        origin[1] + np.arange(grid.ny) * grid.vy,
        origin[0] + np.arange(grid.nx) * grid.vx,
        indexing="ij",
    )
    truth = np.exp(-0.5 * ((x - 1.2) ** 2 + (y + 2) ** 2 + (z - 0.7) ** 2) / sigma**2)
    assert np.linalg.norm(result.volume - truth) / np.linalg.norm(truth) < 0.007
    assert result.scan.geometry is scan.geometry


def test_fourier_rejects_rotational_pose_corrections_before_writing():
    grid, detector = tj.Grid(8, 8, 2, 1, 1, 1), tj.Detector(12, 2, 1, 1)
    scan = tj.Scan(
        np.ones((8, 2, 12), np.float32), tj.ParallelGeometry(grid, detector, np.arange(8) * 20)
    )
    poses = np.zeros((8, 6), np.float32)
    poses[1, 0] = 0.01
    out = np.full((2, 8, 8), 123, np.float32)
    with pytest.raises(ValueError, match="translation-only"):
        tj.reconstruct(
            scan.with_poses(poses), "fourier", config=FourierConfig(backend="numpy"), out=out
        )
    np.testing.assert_array_equal(out, 123)
