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
    center_u = 1.2 * np.cos(theta) + 2.0 * np.sin(theta)
    data = (
        np.sqrt(2 * np.pi)
        * sigma
        * np.exp(
            -0.5
            * (
                (u[None, None, :] - center_u[:, None, None] - poses[:, 3, None, None]) ** 2
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


@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def test_fourier_resamples_a_rolled_parallel_detector(backend):
    if backend == "cupy" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid = tj.Grid(32, 32, 9, 1.0, 1.0, 0.8)
    detector = tj.Detector(64, 41, 1.0, 0.7, center=(-0.6, 0.3))
    angles = np.arange(0, 180, 2.0)
    theta = np.deg2rad(angles)
    roll = np.deg2rad(4.0)
    # Rolled pixel offsets (x', z') sit at R(roll) @ (x', z') from the detector center.
    xr = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du
    zr = (np.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv
    u = detector.center[0] + np.cos(roll) * xr[None, :] - np.sin(roll) * zr[:, None]
    v = detector.center[1] + np.sin(roll) * xr[None, :] + np.cos(roll) * zr[:, None]
    sigma = 3.1
    center_u = 1.2 * np.cos(theta) + 2.0 * np.sin(theta)
    data = (
        np.sqrt(2 * np.pi)
        * sigma
        * np.exp(-0.5 * ((u - center_u[:, None, None]) ** 2 + (v - 0.7) ** 2) / sigma**2)
    ).astype(np.float32)
    geometry = tj.ParallelGeometry(grid, detector, angles, detector_roll=4.0)
    result = tj.reconstruct(
        tj.Scan(data, geometry),
        "fourier",
        config=FourierConfig(backend=backend, slices_per_batch=4),
    )
    origin = grid_volume_origin(grid)
    z, y, x = np.meshgrid(
        origin[2] + np.arange(grid.nz) * grid.vz,
        origin[1] + np.arange(grid.ny) * grid.vy,
        origin[0] + np.arange(grid.nx) * grid.vx,
        indexing="ij",
    )
    truth = np.exp(-0.5 * ((x - 1.2) ** 2 + (y + 2) ** 2 + (z - 0.7) ** 2) / sigma**2)
    assert np.linalg.norm(result.volume - truth) / np.linalg.norm(truth) < 0.012


def test_fourier_rejects_out_of_plane_pose_corrections_before_writing():
    grid, detector = tj.Grid(8, 8, 2, 1, 1, 1), tj.Detector(12, 2, 1, 1)
    scan = tj.Scan(
        np.ones((8, 2, 12), np.float32), tj.ParallelGeometry(grid, detector, np.arange(8) * 20)
    )
    poses = np.zeros((8, 6), np.float32)
    poses[1, 0] = 0.01
    out = np.full((2, 8, 8), 123, np.float32)
    with pytest.raises(ValueError, match="rotations about z"):
        tj.reconstruct(
            scan.with_poses(poses), "fourier", config=FourierConfig(backend="numpy"), out=out
        )
    np.testing.assert_array_equal(out, 123)


def test_fourier_requires_rigid_axial_poses():
    class Sheared(tj.ParallelGeometry):
        def pose_matrix(self, i):
            pose = np.array(super().pose_matrix(i))
            pose[0, 2] = 0.1
            return pose

    grid, detector = tj.Grid(4, 4, 2, 1, 1, 1), tj.Detector(8, 3, 1, 1)
    geometry = Sheared(grid, detector, np.arange(8) * 20)
    scan = tj.Scan(np.ones((8, 3, 8), np.float32), geometry)
    with pytest.raises(ValueError, match="rotations about z"):
        tj.reconstruct(scan, "fourier")


def test_preflight_reads_bounded_view_and_detector_row_batches():
    from tomojax._workflow.planning import resolve_reconstruction

    reads = []

    class Recorded(np.ndarray):
        def __getitem__(self, key):
            batch = super().__getitem__(key)
            reads.append(batch.size)
            return batch

    grid = tj.Grid(3, 3, 17, 1, 1, 1)
    detector = tj.Detector(7, 101, 1, 1)
    geom = tj.ParallelGeometry(grid, detector, np.arange(73) * 2.0)
    data = np.full((73, 101, 7), np.nan, np.float32)
    data[:, 42:59] = 1
    resolve_reconstruction(tj.Scan(data.view(Recorded), geom), "fourier")
    assert max(reads) < 73 * 7 * 17
    assert sum(reads) == 73 * 7 * 17


def test_preflight_uses_each_views_translated_rows_on_a_device():
    from tomojax._workflow.planning import resolve_reconstruction

    grid = tj.Grid(5, 5, 1, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(7, 9, 1, 1), [130, 25, 80])
    poses = np.zeros((3, 6), np.float32)
    poses[:, 4] = [2.5, 0.5, -1.5]
    measured = np.full((3, 9, 7), np.nan, np.float32)
    for view, lower in enumerate((6, 4, 2)):
        measured[view, lower : lower + 2] = 1
    scan = tj.Scan(jax.device_put(measured), geometry).with_poses(poses)
    resolve_reconstruction(scan, "fourier")
    measured[0, 7, 2] = np.nan
    with pytest.raises(tj.InvalidParameter) as err:
        resolve_reconstruction(
            tj.Scan(jax.device_put(measured), geometry).with_poses(poses), "fourier"
        )
    assert err.value.path == "source"
