"""Native detector registration and its public translation workflow."""

import numpy as np
import pytest
from scipy import ndimage


def test_vertical_registration_retains_subpixel_texture():
    # check-public-imports: allow-private
    from tomojax.alignment._projection_matching import _vertical_shifts

    rng = np.random.default_rng(71)
    nu, nv, count = 128, 96, 17
    u, v = np.arange(nu), np.arange(nv)
    model = sum(np.exp(-(((u - c) / w) ** 2)) for c, w in [(31, 4), (62, 6), (87, 3)])
    profile = sum(np.exp(-(((v - c) / w) ** 2)) for c, w in [(23, 5), (45, 7), (73, 4)])
    shifts = rng.uniform(-1, 1, count)
    data = np.stack([np.interp(v - s, v, profile)[:, None] * model for s in shifts])
    vertical = _vertical_shifts(data)
    np.testing.assert_allclose(vertical, shifts - shifts.mean(), atol=0.025)


@pytest.mark.gpu
def test_public_matching_recovers_motion_without_resampling_input():
    import tomojax as tj
    from tomojax.alignment import ProjectionMatchingConfig

    n, nu, nv = 48, 64, 48
    x, y, z = np.meshgrid(np.arange(nu), np.arange(nu), np.arange(nv), indexing="ij")
    volume = np.zeros((nu, nu, nv), np.float32)
    for cx, cy, cz, radius in [(22, 27, 16, 5), (39, 26, 28, 7), (31, 40, 32, 4)]:
        volume += np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * radius**2)).astype(
            np.float32
        )
    geometry = tj.ParallelGeometry(
        tj.Grid(nu, nu, nv, 0.7, 0.7, 1.2),
        tj.Detector(nu, nv, 0.7, 1.2),
        np.linspace(0, 180, n, endpoint=False),
    )
    measured = np.asarray(tj.project(geometry, volume))
    t = np.arange(n)
    truth = np.column_stack([1.5 * np.sin(1.9 * t), 1.1 * np.cos(0.7 * t)])
    moved = np.stack(
        [
            ndimage.shift(p, (s[1], s[0]), order=3, mode="constant")
            for p, s in zip(measured, truth, strict=True)
        ]
    ).astype(np.float32)
    scan = tj.Scan(moved, geometry)
    result = tj.align(scan, mode="pose", config=ProjectionMatchingConfig())
    assert result.scan.projections is moved
    assert result.volume.shape == volume.shape
    # Compare after removing only the common object translation gauge.
    from tomojax.geometry import stack_view_poses

    pitch = np.array([geometry.detector.du, geometry.detector.dv])
    error = result.poses[:, 3:5] - truth * pitch
    frame = np.asarray(stack_view_poses(geometry, n))[:, [0, 2], :3].reshape(-1, 3)
    common = np.linalg.lstsq(frame, error.reshape(-1), rcond=None)[0]
    residual = (error - (frame @ common).reshape(-1, 2)) / pitch
    assert np.sqrt(np.mean(np.sum(residual**2, axis=1))) < 0.05
    np.testing.assert_array_equal(result.scan.poses, result.poses)
    np.testing.assert_array_equal(result.volume, tj.reconstruct(result.scan, "fourier").volume)
