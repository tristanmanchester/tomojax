"""Native detector registration and its public translation workflow."""

import numpy as np
import pytest
from scipy import ndimage
from scipy.signal import lfilter

# check-public-imports: allow-private
from tomojax.alignment import _profile_likelihood as likelihood, _projection_matching as matching


def test_vertical_registration_retains_subpixel_texture():
    rng = np.random.default_rng(71)
    nu, nv, count = 128, 96, 17
    u, v = np.arange(nu), np.arange(nv)
    model = sum(np.exp(-(((u - c) / w) ** 2)) for c, w in [(31, 4), (62, 6), (87, 3)])
    profile = sum(np.exp(-(((v - c) / w) ** 2)) for c, w in [(23, 5), (45, 7), (73, 4)])
    shifts = rng.uniform(-1, 1, count)
    data = np.stack([np.interp(v - s, v, profile)[:, None] * model for s in shifts])
    profiles = data.sum(axis=2)
    vertical = likelihood.fit_profile_likelihood(profiles, matching._profile_seed(profiles)).shifts
    np.testing.assert_allclose(vertical, shifts - shifts.mean(), atol=0.025)


@pytest.mark.gpu
def test_public_matching_recovers_motion_without_resampling_input():
    import tomojax as tj
    from tomojax.alignment import ProjectionMatchingConfig

    n, nu, nv = 48, 64, 48
    z, y, x = np.meshgrid(np.arange(nv), np.arange(nu), np.arange(nu), indexing="ij")
    volume = np.zeros((nv, nu, nu), np.float32)
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
    uncertainty = result.info["vertical_uncertainty"]
    standard = np.asarray(uncertainty["standard_deviation"])
    assert standard.shape == (n,) and np.all(np.isfinite(standard)) and np.all(standard > 0)
    assert uncertainty["translation_frame"] == "detector"
    assert uncertainty["units"] == "geometry_length"
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


@pytest.mark.gpu
def test_large_motion_from_independent_gaussian_line_integrals():
    import tomojax as tj
    from tomojax.alignment import ProjectionMatchingConfig
    from tomojax.geometry import stack_view_poses

    n, nu, nv = 48, 96, 40
    angles = np.linspace(0, 180, n, endpoint=False)
    u, v = np.arange(nu) - (nu - 1) / 2, np.arange(nv) - (nv - 1) / 2
    truth = np.column_stack([6 * np.sin(np.arange(n) * 1.9), 2 * np.cos(np.arange(n) * 0.7)])
    data = np.zeros((n, nv, nu))
    # Analytic Gaussian line integrals, independent of the voxel projector.
    for x, y, amplitude, radius in [(-14, -5, 1, 4), (8, 5, 1.4, 5), (1, -10, 0.8, 3)]:
        centre = x * np.cos(np.deg2rad(angles)) + y * np.sin(np.deg2rad(angles))
        du = u[None, None, :] - truth[:, 0, None, None] - centre[:, None, None]
        dv = v[None, :, None] - truth[:, 1, None, None]
        data += amplitude * np.exp(-(du**2 + dv**2) / (2 * radius**2))
    geometry = tj.ParallelGeometry(
        tj.Grid(nu, nu, nv, 0.7, 0.7, 1.2), tj.Detector(nu, nv, 0.7, 1.2), angles
    )
    scan = tj.Scan(data.astype(np.float32), geometry)
    result = tj.align(scan, mode="pose", config=ProjectionMatchingConfig())
    pitch = np.array([geometry.detector.du, geometry.detector.dv])
    error = np.asarray(result.poses)[:, 3:5] - truth * pitch
    frame = np.asarray(stack_view_poses(geometry, n))[:, [0, 2], :3].reshape(-1, 3)
    error -= (frame @ np.linalg.lstsq(frame, error.ravel(), rcond=None)[0]).reshape(-1, 2)
    assert np.sqrt(np.mean(np.sum((error / pitch) ** 2, axis=1))) < 0.01
    assert result.scan.projections is scan.projections


@pytest.mark.parametrize("motion", [False, True])
def test_correlated_profile_likelihood_is_calibrated(motion):
    rng = np.random.default_rng(72401)
    count, width, rho = 160, 192, 0.7
    axis = np.arange(width, dtype=float)
    frequency = np.array([1 / 23, 1 / 37, 1 / 57])
    phase = rng.uniform(-np.pi, np.pi, 3)
    truth = 2 * np.sin(np.arange(count) * 2 * np.pi / count) + rng.normal(0, 0.4, count)
    truth = (truth - truth.mean()) if motion else np.zeros(count)
    signal = np.sin((axis[None, :, None] - truth[:, None, None]) * 2 * np.pi * frequency + phase)
    gain = rng.uniform(0.8, 1.2, count)
    background = rng.normal(0, 0.3, (count, 2))
    noise = lfilter([1], [1, -rho], rng.normal(size=(count, width)), axis=1)
    data = gain[:, None] * (signal @ np.array([0.7, 1, 0.4]))
    data += background[:, 0, None] + background[:, 1, None] * axis / width
    amplitude = np.where(np.arange(count) < count // 2, 0.08, 0.24) if motion else 0.15
    data += np.asarray(amplitude).reshape(-1, 1) * np.sqrt(1 - rho**2) * noise
    result = likelihood.fit_profile_likelihood(data, truth + rng.normal(0, 0.25, count))
    standardized = (result.shifts - truth) / result.standard_error
    assert 0.7 < np.sqrt(np.mean(standardized**2)) < 1.3
    assert 0.55 < np.mean(abs(standardized) <= 1) < 0.8
    if motion:
        first, second = np.array_split(result.relative_noise_variance, 2)
        assert 6 < second.mean() / first.mean() < 12
        for group in np.array_split(standardized, 2):
            assert 0.5 < np.mean(abs(group) <= 1) < 0.85
    projector = np.eye(count) - np.ones((count, count)) / count
    expected = np.diag((projector * result.ungauged_variance) @ projector)
    np.testing.assert_allclose(result.standard_error**2, expected)
    assert abs(result.shifts.mean()) < 1e-12
    assert np.linalg.eigvalsh(result.covariance).min() > 0
