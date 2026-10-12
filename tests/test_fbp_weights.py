"""Exact angular weights and tilted-axis FBP against frequency-limited truth."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.recon import FBPConfig, FBPHostConfig, fbp, fbp_host

# check-public-imports: allow-private
from tomojax.recon._fbp_weights import fbp_weights


def poses_for(geometry: ParallelGeometry | LaminographyGeometry, n: int) -> np.ndarray:
    return np.asarray([geometry.pose_for_view(i) for i in range(n)])


def parallel(angles: np.ndarray) -> np.ndarray:
    grid = Grid(4, 4, 4, 1.0, 1.0, 1.0)
    return poses_for(ParallelGeometry(grid, Detector(4, 4, 1.0, 1.0), angles), len(angles))


@pytest.mark.parametrize("indices", [[0, 0, 1], [1, 0, 1, 1], [0, 0, 1, 1]])
@pytest.mark.parametrize("filter_name", ["ramp", "hann", "shepp-logan"])
@pytest.mark.parametrize("backend", ["jax", "auto"])
@pytest.mark.parametrize("host_slabs", [False, True])
def test_sparse_fbp_averages_repeated_exposures(indices, filter_name, backend, host_slabs):
    """Underdetermined axis fitting must not weight a plane by its exposure count."""
    width, sigma = 33, 3.2
    positions = np.arange(width) - (width - 1) / 2
    angles = np.array([0.0, 90.0])
    theta = np.deg2rad(angles)
    centre = 1.7 * np.cos(theta) + 0.6 * np.sin(theta)
    data = (
        np.sqrt(2 * np.pi)
        * sigma
        * np.exp(-0.5 * ((positions[None, :] - centre[:, None]) / sigma) ** 2)
    ).astype(np.float32)[:, None, :]
    repeated = data[indices].copy()
    # Vary individual exposures without changing each plane's mean signal.
    noise = np.random.default_rng(452).normal(0, 0.01, repeated.shape)
    for plane in range(2):
        own = np.flatnonzero(np.asarray(indices) == plane)
        noise[own] -= noise[own].mean(axis=0)
    repeated += noise.astype(np.float32)
    grid = Grid(width, width, 1, 1.0, 1.0, 1.0)
    detector = Detector(width, 1, 1.0, 1.0)

    def run(views, rows):
        geometry = ParallelGeometry(grid, detector, views)
        if host_slabs:
            return fbp_host(
                geometry,
                grid,
                detector,
                rows,
                config=FBPHostConfig(filter=filter_name, backprojector=backend, views_per_batch=2),
            )
        return np.asarray(
            fbp(
                geometry,
                grid,
                detector,
                jnp.asarray(rows),
                config=FBPConfig(filter=filter_name, backprojector=backend),
            )
        )

    expected = run(angles, data)
    actual = run(angles[indices], repeated)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-6)
    truth = np.exp(
        -0.5 * ((positions[:, None] - 1.7) ** 2 + (positions[None, :] + 0.6) ** 2) / sigma**2
    )[:, :, None]
    error = np.linalg.norm(actual - truth) / np.linalg.norm(truth)
    # Two views deliberately leave severe streaks; repeats must not alter the
    # independently scored reconstruction error, not just its total amplitude.
    assert abs(error - np.linalg.norm(expected - truth) / np.linalg.norm(truth)) < 2e-6


@pytest.mark.parametrize("angles", [[0.0], [0.0, 180.0], [-360.0, 0.0, 180.0]])
def test_single_projection_plane_fbp_fallback_remains_finite(angles):
    grid = Grid(9, 9, 1, 1.0, 1.0, 1.0)
    detector = Detector(9, 1, 1.0, 1.0)
    positions = np.arange(9) - 4
    row = np.exp(-0.5 * positions**2 / 4).astype(np.float32)
    geometry = ParallelGeometry(grid, detector, angles)
    actual = np.asarray(
        fbp(geometry, grid, detector, jnp.broadcast_to(jnp.asarray(row), (len(angles), 1, 9)))
    )
    expected = np.asarray(
        fbp(
            ParallelGeometry(grid, detector, [0.0]), grid, detector, jnp.asarray(row)[None, None, :]
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_sparse_fbp_explicit_scale_keeps_per_exposure_weight():
    grid = Grid(9, 9, 1, 1.0, 1.0, 1.0)
    detector = Detector(9, 1, 1.0, 1.0)
    data = jnp.arange(18, dtype=jnp.float32).reshape(2, 1, 9) / 10
    config = FBPConfig(scale=0.1)

    def run(angles, rows):
        return np.asarray(
            fbp(ParallelGeometry(grid, detector, angles), grid, detector, rows, config=config)
        )

    expected = run([0.0, 90.0], data) + run([0.0], data[:1])
    actual = run([0.0, 0.0, 90.0], data[jnp.array([0, 0, 1])])
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("backend", ["jax", "auto"])
def test_sparse_fbp_matches_independent_dense_plane_cells(backend):
    """Independent spatial ramp/linear sampling anchors the fallback's measure."""
    width, sigma = 33, 3.2
    positions = np.arange(width) - (width - 1) / 2
    rows = (
        np.sqrt(2 * np.pi)
        * sigma
        * np.exp(-0.5 * ((positions[None, :] - np.array([1.7, 0.6])[:, None]) / sigma) ** 2)
    )
    radius = int(np.ceil(np.sqrt(2) * np.max(abs(positions))))
    filtered_positions = np.arange(-radius, radius + 1)
    distance = filtered_positions[:, None] - positions[None, :]
    ramp = np.zeros(distance.shape)
    ramp[distance == 0] = 0.25
    odd = distance % 2 != 0
    ramp[odd] = -1 / (np.pi**2 * distance[odd] ** 2)
    filtered = rows @ ramp.T
    x, y = np.meshgrid(positions, positions, indexing="ij")
    expected = (np.pi / 2) * (
        np.interp(x, filtered_positions, filtered[0])
        + np.interp(-y, filtered_positions, filtered[1])
    )
    grid = Grid(width, width, 1, 1.0, 1.0, 1.0)
    detector = Detector(width, 1, 1.0, 1.0)
    geometry = ParallelGeometry(grid, detector, [0.0, 0.0, 90.0])
    data = jnp.asarray(rows[[0, 0, 1], None, :], jnp.float32)
    actual = np.asarray(
        fbp(geometry, grid, detector, data, config=FBPConfig(backprojector=backend))
    )[:, :, 0]
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("start", [-179.0, -100.0, -1.0, 0.0, 7.0, 13.0, 37.0, 150.0, 359.0])
@pytest.mark.parametrize("order", ["forward", "reverse", "shuffled"])
def test_uniform_half_turn_reduces_to_standard_fbp(start, order) -> None:
    angles = start + np.linspace(0, 180, 90, endpoint=False)
    if order == "reverse":
        angles = angles[::-1]
    elif order == "shuffled":
        angles = angles[np.random.default_rng(574).permutation(len(angles))]
    weights = fbp_weights(parallel(angles))
    assert weights.separable and not weights.full_turn
    np.testing.assert_allclose(weights.view_scale, np.pi / 90, rtol=1e-5)
    np.testing.assert_allclose(weights.arc_length, np.pi, rtol=1e-6)


@pytest.mark.parametrize("start", [7.0, 150.0, -100.0])
@pytest.mark.parametrize("span", [30.0, 90.0, 180.0])
@pytest.mark.parametrize("tilt", [0.0, 30.0])
def test_offset_short_arcs_have_positive_physical_midpoint_quadrature(start, span, tilt):
    grid = Grid(4, 4, 4, 1, 1, 1)
    detector = Detector(4, 4, 1, 1)
    angles = start + np.linspace(0, span, 60, endpoint=False)
    geometry = (
        ParallelGeometry(grid, detector, angles)
        if tilt == 0
        else LaminographyGeometry(grid, detector, angles, tilt_deg=tilt)
    )
    weights = fbp_weights(poses_for(geometry, len(angles)))
    assert not weights.full_turn
    np.testing.assert_allclose(weights.params[:, 5], np.deg2rad(span) / 60, rtol=1e-5)
    np.testing.assert_allclose(weights.arc_length, np.deg2rad(span), rtol=1e-6)
    assert np.all((weights.params[:, 4] > 0) & (weights.params[:, 4] < weights.arc_length))


def test_full_turn_counts_each_frequency_twice() -> None:
    weights = fbp_weights(parallel(np.linspace(0, 360, 120, endpoint=False)))
    assert weights.separable and weights.full_turn
    np.testing.assert_allclose(weights.view_scale, (2 * np.pi / 120) / 2, rtol=1e-5)


@pytest.mark.parametrize("start", [0.0, 150.0, -100.0])
@pytest.mark.parametrize("order", ["forward", "reverse", "shuffled"])
def test_irregular_full_turn_uses_closed_circle_quadrature(start, order) -> None:
    angles = start + np.array([0.0, 60.0, 150.0, 260.0])
    expected = np.deg2rad([40.0, 37.5, 50.0, 52.5])
    indices = np.arange(len(angles))
    if order == "reverse":
        indices = indices[::-1]
    elif order == "shuffled":
        indices = np.random.default_rng(564).permutation(indices)
    weights = fbp_weights(parallel(angles[indices]))
    assert weights.full_turn and weights.separable
    np.testing.assert_allclose(weights.arc_length, 2 * np.pi)
    np.testing.assert_allclose(weights.view_scale, expected[indices], rtol=1e-5)
    np.testing.assert_allclose(weights.view_scale.sum(), np.pi, rtol=1e-6)


@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_jittered_full_turn_recovers_gaussian_in_physical_units(backend) -> None:
    n, sigma = 48, 5.0
    grid, detector = Grid(n, n, 1, 1, 1, 1), Detector(80, 1, 1, 1)
    angles = np.arange(0.0, 360.0, 6.0)
    angles += np.random.default_rng(451).uniform(-0.6, 0.6, angles.shape)
    geometry = ParallelGeometry(grid, detector, angles)
    u = np.arange(detector.nu) - (detector.nu - 1) / 2
    profile = np.sqrt(2 * np.pi) * sigma * np.exp(-(u**2) / (2 * sigma**2))
    projections = jnp.asarray(np.broadcast_to(profile, (len(angles), 1, detector.nu)))
    actual = np.asarray(
        fbp(geometry, grid, detector, projections, config=FBPConfig(backprojector=backend))
    )[:, :, 0]
    x = np.arange(n) - (n - 1) / 2
    truth = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma**2))
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.007
    np.testing.assert_allclose(actual[n // 2, n // 2], truth[n // 2, n // 2], rtol=0.006)


@pytest.mark.parametrize("start", [0.0, 150.0, -100.0])
def test_partial_arc_halves_views_with_an_opposite(start: float) -> None:
    """Over 270 degrees, only views whose opposite was also acquired share frequencies."""
    angles = start + np.arange(270) + 0.5
    weights = fbp_weights(parallel(angles))
    opposite = (np.arange(270) + 0.5 + 180) % 360 < 270
    expected = np.where(opposite, 0.5, 1.0) * np.deg2rad(1.0)
    np.testing.assert_allclose(weights.view_scale, expected, rtol=1e-5)
    np.testing.assert_allclose(weights.arc_length, np.deg2rad(270), rtol=1e-6)


def test_irregular_angles_use_midpoint_quadrature() -> None:
    angles = np.array([10.0, 12.0, 40.0, 41.0, 100.0])
    weights = fbp_weights(parallel(angles[::-1]))
    expected = np.deg2rad([2.0, 15.0, 14.5, 30.0, 59.0])[::-1]
    np.testing.assert_allclose(weights.view_scale, expected, rtol=1e-5)


@pytest.mark.parametrize("start", [0.0, 150.0, -100.0])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("repeats", [None, (1, 3, 2, 1, 2, 4)])
def test_irregular_half_turn_uses_periodic_direction_cells(start, reverse, repeats) -> None:
    angles = start + np.array([0.0, 1.0, 45.0, 90.0, 170.0, 180.0])
    # Unique directions have circular cell widths 5.5,22.5,44.5,62.5,45
    # degrees. The 0/180 pair shares the first cell equally, independently
    # of how many repeated exposures each ray direction contains.
    expected = np.deg2rad([2.75, 22.5, 44.5, 62.5, 45.0, 2.75])
    if repeats is not None:
        angles = np.repeat(angles, repeats)
        expected = np.repeat(expected / np.asarray(repeats), repeats)
    if reverse:
        angles, expected = angles[::-1], expected[::-1]
    weights = fbp_weights(parallel(angles))
    assert weights.separable and not weights.full_turn
    np.testing.assert_allclose(weights.view_scale, expected, rtol=1e-5)
    np.testing.assert_allclose(weights.view_scale.sum(), np.pi, rtol=1e-6)


@pytest.mark.parametrize("backend", ["jax", "host", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_missing_early_views_preserve_gaussian_attenuation(backend) -> None:
    n, sigma = 64, 8.0
    grid, detector = Grid(n, n, 1, 1, 1, 1), Detector(96, 1, 1, 1)
    angles = np.concatenate(([0.0], np.arange(10.0, 181.0)))
    geometry = ParallelGeometry(grid, detector, angles)
    u = np.arange(detector.nu) - (detector.nu - 1) / 2
    profile = np.sqrt(2 * np.pi) * sigma * np.exp(-(u**2) / (2 * sigma**2))
    projections = jnp.asarray(np.broadcast_to(profile, (len(angles), 1, detector.nu)))
    volume = (
        fbp_host(geometry, grid, detector, np.asarray(projections))
        if backend == "host"
        else fbp(geometry, grid, detector, projections, config=FBPConfig(backprojector=backend))
    )
    actual = np.asarray(volume)[:, :, 0]
    x = np.arange(n) - (n - 1) / 2
    truth = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma**2))
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.005
    np.testing.assert_allclose(actual[n // 2, n // 2], truth[n // 2, n // 2], rtol=0.003)


@pytest.mark.parametrize("backend", ["jax", "host", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_missing_views_recover_an_off_axis_anisotropic_gaussian(backend) -> None:
    n = 64
    grid, detector = Grid(n, n, 1, 1, 1, 1), Detector(96, 1, 1, 1)
    angles = np.concatenate(([0.0], np.arange(10.0, 181.0)))
    geometry = ParallelGeometry(grid, detector, angles)
    cx, cy, sx, sy = 8.0, -5.0, 4.0, 8.0
    phi = np.deg2rad(angles)
    sigma = np.sqrt(sx**2 * np.cos(phi) ** 2 + sy**2 * np.sin(phi) ** 2)
    centre = cx * np.cos(phi) - cy * np.sin(phi)
    u = np.arange(detector.nu) - (detector.nu - 1) / 2
    profile = (
        np.sqrt(2 * np.pi)
        * sx
        * sy
        / sigma[:, None]
        * np.exp(-((u[None, :] - centre[:, None]) ** 2) / (2 * sigma[:, None] ** 2))
    )
    projections = jnp.asarray(profile[:, None, :])
    volume = (
        fbp_host(geometry, grid, detector, np.asarray(projections))
        if backend == "host"
        else fbp(geometry, grid, detector, projections, config=FBPConfig(backprojector=backend))
    )
    x = np.arange(n) - (n - 1) / 2
    truth = np.exp(-((x[:, None] - cx) ** 2) / (2 * sx**2) - (x[None, :] - cy) ** 2 / (2 * sy**2))
    assert np.linalg.norm(np.asarray(volume)[:, :, 0] - truth) / np.linalg.norm(truth) < 0.04


@pytest.mark.parametrize("backend", ["jax", "host", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_missing_views_preserve_attenuation_with_a_rolled_detector(backend) -> None:
    class RolledGeometry(ParallelGeometry):
        def pose_for_view(self, i):
            roll = np.eye(4)
            sine, cosine = np.sin(0.4), np.cos(0.4)
            roll[0, 0] = roll[2, 2] = cosine
            roll[0, 2], roll[2, 0] = sine, -sine
            return roll @ super().pose_for_view(i)

    n, sigma = 24, 4.0
    grid, detector = Grid(n, n, n, 1, 1, 1), Detector(48, 48, 1, 1)
    angles = np.concatenate(([0.0], np.arange(10.0, 181.0, 2.0)))
    geometry = RolledGeometry(grid, detector, angles)
    u = np.arange(detector.nu) - (detector.nu - 1) / 2
    profile = (
        np.sqrt(2 * np.pi) * sigma * np.exp(-(u[:, None] ** 2 + u[None, :] ** 2) / (2 * sigma**2))
    )
    projections = jnp.asarray(np.broadcast_to(profile, (len(angles), 48, 48)))
    volume = (
        fbp_host(geometry, grid, detector, np.asarray(projections))
        if backend == "host"
        else fbp(geometry, grid, detector, projections, config=FBPConfig(backprojector=backend))
    )
    actual = np.asarray(volume)
    x = np.arange(n) - (n - 1) / 2
    truth = np.exp(
        -(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2) / (2 * sigma**2)
    )
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.015
    np.testing.assert_allclose(
        actual[n // 2, n // 2, n // 2], truth[n // 2, n // 2, n // 2], rtol=0.015
    )


@pytest.mark.parametrize("span", [90.0, 180.0, 270.0, 360.0])
@pytest.mark.parametrize("start", [7.0, 150.0])
@pytest.mark.parametrize("tilt", [0.0, 30.0])
@pytest.mark.parametrize("repeats", [2, 3, 5])
def test_repeated_directions_share_weights_without_changing_the_acquired_arc(
    span, start, tilt, repeats
):
    grid, detector = Grid(4, 4, 4, 1, 1, 1), Detector(4, 4, 1, 1)
    angles = start + np.arange(120) * span / 120
    geometry = (
        ParallelGeometry(grid, detector, angles)
        if tilt == 0
        else LaminographyGeometry(grid, detector, angles, tilt_deg=tilt)
    )
    original = fbp_weights(poses_for(geometry, len(angles)))
    poses = np.repeat(poses_for(geometry, len(angles)), repeats, axis=0)
    order = np.random.default_rng(432).permutation(len(poses))
    repeated = fbp_weights(poses[order])
    assert repeated.full_turn == original.full_turn
    assert repeated.separable == original.separable
    np.testing.assert_allclose(repeated.arc_length, original.arc_length, rtol=1e-6)
    np.testing.assert_allclose(
        repeated.params[:, 5], np.repeat(original.params[:, 5] / repeats, repeats)[order], rtol=1e-5
    )
    np.testing.assert_allclose(
        repeated.view_scale, np.repeat(original.view_scale / repeats, repeats)[order], rtol=1e-5
    )
    assert np.all(repeated.params[:, 5] > 0)


@pytest.mark.parametrize("start", [0.0, 150.0])
def test_whole_turn_repeats_share_cells_despite_pose_rounding(start: float) -> None:
    """Views labelled theta and theta + 360 differ only by rounding; they are one direction."""
    angles = start + np.linspace(0, 180, 90, endpoint=False)
    weights = fbp_weights(parallel(np.concatenate([angles, angles + 360.0])))
    np.testing.assert_allclose(weights.view_scale, np.pi / 180, rtol=1e-5)
    np.testing.assert_allclose(weights.arc_length, np.pi, rtol=1e-6)


def test_nearby_distinct_directions_keep_their_own_angular_cells() -> None:
    angles = np.array([10.0, 10.001, 40.0, 41.0, 100.0])
    weights = fbp_weights(parallel(angles))
    expected = np.deg2rad([0.001, 15.0, 15.4995, 30.0, 59.0])
    np.testing.assert_allclose(weights.params[:, 5], expected, rtol=1e-5)


@pytest.mark.parametrize("order", ["forward", "reverse", "shuffled"])
def test_unequal_repeated_directions_share_independent_midpoint_quadrature(order):
    angles = np.array([10.0, 12.0, 40.0, 41.0, 100.0])
    repeats = np.array([1, 4, 2, 3, 1])
    expected = np.repeat(np.deg2rad([2.0, 15.0, 14.5, 30.0, 59.0]) / repeats, repeats)
    indices = np.arange(sum(repeats))
    if order == "reverse":
        indices = indices[::-1]
    elif order == "shuffled":
        indices = np.random.default_rng(576).permutation(indices)
    weights = fbp_weights(parallel(np.repeat(angles, repeats)[indices]))
    np.testing.assert_allclose(weights.view_scale, expected[indices], rtol=1e-5)


def test_tilted_scans_scale_ramp_by_ray_axis_sine() -> None:
    grid = Grid(4, 4, 4, 1.0, 1.0, 1.0)
    detector = Detector(4, 4, 1.0, 1.0)
    full = LaminographyGeometry(
        grid, detector, np.linspace(0, 360, 60, endpoint=False), tilt_deg=30
    )
    weights = fbp_weights(poses_for(full, 60))
    # Rays along +y meet an axis tilted 30 degrees from z towards y at 60 degrees.
    assert weights.separable
    np.testing.assert_allclose(weights.view_scale, np.sin(np.deg2rad(60)) * np.pi / 60, rtol=1e-5)
    half = LaminographyGeometry(
        grid, detector, np.linspace(0, 180, 60, endpoint=False), tilt_deg=30
    )
    assert not fbp_weights(poses_for(half, 60)).separable


def test_degenerate_scans_are_rejected() -> None:
    with pytest.raises(ValueError, match="three views"):
        fbp_weights(parallel(np.array([0.0, 90.0])))
    with pytest.raises(ValueError, match="rotate"):
        fbp_weights(parallel(np.zeros(5)))


def gaussian_case(span: float, n_views: int, start: float = 0.0) -> tuple:
    """Analytic projections of an off-centre Gaussian in a 30-degree laminography scan."""
    n, sigma, centre = 40, 3.0, np.array([2.0, -3.0, 1.5])
    grid = Grid(n, n, n, 1.0, 1.0, 1.0)
    detector = Detector(64, 64, 1.0, 1.0)
    angles = start + np.linspace(0, span, n_views, endpoint=False)
    geometry = LaminographyGeometry(grid, detector, angles, tilt_deg=30)
    poses = poses_for(geometry, n_views)
    u = np.arange(detector.nu) - (detector.nu - 1) / 2
    v = np.arange(detector.nv) - (detector.nv - 1) / 2
    world = np.stack(np.meshgrid(u, 0.0, v, indexing="xy"), axis=-1)[0]
    projections = np.empty((n_views, detector.nv, detector.nu), np.float32)
    for i, pose in enumerate(poses):
        rotation, translation = pose[:3, :3], pose[:3, 3]
        points = (world.reshape(-1, 3) - translation) @ rotation
        offset = points - centre
        along = offset @ rotation[1]
        distance2 = np.sum(offset**2, axis=1) - along**2
        line = np.sqrt(2 * np.pi) * sigma * np.exp(-distance2 / (2 * sigma**2))
        projections[i] = line.reshape(detector.nu, detector.nv).T
    axes = [np.arange(n) - (n - 1) / 2] * 3
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    truth = np.exp(-((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (zz - centre[2]) ** 2) / 18)
    return geometry, grid, detector, poses, projections, truth


def measured_truth(truth: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Keep only frequencies lying in some view's projection plane."""
    shape = [2 * s for s in truth.shape]
    spectrum = np.fft.fftn(truth, shape, axes=(0, 1, 2))
    k = np.stack(np.meshgrid(*[np.fft.fftfreq(s) for s in shape], indexing="ij"), axis=-1)
    norm = np.linalg.norm(k, axis=-1, keepdims=True)
    direction = k / np.where(norm == 0, 1, norm)
    rays = poses[:, 1, :3]
    # Views are 3 degrees apart: a frequency is measured where some ray is nearly normal to it.
    nearest = np.min(np.abs(direction.reshape(-1, 3) @ rays.T), axis=1).reshape(shape)
    measured = (nearest <= np.sin(np.deg2rad(1.6))) | (norm[..., 0] == 0)
    restricted = np.real(np.fft.ifftn(spectrum * measured))
    return restricted[tuple(slice(0, s) for s in truth.shape)]


@pytest.mark.parametrize(("span", "n_views"), [(360.0, 120), (180.0, 60)])
@pytest.mark.parametrize("start", [0.0, 7.0])
def test_tilted_fbp_recovers_every_measured_frequency(span: float, n_views: int, start) -> None:
    geometry, grid, detector, poses, projections, truth = gaussian_case(span, n_views, start)
    actual = np.asarray(
        fbp(
            geometry,
            grid,
            detector,
            jnp.asarray(projections),
            config=FBPConfig(backprojector="jax"),
        )
    )
    reference = measured_truth(truth, poses)
    # The missing cone is a large part of the error against the full truth. The
    # remainder is discretization of a 3-voxel Gaussian; uniform pi/n weights
    # give 0.14 (full turn) and 0.42 (half turn) here.
    assert np.linalg.norm(reference - truth) / np.linalg.norm(truth) > 0.1
    assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < 0.07


@pytest.mark.parametrize(("span", "n_views"), [(360.0, 120), (180.0, 60)])
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_repeated_laminography_exposures_equal_their_mean_and_preserve_physical_truth(
    span, n_views, backend
):
    geometry, grid, detector, poses, projections, truth = gaussian_case(span, n_views, 7.0)
    noise = (
        np.random.default_rng(132)
        .normal(0, 0.001 * projections.max(), projections.shape)
        .astype(np.float32)
    )
    repeated = np.stack((projections + noise, projections, projections - noise), axis=1)
    averaged = repeated.mean(axis=1, dtype=np.float64).astype(np.float32)
    repeated = repeated.reshape((-1, detector.nv, detector.nu))
    order = np.random.default_rng(432).permutation(len(repeated))
    repeated_geometry = LaminographyGeometry(
        grid, detector, np.repeat(geometry.angles, 3)[order], tilt_deg=30
    )
    cfg = FBPConfig(backprojector=backend)
    expected = np.asarray(fbp(geometry, grid, detector, averaged, config=cfg))
    actual = np.asarray(fbp(repeated_geometry, grid, detector, repeated[order], config=cfg))
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 2e-6
    reference = measured_truth(truth, poses)
    assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < 0.07
