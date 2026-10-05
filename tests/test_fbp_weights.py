"""Exact angular weights and tilted-axis FBP against frequency-limited truth."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.recon import FBPConfig, fbp

# check-public-imports: allow-private
from tomojax.recon._fbp_weights import fbp_weights


def poses_for(geometry: ParallelGeometry | LaminographyGeometry, n: int) -> np.ndarray:
    return np.asarray([geometry.pose_for_view(i) for i in range(n)])


def parallel(angles: np.ndarray) -> np.ndarray:
    grid = Grid(4, 4, 4, 1.0, 1.0, 1.0)
    return poses_for(ParallelGeometry(grid, Detector(4, 4, 1.0, 1.0), angles), len(angles))


def test_uniform_half_turn_reduces_to_standard_fbp() -> None:
    weights = fbp_weights(parallel(np.linspace(0, 180, 90, endpoint=False)))
    assert weights.separable and not weights.full_turn
    np.testing.assert_allclose(weights.view_scale, np.pi / 90, rtol=1e-5)


def test_full_turn_counts_each_frequency_twice() -> None:
    weights = fbp_weights(parallel(np.linspace(0, 360, 120, endpoint=False)))
    assert weights.separable and weights.full_turn
    np.testing.assert_allclose(weights.view_scale, (2 * np.pi / 120) / 2, rtol=1e-5)


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


def gaussian_case(span: float, n_views: int) -> tuple:
    """Analytic projections of an off-centre Gaussian in a 30-degree laminography scan."""
    n, sigma, centre = 40, 3.0, np.array([2.0, -3.0, 1.5])
    grid = Grid(n, n, n, 1.0, 1.0, 1.0)
    detector = Detector(64, 64, 1.0, 1.0)
    angles = np.linspace(0, span, n_views, endpoint=False)
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
def test_tilted_fbp_recovers_every_measured_frequency(span: float, n_views: int) -> None:
    geometry, grid, detector, poses, projections, truth = gaussian_case(span, n_views)
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
