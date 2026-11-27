"""Independent Radon integrals on an offset detector through a complete circular orbit."""

import numpy as np
import pytest

import tomojax as tj


@pytest.mark.parametrize("method", ["fbp", "fourier"])
@pytest.mark.parametrize("side", [-1, 1])
def test_full_turn_resolves_the_axis_and_outer_field(method, side):
    # Gaussian Radon integrals on irregular, unpaired directions (including a repeat).
    angles = np.linspace(3, 363, 243, endpoint=False) + 0.08 * np.sin(np.arange(243))
    angles = np.r_[angles, angles[0]]
    grid = tj.Grid(96, 96, 1, 0.7, 0.7, 1.0)
    detector = tj.Detector(54, 1, 0.7, 1.0, center=(side * 14.7, 0))
    g = tj.ParallelGeometry(grid, detector, angles)
    u = (np.arange(54) - 26.5) * detector.du + detector.center[0]
    theta = np.deg2rad(angles)[:, None]
    x = (np.arange(96) - 47.5) * grid.vx
    truth = np.zeros((96, 96))
    p = np.zeros((len(angles), 54))
    for cx, cy, sigma, peak in [(0, 0, 8, 0.8), (21, -5, 2, 1.3), (-2, 1, 1, 0.6)]:
        distance = u - cx * np.cos(theta) + cy * np.sin(theta)
        p += np.sqrt(2 * np.pi) * sigma * peak * np.exp(-0.5 * (distance / sigma) ** 2)
        truth += peak * np.exp(-0.5 * ((x - cx) ** 2 + (x[:, None] - cy) ** 2) / sigma**2)
    volume = np.asarray(tj.reconstruct(tj.Scan(p[:, None].astype("f4"), g), method).volume)[0]
    radius = np.hypot(x, x[:, None])
    for mask in (radius < 4, radius < 16, (radius > 16) & (radius < 32)):
        # Detector sampling and linear angular interpolation broaden the narrow peaks.
        assert np.linalg.norm((volume - truth)[mask]) < 0.035 * np.linalg.norm(truth[mask])
