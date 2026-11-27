"""Cone FDK and axial Fourier inversion depend on the effective circular orbit."""

from dataclasses import replace

import jax
import numpy as np
import pytest

import tomojax as tj
from tomojax.recon import FBPConfig, FourierConfig


@pytest.mark.parametrize("method", ["fbp", "fourier"])
@pytest.mark.parametrize("magnitude", [1, 5])
@pytest.mark.parametrize("span", [220, 360])
def test_pose_carried_angles_reconstruct_the_same_measurements(method, magnitude, span):
    grid = tj.Grid(11, 13, 3, 0.9, 1, 1)
    detector = tj.Detector(27, 9, 0.9, 1)
    angles = np.linspace(-31, span - 31, 54, endpoint=False)
    delta = magnitude * np.sin(2 * np.deg2rad(angles))
    # Repeat the same physical direction under a different encoder label.
    angles = np.r_[angles, angles[2] + 1]
    delta = np.r_[delta, delta[2] - 1]
    if method == "fourier":
        geometry = tj.ParallelGeometry(grid, detector, angles)
        config = FourierConfig(backend="numpy")
    else:
        geometry = tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(90, 130))
        config = FBPConfig()
    # Smooth, asymmetric measurements. The two scans attach exactly the same
    # values to the same rays, including the duplicate exposure.
    v, u = np.mgrid[-4:5, -13:14]
    center = 2 * np.sin(np.deg2rad(angles + delta))
    data = np.exp(-((u - center[:, None, None]) ** 2 + (v - 0.4) ** 2) / 11).astype(np.float32)
    table = np.zeros((len(angles), 6))
    table[:, 2] = delta
    direct = tj.Scan(data, replace(geometry, angles=angles + delta))
    posed = tj.Scan(data, geometry).with_poses(table)
    expected = tj.reconstruct(direct, method, config=config).volume
    actual = tj.reconstruct(posed, method, config=config).volume
    # The float32 composed poses can move interpolation coordinates by a few ulps.
    # CUDA texture coordinates have eight fractional bits; crossing a bin in
    # a few voxels amplifies these coordinate ulps.
    tolerance = 3e-5 if method == "fbp" and jax.default_backend() == "gpu" else 5e-6
    assert np.linalg.norm(actual - expected) < tolerance * np.linalg.norm(expected)


@pytest.mark.parametrize(("nominal", "actual"), [(170, 215), (215, 170)])
def test_fdk_short_scan_coverage_uses_the_corrected_source_arc(nominal, actual):
    from tomojax._workflow.planning import resolve_reconstruction

    angles = np.linspace(0, nominal, 40)
    grid, panel = tj.Grid(5, 5, 3, 1, 1, 1), tj.Detector(17, 5, 1, 1)
    geometry = tj.ConeGeometry(grid, panel, angles, tj.ConeBeam(80, 120))
    data = np.ones((40, 5, 17), np.float32)
    table = np.zeros((len(angles), 6))
    table[:, 2] = np.linspace(0, actual - nominal, len(angles))
    scan = tj.Scan(data, geometry).with_poses(table)
    plan = resolve_reconstruction(scan, "auto")
    assert plan.method == ("fbp" if actual > 180 else "cgls")
    if actual < 180:
        with pytest.raises(ValueError, match="short scans"):
            tj.reconstruct(scan, "fbp")
