"""Physical detector coordinates in analytic inverses, including oblique filter lines."""

from dataclasses import replace

import jax
import numpy as np
import pytest

import tomojax as tj
from tomojax.recon import FBPConfig, FDKConfig, FDKHostConfig, FourierConfig, fdk, fdk_host


def gaussian(grid, detector, angles, roll):
    """An anisotropic off-center Gaussian and its exact infinite cone-ray integrals."""
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du
    v = (np.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv
    c, s = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))
    x = c * u[None, :] - s * v[:, None] + detector.center[0]
    z = s * u[None, :] + c * v[:, None] + detector.center[1]
    rays = np.stack([x, np.full_like(x, 144), z], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    center = np.array([1.5, -1.2, 0.6])
    widths = np.array([2.4, 2, 1.9])
    data = []
    for theta in np.deg2rad(angles):
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        direction = (rays @ rotation) / widths
        source = (np.array([0, -96, 0]) @ rotation - center) / widths
        a = (direction**2).sum(axis=-1)
        distance2 = (source**2).sum() - (direction @ source) ** 2 / a
        data.append(np.sqrt(2 * np.pi / a) * np.exp(-0.5 * distance2))
    z, y, x = np.ogrid[: grid.nz, : grid.ny, : grid.nx]
    q = [
        (i - (n - 1) / 2) * h - c
        for i, n, h, c in zip(
            (x, y, z), (grid.nx, grid.ny, grid.nz), (grid.vx, grid.vy, grid.vz), center, strict=True
        )
    ]
    truth = np.exp(-0.5 * sum((coord / w) ** 2 for coord, w in zip(q, widths, strict=True)))
    return np.asarray(data, np.float32), truth


@pytest.mark.parametrize("exterior", ["zero", "cylinder"])
@pytest.mark.parametrize("roll", [5, 15, 45, 90])
def test_fdk_filters_the_measured_horizontal_gaussian_chords(roll, exterior):
    grid = tj.Grid(25, 27, 9, 0.7, 0.7, 0.8)
    detector = tj.Detector(51, 51, 0.75, 0.75)
    angles = np.arange(0, 360, 5)
    geometry = tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(96, 144, detector_roll=roll))
    data, truth = gaussian(grid, detector, angles, roll)
    image = tj.reconstruct(
        tj.Scan(data, geometry), "fbp", config=FBPConfig(exterior=exterior)
    ).volume
    # One-pixel linear backprojection of a 2-unit-wide peak on a 0.5-unit grid.
    assert np.linalg.norm(image - truth) < 0.018 * np.linalg.norm(truth)


@pytest.mark.parametrize("arc", [220, 360])
@pytest.mark.parametrize("exterior", ["zero", "cylinder"])
def test_reindexed_offset_panel_keeps_its_rays_and_short_scan_weights(arc, exterior):
    grid = tj.Grid(17, 19, 7, 0.8, 0.8, 0.8)
    panel = tj.Detector(41, 27, 0.8, 0.9, center=(2.4, -0.3))
    angles = np.linspace(15, 15 + arc, 60, endpoint=False)
    geometry = tj.ConeGeometry(grid, panel, angles, tj.ConeBeam(96, 144))
    data, truth = gaussian(grid, panel, angles, 0)
    cfg = FDKConfig(exterior=exterior, devices=jax.local_devices()[:2], views_per_batch=11)
    direct = fdk(geometry, grid, panel, data, config=cfg)
    rotated = replace(panel, nu=panel.nv, nv=panel.nu, du=panel.dv, dv=panel.du)
    rolled = replace(geometry, detector=rotated, beam=replace(geometry.beam, detector_roll=90))
    indexed = np.rot90(data, axes=(1, 2))
    np.testing.assert_allclose(
        indexed, gaussian(grid, rotated, angles, 90)[0], rtol=1e-6, atol=1e-7
    )
    for image in (
        fdk(rolled, grid, rotated, indexed, config=cfg),
        fdk_host(rolled, grid, rotated, indexed, config=FDKHostConfig(slices_per_batch=3, fdk=cfg)),
    ):
        # CUDA uses per-batch normalized half textures for the voxel interpolation.
        assert np.linalg.norm(image - direct) < 0.001 * np.linalg.norm(direct)
        assert np.linalg.norm(image - truth) < 0.06 * np.linalg.norm(truth)


@pytest.mark.parametrize("kind", ["parallel", "lamino", "fourier"])
@pytest.mark.parametrize("exterior", ["zero", "cylinder"])
def test_parallel_reindexing_preserves_anisotropic_pixels_and_truncated_edges(kind, exterior):
    grid, panel = (
        tj.Grid(13, 15, 5, 0.7, 0.8, 0.7),
        tj.Detector(23, 11, 0.8, 0.9, center=(1.3, -0.5)),
    )
    angles = np.linspace(7, 367, 64, endpoint=False)
    geometry = (
        tj.LaminographyGeometry(grid, panel, angles, tilt=28)
        if kind == "lamino"
        else tj.ParallelGeometry(grid, panel, angles)
    )
    cfg = (
        FourierConfig(exterior=exterior, backend="numpy")
        if kind == "fourier"
        else FBPConfig(exterior=exterior)
    )
    method = "fourier" if kind == "fourier" else "fbp"
    # Nonzero edges and unequal pixels expose continuation in the stored column direction.
    data = (
        np.random.default_rng(73)
        .uniform(0.3, 2, (len(angles), panel.nv, panel.nu))
        .astype(np.float32)
    )
    direct = tj.reconstruct(tj.Scan(data, geometry), method, config=cfg).volume
    swapped = replace(panel, nu=panel.nv, nv=panel.nu, du=panel.dv, dv=panel.du)
    rolled = replace(geometry, detector_roll=90, detector=swapped)
    actual = tj.reconstruct(tj.Scan(np.rot90(data, axes=(1, 2)), rolled), method, config=cfg).volume
    np.testing.assert_allclose(actual, direct, atol=1e-6, rtol=1e-5)
