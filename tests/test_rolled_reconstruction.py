"""Physical detector coordinates in analytic inverses, including oblique filter lines."""

from dataclasses import replace

import jax
import numpy as np
import pytest

import tomojax as tj
from tomojax.recon import (
    CGLSConfig,
    FBPConfig,
    FDKConfig,
    FDKHostConfig,
    FourierConfig,
    fdk,
    fdk_host,
)


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


@pytest.mark.parametrize("method", ["auto", "cgls"])
def test_oblique_fdk_auto_and_warm_start_follow_the_same_measured_rays(method):
    grid = tj.Grid(21, 21, 17, 0.8, 0.8, 0.8)
    panel = tj.Detector(39, 31, 0.8, 0.9, center=(1.3, -0.5))
    angles = np.linspace(17, 237, 64, endpoint=False)
    geometry = tj.ConeGeometry(grid, panel, angles, tj.ConeBeam(96, 144, detector_roll=15))
    data, truth = gaussian(grid, panel, angles, 15)
    swapped = replace(panel, nu=panel.nv, nv=panel.nu, du=panel.dv, dv=panel.du)
    rolled = replace(geometry, detector=swapped, beam=replace(geometry.beam, detector_roll=105))
    options = {"iterations": 1, "warm_start": True} if method == "cgls" else {}
    direct = tj.reconstruct(tj.Scan(data, geometry), method, **options)
    same = tj.reconstruct(tj.Scan(np.rot90(data, axes=(1, 2)), rolled), method, **options)
    assert direct.method == same.method == ("fbp" if method == "auto" else "cgls")
    for name, value in direct.info["truncation"].items():
        assert same.info["truncation"][name] == pytest.approx(value, rel=2e-5, abs=1e-7)
    assert np.linalg.norm(direct.volume - truth) < 0.08 * np.linalg.norm(truth)
    assert np.linalg.norm(same.volume - direct.volume) < 0.001 * np.linalg.norm(direct.volume)


@pytest.mark.parametrize("method", ["fbp", "cgls"])
def test_rotating_some_exposures_in_storage_preserves_offset_chord_fits(method):
    from tomojax.geometry import ConeSegments

    angles = np.arange(0, 360, 6)
    grid, panel = tj.Grid(11, 11, 5, 1, 1, 1), tj.Detector(23, 23, 1, 1, (2, 0))
    native = tj.ConeGeometry(grid, panel, angles, tj.ConeBeam(96, 144, detector_roll=15))
    # The two halves have different profiles at the truncated edges.
    v, u = np.mgrid[-11:12, -11:12]
    radius = np.linspace(12, 22, len(angles))[:, None, None]
    data = np.sqrt(np.maximum(radius**2 - (u - 2) ** 2 - v**2 / 4, 0)).astype(np.float32)
    half = len(angles) // 2
    turned = replace(native, angles=angles[half:], beam=replace(native.beam, detector_roll=105))
    same = ConeSegments(
        (replace(native, angles=angles[:half]), turned), trajectory_lengths=(len(angles),)
    )
    reindexed = np.concatenate((data[:half], np.rot90(data[half:], axes=(1, 2))))
    options = {"iterations": 2} if method == "cgls" else {}
    expected = tj.reconstruct(tj.Scan(data, native), method, **options)
    actual = tj.reconstruct(tj.Scan(reindexed, same), method, **options)
    assert actual.method == expected.method == method
    if method == "cgls":
        assert expected.info["truncation"]["outside_grid"] > 0
    for name, value in expected.info["truncation"].items():
        assert actual.info["truncation"][name] == pytest.approx(value, rel=2e-5, abs=1e-7)
    assert np.linalg.norm(actual.volume - expected.volume) < 3e-4 * np.linalg.norm(expected.volume)


def test_raw_oblique_cone_slabs_follow_reindexed_rays_and_share_the_whole_scan_record():
    grid = tj.Grid(11, 13, 37, 0.7, 0.7, 0.45)
    panel = tj.Detector(31, 43, 0.8, 0.9)
    angles = np.linspace(0, 360, 48, endpoint=False)
    geometry = tj.ConeGeometry(grid, panel, angles, tj.ConeBeam(96, 144, detector_roll=15))
    data, _ = gaussian(grid, panel, angles, 15)
    counts = 1000 * np.exp(-0.1 * data)
    expected = tj.reconstruct(tj.Frames(counts, geometry, white_level=1000).corrected(), "fbp")
    swapped = replace(panel, nu=panel.nv, nv=panel.nu, du=panel.dv, dv=panel.du)
    rolled = replace(geometry, detector=swapped, beam=replace(geometry.beam, detector_roll=105))
    frames = tj.Frames(np.rot90(counts, axes=(1, 2)), rolled, white_level=1000)
    info = {}
    with tj.reconstruct_slabs(frames, "fbp", info=info) as slabs:
        parts = [r.volume for _, r in slabs]
    assert len(parts) > 1
    assert np.linalg.norm(np.concatenate(parts) - expected.volume) < 0.001 * np.linalg.norm(
        expected.volume
    )
    for name, value in expected.info["truncation"].items():
        assert info["truncation"][name] == pytest.approx(value, rel=2e-5, abs=1e-7)


@pytest.mark.parametrize("on_device", [False, True])
def test_quarter_turn_preserves_the_fitted_rays_of_independent_planes(on_device):
    # check-public-imports: allow-private
    from tomojax.recon._exterior import outside_data

    width, rows = 80, 3
    pitch = 2 / width
    grid = tj.Grid(width, width, rows, pitch, pitch, pitch)
    det = tj.Detector(width, rows, pitch, pitch)
    g = tj.ParallelGeometry(grid, det, np.linspace(0, 180, width, endpoint=False))
    x = (np.arange(width) - (width - 1) / 2) * pitch
    radii = np.array([0.8, 1.2, 1.6])[:, None]
    profiles = 0.06 * np.sqrt(np.maximum(radii**2 - x**2, 0))
    data = np.broadcast_to(profiles, (width, rows, width)).astype(np.float32)
    rotated = np.ascontiguousarray(np.rot90(data, axes=(1, 2)))
    panel = replace(det, nu=rows, nv=width)
    same = replace(g, detector=panel, detector_roll=90)
    if on_device:
        data, rotated = jax.device_put(data), jax.device_put(rotated)
    bg, share = outside_data(g, grid, det, data, CGLSConfig())
    other, turned_share = outside_data(same, grid, panel, rotated, CGLSConfig())
    assert other is not None and bg is not None and share > 0
    assert turned_share == pytest.approx(share, rel=1e-6)
    np.testing.assert_array_equal(np.rot90(np.asarray(other), -1, axes=(1, 2)), bg)
