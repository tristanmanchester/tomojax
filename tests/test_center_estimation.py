"""Projection-only centre estimation preserves physical scan context."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import BeamHardening
from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.io import ProjectionDataset, save_dataset


def _scan(*, pitch=0.8, views=180):
    angles = 350 + np.linspace(0, 180, views, endpoint=False)
    detector = Detector(64, 8, pitch, 1.3, (2.7, -4.2))
    grid = Grid(80, 80, 8, pitch, pitch, 1.3, vol_center=(0, 0, -4.2))
    true_u = -5.25 * pitch
    u = (np.arange(64) - 31.5) * pitch + true_u
    theta = np.deg2rad(angles)
    sino = np.zeros((views, 64))
    truth = np.zeros((80, 80))
    x, y = np.meshgrid(
        (np.arange(80) - 39.5) * pitch, (np.arange(80) - 39.5) * pitch, indexing="ij"
    )
    components = np.asarray([(5, 2, 3, 1), (-6, -3, 2, 0.7), (0, 9, 1.2, 0.4)])
    # Change physical units, not sampling bandwidth, with the pitch.
    components[:, :3] = components[:, :3] * pitch / 0.8
    for cx, cy, r, amplitude in components:
        projected = np.cos(theta) * cx - np.sin(theta) * cy
        sino += (
            amplitude * np.sqrt(2 * np.pi) * r * np.exp(-0.5 * ((u - projected[:, None]) / r) ** 2)
        )
        truth += amplitude * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / r**2)
    data = np.broadcast_to(sino[:, None], (views, 8, 64)).astype(np.float32).copy()
    return (
        tj.Scan(data, ParallelGeometry(grid, detector, angles), name="centre fixture"),
        true_u,
        truth,
    )


@pytest.mark.parametrize("pitch", [0.8, 2.4])
@pytest.mark.parametrize("device", [False, True])
def test_estimate_center_preserves_physical_scan_context(pitch, device, tmp_path):
    scan, true_u, _ = _scan(pitch=pitch)
    scan = scan.corrected(BeamHardening((1,)))
    # A real source record tests ownership as well as the corrected geometry on save.
    tj.save(tmp_path / "input.npz", scan)
    scan = tj.load(tmp_path / "input.npz")
    if device:
        scan = replace(scan, projections=jnp.asarray(scan.projections))
    result = tj.estimate_center(scan)
    assert result is not scan
    assert result.projections is scan.projections
    assert result.grid is scan.grid
    assert result.source is scan.source
    assert result.corrections is scan.corrections
    assert result.name == scan.name
    assert result.detector.center[1] == scan.detector.center[1]
    assert abs(result.detector.center[0] - true_u) <= 0.3 * pitch
    assert scan.detector.center[0] == 2.7
    tj.save(tmp_path / "estimated.npz", result)
    restored = tj.load(tmp_path / "estimated.npz")
    assert restored.detector == result.detector
    assert restored.grid == result.grid
    np.testing.assert_array_equal(restored.projections, scan.projections)


def test_estimate_center_after_crop_and_bin_keeps_absolute_physical_units():
    scan, true_u, _ = _scan()
    known = replace(
        scan,
        geometry=replace(scan.geometry, detector=replace(scan.detector, center=(true_u, -4.2))),
    )
    for factor in (1, 2):
        preview = scan.cropped(slice(1, 7), slice(2, 61)).binned(factor)
        expected = known.cropped(slice(1, 7), slice(2, 61)).binned(factor)
        result = tj.estimate_center(preview)
        assert result.grid is scan.grid
        assert result.projections is preview.projections
        assert result.detector.center[1] == expected.detector.center[1]
        assert (
            abs(result.detector.center[0] - expected.detector.center[0])
            <= 0.4 * preview.detector.du
        )


@pytest.mark.parametrize("order", ["reverse", "repeat", "irregular"])
def test_estimate_center_uses_physical_angular_sampling(order):
    scan, true_u, _ = _scan()
    if order == "reverse":
        indices = np.arange(179, -1, -1)
    elif order == "repeat":
        indices = np.repeat(np.arange(180), 2)
    else:
        indices = np.delete(np.arange(180), np.arange(8, 165, 3))
    angles = scan.angles[indices] % 360
    angles += 360 * np.random.default_rng(8).integers(-2, 3, len(indices))
    scan = replace(
        scan, projections=scan.projections[indices], geometry=replace(scan.geometry, angles=angles)
    )
    result = tj.estimate_center(scan)
    assert abs(result.detector.center[0] - true_u) <= 0.3 * scan.detector.du
    np.testing.assert_array_equal(result.angles, angles)


@pytest.mark.numerical
def test_estimated_scan_reconstructs_independent_physical_truth():
    scan, true_u, truth = _scan()
    result = tj.estimate_center(scan)
    volume = np.asarray(tj.reconstruct(result).volume)
    assert abs(result.detector.center[0] - true_u) <= 0.3 * scan.detector.du
    assert np.linalg.norm(volume[:, :, 4] - truth) / np.linalg.norm(truth) < 0.025


@pytest.mark.parametrize("kind", ["short_arc", "constant", "cone", "posed"])
def test_estimate_center_rejects_unsupported_evidence(kind, tmp_path):
    scan, _, _ = _scan()
    if kind == "short_arc":
        scan = scan.selected(slice(0, 100))
    elif kind == "constant":
        scan = replace(scan, projections=np.ones_like(scan.projections))
    elif kind == "cone":
        geometry = tj.ConeGeometry(scan.grid, scan.detector, scan.angles, tj.ConeBeam(100, 200))
        scan = replace(scan, geometry=geometry)
    else:
        poses = np.zeros((len(scan.angles), 6), np.float32)
        poses[:, 3] = np.linspace(-1, 1, len(poses))
        record = ProjectionDataset(
            projections=scan.projections,
            angles=scan.angles,
            detector=scan.detector,
            grid=scan.grid,
            align_params=poses,
        )
        save_dataset(tmp_path / "posed.npz", record)
        scan = tj.load(tmp_path / "posed.npz")
        assert scan.poses is not None
    with pytest.raises(ValueError, match="estimate_center"):
        tj.estimate_center(scan)
