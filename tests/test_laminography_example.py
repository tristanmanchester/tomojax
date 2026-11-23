"""Declared geometry gets one bounded smooth-motion workflow, without setup guesses."""

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj
from tomojax import _smooth_alignment as automatic

# check-public-imports: allow-private
from tomojax.alignment._joint import working_scale
from tomojax.alignment._joint._fit import _finest  # check-public-imports: allow-private
from tomojax.geometry import grid_volume_origin


@pytest.mark.parametrize("shape", [(2112, 2112, 704), (9000, 9000, 1)])
@pytest.mark.parametrize("detector_shape", [(323, 7), (640, 1), (1441, 1080)])
def test_working_scale_is_bounded_and_keeps_the_support(shape, detector_shape):
    nu, nv = detector_shape
    grid = tj.Grid(*shape, 0.7, 1.1, 0.9, vol_origin=(1.2, -4, 0.5))
    work, dbin, bins = working_scale(grid, tj.Detector(nu, nv, 1, 1))
    assert work.nx * work.ny * work.nz <= 16_000_000
    assert max(nu, nv) // dbin <= 320 or dbin == min(nu, nv)
    assert all(min(nu, nv) // (dbin * b) >= 1 for b in bins)
    # Every working ray's footprint is at most one working voxel.
    assert dbin <= _finest(work) * (1 + 1e-9)
    bounds = []
    for g in (grid, work):
        pitch = np.array((g.vx, g.vy, g.vz))
        lower = np.asarray(grid_volume_origin(g)) - pitch / 2
        bounds.append((lower, lower + pitch * (g.nx, g.ny, g.nz)))
    np.testing.assert_allclose(bounds[1], bounds[0])


@pytest.mark.parametrize("explicit", [False, True])
def test_automatic_route_keeps_native_data_and_declared_geometry(explicit, monkeypatch):
    grid = tj.Grid(40, 36, 20, 0.7, 1.1, 0.9, vol_origin=(1.2, -4, 0.5))
    detector = tj.Detector(23, 7, 1.5, 0.5)
    geometry = tj.LaminographyGeometry(grid, detector, np.arange(25), tilt=30)
    scan = tj.Scan(np.ones((25, 7, 23), np.float32), geometry)
    supplied = tj.Grid(7, 9, 5, 0.7, 1.1, 0.9) if explicit else None
    shifts = np.stack([np.sin(scan.angles), np.cos(scan.angles)], 1)

    table = np.zeros((25, 6))
    table[:, 3:5] = shifts

    def fit(
        projections,
        nominal,
        g,
        d,
        *,
        fit,
        keep_grid,
        frames,
        detector_roll,
        view_positions,
        schedule,
        config=None,
    ):
        assert projections is scan.projections and d == detector and keep_grid == explicit
        assert view_positions is None
        assert g == (supplied or grid) and nominal.shape == (25, 4, 4)
        assert fit == "translations" and frames is None and detector_roll == 0
        volume = np.zeros((g.nz, g.ny, g.nx), np.float32)
        report = {"isolated_views": [], "unusual_views": [], "advisories": []}
        return volume, g, table, {"fit": ["translations"], "calibration": {}, "report": report}

    monkeypatch.setattr(automatic, "fit_poses", fit)
    result = tj.align(scan, grid=supplied)
    assert result.scan.projections is scan.projections and result.scan.detector == detector
    np.testing.assert_allclose(result.scan.poses[:, 3:5], shifts)
    np.testing.assert_array_equal(result.scan.poses[:, [0, 1, 2, 5]], 0)
    np.testing.assert_allclose(result.scan.geometry.rotation_axis(), scan.geometry.rotation_axis())
    assert result.info["method"] == "joint_volume_pose" and result.info["calibration"] == {}
    assert result.scan.grid == (supplied or grid)
    assert scan.source is None


def test_missing_axis_is_not_silently_inferred():
    geometry = tj.ParallelGeometry(tj.Grid(4, 4, 4, 1, 1, 1), tj.Detector(4, 4, 1, 1), [0, 90])
    scan = tj.Scan(np.ones((2, 4, 4), np.float32), geometry)
    scan = replace(scan, source=automatic.record_of(scan))
    scan.source.geometry_metadata["acquisition"] = {"geometry_status": "not_recorded"}
    for operation in (tj.align, tj.reconstruct):
        with pytest.raises(ValueError, match="declared acquisition geometry"):
            operation(scan)


def test_saved_alignment_reuse(tmp_path):
    # Align, save, load, and reconstruct the loaded scan through the real automatic route
    # (FBP): its saved poses are used, and on data that never moved the image is the one the
    # declared geometry gives.
    grid = tj.Grid(8, 8, 8, 1, 1, 1)
    angles = np.linspace(0, 360, 128, endpoint=False)
    geometry = tj.LaminographyGeometry(grid, tj.Detector(12, 12, 1, 1), angles, tilt=30)
    xyz = np.indices((8, 8, 8)).astype(float)
    volume = np.exp(-np.sum((xyz - 3.5) ** 2, axis=0) / 3).astype(np.float32)
    scan = tj.Scan(np.asarray(tj.project(geometry, volume)), geometry)
    result = tj.align(scan)
    assert np.isfinite(result.image).all()
    assert result.scan.projections is scan.projections
    assert result.info["fit"] == ["translations"]
    tj.save(tmp_path / "alignment.nxs", result)
    reused = tj.load(tmp_path / "alignment.nxs")
    np.testing.assert_array_equal(reused.poses, result.scan.poses)
    again = tj.reconstruct(reused)
    assert again.method == "fbp"
    assert again.info["automatic"]["geometry"] == "preserved"
    declared = np.asarray(tj.reconstruct(scan).volume)
    distance = np.linalg.norm(np.asarray(again.volume) - declared) / np.linalg.norm(declared)
    assert distance < 0.02  # 5e-5 measured: the fit finds the scan still


def test_a_posed_laminography_scan_reconstructs_with_its_poses():
    # tj.reconstruct's automatic route (FBP) on a scan carrying pose corrections uses them:
    # data moved by known per-view shifts reconstruct with those shifts to the image the
    # unmoved data give, and without them to a blur. (FBP leaves the missing cone empty, so
    # the oracle is the same method on unmoved data, not the phantom.)
    grid, detector = tj.Grid(16, 16, 16, 1, 1, 1), tj.Detector(24, 24, 1, 1)
    views = 128
    geometry = tj.LaminographyGeometry(
        grid, detector, np.linspace(0, 360, views, endpoint=False), tilt=30
    )
    xyz = np.indices((16, 16, 16)).astype(float)
    volume = np.exp(-np.sum((xyz - 7.5) ** 2, axis=0) / 8).astype(np.float32)
    turn = np.linspace(0, 2 * np.pi, views, endpoint=False)
    shifts = np.zeros((views, 6))
    shifts[:, 3], shifts[:, 4] = 1.5 * np.sin(2 * turn), np.cos(turn)
    blank = tj.Scan(np.zeros((views, 24, 24), np.float32), geometry)
    moved = tj.Scan(np.asarray(tj.project(blank.with_poses(shifts).geometry, volume)), geometry)
    posed = tj.reconstruct(moved.with_poses(shifts))
    unposed = tj.reconstruct(moved.with_poses(np.zeros((views, 6))))
    still = tj.reconstruct(tj.Scan(np.asarray(tj.project(geometry, volume)), geometry))
    assert posed.info["automatic"]["geometry"] == "preserved"

    def distance(result):
        reference = np.asarray(still.volume)
        return np.linalg.norm(np.asarray(result.volume) - reference) / np.linalg.norm(reference)

    assert distance(posed) < 0.03  # 0.0096 measured
    assert distance(unposed) > 0.2  # 0.33 measured


@pytest.mark.parametrize("keep_grid", [False, True])
def test_working_rays_sample_every_voxel(keep_grid):
    # A 656-pixel detector bins by 3; a grid at the detector pitch must follow it,
    # and a kept grid at that pitch must stop the binning instead.
    grid = tj.Grid(656, 656, 104, 4, 4, 4)
    detector = tj.Detector(656, 104, 4, 4)
    work, dbin, _ = working_scale(grid, detector, keep_grid=keep_grid)
    assert dbin * detector.du <= work.vz * (1 + 1e-9)
    assert (work == grid) if keep_grid else dbin == 3
    _, cone_bin, _ = working_scale(grid, detector, keep_grid=True, magnification=2)
    assert cone_bin == 2
