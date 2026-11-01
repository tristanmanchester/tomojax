"""Declared geometry gets one bounded smooth-motion workflow, without setup guesses."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import tomojax as tj
from tomojax import _laminography as automatic
from tomojax.geometry import grid_volume_origin


@pytest.mark.parametrize("shape", [(2112, 2112, 704), (9000, 9000, 1)])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("detector_shape", [(323, 7), (640, 1)])
def test_coarse_policy_preserves_support_native_data_and_poses(shape, explicit, detector_shape):
    nu, nv = detector_shape
    grid = tj.Grid(*shape, 0.7, 1.1, 0.9, vol_origin=(1.2, -4, 0.5))
    geometry = tj.LaminographyGeometry(grid, tj.Detector(nu, nv, 1, 1), np.arange(25), tilt_deg=30)
    scan = tj.Scan(np.ones((25, nv, nu), np.float32), geometry)
    supplied = tj.Grid(7, 9, 5, 0.7, 1.1, 0.9) if explicit else None
    expected = supplied or grid
    poses = np.zeros((25, 6), np.float32)
    poses[:, 3] = np.sin(scan.angles)

    def align(work, *, mode, levels, config):
        assert mode == "pose" and levels == (1,)
        assert config.pose_model == "spline"
        assert config.pose_translation_frame == "detector" and config.optimise_dofs == ("dx", "dz")
        assert config.outer_iterations == 8 and not config.seed_translations
        assert max(work.detector.nu, work.detector.nv) <= (320 if nv > 1 else nu)
        assert work.grid.nx * work.grid.ny * work.grid.nz <= automatic._ALIGNMENT_VOXELS
        for g in (expected, work.grid):
            pitch = np.array((g.vx, g.vy, g.vz))
            lower = np.asarray(grid_volume_origin(g)) - pitch / 2
            upper = lower + pitch * (g.nx, g.ny, g.nz)
            if g is expected:
                bounds = (lower, upper)
            else:
                np.testing.assert_allclose((lower, upper), bounds)
        return SimpleNamespace(volume=None, scan=work.with_poses(poses), info={})

    result = automatic.reconstruct_laminography(scan, align=align, reconstruct=None, grid=supplied)
    assert result.scan.projections is scan.projections and result.scan.detector == scan.detector
    np.testing.assert_array_equal(result.scan.poses, poses)
    np.testing.assert_allclose(result.scan.geometry.axis_unit_lab, scan.geometry.axis_unit_lab)
    assert result.info["automatic"]["geometry"] == "declared"
    if explicit:
        assert result.scan.grid == supplied
    assert scan.source is None


def test_missing_axis_is_not_silently_inferred():
    geometry = tj.ParallelGeometry(tj.Grid(4, 4, 4, 1, 1, 1), tj.Detector(4, 4, 1, 1), [0, 90])
    scan = tj.Scan(np.ones((2, 4, 4), np.float32), geometry)
    scan = replace(scan, source=automatic.record_of(scan))
    scan.source.geometry_metadata["acquisition"] = {"geometry_status": "not_recorded"}
    for operation in (tj.align, tj.reconstruct):
        with pytest.raises(ValueError, match="declared geometry"):
            operation(scan)


@pytest.mark.parametrize("operation", [tj.align, tj.reconstruct])
def test_automatic_facades_and_saved_pose_reuse(operation, tmp_path):
    grid = tj.Grid(8, 8, 8, 1, 1, 1)
    geometry = tj.LaminographyGeometry(
        grid, tj.Detector(12, 12, 1, 1), np.arange(0, 360, 12), tilt_deg=30
    )
    xyz = np.indices((8, 8, 8)).astype(float)
    volume = np.exp(-np.sum((xyz - 3.5) ** 2, axis=0) / 3).astype(np.float32)
    scan = tj.Scan(np.asarray(tj.project(geometry, volume)), geometry)
    result = operation(scan)
    assert np.isfinite(result.volume).all()
    assert result.scan.projections is scan.projections
    assert result.info["automatic"]["pose_model"] == "spline"
    tj.save(tmp_path / "alignment.nxs", result)
    reused = tj.load(tmp_path / "alignment.nxs")
    expected = result.scan.poses

    def reconstruct(s, method, *, config):
        assert s.grid == reused.grid and method == "fista"
        np.testing.assert_array_equal(s.poses, expected)
        return tj.Reconstruction(None, s, method, {})

    again = automatic.reconstruct_laminography(reused, align=None, reconstruct=reconstruct)
    assert again.info["automatic"]["geometry"] == "preserved_pose_corrections"
