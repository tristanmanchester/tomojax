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
    geometry = tj.LaminographyGeometry(grid, detector, np.arange(25), tilt_deg=30)
    scan = tj.Scan(np.ones((25, 7, 23), np.float32), geometry)
    supplied = tj.Grid(7, 9, 5, 0.7, 1.1, 0.9) if explicit else None
    shifts = np.stack([np.sin(scan.angles), np.cos(scan.angles)], 1)

    table = np.zeros((25, 6))
    table[:, 3:5] = shifts

    def fit(projections, nominal, g, d, *, fit, keep_grid, frames, detector_roll_deg):
        assert projections is scan.projections and d == detector and keep_grid == explicit
        assert g == (supplied or grid) and nominal.shape == (25, 4, 4)
        assert fit == "translations" and frames is None and detector_roll_deg == 0
        volume = np.zeros((g.nz, g.ny, g.nx), np.float32)
        return volume, g, table, {"fit": ["translations"], "corrections": {}}

    monkeypatch.setattr(automatic, "fit_poses", fit)
    result = automatic.reconstruct_with_motion(scan, reconstruct=None, grid=supplied)
    assert result.scan.projections is scan.projections and result.scan.detector == detector
    np.testing.assert_allclose(result.scan.poses[:, 3:5], shifts)
    np.testing.assert_array_equal(result.scan.poses[:, [0, 1, 2, 5]], 0)
    np.testing.assert_allclose(result.scan.geometry.axis_unit_lab, scan.geometry.axis_unit_lab)
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


@pytest.mark.parametrize(
    ("kind", "operation"),
    [("lamino", tj.align), ("lamino", tj.reconstruct), ("parallel", tj.align)],
)
def test_automatic_facades_and_saved_pose_reuse(kind, operation, tmp_path):
    grid = tj.Grid(8, 8, 8, 1, 1, 1)
    geometry = tj.LaminographyGeometry(
        grid, tj.Detector(12, 12, 1, 1), np.arange(0, 360, 12), tilt_deg=30
    )
    if kind == "parallel":
        geometry = tj.ParallelGeometry(grid, geometry.detector, geometry.angles)
    xyz = np.indices((8, 8, 8)).astype(float)
    volume = np.exp(-np.sum((xyz - 3.5) ** 2, axis=0) / 3).astype(np.float32)
    scan = tj.Scan(np.asarray(tj.project(geometry, volume)), geometry)
    result = operation(scan)
    assert np.isfinite(result.volume).all()
    assert result.scan.projections is scan.projections
    assert result.info["fit"] == ["translations"]
    tj.save(tmp_path / "alignment.nxs", result)
    reused = tj.load(tmp_path / "alignment.nxs")
    expected = result.scan.poses

    def reconstruct(s, method, *, config):
        assert s.grid == reused.grid and method == "fista"
        np.testing.assert_array_equal(s.poses, expected)
        return tj.Reconstruction(None, s, method, {})

    again = automatic.reconstruct_with_motion(reused, reconstruct=reconstruct)
    assert again.info["automatic"]["geometry"] == "preserved_pose_corrections"


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
