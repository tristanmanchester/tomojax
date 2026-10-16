"""The runnable recipe keeps validation separate and preserves conservative fallback."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import tomojax as tj
from tomojax import _laminography as automatic


def test_uncertain_recipe_validates_projection_seed(monkeypatch, tmp_path):
    grid = tj.Grid(8, 8, 8, 1, 1, 1)
    detector = tj.Detector(12, 12, 1, 1)
    angles = np.linspace(0, 360, 32, endpoint=False)
    geometry = tj.LaminographyGeometry(grid, detector, angles, tilt_deg=30.9)
    xyz = np.indices((8, 8, 8)).astype(float)
    volume = np.exp(-np.sum((xyz - 3.5) ** 2, axis=0) / 3).astype(np.float32)
    scan = tj.Scan(np.asarray(tj.project(geometry, volume)), geometry)
    monkeypatch.setattr(
        automatic,
        "estimate_region",
        lambda s: SimpleNamespace(scan=s, grid=s.grid, info={"status": "broad_uncertain"}),
        raising=False,
    )
    result = tj.align(scan)
    info = result.info["automatic"]
    assert np.isfinite(result.volume).all()
    assert result.scan.projections is scan.projections
    assert result.scan.poses.shape == (32, 6)
    assert not info["accepted"]
    assert [item["label"] for item in info["internal"]] == ["zero", "seed"]
    assert isinstance(info["motion_accepted"], bool)
    tj.save(tmp_path / "alignment.nxs", result)
    loaded = tj.load_reconstruction(tmp_path / "alignment.nxs").info["automatic"]
    assert loaded["status"] == info["status"] and loaded["internal"] == info["internal"]
    for reused in (result.scan, tj.load(tmp_path / "alignment.nxs")):
        again = automatic.reconstruct_laminography(
            reused, align=tj.align, reconstruct=tj.reconstruct
        )
        assert again.info["automatic"]["status"] == info["status"] == "limited"
        assert again.info["automatic"]["reasons"] == info["reasons"]
    assert result.scan.source is not None
    unknown = replace(
        result.scan,
        geometry=tj.ParallelGeometry(grid, detector, angles.tolist()),
        source=replace(
            result.scan.source,
            geometry_metadata={
                "acquisition": {"geometry_status": "not_recorded"},
            },
        ),
    )
    with pytest.raises(ValueError, match="explicit mode"):
        tj.align(unknown, levels=(1,))


@pytest.mark.parametrize("improves", [False, True])
@pytest.mark.parametrize("broad", [False, True])
@pytest.mark.parametrize("n", [25, 500])
def test_recipe_excludes_validation_and_replaces_selected_poses(monkeypatch, improves, broad, n):
    angles = np.arange(n, dtype=float)
    geometry = tj.LaminographyGeometry(
        tj.Grid(4, 4, 4, 1, 1, 1), tj.Detector(4, 4, 1, 1), angles, tilt_deg=30.9
    )
    data = np.random.default_rng(3).uniform(0.2, 1, (n, 4, 4)).astype(np.float32)
    scan = tj.Scan(data, geometry)
    held = np.arange(n) % 5 == 2
    modes = []

    def align(s, *, mode, **kwargs):
        modes.append(mode)
        np.testing.assert_array_equal(s.angles, angles[~held])
        poses = np.zeros((len(s.angles), 6), np.float32)
        poses[:, 3] = 1 + float(mode == "pose")
        return SimpleNamespace(scan=s.with_poses(poses), poses=poses, info={})

    def reconstruct(s, *args, **kwargs):
        count = min(320, n * 4 // 5) if broad else n * 4 // 5
        assert len(s.angles) in {n, count}
        return tj.Reconstruction(np.full((4, 4, 4), s.poses[0, 3]), s, "fista", {})

    def project(g, volume):
        np.testing.assert_array_equal(g.angles, angles[held])
        later = volume[0, 0, 0] == (1 if broad else 2)
        error = (0.1 if improves else 0.3) if later else 0.2
        return data[held] * (1 + error)

    monkeypatch.setattr(
        automatic,
        "estimate_region",
        lambda s: SimpleNamespace(
            scan=s,
            grid=s.grid,
            info={"status": "broad_uncertain" if broad else "conditional_cv_region"},
        ),
        raising=False,
    )
    monkeypatch.setattr(automatic, "project", project)
    result = automatic.reconstruct_laminography(scan, align=align, reconstruct=reconstruct)
    info = result.info["automatic"]
    assert modes == (["full"] if broad else ["full", "pose"])
    assert info["accepted"] is (improves and not broad)
    assert info["motion_accepted"] is (improves if broad else None)
    np.testing.assert_array_equal(result.scan.poses[:, 3], int(not broad) + float(improves))


@pytest.mark.parametrize("status", ["not_recorded", "provided"])
@pytest.mark.parametrize("failure", ["boundary", "pole", "resolved"])
@pytest.mark.parametrize("pitch", [1.0, 3.0])
def test_recipe_declines_unresolved_axis_only_without_supplied_geometry(
    monkeypatch, status, failure, pitch
):
    geometry = tj.ParallelGeometry(
        tj.Grid(4, 4, 4, pitch, pitch, pitch), tj.Detector(4, 4, pitch, pitch), [0, 90]
    )
    scan = tj.Scan(np.ones((2, 4, 4), np.float32), geometry)
    scan = replace(scan, source=automatic.record_of(scan))
    scan.source.geometry_metadata["acquisition"] = {"geometry_status": status}
    scan.source.geometry_metadata["detector_center_status"] = "not_recorded"
    z = 0.2 if failure == "pole" else 0.3  # 0.8 versus 1.2 working pixels.
    seeded = replace(
        scan,
        geometry=automatic.RotationAxisGeometry(
            scan.grid, scan.detector, [0, 90], (0, np.sqrt(1 - z * z), z)
        ),
    )
    poses = np.zeros((2, 6), np.float32)

    def align(*args, **kwargs):
        stats = {"axis_seed_on_search_boundary": failure == "boundary"}
        stats["detector_center_seed_applied"] = failure == "resolved"
        return SimpleNamespace(scan=seeded, poses=poses, info={"outer_stats": [stats]})

    monkeypatch.setattr(
        automatic,
        "estimate_region",
        lambda s: SimpleNamespace(scan=s, grid=s.grid, info={"status": "broad_uncertain"}),
    )
    work, region, _ = automatic._seed(scan, np.arange(2), align)
    declined = status == "not_recorded" and failure != "resolved"
    expected_center = "inferred" if failure == "resolved" else "not_recorded"
    assert work.source.geometry_metadata["detector_center_status"] == expected_center
    assert scan.source.geometry_metadata["detector_center_status"] == "not_recorded"
    assert region["status"] == ("axial_or_unresolved" if declined else "broad_uncertain")
    if declined:
        reason = "search boundary" if failure == "boundary" else "depth parallax"
        assert work is scan and reason in region["reason"]
