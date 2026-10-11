"""Saved poses have one validated six-column meaning, also across cone orbits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import tomojax as tj
from tomojax.io import ProjectionDataset, build_geometry_from_dataset_metadata, save_dataset


def _metadata(columns: int, frame: str, segmented: bool) -> dict:
    grid = tj.Grid(6, 5, 4, 1.0, 0.8, 1.2)
    detector = tj.Detector(9, 7, 0.9, 1.1, (0.2, -0.3))
    beams = [
        tj.ConeBeam(18.0, 30.0, axis_offset=0.4, detector_roll_deg=7.0),
        tj.ConeBeam(22.0, 35.0, detector_pitch_deg=-5.0, axis_offset=-0.2),
    ]
    angles = np.asarray([0.0, 55.0, 130.0, 185.0, 250.0, 310.0], np.float32)
    table = np.zeros((6, columns), np.float32)
    table[:3, :5] = [0.01, -0.02, 0.03, 0.35, -0.25]
    if columns >= 6:
        table[:3, 5] = 0.15
    if columns > 6:
        table[:, 6:] = 99.0  # Not pose coordinates: documented as ignored.
    meta = {
        "grid": grid.to_dict(),
        "detector": detector.to_dict(),
        "angles": angles,
        "geometry_type": "cone",
        "cone_beam": beams[0].to_dict(),
        "align_params": table,
        "align_gauge": {"pose_translation_frame": frame},
    }
    if segmented:
        meta["cone_segments"] = [
            {"views": 3, "detector": detector.to_dict(), "cone_beam": beam.to_dict()}
            for beam in beams
        ]
    return meta


def _scan(meta: dict, *, poses: bool = True) -> tj.Scan:
    _, detector, geometry = build_geometry_from_dataset_metadata(meta, poses=poses)
    data = np.zeros((len(meta["angles"]), detector.nv, detector.nu), np.float32)
    return tj.Scan(data, geometry)


def _expected_poses(meta: dict) -> np.ndarray:
    """Independent FP64 Euler and homogeneous-matrix composition of saved coordinates."""
    table = np.asarray(meta["align_params"], np.float64)
    frame = meta.get("align_gauge", {}).get(
        "pose_translation_frame", "detector" if "cone_segments" in meta else "object"
    )
    result = np.broadcast_to(np.eye(4), (len(table), 4, 4)).copy()
    for i, (angle, params) in enumerate(zip(meta["angles"], table, strict=True)):
        entries = meta.get("cone_segments")
        beam = meta["cone_beam"] if entries is None else entries[i // 3]["cone_beam"]
        nominal = Rotation.from_euler("Z", float(angle), degrees=True).as_matrix()
        delta = Rotation.from_euler("YXZ", params[[1, 0, 2]]).as_matrix()
        shift = np.asarray([params[3], params[5] if len(params) >= 6 else 0.0, params[4]])
        result[i, :3, :3] = nominal @ delta
        result[i, :3, 3] = [beam["axis_offset"], 0.0, 0.0]
        result[i, :3, 3] += shift if frame == "detector" else nominal @ shift
    return result


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("frame", ["object", "detector"])
@pytest.mark.parametrize("columns", [5, 6, 8])
def test_saved_pose_tables_have_six_coordinates_and_independent_frames(
    segmented: bool, frame: str, columns: int
) -> None:
    meta = _metadata(columns, frame, segmented)
    original = meta["align_params"].copy()
    scan = _scan(meta)
    assert scan.poses is not None and scan.poses.shape == (6, 6)
    actual = [scan.geometry.pose_for_view(i) for i in range(6)]
    np.testing.assert_allclose(actual, _expected_poses(meta), rtol=2e-6, atol=1e-7)
    np.testing.assert_array_equal(meta["align_params"], original)
    # An unposed orbit remains exactly unposed, including its optional dy.
    np.testing.assert_array_equal(scan.poses[3:], np.zeros((3, 6), np.float32))


@pytest.mark.parametrize("segmented", [False, True])
def test_posed_and_unposed_orbits_combine_and_round_trip(segmented: bool, tmp_path: Path) -> None:
    meta = _metadata(6, "object", segmented)
    posed, plain = _scan(meta), _scan(meta, poses=False)
    combined = tj.Scan.combine([posed, plain])
    assert combined.poses is not None and combined.poses.shape == (12, 6)
    np.testing.assert_array_equal(combined.poses[6:], np.zeros((6, 6), np.float32))
    expected = np.concatenate(
        [
            _expected_poses(meta),
            _expected_poses({**meta, "align_params": np.zeros((6, 6), np.float32)}),
        ]
    )
    path = tmp_path / "combined.nxs"
    tj.save(path, combined)
    restored = tj.load(path)
    np.testing.assert_allclose(
        [restored.geometry.pose_for_view(i) for i in range(12)],
        expected,
        rtol=2e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(restored.poses, combined.poses, atol=1e-7)
    assert tj.load(path, poses=False).poses is None


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize(
    ("table", "frame", "message"),
    [
        (np.ones(6), "detector", "2-D array"),
        (np.ones((5, 6)), "detector", "row count"),
        (np.ones((6, 4)), "detector", "at least 5 columns"),
        (np.ones((6, 6)), "gantry", "unknown saved pose translation frame"),
    ],
)
def test_saved_pose_validation_is_the_same_for_one_and_several_orbits(
    segmented: bool, table: np.ndarray, frame: str, message: str
) -> None:
    meta = _metadata(6, frame, segmented)
    meta["align_params"] = table
    with pytest.raises(ValueError, match=message):
        _scan(meta)
    assert _scan(meta, poses=False).poses is None


@pytest.mark.parametrize("segmented", [False, True])
def test_omitted_saved_pose_frame_keeps_its_existing_meaning(segmented: bool) -> None:
    meta = _metadata(5, "detector" if segmented else "object", segmented)
    del meta["align_gauge"]
    scan = _scan(meta)
    np.testing.assert_allclose(
        [scan.geometry.pose_for_view(i) for i in range(6)],
        _expected_poses(meta),
        rtol=2e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("extension", ["nxs", "npz"])
def test_loading_five_column_pose_files_sets_missing_dy_to_zero(
    extension: str, tmp_path: Path
) -> None:
    meta = _metadata(5, "detector", segmented=True)
    nominal = _scan(meta, poses=False)
    dataset = ProjectionDataset(
        projections=nominal.projections,
        angles=meta["angles"],
        grid=nominal.grid,
        detector=nominal.detector,
        geometry_type="cone",
        geometry_metadata={k: meta[k] for k in ("cone_beam", "cone_segments") if k in meta},
        align_params=meta["align_params"],
        align_gauge=meta["align_gauge"],
    )
    path = tmp_path / f"five_columns.{extension}"
    save_dataset(path, dataset)
    loaded = tj.load(path)
    assert loaded.poses is not None and loaded.poses.shape == (6, 6)
    np.testing.assert_allclose(
        [loaded.geometry.pose_for_view(i) for i in range(6)],
        _expected_poses(meta),
        rtol=2e-6,
        atol=1e-7,
    )
