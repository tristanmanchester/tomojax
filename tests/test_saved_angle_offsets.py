"""Requested angle corrections cannot silently disappear from saved geometry."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import tomojax as tj
from tomojax.io import ProjectionDataset, build_geometry_from_dataset_metadata, save_dataset

_BAD_OFFSETS = [np.zeros(3), np.zeros((4, 1)), np.asarray([0.0, np.nan, 0.0, 0.0])]


def _dataset(kind, offsets):
    beam = tj.ConeBeam(24, 40, axis_offset=0.2)
    metadata = {} if kind == "parallel" else {"tilt_deg": 27.0, "tilt_about": "x"}
    if kind in {"cone", "segments"}:
        metadata["cone_beam"] = beam.to_dict()
    if kind == "segments":
        metadata["cone_segments"] = [
            {"views": 2, "cone_beam": beam.to_dict(), "tilt_deg": 27.0},
            {
                "views": 2,
                "cone_beam": tj.ConeBeam(26, 42, axis_offset=0.4).to_dict(),
                "tilt_deg": 27.0,
            },
        ]
    if kind == "segments":
        for entry in metadata["cone_segments"]:
            entry["detector"] = tj.Detector(5, 3, 0.8, 0.85).to_dict()
    return ProjectionDataset(
        projections=np.zeros((4, 3, 5), np.float32),
        angles=np.asarray([-31.123456789, 17.234567891, 168.345678912, 311.456789123]),
        grid=tj.Grid(3, 4, 2, 0.9, 1.1, 0.8),
        detector=tj.Detector(5, 3, 0.8, 0.85),
        geometry_type="cone" if kind == "segments" else kind,
        geometry_metadata=metadata,
        angle_offset_deg=offsets,
    )


@pytest.mark.parametrize("kind", ["parallel", "segments"])
@pytest.mark.parametrize("offsets", _BAD_OFFSETS)
@pytest.mark.parametrize("poses", [False, True])
def test_invalid_requested_offsets_are_reported_but_nominal_geometry_can_be_loaded(
    kind, offsets, poses
):
    dataset = _dataset(kind, offsets)
    if poses:
        with pytest.raises(ValueError, match="angle_offset_deg"):
            build_geometry_from_dataset_metadata(dataset.geometry_inputs(), poses=True)
    else:
        _, _, geometry = build_geometry_from_dataset_metadata(
            dataset.geometry_inputs(), poses=False
        )
        np.testing.assert_allclose(geometry.angles, dataset.angles, rtol=1e-6)


@pytest.mark.parametrize("suffix", [".npz", ".nxs"])
@pytest.mark.parametrize("offsets", _BAD_OFFSETS)
@pytest.mark.parametrize("poses", [False, True])
def test_public_scan_load_does_not_hide_malformed_saved_angle_offsets(
    tmp_path, suffix, offsets, poses
):
    dataset = _dataset("parallel", np.zeros(4))
    path = tmp_path / ("offsets" + suffix)
    save_dataset(path, dataset)
    # Inject malformed metadata without asking the writer to compress a scalar.
    if suffix == ".nxs":
        with h5py.File(path, "r+") as handle:
            key = "/entry/processing/tomojax/align/angle_offset_deg"
            del handle[key]
            handle.create_dataset(key, data=offsets)
    else:
        with np.load(path, allow_pickle=True) as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays["angle_offset_deg"] = offsets
        np.savez_compressed(path, **arrays)
    if poses:
        with pytest.raises(ValueError, match="angle_offset_deg"):
            tj.load(path, poses=True)
    else:
        np.testing.assert_allclose(tj.load(path, poses=False).angles, dataset.angles, rtol=1e-6)


@pytest.mark.parametrize("kind", ["parallel", "lamino", "cone", "segments"])
@pytest.mark.parametrize("offsets", [np.full(4, 2e-9), np.asarray([0.1, -0.2, 0.3, -0.4])])
def test_finite_offsets_even_below_allclose_tolerance_match_independent_poses(kind, offsets):
    dataset = _dataset(kind, offsets)
    original = dataset.angles.copy()
    _, _, geometry = build_geometry_from_dataset_metadata(dataset.geometry_inputs(), poses=True)
    expected_angles = original + offsets
    np.testing.assert_allclose(geometry.angles, expected_angles, rtol=1e-6)
    expected = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
    tilt = (
        np.eye(3) if kind == "parallel" else Rotation.from_euler("X", -27, degrees=True).as_matrix()
    )
    expected[:, :3, :3] = (
        tilt @ Rotation.from_euler("Z", expected_angles[:, None], degrees=True).as_matrix()
    )
    if kind in {"cone", "segments"}:
        expected[:, 0, 3] = 0.2
    if kind == "segments":
        expected[2:, 0, 3] = 0.4
    np.testing.assert_allclose(
        [geometry.pose_for_view(i) for i in range(4)], expected, rtol=0, atol=1e-6
    )
    np.testing.assert_array_equal(dataset.angles, original)


@pytest.mark.parametrize("offsets", [*_BAD_OFFSETS, np.full(4, 0.123456789)])
def test_baked_misalignment_schedule_is_provenance_not_a_second_correction(offsets):
    dataset = _dataset("parallel", offsets)
    dataset.misalign_spec = {"angle_offset_deg": "already baked into angles"}
    _, _, geometry = build_geometry_from_dataset_metadata(dataset.geometry_inputs(), poses=True)
    np.testing.assert_allclose(geometry.angles, dataset.angles, rtol=1e-6)
