"""Detector windows retain their physical pixel positions, including tilted planes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import tomojax as tj
from tomojax.geometry import ConeSegments, stack_view_poses
from tomojax.geometry.api import view_frames
from tomojax.io import ProjectionDataset, build_geometry_from_dataset_metadata


def _problem(rotations, structure):
    grid = tj.Grid(3, 2, 2, 0.7, 0.9, 1.1)
    detector = tj.Detector(11, 9, 0.43, 0.57, (0.21, -0.19))
    beam = tj.ConeBeam(
        18.0,
        30.0,
        detector_roll_deg=rotations[0],
        detector_pitch_deg=rotations[1],
        detector_yaw_deg=rotations[2],
        axis_offset=0.17,
    )
    first = tj.ConeGeometry(grid, detector, [13, 53, 97, 149, 217, 293], beam, tilt_deg=19)
    if structure == "segmented":
        second = replace(
            first,
            detector=replace(detector, center=(-0.31, 0.27)),
            beam=replace(beam, source_to_axis=20, source_to_detector=33, detector_roll_deg=-24),
        )
        geometry = ConeSegments((first, second))
    elif structure == "posed":
        params = np.array([[0.01, -0.02, 0.03, 0.11, -0.13, 0.17]] * 6, dtype=np.float32)
        record = ProjectionDataset(
            projections=np.zeros((6, 9, 11), np.float32),
            angles=np.asarray(first.angles, np.float32),
            detector=detector,
            grid=grid,
            geometry_type="cone",
            geometry_metadata=first.geometry_metadata(),
            align_params=params,
            align_gauge={"pose_translation_frame": "detector"},
        )
        _, _, geometry = build_geometry_from_dataset_metadata(record.geometry_inputs(), poses=True)
    else:
        geometry = first
    data = np.arange(len(geometry.angles) * 9 * 11, dtype=np.float32).reshape(-1, 9, 11)
    return tj.Scan(data, geometry, name="window fixture")


def _reference_frames(scan, row_offset, col_offset, rows, cols, factor):
    parts = scan.geometry.segments if isinstance(scan.geometry, ConeSegments) else (scan.geometry,)
    expected = []
    for part in parts:
        beam, detector = part.beam, part.detector
        # Independent Euler matrices, not ConeBeam.detector_frame or window helpers.
        rotation = (
            Rotation.from_euler("z", beam.detector_yaw_deg, degrees=True).as_matrix()
            @ Rotation.from_euler("x", beam.detector_pitch_deg, degrees=True).as_matrix()
            @ Rotation.from_euler("y", beam.detector_roll_deg, degrees=True).as_matrix()
        )
        u = rotation[:, 0] * detector.du
        v = rotation[:, 2] * detector.dv
        centre = np.array(
            [detector.center[0], beam.source_to_detector - beam.source_to_axis, detector.center[1]]
        )
        centre += (col_offset + (factor * cols - detector.nu) / 2) * u
        centre += (row_offset + (factor * rows - detector.nv) / 2) * v
        source = np.array([0, -beam.source_to_axis, 0])
        poses = np.asarray(stack_view_poses(part, len(part.angles)), np.float64)
        for pose in poses:
            turn, shift = pose[:3, :3], pose[:3, 3]
            expected.append(
                np.concatenate(
                    [
                        turn.T @ (source - shift),
                        turn.T @ (centre - shift),
                        turn.T @ (u * factor),
                        turn.T @ (v * factor),
                    ]
                )
            )
    return np.asarray(expected)


TILTED = (30, 20, -15)


@pytest.mark.parametrize("rotations", [(0, 0, 0), TILTED])
@pytest.mark.parametrize("structure", ["plain", "posed", "segmented"])
@pytest.mark.parametrize("operation", ["crop", "bin", "crop_then_bin"])
def test_detector_windows_match_independent_physical_frames(rotations, structure, operation):
    original = _problem(rotations, structure)
    if operation == "crop":
        actual = original.cropped(slice(1, 8), slice(2, 10))
        expected_data = original.projections[:, 1:8, 2:10]
        expected_frames = _reference_frames(original, 1, 2, 7, 8, 1)
    elif operation == "bin":
        actual = original.binned(3)
        expected_data = original.projections[:, :9, :9].reshape(-1, 3, 3, 3, 3).mean(axis=(2, 4))
        expected_frames = _reference_frames(original, 0, 0, 3, 3, 3)
    else:
        actual = original.cropped(slice(1, 8), slice(2, 10)).binned(3)
        expected_data = None
        expected_frames = _reference_frames(original, 1, 2, 2, 2, 3)
    np.testing.assert_allclose(view_frames(actual.geometry), expected_frames, rtol=1e-7, atol=2e-6)
    if expected_data is not None:
        np.testing.assert_array_equal(actual.projections, expected_data)
    np.testing.assert_array_equal(
        stack_view_poses(actual.geometry, len(actual.angles)),
        stack_view_poses(original.geometry, len(original.angles)),
    )
    assert actual.name == original.name
    assert actual.grid == original.grid


def test_raw_frame_crop_moves_counts_flat_dark_and_geometry_together():
    original = _problem(TILTED, "plain")
    frames = tj.Frames(
        original.projections.astype(np.uint16),
        original.geometry,
        flats=np.ones((2, 9, 11), np.uint16) * 2000,
        darks=np.ones((1, 9, 11), np.uint16),
        flat_positions=np.array([0, 8]),
        view_positions=np.array([0.5, 1.5, 3.5, 4.5, 6.5, 7.5]),
    )
    actual = frames.cropped(slice(1, 8), slice(2, 10))
    np.testing.assert_array_equal(actual.counts, frames.counts[:, 1:8, 2:10])
    np.testing.assert_array_equal(actual.flats, frames.flats[:, 1:8, 2:10])
    np.testing.assert_array_equal(actual.darks, frames.darks[:, 1:8, 2:10])
    np.testing.assert_array_equal(actual.flat_positions, frames.flat_positions)
    np.testing.assert_array_equal(actual.view_positions, frames.view_positions)
    expected = _reference_frames(original, 1, 2, 7, 8, 1)
    np.testing.assert_allclose(view_frames(actual.geometry), expected, rtol=1e-7, atol=2e-6)


@pytest.mark.parametrize("operation", ["crop", "bin"])
def test_pitched_detector_asymmetric_row_window_retains_lab_y(operation):
    original = _problem((0, 20, 0), "plain")
    geometry = replace(original.geometry, detector=replace(original.detector, nv=10))
    original = replace(original, geometry=geometry, projections=np.zeros((6, 10, 11), np.float32))
    if operation == "crop":
        actual = original.cropped(slice(2, 9), slice(1, 10))
        reference = _reference_frames(original, 2, 1, 7, 9, 1)
    else:
        actual = original.binned(3)
        reference = _reference_frames(original, 0, 0, 3, 3, 3)
    assert actual.geometry.beam.source_to_detector != original.geometry.beam.source_to_detector
    np.testing.assert_allclose(view_frames(actual.geometry), reference, rtol=1e-7, atol=2e-6)
