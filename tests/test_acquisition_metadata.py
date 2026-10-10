"""Acquisition metadata must describe the same physical views in every HDF5 path."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import tomojax as tj
from tomojax.io import load_dataset
from tomojax.io.api import load_nxtomo, locate_frames

from ._helpers import write_projection_dataset, write_raw_nxtomo

pytestmark = pytest.mark.surface


@pytest.mark.parametrize("units", ["rad", "radian", " RADIANS ", np.bytes_("rad")])
def test_processed_nxtomo_angles_are_converted_to_degrees(tmp_path, units):
    path = tmp_path / "scan.nxs"
    dataset = write_projection_dataset(path, angles=np.asarray([-37.0, 113.0]))
    with h5py.File(path, "a") as handle:
        angles = handle["entry/sample/transformations/rotation_angle"]
        angles[...] = np.radians(dataset.angles)
        angles.attrs["units"] = units
    np.testing.assert_allclose(load_nxtomo(str(path)).angles, dataset.angles, atol=1e-5)
    np.testing.assert_allclose(load_dataset(path).angles, dataset.angles, atol=1e-5)
    np.testing.assert_allclose(tj.load(path).angles, dataset.angles, atol=1e-5)


@pytest.mark.parametrize("units", [None, "degree", "degrees", " DEG "])
def test_degree_angles_and_missing_units_keep_their_values(tmp_path, units):
    path = tmp_path / "scan.nxs"
    dataset = write_projection_dataset(path, angles=np.asarray([-37.0, 413.0]))
    with h5py.File(path, "a") as handle:
        angles = handle["entry/sample/transformations/rotation_angle"]
        del angles.attrs["units"]
        if units is not None:
            angles.attrs["units"] = units
    np.testing.assert_array_equal(load_dataset(path).angles, dataset.angles)
    np.testing.assert_array_equal(locate_frames(str(path)).angles, dataset.angles)


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize(
    "values",
    [np.asarray([0.9, 0.0]), np.asarray([0, 99]), np.asarray([0, 2**32], np.uint64)],
)
def test_invalid_image_keys_are_not_cast_or_silently_dropped(tmp_path, raw, values):
    path = tmp_path / "scan.nxs"
    if raw:
        write_raw_nxtomo(path)
        values = np.concatenate([values, [0, 2]]).astype(values.dtype)
    else:
        write_projection_dataset(path)
    with h5py.File(path, "a") as handle:
        detector = handle["entry/instrument/detector"]
        del detector["image_key"]
        detector.create_dataset("image_key", data=values)
    for read in (tj.load, tj.load_frames, load_dataset):
        with pytest.raises(ValueError, match="image_key"):
            read(path)


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("invalid", ["nonfinite", "matrix", "units", "overflow"])
def test_invalid_angles_fail_instead_of_changing_the_acquisition(tmp_path, raw, invalid):
    path = tmp_path / "scan.nxs"
    (write_raw_nxtomo if raw else write_projection_dataset)(path)
    count = 4 if raw else 2
    values = np.arange(count, dtype=np.float64)
    if invalid == "nonfinite":
        values[0] = np.nan
    elif invalid == "overflow":
        values[0] = 1e40
    elif invalid == "matrix":
        values = values.reshape(1, count)
    with h5py.File(path, "a") as handle:
        transformations = handle["entry/sample/transformations"]
        del transformations["rotation_angle"]
        angles = transformations.create_dataset("rotation_angle", data=values)
        angles.attrs["units"] = "turns" if invalid == "units" else "degree"
    for read in (tj.load, tj.load_frames, load_dataset):
        with pytest.raises(ValueError, match="angles|units"):
            read(path)


def test_calibration_frames_need_no_finite_rotation_angle(tmp_path):
    path = tmp_path / "raw.nxs"
    write_raw_nxtomo(path)
    with h5py.File(path, "a") as handle:
        angles = handle["entry/sample/transformations/rotation_angle"]
        angles[...] = [0.0, np.nan, np.pi / 2, np.nan]
        angles.attrs["units"] = "rad"
    for scan in (tj.load(path), tj.load_frames(path).corrected()):
        np.testing.assert_allclose(scan.angles, [0, 90], atol=1e-5)
        np.testing.assert_allclose(scan.projections[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6)


@pytest.mark.parametrize("layout", ["beamline", "exchange"])
def test_external_hdf5_layouts_decode_the_same_angles_and_calibration(tmp_path, layout):
    path = tmp_path / "frames.h5"
    counts = np.stack([np.full((2, 3), v, np.uint16) for v in (500, 250)])
    with h5py.File(path, "w") as handle:
        if layout == "beamline":
            handle.create_dataset("entry1/tomo/data", data=counts)
            angles = handle.create_dataset("entry1/tomo/rotation_angle", data=[0, np.pi / 2])
        else:
            handle.create_dataset("exchange/data", data=counts)
            handle.create_dataset("exchange/data_white", data=np.full((1, 2, 3), 1000))
            handle.create_dataset("exchange/data_dark", data=np.zeros((1, 2, 3)))
            angles = handle.create_dataset("exchange/theta", data=[0, np.pi / 2])
        angles.attrs["units"] = "radians"
    frames = (
        tj.load_frames(path, white_level=1000) if layout == "beamline" else tj.load_frames(path)
    )
    scan = frames.corrected()
    np.testing.assert_allclose(scan.angles, [0, 90], atol=1e-5)
    np.testing.assert_allclose(scan.projections[:, 0, 0], -np.log([0.5, 0.25]), rtol=1e-6)


def test_raw_nxtomo_does_not_discard_an_invalid_geometry_type(tmp_path):
    path = tmp_path / "raw.nxs"
    write_raw_nxtomo(path)
    with h5py.File(path, "a") as handle:
        handle["entry/geometry"].attrs["type"] = "unknown"
    for read in (tj.load, tj.load_frames):
        with pytest.raises(ValueError, match="geometry_type"):
            read(path)


@pytest.mark.numerical
def test_radian_file_recovers_the_same_volume_and_pose_geometry(tmp_path):
    grid = tj.Grid(5, 4, 3, 0.8, 1.1, 1.3)
    detector = tj.Detector(8, 5, 0.9, 1.2, (0.17, -0.23))
    angles = np.linspace(-17, 163, 18, endpoint=False)
    geometry = tj.ParallelGeometry(grid, detector, angles)
    truth = np.random.default_rng(6).uniform(size=(5, 4, 3)).astype(np.float32)
    scan = tj.Scan(np.asarray(tj.project(geometry, truth)), geometry)
    path = tmp_path / "radians.nxs"
    tj.save(path, scan)
    with h5py.File(path, "a") as handle:
        stored = handle["entry/sample/transformations/rotation_angle"]
        stored[...] = np.radians(angles)
        stored.attrs["units"] = "radians"
    loaded = tj.load(path)
    for view in range(len(angles)):
        np.testing.assert_allclose(
            loaded.geometry.pose_for_view(view), geometry.pose_for_view(view), atol=3e-7
        )
    direct = np.asarray(tj.reconstruct(scan, "cgls", iterations=40).volume)
    actual = np.asarray(tj.reconstruct(loaded, "cgls", iterations=40).volume)
    np.testing.assert_allclose(actual, direct, rtol=3e-4, atol=3e-4)
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.06
