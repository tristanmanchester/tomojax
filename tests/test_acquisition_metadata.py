"""Acquisition metadata must describe the same physical views in every HDF5 path."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import tomojax as tj
from tomojax.io import load_dataset
from tomojax.io.api import inspect_dataset, load_nxtomo, locate_frames, validate_nxtomo

from ._helpers import write_projection_dataset, write_raw_nxtomo

pytestmark = pytest.mark.surface


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("standard_path", [False, True])
def test_standard_nxtomo_angles_are_loaded_and_found_before_other_detectors(
    tmp_path, raw, standard_path
):
    path = tmp_path / "scan.nxs"
    (write_raw_nxtomo if raw else write_projection_dataset)(path)
    with h5py.File(path, "a") as handle:
        if standard_path:
            handle.move(
                "entry/sample/transformations/rotation_angle", "entry/sample/rotation_angle"
            )
        handle.create_dataset("unrelated/rotation_angle", data=np.zeros(4 if raw else 2))
    np.testing.assert_allclose(load_nxtomo(str(path)).angles, [0, 0, 90, 0] if raw else [0, 90])
    np.testing.assert_allclose(tj.load(path).angles, [0, 90])
    np.testing.assert_allclose(
        tj.load_frames(path, **({} if raw else {"flats": 100, "darks": 0})).geometry.angles, [0, 90]
    )
    angle_path = (
        "/entry/sample/rotation_angle"
        if standard_path
        else "/entry/sample/transformations/rotation_angle"
    )
    assert inspect_dataset(path)["angles"]["path"] == angle_path
    assert validate_nxtomo(str(path))["issues"] == []


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
    with h5py.File(path, "a") as handle:
        group = handle["entry/instrument/detector"]
        sinogram = group["data"][:, 0, :]
        del group["data"]
        group["data"] = sinogram
    assert validate_nxtomo(str(path))["issues"] == []


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


def test_raw_load_can_ignore_saved_poses_and_angle_offsets(tmp_path):
    path = tmp_path / "raw.nxs"
    write_raw_nxtomo(path)
    params = np.asarray([[0, 0, 0, 0.25, 0.1, -0.2], [0, 0, 0, -0.4, -0.1, 0.3]], np.float32)
    with h5py.File(path, "a") as handle:
        align = handle.create_group("entry/processing/tomojax/align")
        align.create_dataset("thetas", data=params)
        align.create_dataset("angle_offset_deg", data=[3, -7])
        align.attrs["gauge_fix_json"] = '{"pose_translation_frame": "detector"}'
    nominal = tj.load(path, poses=False)
    assert nominal.poses is None
    np.testing.assert_array_equal(nominal.angles, [0, 90])
    corrected = tj.load(path)
    np.testing.assert_allclose(corrected.poses, params, atol=1e-7)
    np.testing.assert_array_equal(corrected.angles, [3, 83])


@pytest.mark.parametrize("poses", [False, True])
def test_processed_load_keeps_only_sample_frames_and_their_saved_poses(tmp_path, poses):
    path = tmp_path / "processed.nxs"
    grid = tj.Grid(3, 3, 2, 0.8, 1.1, 0.9)
    detector = tj.Detector(3, 2, 0.9, 1.0, (0.13, -0.24))
    angles = np.asarray([0.0, -37.0, 90.0, 113.0])
    geometry = tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(40, 60))
    values = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3) / 20
    tj.save(path, tj.Scan(values, geometry))
    params = np.zeros((4, 6), np.float32)
    params[1, 3:] = [0.25, 0.1, -0.2]
    params[3, 3:] = [-0.4, -0.1, 0.3]
    with h5py.File(path, "a") as handle:
        handle["entry/instrument/detector/image_key"][...] = [3, 0, 3, 0]
        handle["entry/sample/transformations/rotation_angle"][[0, 2]] = np.nan
        align = handle.create_group("entry/processing/tomojax/align")
        align.create_dataset("thetas", data=params)
        align.create_dataset("angle_offset_deg", data=[0, 3, 0, -7])
        align.attrs["gauge_fix_json"] = '{"pose_translation_frame": "detector"}'
    loaded = tj.load(path, poses=poses)
    np.testing.assert_array_equal(loaded.projections, values[[1, 3]])
    np.testing.assert_array_equal(loaded.angles, [-34.0, 106.0] if poses else [-37.0, 113.0])
    assert (loaded.poses is not None) == poses
    saved = tmp_path / "samples.nxs"
    tj.save(saved, loaded)
    np.testing.assert_array_equal(tj.load(saved).projections, values[[1, 3]])
    np.testing.assert_array_equal(tj.load(saved).angles, loaded.angles)


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
        tj.load_frames(path, flats=1000, detector=tj.Detector(3, 2, 1, 1), darks=0)
        if layout == "beamline"
        else tj.load_frames(path, detector=tj.Detector(3, 2, 1, 1), darks=0)
    )
    scan = frames.corrected()
    np.testing.assert_allclose(scan.angles, [0, 90], atol=1e-5)
    np.testing.assert_allclose(scan.projections[:, 0, 0], -np.log([0.5, 0.25]), rtol=1e-6)


@pytest.mark.parametrize("marked", [1, 2])
def test_separate_fields_fill_only_calibrations_missing_from_image_key(tmp_path, marked):
    path = tmp_path / "hybrid.h5"
    levels = (100 if marked == 1 else 4, 52, 28)
    counts = np.stack([np.full((2, 3), level, np.uint16) for level in levels])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("exchange/data", data=counts)
        handle.create_dataset("exchange/image_key", data=[marked, 0, 0])
        handle.create_dataset("exchange/theta", data=[np.nan, 0, 90])
        # The marked calibration takes precedence; the sibling field fills the other.
        white = np.full((1, 2, 3), 999 if marked == 1 else 100)
        dark = np.full((1, 2, 3), 40 if marked == 2 else 4)
        handle.create_dataset("exchange/data_white", data=white)
        handle.create_dataset("exchange/data_dark", data=dark)
    located = locate_frames(str(path))
    assert (located.flats is None) == (marked == 1)
    assert (located.darks is None) == (marked == 2)
    for read in (tj.load, tj.load_frames):
        with pytest.raises(ValueError, match="detector="):
            read(path)
    for scan in (tj.load_frames(path, detector=tj.Detector(3, 2, 1, 1)).corrected(),):
        np.testing.assert_array_equal(scan.angles, [0, 90])
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
    truth = np.random.default_rng(6).uniform(size=(3, 4, 5)).astype(np.float32)
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


def _write_beamline_frames(path):
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry1/tomo")
        entry.create_dataset("data", data=np.stack([np.full((2, 2), v) for v in (500, 1000, 250)]))
        entry.create_dataset("image_key", data=[0, 1, 0])
        entry.create_dataset("rotation_angle", data=[0.0, np.nan, 90.0]).attrs["units"] = "deg"


@pytest.mark.parametrize("name", ["image_key", "rotation_angle"])
@pytest.mark.parametrize("shape", [(1, 3), (3, 1), (2,), ()])
def test_unique_malformed_beamline_metadata_is_not_treated_as_absent(tmp_path, name, shape):
    path = tmp_path / "beamline.h5"
    _write_beamline_frames(path)
    with h5py.File(path, "a") as handle:
        entry = handle["entry1/tomo"]
        del entry[name]
        entry.create_dataset(name, data=np.zeros(shape, dtype=np.int32))
    for read in (lambda p: locate_frames(str(p)), tj.load_frames):
        with pytest.raises(ValueError, match="shape"):
            read(path)


def test_beamline_metadata_shape_still_disambiguates_different_detectors(tmp_path):
    path = tmp_path / "beamline.h5"
    _write_beamline_frames(path)
    with h5py.File(path, "a") as handle:
        entry = handle.create_group("entry2/tomo")
        entry.create_dataset("data", data=np.zeros((4, 2, 2)))
        entry.create_dataset("image_key", data=[0, 0, 0, 0])
        entry.create_dataset("rotation_angle", data=[0, 45, 90, 135]).attrs["units"] = "deg"
    first = tj.load_frames(
        path, data_path="/entry1/tomo/data", detector=tj.Detector(2, 2, 1, 1), darks=0
    ).corrected()
    np.testing.assert_array_equal(first.angles, [0, 90])
    np.testing.assert_allclose(first.projections[:, 0, 0], -np.log([0.5, 0.25]))
    second = tj.load_frames(
        path,
        data_path="/entry2/tomo/data",
        flats=1,
        detector=tj.Detector(2, 2, 1, 1),
        darks=0,
    )
    assert second.views == 4
    np.testing.assert_array_equal(second.geometry.angles, [0, 45, 90, 135])


@pytest.mark.parametrize("name", ["image_key", "rotation_angle"])
@pytest.mark.parametrize("matching", [False, True])
def test_ambiguous_beamline_metadata_requires_an_explicit_path(tmp_path, name, matching):
    path = tmp_path / "beamline.h5"
    _write_beamline_frames(path)
    with h5py.File(path, "a") as handle:
        if not matching:
            entry = handle["entry1/tomo"]
            del entry[name]
            entry.create_dataset(name, data=np.zeros(4, np.int32)).attrs["units"] = "deg"
        handle.create_dataset(f"entry2/{name}", data=np.zeros(3 if matching else 5, np.int32))
    with pytest.raises(KeyError, match=f"several '{name}' datasets"):
        tj.load_frames(path, darks=0)
    keyword = "image_key_path" if name == "image_key" else "angles_path"
    arguments = {keyword: f"/entry1/tomo/{name}", "detector": tj.Detector(2, 2, 1, 1)}
    if matching:
        scan = tj.load_frames(path, **arguments, darks=0).corrected()
        np.testing.assert_array_equal(scan.angles, [0, 90])
        np.testing.assert_allclose(scan.projections[:, 0, 0], -np.log([0.5, 0.25]))
    else:
        with pytest.raises(ValueError, match="shape"):
            tj.load_frames(path, **arguments, darks=0)


def test_a_missing_beamline_image_key_remains_supported(tmp_path):
    path = tmp_path / "beamline.h5"
    _write_beamline_frames(path)
    with h5py.File(path, "a") as handle:
        del handle["entry1/tomo/image_key"]
        handle["entry1/tomo/rotation_angle"][...] = [0, 45, 90]
    frames = tj.load_frames(path, flats=1000, detector=tj.Detector(2, 2, 1, 1), darks=0)
    assert frames.views == 3 and frames.flats is None
    np.testing.assert_array_equal(frames.geometry.angles, [0, 45, 90])


def test_data_exchange_angles_take_priority_over_unmatched_rotation_datasets(tmp_path):
    path = tmp_path / "exchange.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("exchange/data", data=np.full((2, 2, 2), 500))
        handle.create_dataset("exchange/data_white", data=np.full((1, 2, 2), 1000))
        handle.create_dataset("exchange/theta", data=[0, np.pi / 2]).attrs["units"] = "rad"
        handle.create_dataset("unrelated/rotation_angle", data=[0, 45, 90])
    scan = tj.load_frames(path, detector=tj.Detector(2, 2, 1, 1), darks=0).corrected()
    np.testing.assert_allclose(scan.angles, [0, 90], atol=1e-5)
    np.testing.assert_allclose(scan.projections, np.log(2.0))


def test_missing_angles_are_not_replaced_by_zero_views(tmp_path):
    path = tmp_path / "scan.nxs"
    write_projection_dataset(path)
    with h5py.File(path, "a") as handle:
        del handle["entry/sample/transformations/rotation_angle"]
    assert load_nxtomo(str(path)).metadata.angles is None
    for read in (tj.load, tj.load_frames):
        with pytest.raises(ValueError, match="angles="):
            read(path)
    frames = tj.load_frames(path, angles=[11, 72], flats=100, darks=0)
    np.testing.assert_array_equal(frames.geometry.angles, [11, 72])


@pytest.mark.parametrize("raw", [False, True])
def test_missing_laminography_tilt_requires_geometry_before_materialisation(tmp_path, raw):
    path = tmp_path / "scan.nxs"
    (write_raw_nxtomo if raw else write_projection_dataset)(path)
    with h5py.File(path, "a") as handle:
        handle["entry/geometry"].attrs["type"] = "lamino"
    for read in (tj.load, tj.load_frames):
        with pytest.raises(ValueError, match="geometry=.*tilt_deg"):
            read(path)
    detector = tj.Detector(2 if raw else 4, 2, 0.7, 0.8, (1.2, -0.3))
    geometry = tj.LaminographyGeometry(
        tj.Grid(4, 4, 2, 0.7, 0.7, 0.8), detector, [0, 90], tilt_deg=57
    )
    frames = tj.load_frames(path, geometry=geometry, **({} if raw else {"flats": 100, "darks": 0}))
    assert frames.geometry is geometry


def test_npz_detector_without_center_stays_incomplete_until_overridden(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(
        path,
        projections=np.ones((2, 2, 3), np.float32),
        thetas_deg=[0, 90],
        detector={"nu": 3, "nv": 2, "du": 0.2, "dv": 0.3},
    )
    for read in (tj.load, tj.load_frames):
        with pytest.raises(ValueError, match="detector="):
            read(path)
    detector = tj.Detector(3, 2, 0.2, 0.3, (0.1, -0.2))
    assert tj.load_frames(path, detector=detector, flats=100, darks=0).geometry.detector == detector


def test_container_conversion_does_not_create_absent_rotation_measurements(tmp_path):
    from tomojax.io.api import convert_dataset

    path = tmp_path / "without_angles.npz"
    np.savez(
        path, projections=np.ones((2, 2, 3), np.float32), detector=tj.Detector(3, 2, 1, 1).to_dict()
    )
    output = tmp_path / "converted.nxs"
    convert_dataset(path, output)
    with pytest.raises(ValueError, match="angles="):
        tj.load(output)
    np.testing.assert_array_equal(
        tj.load_frames(output, angles=[2, 82], flats=100, darks=0).geometry.angles, [2, 82]
    )


def test_generic_motor_without_units_requires_explicit_angles(tmp_path):
    path = tmp_path / "beamline.h5"
    _write_beamline_frames(path)
    with h5py.File(path, "a") as handle:
        del handle["entry1/tomo/rotation_angle"].attrs["units"]
    assert locate_frames(str(path)).angles is None
    with pytest.raises(ValueError, match="angles="):
        tj.load_frames(path, detector=tj.Detector(2, 2, 1, 1), darks=0)
    frames = tj.load_frames(path, angles=[11, 72], detector=tj.Detector(2, 2, 1, 1), darks=0)
    np.testing.assert_array_equal(frames.geometry.angles, [11, 72])
