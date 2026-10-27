from __future__ import annotations

import json
from pathlib import Path

import h5py
import imageio.v3 as iio
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.geometry import Detector, Grid
from tomojax.geometry.api import detector_grid_from_geometry_inputs
from tomojax.io import (
    ProjectionDataset,
    load_dataset,
    load_tiff_stack,
    save_dataset,
    validate_dataset,
)
from tomojax.io.api import (
    convert_dataset,
    inspect_dataset,
    load_nxtomo,
    locate_frames,
    projection_stats,
    save_nxtomo,
)

from ._helpers import make_projection_dataset, write_projection_dataset

pytestmark = pytest.mark.surface


@pytest.mark.parametrize("gtype", ["lamino", "laminography", " LAMINOGRAPHY "])
def test_projection_dataset_roundtrips_nxtomo_with_solver_metadata(tmp_path: Path, gtype) -> None:
    path = tmp_path / "scan.nxs"
    detector = Detector(nu=4, nv=2, du=0.5, dv=0.75, center=(1.0, -2.0))
    grid = Grid(nx=4, ny=4, nz=2, vx=1.0, vy=1.0, vz=2.0)
    align_params = np.ones((2, 5), dtype=np.float32)
    dataset = make_projection_dataset(
        detector=detector,
        grid=grid,
        geometry_type=gtype,
        geometry_metadata={"tilt_deg": 55.0, "tilt_about": "x", "detector_roll_deg": 1.25},
    )
    dataset.angle_offset_deg = np.asarray([1.0, 2.0], dtype=np.float32)
    dataset.align_params = align_params
    dataset.align_gauge = {"mode": "mean_zero"}

    save_dataset(path, dataset)
    loaded = load_dataset(path)
    with h5py.File(path) as file:
        axis = file["entry/sample/transformations/rotation_axis"][()]
    np.testing.assert_allclose(axis, [0, np.sin(np.deg2rad(55)), np.cos(np.deg2rad(55))])

    assert isinstance(loaded, ProjectionDataset)
    np.testing.assert_allclose(loaded.projections, dataset.projections)
    np.testing.assert_allclose(loaded.angles, dataset.angles)
    assert loaded.detector == detector
    assert loaded.grid == grid
    geometry_inputs = loaded.geometry_inputs()
    assert geometry_inputs["geometry_type"] == "lamino"
    assert geometry_inputs["tilt_deg"] == pytest.approx(55.0)
    assert geometry_inputs["tilt_about"] == "x"
    assert geometry_inputs["detector_roll_deg"] == pytest.approx(1.25)
    np.testing.assert_allclose(geometry_inputs["angle_offset_deg"], [1.0, 2.0])
    np.testing.assert_allclose(geometry_inputs["align_params"], align_params)
    assert geometry_inputs["align_gauge"] == {"mode": "mean_zero"}


def test_projection_payload_save_load_preserves_metadata_copy(tmp_path: Path) -> None:
    path = tmp_path / "payload.nxs"
    dataset = make_projection_dataset(geometry_metadata={"axis_unit_lab": [0.0, 0.0, 1.0]})
    save_nxtomo(path, projections=dataset.projections, metadata=dataset.to_nxtomo_metadata())

    payload = load_dataset(path)

    copied = payload.copy_metadata()
    assert copied.sample_name == "fixture"
    assert copied.detector == payload.detector
    assert copied.grid == payload.grid
    assert payload.geometry_inputs()["axis_unit_lab"] == [0.0, 0.0, 1.0]


def test_json_normalization_reports_array_conversion_failures() -> None:
    from tomojax.io.api import normalize_json

    class BrokenArray(np.ndarray):
        def tolist(self) -> object:
            raise RuntimeError("tolist failed")

    value = np.asarray([1.0], dtype=np.float32).view(BrokenArray)

    with pytest.raises(TypeError, match="could not normalize NumPy array"):
        normalize_json(value)


def test_dataset_roundtrips_npz_and_converts_to_nxtomo(tmp_path: Path) -> None:
    npz_path = tmp_path / "scan.npz"
    nxs_path = tmp_path / "scan.nxs"
    dataset = make_projection_dataset()

    save_dataset(npz_path, dataset)
    loaded_npz = load_dataset(npz_path)
    convert_dataset(npz_path, nxs_path)
    loaded_nxs = load_dataset(nxs_path)

    np.testing.assert_allclose(loaded_npz.projections, dataset.projections)
    np.testing.assert_allclose(loaded_nxs.projections, dataset.projections)
    assert validate_dataset(nxs_path)["issues"] == []


def test_files_keep_their_angle_and_detector_centre_keys(tmp_path: Path) -> None:
    # An .npz in the layout earlier versions wrote: angles under `thetas_deg`
    # and the detector centre under `det_center`.
    old_path = tmp_path / "old.npz"
    detector = {"nu": 4, "nv": 2, "du": 1.0, "dv": 1.0, "det_center": [0.5, -1.0]}
    np.savez_compressed(
        old_path,
        projections=np.zeros((2, 2, 4), np.float32),
        thetas_deg=np.asarray([0.0, 90.0], np.float32),
        detector=detector,  # pyright: ignore[reportArgumentType]
        geometry_type="parallel",
    )

    loaded = load_dataset(old_path)

    np.testing.assert_allclose(loaded.angles, [0.0, 90.0])
    assert loaded.detector == Detector(nu=4, nv=2, du=1.0, dv=1.0, center=(0.5, -1.0))

    npz_path, nxs_path = tmp_path / "new.npz", tmp_path / "new.nxs"
    save_dataset(npz_path, loaded)
    save_dataset(nxs_path, loaded)
    with np.load(npz_path, allow_pickle=True) as saved:
        np.testing.assert_allclose(saved["thetas_deg"], [0.0, 90.0])
        assert saved["detector"].item()["det_center"] == [0.5, -1.0]
    with h5py.File(nxs_path, "r") as saved:
        angles = saved["/entry/sample/transformations/rotation_angle"][()]
        np.testing.assert_allclose(angles, [0.0, 90.0])
        meta = json.loads(saved["/entry/instrument/detector"].attrs["detector_meta_json"])
        assert meta["det_center"] == [0.5, -1.0]
    assert load_dataset(npz_path).detector == loaded.detector
    assert load_dataset(nxs_path).detector == loaded.detector


def test_nxtomo_loader_does_not_synthesize_grid_from_volume_only(tmp_path: Path) -> None:
    path = tmp_path / "missing_grid.nxs"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["definition"] = "NXtomo"
        detector = entry.create_group("instrument").create_group("detector")
        detector.create_dataset("data", data=np.zeros((2, 3, 4), dtype=np.float32))
        detector.create_dataset("image_key", data=np.zeros((2,), dtype=np.int32))
        sample = entry.create_group("sample")
        sample.create_dataset("name", data="fixture")
        transformations = sample.create_group("transformations")
        rotation_angle = transformations.create_dataset(
            "rotation_angle",
            data=np.asarray([0.0, 90.0], dtype=np.float32),
        )
        rotation_angle.attrs["units"] = "degree"
        tomojax = entry.create_group("processing").create_group("tomojax")
        tomojax.create_dataset("volume", data=np.zeros((5, 6, 7), dtype=np.float32))

    loaded = load_nxtomo(str(path))

    assert loaded.volume is not None
    assert loaded.grid is None
    assert loaded.volume_axes_order == "unknown"
    assert loaded.disk_volume_axes_order == "unknown"
    assert loaded.volume_axes_source == "heuristic"


def test_nxtomo_loader_marks_equal_xz_volume_axes_ambiguous(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous_axes.nxs"
    grid = Grid(nx=5, ny=6, nz=5, vx=1.0, vy=1.0, vz=1.0)
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["definition"] = "NXtomo"
        entry.attrs["grid_meta_json"] = json.dumps(grid.to_dict())
        detector = entry.create_group("instrument").create_group("detector")
        detector.create_dataset("data", data=np.zeros((2, 3, 4), dtype=np.float32))
        detector.create_dataset("image_key", data=np.zeros((2,), dtype=np.int32))
        sample = entry.create_group("sample")
        sample.create_dataset("name", data="fixture")
        transformations = sample.create_group("transformations")
        rotation_angle = transformations.create_dataset(
            "rotation_angle",
            data=np.asarray([0.0, 90.0], dtype=np.float32),
        )
        rotation_angle.attrs["units"] = "degree"
        tomojax = entry.create_group("processing").create_group("tomojax")
        tomojax.create_dataset("volume", data=np.zeros((5, 6, 5), dtype=np.float32))

    loaded = load_nxtomo(str(path))

    assert loaded.volume is not None
    assert loaded.grid == grid.to_dict()
    assert loaded.volume.shape == (5, 6, 5)
    assert loaded.volume_axes_order == "unknown"
    assert loaded.disk_volume_axes_order == "unknown"
    assert loaded.volume_axes_source == "heuristic"


def test_saving_unknown_axis_volume_reports_actionable_error(tmp_path: Path) -> None:
    path = tmp_path / "missing_axes.nxs"
    out_path = tmp_path / "out.nxs"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["definition"] = "NXtomo"
        detector = entry.create_group("instrument").create_group("detector")
        detector.create_dataset("data", data=np.zeros((2, 3, 4), dtype=np.float32))
        detector.create_dataset("image_key", data=np.zeros((2,), dtype=np.int32))
        sample = entry.create_group("sample")
        sample.create_dataset("name", data="fixture")
        transformations = sample.create_group("transformations")
        rotation_angle = transformations.create_dataset(
            "rotation_angle",
            data=np.asarray([0.0, 90.0], dtype=np.float32),
        )
        rotation_angle.attrs["units"] = "degree"
        tomojax = entry.create_group("processing").create_group("tomojax")
        tomojax.create_dataset("volume", data=np.zeros((5, 6, 7), dtype=np.float32))

    loaded = load_nxtomo(str(path))

    with pytest.raises(ValueError, match="Set metadata.volume_axes_order before saving"):
        save_nxtomo(
            out_path,
            projections=loaded.projections,
            metadata=loaded.copy_metadata(),
        )


def test_unknown_axis_in_place_save_preserves_existing_volume(tmp_path: Path) -> None:
    path = tmp_path / "missing_axes.nxs"
    original_volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["definition"] = "NXtomo"
        detector = entry.create_group("instrument").create_group("detector")
        detector.create_dataset("data", data=np.zeros((2, 3, 4), dtype=np.float32))
        detector.create_dataset("image_key", data=np.zeros((2,), dtype=np.int32))
        sample = entry.create_group("sample")
        sample.create_dataset("name", data="fixture")
        transformations = sample.create_group("transformations")
        rotation_angle = transformations.create_dataset(
            "rotation_angle",
            data=np.asarray([0.0, 90.0], dtype=np.float32),
        )
        rotation_angle.attrs["units"] = "degree"
        tomojax = entry.create_group("processing").create_group("tomojax")
        tomojax.create_dataset("volume", data=original_volume)

    loaded = load_nxtomo(str(path))

    with pytest.raises(ValueError, match="Set metadata.volume_axes_order before saving"):
        save_nxtomo(
            path,
            projections=loaded.projections,
            metadata=loaded.copy_metadata(),
        )

    with h5py.File(path, "r") as handle:
        saved_volume = handle["entry/processing/tomojax/volume"][:]
    np.testing.assert_array_equal(saved_volume, original_volume)


def test_copy_metadata_preserves_disk_volume_axes_order(tmp_path: Path) -> None:
    original_path = tmp_path / "original.nxs"
    resaved_path = tmp_path / "resaved.nxs"
    dataset = make_projection_dataset()
    dataset.volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    save_dataset(original_path, dataset)
    loaded = load_nxtomo(str(original_path))
    assert loaded.disk_volume_axes_order == "zyx"

    save_nxtomo(
        resaved_path,
        projections=loaded.projections,
        metadata=loaded.copy_metadata(),
    )

    resaved = load_nxtomo(str(resaved_path))
    assert resaved.disk_volume_axes_order == "zyx"


def test_copy_metadata_preserves_user_volume_axes_order_change(tmp_path: Path) -> None:
    original_path = tmp_path / "original.nxs"
    dataset = make_projection_dataset()
    dataset.volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    save_dataset(original_path, dataset)
    loaded = load_nxtomo(str(original_path))
    assert loaded.disk_volume_axes_order == "zyx"

    loaded.metadata.volume_axes_order = "xyz"
    copied = loaded.copy_metadata()

    assert copied.volume_axes_order == "xyz"


def test_copy_metadata_copies_array_backed_fields(tmp_path: Path) -> None:
    path = tmp_path / "scan.nxs"
    dataset = make_projection_dataset()
    dataset.volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    dataset.align_params = np.ones((2, 5), dtype=np.float32)
    dataset.angle_offset_deg = np.asarray([0.25, -0.25], dtype=np.float32)

    direct_copy = dataset.copy_metadata()
    assert not np.shares_memory(direct_copy.angles, dataset.angles)
    assert direct_copy.volume is not None
    assert not np.shares_memory(direct_copy.volume, dataset.volume)
    assert direct_copy.align_params is not None
    assert not np.shares_memory(direct_copy.align_params, dataset.align_params)
    assert direct_copy.angle_offset_deg is not None
    assert not np.shares_memory(direct_copy.angle_offset_deg, dataset.angle_offset_deg)

    save_dataset(path, dataset)
    loaded = load_nxtomo(str(path))
    loaded_copy = loaded.copy_metadata()
    assert loaded_copy.angles is not None
    assert loaded.metadata.angles is not None
    assert not np.shares_memory(loaded_copy.angles, loaded.metadata.angles)
    assert loaded_copy.image_key is not None
    assert loaded.metadata.image_key is not None
    assert not np.shares_memory(loaded_copy.image_key, loaded.metadata.image_key)
    assert loaded_copy.volume is not None
    assert loaded.metadata.volume is not None
    assert not np.shares_memory(loaded_copy.volume, loaded.metadata.volume)
    assert loaded_copy.align_params is not None
    assert loaded.metadata.align_params is not None
    assert not np.shares_memory(loaded_copy.align_params, loaded.metadata.align_params)
    assert loaded_copy.angle_offset_deg is not None
    assert loaded.metadata.angle_offset_deg is not None
    assert not np.shares_memory(loaded_copy.angle_offset_deg, loaded.metadata.angle_offset_deg)


@pytest.mark.parametrize("single_row", [False, True])
@pytest.mark.parametrize("compression", [None, "gzip"])
def test_noncontiguous_lazy_frame_reads_match_cropped_numpy_runs(
    tmp_path: Path, single_row: bool, compression: str | None
) -> None:
    path = tmp_path / "frames.h5"
    values = np.arange(9 * 7 * 11).reshape(9, 7, 11).astype(np.uint16)
    values = values[:, :1, :] if single_row else values
    stored = values[:, 0, :] if single_row else values
    rows = slice(0, 1) if single_row else slice(1, 7)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/scan/data", data=stored, chunks=True, compression=compression)
    stack = locate_frames(str(path)).stack.window(rows, slice(2, 10))
    indices = np.asarray([0, 1, 3, 5, 6, 7])
    result = stack[indices]
    np.testing.assert_array_equal(result, values[indices, rows, 2:10])
    np.testing.assert_array_equal(stack[2:4], values[2:4, rows, 2:10])
    np.testing.assert_array_equal(stack[-1], values[-1, rows, 2:10])
    assert stack[:0].shape == (0, stack.rows, stack.cols)
    assert inspect_dataset(path)["projection"]["shape"] == list(values.shape)
    assert result.dtype == values.dtype


def test_read_json_object_reports_path_for_malformed_json(tmp_path: Path) -> None:
    from tomojax.io.api import read_json_object

    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"invalid JSON in {path}"):
        read_json_object(path)


def test_load_tiff_stack_requires_explicit_angles_and_sorts_files(tmp_path: Path) -> None:
    stack_dir = tmp_path / "tiffs"
    stack_dir.mkdir()
    iio.imwrite(stack_dir / "proj10.tif", np.full((2, 3), 10.0, dtype=np.float32))
    iio.imwrite(stack_dir / "proj2.tif", np.full((2, 3), 2.0, dtype=np.float32))
    iio.imwrite(stack_dir / "proj1.tif", np.full((2, 3), 1.0, dtype=np.float32))

    dataset = load_tiff_stack(stack_dir, angles=[0.0, 45.0, 90.0])

    assert dataset.source_format == "tiff_stack"
    assert dataset.projections.shape == (3, 2, 3)
    np.testing.assert_allclose(dataset.projections[:, 0, 0], [1.0, 2.0, 10.0])
    np.testing.assert_allclose(dataset.angles, [0.0, 45.0, 90.0])
    with pytest.raises(ValueError, match="TIFF inputs require angle metadata"):
        load_dataset(stack_dir)
    with pytest.raises(ValueError, match="does not match projection count"):
        load_tiff_stack(stack_dir, angles=[0.0])


def test_validate_dataset_reports_schema_failures(tmp_path: Path) -> None:
    valid = tmp_path / "valid.nxs"
    invalid = tmp_path / "invalid.nxs"
    write_projection_dataset(valid)
    with h5py.File(invalid, "w") as handle:
        handle.create_group("entry1")

    assert validate_dataset(valid)["issues"] == []
    assert "no stack of frames found" in validate_dataset(invalid)["issues"][0]
    with pytest.raises(ValueError, match="supply an acquisition file.*not metadata alone"):
        tj.load(invalid)


def test_validate_dataset_reports_scalar_axis_datasets(tmp_path: Path) -> None:
    path = tmp_path / "scalar_axes.nxs"
    write_projection_dataset(path)
    with h5py.File(path, "a") as handle:
        detector = handle["entry/instrument/detector"]
        del detector["image_key"]
        detector.create_dataset("image_key", data=np.asarray(0, dtype=np.int32))
        transformations = handle["entry/sample/transformations"]
        del transformations["rotation_angle"]
        rotation_angle = transformations.create_dataset(
            "rotation_angle",
            data=np.asarray(0.0, dtype=np.float32),
        )
        rotation_angle.attrs["units"] = "degree"

    issues = validate_dataset(path)["issues"]

    assert "image_key must have shape" in issues[0]


def test_validate_dataset_reports_invalid_image_key_values(tmp_path: Path) -> None:
    path = tmp_path / "invalid_image_key.nxs"
    write_projection_dataset(path)
    with h5py.File(path, "a") as handle:
        detector = handle["entry/instrument/detector"]
        del detector["image_key"]
        detector.create_dataset("image_key", data=np.asarray([0, 99], dtype=np.int32))

    issues = validate_dataset(path)["issues"]

    assert "image_key values must be in {0, 1, 2, 3}" in issues[0]


def test_projection_stats_keep_exact_stats_when_percentile_sample_has_no_finite_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "large_projection.nxs"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            "data",
            shape=(2, 1024, 1024),
            dtype=np.float32,
        )
        dataset[0] = np.nan
        dataset[1] = np.full((1024, 1024), 7.0, dtype=np.float32)

    with h5py.File(path, "r") as handle:
        stats, nonfinite = projection_stats(handle["data"])

    assert stats["min"] == 7.0
    assert stats["mean"] == 7.0
    assert stats["max"] == 7.0
    assert stats["p01"] is None
    assert stats["p50"] is None
    assert stats["p99"] is None
    assert nonfinite["nan_count"] == 1024 * 1024


def test_only_a_nonzero_saved_detector_roll_builds_an_explicit_grid() -> None:
    detector = Detector(4, 3, 1.0, 1.0, (0.5, 0.0))
    assert detector_grid_from_geometry_inputs(detector, {}) is None
    assert detector_grid_from_geometry_inputs(detector, {"detector_roll_deg": 0.0}) is None
    assert detector_grid_from_geometry_inputs(detector, {"detector_roll_deg": 0.5}) is not None


def test_a_volume_is_written_in_tiles_without_a_transposed_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tomojax._data._io_nxtomo._TILE_BYTES", 600)  # one chunk per tile
    volume = np.random.default_rng(5).normal(size=(40, 50, 60)).astype(np.float32)
    dataset = make_projection_dataset()
    for index, stored in enumerate((volume, jnp.asarray(volume))):
        path = tmp_path / f"volume_{index}.nxs"
        dataset.volume = stored
        save_dataset(path, dataset)
        np.testing.assert_array_equal(load_nxtomo(str(path)).volume, volume)
        assert h5py.File(path, "r")["/entry/processing/tomojax/volume"].shape == (60, 50, 40)
