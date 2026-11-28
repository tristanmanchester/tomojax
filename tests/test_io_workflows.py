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
    build_geometry_from_dataset_metadata,
    load_dataset,
    load_tiff_stack,
    save_dataset,
    validate_dataset,
)
from tomojax.io.api import (
    inspect_dataset,
    load_nxtomo,
    locate_frames,
    projection_stats,
    save_nxtomo,
)

from ._helpers import (
    make_projection_dataset,
    write_projection_dataset,
    write_raw_nxtomo,
    write_rc6_dataset,
)

pytestmark = pytest.mark.surface


@pytest.mark.parametrize("gtype", ["lamino", "laminography", " LAMINOGRAPHY "])
def test_projection_dataset_roundtrips_nxtomo_with_solver_metadata(tmp_path: Path, gtype) -> None:
    path = tmp_path / "scan.nxs"
    detector = Detector(nu=4, nv=2, du=0.5, dv=0.75, center=(1.0, -2.0))
    grid = Grid(nx=4, ny=4, nz=2, vx=1.0, vy=1.0, vz=2.0)
    align_params = np.ones((2, 6), dtype=np.float32)
    dataset = make_projection_dataset(
        detector=detector,
        grid=grid,
        geometry_type=gtype,
        geometry_metadata={"tilt": 55.0, "tilt_about": "x", "detector_roll": 1.25},
    )
    dataset.angle_offset = np.asarray([1.0, 2.0], dtype=np.float32)
    dataset.align_params = align_params
    dataset.align_gauge = {"mode": "mean_zero"}

    save_dataset(path, dataset)
    loaded = load_dataset(path)
    with h5py.File(path) as file:
        axis = file["entry/sample/rotation_angle"].attrs["vector"]
    np.testing.assert_allclose(axis, [0, np.sin(np.deg2rad(55)), np.cos(np.deg2rad(55))])

    assert isinstance(loaded, ProjectionDataset)
    np.testing.assert_allclose(loaded.projections, dataset.projections)
    np.testing.assert_allclose(loaded.angles, dataset.angles)
    assert loaded.detector == detector
    assert loaded.grid == grid
    geometry_inputs = loaded.geometry_inputs()
    assert geometry_inputs["geometry_type"] == "lamino"
    assert geometry_inputs["tilt"] == pytest.approx(55.0)
    assert geometry_inputs["tilt_about"] == "x"
    assert geometry_inputs["detector_roll"] == pytest.approx(1.25)
    np.testing.assert_allclose(geometry_inputs["angle_offset"], [1.0, 2.0])
    np.testing.assert_allclose(geometry_inputs["align_params"], align_params)
    assert geometry_inputs["align_gauge"] == {"mode": "mean_zero"}


def test_projection_payload_save_load_preserves_metadata_copy(tmp_path: Path) -> None:
    path = tmp_path / "payload.nxs"
    dataset = make_projection_dataset(geometry_metadata={"axis_unit_lab": [0.0, 0.0, 1.0]})
    save_nxtomo(path, projections=dataset.projections, metadata=dataset.to_nxtomo_metadata())

    payload = load_dataset(path)

    copied = payload.to_nxtomo_metadata()
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


def test_numpy_archives_are_refused_at_the_native_file_boundary(tmp_path):
    path = tmp_path / "scan.npz"
    with pytest.raises(ValueError, match="path ending in"):
        save_dataset(path, make_projection_dataset())
    assert not path.exists()
    np.savez(path, projections=np.array([{}], dtype=object))
    for read in (load_dataset, tj.load):
        with pytest.raises(ValueError, match="unsupported|unrecognized|Unsupported"):
            read(path)


def test_files_keep_their_angle_and_detector_center_keys(tmp_path):
    path = tmp_path / "scan.nxs"
    detector = Detector(4, 2, 1, 1, center=(0.5, -1))
    write_projection_dataset(path, detector=detector)
    with h5py.File(path) as file:
        np.testing.assert_array_equal(file["entry/sample/rotation_angle"], [0, 90])
        assert json.loads(file["entry/record"][()])["geometry"]["detector"]["det_center"] == [
            0.5,
            -1,
        ]
    assert load_dataset(path).detector == detector


def test_rc6_native_files_name_the_migration_release_but_external_nexus_still_loads(tmp_path):
    legacy = tmp_path / "old.nxs"
    write_rc6_dataset(legacy)
    for read in (load_dataset, load_nxtomo, tj.load, tj.load_frames, tj.inspect):
        with pytest.raises(
            ValueError, match=r'rc6.*re-save.*python -m pip install "tomojax\[cpu\]==1\.0\.0rc7"'
        ):
            read(legacy)
    external = tmp_path / "external.nxs"
    write_raw_nxtomo(external)
    np.testing.assert_allclose(tj.load(external).projections[:, 0, 0], -np.log([0.4, 0.8]))


def test_copy_metadata_preserves_canonical_volume_values(tmp_path: Path) -> None:
    original_path = tmp_path / "original.nxs"
    resaved_path = tmp_path / "resaved.nxs"
    dataset = make_projection_dataset(grid=Grid(4, 3, 2, 1, 1, 1))
    dataset.volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    save_dataset(original_path, dataset)
    loaded = load_nxtomo(str(original_path))

    save_nxtomo(
        resaved_path,
        projections=loaded.projections,
        metadata=loaded.copy_metadata(),
    )

    resaved = load_nxtomo(str(resaved_path))
    np.testing.assert_array_equal(resaved.volume, dataset.volume)


def test_copy_metadata_copies_array_backed_fields(tmp_path: Path) -> None:
    path = tmp_path / "scan.nxs"
    dataset = make_projection_dataset(grid=Grid(4, 3, 2, 1, 1, 1))
    dataset.volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    dataset.align_params = np.ones((2, 6), dtype=np.float32)
    dataset.angle_offset = np.asarray([0.25, -0.25], dtype=np.float32)

    direct_copy = dataset.to_nxtomo_metadata()
    assert not np.shares_memory(direct_copy.angles, dataset.angles)
    assert direct_copy.volume is not None
    assert not np.shares_memory(direct_copy.volume, dataset.volume)
    assert direct_copy.align_params is not None
    assert not np.shares_memory(direct_copy.align_params, dataset.align_params)
    assert direct_copy.angle_offset is not None
    assert not np.shares_memory(direct_copy.angle_offset, dataset.angle_offset)

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
    assert loaded_copy.angle_offset is not None
    assert loaded.metadata.angle_offset is not None
    assert not np.shares_memory(loaded_copy.angle_offset, loaded.metadata.angle_offset)


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
    with pytest.raises(KeyError, match="no stack of frames found.*data_path"):
        tj.load(invalid)


def test_validate_dataset_reports_scalar_axis_datasets(tmp_path: Path) -> None:
    path = tmp_path / "scalar_axes.nxs"
    write_raw_nxtomo(path)
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
    assert detector_grid_from_geometry_inputs(detector, {"detector_roll": 0.0}) is None
    assert detector_grid_from_geometry_inputs(detector, {"detector_roll": 0.5}) is not None


def test_a_volume_is_written_in_tiles_without_a_transposed_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tomojax._data._format2._TILE_BYTES", 600)  # one chunk per tile
    volume = np.random.default_rng(5).normal(size=(40, 50, 60)).astype(np.float32)
    dataset = make_projection_dataset(grid=Grid(60, 50, 40, 1, 1, 1))
    for index, stored in enumerate((volume, jnp.asarray(volume))):
        path = tmp_path / f"volume_{index}.nxs"
        dataset.volume = stored
        save_dataset(path, dataset)
        np.testing.assert_array_equal(load_nxtomo(str(path)).volume, volume)
        assert h5py.File(path, "r")["/entry/volume/data"].shape == volume.shape


def test_geometry_volume_shape_is_zyx_but_grid_override_remains_xyz():
    from tomojax.io import build_geometry_from_dataset_metadata

    meta = {"detector": Detector(8, 5, 0.9, 1.2).to_dict(), "angles": [0.0, 90.0]}
    inferred, _, _ = build_geometry_from_dataset_metadata(meta, volume_shape=(3, 4, 5))
    explicit, _, _ = build_geometry_from_dataset_metadata(meta, grid_override=(5, 4, 3))
    assert inferred == explicit and (inferred.nx, inferred.ny, inferred.nz) == (5, 4, 3)


def test_single_sinogram_load_infers_one_slice_and_saves_fixed_rank(tmp_path):
    data = np.arange(30, dtype=np.float32).reshape(3, 10)
    path = tmp_path / "external.h5"
    with h5py.File(path, "w") as file:
        file["entry/instrument/detector/data"] = data
        file["entry/sample/rotation_angle"] = [0, 60, 120]
        file["entry/sample/rotation_angle"].attrs["units"] = "degree"
    scan = tj.load_frames(path, detector=Detector(10, 1, 0.7, 0.9), flats=100, darks=0).corrected()
    assert scan.grid.shape == (1, 10, 10)
    np.testing.assert_allclose(scan.projections[1:], -np.log(data[1:, None] / 100))
    target = tmp_path / "slice.nxs"
    tj.save(target, scan)
    np.testing.assert_array_equal(tj.load(target).projections, scan.projections)


def _labeled_dataset(values, key):
    from tomojax.io.api import LoadedNXTomo, NXTomoMetadata

    frames = len(key)
    metadata = NXTomoMetadata(
        angles=np.arange(frames, dtype=np.float32) * 45.0,
        image_key=np.asarray(key, np.int32),
        grid=tj.Grid(4, 4, 2, 1.0, 1.0, 1.0),
        detector=tj.Detector(4, 2, 1.0, 1.0),
    )
    projections = np.broadcast_to(np.asarray(values, np.float32)[:, None, None], (frames, 2, 4))
    return ProjectionDataset.from_nxtomo(LoadedNXTomo(np.array(projections), metadata))


def test_saved_image_keys_split_raw_frames_in_every_container(tmp_path: Path) -> None:
    path = tmp_path / "raw.nxs"
    save_dataset(path, _labeled_dataset([5, 11, 9, 1], [0, 1, 0, 2]))
    scan = tj.load(path)
    np.testing.assert_allclose(
        np.asarray(scan.projections)[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6
    )
    np.testing.assert_allclose(scan.geometry.angles, [0.0, 90.0])
    frames = tj.load_frames(path)
    assert frames.views == 2 and frames.flats is not None and frames.darks is not None


def test_saved_image_keys_drop_invalid_processed_frames(tmp_path: Path) -> None:
    path = tmp_path / "processed.nxs"
    save_dataset(path, _labeled_dataset([0.5, 7.0, 0.25], [0, 3, 0]))
    scan = tj.load(path)
    np.testing.assert_allclose(np.asarray(scan.projections)[:, 0, 0], [0.5, 0.25])
    np.testing.assert_allclose(scan.geometry.angles, [0.0, 90.0])


def test_saved_line_integrals_declare_their_domain_and_omit_unknown_pitch_units(
    tmp_path: Path,
) -> None:
    grid, detector = Grid(4, 4, 2, 1.0, 1.0, 1.0), Detector(4, 2, 0.0065, 0.0065)
    scan = tj.Scan(
        np.ones((3, 2, 4), np.float32), tj.ParallelGeometry(grid, detector, [0, 60, 120])
    )
    tj.save(tmp_path / "scan.nxs", scan)
    with h5py.File(tmp_path / "scan.nxs") as f:
        assert "definition" not in f["entry"]
        assert f["entry/data/data"].attrs["long_name"] == "line integrals"
        # A pitch of unrecorded unit carries no unit, not "pixel".
        assert "units" not in f["entry/instrument/detector/x_pixel_size"].attrs
    assert not validate_dataset(tmp_path / "scan.nxs")["issues"]


def test_tiled_writes_read_the_source_one_bounded_tile_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # check-public-imports: allow-private
    from tomojax._data import _format2

    monkeypatch.setattr(_format2, "_TILE_BYTES", 4000)
    volume = np.random.default_rng(6).normal(size=(30, 20, 40)).astype(np.float32)

    class Source:
        shape, dtype = volume.shape, volume.dtype

        def __init__(self):
            self.reads: list[int] = []

        def __getitem__(self, region):
            block = volume[region]
            self.reads.append(block.size)
            return block

        def __array__(self, *args, **kwargs):
            raise AssertionError("the whole volume must not be converted at once")

    source = Source()
    with h5py.File(tmp_path / "tiles.h5", "w") as f:
        target = f.create_dataset("v", shape=volume.shape, dtype=np.float32, chunks=(5, 5, 10))
        _format2.copy_pixels(target, source)  # pyright: ignore[reportArgumentType]
        np.testing.assert_array_equal(target[()], volume)
    assert len(source.reads) > 1 and max(source.reads) * 4 <= 4000
    assert sum(source.reads) == volume.size


def test_inspection_reports_sample_angle_range(tmp_path):
    path = tmp_path / "raw.nxs"
    write_raw_nxtomo(path)
    with h5py.File(path, "r+") as file:
        file["entry/sample/transformations/rotation_angle"][...] = [5, np.nan, 90, 180]
    angles = tj.inspect(path)["angles"]
    assert (angles["min"], angles["max"], angles["coverage"]) == (5, 90, 85)


def test_a_broken_frame_label_link_is_an_error_not_missing_labels(tmp_path: Path) -> None:
    # Reading a dangling link as absent would label every flat and dark a sample view.
    path = tmp_path / "scan.nxs"
    save_dataset(path, make_projection_dataset())
    with h5py.File(path, "a") as file:
        detector = file["/entry/instrument/detector"]
        del detector["image_key"]
        detector["image_key"] = h5py.ExternalLink("absent.h5", "/image_key")
    with pytest.raises(ValueError, match="image_key: external data unavailable.*absent.h5"):
        load_dataset(path)


def test_a_grid_override_needs_only_the_saved_pitches() -> None:
    grid, _, _ = build_geometry_from_dataset_metadata(
        {
            "detector": Detector(4, 3, 1.0, 1.0).to_dict(),
            "angles": np.arange(5, dtype=np.float32),
            "grid": {"vx": 0.5, "vy": 0.25, "vz": 2.0},
        },
        grid_override=(7, 6, 5),
    )
    assert (grid.nx, grid.ny, grid.nz, grid.vx, grid.vy, grid.vz) == (7, 6, 5, 0.5, 0.25, 2.0)


@pytest.mark.parametrize("link", ["soft", "external"])
def test_raw_frames_refuse_a_dangling_frame_label_link(tmp_path: Path, link: str) -> None:
    # Reading the dangling link as absent would make the flat and dark sample views.
    path = tmp_path / "raw.nxs"
    write_raw_nxtomo(path)
    with h5py.File(path, "a") as file:
        detector = file["/entry/instrument/detector"]
        del detector["image_key"]
        detector["image_key"] = (
            h5py.SoftLink("/entry/absent")
            if link == "soft"
            else h5py.ExternalLink("absent.h5", "/image_key")
        )
    with pytest.raises(ValueError, match="image_key"):
        tj.load_frames(path)


def test_dataset_readers_keep_their_inputs_protected(tmp_path: Path) -> None:
    from tomojax.io.api import save_projection_quicklook

    source = tmp_path / "scan.nxs"
    save_dataset(source, make_projection_dataset())
    originals = source.read_bytes()
    png = tmp_path / "preview.png"
    png.hardlink_to(source)
    with pytest.raises(ValueError, match="different destination"):
        save_projection_quicklook(source, png)
    loaded = load_dataset(source)
    alias = tmp_path / "other.nxs"
    alias.hardlink_to(source)
    for path in (source, alias):
        with pytest.raises(ValueError, match="different destination"):
            save_dataset(path, loaded)
    assert source.read_bytes() == originals
