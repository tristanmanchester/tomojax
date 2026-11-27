"""Independent NeXus tree and registered-coordinate checks for the native container."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from threading import Barrier, Thread

import h5py
import numpy as np
import pytest

import tomojax as tj
from tomojax.core.host_volume import mapped_storage


def volume_and_scan():
    grid = tj.Grid(4, 3, 2, 0.1, 0.2, 0.3, vol_origin=(-1, 2, 3))
    det = tj.Detector(4, 2, 0.2, 0.3)
    geometry = tj.ParallelGeometry(grid, det, np.array([0, 30, 61], np.float32))
    pixels = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 0.25
    frames = np.arange(24, dtype=np.float32).reshape(3, 2, 4) - 4
    scan = tj.Scan(frames, geometry, name="crystal", view_positions=np.array([0.5, 3.5, 5.5]))
    return tj.Reconstruction(
        pixels, geometry, "cgls", {"iterations": 3}, scan, length_unit="mm"
    ), scan


def test_native_default_signal_registration_and_frame_links(tmp_path):
    result, scan = volume_and_scan()
    path = tmp_path / "result.nxs"
    tj.save(path, result, include_scan=True)
    with h5py.File(path) as f:
        assert f.attrs["tomojax_format"] == 2
        assert f.attrs["default"] == "entry" and f["entry"].attrs["default"] == "volume"
        assert "definition" not in f["entry"] and "definition" not in f["entry"].attrs
        group = f["entry/volume"]
        assert group.attrs["NX_class"] == "NXdata" and group.attrs["signal"] == "data"
        assert list(group.attrs["axes"]) == ["z", "y", "x"]
        np.testing.assert_allclose(group["z"], [3, 3.3])
        np.testing.assert_allclose(group["y"], [2, 2.2, 2.4])
        np.testing.assert_allclose(group["x"], [-1, -0.9, -0.8, -0.7])
        assert group["z"].attrs["units"] == "mm"
        assert group["data"].attrs["units"] == "1/mm"
        for d in (group["data"], f["entry/data/data"]):
            assert d.id.get_offset() is not None and d.chunks is None and d.compression is None
        assert f["entry/data/data"].id == f["entry/instrument/detector/data"].id
        assert f["entry/data/rotation_angle"].id == f["entry/sample/rotation_angle"].id
        assert f["entry/data/image_key"].id == f["entry/instrument/detector/image_key"].id
        assert f["entry/data/data"].attrs["long_name"] == "line integrals"
        np.testing.assert_array_equal(f["entry/sample/rotation_angle"], [0, 30, 61])
        assert group["data"][1, 1, 2] == 18.25
        record = json.loads(f["entry/record"].asstr()[()])
        assert record["kind"] == "reconstruction"
        assert record["tables"]["angles"]["path"] == "/entry/sample/rotation_angle"
    loaded = tj.load_reconstruction(path)
    assert mapped_storage(loaded.scan.projections) is not None
    assert mapped_storage(loaded.volume) is not None
    np.testing.assert_array_equal(loaded.volume, result.volume)
    np.testing.assert_array_equal(loaded.scan.view_positions, scan.view_positions)
    np.testing.assert_array_equal(loaded.scan.projections, scan.projections)


@pytest.mark.parametrize("kind", ["scan", "reconstruction", "alignment", "simulation"])
def test_hand_built_format2_generations(kind):
    path = Path(__file__).parent / "data" / "formats" / f"format2-{kind}.nxs"
    loader = {
        "scan": tj.load,
        "reconstruction": tj.load_reconstruction,
        "alignment": tj.load_alignment,
        "simulation": tj.load_simulation,
    }[kind]
    item = loader(path)
    if kind != "reconstruction":
        scan = item if kind == "scan" else item.scan
        assert scan.name == "golden"
        assert scan.projections[2, 1, 3] == 19
        np.testing.assert_array_equal(scan.angles, [0, 30, 61])
        assert scan.detector.du == 0.2 and scan.detector.dv == 0.3
    if kind != "scan":
        array = {"reconstruction": "volume", "simulation": "phantom", "alignment": "image"}[kind]
        assert getattr(item, array)[1, 2, 1] == 21.25
    report = tj.inspect(path)
    assert report["record"]["kind"] == kind
    assert report["record"]["provenance"]["command"] == "fixture"


def test_remapped_file_uses_original_descriptor(tmp_path, monkeypatch):
    old, _ = volume_and_scan()
    a, b = tmp_path / "active.nxs", tmp_path / "replacement.nxs"
    tj.save(a, old, include_scan=True)
    tj.save(b, replace(old, volume=old.volume + 1000))
    meet = Barrier(2, timeout=20)

    def swap():
        meet.wait()
        os.replace(b, a)
        meet.wait()

    both = Thread(target=swap)
    original = np.memmap

    class RacedMap(original):
        raced = False

        def __new__(cls, *args, **kwargs):
            if not cls.raced:
                cls.raced = True
                meet.wait()
                meet.wait()
            return super().__new__(cls, *args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(np, "memmap", RacedMap)
        both.start()
        loaded = tj.load_reconstruction(a)
        both.join(20)
    assert loaded.scan.projections[2, 1, 3] == 19
    assert loaded.volume[1, 2, 1] == 21.25
    assert tj.load_reconstruction(a).volume[1, 2, 1] == 1021.25


def test_history_survives_scan_free_resave_and_source_removal(tmp_path):
    result, _ = volume_and_scan()
    first, second = tmp_path / "one.nxs", tmp_path / "two.nxs"
    tj.save(first, result)
    before = tj.inspect(first)["record"]
    loaded = tj.load_reconstruction(first)
    first.unlink()
    tj.save(second, loaded)
    after = tj.inspect(second)["record"]
    assert after["history"] == [before["provenance"]]
    assert before["geometry"]["meta"]["view_positions"] == [0.5, 3.5, 5.5]
    assert after["geometry"] == before["geometry"]
    np.testing.assert_array_equal(tj.load_reconstruction(second).volume, result.volume)


def test_future_format_reports_writing_version(tmp_path):
    path = tmp_path / "later.nxs"
    with h5py.File(path, "w") as f:
        f.attrs.update(tomojax_format=3, creator_version="7.3")
    for load in (tj.load, tj.load_reconstruction, tj.inspect):
        with pytest.raises(ValueError, match=r"format 3.*7\.3"):
            load(path)


def test_saved_pixels_read_with_stock_hdf5(tmp_path):
    import shutil
    import subprocess

    command = shutil.which("h5dump")
    if command is None:
        pytest.skip("h5dump is unavailable")
    recon, _ = volume_and_scan()
    path = tmp_path / "volume.nxs"
    tj.save(path, recon)
    result = subprocess.run(
        [command, "-d", "/entry/volume/data", "-s", "1,2,1", "-c", "1,1,1", str(path)],
        text=True,
        check=True,
        capture_output=True,
    )
    assert "(1,2,1): 21.25" in result.stdout


def test_scan_free_save_only_indexes_the_source_views_needed_for_provenance(tmp_path):
    recon, scan = volume_and_scan()

    class Lazy:
        shape = scan.projections.shape
        dtype = scan.projections.dtype
        ndim = 3

        def __array__(self, *args, **kwargs):
            raise AssertionError("read the whole source")

        def __getitem__(self, view):
            assert isinstance(view, int)
            return scan.projections[view]

    # Install a source that can supply individual views without a full array conversion.
    lazy = replace(scan)
    object.__setattr__(lazy, "projections", Lazy())
    tj.save(tmp_path / "export.nxs", replace(recon, scan=lazy))
    with h5py.File(tmp_path / "export.nxs") as file:
        assert "data" not in file["entry/instrument/detector"]


def test_tilted_posed_scan_solves_identically_after_save(tmp_path):
    sim = tj.simulate(size=12, n_views=18, geometry="lamino", tilt=58, seed=4)
    poses = np.zeros((18, 6), dtype=np.float32)
    poses[:, 0] = np.linspace(-1.5, 1.5, 18)
    poses[:, 3] = 0.35 * np.sin(np.arange(18))
    scan = sim.scan.with_poses(poses)
    path = tmp_path / "tilted.nxs"
    tj.save(path, scan)
    loaded = tj.load(path)
    np.testing.assert_array_equal(loaded.poses, scan.poses)
    first = tj.reconstruct(scan, "cgls", iterations=2)
    second = tj.reconstruct(loaded, "cgls", iterations=2)
    np.testing.assert_array_equal(second.volume, first.volume)
    for item in (first, second):
        assert item.grid == scan.grid
    left = tj.align(scan, fit="center")
    right = tj.align(loaded, fit="center")
    np.testing.assert_array_equal(left.poses, right.poses)


@pytest.mark.parametrize("kind", ["scan", "alignment", "simulation", "reconstruction", "with_scan"])
def test_opening_each_kind_maps_all_members_without_an_hdf5_pixel_read(tmp_path, monkeypatch, kind):
    result, scan = volume_and_scan()
    item, loader = {
        "scan": (scan, tj.load),
        "alignment": (tj.Alignment(scan, {}, result.volume, result.grid), tj.load_alignment),
        "simulation": (tj.Simulation(scan, result.volume), tj.load_simulation),
        "reconstruction": (result, tj.load_reconstruction),
        "with_scan": (result, tj.load_reconstruction),
    }[kind]
    path = tmp_path / "registered.nxs"
    tj.save(path, item, include_scan=kind == "with_scan")
    opens = []
    open_file = h5py.File.__init__

    def counted_open(file, name, *args, **kwargs):
        if isinstance(name, str | Path) and Path(name) == path:
            opens.append(path)
        return open_file(file, name, *args, **kwargs)

    monkeypatch.setattr(h5py.File, "__init__", counted_open)
    original = h5py.Dataset.__getitem__
    original_array = h5py.Dataset.__array__

    def checked_getitem(field, *args, **kwargs):
        assert field.ndim < 3, f"Read pixels during load from {field.name}"
        return original(field, *args, **kwargs)

    def checked_array(field, *args, **kwargs):
        assert field.ndim < 3, f"Converted pixels during load from {field.name}"
        return original_array(field, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", checked_getitem)
    monkeypatch.setattr(h5py.Dataset, "__array__", checked_array)
    loaded = loader(path)
    assert opens == [path]
    members = []
    if kind != "scan":
        name = "phantom" if kind == "simulation" else "image" if kind == "alignment" else "volume"
        members.append((getattr(loaded, name), result.volume))
    if kind != "reconstruction":
        members.append(((loaded if kind == "scan" else loaded.scan).projections, scan.projections))
    for pixels, expected in members:
        assert mapped_storage(pixels) is not None
        assert not pixels.flags.writeable
        np.testing.assert_array_equal(pixels[1], expected[1])


def test_opaque_future_members_and_keys_do_not_hide_the_known_volume(tmp_path):
    result, _ = volume_and_scan()
    path = tmp_path / "future-fields.nxs"
    tj.save(path, result)
    with h5py.File(path, "r+") as file:
        dataset = file["entry/record"]
        record = json.loads(dataset[()])
        record["laboratory"] = {"campaign": "local"}
        record["arrays"]["auxiliary"] = {"new_kind": True}
        record["tables"]["unfamiliar"] = {"new_table": True}
        dataset[()] = json.dumps(record)
    np.testing.assert_array_equal(tj.load_reconstruction(path).volume, result.volume)


def test_format2_reads_filtered_and_externally_linked_members_at_their_recorded_paths(
    tmp_path, capsys
):
    from tomojax.cli.export import main

    result, _ = volume_and_scan()
    path = tmp_path / "registered.nxs"
    tj.save(path, result)
    with h5py.File(tmp_path / "pixels.h5", "w") as data:
        data.create_dataset("image", data=result.volume, compression="gzip")
    with h5py.File(path, "r+") as file:
        del file["entry/volume/data"]
        file["entry/volume/data"] = h5py.ExternalLink("pixels.h5", "/image")
    pixels = tj.load_reconstruction(path).volume
    assert not pixels.flags.writeable
    np.testing.assert_array_equal(pixels, result.volume)

    output = tmp_path / "volume.raw"
    assert main([str(path), "-o", str(output), "--progress", "json"]) == 0
    receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
    reference = receipt["arrays"]["source_volume"]["storage"]
    assert reference == {"type": "hdf5", "path": str(path), "dataset": "/entry/volume/data"}
    with h5py.File(reference["path"]) as master:
        np.testing.assert_array_equal(master[reference["dataset"]], result.volume)
    np.testing.assert_array_equal(
        np.fromfile(output, "<f4").reshape(result.volume.shape), result.volume
    )


def test_integer_line_integrals_keep_the_scan_measurement_domain(tmp_path):
    _, scan = volume_and_scan()
    path = tmp_path / "integer.nxs"
    tj.save(path, replace(scan, projections=np.asarray(scan.projections, np.int16)))
    with h5py.File(path) as file:
        record = json.loads(file["entry/record"][()])
        assert record["arrays"]["projections"]["domain"] == "line_integrals"
    np.testing.assert_array_equal(tj.load(path).projections, scan.projections)


def test_saved_and_exported_simulation_keep_the_same_motion_truth(tmp_path):
    from tomojax.io import ProjectionDataset

    result, scan = volume_and_scan()
    truth = np.zeros((3, 6))
    truth[:, 3:5] = [[1.5, -0.2], [0.7, 0.9], [-0.6, 2]]
    source = ProjectionDataset(
        None,
        scan.angles,
        geometry_metadata={
            "length_unit": "mm",
            "simulation_info": {"simulation": {"true_poses": truth.tolist()}},
        },
    )
    sim = tj.Simulation(replace(scan, source=source), result.volume)
    saved, exported = tmp_path / "motion.nxs", tmp_path / "phantom.raw"
    tj.save(saved, sim)
    tj.export(exported, sim)
    with h5py.File(saved) as file:
        record = json.loads(file["entry/record"][()])
        native = record["tables"]["true_poses"]
        np.testing.assert_array_equal(file[native["path"]], truth)
    portable = tj.inspect(exported)["record"]["tables"]["true_poses"]
    assert native["frame"] == portable["frame"] == "detector"
    assert native["columns"] == portable["columns"]
    np.testing.assert_array_equal(portable["values"], truth)
    np.testing.assert_array_equal(tj.load_simulation(saved).true_poses, truth)


def test_copying_loaded_metadata_reads_no_detector_pixels(tmp_path, monkeypatch):
    from tomojax.io.api import load_nxtomo

    _, scan = volume_and_scan()
    saved = tmp_path / "scan.nxs"
    tj.save(saved, scan)
    loaded = load_nxtomo(str(saved), lazy=True)
    original = h5py.Dataset.__getitem__

    def checked_read(field, *args):
        assert field.ndim < 3, f"Read detector pixels from {field.name}"
        return original(field, *args)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", checked_read)
    copied = loaded.copy_metadata()
    copied.history[0]["command"] = "edited"
    assert "command" not in loaded.metadata.history[0]
    np.testing.assert_array_equal(copied.angles, [0, 30, 61])


def test_scan_inside_a_result_uses_the_same_frame_selection_and_calibration(tmp_path):
    from tomojax.io.api import NXTomoMetadata, save_nxtomo

    result, scan = volume_and_scan()
    path = tmp_path / "detector-counts-with-volume.nxs"
    counts = np.broadcast_to(np.array([51, 101, 1, 26], dtype="<f4")[:, None, None], (4, 2, 4))
    save_nxtomo(
        str(path),
        counts,
        metadata=NXTomoMetadata(
            volume=result.volume,
            grid=scan.grid,
            detector=scan.detector,
            angles=np.array([0.0, 0.0, 0.0, 90.0]),
            image_key=np.array([0, 1, 2, 0]),
            projection_domain="counts",
        ),
    )
    for loaded in (tj.load(path), tj.load_reconstruction(path).scan):
        np.testing.assert_array_equal(loaded.angles, [0, 90])
        np.testing.assert_allclose(loaded.projections[:, 0, 0], -np.log([0.5, 0.25]))
        assert [step.name for step in loaded.corrections] == ["flat_dark", "log"]


def test_foreign_endian_counts_and_view_tables_have_portable_storage(tmp_path):
    from tomojax.io.api import NXTomoMetadata, save_nxtomo

    counts = np.arange(12, dtype=">u2").reshape(3, 1, 4)
    meta = NXTomoMetadata(
        angles=np.array([15, 37, 61], dtype=">f4"),
        image_key=np.array([0, 0, 0], dtype=">i4"),
        projection_domain="counts",
    )
    path = tmp_path / "detector.nxs"
    save_nxtomo(str(path), counts, metadata=meta)
    with h5py.File(path) as file:
        for name, dtype in (
            ("entry/data/data", "<u2"),
            ("entry/data/rotation_angle", "<f4"),
            ("entry/data/image_key", "<i4"),
        ):
            assert file[name].dtype.str == dtype
        np.testing.assert_array_equal(file["entry/data/data"][2, 0], [8, 9, 10, 11])
        np.testing.assert_array_equal(file["entry/data/rotation_angle"], [15, 37, 61])

    inconsistent = replace(meta, projection_domain="line_integrals", image_key=np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="image_key.*line integrals"):
        save_nxtomo(str(path), counts, metadata=inconsistent)
    with h5py.File(path) as file:
        np.testing.assert_array_equal(file["entry/data/image_key"], [0, 0, 0])


@pytest.mark.parametrize("include_scan", [False, True])
def test_replacing_a_result_volume_releases_its_original_pixel_mapping(tmp_path, include_scan):
    import gc
    import weakref

    result, _ = volume_and_scan()
    path = tmp_path / "edited.nxs"
    tj.save(path, result, include_scan=include_scan)
    loaded = tj.load_reconstruction(path)
    original = weakref.ref(mapped_storage(loaded.volume))
    edited = replace(loaded, volume=np.zeros_like(result.volume))
    del loaded
    gc.collect()
    assert original() is None
    np.testing.assert_array_equal(edited.volume, 0)


def test_explicit_detector_and_angles_can_replace_missing_saved_datasets(tmp_path):
    _, scan = volume_and_scan()
    path = tmp_path / "alternative.nxs"
    tj.save(path, scan)
    with h5py.File(path, "r+") as file:
        del file["entry/instrument/detector/data"]
        del file["entry/sample/rotation_angle"]
        file.create_dataset("alternative", data=np.full((3, 2, 4), 25, np.float32))
        file.create_dataset("radians", data=np.deg2rad([9, 30, 90])).attrs["units"] = "rad"

    loaded = tj.load(path, data_path="/alternative", angles_path="/radians", flats=100, darks=0)
    np.testing.assert_allclose(loaded.angles, [9, 30, 90])
    np.testing.assert_allclose(loaded.projections, -np.log(0.25))


def test_inspecting_a_volume_only_archive_needs_no_pixel_mapping(tmp_path, monkeypatch):
    result, _ = volume_and_scan()
    path = tmp_path / "header.nxs"
    tj.save(path, result)

    def unexpected_mapping(*args, **kwargs):
        raise AssertionError("inspection mapped the volume")

    monkeypatch.setattr(np.memmap, "__new__", unexpected_mapping)
    report = tj.inspect(path)
    assert report["contents"]["volume"]["shape"] == [2, 3, 4]
    assert report["volume"]["shape"] == [2, 3, 4]


def test_alignment_image_without_a_separate_grid_uses_the_scan_voxels(tmp_path):
    result, scan = volume_and_scan()
    alignment = tj.Alignment(scan, {}, result.volume)
    path = tmp_path / "image.nxs"
    tj.save(path, alignment)
    assert tj.load_alignment(path).grid == scan.grid
    receipt = tj.export(tmp_path / "image.raw", alignment)
    with h5py.File(path) as file:
        saved_axes = file["entry/image"]
        for axis, values in zip(
            receipt["axes"], ([3, 3.3], [2, 2.2, 2.4], [-1, -0.9, -0.8, -0.7]), strict=True
        ):
            np.testing.assert_allclose(saved_axes[axis["name"]], values)
            assert axis["coordinates"]["origin"] == pytest.approx(values[0])
            assert axis["coordinates"]["spacing"] == pytest.approx(values[1] - values[0])


@pytest.mark.parametrize("vertical_pitch", [None, 0.5])
def test_nexus_projection_axes_keep_each_recorded_pitch_without_a_detector_center(
    tmp_path, vertical_pitch
):
    from tomojax.io.api import NXTomoMetadata, save_nxtomo

    detector = {"nu": 4, "nv": 2, "du": 0.3}
    if vertical_pitch is not None:
        detector["dv"] = vertical_pitch
    metadata = NXTomoMetadata(
        angles=np.array([0.0, 90.0]), detector=detector, geometry_meta={"length_unit": "mm"}
    )
    saved = tmp_path / "detector.nxs"
    save_nxtomo(str(saved), np.zeros((2, 2, 4), np.float32), metadata=metadata)
    with h5py.File(saved) as file:
        det = file["entry/instrument/detector"]
        assert det["x_pixel_size"][()] == 0.3
        np.testing.assert_allclose(file["entry/data/u"], [-0.45, -0.15, 0.15, 0.45])
        assert file["entry/data/u"].attrs["units"] == "mm"
        v = file["entry/data/v"]
        if vertical_pitch is None:
            np.testing.assert_array_equal(v, [-0.5, 0.5])
            assert v.attrs["units"] == "pixel" and "y_pixel_size" not in det
        else:
            np.testing.assert_array_equal(v, [-0.25, 0.25])
            assert det["y_pixel_size"][()] == 0.5 and v.attrs["units"] == "mm"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_saved_scan_preserves_the_supplied_angular_coordinates(tmp_path, dtype):
    _, scan = volume_and_scan()
    coordinates = np.array([1.234567890123, 162.1234567891, 179.999999], dtype)
    scan = replace(scan, geometry=replace(scan.geometry, angles=coordinates))
    path = tmp_path / "measured-angles.nxs"
    tj.save(path, scan)
    with h5py.File(path) as file:
        angles = file["entry/sample/rotation_angle"]
        assert angles.dtype == coordinates.dtype
        np.testing.assert_array_equal(angles, coordinates)
    loaded = tj.load(path)
    np.testing.assert_array_equal(loaded.angles, coordinates)
    for i in range(len(coordinates)):
        np.testing.assert_array_equal(loaded.geometry.pose_matrix(i), scan.geometry.pose_matrix(i))


def test_resaving_an_incomplete_acquisition_keeps_known_pitch_and_absent_angles(tmp_path):
    from tomojax.io import load_dataset, save_dataset
    from tomojax.io.api import NXTomoMetadata, save_nxtomo

    old, new = tmp_path / "measured.nxs", tmp_path / "resaved.nxs"
    detector = {"nu": 4, "nv": 2, "du": 0.2}
    save_nxtomo(
        str(old),
        np.zeros((3, 2, 4), np.float32),
        metadata=NXTomoMetadata(detector=detector, geometry_meta={"length_unit": "mm"}),
    )
    save_dataset(new, load_dataset(old))
    with h5py.File(new) as file:
        record = json.loads(file["entry/record"][()])
        assert record["geometry"]["detector"] == detector
        assert "angles" not in record["tables"]
        assert "rotation_angle" not in file["entry/sample"]
        np.testing.assert_allclose(file["entry/data/u"], [-0.3, -0.1, 0.1, 0.3])
        assert file["entry/data/u"].attrs["units"] == "mm"
        assert file["entry/data/v"].attrs["units"] == "pixel"
