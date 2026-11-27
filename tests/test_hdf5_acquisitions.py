"""External HDF5 exports are recognized by content without inventing calibration."""

import h5py
import numpy as np
import pytest

import tomojax as tj
from tomojax.io.api import locate_frames, read_acquisition


@pytest.mark.parametrize("recorded", [False, True])
@pytest.mark.parametrize("single_row", [False, True])
def test_data_exchange_preserves_syrmep_units_and_leaves_an_unrecorded_center_at_the_middle(
    tmp_path, recorded, single_row
):
    path = tmp_path / "renamed.bin"
    shape = (4,) if single_row else (3, 4)
    with h5py.File(path, "w") as file:
        file["exchange/data"] = np.full((2, *shape), 52, np.uint16)
        if recorded:
            file["exchange/data_white"] = np.full((2, *shape), 100, np.uint16)
            file["exchange/data_dark"] = np.full((1, *shape), 4, np.uint16)
        file["exchange/theta"] = [0.0, 90.0]
        file["measurement/instrument/source/beamline"] = "Syrmep"
        file["measurement/sample/name"] = "sample"
        for axis, pitch in (("x", "2.04"), ("y", "3.06")):
            data = file.create_dataset(
                "measurement/instrument/detector/actual_pixel_size_" + axis, data=pitch
            )
            data.attrs["unit"] = "microns"
    raw = read_acquisition(path)
    assert raw.record.source_format == "data_exchange" and raw.record.detector is None
    meta = raw.record.geometry_metadata["acquisition"]
    np.testing.assert_allclose(meta["detector_pixel_size"], [0.00204, 0.00306])
    assert meta["beamline"] == "Syrmep" and meta["length_unit"] == "mm"
    if recorded:  # pitches and no center: the middle, and the record says none was recorded
        loaded = tj.load_frames(path)
        assert loaded.geometry.detector.center == (0.0, 0.0)
        assert loaded.source.geometry_metadata["detector_center_status"] == "not_recorded"
    else:
        with pytest.raises(ValueError, match="flats="):
            tj.load_frames(path)
    detector = tj.Detector(4, 1 if single_row else 3, 0.00204, 0.00306, (0.001, -0.002))
    scan = tj.load_frames(
        path, detector=detector, **({} if recorded else {"flats": 100, "darks": 4})
    ).corrected()
    assert scan.detector == detector
    assert scan.projections.shape == (2, detector.nv, 4) and scan.grid.nz == detector.nv
    np.testing.assert_allclose(scan.projections, np.log(2), atol=1e-7)


def test_external_nxtomo_entry_and_explicit_angle_path_keep_calibration_order(tmp_path):
    path = tmp_path / "renamed.data"
    with h5py.File(path, "w") as file:
        root = file.create_group("entry0000")
        root["definition"] = "NXtomo"
        root["instrument/detector/data"] = np.asarray([1, 101, 51, 76, 101], np.float32)[
            :, None, None
        ]
        root["instrument/detector/image_key"] = [2, 1, 0, 0, 1]
        root["sample/rotation_angle"] = [0, 0, 0, 45, 0]
        root["sample/measured_angle"] = [0, 0, 12, 72, 0]
        root["sample/measured_angle"].attrs["units"] = "deg"
        for axis in ("x", "y"):
            root.create_dataset(f"instrument/detector/{axis}_pixel_size", data=6.5e-6).attrs[
                "units"
            ] = "m"
    detector = tj.Detector(1, 1, 0.0065, 0.0065, (0.003, 0))
    frames = tj.load_frames(path, detector=detector, angles_path="/entry0000/sample/measured_angle")
    np.testing.assert_array_equal(frames.geometry.angles, [12, 72])
    np.testing.assert_array_equal(frames.flat_positions, [0, 2])
    np.testing.assert_allclose(frames.corrected().projections[:, 0, 0], -np.log([0.5, 0.75]))
    with h5py.File(path, "a") as file:
        file.move("entry0000/instrument/detector/image_key", "entry0000/labels")
    with pytest.raises(ValueError, match="image_key_path="):
        tj.load_frames(path, detector=detector)
    assert tj.load_frames(path, detector=detector, image_key_path="/entry0000/labels").views == 2


def test_external_nxtomo_checks_nested_virtual_pixels(tmp_path):
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as f:
        f["pixels"] = np.ones((3, 2, 4), np.uint16)
    nested = tmp_path / "nested.h5"
    with h5py.File(nested, "w") as f:
        layout = h5py.VirtualLayout((3, 2, 4), np.uint16)
        layout[:] = h5py.VirtualSource(str(source), "pixels", shape=(3, 2, 4))
        f.create_virtual_dataset("pixels", layout)
    path = tmp_path / "scan.data"
    with h5py.File(path, "w") as f:
        f["entry/definition"] = "NXtomo"
        f["entry/instrument/detector/image_key"] = [0, 1, 2]
        f["entry/sample/rotation_angle"] = [0, 0, 0]
        layout = h5py.VirtualLayout((3, 2, 4), np.uint16)
        layout[:] = h5py.VirtualSource(str(nested), "pixels", shape=(3, 2, 4))
        f.create_virtual_dataset("entry/instrument/detector/data", layout)
    detector = tj.Detector(4, 2, 1.0, 1.0)
    np.testing.assert_array_equal(tj.load_frames(path, detector=detector).counts[:], 1)
    source.unlink()
    with pytest.raises(ValueError, match="missing virtual detector pixels"):
        tj.load_frames(path, detector=detector)


@pytest.mark.parametrize(
    "nested",
    [
        False,
        True,
        "parent_definition",
        "external_entry",
        "external_nested",
        "alias_path",
        "relative_path",
    ],
)
def test_linked_nxtomo_float_keys_preserve_pixels_and_calibration(tmp_path, nested):
    path, pixels = tmp_path / "scan.nxs", tmp_path / "pixels.h5"
    values = np.asarray([1, 101, 51, 76, 101], np.uint16)[:, None, None]
    with h5py.File(pixels, "w") as file:
        file["pixels"] = values
    entry = "/entry/tomo" if nested in (True, "parent_definition", "external_nested") else "/entry"
    with h5py.File(path, "w") as file:
        file[
            ("/entry" if nested in ("parent_definition", "external_nested") else entry)
            + "/definition"
        ] = "NXtomo"
        for suffix in ("/instrument/detector/data", "/data/data"):
            file[entry + suffix] = h5py.ExternalLink(pixels.name, "/pixels")
        file[entry + "/instrument/detector/image_key"] = [2.0, 1.0, 0.0, 0.0, 1.0]
        file[entry + "/sample/rotation_angle"] = [0.0, 0.0, 12.0, 72.0, 0.0]
        for axis in ("x", "y"):
            file.create_dataset(
                entry + f"/instrument/detector/{axis}_pixel_size", data=np.nan
            ).attrs["units"] = "m"
    options = {}
    with h5py.File(path, "a") as file:
        if nested in ("alias_path", "relative_path"):
            if nested == "relative_path":
                file.copy("entry", "other")
            file["latest"] = file["entry"]
            options["data_path"] = (
                "/" if nested == "alias_path" else ""
            ) + "latest/instrument/detector/data"
        file[entry + "/back"] = h5py.SoftLink("/entry")
    if nested in ("external_entry", "external_nested"):
        master = tmp_path / "master.nxs"
        with h5py.File(master, "w") as file:
            file["entry"] = h5py.ExternalLink(path.name, "/entry")
        path = master
    located = locate_frames(
        str(path),
        **(
            {
                "data_path": entry + "/instrument/detector/data",
                "image_key_path": entry + "/instrument/detector/image_key",
                "angles_path": entry + "/sample/rotation_angle",
            }
            if nested == "external_nested"
            else {}
        ),
    )
    np.testing.assert_array_equal(located.stack[:], values)
    np.testing.assert_array_equal(located.image_key, [2, 1, 0, 0, 1])
    assert located.angles is not None
    loaded = tj.load_frames(path, **options)  # no pixel size recorded: pixel units
    assert loaded.source.geometry_metadata["length_unit"] == "pixel"
    frames = tj.load_frames(path, detector=tj.Detector(1, 1, 1, 1), **options)
    np.testing.assert_allclose(frames.geometry.angles, [12, 72])
    np.testing.assert_allclose(frames.corrected().projections[:, 0, 0], -np.log([0.5, 0.75]))
    pixels.unlink()
    with pytest.raises(ValueError, match="external data unavailable.*pixels.h5"):
        tj.load_frames(path, detector=tj.Detector(1, 1, 1, 1), **options)


def _nxtomo(path, *, control=None, declared_by_attribute=False):
    """A one-pixel NXtomo: dark, flat, three sample-keyed frames, flat; angles without units."""
    with h5py.File(path, "w") as file:
        root = file.create_group("entry0000")
        if declared_by_attribute:
            root.attrs["definition"] = "NXtomo"
        else:
            root["definition"] = "NXtomo"
        root["instrument/detector/data"] = np.asarray([1, 101, 51, 76, 61, 101], np.float32)[
            :, None, None
        ]
        root["instrument/detector/image_key"] = [2, 1, 0, 0, 0, 1]
        if control is not None:
            root["instrument/detector/image_key_control"] = control
        root["sample/rotation_angle"] = [0.0, 0.0, 0.0, 90.0, 0.0, 0.0]


def test_esrf_return_images_are_not_sample_views(tmp_path):
    # ESRF keys a return image 0 in image_key and -1 in image_key_control: an alignment check
    # at an angle already measured, not a view.
    path = tmp_path / "scan.nx"
    _nxtomo(path, control=[2, 1, 0, 0, -1, 1])
    frames = tj.load_frames(path, detector=tj.Detector(1, 1, 1.0, 1.0))
    np.testing.assert_array_equal(frames.geometry.angles, [0.0, 90.0])
    assert frames.source.geometry_metadata["return_images_excluded"] == 1


def test_an_entry_declared_nxtomo_by_attribute_keeps_its_unitless_angles(tmp_path):
    path = tmp_path / "scan.nx"
    _nxtomo(path, declared_by_attribute=True)
    frames = tj.load_frames(path, detector=tj.Detector(1, 1, 1.0, 1.0))
    np.testing.assert_array_equal(frames.geometry.angles, [0.0, 90.0, 0.0])


def test_a_virtual_source_recorded_at_a_missing_absolute_path_is_read_beside_the_file(tmp_path):
    # Beamline files often record the acquisition machine's absolute path; HDF5 then reads the
    # source beside the virtual file, and the pixel check must look there too.
    with h5py.File(tmp_path / "source.h5", "w") as f:
        f["pixels"] = np.full((3, 2, 4), 7, np.uint16)
    path = tmp_path / "scan.data"
    with h5py.File(path, "w") as f:
        f["entry/definition"] = "NXtomo"
        f["entry/instrument/detector/image_key"] = [0, 1, 2]
        f["entry/sample/rotation_angle"] = [0, 0, 0]
        layout = h5py.VirtualLayout((3, 2, 4), np.uint16)
        layout[:] = h5py.VirtualSource("/beamline/gone/source.h5", "pixels", shape=(3, 2, 4))
        f.create_virtual_dataset("entry/instrument/detector/data", layout)
    np.testing.assert_array_equal(
        tj.load_frames(path, detector=tj.Detector(4, 2, 1.0, 1.0)).counts[:], 7
    )


# A GDA (Diamond) NeXus scan: the frames in their own file, linked from the scan file; an
# NXdata group plotting them with per-frame coordinates (the demanded rotation, labeled mm,
# the encoder's unlabeled readback, a beam monitor); the NXdetector holding the frames and the
# planned image_key; the rotation stage's fields recorded once, with units; a second detector
# with a stack and labels of its own; and, optionally, an NXtomo subentry without angles.
_PLANNED_KEY = [1, 1, 2, 0, 0, 0, 0, 1, 2]
_LEVELS = [1100, 1100, 100, 600, 350, 850, 225, 1100, 100]
_DEMAND = [0.0, 0.0, 0.0, 0.0, 45.0, 90.0, 135.0, 135.0, 135.0]


def _gda_scan(tmp_path, *, written=9, encoder=None, nxtomo=True, demand_units="mm", axes=True):
    frames = tmp_path / "scan-imaging.h5"
    with h5py.File(frames, "w") as file:
        file["entry/detector/detector"] = np.stack(
            [np.full((2, 3), level, np.uint16) for level in _LEVELS[:written]]
        )
        readback = (np.asarray(_DEMAND) + 0.01)[: written if encoder is None else encoder]
        file["entry/encoder/value"] = readback
    path = tmp_path / "scan.nxs"
    with h5py.File(path, "w") as file:
        entry = file.create_group("entry")
        entry.attrs.update(NX_class="NXentry", default="imaging")
        plot = entry.create_group("imaging")
        plot["data"] = h5py.ExternalLink(frames.name, "/entry/detector/detector")
        plot.create_dataset("stage_cs_theta", data=_DEMAND).attrs["units"] = demand_units
        plot["stage_theta"] = h5py.ExternalLink(frames.name, "/entry/encoder/value")
        plot.create_dataset("ring_current", data=np.full(written, 300.0)).attrs["units"] = "mA"
        plot.attrs.update(
            NX_class="NXdata",
            signal="data",
            axes=np.array([b"stage_cs_theta" if axes else b".", b".", b"."]),
            stage_cs_theta_indices=[0],
            stage_theta_indices=[0],
            ring_current_indices=[0],
        )
        monitor = entry.create_group("monitor")
        monitor["data"] = np.zeros((written, 1, 4))
        monitor.attrs.update(NX_class="NXdata", signal="data")
        detector = entry.create_group("instrument/imaging")
        detector.attrs["NX_class"] = "NXdetector"
        detector["data"] = h5py.ExternalLink(frames.name, "/entry/detector/detector")
        detector["image_key"] = _PLANNED_KEY
        other = entry.create_group("instrument/monitor")
        other.attrs["NX_class"] = "NXdetector"
        other["data"] = monitor["data"]
        other["image_key"] = np.zeros(written, np.int32)
        stage = entry.create_group("instrument/stage")
        stage.attrs["NX_class"] = "NXpositioner"
        stage.create_dataset("stage_theta", data=0.01).attrs["units"] = "deg"
        if nxtomo:
            tomo = entry.create_group("tomo_entry")
            tomo.attrs["NX_class"] = "NXsubentry"
            tomo["definition"] = "NXtomo"
            tomo.create_group("sample").attrs["NX_class"] = "NXsample"
            tomo["instrument/detector/data"] = h5py.ExternalLink(
                frames.name, "/entry/detector/detector"
            )
            tomo["instrument/detector/image_key"] = detector["image_key"]
            tomo["instrument/detector/x_rotation_axis_pixel_position"] = "NaN"
    return path


@pytest.mark.parametrize("nxtomo", [True, False])
def test_a_gda_scan_loads_from_its_nexus_structure(tmp_path, nxtomo):
    # The frames come from the default plot (not the monitor's stack), their labels from their
    # detector (not the monitor's), and the angles from the encoder: the plot's one coordinate
    # with angular units, taken from the stage's recorded field of the same name.
    frames = tj.load_frames(_gda_scan(tmp_path, nxtomo=nxtomo))
    assert frames.views == 4 and len(frames.flats) == 3 and len(frames.darks) == 2
    np.testing.assert_allclose(frames.geometry.angles, [0.01, 45.01, 90.01, 135.01], atol=1e-5)
    record = frames.source.geometry_metadata
    acquisition = record["acquisition"]
    assert acquisition["angles_path"] == "/entry/imaging/stage_theta"
    assert acquisition["angle_units_path"] == "/entry/instrument/stage/stage_theta"
    assert acquisition["image_key_path"] in (
        "/entry/instrument/imaging/image_key",
        "/entry/tomo_entry/instrument/detector/image_key",
    )
    assert "frame_records" not in acquisition
    assert record["length_unit"] == "pixel"  # no pixel size recorded
    np.testing.assert_allclose(
        frames.corrected().projections[:, 0, 0], -np.log([0.5, 0.25, 0.75, 0.125]), rtol=1e-6
    )


@pytest.mark.parametrize("nxtomo", [True, False])
@pytest.mark.parametrize(("written", "encoder", "views"), [(6, None, 3), (6, 5, 2), (9, 8, 4)])
def test_a_scan_stopped_early_reads_the_frames_every_record_covers(
    tmp_path, caplog, nxtomo, written, encoder, views
):
    # The planned key and demand outrun the frames written; the encoder may stop a frame short.
    frames = tj.load_frames(_gda_scan(tmp_path, written=written, encoder=encoder, nxtomo=nxtomo))
    read = written if encoder is None else encoder
    assert frames.views == views and len(frames.flats) == 2 + (read > 7)
    np.testing.assert_allclose(frames.geometry.angles, np.asarray(_DEMAND[3 : 3 + views]) + 0.01)
    records = frames.source.geometry_metadata["acquisition"]["frame_records"]
    assert records == {"data": written, "image_key": 9, "angles": read}
    assert f"reading the first {read} frames" in caplog.text
    corrected = frames.corrected().projections[:, 0, 0]
    np.testing.assert_allclose(corrected, -np.log([0.5, 0.25, 0.75, 0.125][:views]), rtol=1e-6)


@pytest.mark.parametrize("axes", [True, False])
def test_several_rotation_coordinates_resolve_to_the_scan_axis_or_ask(tmp_path, axes):
    path = _gda_scan(tmp_path, nxtomo=False, demand_units="deg", axes=axes)
    if not axes:
        with pytest.raises(KeyError, match="several rotation coordinates"):
            tj.load_frames(path)
        assert tj.load_frames(path, angles_path="/entry/imaging/stage_cs_theta").views == 4
        return
    frames = tj.load_frames(path)
    np.testing.assert_array_equal(frames.geometry.angles, [0, 45, 90, 135])


@pytest.mark.parametrize(("written", "keys", "angles"), [(4, 6, 5), (6, 4, 6), (6, 6, 4)])
def test_fixed_nxtomo_reads_the_common_frame_prefix(tmp_path, written, keys, angles):
    path = tmp_path / "scan.nxs"
    with h5py.File(path, "w") as file:
        det = file.create_group("entry/instrument/detector")
        det["data"] = np.asarray([1, 101, 51, 26, 201, 11][:written])[:, None, None]
        det["image_key"] = [2, 1, 0, 0, 1, 2][:keys]
        file["entry/sample/rotation_angle"] = [0, 0, 10, 40, 0, 0][:angles]
    scan = tj.load(path)
    np.testing.assert_array_equal(scan.angles, [10, 40])
    np.testing.assert_allclose(scan.projections[:, 0, 0], -np.log([0.5, 0.25]))
    records = {"data": written, "image_key": keys, "angles": angles}
    assert scan.source.geometry_metadata["acquisition"]["frame_records"] == records
    assert tj.inspect(path)["loading"]["frame_records"] == records
