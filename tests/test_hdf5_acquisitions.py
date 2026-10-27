"""External HDF5 exports are recognised by content without inventing calibration."""

import h5py
import numpy as np
import pytest

import tomojax as tj
from tomojax.io.api import locate_frames, read_acquisition


@pytest.mark.parametrize("recorded", [False, True])
@pytest.mark.parametrize("single_row", [False, True])
def test_data_exchange_preserves_syrmep_units_and_accepts_only_explicit_center(
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
    with pytest.raises(ValueError, match="detector=|flats="):
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
    with pytest.raises(ValueError, match="detector="):
        tj.load_frames(path, **options)
    frames = tj.load_frames(path, detector=tj.Detector(1, 1, 1, 1), **options)
    np.testing.assert_allclose(frames.geometry.angles, [12, 72])
    np.testing.assert_allclose(frames.corrected().projections[:, 0, 0], -np.log([0.5, 0.75]))
    pixels.unlink()
    with pytest.raises(ValueError, match="external data unavailable.*pixels.h5"):
        tj.load_frames(path, detector=tj.Detector(1, 1, 1, 1), **options)
