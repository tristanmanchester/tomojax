"""Small raw acquisitions, with geometry and calibrations owned by their files."""

import json
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import pytest

import tomojax as tj
from tomojax.cli.main import main
from tomojax.io import validate_dataset
from tomojax.recon.quicklook import scale_to_uint8


def _edf(path: Path, values: np.ndarray) -> None:
    integer = values.dtype.kind == "i"
    data = np.asarray(values, ">i4" if integer else ">f4")
    fields = {
        "EDF_HeaderSize": 512,
        "ByteOrder": "HighByteFirst",
        "DataType": "SignedInteger" if integer else "FloatValue",
        "Dim_1": data.shape[1],
        "Dim_2": data.shape[0],
        "Size": data.nbytes,
    }
    header = ("{\n" + "".join(f"{k} = {v} ;\n" for k, v in fields.items())).encode()
    path.write_bytes(header + b" " * (510 - len(header)) + b"}\n" + data.tobytes())


def _check_inspection(path, frames, previews, capsys):
    assert main(["inspect", str(path), "--json", "--preview", str(previews)]) == 0
    report = json.loads(capsys.readouterr().out)
    counts = np.asarray(frames.counts)
    assert report["valid"] and report["projection"]["shape"] == list(counts.shape)
    assert report["projection"]["stats"]["mean"] == pytest.approx(float(counts.mean()))
    assert report["detector_metadata"]["du"] == frames.source.detector.du
    assert report["flats_darks"]["flat_count"] == len(frames.flats)
    if report["angles"]["coverage_deg"] is not None:
        np.testing.assert_allclose(report["angles"]["coverage_deg"], np.ptp(frames.source.angles))
    np.testing.assert_array_equal(
        iio.imread(previews / "projection.png"), scale_to_uint8(counts[len(counts) // 2])
    )


def test_edf_acquisition_metadata_calibration_and_lazy_windows(tmp_path, capsys):
    shape = (3, 5)
    dark = np.full(shape, 20.25, np.float32)
    for i in range(4):
        _edf(tmp_path / f"sample_{i:04}.edf", dark + (100 + 5 * (i + 0.5)) * np.exp(-0.2))
    _edf(tmp_path / "sample_0004.edf", dark)  # closing alignment image, not a sample view
    _edf(tmp_path / "darkend0000.edf", (4 * dark).astype(np.int32))
    _edf(tmp_path / "dark.edf", dark * 999)  # raw darkend takes precedence
    for p in (0, 4):
        _edf(tmp_path / f"ref0000_{p:04}.edf", dark + 100 + 5 * p)
    fields = dict(
        TOMO_N=4,
        ScanRange=-180,
        REF_N=1,
        REF_ON=4,
        DARK_N=4,
        Dim_1=5,
        Dim_2=3,
        PixelSize=12.5,
        Distance=130,
    )
    metadata = tmp_path / "renamed.txt"
    metadata.write_text("\n".join(f"{k}={v}" for k, v in fields.items()))
    overrides = {"detector": tj.Detector(5, 3, 0.0125, 0.0125), "angles": [0, -45, -90, -135]}
    with pytest.raises(ValueError, match="angles="):
        tj.load_frames(metadata)
    with pytest.raises(ValueError, match="angles="):
        tj.load_frames(metadata, detector=overrides["detector"])
    frames = tj.load_frames(metadata, **overrides)
    assert frames.counts.shape == (4, 3, 5)
    np.testing.assert_array_equal(frames.flat_positions, [0, 4])
    np.testing.assert_array_equal(frames.darks, dark[None])
    np.testing.assert_array_equal(
        frames.counts.frames_at(np.array([3, 1])).window(slice(1, 3), slice(2, 5))[:],
        np.asarray(frames.counts)[[3, 1], 1:3, 2:5],
    )
    _check_inspection(metadata, frames, tmp_path / "previews", capsys)
    scan = frames.corrected()
    np.testing.assert_allclose(scan.projections, 0.2, atol=2e-7)
    np.testing.assert_allclose(scan.detector.du, 0.0125)
    np.testing.assert_array_equal(scan.angles, [0, -45, -90, -135])
    # XML discovery uses content too, without needing the original suffix/name.
    metadata.write_text(
        "<tomodb2><acquisition>"
        + "".join(f"<{k}>{v}</{k}>" for k, v in fields.items())
        + "</acquisition></tomodb2>"
    )
    np.testing.assert_array_equal(tj.load_frames(tmp_path, **overrides).counts[:], frames.counts[:])
    metadata.write_text(metadata.read_text().replace("<Dim_1>5", "<Dim_1>50"))
    assert "cropped or resized" in validate_dataset(tmp_path)["issues"][0]
    with pytest.raises(ValueError, match="cropped or resized"):
        tj.load_frames(tmp_path)


@pytest.mark.parametrize(
    "rotation_path",
    ["measurement/calc_hrsrot", "measurement/hrsrot", "instrument/hrsrot/value"],
)
def test_bliss_raw_sequence_units_keys_and_virtual_pixels(tmp_path, capsys, rotation_path):
    path = tmp_path / "renamed.bin"
    with h5py.File(path, "w") as file:
        file.attrs["default"] = "1.1"
        root = file.create_group("1.1")
        root["title"] = "tomo:fullturn"
        root["technique/scan/tomo_n"] = 4
        root["measurement/scan_numbers"] = [2, 3, 4, 5, 6]
        root["technique/detector/name"] = "camera"
        root["technique/tomoconfig/rotation"] = ["hrsrot"]
        root["technique/tomoconfig/rotation_is_clockwise"] = True
        for key, value, unit in (
            ("optic/sample_pixel_size", [0.01, 0.02], "mm"),
            ("scan/sample_detector_distance", 13, "cm"),
            ("scan/source_sample_distance", 145, "m"),
        ):
            data = root.create_dataset("technique/" + key, data=value)
            data.attrs["unit"] = unit
        for i, key, value in ((2, 2, 10), (3, 1, 110), (4, 0, 60), (5, 1, 110), (6, 0, 60)):
            entry = file.create_group(f"{i}.1")
            entry["technique/image_key"] = key
            entry["measurement/camera"] = np.full((2, 3, 5), value, np.uint16)
            entry[rotation_path] = np.array([0, np.pi / 4]) + (np.pi / 2 if i == 6 else 0)
            entry[rotation_path].attrs["units"] = "radian"
            entry["instrument/positioners/hrsrot"] = 12345
            entry["instrument/positioners/hrsrot"].attrs["units"] = "degree"
        file["broken_link"] = h5py.ExternalLink("absent.h5", "/entry")
    with h5py.File(tmp_path / "pixels.h5", "w") as file:
        file["data"] = np.full((2, 3, 5), 60, np.float32)
    with h5py.File(path, "a") as file:
        del file["4.1/measurement/camera"]
        layout = h5py.VirtualLayout((2, 3, 5), np.float32)
        layout[:] = h5py.VirtualSource("pixels.h5", "data", shape=(2, 3, 5))
        file.create_virtual_dataset("4.1/measurement/camera", layout)
    detector = tj.Detector(5, 3, 0.01, 0.02)
    with pytest.raises(ValueError, match="detector="):
        tj.load_frames(path)
    frames = tj.load_frames(path, detector=detector)
    np.testing.assert_array_equal(frames.flat_positions, [0, 0, 2, 2])
    np.testing.assert_array_equal(
        frames.counts.frames_at(np.array([3, 0, 2])).window(slice(1, 3), slice(2, 4))[:],
        np.full((3, 2, 2), 60),
    )
    _check_inspection(path, frames, tmp_path / "previews", capsys)
    scan = frames.corrected()
    np.testing.assert_allclose(scan.projections, np.log(2), atol=1e-7)
    np.testing.assert_array_equal(scan.angles, [0, -45, -90, -135])
    meta = frames.source.geometry_metadata["acquisition"]
    assert meta["sample_detector_distance"] == 130
    assert meta["source_sample_distance"] == 145000
    with h5py.File(path, "a") as file:
        for view in (4, 6):
            del file[f"{view}.1/{rotation_path}"]
    with pytest.raises(ValueError, match="angles="):
        tj.load_frames(path, detector=detector)
    tj.load_frames(path, detector=detector, angles=[0, -45, -90, -135])
    with h5py.File(tmp_path / "pixels.h5", "w") as file:
        file["data"] = np.empty((0,), np.float32)
    assert "missing virtual detector pixels" in validate_dataset(path)["issues"][0]
    with pytest.raises(ValueError, match="missing virtual detector pixels"):
        tj.load_frames(path)
