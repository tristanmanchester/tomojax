"""``tomojax preprocess``: raw frames to a scan of line integrals, through tj.load_frames."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import h5py
import imageio.v3 as iio
import numpy as np
import pytest

import tomojax as tj
from tomojax.cli.main import main
from tomojax.io import load_dataset

from ._helpers import write_angle_csv, write_raw_nxtomo, write_tiff_stack

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.surface


def _preprocess(*args: object) -> int:
    return main(["preprocess", *(str(a) for a in args)])


def _set_detector(
    raw: Path, *, record: dict[str, object] | None, sizes: tuple[float, float] | None
) -> None:
    with h5py.File(raw, "a") as handle:
        detector = handle["entry/instrument/detector"]
        if "detector_meta_json" in detector.attrs:
            del detector.attrs["detector_meta_json"]
        if record is not None:
            detector.attrs["detector_meta_json"] = json.dumps(record)
        for name, value in zip(
            ("x_pixel_size", "y_pixel_size"), sizes or (None, None), strict=False
        ):
            if name in detector:
                del detector[name]
            if value is not None:
                detector.create_dataset(name, data=np.float32(value))


def test_a_raw_nxtomo_file_becomes_a_scan_that_records_its_corrections(tmp_path: Path) -> None:
    write_raw_nxtomo(tmp_path / "raw.nxs")  # sample 5, flat 11, sample 9, dark 1
    assert _preprocess(tmp_path / "raw.nxs", "-o", tmp_path / "scan.nxs") == 0
    scan = tj.load(tmp_path / "scan.nxs")
    np.testing.assert_allclose(
        np.asarray(scan.projections)[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6
    )
    np.testing.assert_allclose(scan.angles, [0.0, 90.0])
    assert [c.name for c in scan.corrections] == ["flat_dark", "log"]
    assert scan.detector.nu == 2 and scan.detector.nv == 2


def test_tiff_frames_with_flats_darks_and_angles(tmp_path: Path) -> None:
    write_tiff_stack(tmp_path / "frames", [5.0, 9.0])
    write_tiff_stack(tmp_path / "flats", [11.0])
    write_tiff_stack(tmp_path / "darks", [1.0])
    write_angle_csv(tmp_path / "angles.csv", [0.0, 90.0])
    out = tmp_path / "scan.nxs"
    common = ["--flats", tmp_path / "flats", "--angles", tmp_path / "angles.csv"]
    assert _preprocess(tmp_path / "frames", "-o", out, *common, "--darks", tmp_path / "darks") == 0
    np.testing.assert_allclose(
        load_dataset(out).projections[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6
    )
    # Levels stand in for frames: a flat of 11 everywhere, darks of 1.
    assert _preprocess(tmp_path / "frames", "-o", out, "--force", "--flats", 11, "--darks", 1,
                       "--angles", tmp_path / "angles.csv") == 0  # fmt: skip
    np.testing.assert_allclose(
        load_dataset(out).projections[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6
    )


@pytest.mark.parametrize("omitted", ["--flats", "--angles"])
def test_tiff_frames_need_flats_and_angles(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], omitted: str
) -> None:
    write_tiff_stack(tmp_path / "frames", [5.0, 9.0])
    write_tiff_stack(tmp_path / "flats", [11.0])
    write_angle_csv(tmp_path / "angles.csv", [0.0, 90.0])
    args = ["--flats", tmp_path / "flats", "--angles", tmp_path / "angles.csv"]
    del args[args.index(omitted) : args.index(omitted) + 2]
    assert _preprocess(tmp_path / "frames", "-o", tmp_path / "scan.nxs", *args) == 1
    assert ("no flat fields" if omitted == "--flats" else "angles") in capsys.readouterr().err
    assert not (tmp_path / "scan.nxs").exists()


def test_detector_pixels_from_the_record_then_nxtomo_sizes_then_unit(tmp_path: Path) -> None:
    write_raw_nxtomo(tmp_path / "raw.nxs")
    record = {"nu": 2, "nv": 2, "du": 2.0, "dv": 3.0, "det_center": [4.0, -5.0]}
    cases = [(record, (0.5, 0.75), (2.0, 3.0, (4.0, -5.0))), (None, (0.5, 0.75), (0.5, 0.75, (0.0, 0.0))),
             (None, None, (1.0, 1.0, (0.0, 0.0)))]  # fmt: skip
    for i, (meta, sizes, (du, dv, centre)) in enumerate(cases):
        _set_detector(tmp_path / "raw.nxs", record=meta, sizes=sizes)
        assert _preprocess(tmp_path / "raw.nxs", "-o", tmp_path / f"scan{i}.nxs") == 0
        detector = load_dataset(tmp_path / f"scan{i}.nxs").detector
        assert detector is not None
        assert (detector.du, detector.dv, detector.center) == (du, dv, centre)


def test_views_and_a_detector_block(tmp_path: Path) -> None:
    write_raw_nxtomo(tmp_path / "raw.nxs")
    record = {"nu": 2, "nv": 2, "du": 2.0, "dv": 3.0, "det_center": [10.0, 20.0]}
    _set_detector(tmp_path / "raw.nxs", record=record, sizes=(2.0, 3.0))
    out = tmp_path / "part.nxs"
    assert (
        _preprocess(tmp_path / "raw.nxs", "-o", out, "--select-views", "1", "--crop", "0:1,0:1")
        == 0
    )
    scan = tj.load(out)
    np.testing.assert_allclose(np.asarray(scan.projections), [[[-np.log(0.8)]]], rtol=1e-6)
    np.testing.assert_allclose(scan.angles, [90.0])
    assert (scan.detector.nu, scan.detector.nv, scan.detector.center) == (1, 1, (9.0, 18.5))
    assert _preprocess(tmp_path / "raw.nxs", "-o", out, "--force", "--reject-views", "0") == 0
    np.testing.assert_allclose(tj.load(out).angles, [90.0])
    assert _preprocess(tmp_path / "raw.nxs", "-o", out, "--force", "--reject-views", "0:2") == 1


def test_steps_from_the_command_line(tmp_path: Path) -> None:
    views, nv, nu = 40, 3, 64
    u = np.arange(nu)[None, None, :]
    centre = 32 + 12 * np.sin(np.linspace(0, 2 * np.pi, views))[:, None, None]
    truth = 2.0 * np.maximum(0.0, 1.0 - ((u - centre) / 18.0) ** 2) * np.ones((1, nv, 1))
    gain = np.ones(nu)
    gain[[10, 31]] = [0.97, 1.03]  # pixels the flats do not describe
    (tmp_path / "frames").mkdir()
    for i, image in enumerate(1000.0 * np.exp(-truth) * gain):
        iio.imwrite(tmp_path / "frames" / f"p_{i:03d}.tif", image.astype(np.float32))
    write_angle_csv(tmp_path / "angles.csv", list(np.linspace(0.0, 360.0, views, endpoint=False)))
    out = tmp_path / "scan.nxs"
    assert _preprocess(tmp_path / "frames", "-o", out, "--flats", 1000, "--angles", tmp_path / "angles.csv",
                       "--zingers", "--remove-stripes", 9, "--beam-hardening", "1,0.05",
                       "--reject-outliers") == 0  # fmt: skip
    scan = tj.load(out)
    names = [c.name for c in scan.corrections]
    assert names == ["flat_dark", "zingers", "log", "reject_views", "stripes", "beam_hardening"]
    p = truth + 0.05 * truth**2
    assert np.abs(np.asarray(scan.projections) - p).max() < 3e-3


def test_inspect_lists_the_corrections(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_raw_nxtomo(tmp_path / "raw.nxs")
    assert _preprocess(tmp_path / "raw.nxs", "-o", tmp_path / "scan.nxs") == 0
    _ = capsys.readouterr()
    main(["inspect", str(tmp_path / "scan.nxs")])
    assert "Corrections: flat_dark(flats=1, darks=1, flat_sets=1), log(epsilon=1e-06)" in (
        capsys.readouterr().out
    )
