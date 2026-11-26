"""``tomojax correct``: raw frames to a scan of line integrals, through tj.load_frames."""

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

from ._helpers import (
    write_angle_csv,
    write_projection_dataset,
    write_raw_nxtomo,
    write_tiff_stack,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.surface


def _preprocess(*args: object) -> int:
    return main(["correct", *(str(a) for a in args)])


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


@pytest.mark.parametrize("binning", [1, 2])
def test_a_raw_nxtomo_file_becomes_a_scan_that_records_its_corrections(tmp_path, binning):
    write_raw_nxtomo(tmp_path / "raw.nxs")  # sample 5, flat 11, sample 9, dark 1
    out = tmp_path / "scan.nxs"
    assert _preprocess(tmp_path / "raw.nxs", "-o", out, "--binning", binning) == 0
    scan = tj.load(tmp_path / "scan.nxs")
    np.testing.assert_allclose(
        np.asarray(scan.projections)[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6
    )
    np.testing.assert_allclose(scan.angles, [0.0, 90.0])
    binned = ["binned"] if binning > 1 else []
    assert [c.name for c in scan.corrections] == ["flat_dark", "log", *binned]
    assert scan.detector.nu == 2 // binning and scan.detector.nv == 2 // binning


# The detector of a file that records none: unit pixels, centered.
DETECTOR = ("--pixel-size", 1, "--detector-center", 0, 0)


def test_tiff_frames_with_flats_darks_and_angles(tmp_path: Path) -> None:
    write_tiff_stack(tmp_path / "frames", [5.0, 9.0])
    write_tiff_stack(tmp_path / "flats", [11.0])
    write_tiff_stack(tmp_path / "darks", [1.0])
    write_angle_csv(tmp_path / "angles.csv", [0.0, 90.0])
    out = tmp_path / "scan.nxs"
    common = [
        *DETECTOR,
        "--flats",
        tmp_path / "flats",
        "--angles",
        tmp_path / "angles.csv",
    ]
    assert _preprocess(tmp_path / "frames", "-o", out, *common, "--darks", tmp_path / "darks") == 0
    np.testing.assert_allclose(
        load_dataset(out).projections[:, 0, 0], -np.log([0.4, 0.8]), rtol=1e-6
    )
    # Levels stand in for frames: a flat of 11 everywhere, darks of 1.
    assert _preprocess(tmp_path / "frames", "-o", out, "--force", "--flats", 11, "--darks", 1,
                       "--angles", tmp_path / "angles.csv", *DETECTOR) == 0  # fmt: skip
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
    args = [
        *DETECTOR,
        "--flats",
        tmp_path / "flats",
        "--angles",
        tmp_path / "angles.csv",
    ]
    del args[args.index(omitted) : args.index(omitted) + 2]
    assert _preprocess(tmp_path / "frames", "-o", tmp_path / "scan.nxs", *args) == 1
    assert ("No flat fields" if omitted == "--flats" else "angles") in capsys.readouterr().err
    assert not (tmp_path / "scan.nxs").exists()


def test_detector_comes_from_the_record_or_an_explicit_override(tmp_path: Path) -> None:
    # Recorded pitches with no detector give those pitches and the middle; none give pixel
    # units; --pixel-size and --detector-center replace either.
    write_raw_nxtomo(tmp_path / "raw.nxs")
    for i, sizes in enumerate(((0.5, 0.75), None)):
        _set_detector(tmp_path / "raw.nxs", record=None, sizes=sizes)
        out = tmp_path / f"scan{i}.nxs"
        assert _preprocess(tmp_path / "raw.nxs", "-o", out) == 0
        du, dv = sizes or (1.0, 1.0)
        assert tj.load(out).detector == tj.Detector(2, 2, du, dv)
        assert (tj.load(out).source.geometry_metadata.get("length_unit") == "pixel") == (
            sizes is None
        )
        out = tmp_path / f"configured{i}.nxs"
        assert _preprocess(tmp_path / "raw.nxs", "-o", out, *DETECTOR) == 0
        assert tj.load(out).detector == tj.Detector(2, 2, 1, 1)


def test_views_and_a_detector_block(tmp_path: Path) -> None:
    write_raw_nxtomo(tmp_path / "raw.nxs")
    record = {"nu": 2, "nv": 2, "du": 2.0, "dv": 3.0, "det_center": [10.0, 20.0]}
    _set_detector(tmp_path / "raw.nxs", record=record, sizes=(2.0, 3.0))
    out = tmp_path / "part.nxs"
    assert _preprocess(tmp_path / "raw.nxs", "-o", out, "--views", "1", "--crop", "0:1,0:1") == 0
    scan = tj.load(out)
    np.testing.assert_allclose(np.asarray(scan.projections), [[[-np.log(0.8)]]], rtol=1e-6)
    np.testing.assert_allclose(scan.angles, [90.0])
    assert (scan.detector.nu, scan.detector.nv, scan.detector.center) == (1, 1, (9.0, 18.5))
    assert _preprocess(tmp_path / "raw.nxs", "-o", out, "--force", "--views", "0:3") == 1


@pytest.mark.parametrize("rings", [False, True])
def test_steps_from_the_command_line(tmp_path: Path, rings: bool) -> None:
    views, nv, nu = 40, 3, 64
    u = np.arange(nu)[None, None, :]
    center = 32 + 12 * np.sin(np.linspace(0, 2 * np.pi, views))[:, None, None]
    truth = 2.0 * np.maximum(0.0, 1.0 - ((u - center) / 18.0) ** 2) * np.ones((1, nv, 1))
    gain = np.ones(nu)
    gain[[10, 31]] = [0.97, 1.03]  # pixels the flats do not describe
    (tmp_path / "frames").mkdir()
    for i, image in enumerate(1000.0 * np.exp(-truth) * gain):
        iio.imwrite(tmp_path / "frames" / f"p_{i:03d}.tif", image.astype(np.float32))
    write_angle_csv(tmp_path / "angles.csv", list(np.linspace(0.0, 360.0, views, endpoint=False)))
    out = tmp_path / "scan.nxs"
    assert _preprocess(tmp_path / "frames", "-o", out, "--darks", 0, "--flats", 1000, "--angles", tmp_path / "angles.csv",
                       "--zingers", "--stripes", 9, *(["--rings"] if rings else []),
                       "--reject-views", *DETECTOR) == 0  # fmt: skip
    scan = tj.load(out)
    names = [c.name for c in scan.corrections]
    if rings:
        assert scan.corrections[3].settings == {"mirror_axis": (nu - 1) / 2, "views": 512}
        names.remove("rings")
    assert names == ["flat_dark", "zingers", "log", "reject_views", "stripes"]
    assert np.isfinite(np.asarray(scan.projections)).all()


def test_correct_on_a_dataset_takes_only_its_line_integral_steps(tmp_path, capsys):
    # A saved scan holds line integrals: steps on them apply, and a step on transmission,
    # which must run before the log, is refused with what to run instead.
    write_projection_dataset(tmp_path / "scan.nxs")
    out = tmp_path / "corrected.nxs"
    assert _preprocess(tmp_path / "scan.nxs", "-o", out, "--stripes", 3, "--views", "0:2") == 0
    assert [c.name for c in tj.load(out).corrections][-2:] == ["selected", "stripes"]
    assert _preprocess(tmp_path / "scan.nxs", "-o", tmp_path / "z.nxs", "--zingers") == 1
    assert "acts on transmission" in capsys.readouterr().err
    assert not (tmp_path / "z.nxs").exists()


def test_inspect_lists_the_corrections(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_raw_nxtomo(tmp_path / "raw.nxs")
    assert _preprocess(tmp_path / "raw.nxs", "-o", tmp_path / "scan.nxs") == 0
    _ = capsys.readouterr()
    main(["inspect", str(tmp_path / "scan.nxs")])
    assert "applied: flat_dark(flats=1, darks=1, flat_sets=1), log(epsilon=1e-06)" in (
        capsys.readouterr().out
    )
