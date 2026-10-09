"""Detector frames to line integrals: tj.load_frames, Frames.corrected and the steps."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import h5py
import imageio.v3 as iio
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import BeamHardening, Correction

from ._helpers import tiny_detector, tiny_grid, write_projection_dataset, write_raw_nxtomo

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.surface


def _write_frames(
    path: Path, frames: np.ndarray, image_key: list[int], angles: list[float]
) -> None:
    rows, cols = frames.shape[1:]
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["definition"] = "NXtomo"
        entry.attrs["grid_meta_json"] = json.dumps(tiny_grid(nx=cols, ny=cols, nz=rows).to_dict())
        detector = entry.create_group("instrument/detector")
        detector.create_dataset("data", data=frames)
        detector.create_dataset("image_key", data=np.asarray(image_key, np.int32))
        detector.attrs["detector_meta_json"] = json.dumps(tiny_detector(nu=cols, nv=rows).to_dict())
        sample = entry.create_group("sample")
        sample.create_dataset("name", data="fixture")
        rotation = sample.create_group("transformations").create_dataset(
            "rotation_angle", data=np.asarray(angles, np.float32)
        )
        rotation.attrs["units"] = "degree"


def test_load_corrects_a_raw_nxtomo_file_and_says_so(tmp_path: Path) -> None:
    write_raw_nxtomo(tmp_path / "raw.nxs")  # sample 5, flat 11, sample 9, dark 1
    scan = tj.load(tmp_path / "raw.nxs")
    assert scan.projections.shape == (2, 2, 2)
    np.testing.assert_allclose(scan.angles, [0.0, 90.0])
    np.testing.assert_allclose(
        np.asarray(scan.projections)[:, 0, 0], -np.log([4 / 10, 8 / 10]), rtol=1e-6
    )
    assert [c.name for c in scan.corrections] == ["flat_dark", "log"]
    assert scan.corrections[0].settings == {"flats": 1, "darks": 1, "flat_sets": 1}
    assert "corrections: flat_dark(flats=1, darks=1, flat_sets=1), log" in repr(scan)


def test_load_frames_reads_the_frames_lazily(tmp_path: Path) -> None:
    write_raw_nxtomo(tmp_path / "raw.nxs")
    frames = tj.load_frames(tmp_path / "raw.nxs")
    assert frames.views == 2 and frames.counts.shape == (2, 2, 2)
    assert not isinstance(frames.counts, np.ndarray)  # read only as corrected
    np.testing.assert_array_equal(np.asarray(frames.counts[0:2])[:, 0, 0], [5.0, 9.0])
    np.testing.assert_array_equal(frames.flat_positions, [1])
    assert repr(frames) == (
        "Frames('fixture': 2 views of 2 x 2 float32, 1 flat in 1 set, 1 dark, ParallelGeometry)"
    )


def test_flats_before_and_after_are_interpolated_by_position(tmp_path: Path) -> None:
    # flat 10, four views of 8, flat 20, a dark of 0: view i sits at i + 1/2 of 4.
    frames = np.stack([np.full((2, 3), v, np.uint16) for v in (10, 8, 8, 8, 8, 20, 0)])
    _write_frames(tmp_path / "scan.nxs", frames, [1, 0, 0, 0, 0, 1, 2], [0, 0, 45, 90, 135, 0, 0])
    scan = tj.load(tmp_path / "scan.nxs")
    flats = 10 + 10 * (np.arange(4) + 0.5) / 4
    np.testing.assert_allclose(np.asarray(scan.projections)[:, 0, 0], -np.log(8 / flats), rtol=1e-5)
    settings = scan.corrections[0].settings
    assert settings["flat_sets"] == 2 and settings["flat_positions"] == [0.0, 4.0]


def test_no_darks_warns_and_uses_zero(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    frames = np.stack([np.full((2, 2), v, np.float32) for v in (8, 4)])
    _write_frames(tmp_path / "scan.nxs", frames, [1, 0], [0, 0])
    with caplog.at_level(logging.WARNING):
        scan = tj.load(tmp_path / "scan.nxs")
    assert "no dark fields" in caplog.text
    np.testing.assert_allclose(np.asarray(scan.projections), np.log(2.0), rtol=1e-6)
    assert scan.corrections[0].settings["darks"] == 0


def test_counts_without_flats_are_refused(tmp_path: Path) -> None:
    frames = np.full((2, 2, 2), 100, np.uint16)
    _write_frames(tmp_path / "scan.nxs", frames, [0, 0], [0, 90])
    with pytest.raises(ValueError, match="integer detector counts and no flat frames"):
        tj.load(tmp_path / "scan.nxs")
    with pytest.raises(ValueError, match="no flat fields: pass flats"):
        tj.load_frames(tmp_path / "scan.nxs").corrected()
    scan = tj.load_frames(tmp_path / "scan.nxs", white_level=200).corrected()
    np.testing.assert_allclose(np.asarray(scan.projections), np.log(2.0), rtol=1e-6)
    assert scan.corrections[0].settings == {"white_level": 200.0, "darks": 0}


def test_tiff_stacks_go_through_load_frames(tmp_path: Path) -> None:
    views = tmp_path / "views"
    views.mkdir()
    for i, value in enumerate((50, 25)):
        iio.imwrite(views / f"view_{i:04d}.tif", np.full((2, 3), value, np.uint16))
    flat = tmp_path / "flat.tif"
    iio.imwrite(flat, np.full((2, 3), 100, np.uint16))
    with pytest.raises(ValueError, match="load_frames"):
        tj.load(views)
    with pytest.raises(ValueError, match="angles"):
        tj.load_frames(views / "view_0000.tif")
    frames = tj.load_frames(views, angles=[0, 90], flats=flat)
    scan = frames.corrected()
    np.testing.assert_allclose(np.asarray(scan.projections)[:, 0, 0], -np.log([0.5, 0.25]))
    assert scan.detector.nu == 3 and scan.detector.nv == 2


def test_corrections_are_saved_with_the_scan(tmp_path: Path) -> None:
    write_raw_nxtomo(tmp_path / "raw.nxs")
    scan = tj.load(tmp_path / "raw.nxs").corrected(BeamHardening((1.0, 0.05)))
    tj.save(tmp_path / "scan.nxs", scan)
    again = tj.load(tmp_path / "scan.nxs")
    assert again.corrections == scan.corrections
    assert again.corrections[-1] == Correction("beam_hardening", {"coefficients": [1.0, 0.05]})
    np.testing.assert_allclose(np.asarray(again.projections), np.asarray(scan.projections))


def test_a_processed_file_loads_as_it_is(tmp_path: Path) -> None:
    dataset = write_projection_dataset(tmp_path / "scan.nxs")
    scan = tj.load(tmp_path / "scan.nxs")
    assert scan.corrections == ()
    np.testing.assert_array_equal(np.asarray(scan.projections), dataset.projections)


def test_batches_give_the_same_result(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    counts = rng.integers(200, 1000, (7, 3, 4)).astype(np.uint16)
    stack = np.concatenate(
        [np.full((2, 3, 4), 1200, np.uint16), counts, np.full((1, 3, 4), 1300, np.uint16)]
    )
    _write_frames(tmp_path / "scan.nxs", stack, [1, 1, *[0] * 7, 1], list(range(10)))
    frames = tj.load_frames(tmp_path / "scan.nxs")
    whole = frames.corrected(BeamHardening((1.0, 0.1)))
    batched = frames.corrected(BeamHardening((1.0, 0.1)), batch_views=2)
    np.testing.assert_allclose(
        np.asarray(batched.projections), np.asarray(whole.projections), rtol=1e-6
    )


def test_beam_hardening_is_the_polynomial(tmp_path: Path) -> None:
    write_raw_nxtomo(tmp_path / "raw.nxs")
    scan = tj.load(tmp_path / "raw.nxs")
    p = np.asarray(scan.projections)
    hardened = scan.corrected(BeamHardening((1.0, 0.05)))
    np.testing.assert_allclose(np.asarray(hardened.projections), p + 0.05 * p**2, rtol=1e-6)
    with pytest.raises(ValueError, match="at least one coefficient"):
        BeamHardening(())


def test_scan_corrected_refuses_steps_before_the_log(tmp_path: Path) -> None:
    from dataclasses import dataclass
    from typing import ClassVar

    from tomojax.corrections import Step

    @dataclass(frozen=True)
    class Gain(Step):
        domain: ClassVar = "counts"

        def apply(self, batch):
            return batch * 2

    write_raw_nxtomo(tmp_path / "raw.nxs")
    with pytest.raises(ValueError, match=r"load the frames with tomojax.load_frames"):
        tj.load(tmp_path / "raw.nxs").corrected(Gain())
    # On the frames a gain on the counts cancels against nothing: 2 I / F.
    scan = tj.load_frames(tmp_path / "raw.nxs").corrected(Gain())
    np.testing.assert_allclose(
        np.asarray(scan.projections)[:, 0, 0], -np.log([9 / 10, 17 / 10]), rtol=1e-6
    )
    assert [c.name for c in scan.corrections] == ["gain", "flat_dark", "log"]


def test_a_nikon_scan_is_corrected_with_its_white_level(tmp_path: Path) -> None:
    image = np.asarray([[30000, 30000], [15000, 15000]], np.uint16)  # top row first
    for i in range(2):
        iio.imwrite(tmp_path / f"part_{i + 1:04d}.tif", image)
    (tmp_path / "part.xtekct").write_text(
        "[XTekCT]\nName=part\nSrcToObject=12.0\nSrcToDetector=18.0\nDetectorPixelSizeX=0.1\n"
        "DetectorPixelSizeY=0.1\nWhiteLevel=60000\nInitialAngle=0\nAngularStep=90\n"
    )
    scan = tj.load(tmp_path / "part.xtekct")
    # Detector v points up: the image's bottom row is the scan's first.
    np.testing.assert_allclose(
        np.asarray(scan.projections)[0, :, 0], -np.log([0.25, 0.5]), rtol=1e-6
    )
    assert scan.corrections[0] == Correction("flat_dark", {"white_level": 60000.0, "darks": 0})
    assert type(scan.geometry).__name__ == "ConeGeometry"
