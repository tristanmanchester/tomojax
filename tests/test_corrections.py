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
from tomojax.corrections import BeamHardening, Correction, Paganin, RejectViews, Stripes, Zingers

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


def test_stripes_removes_pixel_offsets_the_flats_miss(tmp_path: Path) -> None:
    views, nv, nu = 40, 3, 64
    u = np.arange(nu)[None, None, :]
    centre = 32 + 12 * np.sin(np.linspace(0, 2 * np.pi, views))[:, None, None]
    truth = 2.0 * np.maximum(0.0, 1.0 - ((u - centre) / 18.0) ** 2) * np.ones((1, nv, 1))
    gain = np.ones(nu)
    gain[[10, 31, 45]] = [0.97, 1.03, 0.98]  # pixels the flats do not describe
    stack = tmp_path / "views"
    stack.mkdir()
    for i, image in enumerate(1000.0 * np.exp(-truth) * gain):
        iio.imwrite(stack / f"p_{i:03d}.tif", image.astype(np.float32))
    frames = tj.load_frames(
        stack,
        angles=np.linspace(0.0, 360.0, views, endpoint=False),
        flats=np.full((nv, nu), 1000.0),
    )
    plain = np.asarray(frames.corrected().projections)
    cleaned = frames.corrected(Stripes(9))
    # Columns 10 and 31 stand out from their neighbours; column 45's offset is
    # smaller than the gradient across it and stays.
    error = np.abs(np.asarray(cleaned.projections) - truth).max(axis=(0, 1))
    assert error[[10, 31]].max() < 2e-3
    assert np.abs(plain - truth).max(axis=(0, 1))[[10, 31]].min() > 0.02
    assert np.delete(error, [10, 31, 45]).max() < 2e-3
    assert cleaned.corrections[-1] == Correction("stripes", {"width": 9})
    # A step after a whole-row step runs on its result.
    both = frames.corrected(Stripes(9), BeamHardening((1.0, 0.05)), batch_views=7)
    p = np.asarray(cleaned.projections)
    np.testing.assert_allclose(np.asarray(both.projections), p + 0.05 * p**2, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match=">= 3"):
        Stripes(2)


def test_reject_views_drops_outliers_and_their_geometry(tmp_path: Path) -> None:
    views = 12
    counts = np.random.default_rng(1).integers(495, 506, (views, 2, 3)).astype(np.uint16)
    counts[[4, 9]] = 5  # the shutter closed for two views
    _write_frames(
        tmp_path / "scan.nxs",
        np.concatenate([np.full((1, 2, 3), 1000, np.uint16), counts]),
        [1, *[0] * views],
        [0, *np.linspace(0, 180, views, endpoint=False)],
    )
    scan = tj.load_frames(tmp_path / "scan.nxs").corrected(RejectViews(), BeamHardening((1.0, 0.1)))
    keep = np.delete(np.arange(views), [4, 9])
    assert scan.projections.shape == (10, 2, 3)
    np.testing.assert_allclose(scan.angles, np.linspace(0, 180, views, endpoint=False)[keep])
    p = -np.log(counts[keep] / 1000.0)
    np.testing.assert_allclose(np.asarray(scan.projections), p + 0.1 * p**2, rtol=1e-5)
    rejected = scan.corrections[-2]
    assert rejected.name == "reject_views" and rejected.found["rejected"] == [4, 9]
    # A scan of line integrals rejects views the same way, numbered in that scan.
    again = tj.load(tmp_path / "scan.nxs").selected(slice(2, None)).corrected(RejectViews())
    assert again.corrections[-1].found["rejected"] == [2, 7]
    assert again.projections.shape[0] == 8


def test_selected_views_keep_their_geometry_and_flats(tmp_path: Path) -> None:
    frames = np.stack([np.full((2, 2), v, np.float32) for v in (10, 8, 8, 8, 8, 20)])
    _write_frames(tmp_path / "scan.nxs", frames, [1, 0, 0, 0, 0, 1], [0, 0, 45, 90, 135, 0])
    loaded = tj.load_frames(tmp_path / "scan.nxs")
    picked = loaded.selected([0, 3])
    np.testing.assert_allclose(picked.geometry.angles, [0, 135])
    flats = 10 + 10 * np.asarray([0.5, 3.5]) / 4  # interpolated at their places in the scan
    np.testing.assert_allclose(
        np.asarray(picked.corrected().projections)[:, 0, 0], -np.log(8 / flats), rtol=1e-5
    )
    whole = loaded.corrected()
    by_mask = whole.selected(np.asarray([True, False, False, True]))
    np.testing.assert_allclose(
        np.asarray(by_mask.projections), np.asarray(picked.corrected().projections)
    )
    with pytest.raises(ValueError, match="increasing"):
        whole.selected([3, 0])


def test_selected_views_of_a_posed_multi_orbit_scan() -> None:
    from tomojax.geometry import ConeBeam, ConeGeometry

    grid, detector = tiny_grid(nx=4, ny=4, nz=4), tiny_detector(nu=4, nv=4)
    orbits = [
        tj.Scan(
            np.full((3, 4, 4), k, np.float32),
            ConeGeometry(grid, detector, [0.0, 120.0, 240.0], ConeBeam(20.0, 30.0 + k)),
        )
        for k in range(2)
    ]
    combined = tj.Scan.combine(orbits)
    picked = combined.selected([1, 2, 4])
    assert picked.projections[:, 0, 0].tolist() == [0.0, 0.0, 1.0]
    np.testing.assert_allclose(picked.angles, [120.0, 240.0, 120.0])
    assert [s.beam.source_to_detector for s in picked.geometry.segments] == [30.0, 31.0]
    one_orbit = combined.selected([3, 5])
    assert isinstance(one_orbit.geometry, ConeGeometry)


def _frames(counts: np.ndarray, flat: float) -> tj.Frames:
    views, rows, cols = counts.shape
    geometry = tj.ParallelGeometry(
        tiny_grid(nx=cols, ny=cols, nz=rows),
        tiny_detector(nu=cols, nv=rows),
        np.linspace(0.0, 180.0, views, endpoint=False),
    )
    return tj.Frames(counts, geometry, flats=np.full((1, rows, cols), flat, np.float32))


def test_zingers_are_replaced_by_their_neighbours_median() -> None:
    counts = np.full((2, 5, 6), 500.0, np.float32)
    counts[0, 2, 3] = 1000.0  # a zinger
    counts[1, 1:4, 1:4] = 520.0  # a faint patch: not a zinger
    scan = _frames(counts, 1000.0).corrected(Zingers())
    p = np.asarray(scan.projections)
    np.testing.assert_allclose(p[0], np.log(2.0), rtol=1e-6)
    np.testing.assert_allclose(p[1], -np.log(counts[1] / 1000.0), rtol=1e-6)
    assert scan.corrections[1] == Correction("zingers", {"threshold": 0.1, "size": 3})
    with pytest.raises(ValueError, match="odd"):
        Zingers(size=4)


def test_paganin_recovers_the_attenuation_of_a_phase_contrast_image() -> None:
    n, pixel, distance, energy, delta_beta = 64, 1e-6, 0.05, 20.0, 300.0
    y, x = np.mgrid[:n, :n] - n / 2 + 0.5
    mu_t = 0.4 * np.exp(-(x**2 + y**2) / (2 * 5.0**2))  # a smooth blob, mu * thickness
    absorbed = np.exp(-mu_t)
    # Near-field propagation (transport of intensity): I = (1 - z delta / mu laplacian) e^{-mu t}.
    wavelength = 12.398419843320026e-10 / energy
    k = 2 * np.pi * np.fft.fftfreq(n, pixel)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    scale = distance * wavelength * delta_beta / (4 * np.pi)
    image = np.fft.ifft2(np.fft.fft2(absorbed) * (1 + scale * k2)).real
    assert np.abs(image - absorbed).max() > 0.05  # strong edge fringes
    frames = _frames((1000.0 * image)[None].astype(np.float32), 1000.0)
    step = Paganin(delta_beta=delta_beta, distance=distance, energy_kev=energy, pixel_size=pixel)
    scan = frames.corrected(step)
    np.testing.assert_allclose(np.asarray(scan.projections)[0], mu_t, atol=2e-3)
    assert [c.name for c in scan.corrections] == ["flat_dark", "paganin", "log"]
    with pytest.raises(ValueError, match="positive"):
        Paganin(delta_beta=0, distance=1, energy_kev=1, pixel_size=1)
