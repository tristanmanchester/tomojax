"""Detector frames to line integrals: tj.load_frames, Frames.corrected and the steps."""

from __future__ import annotations

from dataclasses import replace
import importlib
import json
from typing import TYPE_CHECKING, Any

import h5py
import imageio.v3 as iio
import numpy as np
import pytest
import tifffile

import tomojax as tj
from tomojax.corrections import (
    BeamHardening,
    Correction,
    Paganin,
    RejectViews,
    Stripes,
    Zingers,
)
from tomojax.io.api import locate_frames, read_tiff_frames

from ._helpers import tiny_detector, tiny_grid, write_projection_dataset, write_raw_nxtomo

if TYPE_CHECKING:
    from pathlib import Path

    from tomojax.corrections import Step

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
    write_raw_nxtomo(tmp_path / "raw.hdf")  # sample 5, flat 11, sample 9, dark 1
    scan = tj.load(tmp_path / "raw.hdf")
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


@pytest.mark.parametrize("recorded", [True, False])
def test_frame_center_provenance_preserves_explicit_overrides(tmp_path, recorded):
    path = tmp_path / "raw.nxs"
    write_raw_nxtomo(path)
    if not recorded:
        with h5py.File(path, "r+") as handle:
            del handle["entry/instrument/detector"].attrs["detector_meta_json"]
    if not recorded:
        with pytest.raises(ValueError, match="detector="):
            tj.load_frames(path)
    frames = tj.load_frames(path, **({} if recorded else {"detector": tiny_detector(nu=2)}))
    status = "recorded" if recorded else "provided"
    assert frames.source.geometry_metadata["detector_center_status"] == status
    for kwargs in ({"detector": frames.geometry.detector}, {"geometry": frames.geometry}):
        explicit = tj.load_frames(path, **kwargs)
        assert explicit.source.geometry_metadata["detector_center_status"] == "provided"
    saved = tmp_path / "saved.nxs"
    tj.save(saved, frames.corrected())
    assert tj.load(saved).source.geometry_metadata["detector_center_status"] == status


@pytest.mark.parametrize("index", [0, 1, -1, -2, np.int64(-1)])
def test_lazy_frame_integer_indices_match_numpy(tmp_path: Path, index: int) -> None:
    write_raw_nxtomo(tmp_path / "raw.nxs")
    frames = tj.load_frames(tmp_path / "raw.nxs").cropped(slice(1, 2), slice(0, 1))
    expected = np.asarray(frames.counts[:])
    np.testing.assert_array_equal(frames.counts[index], expected[index])


@pytest.mark.parametrize("index", [2, 3, -3, -4])
def test_lazy_frame_integer_indices_out_of_bounds_raise(tmp_path: Path, index: int) -> None:
    write_raw_nxtomo(tmp_path / "raw.nxs")
    frames = tj.load_frames(tmp_path / "raw.nxs")
    with pytest.raises(IndexError):
        _ = frames.counts[index]


def test_flats_before_and_after_are_interpolated_by_position(tmp_path: Path) -> None:
    # flat 10, four views of 8, flat 20, a dark of 0: view i sits at i + 1/2 of 4.
    frames = np.stack([np.full((2, 3), v, np.uint16) for v in (10, 8, 8, 8, 8, 20, 0)])
    _write_frames(tmp_path / "scan.nxs", frames, [1, 0, 0, 0, 0, 1, 2], [0, 0, 45, 90, 135, 0, 0])
    scan = tj.load(tmp_path / "scan.nxs")
    flats = 10 + 10 * (np.arange(4) + 0.5) / 4
    np.testing.assert_allclose(np.asarray(scan.projections)[:, 0, 0], -np.log(8 / flats), rtol=1e-5)
    settings = scan.corrections[0].settings
    assert settings["flat_sets"] == 2 and settings["flat_positions"] == [0.0, 4.0]


def test_missing_dark_requires_explicit_calibration(tmp_path: Path) -> None:
    frames = np.stack([np.full((2, 2), v, np.float32) for v in (8, 4)])
    _write_frames(tmp_path / "scan.nxs", frames, [1, 0], [0, 0])
    for load in (tj.load, tj.load_frames):
        with pytest.raises(ValueError, match="darks="):
            load(tmp_path / "scan.nxs")
    scan = tj.load_frames(tmp_path / "scan.nxs", darks=0).corrected()
    np.testing.assert_allclose(np.asarray(scan.projections), np.log(2.0), rtol=1e-6)
    assert scan.corrections[0].settings["darks"] == 1


def test_counts_without_flats_are_refused(tmp_path: Path) -> None:
    frames = np.full((2, 2, 2), 100, np.uint16)
    _write_frames(tmp_path / "scan.nxs", frames, [0, 0], [0, 90])
    with pytest.raises(ValueError, match="integer detector counts and no flat frames"):
        tj.load(tmp_path / "scan.nxs")
    with pytest.raises(ValueError, match="No flat fields: pass flats"):
        tj.load_frames(tmp_path / "scan.nxs").corrected()
    scan = tj.load_frames(tmp_path / "scan.nxs", white_level=200, darks=0).corrected()
    np.testing.assert_allclose(np.asarray(scan.projections), np.log(2.0), rtol=1e-6)
    assert scan.corrections[0].settings == {"white_level": 200.0, "darks": 1}


@pytest.mark.parametrize("suffix", [".tif", ".hdf"])
def test_tiff_stacks_go_through_load_frames(tmp_path: Path, suffix: str) -> None:
    views = tmp_path / "views"
    views.mkdir()
    for i, value in enumerate((50, 25)):
        iio.imwrite(views / f"view_{i:04d}.tif", np.full((2, 3), value, np.uint16))
    flat, dark = (tmp_path / f"{kind}{suffix}" for kind in ("flat", "dark"))
    for path, value in ((flat, 100), (dark, 0)):
        data = np.full((2, 3), value, np.uint16)
        if suffix == ".hdf":
            with h5py.File(path, "w") as handle:
                handle.create_dataset("entry/data/data", data=data[None])
        else:
            iio.imwrite(path, data)
    with pytest.raises(ValueError, match="load_frames"):
        tj.load(views)
    with pytest.raises(ValueError, match="angles"):
        tj.load_frames(views / "view_0000.tif")
    frames = tj.load_frames(
        views, angles=[0, 90], flats=flat, darks=dark, detector=tiny_detector(nu=3)
    )
    assert not any(isinstance(field, np.ndarray) for field in (frames.flats, frames.darks))
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
    frames = tj.load_frames(tmp_path / "scan.nxs", darks=0)
    whole = frames.corrected(BeamHardening((1.0, 0.1)))
    batched = frames.corrected(BeamHardening((1.0, 0.1)), batch_views=2)
    np.testing.assert_allclose(
        np.asarray(batched.projections), np.asarray(whole.projections), rtol=1e-6
    )


def test_automatic_batch_budget_preserves_workspace_steps() -> None:
    # check-public-imports: allow-private
    from tomojax.corrections._engine import _frame_batch_size

    assert _frame_batch_size(1000, 512, 512, ()) == 32
    assert _frame_batch_size(7, 512, 512, ()) == 7
    assert _frame_batch_size(10, 4096, 4096, ()) == 1
    phase = Paganin(delta_beta=100, distance=0.1, energy_kev=20, pixel_size=1e-6)
    pixels = 512 * 512 + phase.workspace_pixels(512, 512)
    assert _frame_batch_size(1000, 512, 512, (phase,)) == (256 << 20) // (4 * pixels)


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
        "[XTekCT]\nDetectorOffsetX=0\nDetectorOffsetY=0\n"
        "CentreOfRotationTop=0\nCentreOfRotationBottom=0\nName=part\nSrcToObject=12.0\nSrcToDetector=18.0\nDetectorPixelSizeX=0.1\n"
        "DetectorPixelSizeY=0.1\nWhiteLevel=60000\nInitialAngle=0\nAngularStep=90\n"
    )
    frames = tj.load_frames(tmp_path / "part.xtekct", darks=0)
    assert not isinstance(frames.counts, np.ndarray) and frames.counts.dtype == np.uint16
    window = frames.cropped(slice(0, 1), slice(0, 1)).selected([1])
    np.testing.assert_array_equal(np.asarray(window.counts), [[[15000]]])
    scan = frames.corrected()
    # Detector v points up: the image's bottom row is the scan's first.
    np.testing.assert_allclose(
        np.asarray(scan.projections)[0, :, 0], -np.log([0.25, 0.5]), rtol=1e-6
    )
    assert scan.corrections[0] == Correction(
        "flat_dark", {"white_level": 60000.0, "darks": 1}, {"pixels_without_light": 0}
    )
    assert type(scan.geometry).__name__ == "ConeGeometry"


@pytest.mark.parametrize("measured", [False, True])
def test_nikon_import_takes_its_own_angles_and_numbered_projections(
    tmp_path: Path, measured: bool
) -> None:
    for view, level in [(10, 250), (2, 500)]:
        iio.imwrite(tmp_path / f"part_{view}.tif", np.full((2, 3), level, np.uint16))
    for suffix in ["flat", "2_thumbnail", "reconstruction"]:
        iio.imwrite(tmp_path / f"part_{suffix}.TIFF", np.full((1, 1), 123, np.uint16))
    path = tmp_path / "renamed.xtekct"
    path.write_text(
        "[XTekCT]\nDetectorOffsetX=0\nDetectorOffsetY=0\n"
        "CentreOfRotationTop=0\nCentreOfRotationBottom=0\nName=part\nSrcToObject=12\nSrcToDetector=18\nDetectorPixelSizeX=0.1\n"
        "DetectorPixelSizeY=0.1\nWhiteLevel=1000\nInitialAngle=5\nAngularStep=90\n"
    )
    (tmp_path / "another_ctdata.txt").write_text("Projection Angle\n1 -60\n2 -30\n")
    if measured:
        (tmp_path / "part_ctdata.txt").write_text("Projection Angle\n10 97\n2 7\n")
    frames = tj.load_frames(path, darks=0)
    np.testing.assert_array_equal(np.asarray(frames.counts)[:, 0, 0], [500, 250])
    np.testing.assert_array_equal(frames.geometry.angles, [7, 97] if measured else [5, 95])

    # A partial measured sidecar cannot silently become a nominal uniform scan.
    (tmp_path / "part_ctdata.txt").write_text("Projection Angle\n1 7\n")
    with pytest.raises(ValueError, match="angles="):
        tj.load_frames(path, darks=0)
    (tmp_path / "part_ctdata.txt").unlink()
    text = path.read_text()
    for field, argument in {
        "InitialAngle": "angles=",
        "AngularStep": "angles=",
        "WhiteLevel": "white_level=",
        "DetectorOffsetX": "detector=",
        "CentreOfRotationTop": "geometry=",
    }.items():
        path.write_text("\n".join(line for line in text.splitlines() if not line.startswith(field)))
        with pytest.raises(ValueError, match=argument):
            tj.load_frames(path, darks=0)
        explicit = tj.load_frames(
            path, geometry=frames.geometry, white_level=1000, darks=0
        ).corrected()
        np.testing.assert_allclose(explicit.projections[:, 0, 0], -np.log([0.5, 0.25]))
        tj.save(tmp_path / "explicit.nxs", explicit)
        assert tj.load(tmp_path / "explicit.nxs").geometry is not None


def test_big_endian_counts_are_corrected_like_native_ones() -> None:
    counts = (300 + np.arange(2 * 3 * 4).reshape(2, 3, 4) * 3).astype(">u2")
    geometry = tj.ParallelGeometry(tiny_grid(), tj.Detector(4, 3, 1.0, 1.0), [0, 90])
    flats, darks = np.full((1, 3, 4), 950, ">u2"), np.full((1, 3, 4), 20, ">u2")
    scan = tj.Frames(counts, geometry, flats=flats, darks=darks).corrected(batch_views=1)
    expected = -np.log((counts.astype(np.float64) - 20) / (950 - 20))
    np.testing.assert_allclose(scan.projections, expected, rtol=2e-6)


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
        darks=0,
        detector=tj.Detector(nu, nv, 1, 1),
    )
    plain = np.asarray(frames.corrected().projections)
    cleaned = frames.corrected(Stripes(9))
    # Columns 10 and 31 stand out from their neighbours; column 45's offset is
    # smaller than the gradient across it and stays.
    error = np.abs(np.asarray(cleaned.projections) - truth).max(axis=(0, 1))
    assert error[[10, 31]].max() < 2e-3
    assert np.abs(plain - truth).max(axis=(0, 1))[[10, 31]].min() > 0.02
    assert np.delete(error, [10, 31, 45]).max() < 2e-3
    assert cleaned.corrections[-1] == Correction("stripes", {"width": 9, "varying": False})
    # A step after a whole-row step runs on its result.
    both = frames.corrected(Stripes(9), BeamHardening((1.0, 0.05)), batch_views=7)
    p = np.asarray(cleaned.projections)
    np.testing.assert_allclose(np.asarray(both.projections), p + 0.05 * p**2, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match=">= 3"):
        Stripes(2)


def test_reject_views_drops_outliers_and_their_geometry(tmp_path: Path) -> None:
    views = 12
    counts = np.random.default_rng(1).integers(495, 506, (views, 2, 3)).astype(np.uint16)
    counts[[0, 4, 8, 11]] = 5  # the shutter closed, including the first and last views
    _write_frames(
        tmp_path / "scan.nxs",
        np.concatenate([np.full((1, 2, 3), 1000, np.uint16), counts]),
        [1, *[0] * views],
        [0, *np.linspace(0, 180, views, endpoint=False)],
    )
    scan = tj.load_frames(tmp_path / "scan.nxs", darks=0).corrected(
        RejectViews(), BeamHardening((1.0, 0.1))
    )
    keep = np.delete(np.arange(views), [0, 4, 8, 11])
    assert scan.projections.shape == (8, 2, 3)
    np.testing.assert_allclose(scan.angles, np.linspace(0, 180, views, endpoint=False)[keep])
    p = -np.log(counts[keep] / 1000.0)
    np.testing.assert_allclose(np.asarray(scan.projections), p + 0.1 * p**2, rtol=1e-5)
    rejected = scan.corrections[-2]
    assert rejected.name == "reject_views" and rejected.found["rejected"] == [0, 4, 8, 11]
    # A scan of line integrals rejects views the same way, numbered in that scan.
    again = (
        tj.load_frames(tmp_path / "scan.nxs", darks=0)
        .corrected()
        .selected(slice(2, None))
        .corrected(RejectViews())
    )
    assert again.corrections[-1].found["rejected"] == [2, 6, 9]
    assert again.projections.shape[0] == 7


def test_selected_views_keep_their_geometry_and_flats(tmp_path: Path) -> None:
    frames = np.stack([np.full((2, 2), v, np.float32) for v in (10, 8, 8, 8, 8, 20)])
    _write_frames(tmp_path / "scan.nxs", frames, [1, 0, 0, 0, 0, 1], [0, 0, 45, 90, 135, 0])
    loaded = tj.load_frames(tmp_path / "scan.nxs", darks=0)
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


def test_cropping_reads_a_block_and_moves_the_detector(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    stack = np.concatenate(
        [np.full((1, 6, 8), 1000, np.uint16), rng.integers(100, 900, (3, 6, 8)).astype(np.uint16)]
    )
    _write_frames(tmp_path / "scan.nxs", stack, [1, 0, 0, 0], [0, 0, 60, 120])
    frames = tj.load_frames(tmp_path / "scan.nxs", darks=0)
    part = frames.cropped(slice(1, 4), slice(2, 7))
    assert part.counts.shape == (3, 3, 5) and part.flats.shape == (1, 3, 5)
    np.testing.assert_array_equal(np.asarray(part.counts[0:3]), stack[1:, 1:4, 2:7])
    # The block's centre, in the detector's units: columns 2..6 of 8, rows 1..3 of 6.
    assert part.geometry.detector.center == (0.5, -0.5)
    whole = frames.corrected()
    np.testing.assert_allclose(
        np.asarray(part.corrected().projections),
        np.asarray(whole.cropped(slice(1, 4), slice(2, 7)).projections),
    )
    with pytest.raises(ValueError, match="non-empty block"):
        whole.cropped(slice(3, 3), slice(None))


def test_frames_are_found_in_other_hdf5_layouts(tmp_path: Path) -> None:
    # A Diamond-style file: frames, keys and angles (radians) under entry1/tomo_entry.
    stack = np.stack([np.full((2, 3), v, np.uint16) for v in (0, 1000, 500, 250, 1000)])
    with h5py.File(tmp_path / "beamline.nxs", "w") as handle:
        entry = handle.create_group("entry1/tomo_entry")
        entry.create_dataset("data/data", data=stack)
        entry.create_dataset("instrument/detector/image_key", data=[2, 1, 0, 0, 1])
        angles = entry.create_dataset("data/rotation_angle", data=np.radians([0, 0, 0, 90, 90]))
        angles.attrs["units"] = "rad"
        handle.create_dataset("extra/images", data=stack)  # an unrelated stack
    scan = tj.load_frames(tmp_path / "beamline.nxs", detector=tiny_detector(nu=3)).corrected()
    np.testing.assert_allclose(scan.angles, [0.0, 90.0])
    np.testing.assert_allclose(
        np.asarray(scan.projections)[:, 0, 0], -np.log([0.5, 0.25]), rtol=1e-6
    )
    assert scan.detector.nu == 3 and scan.detector.du == 1.0
    named = tj.load_frames(
        tmp_path / "beamline.nxs",
        data_path="/extra/images",
        angles=[0, 45],
        detector=tiny_detector(nu=3),
    )
    assert named.views == 2 and len(named.flats) == 2  # the file's one image_key fits it
    np.testing.assert_allclose(named.geometry.angles, [0, 45])
    with pytest.raises(KeyError, match="no dataset"):
        tj.load_frames(tmp_path / "beamline.nxs", data_path="/missing")


@pytest.mark.parametrize("command", [True, False])
def test_gda_frames_use_paired_rotation_and_key(tmp_path, caplog, command):
    path = tmp_path / "arbitrary-name.nxs"
    stack = np.stack([np.full((2, 3), v, np.int32) for v in (100, 1100, 600, 350)])
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry7")
        entry.create_dataset("imaging/data", data=stack)
        entry.create_dataset("instrument/EtherCAT/image_key", data=[2, 1, 0, 0])
        name = "imaging/smaract_zrot_value_set" if command else "imaging_sum/smaract_zrot"
        angle = entry.create_dataset(name, data=[0.0, 0.0, 0.0, 90.0])
        angle.attrs["units"] = "mm" if command else "degree"
        entry.create_dataset("imaging/smaract_zrot_value", data=np.arange(4) * 1e6)
        handle.create_dataset("other/data", data=stack)
        handle.create_dataset("other/image_key", data=[0, 0, 0, 0])
    frames = tj.load_frames(path, detector=tiny_detector(nu=3))
    assert frames.views == 2 and len(frames.flats) == len(frames.darks) == 1
    assert frames.counts.dataset == "/entry7/imaging/data"
    np.testing.assert_array_equal(frames.geometry.angles, [0, 90])
    acquisition = frames.source.geometry_metadata["acquisition"]
    assert acquisition["geometry_status"] == "not_recorded"
    assert frames.source.geometry_metadata["detector_center_status"] == "provided"
    assert ("interpreting it as degrees" in caplog.text) == command
    np.testing.assert_allclose(frames.corrected().projections[:, 0, 0], -np.log([0.5, 0.25]))
    explicit = tj.load_frames(path, geometry=frames.geometry)
    assert explicit.source.geometry_metadata["acquisition"]["geometry_status"] == "provided"


def test_gda_unit_correction_does_not_accept_other_linear_axes(tmp_path):
    path = tmp_path / "unknown-units.nxs"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/imaging/data", data=np.ones((2, 2, 3)))
        handle.create_dataset("entry/instrument/EtherCAT/image_key", data=[0, 0])
        angle = handle.create_dataset("entry/imaging_sum/smaract_zrot", data=[0, 90])
        angle.attrs["units"] = "mm"
    with pytest.raises(ValueError, match="unsupported angle units"):
        tj.load_frames(path)


def test_constant_flat_and_dark_levels(tmp_path: Path) -> None:
    _write_frames(tmp_path / "scan.nxs", np.full((2, 2, 2), 600, np.uint16), [0, 0], [0, 90])
    scan = tj.load_frames(tmp_path / "scan.nxs", flats=1100, darks=100).corrected()
    np.testing.assert_allclose(np.asarray(scan.projections), np.log(2.0), rtol=1e-6)
    assert scan.corrections[0].settings == {"white_level": 1100.0, "darks": 1}


def test_tiff_frames_take_a_measured_geometry(tmp_path: Path) -> None:
    for i, value in enumerate((50, 25, 40)):
        iio.imwrite(tmp_path / f"view_{i:04d}.tif", np.full((2, 3), value, np.uint16))
    geometry = tj.LaminographyGeometry(
        tiny_grid(nx=3, ny=3, nz=2),
        tj.Detector(nu=3, nv=2, du=0.65, dv=0.65),
        [0.0, 120.0, 240.0],
        tilt_deg=30.0,
    )
    scan = tj.load_frames(tmp_path, flats=100, darks=0, geometry=geometry).corrected()
    assert scan.geometry is geometry and scan.detector.du == 0.65
    np.testing.assert_allclose(np.asarray(scan.projections)[:, 0, 0], -np.log([0.5, 0.25, 0.4]))


def test_aps_data_exchange_files_load_with_their_flats_and_darks(tmp_path: Path) -> None:
    # APS's layout (TomoPy, tomocupy): flats and darks as stacks of their own.
    with h5py.File(tmp_path / "aps.h5", "w") as handle:
        handle.create_dataset(
            "exchange/data", data=np.stack([np.full((2, 3), v, np.uint16) for v in (300, 150)])
        )
        handle.create_dataset("exchange/data_white", data=np.full((3, 2, 3), 500, np.uint16))
        handle.create_dataset("exchange/data_dark", data=np.full((2, 2, 3), 100, np.uint16))
        handle.create_dataset("exchange/theta", data=np.asarray([0.0, 90.0], np.float32))
        handle.create_dataset("defaults/NDArrayUniqueId", data=np.arange(5))
    scan = tj.load_frames(tmp_path / "aps.h5", detector=tiny_detector(nu=3)).corrected()
    np.testing.assert_allclose(
        np.asarray(scan.projections)[:, 0, 0], -np.log([0.5, 0.125]), rtol=1e-6
    )
    np.testing.assert_allclose(scan.angles, [0.0, 90.0])
    assert scan.corrections[0].settings == {"flats": 3, "darks": 2, "flat_sets": 1}
    # Without flats a foreign file says where to go.
    with h5py.File(tmp_path / "bare.h5", "w") as handle:
        handle.create_dataset("exchange/data", data=np.full((2, 2, 3), 7, np.uint16))
    with pytest.raises(ValueError, match="load_frames"):
        tj.load(tmp_path / "bare.h5")


def test_pixels_without_a_resolved_flat_take_no_absorption() -> None:
    rng = np.random.default_rng(6)
    views, rows, cols = 4, 5, 8
    flats = np.full((20, rows, cols), 1000.0) + rng.normal(0, 3, (20, rows, cols))
    darks = np.full((20, rows, cols), 10.0) + rng.normal(0, 3, (20, rows, cols))
    flats[..., :2] = darks[..., :2] = 10 + rng.normal(0, 3, (20, rows, 2))  # outside the beam
    counts = rng.uniform(100.0, 900.0, (views, rows, cols))
    counts[..., :2] = rng.uniform(5.0, 15.0, (views, rows, 2))
    geometry = tj.ParallelGeometry(
        tj.Grid(cols, cols, rows, 1.0, 1.0, 1.0),
        tj.Detector(cols, rows, 1.0, 1.0),
        np.arange(views) * 45.0,
    )
    scan = tj.Frames(counts, geometry, flats=flats, darks=darks).corrected()
    projections = np.asarray(scan.projections)
    np.testing.assert_array_equal(projections[..., :2], 0.0)
    lit = np.s_[..., 2:]
    expected = -np.log((counts - darks.mean(0))[lit] / (flats.mean(0) - darks.mean(0))[lit])
    np.testing.assert_allclose(projections[lit], expected, rtol=1e-5, atol=1e-6)
    assert scan.corrections[0].found == {"pixels_without_light": 2 * rows}


@pytest.mark.parametrize("source", ["array", "hdf5", "tiff"])
@pytest.mark.parametrize("steps", [(), (Stripes(5),), (Stripes(5), RejectViews())])
@pytest.mark.parametrize("factor", [1, 2])
def test_row_slabs_match_view_batches_and_optional_binning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    steps: tuple[Step, ...],
    source: str,
    factor: int,
) -> None:
    rng = np.random.default_rng(7)
    views, rows, cols = 12, 10, 14
    counts = rng.uniform(300.0, 900.0, (views, rows, cols)).astype(np.uint16)
    counts[5] //= 4  # a view for RejectViews to drop
    stack = np.concatenate([np.full((1, rows, cols), 1000, np.uint16), counts])
    _write_frames(tmp_path / "scan.nxs", stack, [1, *[0] * views], [0.0, *range(0, views * 15, 15)])
    frames = tj.load_frames(tmp_path / "scan.nxs", darks=0)
    if source == "array":
        frames = replace(frames, counts=np.asarray(frames.counts), flats=np.asarray(frames.flats))
    elif source == "tiff":
        tifffile.imwrite(tmp_path / "counts.tif", counts, photometric="minisblack")
        frames = replace(frames, counts=read_tiff_frames(tmp_path / "counts.tif", lazy=True))
    expected = frames.corrected(*steps, batch_views=3).binned(factor)

    # A small budget makes the row-by-row path take several slabs where it applies.
    engine = importlib.import_module("tomojax.corrections._engine")
    monkeypatch.setattr(engine, "_BATCH_BYTES", 5376)
    slabs: list[int] = []
    correct_slabs = engine._correct_row_slabs

    def spy(*args: Any, **kwargs: Any) -> Any:
        slabs.append(kwargs["slab"])
        return correct_slabs(*args, **kwargs)

    monkeypatch.setattr(engine, "_correct_row_slabs", spy)
    actual = frames.corrected(*steps, binning=factor)

    np.testing.assert_allclose(actual.projections, expected.projections, rtol=1e-5, atol=1e-6)
    assert actual.detector == expected.detector and actual.corrections == expected.corrections
    np.testing.assert_array_equal(actual.angles, expected.angles)
    streamed = steps == (Stripes(5),) and (source != "tiff" or factor > 1)
    assert slabs == ([4] if streamed else [])


def test_cropping_reads_lazy_flats_and_darks_as_a_block(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    stack = rng.integers(100, 900, (5, 6, 8)).astype(np.uint16)
    _write_frames(tmp_path / "scan.nxs", stack, [1, 2, 0, 0, 0], [0, 0, 0, 60, 120])
    part = tj.load_frames(tmp_path / "scan.nxs").cropped(slice(1, 4), slice(2, 7))
    assert type(part.flats).__name__ == type(part.darks).__name__ == "Hdf5Frames"
    np.testing.assert_array_equal(np.asarray(part.flats), stack[:1, 1:4, 2:7])
    np.testing.assert_array_equal(np.asarray(part.darks), stack[1:2, 1:4, 2:7])


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_tiff_stacks_are_read_lazily_by_window(tmp_path: Path, compression: str | None) -> None:
    rng = np.random.default_rng(4)
    pages = rng.integers(0, 60000, (5, 6, 9)).astype(np.uint16)
    for name, part in (("a.tif", pages[:2]), ("b.tif", pages[2:])):
        with tifffile.TiffWriter(tmp_path / name) as writer:
            for page in part:
                writer.write(page, compression=compression, contiguous=False)
    frames = read_tiff_frames(tmp_path, lazy=True)
    assert frames.shape == pages.shape and frames._efficient_row_windows == (compression is None)
    np.testing.assert_array_equal(np.asarray(frames), pages)
    part = frames.frames_at(np.array([1, 2, 4])).window(slice(1, 4), slice(2, 7))
    assert part.shape == (3, 3, 5)
    np.testing.assert_array_equal(part[:], pages[[1, 2, 4], 1:4, 2:7])
    np.testing.assert_array_equal(part[1], pages[2, 1:4, 2:7])


@pytest.mark.parametrize(
    ("options", "efficient"),
    [
        ({}, True),
        ({"chunks": (1, 2, 16)}, True),
        ({"chunks": (1, 16, 16)}, True),  # unfiltered windows bypass the chunk cache
        ({"chunks": (2, 16, 16)}, True),
        ({"chunks": (1, 2, 16), "compression": "gzip"}, False),
    ],
)
def test_hdf5_unfiltered_row_windows_use_direct_partial_chunk_reads(
    tmp_path: Path, options: dict[str, Any], efficient: bool
) -> None:
    data = np.arange(3 * 16 * 16, dtype=np.uint16).reshape(3, 16, 16)
    with h5py.File(tmp_path / "scan.h5", "w") as handle:
        handle.create_dataset("exchange/data", data=data, **options)
    stack = locate_frames(str(tmp_path / "scan.h5")).stack
    assert stack._efficient_row_windows is efficient
    part = stack.window(slice(3, 7), slice(2, 8))
    np.testing.assert_array_equal(np.asarray(part), data[:, 3:7, 2:8])


def test_raw_acquisition_discovery_uses_content_and_rejects_ambiguous_directories(tmp_path):
    image = np.array([[300, 300, 300], [150, 150, 150]], np.uint16)
    for i in range(2):
        iio.imwrite(tmp_path / f"object_{i + 1:04d}.tif", image)
    path = tmp_path / "metadata.unrelated"
    path.write_text(
        "[XTekCT]\nDetectorOffsetX=0\nDetectorOffsetY=0\n"
        "CentreOfRotationTop=0\nCentreOfRotationBottom=0\nName=object\nSrcToObject=12\nSrcToDetector=18\n"
        "DetectorPixelSizeX=.1\nDetectorPixelSizeY=.1\nWhiteLevel=600\nInitialAngle=0\nAngularStep=90\n"
    )
    frames = tj.load_frames(path, darks=0)
    by_directory = tj.load_frames(tmp_path, darks=0).corrected()
    np.testing.assert_allclose(by_directory.projections[0, :, 0], -np.log([0.25, 0.5]))
    override = tj.Detector(3, 2, 0.2, 0.3)
    explicit = tj.load_frames(path, detector=override, white_level=1200, darks=0)
    assert explicit.geometry.detector == override and explicit.white_level == 1200
    np.testing.assert_array_equal(frames.counts[0], image[::-1])
    path.rename(tmp_path / "metadata.also_unrelated")
    assert tj.load_frames(tmp_path, darks=0).geometry.detector == frames.geometry.detector
    path.write_text((tmp_path / "metadata.also_unrelated").read_text())
    with pytest.raises(ValueError, match="several Nikon"):
        tj.load_frames(tmp_path, darks=0)


@pytest.mark.parametrize("modern", [False, True])
def test_nikon_calibrated_axis_and_closing_turn_are_physical_metadata(
    tmp_path: Path, modern: bool
) -> None:
    folder = tmp_path / "projections"
    folder.mkdir()
    for i in range(3):
        iio.imwrite(folder / f"part_{i + 1:04d}.tif", np.full((4, 6), 400, np.uint16))
    path = tmp_path / "scan.xtekct"
    path.write_text(
        "[XTekCT]\nDetectorOffsetX=0\nDetectorOffsetY=0\n"
        "CentreOfRotationTop=0\nCentreOfRotationBottom=0\nName=part\nInputFolderName=projections\nProjections=2\n"
        "SrcToObject=10\nSrcToDetector=20\nDetectorPixelSizeX=0.5\n"
        "DetectorPixelSizeY=0.5\nInitialAngle=0\nAngularStep=180\nWhiteLevel=1000\n"
        "CentreOfRotationTop=1\nCentreOfRotationBottom=-0.5\n"
        + ("ObjectOffsetX=0.3\nObjectRoll=0\nObjectTilt=30\n" if modern else "")
        + "[CTPro]\nAngleFile_Use=False\n"
    )
    (tmp_path / "part_ctdata.txt").write_text("1 20\n2 40\n")
    frames = tj.load_frames(path, darks=0)
    geometry = frames.geometry
    assert frames.counts.shape == (2, 4, 6)  # closing frame is not a third view
    np.testing.assert_array_equal(geometry.angles, [0, 180])
    axis = geometry.rotation_axis()
    if modern:
        assert geometry.beam.axis_offset == 0.3  # already object-plane millimetres
        np.testing.assert_allclose(axis, [0, 0.5, np.sqrt(3) / 2], atol=1e-12)
    else:
        # Project two points on the physical rotation axis to the panel's ends:
        # the recorded CORs are detector-plane millimetres, not pixel indices.
        source = geometry.beam.source()
        for detector_height, expected_cor in [(1.0, 1.0), (-1.0, -0.5)]:
            z = detector_height / (2 * axis[2])
            point = np.asarray(geometry.pose_for_view(0)) @ [0, 0, z, 1]
            hit = source + 2 * (point[:3] - source)
            np.testing.assert_allclose(hit[[0, 2]], [expected_cor, detector_height], atol=1e-12)

    # Unknown vendor ROI coordinates may not silently turn into a centred grid.
    path.write_text(
        path.read_text().replace(
            "[CTPro]",
            "VoxelsX=6\nVoxelsY=6\nVoxelsZ=4\n"
            "VoxelSizeX=.25\nVoxelSizeY=.25\nVoxelSizeZ=.25\n"
            "OffsetX=1\nOffsetY=0\nOffsetZ=0\n[CTPro]",
        )
    )
    with pytest.raises(ValueError, match="grid="):
        tj.load_frames(path, darks=0)
    chosen = tj.load_frames(path, darks=0, grid=geometry.grid)
    assert chosen.geometry.grid == geometry.grid
    assert chosen.source.geometry_metadata["nikon_xtekct"]["volume"]["OffsetX"] == 1


@pytest.mark.parametrize("suffix", [".nxs", ".tif"])
def test_lazy_selections_compose_without_reading_storage(tmp_path, suffix):
    values = np.arange(5 * 6 * 9, dtype=np.uint16).reshape(5, 6, 9)
    path = tmp_path / f"frames{suffix}"
    angles = np.arange(5, dtype=np.float32)
    if suffix == ".nxs":
        _write_frames(path, values, [0] * 5, angles.tolist())
    else:
        tifffile.imwrite(path, values, photometric="minisblack")
    counts = tj.load_frames(
        path, angles=angles, detector=tiny_detector(nu=9, nv=6), flats=1000, darks=0
    ).counts
    moved = path.rename(tmp_path / "unavailable")
    selected = counts.frames_at(np.array([True, False, True, True, False]))
    selected = selected.window(slice(1, 6), slice(2, 9)).window(slice(1, 4), slice(1, 5))
    selected = selected.frames_at(slice(1, None))
    moved.rename(path)
    expected = values[[2, 3], 2:5, 3:7]
    np.testing.assert_array_equal(selected[-1], expected[-1])
    np.testing.assert_array_equal(np.asarray(selected, dtype=np.float32), expected)
    assert selected[:0].shape == (0, 3, 4)
    with pytest.raises(ValueError, match="without copying"):
        np.array(selected, copy=False)
    with pytest.raises(ValueError, match="non-empty block"):
        selected.window(slice(None, None, -1), slice(None))


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("tilted", [False, True])
def test_only_detector_derived_output_grids_follow_crop_and_bin(tmp_path, explicit, tilted):
    path = tmp_path / "raw.nxs"
    _write_frames(path, np.full((3, 6, 8), 500, np.uint16), [0, 0, 0], [0, 60, 120])
    with h5py.File(path, "a") as handle:
        if not explicit:
            del handle["entry"].attrs["grid_meta_json"]
        if tilted:
            geometry = handle.require_group("entry/geometry")
            geometry.attrs["type"] = "lamino"
            geometry.attrs["geometry_meta_json"] = '{"tilt_deg": 47}'
    frames = tj.load_frames(path, flats=1000, darks=0)
    scan = frames.cropped(slice(2, 6), slice(1, 7)).corrected(binning=2)
    assert scan.projections.shape == (3, 2, 3)
    np.testing.assert_allclose(scan.projections, np.log(2), atol=1e-7)
    if explicit:
        assert scan.grid == frames.geometry.grid
    elif tilted:
        assert scan.grid == frames.geometry.grid.resampled((4, 4, 3))
    else:
        assert scan.grid == tj.Grid(3, 3, 2, 2, 2, 2, vol_center=(0, 0, 1))
