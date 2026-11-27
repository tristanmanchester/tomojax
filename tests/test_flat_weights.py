"""Flat sets around a scan are weighted by what its views show of each."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import tomojax as tj
from tomojax import progress
from tomojax.cli.main import main
from tomojax.corrections import Rings
from tomojax.errors import InvalidParameter
from tomojax.interactive import correction_preview

if TYPE_CHECKING:
    from pathlib import Path

ROWS, COLS, VIEWS, FLATS, DARK, PHOTONS = 48, 256, 96, 16, 100.0, 20000.0


def _texture(rng: np.random.Generator, scale: float) -> np.ndarray:
    pattern = gaussian_filter(rng.normal(size=(ROWS, COLS)), scale, mode="wrap")
    return pattern / pattern.std()


def _line_integrals() -> np.ndarray:
    """Blobs crossing the detector on sinusoidal tracks: positive, with moving fine structure."""
    rng = np.random.default_rng(3)
    angle = np.linspace(0, np.pi, VIEWS, endpoint=False)[:, None, None]
    row, col = np.arange(ROWS)[None, :, None], np.arange(COLS)[None, None, :]
    total = np.full((VIEWS, ROWS, COLS), 0.3)
    for _ in range(40):
        radius, phase, height = rng.uniform(5, 110), rng.uniform(0, 2 * np.pi), rng.uniform(0, ROWS)
        width, strength = rng.uniform(3, 10), rng.uniform(0.05, 0.25)
        at = COLS / 2 + radius * np.cos(angle + phase)
        total += strength * np.exp(-((col - at) ** 2 + (row - height) ** 2) / (2 * width**2))
    return total


def _beam(rng: np.random.Generator) -> np.ndarray:
    """An illumination with rows of different brightness and a per-pixel detector response."""
    rows = 0.8 + 0.2 * np.cos(2 * np.pi * np.arange(ROWS) / 37.0)[:, None]
    return (
        PHOTONS
        * rows
        * (1 + 0.1 * np.sin(np.arange(COLS) / 90.0))
        * (1 + 0.02 * rng.normal(size=(ROWS, COLS)))
    )


def _frames(
    case: str, *, seed: int = 0, middle_set: bool = False, flats: int = FLATS
) -> tuple[tj.Frames, np.ndarray, np.ndarray]:
    """Frames of a named case, their true line integrals and each view's true flat."""
    rng = np.random.default_rng(seed)
    truth = _line_integrals()
    beam = _beam(rng)
    col = np.arange(COLS)
    change = np.zeros((ROWS, COLS))  # the beam at position w of the scan is beam * (1 + w * change)
    shadows = [np.ones((ROWS, COLS)), np.ones((ROWS, COLS))]

    def wall(level: float) -> np.ndarray:
        return np.exp(-(level + 0.03 * _texture(rng, 3.0)))

    if case == "fading":  # the beam fades by 5% through the scan
        change -= 0.05
    elif case == "fading with specks":
        change -= 0.1
        shadows[0] = np.where(col < COLS // 3, np.exp(0.025 * _texture(rng, 1.5)), 1.0)
    elif case == "drifting pattern":  # fine structure grows through the scan
        change = 0.03 * _texture(rng, 2.0)
    elif case == "first set shadowed":
        shadows[0] = np.where(col < 0.6 * COLS, wall(0.2), 1.0)
    elif case == "edge":  # the first set shadowed everywhere, the second beyond an edge
        shadows[0] = wall(0.2)
        shadows[1] = np.where(col > 0.68 * COLS, wall(0.16), 1.0)
    elif case != "clean":
        raise ValueError(case)
    w = ((np.arange(VIEWS) + 0.5) / VIEWS)[:, None, None]
    flat = beam * (1 + w * change)
    counts = rng.poisson(flat * np.exp(-truth)).astype(np.float32) + DARK
    ends = [beam, beam * (1 + change)]
    sets = [
        rng.poisson(np.broadcast_to(e * s, (flats, ROWS, COLS)))
        for e, s in zip(ends, shadows, strict=True)
    ]
    positions = [0, VIEWS]
    if middle_set:  # a third set halfway, through a wall of its own
        sets.insert(1, rng.poisson(np.broadcast_to(beam * wall(0.2), (flats, ROWS, COLS))))
        positions.insert(1, VIEWS // 2)
    geometry = tj.ParallelGeometry(
        tj.Grid(COLS, COLS, ROWS, 1.0, 1.0, 1.0),
        tj.Detector(COLS, ROWS, 1.0, 1.0),
        np.linspace(0, 180, VIEWS, endpoint=False),
    )
    frames = tj.Frames(
        counts,
        geometry,
        flats=np.concatenate(sets).astype(np.float32) + DARK,
        darks=np.full((2, ROWS, COLS), DARK, np.float32),
        flat_positions=np.repeat(positions, flats),
    )
    return frames, truth, flat


def _by_position(frames: tj.Frames) -> np.ndarray:
    """Line integrals with the two flat sets interpolated linearly in the views' positions."""
    flats = np.asarray(frames.flats, np.float64) - DARK
    first, last = flats[: len(flats) // 2].mean(0), flats[len(flats) // 2 :].mean(0)
    w = ((np.arange(VIEWS) + 0.5) / VIEWS)[:, None, None]
    return -np.log((np.asarray(frames.counts, np.float64) - DARK) / ((1 - w) * first + w * last))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def _noise(truth: np.ndarray, flat: np.ndarray) -> float:
    """RMS error that photon noise alone leaves in the line integrals."""
    return float(np.sqrt(np.mean(1 / (flat * np.exp(-truth)))))


def test_a_flat_set_taken_through_an_obstruction_is_left_out() -> None:
    frames, truth, flat = _frames("first set shadowed")
    scan = frames.corrected()
    error = np.asarray(scan.projections, np.float64) - truth
    assert _rms(_by_position(frames) - truth) > 5 * _noise(truth, flat)
    assert _rms(error) < 1.25 * _noise(truth, flat)
    found = scan.corrections[0].found
    assert found["flat_weights_from_views"] > 0.5
    np.testing.assert_allclose(found["later_flat_weight"], [1.0, 1.0], atol=0.05)
    assert json.loads(json.dumps(found)) == found


@pytest.mark.parametrize("case", ["clean", "fading"])
def test_sets_that_differ_by_noise_or_level_stay_interpolated_by_position(case: str) -> None:
    frames, _, _ = _frames(case)
    scan = frames.corrected()
    np.testing.assert_allclose(np.asarray(scan.projections), _by_position(frames), atol=1e-5)
    assert scan.corrections[0].found == {"pixels_without_light": 0, "flat_weights_from_views": 0.0}


@pytest.mark.parametrize("unlit_margin", [0, 2 * COLS])
def test_structure_in_one_region_does_not_transfer_a_beam_wide_decay(unlit_margin: int) -> None:
    frames, truth, _ = _frames("fading with specks")
    if unlit_margin:
        cols = COLS + unlit_margin
        geometry = tj.ParallelGeometry(
            tj.Grid(cols, cols, ROWS, 1.0, 1.0, 1.0),
            tj.Detector(cols, ROWS, 1.0, 1.0),
            frames.geometry.angles,
        )
        padding = ((0, 0), (0, 0), (0, unlit_margin))
        frames = tj.Frames(
            np.pad(frames.counts, padding, constant_values=DARK),
            geometry,
            flats=np.pad(frames.flats, padding, constant_values=DARK),
            darks=np.pad(frames.darks, padding, constant_values=DARK),
            flat_positions=frames.flat_positions,
        )
    region = np.s_[:12, :, : COLS // 4]
    baseline = (
        np.asarray(frames.corrected(flat_weights="position").projections)[region] - truth[region]
    )
    error = np.asarray(frames.corrected().projections)[region] - truth[region]
    assert abs(error.mean()) < 0.005
    assert _rms(error) < 0.6 * _rms(baseline)


def test_sets_of_one_frame_have_no_measured_noise_and_stay_interpolated_by_position() -> None:
    frames, _, _ = _frames("first set shadowed", flats=1)
    scan = frames.corrected()
    np.testing.assert_allclose(np.asarray(scan.projections), _by_position(frames), atol=1e-5)
    assert "flat_weights_from_views" not in scan.corrections[0].found


def test_structure_that_drifts_through_the_scan_is_followed_through_it() -> None:
    frames, truth, flat = _frames("drifting pattern")
    scan = frames.corrected()
    found = scan.corrections[0].found
    # The views show the pattern arriving: none of the later set at first, all of it at the end.
    assert found["flat_weights_from_views"] > 0.9
    np.testing.assert_allclose(found["later_flat_weight"], [0.0, 1.0], atol=0.05)
    assert found["flat_structure_unmatched"] < 2e-3
    assert _rms(np.asarray(scan.projections, np.float64) - truth) < 1.1 * _noise(truth, flat)


def test_the_level_carries_across_an_edge_that_only_one_set_has() -> None:
    frames, truth, _ = _frames("edge")
    beyond = slice(int(0.75 * COLS), COLS)  # both sets are shadowed here, by 0.2 and 0.16
    error = np.asarray(frames.corrected().projections, np.float64) - truth
    blended = _by_position(frames) - truth
    assert abs(blended[..., beyond].mean()) > 0.15
    assert abs(error[..., beyond].mean()) < 0.04
    assert abs(error[..., : int(0.6 * COLS)].mean()) < 0.01


def test_each_interval_between_three_sets_is_weighted_on_its_own() -> None:
    frames, truth, flat = _frames("clean", middle_set=True)
    scan = frames.corrected(views_per_batch=40)  # batches that straddle the middle set
    error = np.asarray(scan.projections, np.float64) - truth
    assert _rms(error) < 1.2 * _noise(truth, flat)
    np.testing.assert_allclose(scan.corrections[0].found["later_flat_weight"], [0.5, 0.5], atol=0.1)


@pytest.mark.parametrize("middle_set", [False, True])
def test_rows_corrected_alone_and_in_slabs_equal_the_whole_detector(middle_set: bool) -> None:
    frames, _, _ = _frames("edge", middle_set=middle_set)
    whole = np.asarray(frames.corrected().projections)
    band = correction_preview(frames, (16, 24))
    np.testing.assert_allclose(band.after, whole[:, 16:24], atol=1e-5)
    with_rings = np.asarray(frames.corrected(Rings()).projections)  # corrected in slabs of rows
    staged = np.asarray(frames.corrected().corrected(Rings()).projections)
    np.testing.assert_allclose(with_rings, staged, atol=2e-4)


def test_views_sampled_in_several_batches_give_the_weights_of_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, _, _ = _frames("first set shadowed")
    whole = frames.corrected()
    # Real detectors fill a batch with a few views; here 5 views make one.
    monkeypatch.setattr("tomojax.corrections._flat_weights._VIEW_BYTES", 5 * 4 * ROWS * COLS)
    batched = frames.corrected()
    assert batched.corrections[0].found == whole.corrections[0].found
    # Batches of another size sum in another order: equal to float32 rounding.
    np.testing.assert_allclose(
        np.asarray(batched.projections), np.asarray(whole.projections), rtol=1e-5, atol=1e-6
    )


def _warnings(frames: tj.Frames, **options: str) -> tuple[tj.Scan, list[str]]:
    """The corrected scan and the warnings a watching client (the worker, MCP) receives."""
    events: list[progress.Event] = []
    with progress.watch(events.append):
        scan = frames.corrected(**options)  # pyright: ignore[reportArgumentType]
    notes = [e.message for e in events if isinstance(e, progress.Note) and e.level == "warning"]
    return scan, notes


def test_position_blends_the_flat_sets_by_position_alone_and_says_so() -> None:
    frames, _, _ = _frames("first set shadowed")
    scan, notes = _warnings(frames, flat_weights="position")
    np.testing.assert_allclose(np.asarray(scan.projections), _by_position(frames), atol=1e-5)
    record = scan.corrections[0]
    assert record.settings["flat_weights"] == "position"
    assert record.found == {"pixels_without_light": 0}
    assert notes == []
    assert frames.corrected().corrections[0].settings["flat_weights"] == "views"


def test_a_flat_set_the_views_reject_is_reported_with_what_to_do() -> None:
    frames, _, _ = _frames("first set shadowed")
    scan, notes = _warnings(frames)
    # The shadow covers 60% of the width, and the views hold the later set there throughout.
    assert 0.5 < scan.corrections[0].found["flat_moved_by_views"] < 0.7
    (note,) = notes
    assert "over 60% of the detector" in note
    held = re.search(r"later set at ([\d.]+) at the start and ([\d.]+) at the end", note)
    assert held is not None
    assert min(map(float, held.groups())) > 0.9
    assert "flats=" in note
    assert 'flat_weights="position"' in note


@pytest.mark.parametrize("case", ["clean", "fading", "drifting pattern"])
def test_flats_the_views_follow_raise_no_warning(case: str) -> None:
    scan, notes = _warnings(_frames(case)[0])
    assert notes == []
    assert scan.corrections[0].found.get("flat_moved_by_views", 0.0) < 0.01


def test_an_unknown_flat_weighting_is_refused() -> None:
    frames, _, _ = _frames("clean")
    with pytest.raises(InvalidParameter, match="flat_weights"):
        frames.corrected(flat_weights="time")  # pyright: ignore[reportArgumentType]


def _write_nxtomo(frames: tj.Frames, path: Path) -> None:
    """The frames as a raw NXtomo file: darks, the first flat set, the views, the second set."""
    flats = np.asarray(frames.flats)
    half = len(flats) // 2
    darks = np.asarray(frames.darks)
    data = np.concatenate([darks, flats[:half], np.asarray(frames.counts), flats[half:]])
    key = np.repeat([2, 1, 0, 1], [len(darks), half, VIEWS, half])
    angles = np.asarray(frames.geometry.angles)
    angle = np.concatenate([np.zeros(len(darks) + half), angles, np.full(half, angles[-1])])
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["definition"] = "NXtomo"
        detector = entry.create_group("instrument/detector")
        detector.create_dataset("data", data=data.astype(np.uint16))
        detector.create_dataset("image_key", data=key.astype(np.int32))
        rotation = entry.create_group("sample").create_dataset("rotation_angle", data=angle)
        rotation.attrs["units"] = "degree"


def test_the_command_line_chooses_the_blend_and_the_saved_scan_records_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frames, _, _ = _frames("first set shadowed")
    raw, out = tmp_path / "raw.nxs", tmp_path / "scan.nxs"
    _write_nxtomo(frames, raw)
    argv = ["correct", str(raw), "-o", str(out), "--progress", "none"]
    assert main([*argv, "--flat-weights", "position"]) == 0
    assert "do not follow" not in capsys.readouterr().err
    saved = tj.load(out)
    assert saved.corrections[0].settings["flat_weights"] == "position"
    np.testing.assert_allclose(
        np.asarray(saved.projections), _by_position(frames), atol=2e-5, rtol=1e-5
    )
    assert main([*argv, "--force"]) == 0
    assert "The views do not follow the flat sets' positions" in capsys.readouterr().err
    assert tj.load(out).corrections[0].settings["flat_weights"] == "views"
