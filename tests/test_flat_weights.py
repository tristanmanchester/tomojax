"""Flat sets around a scan are weighted by what its views show of each."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import tomojax as tj
from tomojax.corrections import Rings
from tomojax.interactive import correction_preview

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
    np.testing.assert_allclose(np.asarray(batched.projections), np.asarray(whole.projections))
