"""Public calibrated terminal binning keeps measured optical halos and owned statistics."""

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import BeamHardening, RadialDistortion, Stripes


class RowSource:
    _efficient_row_windows = True

    def __init__(self, data, reads=None, first=0):
        self.data = data
        self.shape, self.dtype = data.shape, data.dtype
        self.reads = [] if reads is None else reads
        self.first = first

    def window(self, rows, cols):
        return RowSource(self.data[:, rows, cols], self.reads, self.first + rows.start)

    def __getitem__(self, views):
        result = self.data[views]
        self.reads.append((self.first, result.shape))
        return result.copy()


def _frames(dtype):
    views, rows, cols = 13, 23, 31
    y, x = np.indices((rows, cols), dtype=np.float32)
    flat = 1600 + x * 3 + y
    dark = 3 + x * 0 + y * 0.1
    raw = dark + (flat - dark) * np.exp(
        -np.random.default_rng(7).uniform(0.1, 1, (views, rows, cols))
    )
    if np.issubdtype(np.dtype(dtype), np.floating):
        raw[0, -1, -1], raw[1, -1, -2] = np.nan, -1
    flat1 = flat.copy()
    flat1[4:6, 9] = np.nan
    # Endpoint views do not use the missing second calibration.
    return tj.Frames(
        raw.astype(dtype),
        tj.ParallelGeometry(
            tj.Grid(31, 31, 23, 0.8, 0.9, 1.1, (0.3, -0.4, 0.2)),
            tj.Detector(cols, rows, 0.8, 1.1, (0.37, -0.23)),
            np.linspace(0, 180, views),
        ),
        flats=np.stack([flat, flat1]),
        darks=dark[None],
        flat_positions=np.array([0.5, views - 0.5]),
        name="calibrated rows",
    )


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64, ">f4"])
@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize(
    "lens",
    [
        RadialDistortion(center=(11.4, 9.3), coefficients=(1,)),
        RadialDistortion(center=(-20.3, 34.7), coefficients=(0.93, 0.001)),
        RadialDistortion(center=(15.4, 10.3), coefficients=(1.2, -0.004)),
    ],
)
def test_calibrated_terminal_binning_matches_full_support_without_full_stack(
    tmp_path,
    monkeypatch,
    dtype,
    factor,
    lens,
):
    frames = _frames(dtype)
    steps = (BeamHardening((1, 0.02)), lens, Stripes(5), BeamHardening((1, -0.01)))
    expected = frames.corrected(*steps, batch_views=4).binned(factor)
    views, rows, cols = frames.counts.shape
    source = RowSource(frames.counts)
    frames = replace(frames, counts=source)
    # Own six rows at a time, so the lens needs several different measured halos.
    monkeypatch.setattr("tomojax.corrections._engine._BATCH_BYTES", 8 * views * cols * 6)
    out = np.memmap(
        tmp_path / "out", mode="w+", dtype="f4", shape=(views, rows // factor, cols // factor)
    )
    original = np.empty

    def bounded(shape, *args, **kwargs):
        assert tuple(np.atleast_1d(shape)) != (views, rows, cols)
        return original(shape, *args, **kwargs)

    monkeypatch.setattr(np, "empty", bounded)
    actual = frames.corrected(*steps, binning=factor, out=out)
    np.testing.assert_allclose(actual.projections, expected.projections, atol=1e-6, rtol=1e-6)
    assert actual.projections is out
    assert actual.corrections == expected.corrections
    assert actual.geometry == expected.geometry
    assert actual.name == expected.name
    assert len(source.reads) > 4 and all(shape[2] == cols for _, shape in source.reads)


@pytest.mark.parametrize("mode", ["phase", "selector", "counts", "explicit", "unknown"])
def test_lens_row_route_does_not_window_global_steps_or_unknown_readers(monkeypatch, mode):
    from tomojax.corrections import Paganin, RejectViews, Zingers

    frames = _frames(np.float32)
    lens = RadialDistortion(center=(12.3, 10.4), coefficients=(0.97, 0.001))
    steps = [lens, Stripes(5)]
    kwargs = {}
    if mode == "phase":
        steps.append(Paganin(delta_beta=10, distance=0.2, energy_kev=30, pixel_size=1e-5))
    elif mode == "selector":
        steps.append(RejectViews())
    elif mode == "counts":
        steps.append(Zingers())
    elif mode == "explicit":
        kwargs["batch_views"] = 4
    else:
        source = RowSource(frames.counts)
        source._efficient_row_windows = False
        frames = replace(frames, counts=source)

    def forbidden(*args, **kwargs):
        raise AssertionError("ineligible window path")

    monkeypatch.setattr("tomojax.corrections._engine.correct_lens_rows", forbidden)
    frames.corrected(*steps, binning=2, **kwargs)


@pytest.mark.parametrize(
    "lens",
    [
        RadialDistortion(center=(12.4, 11.3), coefficients=(0.55,)),
        RadialDistortion(center=(12.4, 11.3), coefficients=(1.35, -0.003)),
    ],
)
def test_raw_ceiling_records_own_acquired_rows_once_not_overlapping_optical_halos(
    monkeypatch, lens
):
    frames = _frames(np.uint16)
    counts = frames.counts.copy()
    # Include discarded output-edge pixels and camera rows a contracting lens never samples.
    counts[0, 0, 0] = counts[1, -1, -1] = counts[2, 8, 9] = 65535
    frames = replace(frames, counts=counts)
    expected = frames.corrected(lens, Stripes(5), batch_views=4).binned(3)
    source = RowSource(counts)
    frames = replace(frames, counts=source)
    monkeypatch.setattr("tomojax.corrections._engine._BATCH_BYTES", 8 * 13 * 31 * 6)
    actual = frames.corrected(lens, Stripes(5), binning=3)
    np.testing.assert_allclose(actual.projections, expected.projections, atol=1e-6, rtol=1e-6)
    assert actual.corrections == expected.corrections
    log = next(c for c in actual.corrections if c.name == "log")
    assert log.found["counts_at_dtype_ceiling"] == 3
    assert log.settings["counts_dtype_ceiling"] == 65535
    assert sum(shape[0] * shape[1] for _, shape in source.reads) > 13 * 23
