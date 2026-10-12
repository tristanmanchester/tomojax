"""Terminal binning keeps correction order and original-detector statistics."""

from dataclasses import dataclass
import importlib
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import BeamHardening, Step, Stripes


@dataclass(frozen=True)
class KeepPattern(Step):
    rows: int
    cols: int
    domain: ClassVar[str] = "line integrals"
    selects_views: ClassVar[bool] = True

    def statistic(self, batch):
        assert batch.shape[1:] == (self.rows, self.cols)
        return jnp.mean(batch, axis=(1, 2))

    def select(self, stats, kept):
        keep = (kept % 3 != 1) & (stats >= 0)
        return keep, {"kept": int(keep.sum())}


def _frames(shape):
    views, rows, cols = shape
    v, u = np.meshgrid(np.linspace(-1, 1, rows), np.linspace(-1, 1, cols), indexing="ij")
    f0 = (12000 + 500 * u + 100 * v).astype(np.float32)
    f1 = (10000 + 800 * u - 300 * v).astype(np.float32)
    dark = (5 + 0.1 * v).astype(np.float32)
    t = (np.arange(views) + 0.5) / views
    line = np.random.default_rng(37).uniform(0.01, 2, shape).astype(np.float32)
    raw = ((1 - t[:, None, None]) * f0 + t[:, None, None] * f1) * np.exp(-line) + dark
    raw = raw.astype(np.float32)
    raw[0, -1, -1] = np.nan
    raw[1, -1, -2] = -1
    geometry = tj.ParallelGeometry(
        tj.Grid(8, 8, 3, 1, 1, 1),
        tj.Detector(cols, rows, 0.7, 0.9, (0.31, -0.23)),
        np.linspace(0, 180, views),
    )
    return tj.Frames(
        raw,
        geometry,
        flats=np.stack([f0, f0, f1, f1]),
        darks=np.stack([dark, dark]),
        flat_positions=np.array([0, 0, views, views]),
        name="terminal control",
    )


def _steps(order, rows, cols):
    choices = {
        "none": (),
        "per_view": (BeamHardening((1, 0.01)),),
        "rows": (Stripes(),),
        "head_rows": (BeamHardening((1, 0.01)), Stripes()),
        "select": (KeepPattern(rows, cols),),
        "head_select": (BeamHardening((1, 0.01)), KeepPattern(rows, cols)),
        "two_selects": (KeepPattern(rows, cols), KeepPattern(rows, cols)),
        "select_rows": (KeepPattern(rows, cols), Stripes()),
        "rows_select": (Stripes(), KeepPattern(rows, cols)),
        "rows_tail": (Stripes(), BeamHardening((1, 0.01))),
    }
    return choices[order]


@pytest.mark.parametrize("shape", [(13, 14, 19), (8, 12, 18), (7, 5, 7)])
@pytest.mark.parametrize("factor", [1, 2, 3])
@pytest.mark.parametrize(
    "order",
    [
        "none",
        "per_view",
        "rows",
        "head_rows",
        "select",
        "head_select",
        "two_selects",
        "select_rows",
        "rows_select",
        "rows_tail",
    ],  # fmt: skip
)
@pytest.mark.parametrize("storage", ["allocated", "memmap"])
def test_corrected_binning_is_ordered_correction_then_float32_mean(
    tmp_path, shape, factor, order, storage
):
    frames = _frames(shape)
    steps = _steps(order, shape[1], shape[2])
    expected = frames.corrected(*steps, batch_views=4).binned(factor)
    out = None
    if storage == "memmap":
        out = np.memmap(
            tmp_path / "corrected.dat",
            mode="w+",
            dtype=np.float32,
            shape=(shape[0], shape[1] // factor, shape[2] // factor),
        )
        out[:] = -123
    actual = frames.corrected(*steps, batch_views=4, binning=factor, out=out)
    np.testing.assert_allclose(actual.projections, expected.projections, rtol=3e-7, atol=3e-6)
    np.testing.assert_array_equal(actual.angles, expected.angles)
    assert actual.detector == expected.detector
    assert actual.grid == expected.grid
    assert actual.corrections == expected.corrections
    assert actual.name == expected.name
    assert actual.source is expected.source
    assert actual.projections.dtype == np.float32
    assert actual.projections.flags.writeable
    if out is not None:
        assert np.shares_memory(actual.projections, out)
        out.flush()
        reopened = np.memmap(
            tmp_path / "corrected.dat", mode="r", dtype=np.float32, shape=out.shape
        )
        np.testing.assert_array_equal(actual.projections, reopened[: len(actual.angles)])


@dataclass(frozen=True)
class PromotePixels(Step):
    whole_rows: bool = False
    domain: ClassVar[str] = "line integrals"

    def apply(self, batch):
        _, row, col = jnp.indices(batch.shape)
        extra = jnp.where((row % 2 == 1) & (col % 2 == 1), 3.0, 1.0)
        return jnp.ones(batch.shape, jnp.float64) + extra * 2**-24


@pytest.mark.parametrize("whole_rows", [False, True])
def test_float64_step_is_cast_before_terminal_mean(whole_rows):
    frames = _frames((8, 12, 18))
    with jax.enable_x64():
        expected = frames.corrected(PromotePixels(whole_rows)).binned(2)
        actual = frames.corrected(PromotePixels(whole_rows), binning=2)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    assert actual.corrections == expected.corrections


def test_reduced_output_alias_of_raw_counts_is_rejected():
    frames = _frames((8, 12, 18))
    with pytest.raises(ValueError, match="overlap"):
        frames.corrected(binning=2, out=frames.counts[:, :6, :9])


def test_binning_requires_reduced_output_shape_and_a_positive_factor():
    frames = _frames((8, 12, 18))
    with pytest.raises(ValueError, match="shape"):
        frames.corrected(binning=2, out=np.empty_like(frames.counts))
    with pytest.raises(ValueError, match="at least 1"):
        frames.corrected(binning=0)


def test_large_factor_does_not_raise_the_whole_row_memory_budget(monkeypatch):
    # Two rows fit the old two-buffer budget; a 3-row bin must use the safe fallback.
    engine = importlib.import_module("tomojax.corrections._engine")

    frames = _frames((8, 12, 18))
    expected = frames.corrected(Stripes()).binned(3)
    monkeypatch.setattr(engine, "_BATCH_BYTES", 8 * 8 * 18 * 2)
    actual = frames.corrected(Stripes(), binning=3)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    assert actual.corrections == expected.corrections


@dataclass(frozen=True)
class CountsPixels(Step):
    domain: ClassVar[str] = "counts"

    def apply(self, batch):
        assert batch.shape[1:] == (14, 19)
        gain = 0.8 + 0.02 * jnp.arange(batch.shape[2])
        return batch * gain


@dataclass(frozen=True)
class TransmissionPixels(Step):
    domain: ClassVar[str] = "transmission"

    def apply(self, batch):
        assert batch.shape[1:] == (14, 19)
        return batch + 0.03 * batch**2


@pytest.mark.parametrize("tail", [(), (Stripes(),), (Stripes(), BeamHardening((1, 0.01)))])
@pytest.mark.parametrize("factor", [2, 3])
def test_count_and_transmission_steps_run_at_original_coordinates(tail, factor):
    frames = _frames((13, 14, 19))
    steps = (CountsPixels(), TransmissionPixels(), BeamHardening((1, 0.02)), *tail)
    expected = frames.corrected(*steps).binned(factor)
    actual = frames.corrected(*steps, binning=factor)
    np.testing.assert_allclose(actual.projections, expected.projections, rtol=3e-7, atol=3e-6)
    assert actual.corrections == expected.corrections


@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize("batch_views", [None, 4])
@pytest.mark.parametrize("caller_storage", [False, True])
def test_terminal_selector_uses_full_pixels_but_only_stores_detector_means(
    monkeypatch, factor, batch_views, caller_storage
):
    frames = _frames((13, 14, 19))
    steps = (CountsPixels(), TransmissionPixels(), BeamHardening((1, 0.02)), KeepPattern(14, 19))
    expected = frames.corrected(*steps, batch_views=4).binned(factor)
    target = np.empty((13, 14 // factor, 19 // factor), np.float32) if caller_storage else None
    allocate, convert = np.empty, np.asarray
    downloads = []

    def bounded_empty(shape, *args, **kwargs):
        assert tuple(np.atleast_1d(shape)) != frames.counts.shape
        return allocate(shape, *args, **kwargs)

    def downloaded(value, *args, **kwargs):
        if isinstance(value, jax.Array) and value.ndim == 3:
            downloads.append(value.shape)
        return convert(value, *args, **kwargs)

    monkeypatch.setattr(np, "empty", bounded_empty)
    monkeypatch.setattr(np, "asarray", downloaded)
    actual = frames.corrected(*steps, binning=factor, batch_views=batch_views, out=target)
    assert downloads and all(shape[1:] == (14 // factor, 19 // factor) for shape in downloads)
    np.testing.assert_allclose(actual.projections, expected.projections, rtol=4e-7, atol=1e-6)
    np.testing.assert_array_equal(actual.angles, expected.angles)
    assert actual.corrections == expected.corrections
    assert actual.detector == expected.detector and actual.grid == expected.grid
    assert actual.projections.flags.writeable
    if target is not None:
        assert np.shares_memory(actual.projections, target)


@pytest.mark.parametrize("steps", [(), (Stripes(),)])
def test_terminal_device_downloads_are_binned_and_row_groups_stay_bounded(monkeypatch, steps):
    engine = importlib.import_module("tomojax.corrections._engine")
    frames = _frames((13, 14, 19))
    expected = frames.corrected(*steps, batch_views=4).binned(2)
    monkeypatch.setattr(engine, "_BATCH_BYTES", 8 * 13 * 19 * 6)
    downloads = []
    original = np.asarray

    def downloaded(array, *args, **kwargs):
        if isinstance(array, jax.Array) and array.ndim == 3:
            downloads.append(array.shape)
        return original(array, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", downloaded)
    actual = frames.corrected(*steps, binning=2, batch_views=4)
    np.testing.assert_allclose(actual.projections, expected.projections, rtol=3e-7, atol=3e-6)
    if steps:
        # Original-detector intermediates are needed, but not a full-size final download.
        assert [shape for shape in downloads if shape[0] == 13] == [
            (13, 3, 9),
            (13, 3, 9),
            (13, 1, 9),
        ]
    else:
        assert downloads and all(shape[0] <= 4 and shape[1:] == (7, 9) for shape in downloads)
