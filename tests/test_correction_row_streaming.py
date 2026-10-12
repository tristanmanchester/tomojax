"""Raw row streaming preserves correction order, original statistics and bounded reads."""

from dataclasses import dataclass, replace
import importlib
from typing import ClassVar

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import BeamHardening, Step, Stripes
from tomojax.io.api import locate_frames


class RowSource:
    """A lazy source whose row-window capability is explicitly advertised."""

    def __init__(self, data, *, efficient=True, reads=None):
        self.data = data
        self.shape, self.dtype = data.shape, data.dtype
        self._efficient_row_windows = efficient
        self.reads = [] if reads is None else reads

    def window(self, rows, cols):
        return RowSource(
            self.data[:, rows, cols], efficient=self._efficient_row_windows, reads=self.reads
        )

    def __getitem__(self, views):
        part = self.data[views]
        self.reads.append(part.shape)
        return part.copy()


@dataclass(frozen=True)
class IndependentRows(Step):
    """A custom whole-row step requiring every original view and column."""

    gain: float = 0.1
    domain: ClassVar[str] = "line integrals"
    whole_rows: ClassVar[bool] = True

    def apply(self, batch):
        assert batch.shape[0] == 11 and batch.shape[2] == 19
        return batch + self.gain * jnp.mean(batch, axis=0, keepdims=True)


def _frames(dtype=np.float32):
    shape = (11, 14, 19)
    u = np.arange(shape[2], dtype=np.float32)[None]
    v = np.arange(shape[1], dtype=np.float32)[:, None]
    flat0, flat1, dark = 1000 + 7 * u + v, 1200 + 5 * u - v, 2 + v / 20 + u * 0
    where = (np.arange(shape[0]) + 0.5) / shape[0]
    flat = (1 - where[:, None, None]) * flat0 + where[:, None, None] * flat1
    p = np.random.default_rng(31).uniform(0.1, 1.0, shape)
    raw = (dark + (flat - dark) * np.exp(-p)).astype(dtype)
    raw[0, -1, -1], raw[1, -1, -2] = np.nan, -1
    geometry = tj.ParallelGeometry(
        tj.Grid(7, 8, 3, 0.7, 0.8, 0.9),
        tj.Detector(shape[2], shape[1], 0.7, 0.9, (0.31, -0.23)),
        np.linspace(0, 180, shape[0]),
    )
    return tj.Frames(
        raw, geometry, flats=np.stack([flat0, flat1]), darks=dark[None],
        flat_positions=np.array([0, shape[0]]), name="row stream control",
    )  # fmt: skip


@pytest.mark.parametrize("dtype", [np.float32, np.float64, ">f4"])
@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("storage", ["allocated", "memmap"])
@pytest.mark.parametrize(
    "steps", [(Stripes(),), (IndependentRows(), Stripes(5), IndependentRows(0.2))]
)
def test_row_streaming_matches_original_correction_chain_and_never_allocates_full_stack(
    tmp_path, monkeypatch, dtype, factor, lazy, storage, steps
):
    frames = _frames(dtype)
    expected = frames.corrected(*steps, batch_views=4).binned(factor)
    views, rows, cols = frames.counts.shape
    budget = 8 * views * cols * 6
    monkeypatch.setattr("tomojax.corrections._engine._BATCH_BYTES", budget)
    if lazy:
        frames = replace(frames, counts=RowSource(frames.counts))
    out = None
    if storage == "memmap":
        out = np.memmap(
            tmp_path / "corrected.dat", mode="w+", dtype="f4",
            shape=(views, rows // factor, cols // factor),
        )  # fmt: skip
    original_empty = np.empty

    def bounded_empty(shape, *args, **kwargs):
        assert tuple(np.atleast_1d(shape)) != (views, rows, cols)
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(np, "empty", bounded_empty)
    actual = frames.corrected(*steps, binning=factor, out=out)
    np.testing.assert_allclose(actual.projections, expected.projections, rtol=4e-7, atol=1e-6)
    np.testing.assert_array_equal(actual.angles, expected.angles)
    assert actual.corrections == expected.corrections
    assert actual.detector == expected.detector and actual.grid == expected.grid
    assert actual.name == expected.name and actual.source is expected.source
    if out is not None:
        assert actual.projections is out
    if lazy:
        assert sum(nr for _, nr, _ in frames.counts.reads) == rows
        assert all(nv == views and nc == cols for nv, _, nc in frames.counts.reads)
        itemsize = max(4, np.dtype(dtype).itemsize)
        assert all(2 * itemsize * views * nr * cols <= budget for _, nr, _ in frames.counts.reads)
    # Both events are in incomplete edge blocks for factor3, but still counted.
    log = next(record for record in actual.corrections if record.name == "log")
    assert log.found == {"nonpositive_transmission": 1, "nonfinite_set_to_zero": 1}


@dataclass(frozen=True)
class HalfwayPixels(Step):
    domain: ClassVar[str] = "line integrals"
    whole_rows: ClassVar[bool] = True

    def apply(self, batch):
        return jnp.full(batch.shape, 1.0 + 2**-24, jnp.float64)


@dataclass(frozen=True)
class CheckRounding(HalfwayPixels):
    def apply(self, batch):
        return jnp.where(batch > 1.0, 2.0, 1.0)


def test_each_whole_row_step_retains_its_float32_storage_barrier(monkeypatch):
    frames = _frames()
    monkeypatch.setattr("tomojax.corrections._engine._BATCH_BYTES", 8 * 11 * 19 * 6)
    with jax.enable_x64():
        expected = frames.corrected(HalfwayPixels(), CheckRounding(), batch_views=4).binned(3)
        actual = frames.corrected(HalfwayPixels(), CheckRounding(), binning=3)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    np.testing.assert_array_equal(actual.projections, np.ones_like(actual.projections))
    assert actual.corrections == expected.corrections


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_x64_raw_counts_preserve_float32_correction_storage(monkeypatch, dtype):
    frames = _frames(dtype)
    monkeypatch.setattr("tomojax.corrections._engine._BATCH_BYTES", 8 * 11 * 19 * 6)
    with jax.enable_x64():
        steps = (Stripes(5), HalfwayPixels(), CheckRounding())
        expected = frames.corrected(*steps, batch_views=4).binned(3)
        actual = frames.corrected(*steps, binning=3)
    assert actual.projections.dtype == np.float32
    np.testing.assert_array_equal(actual.projections, expected.projections)
    assert actual.corrections == expected.corrections


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_x64_integer_counts_normalize_before_compiling_row_steps(monkeypatch, dtype):
    frames = _frames()
    counts = np.maximum(np.nan_to_num(frames.counts, nan=0.0), 0).astype(dtype)
    frames = replace(frames, counts=counts)
    monkeypatch.setattr("tomojax.corrections._engine._BATCH_BYTES", 8 * 11 * 19 * 6)

    def fused(*args, **kwargs):
        pytest.fail("64-bit counts must not fuse normalization into the row reduction")

    monkeypatch.setattr("tomojax.corrections._engine._fused_rows_binned", fused)
    with jax.enable_x64():
        expected = frames.corrected(Stripes(5), batch_views=4).binned(3)
        actual = frames.corrected(Stripes(5), binning=3)
    assert actual.projections.dtype == np.float32
    np.testing.assert_allclose(actual.projections, expected.projections, rtol=4e-7, atol=1e-6)
    assert actual.corrections == expected.corrections


@pytest.mark.parametrize(
    "layout", ["contiguous", "single_view", "gzip", "shuffle", "checksum", "virtual", "multi_view"]
)
def test_discovered_hdf_layout_controls_raw_row_streaming(tmp_path, monkeypatch, layout):
    frames = _frames(">f4")
    expected = frames.corrected(Stripes(), batch_views=4).binned(3)
    path = tmp_path / "raw.h5"
    with h5py.File(path, "w") as file:
        if layout == "virtual":
            file.create_dataset("source", data=frames.counts)
            virtual = h5py.VirtualLayout(shape=frames.counts.shape, dtype=frames.counts.dtype)
            virtual[:] = h5py.VirtualSource(".", "/source", shape=frames.counts.shape)
            file.create_virtual_dataset("data", virtual)
        else:
            options = {
                "gzip": {"compression": "gzip"},
                "shuffle": {"shuffle": True},
                "checksum": {"fletcher32": True},
            }.get(layout, {})
            file.create_dataset(
                "data", data=frames.counts,
                chunks=None if layout == "contiguous" else (3 if layout == "multi_view" else 1, 14, 19),
                **options,
            )  # fmt: skip
    counts = locate_frames(str(path), data_path="/data").stack
    frames = replace(frames, counts=counts)
    windows = []
    original_window = type(counts).window

    def window(source, rows, cols):
        result = original_window(source, rows, cols)
        windows.append(result.shape)
        return result

    monkeypatch.setattr(type(counts), "window", window)
    monkeypatch.setattr("tomojax.corrections._engine._BATCH_BYTES", 8 * 11 * 19 * 6)
    actual = frames.corrected(Stripes(), binning=3)
    np.testing.assert_allclose(actual.projections, expected.projections, rtol=4e-7, atol=1e-6)
    assert actual.corrections == expected.corrections
    if layout in {"contiguous", "single_view"}:
        assert windows == [(11, 6, 19), (11, 6, 19), (11, 2, 19)]
    else:
        assert not windows


@dataclass(frozen=True)
class KeepAll(Step):
    domain: ClassVar[str] = "line integrals"
    selects_views: ClassVar[bool] = True

    def statistic(self, batch):
        return jnp.mean(batch, axis=(1, 2))

    def select(self, stats, kept):
        return np.ones(len(kept), bool), {"kept": len(stats)}


@dataclass(frozen=True)
class Transmission(Step):
    domain: ClassVar[str] = "transmission"

    def apply(self, batch):
        return batch + 0.01 * batch**2


@pytest.mark.parametrize(
    "route",
    ["explicit", "unbinned", "none", "unknown", "head", "tail", "selector", "domain", "large"],
)
def test_ineligible_chains_keep_the_existing_route(monkeypatch, route):
    engine = importlib.import_module("tomojax.corrections._engine")
    frames = _frames()
    steps, factor, batch = (Stripes(),), 3, None
    if route == "explicit":
        batch = 4
    elif route == "unbinned":
        factor = 1
    elif route == "none":
        steps = ()
    elif route == "unknown":
        frames = replace(frames, counts=RowSource(frames.counts, efficient=False))
    elif route in {"head", "tail"}:
        steps = (BeamHardening((1, 0.01)), Stripes())
        if route == "tail":
            steps = tuple(reversed(steps))
    elif route == "selector":
        steps = (Stripes(), KeepAll())
    elif route == "domain":
        steps = (Transmission(), Stripes())
    else:
        monkeypatch.setattr(engine, "_BATCH_BYTES", 8 * 11 * 19 * 2)
    expected = frames.corrected(*steps, batch_views=batch).binned(factor)

    def forbidden(*args, **kwargs):
        pytest.fail("ineligible raw rows were streamed")

    monkeypatch.setattr(engine, "_correct_rows_binned", forbidden)
    actual = frames.corrected(*steps, batch_views=batch, binning=factor)
    np.testing.assert_allclose(actual.projections, expected.projections, rtol=4e-7, atol=1e-6)
    assert actual.corrections == expected.corrections
