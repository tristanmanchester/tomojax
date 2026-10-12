"""Caller-owned correction storage stays bounded, ordered and separate from its inputs."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import BeamHardening, RejectViews, Stripes

if TYPE_CHECKING:
    from pathlib import Path


def _frames() -> tj.Frames:
    shape = (17, 4, 7)
    levels = np.linspace(0.45, 0.55, shape[0], dtype=np.float32)
    levels[[5, 13]] = [0.03, 0.95]
    columns = np.array([1, 1, 0.99, 0.98, 1, 1, 1], np.float32)
    counts = np.broadcast_to(levels[:, None, None] * columns * 1000, shape).copy()
    geometry = tj.ParallelGeometry(
        tj.Grid(7, 7, 4, 1, 1, 1), tj.Detector(7, 4, 1, 1), np.arange(17) * 10.0
    )
    return tj.Frames(counts, geometry, white_level=1000.0)


@pytest.mark.parametrize("kind", ["raw", "scan"])
@pytest.mark.parametrize("storage", ["array", "memmap", "buffer_view", "strided"])
@pytest.mark.parametrize("selection", [False, True])
def test_correction_output_keeps_storage_geometry_records_and_input(
    tmp_path: Path, kind: str, storage: str, selection: bool
) -> None:
    frames = _frames()
    source = frames if kind == "raw" else frames.corrected()
    data = source.counts if kind == "raw" else source.projections
    original = data.copy()
    shape = data.shape
    if storage in {"memmap", "buffer_view"}:
        out = np.memmap(tmp_path / "corrected.bin", mode="w+", dtype=np.float32, shape=shape)
        if storage == "buffer_view":
            out = np.frombuffer(memoryview(out), dtype=out.dtype).reshape(shape)
    elif storage == "strided":
        out = np.empty((shape[0], shape[1], shape[2] * 2), np.float32)[:, :, ::2]
    else:
        out = np.empty(shape, np.float32)
    steps = [Stripes(3), BeamHardening((0.9, 0.1))]
    if selection:
        # Multiple selectors, with whole-row and pointwise work after compaction.
        steps = [RejectViews(), *steps, RejectViews(), BeamHardening((0.8,))]
    expected = source.corrected(*steps, batch_views=3)
    actual = source.corrected(*steps, batch_views=3, out=out)
    np.testing.assert_array_equal(data, original)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    np.testing.assert_array_equal(actual.angles, expected.angles)
    assert actual.corrections == expected.corrections
    assert np.shares_memory(actual.projections, out)
    if selection:
        assert len(actual.angles) == 15
        np.testing.assert_array_equal(actual.projections, out[:15])
    else:
        assert actual.projections is out
    if storage == "memmap":
        assert isinstance(out, np.memmap)
        out.flush()
        saved = np.memmap(tmp_path / "corrected.bin", mode="r", dtype=np.float32, shape=shape)
        np.testing.assert_array_equal(saved[: len(actual.angles)], expected.projections)


class _UnreadCounts:
    shape = (17, 4, 7)
    dtype = np.dtype(np.uint16)

    def __getitem__(self, key: object) -> np.ndarray:
        raise AssertionError("invalid output must fail before reading counts")


@pytest.mark.parametrize("invalid", ["dtype", "shape", "readonly", "notarray"])
def test_invalid_storage_is_rejected_before_reading_counts(invalid: str) -> None:
    frames = _frames()
    source = tj.Frames(_UnreadCounts(), frames.geometry, white_level=1000.0)
    if invalid == "notarray":
        out = [[[0.0]]]
    else:
        out = np.empty(
            (17, 4, 6) if invalid == "shape" else (17, 4, 7),
            np.float64 if invalid == "dtype" else np.float32,
        )
        if invalid == "readonly":
            out.flags.writeable = False
    with pytest.raises(ValueError, match="out must"):
        source.corrected(out=out)


@pytest.mark.parametrize("kind", ["raw", "scan"])
@pytest.mark.parametrize("mapping", [False, True])
def test_correction_refuses_overlapping_input_without_changing_it(
    tmp_path: Path, kind: str, mapping: bool
) -> None:
    frames = _frames()
    if mapping:
        path = tmp_path / "same.bin"
        counts = np.memmap(path, mode="w+", dtype=np.float32, shape=frames.counts.shape)
        counts[:] = frames.counts
        out = np.memmap(path, mode="r+", dtype=np.float32, shape=counts.shape)
        frames = tj.Frames(np.asarray(counts), frames.geometry, white_level=1000.0)
    else:
        out = frames.counts.view()
    source = frames if kind == "raw" else tj.Scan(frames.counts, frames.geometry)
    before = frames.counts.copy()
    with pytest.raises(ValueError, match="overlap|separate .*files"):
        source.corrected(out=out)
    np.testing.assert_array_equal(frames.counts, before)


def test_correction_refuses_calibration_aliases() -> None:
    frames = _frames()
    out = np.zeros(frames.counts.shape, np.float32)
    source = tj.Frames(frames.counts, frames.geometry, flats=out[:2], darks=out[2:4])
    with pytest.raises(ValueError, match="overlap"):
        source.corrected(out=out)
    np.testing.assert_array_equal(out, 0)


@pytest.mark.parametrize("role", ["raw", "scan", "flats", "darks"])
@pytest.mark.parametrize("wrapped", ["source", "output", "both"])
def test_buffer_views_preserve_mapped_input_and_output_ownership(
    tmp_path: Path, role: str, wrapped: str
) -> None:
    frames = _frames()
    path = tmp_path / "same.bin"
    owner = np.memmap(path, mode="w+", dtype=np.float32, shape=frames.counts.shape)
    owner[:] = frames.counts
    owner.flush()
    data = np.memmap(path, mode="r", dtype=owner.dtype, shape=owner.shape)
    out = np.memmap(path, mode="r+", dtype=owner.dtype, shape=owner.shape)
    if wrapped in {"source", "both"}:
        data = np.frombuffer(memoryview(data), dtype=data.dtype).reshape(data.shape)
    if wrapped in {"output", "both"}:
        out = np.frombuffer(memoryview(out), dtype=out.dtype).reshape(out.shape)
    # Separate mappings have different addresses, so pointer overlap cannot
    # detect that they refer to the same physical acquisition file.
    assert not np.may_share_memory(data, out)
    if role == "scan":
        source = tj.Scan(data, frames.geometry)
    elif role == "raw":
        source = tj.Frames(data, frames.geometry, white_level=1000.0)
    elif role == "flats":
        source = tj.Frames(frames.counts, frames.geometry, flats=data[:2])
    else:
        source = tj.Frames(frames.counts, frames.geometry, darks=data[:2])
    before = path.read_bytes()
    with pytest.raises(ValueError, match="separate .*files"):
        source.corrected(out=out)
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", ["raw", "scan", "calibration"])
def test_correction_refuses_cpu_jax_buffers_that_share_writable_output(kind: str) -> None:
    frames = _frames()
    shape = frames.counts.shape
    owner = np.empty(frames.counts.nbytes + 64, np.uint8)
    out = np.ndarray(shape, np.float32, buffer=owner, offset=(-owner.ctypes.data) % 64)
    out[:] = frames.counts
    # NumPy's CPU DLPack buffer stays shared; ordinary unaligned device_put may copy.
    data = jax.dlpack.from_dlpack(out[:2] if kind == "calibration" else out)
    assert np.shares_memory(np.asarray(data), out)
    if kind == "scan":
        source = tj.Scan(data, frames.geometry)
    elif kind == "calibration":
        source = tj.Frames(frames.counts, frames.geometry, flats=data)
    else:
        source = tj.Frames(data, frames.geometry, white_level=1000.0)
    before = out.copy()
    with pytest.raises(ValueError, match="overlap"):
        source.corrected(out=out)
    np.testing.assert_array_equal(out, before)


@pytest.mark.parametrize("same_file", [False, True])
def test_two_unnamed_maps_have_a_clear_separate_file_error(same_file: bool) -> None:
    frames = _frames()
    with tempfile.TemporaryFile() as first, tempfile.TemporaryFile() as second:
        counts = np.memmap(first, mode="w+", dtype=np.float32, shape=frames.counts.shape)
        counts[:] = frames.counts
        out = np.memmap(
            first if same_file else second,
            mode="r+" if same_file else "w+",
            dtype=np.float32,
            shape=counts.shape,
        )
        if counts.filename is not None or out.filename is not None:
            pytest.skip("temporary mappings have names on this platform")
        source = tj.Frames(counts, frames.geometry, white_level=1000.0)
        before = counts.copy()
        with pytest.raises(ValueError, match="separate named files"):
            source.corrected(out=out)
        np.testing.assert_array_equal(counts, before)


def test_an_unnamed_output_map_with_ordinary_input_is_supported() -> None:
    frames = _frames()
    with tempfile.TemporaryFile() as file:
        out = np.memmap(file, mode="w+", dtype=np.float32, shape=frames.counts.shape)
        actual = frames.corrected(out=out)
        assert actual.projections is out
        np.testing.assert_array_equal(actual.projections, frames.corrected().projections)
