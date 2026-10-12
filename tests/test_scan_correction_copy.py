"""Caller-owned correction output bounds device and lazy source reads as well as writes."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import BeamHardening, RejectViews, Stripes

if TYPE_CHECKING:
    from pathlib import Path


class LazyProjections:
    """Constant fields that support bounded indexing, never whole-stack conversion."""

    def __init__(self, shape: tuple[int, int, int], dtype: np.dtype) -> None:
        self.shape = shape
        self.dtype = dtype
        self.levels = np.linspace(0.1, 0.9, shape[0]).astype(dtype)
        self.reads: list[slice] = []

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("caller-owned output must not materialize a whole source stack")

    def __getitem__(self, views: slice) -> np.ndarray:
        self.reads.append(views)
        levels = self.levels[views]
        return np.broadcast_to(levels[:, None, None], (len(levels), *self.shape[1:]))


def geometry(shape: tuple[int, int, int]) -> tj.ParallelGeometry:
    views, rows, cols = shape
    return tj.ParallelGeometry(
        tj.Grid(7, 7, 4, 1, 1, 1), tj.Detector(cols, rows, 1, 1), np.arange(views) * 180 / views
    )


def test_automatic_copy_budget_counts_source_dtype_without_materializing_it() -> None:
    source = LazyProjections((65, 1024, 1024), np.dtype(np.float64))
    out = np.empty(source.shape, np.float32)
    actual = tj.Scan(source, geometry(source.shape)).corrected(out=out)
    assert actual.projections is out
    assert source.reads == [slice(0, 32), slice(32, 64), slice(64, 65)]
    for level, view in zip(source.levels, out, strict=True):
        np.testing.assert_array_equal(view, np.broadcast_to(np.float32(level), view.shape))
    np.testing.assert_array_equal(actual.angles, np.arange(65) * 180 / 65)
    assert actual.corrections == ()


@pytest.mark.parametrize("dtype", ["float32", ">f8"])
@pytest.mark.parametrize("selection", [False, True])
def test_explicit_copy_batches_preserve_lazy_pixels_storage_records_and_selection(
    tmp_path: Path, dtype: str, selection: bool
) -> None:
    source = LazyProjections((17, 4, 7), np.dtype(dtype))
    source.levels[[5, 13]] = [3.0, 0.01]
    scan = tj.Scan(source, geometry(source.shape))
    before = source.levels.copy()
    steps = [Stripes(3), BeamHardening((0.9, 0.1))]
    if selection:
        steps = [RejectViews(), *steps, RejectViews(), BeamHardening((0.8,))]
    expected = replace(scan, projections=np.asarray(source[:])).corrected(*steps, batch_views=3)
    source.reads.clear()
    out = np.memmap(tmp_path / "output.bin", mode="w+", dtype=np.float32, shape=source.shape)
    actual = scan.corrected(*steps, batch_views=3, out=out)
    assert source.reads == [slice(i, min(i + 3, 17)) for i in range(0, 17, 3)]
    np.testing.assert_array_equal(actual.projections, expected.projections)
    np.testing.assert_array_equal(actual.angles, expected.angles)
    np.testing.assert_array_equal(source.levels, before)
    assert actual.corrections == expected.corrections
    assert np.shares_memory(actual.projections, out)
    out.flush()
    saved = np.memmap(tmp_path / "output.bin", mode="r", dtype=np.float32, shape=source.shape)
    np.testing.assert_array_equal(saved[: len(actual.angles)], expected.projections)


@pytest.mark.parametrize("batch", [None, 3])
def test_device_scan_copy_preserves_whole_array_result(tmp_path: Path, batch: int | None) -> None:
    shape = (17, 4, 7)
    data = np.linspace(0.01, 2, np.prod(shape), dtype=np.float32).reshape(shape)
    expected_scan = tj.Scan(data, geometry(shape))
    actual_scan = replace(expected_scan, projections=jax.device_put(data))
    out = np.memmap(tmp_path / "output.bin", mode="w+", dtype=np.float32, shape=shape)
    steps = (BeamHardening((0.9, 0.1)),)
    expected = expected_scan.corrected(*steps, batch_views=batch)
    actual = actual_scan.corrected(*steps, batch_views=batch, out=out)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    np.testing.assert_array_equal(actual_scan.projections, data)
    assert actual.projections is out
    assert actual.corrections == expected.corrections
