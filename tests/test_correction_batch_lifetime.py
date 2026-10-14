"""First-pass device batches and calibration fields are gone before host-tail work."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from typing import ClassVar
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import Step


@pytest.mark.parametrize("large", [False, True])
def test_first_pass_device_owners_are_released_before_view_selection(monkeypatch, large):
    rows, cols = (512, 1024) if large else (3, 7)
    downloaded = []
    calibrations = []
    array_type = type(jnp.asarray(0))
    original_copy = array_type.copy_to_host_async
    original_put = jax.device_put

    def record_copy(source):
        if source.ndim == 3 and source.shape[1:] == (rows, cols):
            downloaded.append(weakref.ref(source))
        return original_copy(source)

    def record_put(value, *args, **kwargs):
        source = original_put(value, *args, **kwargs)
        if (
            isinstance(value, np.ndarray)
            and value.dtype == np.float32
            and value.shape in ((1, rows, cols), (rows, cols))
        ):
            calibrations.append(weakref.ref(source))
        return source

    monkeypatch.setattr(array_type, "copy_to_host_async", record_copy)
    monkeypatch.setattr(jax, "device_put", record_put)
    if large and jnp.asarray(0).__dlpack_device__()[0] == 2:
        cupy = pytest.importorskip("cupy")
        original_dlpack = cupy.from_dlpack

        def record_dlpack(source):
            downloaded.append(weakref.ref(source))
            downloaded.append(weakref.ref(source.addressable_data(0)))
            return original_dlpack(source)

        monkeypatch.setattr(cupy, "from_dlpack", record_dlpack)

    @dataclass(frozen=True)
    class DrainProbe(Step):
        domain: ClassVar = "line integrals"
        selects_views: ClassVar = True

        def statistic(self, batch):
            return jnp.mean(batch, axis=(1, 2))

        def select(self, statistics, views):
            if large and jnp.asarray(0).__dlpack_device__()[0] == 2:
                # DLPack's child wrapper can survive until a later GC cycle,
                # but its device allocation must already have been released.
                assert all(ref() is None or ref().is_deleted() for ref in downloaded + calibrations)
            gc.collect()
            assert len(downloaded) >= 3
            assert len(calibrations) >= 2
            assert all(ref() is None for ref in downloaded + calibrations)
            return np.ones(len(views), bool), {}

    g = tj.ParallelGeometry(
        tj.Grid(7, 7, 3, 1, 1, 1), tj.Detector(cols, rows, 1, 1), np.arange(9) * 10
    )
    frames = tj.Frames(np.full((9, rows, cols), 500, np.uint16), g, white_level=1000)
    actual = frames.corrected(DrainProbe(), batch_views=4)
    np.testing.assert_allclose(actual.projections, np.float32(-np.log(0.5)), rtol=2e-6)
    np.testing.assert_array_equal(actual.angles, frames.geometry.angles)


def test_zero_view_correction_drain_has_no_pending_batch():
    g = tj.ParallelGeometry(tj.Grid(7, 7, 3, 1, 1, 1), tj.Detector(7, 3, 1, 1), np.empty(0))
    actual = tj.Frames(np.empty((0, 3, 7), np.uint16), g, white_level=1000).corrected()
    assert actual.projections.shape == (0, 3, 7)
    assert actual.angles.shape == (0,)


def test_batch_release_preserves_captured_arrays_and_input_alias_statistics() -> None:
    constant = jnp.asarray(
        np.random.default_rng(180).uniform(0.1, 0.9, (4, 512, 1024)).astype(np.float32)
    )
    expected = np.asarray(constant).copy()
    projections = np.concatenate([expected, expected, expected[:1]])

    @dataclass(frozen=True)
    class CapturedLine(Step):
        domain: ClassVar = "line integrals"

        def apply(self, batch):
            return constant[: batch.shape[0]]

    @dataclass(frozen=True)
    class IdentitySelect(Step):
        domain: ClassVar = "line integrals"
        selects_views: ClassVar = True

        def statistic(self, batch):
            return batch

        def select(self, statistics, views):
            np.testing.assert_array_equal(statistics, projections)
            return np.ones(len(views), bool), {}

    geometry = tj.ParallelGeometry(
        tj.Grid(7, 7, 3, 1, 1, 1), tj.Detector(1024, 512, 1, 1), np.arange(9) * 10
    )
    frames = tj.Frames(np.full((9, 512, 1024), 500, np.uint16), geometry, white_level=1000)
    for _ in range(2):
        actual = frames.corrected(CapturedLine(), IdentitySelect(), batch_views=4)
        np.testing.assert_array_equal(actual.projections, projections)
        np.testing.assert_array_equal(np.asarray(constant), expected)
        # A fresh device operation must work too, not just a cached host value.
        np.testing.assert_array_equal(
            np.asarray(constant + np.float32(1)), expected + np.float32(1)
        )
