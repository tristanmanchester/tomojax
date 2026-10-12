"""Raw correction downloads preserve batching, optional dependencies and host layouts."""

from __future__ import annotations

import builtins
import sys

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import RejectViews


def _frames(*, large: bool = False) -> tj.Frames:
    rows, cols = (512, 1024) if large else (3, 7)
    counts = np.random.default_rng(118).integers(400, 700, (9, rows, cols), dtype=np.uint16)
    counts[4] = 5
    geometry = tj.ParallelGeometry(
        tj.Grid(7, 7, 3, 1.0, 1.0, 1.0),
        tj.Detector(cols, rows, 1.0, 1.0),
        np.arange(9) * 10.0,
    )
    return tj.Frames(counts, geometry, white_level=1000.0)


def _forbid_cupy_import(monkeypatch) -> None:
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        assert not name.startswith("cupy"), "this correction should not import CuPy"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)


def test_small_correction_does_not_import_optional_cupy(monkeypatch) -> None:
    _forbid_cupy_import(monkeypatch)
    frames = _frames()
    actual = frames.corrected(RejectViews(), batch_views=4)
    kept = np.delete(np.arange(9), 4)
    np.testing.assert_allclose(
        actual.projections, -np.log(frames.counts[kept] / np.float32(1000)), rtol=2e-6
    )
    np.testing.assert_array_equal(actual.angles, frames.geometry.angles[kept])
    assert actual.corrections[-1].found["rejected"] == [4]


def test_large_cpu_correction_does_not_import_optional_cupy(monkeypatch) -> None:
    if jnp.asarray(0).__dlpack_device__()[0] == 2:
        pytest.skip("checks the CPU download boundary")
    _forbid_cupy_import(monkeypatch)
    frames = _frames(large=True)
    actual = frames.corrected(batch_views=4)
    np.testing.assert_allclose(
        actual.projections, -np.log(frames.counts / np.float32(1000)), rtol=2e-6
    )


def test_large_batches_work_without_optional_cupy(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cupy", None)
    frames = _frames(large=True)
    actual = frames.corrected(RejectViews(), batch_views=4)
    expected = frames.corrected(batch_views=4).corrected(RejectViews(), batch_views=4)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    np.testing.assert_array_equal(actual.angles, expected.angles)
    assert actual.corrections == expected.corrections


def test_cuda_downloads_use_dlpack_and_preserve_partial_batches(monkeypatch) -> None:
    if jnp.asarray(0).__dlpack_device__()[0] != 2:
        pytest.skip("requires CUDA")
    cupy = pytest.importorskip("cupy")
    frames = _frames(large=True)
    with monkeypatch.context() as fallback:
        fallback.setitem(sys.modules, "cupy", None)
        expected = frames.corrected(RejectViews(), batch_views=4)
    original = cupy.from_dlpack
    transferred = []

    def checked(source):
        transferred.append(source.shape)
        return original(source)

    monkeypatch.setattr(cupy, "from_dlpack", checked)
    actual = frames.corrected(RejectViews(), batch_views=4)
    # The last one-view batch stays on the small-buffer JAX path.
    assert transferred == [(4, 512, 1024), (4, 512, 1024)]
    np.testing.assert_array_equal(actual.projections, expected.projections)
    np.testing.assert_array_equal(actual.angles, expected.angles)
    assert actual.corrections == expected.corrections


@pytest.mark.parametrize("layout", ["fortran", "reversed"])
def test_noncontiguous_correction_storage_uses_the_jax_download(monkeypatch, layout) -> None:
    frames = _frames(large=True)
    expected = frames.corrected(batch_views=4)
    storage = np.empty(frames.counts.shape, np.float32, order="F" if layout == "fortran" else "C")
    if layout == "reversed":
        storage = storage[..., ::-1]
    original = np.empty
    allocated = []

    def allocate(shape, dtype=float, *args, **kwargs):
        if shape == frames.counts.shape and np.dtype(dtype) == np.float32:
            allocated.append(shape)
            return storage
        return original(shape, dtype, *args, **kwargs)

    monkeypatch.setattr(np, "empty", allocate)
    _forbid_cupy_import(monkeypatch)
    actual = frames.corrected(batch_views=4)
    assert allocated == [frames.counts.shape]
    assert np.shares_memory(actual.projections, storage)
    np.testing.assert_array_equal(actual.projections, expected.projections)


def test_cuda_download_does_not_hide_unrelated_failures(monkeypatch) -> None:
    if jnp.asarray(0).__dlpack_device__()[0] != 2:
        pytest.skip("requires CUDA")
    cupy = pytest.importorskip("cupy")

    def fail(_source):
        raise RuntimeError("download sentinel")

    monkeypatch.setattr(cupy, "from_dlpack", fail)
    with pytest.raises(RuntimeError, match="download sentinel"):
        _frames(large=True).corrected(batch_views=4)


@pytest.mark.parametrize("large", [False, True])
def test_replicated_correction_uses_host_transfer_without_dlpack(monkeypatch, large) -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("checks replicated CPU correction")
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two CPU devices; set --xla_force_host_platform_device_count=2")
    _forbid_cupy_import(monkeypatch)
    frames = _frames(large=large)
    with jax.set_mesh(Mesh(np.asarray(devices), ("replica",))):
        actual = frames.corrected(RejectViews(), batch_views=4)
    kept = np.delete(np.arange(9), 4)
    np.testing.assert_allclose(
        actual.projections, -np.log(frames.counts[kept] / np.float32(1000)), rtol=2e-6
    )
    np.testing.assert_array_equal(actual.angles, frames.geometry.angles[kept])
