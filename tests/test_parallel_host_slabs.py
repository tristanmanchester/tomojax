"""Global-coordinate host slabs agree with the resident matched Joseph operator."""

from importlib import import_module

import numpy as np
import pytest

import tomojax as tj
from tomojax.recon import CGLSConfig, cgls

slabs = import_module("tomojax.recon._parallel_slabs")
arrays = import_module("tomojax.recon._host_arrays")


@pytest.mark.gpu
@pytest.mark.parametrize("tilt", [0, 35, 75])
def test_parallel_slab_sum_and_transpose_keep_global_weights(monkeypatch, tilt):
    import cupy as cp

    from tomojax.core.joseph_cuda import JosephCUDA

    grid = tj.Grid(9, 7, 11, 1.1, 0.8, 1.3, vol_center=(0.3, -0.7, 0.9))
    detector = tj.Detector(17, 15, 1.2, 1.0, (0.2, -0.4))
    geometry = tj.LaminographyGeometry(grid, detector, np.arange(0, 180, 23), tilt)
    poses = np.asarray([geometry.pose_for_view(i) for i in range(8)], np.float32)
    rng = np.random.default_rng(913)
    x = rng.normal(size=(11, 7, 9)).astype(np.float32)
    y = rng.normal(size=(8, 15, 17)).astype(np.float32)
    monkeypatch.setattr(slabs, "slab_depth", lambda *args: 4)
    problem = slabs.parallel_problem(geometry, grid, detector, y, poses=poses, batch=3)
    source, target = problem.volume(x), problem.volume(np.empty_like(x))
    projection = np.empty_like(y)

    def finish(batch, first, stop, rows):
        projection[batch.views, first:stop] = cp.asnumpy(rows[0])
        rows[0][...] = cp.asarray(y[batch.views, first:stop])

    problem.sweep.run(
        np.arange(8),
        source=lambda k: (source.get(*problem.plan.planes(k)),),
        finish=finish,
        sink=lambda k, values, _: target.put(*problem.plan.planes(k), values[0]),
    )
    ops = JosephCUDA(poses, grid, detector, views_per_batch=3)
    expected = cp.asnumpy(ops.forward(cp.asarray(x)))
    transpose = cp.asnumpy(ops.adjoint(cp.asarray(y)))
    np.testing.assert_allclose(projection, expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(target.array, transpose)
    lhs = np.vdot(projection.astype(np.float64), y)
    rhs = np.vdot(x.astype(np.float64), target.array)
    assert abs(lhs - rhs) < 2e-6 * np.linalg.norm(projection) * np.linalg.norm(y)


@pytest.mark.gpu
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("data_term", ["absolute", "per_view_contrast"])
def test_parallel_host_cgls_matches_resident_with_penalties(monkeypatch, initial, data_term):
    grid = tj.Grid(9, 7, 11, 1.1, 0.8, 1.3)
    detector = tj.Detector(17, 15, 1.2, 1.0)
    geometry = tj.LaminographyGeometry(grid, detector, np.arange(0, 180, 23), 35)
    rng = np.random.default_rng(914)
    data = rng.normal(size=(8, 15, 17)).astype(np.float32)
    start = rng.normal(size=(11, 7, 9)).astype(np.float32) if initial else None
    cfg = CGLSConfig(
        data_term=data_term,
        iterations=5,
        rtol=0,
        damping=0.2,
        gradient_damping=0.1,
        views_per_batch=3,
        backend="cuda",
        stream_projections=True,
    )
    expected, _ = cgls(geometry, grid, detector, data, init_x=start, config=cfg)
    monkeypatch.setattr(slabs, "slab_depth", lambda *args: 4)
    out = np.empty((11, 7, 9), np.float32)
    result, info = cgls(geometry, grid, detector, data, init_x=start, config=cfg, out=out)
    assert result is out and info["host_slabs"]
    relative = np.linalg.norm(result - np.asarray(expected)) / np.linalg.norm(expected)
    assert relative < 2e-5


def test_automatic_disk_output_is_owned_and_explicit_output_is_preserved(monkeypatch, tmp_path):
    from tomojax.backends import api as backends

    monkeypatch.setattr(backends, "host_available_memory_bytes", lambda: 1)
    monkeypatch.setattr(arrays.tempfile, "gettempdir", lambda: str(tmp_path))
    data = np.ones((2, 3, 4), np.float32)
    mapped = arrays.automatic_host_output(data, None, (5, 4, 3), 4, "test")
    assert isinstance(mapped, np.memmap)
    assert not list(tmp_path.iterdir())
    mapped[2] = 7
    np.testing.assert_array_equal(mapped[2], 7)
    out = np.empty_like(mapped, subok=False)
    assert arrays.automatic_host_output(data, out, out.shape, 4, "test") is out


def test_owned_unlinked_output_can_be_reused_with_mapped_input(monkeypatch, tmp_path):
    import tomojax.backends.api as backends

    monkeypatch.setattr(backends, "host_available_memory_bytes", lambda: 1)
    monkeypatch.setattr(arrays.tempfile, "gettempdir", lambda: str(tmp_path))
    data = np.memmap(tmp_path / "data", np.float32, "w+", shape=(2, 3, 4))
    out = arrays.automatic_host_output(data, None, (5, 4, 3), 4, "test")
    # Explicit outputs neither query memory nor initialize a host device.
    monkeypatch.setattr(backends, "host_available_memory_bytes", lambda: pytest.fail("probed"))
    assert arrays.automatic_host_output(data, out, out.shape, 4, "test") is out
    view = out.view()
    assert arrays.automatic_host_output(data, view, view.shape, 4, "test") is view
    import os

    from tomojax.core.host_volume import slab_file

    descriptor = slab_file(view)
    assert descriptor is not None
    os.close(descriptor[0])
