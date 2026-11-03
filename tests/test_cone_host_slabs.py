"""Cone volumes in host slabs: the slab sum is the matched projector, solvers match resident ones."""

from dataclasses import replace
from importlib import import_module
import os
import weakref

import numpy as np
import pytest

import tomojax as tj
from tomojax.core.cone import cone_backproject, cone_model, cone_project, use_cuda_cone
from tomojax.geometry import (
    ConeBeam,
    ConeGeometry,
    ConeSegments,
    Detector,
    Grid,
    stack_view_poses,
)
from tomojax.recon import CGLSConfig, SIRTConfig, cgls, sirt

_cone_slabs = import_module("tomojax.recon._cone_slabs")

GRID = Grid(5, 4, 9, 1.0, 1.1, 0.9)
DETECTOR = Detector(8, 11, 1.6, 1.5, (0.3, -0.4))
ANGLES = np.linspace(0, 360, 7, endpoint=False) + 3


def _geometry(kind):
    if kind == "cone":
        return ConeGeometry(GRID, DETECTOR, ANGLES, ConeBeam(12, 26))
    return ConeSegments(
        (
            ConeGeometry(GRID, DETECTOR, ANGLES[:3], ConeBeam(12, 26)),
            ConeGeometry(GRID, replace(DETECTOR, center=(-0.2, 0.4)), ANGLES[3:], ConeBeam(15, 27)),
        )
    )


def _data(geometry, seed=1):
    truth = np.random.default_rng(seed).random((9, 4, 5), np.float32)
    return np.asarray(tj.project(geometry, truth), np.float32)


def _relative(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.fixture
def host(monkeypatch):
    """Force host slabs of ``depth`` planes for the cone solvers."""

    def force(depth):
        monkeypatch.setattr(_cone_slabs, "slab_depth", lambda *_, **__: depth)
        monkeypatch.setattr(_cone_slabs, "ROW_BAND", 3)  # rows finished in bands of three
        for name in ("tomojax.recon._cgls_host", "tomojax.recon.sirt"):
            monkeypatch.setattr(import_module(name), "host_volumes", lambda *_: True)

    return force


@pytest.mark.parametrize("kind", ["cone", "segments"])
@pytest.mark.parametrize("depth", [1, 4])
@pytest.mark.parametrize("prepare", [False, True])
def test_slab_sweep_is_the_matched_normal_operator(kind, depth, host, prepare):
    host(depth)
    geometry = _geometry(kind)
    model, backend = cone_model(geometry), "cuda" if use_cuda_cone() else "jax"
    poses = stack_view_poses(geometry, len(ANGLES))
    problem = _cone_slabs.cone_problem(
        geometry, GRID, DETECTOR, _data(geometry), model=model, backend=backend, poses=poses,
        batch=3, channels=1,
    )  # fmt: skip
    direction = np.random.default_rng(2).random((9, 4, 5), np.float32)
    source = problem.volume(np.ascontiguousarray(direction))
    applied = np.zeros_like(direction, order="C")
    target, energy, prepared = problem.volume(applied), [], []
    if prepare:
        problem.sweep.plan = replace(problem.plan, ring=DETECTOR.nv)

    def prepare_rows(rows):
        prepared.append(problem.engine.norm2(rows[0]))
        rows[0][...] *= 0.5

    def finish(batch, first, stop, rows):
        assert not prepare or prepared
        energy.append(float(np.sum(np.square(rows[0], dtype=np.float64))))

    problem.sweep.run(
        np.arange(len(ANGLES)),
        source=lambda k: (source.get(*problem.plan.planes(k)),),
        finish=finish,
        sink=lambda k, accumulated, _: target.put(*problem.plan.planes(k), accumulated[0]),
        keep=not prepare,
        prepare=prepare_rows if prepare else None,
    )
    coefficients = model.coefficients(poses, GRID)
    projected = cone_project(direction, coefficients, GRID, DETECTOR, backend=backend)
    if prepare:
        projected *= 0.5
    expected = cone_backproject(projected, coefficients, GRID, DETECTOR, backend=backend)
    assert sum(energy) == pytest.approx(
        float(np.sum(np.square(projected, dtype=np.float64))), rel=1e-6
    )
    assert _relative(applied, expected) < 1e-6
    if prepare:
        assert prepared == pytest.approx([4 * sum(energy)], rel=1e-6)


@pytest.mark.parametrize("depth", [3, 4])
def test_host_cgls_matches_streamed_normal_equations(host, depth):
    geometry = _geometry("cone")
    data = _data(geometry)
    start = np.random.default_rng(3).random((9, 4, 5), np.float32)
    cases = [
        (CGLSConfig(iterations=6, rtol=0), None),
        (CGLSConfig(iterations=4, rtol=0, damping=0.7), start),
        (CGLSConfig(iterations=60, rtol=1e-4, damping=0.3), None),
        (CGLSConfig(iterations=5, rtol=0, gradient_damping=0.5, damping=0.3), start),
    ]
    expected = [
        cgls(
            geometry,
            GRID,
            DETECTOR,
            data,
            init_x=initial,
            config=replace(config, stream_projections=True),
        )  # fmt: skip
        for config, initial in cases
    ]
    host(depth)
    for (config, initial), (volume, reference) in zip(cases, expected, strict=True):
        result, info = cgls(geometry, GRID, DETECTOR, data, init_x=initial, config=config)
        assert isinstance(result, np.ndarray) and info["host_slabs"]
        assert _relative(result, volume) < 5e-5
        for key in ("effective_iterations", "termination", "residual_recomputations"):
            assert info[key] == reference[key]
        # Near the tolerance the norm is FP32 rounding of A^T (y - A x).
        floor = 1e-5 * reference["initial_normal_residual_norm"]
        assert info["normal_residual_norm"] == pytest.approx(
            reference["normal_residual_norm"], rel=1e-4, abs=floor
        )


@pytest.mark.parametrize(
    "config",
    [
        SIRTConfig(iterations=3),
        SIRTConfig(iterations=3, subsets=3, lower_bound=0.0, upper_bound=0.9),
        SIRTConfig(iterations=3, support=np.arange(180).reshape(9, 4, 5) % 7 > 0),
        SIRTConfig(iterations=4, acceleration=True, relaxation=0.9),
    ],
)
@pytest.mark.parametrize("depth", [2, 9])  # on CUDA, whole-volume slabs keep it on the device
def test_host_sirt_matches_resident_updates(config, depth, host):
    geometry = _geometry("segments")
    data = _data(geometry)
    expected, _ = sirt(geometry, GRID, DETECTOR, data, config=config)
    host(depth)
    volume, info = sirt(geometry, GRID, DETECTOR, data, config=config)
    assert info["slices_per_batch"] == depth
    assert _relative(volume, expected) < 1e-6


def test_public_reconstruction_returns_host_volume(host):
    host(4)
    scan = tj.Scan(_data(_geometry("cone")), _geometry("cone"))
    result = tj.reconstruct(scan, "cgls", iterations=2)
    assert isinstance(result.volume, np.ndarray) and result.volume.flags.c_contiguous
    assert result.info["host_slabs"] and result.info["effective_iterations"] == 2


def test_memmapped_out_holds_iterate_and_work_volumes_are_unnamed(tmp_path, host):
    geometry = _geometry("cone")
    scan = tj.Scan(_data(geometry), geometry)
    expected = tj.reconstruct(scan, "cgls", iterations=3).volume
    host(4)
    out = np.lib.format.open_memmap(
        tmp_path / "x.npy", "w+", np.float32, (9, 4, 5), fortran_order=False
    )
    result = tj.reconstruct(scan, "cgls", iterations=3, out=out)
    assert result.volume is out and [p.name for p in tmp_path.iterdir()] == ["x.npy"]
    assert _relative(out, expected) < 2e-5
    warm = tj.reconstruct(scan, "sirt", iterations=2, warm_start=True, out=out)
    assert (
        warm.volume is out
        and _relative(out, tj.reconstruct(scan, "sirt", iterations=2, warm_start=True).volume)
        < 1e-4
    )
    out[...] = np.nan  # a zero start ignores what out held
    cold = tj.reconstruct(scan, "sirt", iterations=2, out=out).volume
    assert _relative(cold, tj.reconstruct(scan, "sirt", iterations=2).volume) < 1e-6
    with pytest.raises(ValueError, match="cone-beam"):
        tj.reconstruct(
            tj.Scan(
                scan.projections[:, :1],
                tj.ParallelGeometry(replace(GRID, nz=1), replace(DETECTOR, nv=1), ANGLES),
            ),
            "sirt",
            out=out,
        )


def test_host_volume_storage_fills_checks_and_unlinks_work_files(tmp_path):
    host_volume = import_module("tomojax.core.host_volume")
    named = np.lib.format.open_memmap(
        tmp_path / "v.npy", "w+", np.float32, (70, 2, 3), fortran_order=False
    )
    work = host_volume.host_volume((70, 2, 3), host_volume.scratch_directory(named))
    host_volume.fill(work, 1.5)
    host_volume.fill(named, work)
    assert isinstance(work, np.memmap) and [p.name for p in tmp_path.iterdir()] == ["v.npy"]
    np.testing.assert_array_equal(named, 1.5)
    block = np.arange(12, dtype=np.float32).reshape(2, 2, 3)  # planes 5 and 6, (z, y, x)
    for array in (named, work):  # file reads and writes meet the mapped volume
        descriptor, first = host_volume.slab_file(array)
        host_volume.write_at(descriptor, block, first + 5 * 24, drop=True)
        back = np.empty_like(block)
        host_volume.read_at(descriptor, back, first + 5 * 24, drop=True)
        os.close(descriptor)
        np.testing.assert_array_equal(np.asarray(array[5:7, :, :]), back)
        np.testing.assert_array_equal(back, block)
    data = np.ones((20, 2, 2), np.float32)
    host_volume.check_rows(data, "test", nonnegative=True)
    data[17, 1, 0] = -1
    with pytest.raises(ValueError, match="test: projections must be finite and nonnegative"):
        host_volume.check_rows(data, "test", nonnegative=True)
    data[17, 1, 0] = np.inf
    with pytest.raises(ValueError, match="finite in FP32"):
        host_volume.check_rows(data, "test")


@pytest.mark.parametrize("layout", ["C", "F", "mapped"])
def test_host_volume_transfers_leading_slabs_and_short_tail(tmp_path, layout):
    shape = (9, 4, 5)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    output = (
        np.memmap(tmp_path / "volume.bin", mode="w+", dtype=np.float32, shape=shape)
        if layout == "mapped"
        else np.empty(shape, np.float32, order=layout)
    )
    output[:] = values
    engine = _cone_slabs.engine_for(GRID, DETECTOR, "cuda" if use_cuda_cone() else "jax")
    volume = _cone_slabs.HostVolume(output, engine, depth=4)
    try:
        for first in range(0, shape[0], 4):
            stop = min(first + 4, shape[0])
            volume.prefetch(first, stop)
            block = volume.get(first, stop)
            actual = np.empty((stop - first, *shape[1:]), np.float32)
            engine.download(block[: stop - first], actual)
            np.testing.assert_array_equal(actual, values[first:stop])
            volume.put(first, stop, block + 1)
    finally:
        volume.close()
    np.testing.assert_array_equal(output, values + 1)


@pytest.mark.gpu
def test_upload_keeps_source_until_copy_completion():
    engine = _cone_slabs.CudaEngine(GRID, DETECTOR)
    source = np.arange(8, dtype=np.float32)
    reference = weakref.ref(source)
    uploaded = engine.upload(source)
    del source
    assert reference() is not None
    engine.xp.cuda.get_current_stream().synchronize()
    engine.upload(np.zeros(1, np.float32))
    assert reference() is None
    np.testing.assert_array_equal(uploaded.get(), np.arange(8, dtype=np.float32))


def test_row_reader_reads_each_row_once_per_sweep(host):
    host(2)
    geometry = _geometry("cone")
    data = _data(geometry)
    config = SIRTConfig(iterations=2, subsets=2)
    expected, _ = sirt(geometry, GRID, DETECTOR, data, config=config)
    reads = []

    def read(first, stop):
        reads.append((first, stop))
        return data[:, first:stop] * 1.0

    rows = _cone_slabs.RowReader(read, data.shape)
    volume, info = sirt(geometry, GRID, DETECTOR, rows, config=config)
    assert info["slices_per_batch"] and _relative(volume, expected) < 1e-6
    # four sweeps read each of the four bands at most once, and one band ahead
    assert all(stop == min(first + 3, DETECTOR.nv) and first % 3 == 0 for first, stop in reads)
    assert len(reads) <= 4 * 4 + 1


def test_cgls_lazy_rows_preserve_views_and_close_on_failure(monkeypatch, tmp_path):
    host = import_module("tomojax.recon._cgls_host")
    geometry = _geometry("cone")
    data = _data(geometry)
    cfg = CGLSConfig(iterations=2, backend="jax")
    output = np.memmap(tmp_path / "out", np.float32, "w+", shape=(9, 4, 5))
    original_gradient = host._HostCGLS.gradient

    def gradient(solver):
        zero = solver.zero
        result = original_gradient(solver)
        if zero:
            assert not solver.X._reads  # zero-start source is never read
        return result

    monkeypatch.setattr(host._HostCGLS, "gradient", gradient)
    expected, _ = cgls(geometry, GRID, DETECTOR, data, config=cfg, out=output)
    expected = expected.copy()
    rows = _cone_slabs.RowReader(lambda first, stop: data[:, first:stop], data.shape)
    result, _ = cgls(geometry, GRID, DETECTOR, rows, config=cfg, out=output)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert rows._worker is None
    closed = []
    original = _cone_slabs.HostVolume.close

    def close(volume):
        original(volume)
        closed.append(volume._file)

    monkeypatch.setattr(_cone_slabs.HostVolume, "close", close)
    monkeypatch.setattr(host._HostCGLS, "gradient", lambda _: (_ for _ in ()).throw(RuntimeError()))
    with pytest.raises(RuntimeError):
        cgls(geometry, GRID, DETECTOR, rows, config=cfg, out=output)
    assert closed == [None] * 4


def test_unpenalized_cgls_sinks_do_not_read_unused_volumes(monkeypatch, host):
    host(4)
    module = import_module("tomojax.recon._cgls_host")

    def run(solver):
        solver.keep, solver.zero = False, False
        solver.gradient()
        solver.bounded()
        assert not solver.X._held
        monkeypatch.setattr(solver.D, "get", lambda *_: pytest.fail("unused direction read"))
        solver.normal(None)
        return {}

    monkeypatch.setattr(module._HostCGLS, "run", run)
    geometry = _geometry("cone")
    cgls(geometry, GRID, DETECTOR, _data(geometry), config=CGLSConfig(iterations=1))


@pytest.mark.parametrize("depth", [2, 9])
@pytest.mark.parametrize("reader", [False, True])
def test_whole_view_statistics_include_zero_ray_rows(host, depth, reader):
    from tomojax.recon._view_statistics import ViewStatistics  # check-public-imports: allow-private

    host(depth)
    detector = replace(DETECTOR, nv=31)
    geometry = ConeGeometry(GRID, detector, ANGLES, ConeBeam(12, 26))
    model, backend = cone_model(geometry), "cuda" if use_cuda_cone() else "jax"
    poses = stack_view_poses(geometry, len(ANGLES))
    rng = np.random.default_rng(81)
    data = rng.normal(size=(len(ANGLES), detector.nv, detector.nu)).astype(np.float32)
    source_data = _cone_slabs.RowReader(lambda a, b: data[:, a:b], data.shape) if reader else data
    problem = _cone_slabs.cone_problem(
        geometry, GRID, detector, source_data, model=model, backend=backend, poses=poses,
        batch=3, channels=1,
    )  # fmt: skip
    xp = problem.engine.xp
    value = rng.normal(size=(9, 4, 5)).astype(np.float32)
    volume = problem.volume(value)
    stats = ViewStatistics(problem, np.arange(len(ANGLES)))

    def read(k):
        return (volume.get(*problem.plan.planes(k)),)

    means, magnitudes = stats.project(read), stats.project(read, absolute=True)
    coefficients = model.coefficients(poses, GRID)
    projected = np.asarray(cone_project(value, coefficients, GRID, detector, backend=backend))

    def download(a):
        return np.asarray(a) if xp is np else xp.asnumpy(a)

    np.testing.assert_allclose(download(means), projected.mean((1, 2)), atol=2e-7)
    np.testing.assert_allclose(download(magnitudes), abs(projected).mean((1, 2)), atol=2e-7)
    np.testing.assert_allclose(
        download(stats.projection_mean(xp.asarray(projected))), download(means), atol=2e-7
    )
    np.testing.assert_allclose(
        download(stats.projection_mean(xp.asarray(projected), absolute=True)),
        download(magnitudes),
        atol=2e-7,
    )
    np.testing.assert_allclose(download(stats.mean), data.mean((1, 2)), atol=2e-7)
    np.testing.assert_allclose(download(stats.absolute_mean), abs(data).mean((1, 2)), rtol=1e-6)
    assert stats.outside_pixels > 0
    mean = means - stats.mean
    applied = np.zeros_like(value)
    target, energies = problem.volume(applied), []

    def finish(batch, first, stop, rows):
        rows[0][...] -= problem.measured(batch, first, stop) + mean[batch.index, None, None]
        energies.append(problem.engine.norm2(rows[0]))

    problem.sweep.run(
        np.arange(len(ANGLES)), source=read, finish=finish,
        sink=lambda k, a, _: target.put(*problem.plan.planes(k), a[0]), keep=False,
    )  # fmt: skip
    residual = projected - data
    residual -= residual.mean((1, 2), keepdims=True)
    expected = cone_backproject(residual, coefficients, GRID, detector, backend=backend)
    assert _relative(applied, expected) < 2e-6
    assert np.vdot(value.astype(float), applied) == pytest.approx(
        np.vdot(projected.astype(float), residual), rel=2e-6
    )
    assert sum(energies) + float(stats.outside_energy(-mean)) == pytest.approx(
        np.sum(residual.astype(float) ** 2), rel=2e-6
    )

    if reader:
        source_data.close()


@pytest.mark.parametrize("depth", [3, 4])
def test_host_contrast_cgls_matches_dense_optimum_and_true_stopping(host, depth):
    import jax
    import jax.numpy as jnp

    detector = replace(DETECTOR, nv=31)
    geometry = replace(_geometry("cone"), detector=detector)
    data = _data(geometry) + np.linspace(-0.2, 0.3, len(ANGLES))[:, None, None]
    cfg = CGLSConfig(iterations=100, rtol=1e-5, damping=0.3, data_term="per_view_contrast")
    matrix = np.asarray(
        jax.jacfwd(lambda x: tj.project(geometry, x.reshape(9, 4, 5)))(jnp.zeros(180)),
        dtype=np.float64,
    )
    matrix = (matrix - matrix.mean((1, 2), keepdims=True)).reshape(-1, 180)
    target = (data - data.mean((1, 2), keepdims=True)).ravel()
    reference = np.linalg.solve(
        matrix.T @ matrix + cfg.damping**2 * np.eye(180), matrix.T @ target
    ).reshape(9, 4, 5)
    host(depth)
    actual, info = cgls(geometry, GRID, detector, data, config=cfg)
    # Compare the converged objective, not two different finite FP32 CG sequences.
    assert _relative(actual, reference) < 2e-4
    residual = np.asarray(tj.project(geometry, actual)) - data
    residual -= residual.mean((1, 2), keepdims=True)
    gradient = np.asarray(tj.backproject(geometry, residual)) + cfg.damping**2 * actual
    assert info["converged"] and info["normal_residual_is_recomputed"]
    assert np.linalg.norm(gradient) <= 1.01 * info["normal_residual_tolerance"]
    assert info["data_term"] == "per_view_contrast" and info["background_unresolved"]
