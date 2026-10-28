"""Cone volumes in host slabs: the slab sum is the matched projector, solvers match resident ones."""

from dataclasses import replace
from importlib import import_module
import os

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
    truth = np.random.default_rng(seed).random((5, 4, 9), np.float32)
    return np.asarray(tj.project(geometry, truth), np.float32)


def _relative(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.fixture
def host(monkeypatch):
    """Force host slabs of ``depth`` planes for the cone solvers."""

    def force(depth):
        monkeypatch.setattr(_cone_slabs, "slab_depth", lambda *_, **__: depth)
        monkeypatch.setattr(_cone_slabs, "_BAND", 3)  # rows finished in bands of three
        for name in ("tomojax.recon._cgls_host", "tomojax.recon.sirt"):
            monkeypatch.setattr(import_module(name), "host_volumes", lambda *_: True)

    return force


@pytest.mark.parametrize("kind", ["cone", "segments"])
@pytest.mark.parametrize("depth", [1, 4])
def test_slab_sweep_is_the_matched_normal_operator(kind, depth, host):
    host(depth)
    geometry = _geometry(kind)
    model, backend = cone_model(geometry), "cuda" if use_cuda_cone() else "jax"
    poses = stack_view_poses(geometry, len(ANGLES))
    problem = _cone_slabs.cone_problem(
        geometry, GRID, DETECTOR, _data(geometry), model=model, backend=backend, poses=poses,
        batch=3, channels=1,
    )  # fmt: skip
    direction = np.random.default_rng(2).random((5, 4, 9), np.float32)
    source = problem.volume(np.asfortranarray(direction))
    applied = np.zeros_like(direction, order="F")
    target, energy = problem.volume(applied), []

    def finish(batch, first, stop, rows):
        energy.append(float(np.sum(np.square(rows[0], dtype=np.float64))))

    problem.sweep.run(
        np.arange(len(ANGLES)),
        source=lambda k: (source.get(*problem.plan.planes(k)),),
        finish=finish,
        sink=lambda k, accumulated, _: target.put(*problem.plan.planes(k), accumulated[0]),
    )
    coefficients = model.coefficients(poses, GRID)
    projected = cone_project(direction, coefficients, GRID, DETECTOR, backend=backend)
    expected = cone_backproject(projected, coefficients, GRID, DETECTOR, backend=backend)
    assert sum(energy) == pytest.approx(
        float(np.sum(np.square(projected, dtype=np.float64))), rel=1e-6
    )
    assert _relative(applied, expected) < 1e-6


@pytest.mark.parametrize("depth", [3, 4])
def test_host_cgls_matches_streamed_normal_equations(host, depth):
    geometry = _geometry("cone")
    data = _data(geometry)
    start = np.random.default_rng(3).random((5, 4, 9), np.float32)
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
        SIRTConfig(iterations=3, subsets=3, lower_bound=0.0, upper_bound=0.9),
        SIRTConfig(iterations=3, support=np.arange(180).reshape(5, 4, 9) % 7 > 0),
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
    assert isinstance(result.volume, np.ndarray) and result.volume.flags.f_contiguous
    assert result.info["host_slabs"] and result.info["effective_iterations"] == 2


def test_memmapped_out_holds_iterate_and_work_volumes_are_unnamed(tmp_path, host):
    geometry = _geometry("cone")
    scan = tj.Scan(_data(geometry), geometry)
    expected = tj.reconstruct(scan, "cgls", iterations=3).volume
    host(4)
    out = np.lib.format.open_memmap(
        tmp_path / "x.npy", "w+", np.float32, (5, 4, 9), fortran_order=True
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
        tmp_path / "v.npy", "w+", np.float32, (3, 2, 70), fortran_order=True
    )
    work = host_volume.host_volume((3, 2, 70), host_volume.scratch_directory(named))
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
        np.testing.assert_array_equal(np.asarray(array[:, :, 5:7]).T, back)
        np.testing.assert_array_equal(back, block)
    data = np.ones((20, 2, 2), np.float32)
    host_volume.check_rows(data, "test", nonnegative=True)
    data[17, 1, 0] = -1
    with pytest.raises(ValueError, match="test: projections must be finite and nonnegative"):
        host_volume.check_rows(data, "test", nonnegative=True)
    data[17, 1, 0] = np.inf
    with pytest.raises(ValueError, match="finite in FP32"):
        host_volume.check_rows(data, "test")


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
