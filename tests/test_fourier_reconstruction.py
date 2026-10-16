"""Independent physical and numerical checks for Fourier-slice reconstruction."""

from __future__ import annotations

from dataclasses import replace
import math
import subprocess
import sys

import jax
import numpy as np
import pytest
from scipy.fft import next_fast_len

from tomojax.geometry import (
    Detector,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
    grid_volume_origin,
)
from tomojax.recon import FourierConfig, fourier_reconstruct

BACKENDS = ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_host_fourier_does_not_import_unneeded_runtimes(backend):
    code = f"""
import sys, numpy as np
from tomojax.geometry import Grid, Detector, ParallelGeometry
from tomojax.recon import FourierConfig, fourier_reconstruct
g, d = Grid(4, 4, 1, 1, 1, 1), Detector(4, 1, 1, 1)
x = fourier_reconstruct(ParallelGeometry(g, d, [0., 90.]), g, d,
    np.zeros((2, 1, 4), np.float32), config=FourierConfig(backend='{backend}'),
    out=np.empty((4, 4, 1), np.float32))
assert 'jax' not in sys.modules and np.all(x == 0)
assert 'cupyx.scipy.fft' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.fixture
def row_grid():
    # check-public-imports: allow-private
    import tomojax.recon._fourier_grid as implementation

    return implementation


@pytest.fixture
def staged_rows():
    # check-public-imports: allow-private
    import tomojax.recon._fourier_rows as implementation

    return implementation


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
@pytest.mark.parametrize("nv", [1, 2, 5, 11])
def test_bounded_row_staging_matches_host_rounding_and_partial_padding(
    dtype, nv, row_grid, staged_rows
):
    sample_projection_rows = row_grid.sample_projection_rows
    ProjectionRows = staged_rows.ProjectionRows
    interpolate_projection_rows = staged_rows.interpolate_projection_rows
    prepare_projection_rows = staged_rows.prepare_projection_rows

    xp = pytest.importorskip("cupy")
    rng = np.random.default_rng(697)
    backing = (rng.normal(size=(7, nv, 26)) * 10).astype(dtype)
    backing.flat[::13] = -0.0
    backing.flat[::17] = 0.0
    data = backing[:, :, ::2]
    saved = data.copy()
    data.flags.writeable = False
    order = np.array([5, 2, 0, 6, 1])
    positions = np.array([-1e300, -1, -0.5, 0, 0.2, 0.99999999, 1, nv - 0.5, nv, 1e300])
    rows = prepare_projection_rows(data, order, positions)
    assert isinstance(rows, ProjectionRows)
    assert rows.samples.shape[0] <= 2 * len(positions)
    assert rows.samples.flags.c_contiguous
    actual, _owners = interpolate_projection_rows(
        rows, xp.asarray(rows.samples), len(positions) + 2, xp
    )
    actual = xp.asnumpy(actual)
    expected = np.pad(sample_projection_rows(data, order, positions), ((0, 2), (0, 0), (0, 0)))
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(data, saved)


def test_bounded_staging_ignores_unused_invalid_rows_without_taking_full_source(
    monkeypatch, staged_rows
):
    ProjectionRows = staged_rows.ProjectionRows
    prepare_projection_rows = staged_rows.prepare_projection_rows

    data = np.full((5, 1000, 7), np.nan, np.float64)[:, :, ::2]
    data[:, 499:501] = 2
    data.flags.writeable = False

    def forbidden(*args, **kwargs):
        raise AssertionError("np.take may make a full contiguous copy of strided input")

    monkeypatch.setattr(np, "take", forbidden)
    rows = prepare_projection_rows(data, np.arange(5), np.array([499.25, 499.75]))
    assert isinstance(rows, ProjectionRows)
    assert rows.samples.shape == (2, 5, 4)
    np.testing.assert_array_equal(rows.samples, 2)


@pytest.mark.gpu
def test_fractional_pipeline_commits_completed_slabs_before_late_invalid_neighbor(monkeypatch):
    import tomojax.recon.fourier as implementation

    pytest.importorskip("cupy")
    grid = Grid(17, 19, 13, 1, 1, 1, vol_origin=(-8, -9, -6.25))
    detector = Detector(35, 14, 1, 1)
    angles = np.linspace(0, 180, 17, endpoint=False)
    data, _ = physical_gaussian(grid, detector, angles)
    data[:, -1] = np.nan
    out = np.full((grid.nx, grid.ny, grid.nz), np.nan, np.float32)
    monkeypatch.setattr(implementation, "_HOST_PIPELINE_MIN_BYTES", 0)
    with pytest.raises(ValueError, match="sampled projection rows must be finite"):
        fourier_reconstruct(
            ParallelGeometry(grid, detector, angles),
            grid,
            detector,
            data,
            config=FourierConfig(backend="cupy", slices_per_batch=4),
            out=out,
        )
    assert np.isfinite(out[:, :, :12]).all()
    assert np.isnan(out[:, :, 12]).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_fractional_rows_preserve_readonly_strided_input_and_rounding(dtype, row_grid):
    sample_projection_rows = row_grid.sample_projection_rows

    rng = np.random.default_rng(553)
    data = (rng.normal(size=(7, 8, 22)) * 10).astype(dtype)[:, :, ::2]
    data[0, :2] = -0.0
    saved = data.copy()
    data.flags.writeable = False
    order = np.array([3, 0, 6, 1, 5, 4, 2])
    positions = np.array([0.5, 2.7, 4.2, 6.5])
    expected = np.zeros((4, 7, 11), np.float32)
    for index, position in enumerate(positions):
        row = int(np.floor(position))
        fraction = np.float32(position - row)
        expected[index] += np.asarray(data[order, row], np.float32) * (1 - fraction)
        expected[index] += np.asarray(data[order, row + 1], np.float32) * fraction
    actual = sample_projection_rows(data, order, positions)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(data, saved)


@pytest.mark.gpu
@pytest.mark.parametrize("threshold", [0, 10**12])
def test_cached_geometry_keeps_concurrent_calls_independent(monkeypatch, threshold):
    import tomojax.recon.fourier as implementation

    monkeypatch.setattr(implementation, "_HOST_PIPELINE_MIN_BYTES", threshold)
    from concurrent.futures import ThreadPoolExecutor

    cp = pytest.importorskip("cupy")
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid, detector, geometry, data, _ = scan()
    config = FourierConfig(backend="cupy", slices_per_batch=3)
    inputs = [data, data * np.float32(1.7)]
    expected = [fourier_reconstruct(geometry, grid, detector, x, config=config) for x in inputs]
    device = cp.cuda.runtime.getDevice()

    def reconstruct(x):
        with cp.cuda.Device(device), cp.cuda.Stream(non_blocking=True):
            return fourier_reconstruct(geometry, grid, detector, x, config=config)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reconstruct, inputs))
    for actual, reference in zip(results, expected, strict=True):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.gpu
def test_repeated_host_pipeline_calls_reuse_allocator_storage(monkeypatch):
    import tomojax.recon.fourier as implementation

    cp = pytest.importorskip("cupy")
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    monkeypatch.setattr(implementation, "_HOST_PIPELINE_MIN_BYTES", 0)
    grid = Grid(64, 64, 64, 1.0, 1.0, 1.0)
    detector = Detector(65, 64, 1.0, 1.0)
    angles = np.linspace(0, 180, 90, endpoint=False)
    data, _ = physical_gaussian(grid, detector, angles)
    geometry = ParallelGeometry(grid, detector, angles)
    config = FourierConfig(backend="cupy", slices_per_batch=16)
    reference = fourier_reconstruct(geometry, grid, detector, data, config=config)
    pool = cp.get_default_memory_pool()
    initial = pool.total_bytes()
    for _ in range(6):
        result = fourier_reconstruct(geometry, grid, detector, data, config=config)
        np.testing.assert_array_equal(result, reference)
    # Allow allocator fragmentation and FFT workspace variation, but reject a
    # new set of retained stream arenas on every completed reconstruction.
    assert pool.total_bytes() - initial < 4 * (data.nbytes + reference.nbytes)


@pytest.mark.parametrize("stage", ["construction", "later_slab"])
def test_cuda_allocation_exhaustion_retries_smaller_slabs_and_rewrites_output(monkeypatch, stage):
    import tomojax.recon.fourier as implementation

    original_plan = implementation.FourierPlan
    grid, detector, geometry, data, _ = scan()
    expected = fourier_reconstruct(
        geometry, grid, detector, data, config=FourierConfig(backend="numpy")
    )
    depths, destroyed = [], []

    class Exhausted(MemoryError):
        pass

    class Plan:
        def __init__(self, grid, detector, nviews, flipped, theta0, depth, backend, **kwargs):
            depths.append(depth)
            self.depth = depth
            self.calls = 0
            if stage == "construction" and depth > 2:
                raise Exhausted("test allocation")
            self.original = original_plan(grid, detector, nviews, flipped, theta0, depth, "numpy")

        def reconstruct(self, rows, *, count):
            self.calls += 1
            if self.depth > 2 and self.calls > 1:
                raise Exhausted("test later allocation")
            result = self.original.reconstruct(rows, count=count)
            # Deliberately dirty the output from the failed attempt. A retry
            # must recompute it, not just continue after the last complete slab.
            return result + (100 if self.depth > 2 else 0)

        def __del__(self):
            destroyed.append(self.depth)

    monkeypatch.setattr(implementation, "FourierPlan", Plan)
    monkeypatch.setattr(implementation, "is_cuda_oom", lambda error: isinstance(error, Exhausted))
    monkeypatch.setattr(implementation, "_HOST_PIPELINE_MIN_BYTES", 10**12)
    # The NumPy stand-in plan takes host-sampled rows, not CUDA-staged ones.
    monkeypatch.setattr(
        implementation, "prepare_projection_rows", implementation.sample_projection_rows
    )
    output = np.full(expected.shape, -123, np.float32)
    actual = fourier_reconstruct(
        geometry,
        grid,
        detector,
        data,
        config=FourierConfig(backend="cupy", slices_per_batch=4),
        out=output,
    )
    np.testing.assert_array_equal(actual, expected)
    assert depths == [4, 2]
    assert destroyed == [4, 2]


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("depth", [1, 4])
@pytest.mark.parametrize(
    "exception",
    [MemoryError("CPU memory"), ValueError("bad data"), ImportError("optional backend")],
)
def test_only_cuda_allocation_exhaustion_is_retried(monkeypatch, backend, depth, exception):
    import tomojax.recon.fourier as implementation

    grid, detector, geometry, data, _ = scan()
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(args[5])
        raise exception

    monkeypatch.setattr(implementation, "FourierPlan", fail)
    with pytest.raises(type(exception), match=str(exception)):
        fourier_reconstruct(
            geometry,
            grid,
            detector,
            data,
            config=FourierConfig(backend=backend, slices_per_batch=depth),
        )
    assert attempts == [depth]


def test_host_pipeline_drains_pending_transfers_after_output_failure():
    from typing import cast

    # check-public-imports: allow-private
    from tomojax.recon._fourier_backend import FourierPlan
    from tomojax.recon.fourier import _run_host_pipeline

    queued, waited, attempted = [], [], []

    class Job:
        def __init__(self, start):
            self.start = start

        def wait(self):
            waited.append(self.start)
            return np.zeros((4, 2, 2), np.float32)

    class Plan:
        def enqueue(self, data, *, count):
            assert count == 4
            queued.append(int(data[0]))
            return Job(int(data[0]))

    def write(start, _):
        attempted.append(start)
        if start == 4:
            raise OSError("output write failed")

    with pytest.raises(OSError, match="output write failed"):
        _run_host_pipeline(cast(FourierPlan, Plan()), 16, 4, lambda start: np.array([start]), write)
    assert attempted == [0, 4]
    assert set(waited) == set(queued)


@pytest.mark.gpu
@pytest.mark.parametrize("storage", ["array", "mapped", "mapped_F", "strided"])
def test_host_pipeline_matches_sequential_with_partial_slabs(monkeypatch, tmp_path, storage):
    import tomojax.recon.fourier as implementation

    pytest.importorskip("cupy")
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid = Grid(17, 19, 13, 0.9, 1.1, 0.8)
    detector = Detector(35, 19, 0.9, 0.8, (0.2, -0.1))
    angles = np.linspace(0, 180, 17, endpoint=False, dtype=np.float32)
    data, _ = physical_gaussian(grid, detector, angles)
    # Host staging must preserve strided input and FP64-to-FP32 conversion.
    backing = np.zeros((*data.shape[:2], data.shape[2] * 2), dtype=np.float64)
    backing[:, :, ::2] = data
    projections = backing[:, :, ::2]
    geometry = ParallelGeometry(grid, detector, angles)
    config = FourierConfig(backend="cupy", slices_per_batch=4)
    monkeypatch.setattr(implementation, "_HOST_PIPELINE_MIN_BYTES", 10**12)
    expected = fourier_reconstruct(geometry, grid, detector, projections, config=config)
    if storage.startswith("mapped"):
        out = np.lib.format.open_memmap(
            tmp_path / "pipeline.npy",
            mode="w+",
            dtype=np.float32,
            shape=expected.shape,
            fortran_order=storage == "mapped_F",
        )
    elif storage == "strided":
        out = np.empty((grid.nx * 2, grid.ny, grid.nz), dtype=np.float32)[::2]
    else:
        out = np.empty_like(expected)
    monkeypatch.setattr(implementation, "_HOST_PIPELINE_MIN_BYTES", 0)
    actual = fourier_reconstruct(geometry, grid, detector, projections, config=config, out=out)
    assert actual is out
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(projections, data)


@pytest.mark.gpu
def test_host_pipeline_finishes_completed_slabs_before_late_input_error(monkeypatch):
    import tomojax.recon.fourier as implementation

    pytest.importorskip("cupy")
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid = Grid(17, 19, 13, 1.0, 1.0, 1.0)
    detector = Detector(35, 13, 1.0, 1.0)
    angles = np.linspace(0, 180, 17, endpoint=False, dtype=np.float32)
    data, _ = physical_gaussian(grid, detector, angles)
    data[:, -1, :] = np.nan
    out = np.full((grid.nx, grid.ny, grid.nz), np.nan, dtype=np.float32)
    monkeypatch.setattr(implementation, "_HOST_PIPELINE_MIN_BYTES", 0)
    with pytest.raises(ValueError, match="sampled projection rows must be finite"):
        fourier_reconstruct(
            ParallelGeometry(grid, detector, angles),
            grid,
            detector,
            data,
            config=FourierConfig(backend="cupy", slices_per_batch=4),
            out=out,
        )
    assert np.isfinite(out[:, :, :12]).all()
    assert np.isnan(out[:, :, 12]).all()


@pytest.fixture(params=BACKENDS)
def backend(request):
    if request.param == "cupy":
        cp = pytest.importorskip("cupy")
        if jax.default_backend() != "gpu" or cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("requires CUDA")
    return request.param


def physical_gaussian(grid, detector, angles, length_scale=1.0):
    """Analytic line integrals of an off-axis Gaussian, independent of TomoJAX."""
    mu = np.array([-2.4, 1.1, 0.4]) * length_scale
    sigma = np.array([3.0, 2.1, 4.0]) * length_scale
    coords = [
        first + np.arange(count) * spacing
        for first, count, spacing in zip(
            grid_volume_origin(grid),
            (grid.nx, grid.ny, grid.nz),
            (grid.vx, grid.vy, grid.vz),
            strict=True,
        )
    ]
    x, y, z = np.meshgrid(*coords, indexing="ij", sparse=True)
    truth = np.exp(
        -0.5
        * (
            ((x - mu[0]) / sigma[0]) ** 2
            + ((y - mu[1]) / sigma[1]) ** 2
            + ((z - mu[2]) / sigma[2]) ** 2
        )
    )
    u = detector.center[0] + (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du
    v = detector.center[1] + (np.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv
    c, s = np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))
    mean = c * mu[0] - s * mu[1]
    variance = (c * sigma[0]) ** 2 + (s * sigma[1]) ** 2
    profile = np.sqrt(2 * np.pi) * sigma[0] * sigma[1] / np.sqrt(variance[:, None])
    profile = profile * np.exp(-0.5 * (u[None, :] - mean[:, None]) ** 2 / variance[:, None])
    axial = np.exp(-0.5 * ((v - mu[2]) / sigma[2]) ** 2)
    data = (profile[:, None, :] * axial[None, :, None]).astype(np.float32)
    return data, truth


def scan(kind="offset", length_scale=1.0):
    grid = Grid(33, 29, 7, 0.8, 1.2, 0.9, vol_center=(1.3, -0.7, 0.1))
    detector = Detector(65, 11, 0.8, 1.1, (0.53, -0.21))
    angles = np.linspace(0.0, 180.0, 90, endpoint=False)
    if kind == "cropped":
        grid = Grid(13, 11, 5, 0.8, 1.2, 0.9, vol_origin=(2.4, -7.2, -1.8))
    elif kind == "reordered":
        angles = angles[np.random.default_rng(314).permutation(len(angles))]
    elif kind == "opposing":
        angles[::2] += 180
    elif kind == "reversed":
        angles = (angles + 37.5)[::-1] - 360
    elif kind == "inclusive":
        angles = np.linspace(0, 180, 91)
    elif kind == "repeated":
        angles = np.repeat(angles, 3)
        angles[1::3] += 180
    elif kind == "nonuniform":
        # Uneven encoder positions, rejected views, arbitrary order and opposing
        # orientations must use the actual angles rather than view indices.
        angles += 0.7 * np.sin(np.arange(len(angles)) * 1.3)
        angles = np.delete(angles, [11, 42])
        angles[::3] += 180
        angles = angles[np.random.default_rng(27).permutation(len(angles))]
    elif kind == "even":
        detector = replace(detector, nu=64)
    grid = replace(
        grid,
        vx=grid.vx * length_scale,
        vy=grid.vy * length_scale,
        vz=grid.vz * length_scale,
        vol_origin=tuple(v * length_scale for v in grid_volume_origin(grid)),
        vol_center=None,
    )
    detector = replace(
        detector,
        du=detector.du * length_scale,
        dv=detector.dv * length_scale,
        center=tuple(v * length_scale for v in detector.center),
    )
    geometry = ParallelGeometry(grid, detector, angles)
    data, truth = physical_gaussian(grid, detector, angles, length_scale)
    return grid, detector, geometry, data, truth


@pytest.mark.parametrize(
    "kind",
    [
        *("offset", "cropped", "reordered", "opposing", "reversed", "even"),
        *("nonuniform", "inclusive", "repeated"),
    ],
)
def test_independent_physical_gaussian(backend, kind):
    grid, detector, geometry, data, truth = scan(kind)
    original = data.copy()
    actual = fourier_reconstruct(
        geometry, grid, detector, data, config=FourierConfig(slices_per_batch=3, backend=backend)
    )
    assert actual.dtype == np.float32
    assert actual.shape == truth.shape
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.008
    np.testing.assert_array_equal(data, original)


def test_redundant_exposures_share_one_angular_cell(backend):
    grid, detector, geometry, data, _ = scan()
    exposures = np.repeat(data, 3, axis=0)
    exposures += np.random.default_rng(723).normal(0, 0.02, exposures.shape).astype(np.float32)
    config = FourierConfig(backend=backend)
    expected = fourier_reconstruct(
        geometry,
        grid,
        detector,
        exposures.reshape(len(data), 3, *data.shape[1:]).mean(axis=1),
        config=config,
    )
    actual = fourier_reconstruct(
        ParallelGeometry(grid, detector, np.repeat(geometry.angles, 3)),
        grid,
        detector,
        exposures,
        config=config,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("length_scale", [0.5, 2.0])
def test_physical_units_preserve_attenuation(backend, length_scale):
    grid, detector, geometry, data, truth = scan(length_scale=length_scale)
    actual = fourier_reconstruct(
        geometry, grid, detector, data, config=FourierConfig(slices_per_batch=20, backend=backend)
    )
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.008


@pytest.mark.parametrize("nu", [1, 32, 65, 131])
@pytest.mark.parametrize("signal", ["signed", "constant", "edge"])
def test_kaiser_interpolation_matches_independent_dense_fourier_sum(nu, signal, row_grid):
    interpolate_numpy, radial_coefficients = (
        row_grid.interpolate_numpy,
        row_grid.radial_coefficients,
    )

    rng = np.random.default_rng(818)
    values = rng.normal(size=nu)
    if signal == "constant":
        values[:] = 1
    elif signal == "edge":
        values[:] = 0
        values[-1] = 1
    nfft = max(64, next_fast_len(2 * nu, real=True))
    phase, deapod, table = radial_coefficients(nu, nfft, 0.7)
    spectrum = np.fft.rfft(values * deapod, n=nfft) * phase
    spectrum.flags.writeable = False
    radial = np.concatenate(
        (rng.uniform(0, nfft / 2, 100), [0, 1, nfft // 2, nfft / 2, nfft / 2 + 0.001, nfft])
    )
    angle = np.zeros_like(radial)
    ones = np.asarray(1, dtype=np.complex128)
    actual = interpolate_numpy(
        spectrum[None, None, :],
        angle,
        radial,
        ones,
        ones,
        np.array([0]),
        table,
        nfft,
        np.exp(1j * np.pi * (nu - 1)),
    )[0]
    coordinates = (np.arange(nu) - (nu - 1) / 2) / nfft
    reference = 0.7 * (np.exp(-2j * np.pi * radial[:, None] * coordinates) @ values)
    reference[radial > nfft / 2] = 0  # beyond the sampled disk
    assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < 2e-5


@pytest.mark.gpu
@pytest.mark.parametrize("nu", [1, 32, 65])
@pytest.mark.parametrize("vx", [0.8, 1.3])
@pytest.mark.parametrize("views", [13, 720])
def test_cuda_matches_numpy_for_signed_data_and_partial_slabs(nu, vx, views):
    pytest.importorskip("cupy")
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid = Grid(11, 13, 5, vx, 1.3, 0.7, vol_origin=(-2.4, -5.1, -1.1))
    detector = Detector(nu, 7, 0.9, 1.1, (0.7, -0.23))
    angles = np.linspace(-37, 143, views, endpoint=False)[::-1]
    geometry = ParallelGeometry(grid, detector, angles)
    data = np.random.default_rng(321).normal(size=(views, 7, nu)).astype(np.float32)
    reference = fourier_reconstruct(
        geometry, grid, detector, data, config=FourierConfig(slices_per_batch=2, backend="numpy")
    )
    actual = fourier_reconstruct(
        geometry, grid, detector, data, config=FourierConfig(slices_per_batch=2, backend="cupy")
    )
    assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < 2e-5
    np.testing.assert_allclose(actual, reference, rtol=2e-4, atol=2e-6)


def test_detector_rows_zero_extend_and_ignore_unused_nonfinite_samples(row_grid):
    sample_projection_rows = row_grid.sample_projection_rows

    data = np.array([[[np.nan], [3], [7], [np.nan]]], dtype=np.float32)
    rows = sample_projection_rows(data, np.array([0]), np.array([-1e30, -1, 1, 1.5, 2, 4, 1e30]))
    np.testing.assert_array_equal(rows[:, 0, 0], [0, 0, 3, 5, 7, 0, 0])


def test_unmeasured_axial_slices_are_zero(backend):
    grid = Grid(9, 7, 5, 1.0, 1.0, 1.0)
    detector = Detector(9, 1, 1.0, 1.0)
    geometry = ParallelGeometry(grid, detector, np.arange(10) * 18)
    data = np.ones((10, 1, 9), dtype=np.float32)
    actual = fourier_reconstruct(
        geometry, grid, detector, data, config=FourierConfig(slices_per_batch=3, backend=backend)
    )
    np.testing.assert_array_equal(actual[:, :, [0, 1, 3, 4]], 0)
    assert np.linalg.norm(actual[:, :, 2]) > 0


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 1e300])
def test_nonfinite_sampled_rows_rejected_before_output_written(invalid):
    grid, detector, geometry, data, _ = scan()
    data = data.astype(np.float64)
    data[0, 3, 1] = invalid
    output = np.full((grid.nx, grid.ny, grid.nz), 123, dtype=np.float32)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite in FP32"):
        fourier_reconstruct(geometry, grid, detector, data, out=output)
    np.testing.assert_array_equal(output, 123)


def test_memmap_output_and_batch_size_independence(tmp_path, backend):
    grid, detector, geometry, data, _ = scan()
    source = np.memmap(tmp_path / "source.bin", dtype=np.float32, mode="w+", shape=data.shape)
    source[:] = data
    target = np.memmap(
        tmp_path / "target.bin", dtype=np.float32, mode="w+", shape=(grid.nx, grid.ny, grid.nz)
    )
    actual = fourier_reconstruct(
        geometry,
        grid,
        detector,
        source,
        config=FourierConfig(slices_per_batch=3, backend=backend),
        out=target,
    )
    assert actual is target
    reference = fourier_reconstruct(
        geometry, grid, detector, data, config=FourierConfig(slices_per_batch=7, backend=backend)
    )
    assert reference.flags.f_contiguous and actual.flags.c_contiguous
    np.testing.assert_allclose(actual, reference, rtol=3e-5, atol=3e-6)
    np.testing.assert_array_equal(source, data)


@pytest.mark.parametrize("mapped", [False, True])
def test_overlapping_storage_rejected_before_writing(tmp_path, mapped):
    grid, detector = Grid(5, 4, 3, 1, 1, 1), Detector(3, 4, 1, 1)
    geometry = ParallelGeometry(grid, detector, np.arange(5) * 36)
    if mapped:
        data = np.memmap(tmp_path / "same.bin", mode="w+", dtype=np.float32, shape=(5, 4, 3))
        out = np.memmap(tmp_path / "same.bin", mode="r+", dtype=np.float32, shape=data.shape)
    else:
        data = np.ones((5, 4, 3), dtype=np.float32)
        out = data.view()
    data[:] = 123
    with pytest.raises(ValueError, match="overlap|separate files"):
        fourier_reconstruct(geometry, grid, detector, data, out=out)
    np.testing.assert_array_equal(data, 123)


@pytest.mark.parametrize(
    "invalid",
    [
        "tilted",
        "nonfinite",
        "complex",
        "shape",
        "readonly",
        "dtype",
        "backend",
        "batch",
    ],
)
def test_invalid_inputs_rejected_before_output_written(invalid):
    grid, detector, geometry, data, _ = scan()
    out = np.full((grid.nx, grid.ny, grid.nz), 123, dtype=np.float32)
    config = FourierConfig()
    if invalid == "nonfinite":
        angles = np.asarray(geometry.angles).copy()
        angles[1] = np.nan
        geometry = ParallelGeometry(grid, detector, angles)
    elif invalid == "tilted":
        geometry = LaminographyGeometry(grid, detector, geometry.angles, tilt_deg=30)
    elif invalid == "complex":
        data = data.astype(np.complex64)
    elif invalid == "shape":
        data = data[:, :, :-1]
    elif invalid == "readonly":
        out.flags.writeable = False
    elif invalid == "dtype":
        out = out.astype(np.float64)
    elif invalid == "backend":
        config = FourierConfig(backend="invalid")
    elif invalid == "batch":
        config = FourierConfig(slices_per_batch=0)
    with pytest.raises((ValueError, TypeError), match="fourier_reconstruct"):
        fourier_reconstruct(geometry, grid, detector, data, config=config, out=out)
    np.testing.assert_array_equal(out, 123)


def test_missing_cupy_does_not_mask_original_memory_or_import_errors(monkeypatch):
    import sys

    import tomojax.recon.fourier as implementation

    monkeypatch.setitem(sys.modules, "cupy", None)
    assert not implementation.is_cuda_oom(MemoryError("host allocation"))
    assert not implementation.is_cuda_oom(ImportError("optional backend"))


@pytest.mark.parametrize("nz", [1, 6, 7])
@pytest.mark.parametrize("threshold", [0, 10**12])
def test_fixed_fft_batches_materialize_only_real_output_slices(monkeypatch, backend, nz, threshold):
    import tomojax.recon.fourier as implementation

    grid, detector, geometry, _, _ = scan()
    grid = replace(grid, nz=nz)
    geometry = ParallelGeometry(grid, detector, geometry.angles)
    data, truth = physical_gaussian(grid, detector, geometry.angles)
    saved = data.copy()
    data.flags.writeable = False
    batches = []
    original_plan = implementation.FourierPlan

    class Plan(original_plan):
        def _compute_volume(self, raw, count):
            volume = super()._compute_volume(raw, count)
            batches.append((raw.shape[0], self.frequencies.shape[0], count, volume.shape[0]))
            assert volume.size == count * grid.nx * grid.ny
            return volume

    monkeypatch.setattr(implementation, "FourierPlan", Plan)
    monkeypatch.setattr(implementation, "_HOST_PIPELINE_MIN_BYTES", threshold)
    actual = fourier_reconstruct(
        geometry,
        grid,
        detector,
        data,
        config=FourierConfig(backend=backend, slices_per_batch=3),
    )
    depth = min(3, nz)
    assert batches == [
        (depth, depth, min(depth, nz - start), min(depth, nz - start))
        for start in range(0, nz, depth)
    ]
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.015
    np.testing.assert_array_equal(data, saved)


@pytest.mark.parametrize("layout", ["C", "F", "reversed"])
def test_provided_output_retains_layout_and_identity(backend, layout):
    grid, detector, geometry, data, truth = scan()
    config = FourierConfig(backend=backend, slices_per_batch=3)
    expected = fourier_reconstruct(geometry, grid, detector, data, config=config)
    out = np.empty(truth.shape, dtype=np.float32, order="F" if layout == "F" else "C")
    if layout == "reversed":
        out = out[::-1, ::-1, ::-1]
    strides = out.strides
    result = fourier_reconstruct(geometry, grid, detector, data, config=config, out=out)
    assert result is out and result.strides == strides
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))


def dense_full_spectrum_reference(grid, detector, angles, data):
    """Centred DFT of every sampled detector profile, then a full complex Cartesian FFT."""
    origin = grid_volume_origin(grid)
    radius = abs(detector.center[0]) + detector.nu * detector.du / 2
    shape, crop = [], []
    for count, spacing, first in zip(
        (grid.nx, grid.ny), (grid.vx, grid.vy), origin[:2], strict=True
    ):
        lo = min(0, math.floor((-radius - first) / spacing))
        hi = max(count - 1, math.ceil((radius - first) / spacing))
        support = hi - lo + 1
        padded = next_fast_len(2 * support, real=True)
        shape.append(padded)
        crop.append(-lo + (padded - support) // 2)
    nviews = len(angles)
    order = np.argsort(np.mod(angles, 180))
    theta0 = np.deg2rad(np.mod(angles[order[0]], 180))
    flipped = np.mod(angles[order], 360) >= 180
    raw = np.zeros((grid.nz, nviews, detector.nu), np.float32)
    positions = (origin[2] + np.arange(grid.nz) * grid.vz - detector.center[1]) / detector.dv + (
        detector.nv - 1
    ) / 2
    for z, position in enumerate(positions):
        low = int(np.floor(position))
        fraction = np.float32(position - low)
        for row, weight in [(low, 1 - fraction), (low + 1, fraction)]:
            if 0 <= row < detector.nv and weight != 0:
                raw[z] += data[order, row, :] * weight
    fx = np.fft.fftfreq(shape[0], d=grid.vx)[:, None]
    fy = np.fft.fftfreq(shape[1], d=grid.vy)[None, :]
    frequency = np.sqrt(fx * fx + fy * fy)
    u = detector.center[0] + (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du
    matrix = detector.du * np.exp(-2j * np.pi * frequency.reshape(-1, 1) * u[None, :])
    measured = raw.astype(np.float64) @ matrix.T
    a = (np.mod(np.arctan2(-fy, fx) - theta0, 2 * np.pi) * (nviews / np.pi)).ravel()
    index = np.floor(a).astype(np.int64)
    sites = np.arange(len(a))
    values = np.zeros((grid.nz, len(a)), np.complex128)
    for shift, weight in [(0, 1 - (a - index)), (1, a - index)]:
        view = (index + shift) % (2 * nviews)
        conjugate = (view >= nviews) ^ flipped[view % nviews]
        sampled = measured[:, view % nviews, sites]
        values += np.where(conjugate, sampled.conj(), sampled) * weight
    phase = np.exp(
        2j * np.pi * (fx * (origin[0] - crop[0] * grid.vx) + fy * (origin[1] - crop[1] * grid.vy))
    )
    values = values.reshape(grid.nz, *shape) * phase
    values[:, frequency > 0.5 / detector.du] = 0
    values[:, 0, 0] = raw.astype(np.float64).sum(axis=-1).mean(axis=-1) * detector.du
    full = np.fft.ifft2(values, axes=(-2, -1)).real / (grid.vx * grid.vy)
    cropped = full[:, crop[0] : crop[0] + grid.nx, crop[1] : crop[1] + grid.ny]
    return cropped.transpose(1, 2, 0).astype(np.float32)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ny", [11, 13])
@pytest.mark.parametrize("vx", [0.35, 1.3])
def test_signed_profiles_match_a_dense_full_spectrum_reconstruction(backend, ny, vx):
    # Signed rows have power at the x Nyquist bin, which a half-plane transform
    # must project to a real image as the full transform does.
    if backend == "cupy":
        pytest.importorskip("cupy")
    grid = Grid(13, ny, 5, vx, 1.7, 0.7, vol_center=(0.2, -0.3, 0.1))
    detector = Detector(17, 7, 0.9, 1.1, (0.7, -0.23))
    angles = -37 + np.arange(13) * (180 / 13)
    angles[::3] += 180
    angles = angles[np.random.default_rng(17).permutation(13)]
    data = np.random.default_rng(321).normal(size=(13, 7, 17)).astype(np.float32)
    data += ((-1.0) ** np.arange(17))[None, None, :]
    reference = dense_full_spectrum_reference(grid, detector, angles, data)
    actual = fourier_reconstruct(
        ParallelGeometry(grid, detector, angles),
        grid,
        detector,
        data,
        config=FourierConfig(slices_per_batch=3, backend=backend),
    )
    assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < 2e-5


def test_automatic_slab_depth_bounds_padded_physical_support_and_preserves_overrides(monkeypatch):
    import tomojax.recon.fourier as implementation

    depths = []

    def plan(*args, **kwargs):
        depths.append(args[5])
        raise RuntimeError("planning complete")

    monkeypatch.setattr(implementation, "FourierPlan", plan)
    detector = Detector(64, 31, 1, 1)
    angles = np.linspace(0, 180, 32, endpoint=False)
    data = np.zeros((32, 31, 64), np.float32)
    for pitch, expected in ((1, 16), (1 / 16, 8), (1 / 36, 2)):
        grid = Grid(4, 4, 31, pitch, pitch, 1)
        geometry = ParallelGeometry(grid, detector, angles)
        for requested, selected in ((None, expected), (5, 5)):
            cfg = FourierConfig(backend="cupy", slices_per_batch=requested)
            with pytest.raises(RuntimeError, match="planning complete"):
                fourier_reconstruct(geometry, grid, detector, data, config=cfg)
            assert depths[-1] == selected
