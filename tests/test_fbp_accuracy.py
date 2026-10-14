"""Independent analytic checks for FBP filtering and physical normalization."""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import i0e, polygamma

from tomojax.geometry import (
    Detector,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
    grid_volume_origin,
)
from tomojax.recon import FBPConfig, FBPHostConfig, fbp, fbp_host

# check-public-imports: allow-private
from tomojax.recon._fbp_pallas import backproject_filtered_pallas
from tomojax.recon._fbp_weights import fbp_weights
from tomojax.recon.fbp import _backproject_voxels_jax
from tomojax.recon.filters import fft_filter_rows, rfft_filter_array


@dataclass(frozen=True)
class _CacheDevice:
    id: int
    local_hardware_id: int | None
    platform: str = "gpu"
    version: str = "CUDA 12.0"

    @property
    def client(self) -> SimpleNamespace:
        return SimpleNamespace(platform_version=self.version)


def test_fbp_cache_batch_queries_each_actual_device_once(monkeypatch) -> None:
    module = importlib.import_module("tomojax.recon._fbp_pallas")
    module._cuda_l2_bytes.cache_clear()
    calls = []

    def properties(local):
        calls.append(local)
        return {"l2CacheSize": (32 if local == 7 else 64) * 1024**2}

    monkeypatch.setitem(
        sys.modules, "cupy.cuda.runtime", SimpleNamespace(getDeviceProperties=properties)
    )
    detector = Detector(3900, 64, 1.0, 1.0)
    first, second = _CacheDevice(42, 7), _CacheDevice(43, 3)
    assert (
        module.filtered_view_batch_size(
            128, grid=Grid(2560, 2560, 64, 1.0, 1.0, 1.0), detector=detector, device=first
        )
        == 64
    )
    assert (
        module.filtered_view_batch_size(
            128, grid=Grid(2560, 2560, 64, 1.0, 1.0, 1.0), detector=detector, device=second
        )
        == 128
    )
    assert (
        module.filtered_view_batch_size(
            128, grid=Grid(2560, 2560, 64, 1.0, 1.0, 1.0), detector=detector, device=first
        )
        == 64
    )
    assert calls == [7, 3]


@pytest.mark.parametrize("size", [0, -1, 1, 32 * 1024**2])
@pytest.mark.parametrize("batch", [1, 3, 16, 128])
def test_fbp_cache_batch_never_exceeds_fft_budget(monkeypatch, size, batch) -> None:
    module = importlib.import_module("tomojax.recon._fbp_pallas")
    module._cuda_l2_bytes.cache_clear()
    monkeypatch.setitem(
        sys.modules,
        "cupy.cuda.runtime",
        SimpleNamespace(getDeviceProperties=lambda _: {"l2CacheSize": size}),
    )
    actual = module.filtered_view_batch_size(
        batch,
        grid=Grid(2560, 2560, 64, 1.0, 1.0, 1.0),
        detector=Detector(3900, 64, 1.0, 1.0),
        device=_CacheDevice(42, 7),
    )
    expected = batch if size <= 0 else min(batch, 64 if size == 32 * 1024**2 else 1)
    assert actual == expected


@pytest.mark.parametrize(
    "device",
    [_CacheDevice(42, 7, "cpu"), _CacheDevice(42, 7, version="ROCm"), _CacheDevice(42, None)],
)
def test_fbp_cache_batch_skips_unavailable_cuda_device(monkeypatch, device) -> None:
    module = importlib.import_module("tomojax.recon._fbp_pallas")
    module._cuda_l2_bytes.cache_clear()

    def forbidden(_):
        pytest.fail("non-CUDA or unidentified devices must not query CuPy")

    monkeypatch.setitem(
        sys.modules, "cupy.cuda.runtime", SimpleNamespace(getDeviceProperties=forbidden)
    )
    assert (
        module.filtered_view_batch_size(
            128,
            grid=Grid(2560, 2560, 64, 1.0, 1.0, 1.0),
            detector=Detector(3900, 64, 1.0, 1.0),
            device=device,
        )
        == 128
    )


@pytest.mark.parametrize("missing", [True, False])
def test_fbp_cache_batch_retains_fft_budget_without_optional_probe(monkeypatch, missing) -> None:
    module = importlib.import_module("tomojax.recon._fbp_pallas")
    module._cuda_l2_bytes.cache_clear()

    def unavailable(_):
        raise RuntimeError("optional CUDA runtime is unavailable")

    monkeypatch.setitem(
        sys.modules,
        "cupy.cuda.runtime",
        None if missing else SimpleNamespace(getDeviceProperties=unavailable),
    )
    assert (
        module.filtered_view_batch_size(
            128,
            grid=Grid(2560, 2560, 64, 1.0, 1.0, 1.0),
            detector=Detector(3900, 64, 1.0, 1.0),
            device=_CacheDevice(42, 7),
        )
        == 128
    )


@pytest.mark.parametrize(
    "grid",
    [
        Grid(32, 32, 64, 1.0, 1.0, 1.0),
        Grid(2560, 2560, 129, 1.0, 1.0, 1.0),
        Grid(2560, 2560, 64, 1.0, 1.0, 1.0, vol_center=(0.0, 0.0, 90.0)),
        Grid(2560, 2560, 32, 1.0, 1.0, 3.0),
    ],
)
def test_fbp_cache_batch_keeps_small_or_axially_sparse_output(monkeypatch, grid) -> None:
    module = importlib.import_module("tomojax.recon._fbp_pallas")
    monkeypatch.setattr(module, "_cuda_l2_bytes", lambda _: 32 * 1024**2)
    assert (
        module.filtered_view_batch_size(
            128, grid=grid, detector=Detector(3900, 64, 1.0, 1.0), device=_CacheDevice(42, 7)
        )
        == 128
    )


def test_fbp_cache_batch_accounts_for_fractional_row_spacing(monkeypatch) -> None:
    module = importlib.import_module("tomojax.recon._fbp_pallas")
    monkeypatch.setattr(module, "_cuda_l2_bytes", lambda _: 32 * 1024**2)
    assert (
        module.filtered_view_batch_size(
            128,
            grid=Grid(2560, 2560, 128, 1.0, 1.0, 0.5),
            detector=Detector(3900, 64, 1.0, 1.0),
            device=_CacheDevice(42, 7),
        )
        == 64
    )


@pytest.mark.parametrize(
    ("nz", "nv", "vz", "columns", "capacity"),
    [
        (64, 64, 1.0, 3900, 64),
        (128, 64, 0.5, 3900, 64),
        (32, 129, 4.0, 3900, 16),
        (540, 540, 1.0, 976, 256),
    ],
)
@pytest.mark.parametrize("budget", [1, 3, 16, 512])
@pytest.mark.parametrize("unit", [0.2, 1.3])
def test_fbp_cache_batch_budgets_the_physical_axial_band(
    monkeypatch, nz, nv, vz, columns, capacity, budget, unit
):
    implementation = importlib.import_module("tomojax.recon._fbp_pallas")
    monkeypatch.setattr(implementation, "_cuda_l2_bytes", lambda _: 32 * 1024**2)
    actual = implementation.filtered_view_batch_size(
        budget,
        grid=Grid(2560, 2560, nz, unit, unit, vz * unit),
        detector=Detector(columns, nv, unit, unit),
        device=_CacheDevice(42, 7),
    )
    assert actual == min(budget, capacity)


@pytest.mark.parametrize(
    ("shape", "capacity"),
    [((1024, 2048), 2), ((512, 256), 32), ((64, 64), 1024), ((8192, 8192), 1)],
)
@pytest.mark.parametrize("budget", [1, 3, 16, 128])
@pytest.mark.parametrize("available", [True, False])
def test_2d_fbp_cache_batch_respects_fft_budget_and_actual_device(
    monkeypatch, shape, capacity, budget, available
):
    module = importlib.import_module("tomojax.recon._fbp_pallas")
    device = _CacheDevice(42, 7)

    def cache(target):
        assert target is device
        return 32 * 1024**2 if available else None

    monkeypatch.setattr(module, "_cuda_l2_bytes", cache)
    actual = module.fft_2d_view_batch_size(budget, fft_shape=shape, device=device)
    assert actual == (min(budget, capacity) if available else budget)


@pytest.mark.parametrize(("nu", "nv"), [(128, 16), (128, 64), (128, 256), (2048, 1024)])
@pytest.mark.parametrize("tilt_about", ["x", "z"])
def test_oblique_fbp_budgets_the_actual_rectangular_fft(monkeypatch, nu, nv, tilt_about):
    """A 2D FFT replaces the row dimension; it does not multiply it by nv."""
    import jax

    module = importlib.import_module("tomojax.recon.fbp")
    angles = np.linspace(0, 180, 181, endpoint=False)
    grid = Grid(32, 32, 8, 0.8, 1.1, 0.9, vol_center=(2.1, -1.3, 0.8))
    detector = Detector(nu, nv, 1.0, 1.0, (-1.3, 2.1))
    geometry = LaminographyGeometry(grid, detector, angles, tilt_deg=30, tilt_about=tilt_about)
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(len(angles))])
    # Planning and tracing must not materialize this potentially large stack.
    data = np.broadcast_to(np.float32(0), (len(angles), nv, nu))
    estimates, shapes = [], []
    original_batch, original_fft = module._parallel_filter_batch_size, jnp.fft.rfft2

    def batch(n, rows, columns):
        estimates.append((n, rows, columns))
        return original_batch(n, rows, columns)

    def fft(*args, **kwargs):
        shapes.append(kwargs["s"])
        return original_fft(*args, **kwargs)

    monkeypatch.setattr(module, "_parallel_filter_batch_size", batch)
    monkeypatch.setattr(jnp.fft, "rfft2", fft)
    plan = module._plan(geometry, grid, detector, poses, data, FBPConfig(), cuda=False)
    assert not plan.separable
    jax.eval_shape(
        lambda rows: module._filter_views(
            rows,
            jnp.asarray(plan.view_scale[:1]),
            jnp.asarray(plan.params[:1]),
            plan.spectrum,
            jnp.float32(plan.arc_length),
            detector=plan.detector,
            separable=plan.separable,
        ),
        jax.ShapeDtypeStruct((1, plan.detector.nv, plan.detector.nu), jnp.float32),
    )
    assert len(shapes) == 1
    assert estimates == [(len(angles), *shapes[0])]
    # The existing conservative estimate and power-of-two rounding still apply.
    capacity = (512 * 1024**2) // (16 * np.prod(shapes[0]))
    expected = len(angles) if capacity >= len(angles) else 1 << (int(capacity).bit_length() - 1)
    assert plan.batch == expected


@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("on_host", [True, False])
def test_oblique_fft_batches_preserve_independent_shifted_gaussian(monkeypatch, backend, on_host):
    """A rolled camera has no missing cone; its 2D filter recovers the full truth."""
    import jax

    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    module = importlib.import_module("tomojax.recon.fbp")
    angles = np.linspace(0, 180, 89, endpoint=False)
    grid = Grid(32, 32, 32, 1.0, 0.9, 1.1)
    detector = Detector(128, 64, 0.7, 0.9, (-1.3, 2.1))
    geometry = LaminographyGeometry(grid, detector, angles, tilt_deg=30, tilt_about="z")
    centre, sigma, amplitude = np.array([-3.4, 2.7, 1.5]), 5.0, 0.02
    u = (np.arange(detector.nu) - 63.5) * detector.du + detector.center[0]
    v = (np.arange(detector.nv) - 31.5) * detector.dv + detector.center[1]
    poses = np.asarray([geometry.pose_for_view(i) for i in range(len(angles))])
    world = np.einsum("nij,j->ni", poses[:, :3, :3], centre)
    distance2 = (u[None, None, :] - world[:, 0, None, None]) ** 2
    distance2 = distance2 + (v[None, :, None] - world[:, 2, None, None]) ** 2
    data = (amplitude * np.sqrt(2 * np.pi) * sigma * np.exp(-distance2 / (2 * sigma**2))).astype(
        np.float32
    )
    if not on_host:
        data = jnp.asarray(data)
    config = FBPConfig(backprojector=backend)
    actual = np.asarray(fbp(geometry, grid, detector, data, config=config))
    # An odd view count exercises the final overlapping batch in both the
    # host-streamed and device-loop paths, without changing quadrature/filtering.
    plan = module._plan(geometry, grid, detector, jnp.asarray(poses), data, config, cuda=False)
    bytes_per_view = 8 * module._fft_length(plan.detector.nv) * 2 * (plan.spectrum.shape[0] - 1)
    cache_module = importlib.import_module("tomojax.recon._fbp_pallas")
    monkeypatch.setattr(cache_module, "_cuda_l2_bytes", lambda _: bytes_per_view * 3)
    batches, original_filter = [], module._filter_views

    def filtered(rows, *args, **kwargs):
        batches.append(rows.shape[0])
        return original_filter(rows, *args, **kwargs)

    monkeypatch.setattr(module, "_filter_views", filtered)
    small_batches = np.asarray(fbp(geometry, grid, detector, data, config=config))
    assert batches and set(batches) == {2}
    np.testing.assert_allclose(actual, small_batches, rtol=5e-5, atol=2e-8)
    x = (np.arange(grid.nx) - 15.5) * grid.vx
    y = (np.arange(grid.ny) - 15.5) * grid.vy
    z = (np.arange(grid.nz) - 15.5) * grid.vz
    distance2 = (x[:, None, None] - centre[0]) ** 2 + (y[None, :, None] - centre[1]) ** 2
    distance2 = distance2 + (z[None, None, :] - centre[2]) ** 2
    truth = amplitude * np.exp(-distance2 / (2 * sigma**2))
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.008


@pytest.mark.gpu
@pytest.mark.parametrize("on_host", [True, False])
def test_fbp_cache_limit_preserves_shifted_gaussian_volume(monkeypatch, on_host) -> None:
    import jax

    if jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    module = importlib.import_module("tomojax.recon._fbp_pallas")
    angles = np.linspace(0, 180, 73, endpoint=False, dtype=np.float32)
    grid = Grid(32, 32, 3, 0.7, 0.9, 0.8, vol_center=(1.2, -0.8, 0.0))
    detector = Detector(64, 3, 0.7, 0.8, (2.2, 0.0))
    geometry = ParallelGeometry(grid, detector, angles)
    u = (np.arange(64) - 31.5) * 0.7 + 2.2
    phi = np.deg2rad(angles.astype(np.float64))
    projected_centre = -3.4 * np.cos(phi) - 2.7 * np.sin(phi)
    profile = np.sqrt(2 * np.pi) * 5.0 * np.exp(-((u - projected_centre[:, None]) ** 2) / 50)
    axial_scale = np.array([0.5, 1.0, 1.5])
    projections = (profile[:, None, :] * axial_scale[None, :, None]).astype(np.float32)
    if not on_host:
        projections = jnp.asarray(projections)
    cfg = FBPConfig(backprojector="pallas")
    monkeypatch.setattr(module, "_cuda_l2_bytes", lambda _: None)
    whole = np.asarray(fbp(geometry, grid, detector, projections, config=cfg))
    # Three filtered views fit, so a power-of-two batch has two views and a
    # partial final batch. Both host streaming and device loops must honour it.
    monkeypatch.setattr(module, "_cuda_l2_bytes", lambda _: 4 * detector.nv * detector.nu * 3)
    calls, original = [], module._backprojection_call

    def record(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_backprojection_call", record)
    cached = np.asarray(fbp(geometry, grid, detector, projections, config=cfg))
    assert 2 in calls
    np.testing.assert_allclose(cached, whole, rtol=4e-5, atol=2e-6)
    x = (np.arange(32) - 15.5) * 0.7 + 1.2
    y = (np.arange(32) - 15.5) * 0.9 - 0.8
    expected = np.exp(-((x[:, None] + 3.4) ** 2 + (y[None, :] - 2.7) ** 2) / 50)
    expected = expected[:, :, None] * axial_scale
    assert np.linalg.norm(cached - expected) / np.linalg.norm(expected) < 0.02


@pytest.mark.parametrize("nu", [1, 7, 16, 31])
@pytest.mark.parametrize("du", [0.4, 1.0, 2.5])
def test_ramp_filter_matches_linear_convolution(nu: int, du: float) -> None:
    """The finite detector must not wrap the last pixel onto the first."""
    rows = np.random.default_rng(42).normal(size=(3, nu)).astype(np.float32)
    offsets = np.arange(nu)[:, None] - np.arange(nu)[None, :]
    kernel = np.zeros((nu, nu), dtype=np.float64)
    kernel[offsets == 0] = 1.0 / (4.0 * du)
    odd = (offsets % 2) != 0
    kernel[odd] = -1.0 / (np.pi**2 * offsets[odd] ** 2 * du)
    expected = rows @ kernel.T

    filtered = fft_filter_rows(jnp.asarray(rows), rfft_filter_array("ramp", nu, du, jnp.float32))

    np.testing.assert_allclose(filtered, expected, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("spacing", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("nu", [64, 96])
def test_fbp_reconstructs_analytic_gaussian_in_physical_units(spacing: float, nu: int) -> None:
    n, n_views = 64, 180
    sigma = 8.0 * spacing
    grid = Grid(n, n, 1, spacing, spacing, spacing)
    detector = Detector(nu, 1, spacing, spacing)
    angles = np.linspace(0.0, 180.0, n_views, endpoint=False, dtype=np.float32)
    geometry = ParallelGeometry(grid, detector, angles)
    x = (np.arange(n) - (n - 1) / 2.0) * spacing
    u = (np.arange(nu) - (nu - 1) / 2.0) * spacing
    profile = np.sqrt(2.0 * np.pi) * sigma * np.exp(-(u**2) / (2.0 * sigma**2))
    projections = jnp.asarray(np.broadcast_to(profile, (n_views, 1, nu)), dtype=jnp.float32)
    expected = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2.0 * sigma**2))

    actual = np.asarray(fbp(geometry, grid, detector, projections))[:, :, 0]

    nrmse = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert nrmse < 0.01
    np.testing.assert_allclose(actual[n // 2, n // 2], expected[n // 2, n // 2], rtol=0.005)


def _quarter_turned(geometry, detector, data, sign):
    """The same physical rays and pixels in a quarter-turned camera frame."""

    class TurnedGeometry(ParallelGeometry):
        def pose_for_view(self, i):
            pose = np.asarray(geometry.pose_for_view(i))
            return np.stack([sign * pose[2], pose[1], -sign * pose[0], pose[3]])

    turned_detector = Detector(
        detector.nv,
        detector.nu,
        detector.dv,
        detector.du,
        (sign * detector.center[1], -sign * detector.center[0]),
    )
    turned_data = data.swapaxes(1, 2)
    turned_data = turned_data[:, ::-1, :] if sign == 1 else turned_data[:, :, ::-1]
    return (
        TurnedGeometry(geometry.grid, turned_detector, geometry.angles),
        turned_detector,
        turned_data,
    )


@pytest.mark.parametrize("backend", ["jax", "host", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("filter_name", ["ramp", "hann", "shepp-logan"])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("device_input", [False, True])
def test_fbp_is_invariant_to_quarter_turned_detector_axes(
    backend, filter_name, sign, device_input
) -> None:
    """An exact camera-coordinate change cannot change the filter or physical volume."""
    grid = Grid(12, 10, 3, 0.7, 1.1, 0.9, vol_center=(1.2, -0.7, 0.3))
    detector = Detector(20, 7, 0.8, 1.3, center=(0.27, -0.31))
    angles = [0.0, 31.0, 75.0, 113.0, 150.0, 180.0]
    geometry = ParallelGeometry(grid, detector, angles)
    data = np.random.default_rng(916).normal(size=(6, 7, 20)).astype(np.float32)
    if device_input:
        if backend == "host":
            pytest.skip("fbp_host accepts host storage only")
        data = jnp.asarray(data)
    turned, turned_detector, turned_data = _quarter_turned(geometry, detector, data, sign)
    config = FBPConfig(filter=filter_name, backprojector="jax")
    expected = fbp(geometry, grid, detector, data, config=config)
    if backend == "host":
        actual = fbp_host(
            turned,
            grid,
            turned_detector,
            turned_data,
            config=FBPHostConfig(filter=filter_name, slices_per_batch=5, views_per_batch=4),
        )
    else:
        actual = fbp(
            turned,
            grid,
            turned_detector,
            turned_data,
            config=FBPConfig(filter=filter_name, backprojector=backend),
        )
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-6)


@pytest.mark.parametrize("backend", ["jax", "host", pytest.param("pallas", marks=pytest.mark.gpu)])
def test_quarter_turned_detector_recovers_shifted_anisotropic_gaussian(backend) -> None:
    grid, detector = Grid(24, 24, 5, 0.7, 1.1, 0.9), Detector(48, 17, 0.7, 0.9, (0.4, -0.7))
    angles = np.arange(0.0, 180.0, 2.0)
    geometry = ParallelGeometry(grid, detector, angles)
    phi = np.deg2rad(angles)
    cx, cy, cz, sx, sy, sz = 1.3, -1.7, 0.2, 3.0, 4.0, 2.0
    sigma = np.sqrt(sx**2 * np.cos(phi) ** 2 + sy**2 * np.sin(phi) ** 2)
    centre = cx * np.cos(phi) - cy * np.sin(phi)
    u = (np.arange(48) - 23.5) * detector.du + detector.center[0]
    v = (np.arange(17) - 8) * detector.dv + detector.center[1]
    data = (
        (np.sqrt(2 * np.pi) * sx * sy / sigma)[:, None, None]
        * np.exp(-0.5 * ((u[None, None, :] - centre[:, None, None]) / sigma[:, None, None]) ** 2)
        * np.exp(-0.5 * ((v[None, :, None] - cz) / sz) ** 2)
    ).astype(np.float32)
    turned, turned_detector, turned_data = _quarter_turned(geometry, detector, data, 1)
    actual = (
        fbp_host(turned, grid, turned_detector, turned_data)
        if backend == "host"
        else fbp(
            turned, grid, turned_detector, turned_data, config=FBPConfig(backprojector=backend)
        )
    )
    x, y, z = [(np.arange(n) - (n - 1) / 2) * step for n, step in [(24, 0.7), (24, 1.1), (5, 0.9)]]
    truth = np.exp(
        -0.5 * ((x[:, None, None] - cx) / sx) ** 2
        - 0.5 * ((y[None, :, None] - cy) / sy) ** 2
        - 0.5 * ((z[None, None, :] - cz) / sz) ** 2
    )
    assert np.linalg.norm(np.asarray(actual) - truth) / np.linalg.norm(truth) < 0.035


def test_explicit_scale_keeps_u_only_filter_on_quarter_turned_camera() -> None:
    grid, detector = Grid(7, 6, 3, 0.7, 1.1, 0.9), Detector(20, 7, 0.8, 1.3, (0.27, -0.31))
    geometry = ParallelGeometry(grid, detector, [0.0, 45.0, 90.0, 135.0])
    data = np.random.default_rng(91).normal(size=(4, 7, 20)).astype(np.float32)
    turned, turned_detector, turned_data = _quarter_turned(geometry, detector, data, 1)
    scale = np.pi / 4
    poses = jnp.asarray([turned.pose_for_view(i) for i in range(4)])
    filtered = fft_filter_rows(
        jnp.asarray(turned_data * scale),
        rfft_filter_array("hann", turned_detector.nu, turned_detector.du, jnp.float32),
    )
    expected = _backproject_voxels_jax(poses, filtered, grid=grid, detector=turned_detector)
    actual = fbp(
        turned,
        grid,
        turned_detector,
        turned_data,
        config=FBPConfig(filter="hann", scale=scale, backprojector="jax"),
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-6)


def _rolled_camera(geometry, degrees):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))

    class Camera(ParallelGeometry):
        def pose_for_view(self, i):
            pose = np.asarray(geometry.pose_for_view(i))
            return np.stack(
                [c * pose[0] + s * pose[2], pose[1], -s * pose[0] + c * pose[2], pose[3]]
            )

    return Camera(geometry.grid, geometry.detector, geometry.angles)


def _reflected_parallel_frame(geometry, detector, data):
    """A proper pi camera rotation: same u, opposite v and opposite ray."""

    class Reflected(ParallelGeometry):
        def pose_for_view(self, i):
            pose = np.asarray(geometry.pose_for_view(i))
            return np.stack([pose[0], -pose[1], -pose[2], pose[3]])

    reflected_detector = Detector(
        detector.nu,
        detector.nv,
        detector.du,
        detector.dv,
        (detector.center[0], -detector.center[1]),
    )
    return (
        Reflected(geometry.grid, reflected_detector, geometry.angles),
        reflected_detector,
        data[:, ::-1, :],
    )


@pytest.mark.parametrize("backend", ["jax", "host", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("filter_name", ["ramp", "hann", "shepp-logan"])
@pytest.mark.parametrize("tilt", [0.0, 30.0])
def test_oblique_fbp_is_invariant_to_reflected_parallel_camera_frame(backend, filter_name, tilt):
    """Nyquist aliases cannot pick a physical v sign from a coordinate convention."""
    grid = Grid(15, 13, 7, 0.8, 1.2, 0.75, vol_center=(0.3, -0.4, 0.2))
    detector = Detector(20, 14, 0.9, 1.3, center=(0.2, -0.1))
    angles = [0.0, 21.0, 57.0, 88.0, 123.0, 160.0, 180.0]
    base = (
        ParallelGeometry(grid, detector, angles)
        if tilt == 0
        else LaminographyGeometry(grid, detector, angles, tilt_deg=tilt)
    )
    geometry = _rolled_camera(base, 40.0)
    data = (0.3 + 0.05 * np.random.default_rng(34).normal(size=(7, 14, 20))).astype(np.float32)
    reflected, reflected_detector, reflected_data = _reflected_parallel_frame(
        geometry, detector, data
    )
    expected = np.asarray(
        fbp(
            geometry,
            grid,
            detector,
            data,
            config=FBPConfig(filter=filter_name, backprojector="jax"),
        )
    )
    if backend == "host":
        actual = fbp_host(
            reflected,
            grid,
            reflected_detector,
            reflected_data,
            config=FBPHostConfig(filter=filter_name, slices_per_batch=4, views_per_batch=3),
        )
    else:
        actual = fbp(
            reflected,
            grid,
            reflected_detector,
            reflected_data,
            config=FBPConfig(filter=filter_name, backprojector=backend),
        )
    assert np.linalg.norm(np.asarray(actual) - expected) / np.linalg.norm(expected) < 1e-5
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-6)


@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("counts", [None, 10000])
def test_oblique_fbp_preserves_independent_noisy_gaussian_accuracy(backend, counts):
    grid, detector = Grid(24, 24, 24, 1, 1, 1), Detector(48, 48, 1, 1, (0.2, -0.1))
    angles = np.arange(0.0, 180.0, 2.0)
    geometry = _rolled_camera(ParallelGeometry(grid, detector, angles), 40.0)
    mean, precision = np.array([2.3, -1.2, 1.4]), np.diag(1 / np.array([4.0, 5.0, 3.0]) ** 2)
    u = np.arange(48) - 23.5 + detector.center[0]
    v = np.arange(48) - 23.5 + detector.center[1]
    points = np.stack(np.broadcast_arrays(u[None, :], np.zeros((48, 48)), v[:, None]))
    profiles = []
    for i in range(len(angles)):
        pose = np.asarray(geometry.pose_for_view(i))
        ray = pose[1, :3]
        delta = np.einsum("ji,jvu->ivu", pose[:3, :3], points) - mean[:, None, None]
        a = ray @ precision @ ray
        b = np.einsum("i,ij,jvu->vu", ray, precision, delta)
        c = np.einsum("ivu,ij,jvu->vu", delta, precision, delta)
        profiles.append(0.04 * np.sqrt(2 * np.pi / a) * np.exp(-0.5 * (c - b**2 / a)))
    data = np.asarray(profiles)
    if counts is not None:
        data = -np.log(
            np.maximum(np.random.default_rng(119).poisson(counts * np.exp(-data)), 1) / counts
        )
    coordinate = np.arange(24) - 11.5
    xyz = np.stack(np.meshgrid(coordinate, coordinate, coordinate, indexing="ij"))
    delta = xyz - mean[:, None, None, None]
    truth = 0.04 * np.exp(-0.5 * np.einsum("ixyz,ij,jxyz->xyz", delta, precision, delta))
    actual = np.asarray(
        fbp(
            geometry,
            grid,
            detector,
            data.astype(np.float32),
            config=FBPConfig(backprojector=backend),
        )
    )
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < (
        0.025 if counts is None else 0.10
    )


def _full_turn_filtered_reference(data, params, spectrum):
    """Every nonzero measured frequency has two roots in a complete circle."""
    n = 2 * (len(spectrum) - 1)
    fu = np.fft.rfftfreq(n)[None, None, :]
    fv = np.fft.fftfreq(n)[None, :, None]
    tu, tv, weight = (params[:, i, None, None].astype(np.float64) for i in (0, 1, 5))
    window = spectrum / np.maximum(np.abs(fu), spectrum[0])

    def response(vertical):
        return (
            np.where(
                (fu == 0) & (vertical == 0),
                np.hypot(tu, tv) * spectrum[0],
                np.abs(tu * fu + tv * vertical) * window,
            )
            * weight
            / 2
        )

    factor = response(fv)
    face = slice(n // 2, n // 2 + 1)
    factor[:, face] = 0.5 * (factor[:, face] + response(-fv[:, face]))
    factor[:, :, -1] = 0.5 * (factor[:, :, -1] + factor[:, (-np.arange(n)) % n, -1])
    return np.fft.irfft2(np.fft.rfft2(data, s=(n, n)) * factor, s=(n, n))[:, :64, :64]


@pytest.mark.parametrize("backend", ["jax", "host", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("filter_name", ["ramp", "hann", "shepp-logan"])
@pytest.mark.parametrize("n_views", [6, 18, 90])
def test_full_turn_oblique_fbp_keeps_both_roots_at_the_circular_seam(backend, filter_name, n_views):
    from scipy.ndimage import map_coordinates

    grid = Grid(11, 9, 7, 0.8, 1.2, 0.75, vol_center=(0.3, -0.4, 0.2))
    detector = Detector(64, 64, 1, 1, center=(0.2, -0.1))
    angles = np.linspace(0, 360, n_views, endpoint=False)
    geometry = _rolled_camera(LaminographyGeometry(grid, detector, angles, tilt_deg=30), 45)
    poses = np.asarray([geometry.pose_for_view(i) for i in range(n_views)], dtype=np.float32)
    weights = fbp_weights(poses)
    assert weights.full_turn and not weights.separable
    data = (0.3 + 0.05 * np.random.default_rng(2).normal(size=(n_views, 64, 64))).astype(np.float32)
    spectrum = np.asarray(rfft_filter_array(filter_name, 64, 1, jnp.float32), dtype=np.float64)
    filtered = _full_turn_filtered_reference(data, weights.params, spectrum)
    coordinate = [
        (np.arange(n) - (n - 1) / 2) * spacing + centre
        for n, spacing, centre in zip(
            (grid.nx, grid.ny, grid.nz), (grid.vx, grid.vy, grid.vz), grid.vol_center, strict=True
        )
    ]
    xyz = np.stack(np.meshgrid(*coordinate, indexing="ij"))
    expected = np.zeros((grid.nx, grid.ny, grid.nz))
    for pose, image in zip(poses, filtered, strict=True):
        world = np.einsum("ij,jxyz->ixyz", pose[:3, :3].astype(np.float64), xyz)
        world += pose[:3, 3, None, None, None]
        uv = np.stack([world[2] - detector.center[1], world[0] - detector.center[0]]) + 31.5
        expected += map_coordinates(image, uv, order=1, mode="grid-constant")
    if backend == "host":
        actual = fbp_host(
            geometry,
            grid,
            detector,
            data,
            config=FBPHostConfig(filter=filter_name, slices_per_batch=4, views_per_batch=5),
        )
    else:
        actual = fbp(
            geometry,
            grid,
            detector,
            data,
            config=FBPConfig(filter=filter_name, backprojector=backend),
        )
    assert np.linalg.norm(np.asarray(actual) - expected) / np.linalg.norm(expected) < 1e-5
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-6)


@pytest.mark.parametrize(
    "backend",
    [
        "jax",
        pytest.param("pallas", marks=pytest.mark.gpu),
        pytest.param("helper", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("nu", [4, 5])
@pytest.mark.parametrize("batch", [1, 2, 7])
def test_fbp_keeps_filtered_tails_at_shifted_volume_corners(
    monkeypatch, backend: str, nu: int, batch: int
) -> None:
    """Compare all output voxels with an infinite discrete-convolution sum."""
    import importlib

    import jax

    if backend != "jax" and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    module = importlib.import_module("tomojax.recon.fbp")
    monkeypatch.setattr(module, "_parallel_filter_batch_size", lambda *_: batch)
    grid = Grid(9, 7, 1, 0.8, 1.2, 1.0, vol_center=(1.7, -2.3, 0.0))
    detector = Detector(nu, 1, 0.7, 1.0, (0.27, 0.0))
    angles = np.array([0.0, 31.0, 117.0], dtype=np.float32)
    geometry = ParallelGeometry(grid, detector, angles)
    data = np.random.default_rng(901).uniform(0.1, 2.0, size=(3, 1, nu)).astype(np.float32)
    x = (np.arange(9) - 4) * 0.8 + 1.7
    y = (np.arange(7) - 3) * 1.2 - 2.3
    # Midpoint angular quadrature over the scanned arc: end views cover half a gap beyond.
    weights = np.deg2rad([31.0, 58.5, 86.0])
    if backend == "helper":
        weights = np.full(3, np.pi / len(angles))
    expected = np.zeros((9, 7), dtype=np.float64)
    outside_samples = 0
    for angle, row, view_weight in zip(angles, data[:, 0], weights, strict=True):
        radians = np.deg2rad(float(angle))
        u = np.cos(radians) * x[:, None] - np.sin(radians) * y[None, :]
        if backend == "helper":
            u += 2.2
        pixel = (u - 0.27) / 0.7 + (nu - 1) / 2
        lower = np.floor(pixel).astype(int)
        fraction = pixel - lower
        outside_samples += np.count_nonzero((pixel < 0) | (pixel > nu - 1))
        for index, weight in [(lower, 1 - fraction), (lower + 1, fraction)]:
            offset = index[..., None] - np.arange(nu)
            kernel = np.zeros_like(offset, dtype=np.float64)
            kernel[offset == 0] = 0.25 / 0.7
            odd = offset % 2 != 0
            kernel[odd] = -1 / (np.pi**2 * offset[odd] ** 2 * 0.7)
            expected += view_weight * weight * np.sum(kernel * row, axis=-1)
    assert outside_samples > expected.size
    if backend == "helper":
        from tomojax.recon.api import run_parallel_fbp_direct_pallas

        poses = np.array([geometry.pose_for_view(i) for i in range(len(angles))], np.float32)
        poses[:, 0, 3] = 2.2
        actual = np.asarray(
            run_parallel_fbp_direct_pallas(poses, data, grid=grid, detector=detector, filter="ramp")
        )[:, :, 0] * (np.pi / len(angles))
    else:
        actual = np.asarray(
            fbp(geometry, grid, detector, data, config=FBPConfig(backprojector=backend))
        )[:, :, 0]
    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=2e-6)


@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("batch", [1, 3, 8])
@pytest.mark.parametrize("separable", [True, False])
def test_streamed_fbp_matches_whole_stack_with_partial_batches(
    backend: str, batch: int, separable: bool
) -> None:
    import jax

    # check-public-imports: allow-private
    from tomojax.recon.fbp import _run_fbp_streamed

    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    grid = Grid(7, 6, 3, 0.8, 1.2, 0.7)
    detector = Detector(13, 9, 0.9, 1.0, (0.17, 0.21))
    geometry = ParallelGeometry(grid, detector, [0.0, 31.0, 78.0, 117.0, 161.0])
    poses = jnp.array([geometry.pose_for_view(i) for i in range(5)], dtype=jnp.float32)
    rng = np.random.default_rng(384)
    rows = jnp.array(rng.normal(size=(5, 5, 9)), dtype=jnp.float32)
    scale = jnp.array(rng.uniform(0.5, 1.5, size=5), dtype=jnp.float32)
    params = jnp.array(
        np.column_stack([rng.uniform(0.5, 1, (5, 4)), np.arange(5.0), rng.uniform(0.1, 0.3, 5)]),
        dtype=jnp.float32,
    )
    ramp = rfft_filter_array("ramp", detector.nu, detector.du, jnp.float32)

    def run(batch_size: int, kernel: str) -> jnp.ndarray:
        return _run_fbp_streamed(
            poses,
            rows,
            scale,
            params,
            ramp,
            jnp.float32(3.5),
            grid=grid,
            detector=detector,
            backend=kernel,
            batch_size=batch_size,
            z_mode="general",
            separable=separable,
        )

    np.testing.assert_allclose(run(batch, backend), run(5, "jax"), rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("use_explicit_detector_grid", [False, True])
@pytest.mark.parametrize("axial_pitch_ratio", [0.75, 1.0, 2.0])
def test_fbp_is_invariant_to_change_of_length_units(
    use_explicit_detector_grid: bool, axial_pitch_ratio: float
) -> None:
    """Scaling all lengths and line integrals leaves attenuation values unchanged."""
    angles = np.asarray([0.0, 37.0, 81.0, 127.0], dtype=np.float32)
    projections = np.random.default_rng(17).normal(size=(4, 7, 9)).astype(np.float32)
    reconstructions = []
    for scale in [1.0, 0.1, 1.3, 10.0]:
        grid = Grid(7, 8, 3, 0.8 * scale, 1.1 * scale, axial_pitch_ratio * 1.3 * scale)
        detector = Detector(9, 7, 0.9 * scale, 1.3 * scale)
        geometry = ParallelGeometry(grid, detector, angles)
        det_grid = None
        if use_explicit_detector_grid:
            u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * detector.du
            v = (np.arange(detector.nv) - (detector.nv - 1) / 2) * detector.dv
            det_grid = (
                jnp.asarray(np.tile(u, detector.nv)),
                jnp.asarray(np.repeat(v, detector.nu)),
            )
        reconstructions.append(
            fbp(
                geometry,
                grid,
                detector,
                jnp.asarray(projections * scale),
                config=FBPConfig(views_per_batch=2),
                det_grid=det_grid,
            )
        )
    for result in reconstructions[1:]:
        np.testing.assert_allclose(result, reconstructions[0], rtol=5e-5, atol=5e-6)


@pytest.mark.parametrize("interpret", [True, pytest.param(False, marks=pytest.mark.gpu)])
@pytest.mark.parametrize("z_mode", ["general", "fixed", "integer"])
@pytest.mark.parametrize("n_views", [5, 129])
def test_voxel_pallas_fbp_matches_jax_on_partial_blocks(
    interpret: bool, z_mode: str, n_views: int
) -> None:
    import jax

    if not interpret and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    grid = Grid(7, 6, 3, 0.8, 1.2, 1.0 if z_mode == "integer" else 0.7)
    detector = Detector(9, 5, 0.9, 1.0, (0.17, 0.0 if z_mode == "integer" else 0.21))
    angles = (
        [0.0, 31.0, 78.0, 117.0, 161.0]
        if n_views == 5
        else np.linspace(13, 187, n_views, endpoint=False)
    )
    geometry = ParallelGeometry(grid, detector, angles)
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(n_views)], dtype=jnp.float32)
    if z_mode == "fixed":
        poses = poses.at[:, 2, 3].set(0.375)
    projections = jnp.asarray(
        np.random.default_rng(18).normal(size=(n_views, 5, 9)), dtype=jnp.float32
    )
    ramp = rfft_filter_array("ramp", 9, detector.du, jnp.float32)
    expected = jax.jit(
        lambda y: _backproject_voxels_jax(
            poses, fft_filter_rows(y, ramp), grid=grid, detector=detector
        )
    )(projections)
    # Both automatic long/short-loop tiles and an explicit small override must
    # retain partial-block masks and fractional/integer-row interpolation.
    for block in [None, 64]:
        actual = backproject_filtered_pallas(
            poses,
            fft_filter_rows(projections, ramp),
            grid=grid,
            detector=detector,
            z_mode=z_mode,
            interpret=interpret,
            block_size=block,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("interpret", [True, pytest.param(False, marks=pytest.mark.gpu)])
def test_fixed_rows_preserve_a_common_nonbinary_translation(interpret: bool) -> None:
    """Hoisting agrees with the generic kernel even near cancellation after many views."""
    import jax

    if not interpret and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    grid = Grid(7, 6, 3, 0.8, 1.2, 0.7)
    detector = Detector(9, 5, 0.9, 1.0, (0.17, 0.21))
    geometry = ParallelGeometry(grid, detector, np.linspace(13, 187, 129, endpoint=False))
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(129)], dtype=jnp.float32)
    poses = poses.at[:, 2, 3].set(0.31)
    rows = jnp.asarray(np.random.default_rng(18).normal(size=(129, 5, 9)), dtype=jnp.float32)
    filtered = fft_filter_rows(rows, rfft_filter_array("ramp", 9, detector.du, jnp.float32))
    for block in [None, 64]:
        options = dict(grid=grid, detector=detector, interpret=interpret, block_size=block)
        expected = backproject_filtered_pallas(poses, filtered, z_mode="general", **options)
        actual = backproject_filtered_pallas(poses, filtered, z_mode="fixed", **options)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_explicit_pallas_fbp_requires_cuda_arrays() -> None:
    import jax

    if jax.default_backend() == "gpu":
        pytest.skip("checks the CPU rejection")
    grid = Grid(4, 4, 3, 1.0, 1.0, 1.0)
    detector = Detector(6, 3, 1.0, 1.0)
    geometry = LaminographyGeometry(grid, detector, [0.0, 90.0], tilt_deg=30)
    with pytest.raises(ValueError, match="requires CUDA"):
        fbp(
            geometry,
            grid,
            detector,
            jnp.ones((2, 3, 6)),
            config=FBPConfig(backprojector="pallas"),
        )


@pytest.mark.parametrize(
    ("n_views", "shape", "detector_shape", "override", "expected"),
    [
        (1, (7, 6, 3), (9, 5), None, 128),
        (32, (768, 768, 11), (1104, 12), None, 128),
        (128, (7, 6, 3), (9, 5), None, 128),
        (32, (256, 256, 128), (1024, 256), None, 128),
        (32, (256, 256, 128), (1025, 256), None, 512),
        (32, (4096, 4096, 2), (9, 5), None, 2048),
        (129, (7, 6, 3), (9, 5), None, 128),
        (129, (7, 6, 3), (512, 512), None, 512),
        (129, (512, 512, 11), (767, 11), None, 128),
        (129, (768, 768, 17), (1089, 17), None, 128),
        (129, (4096, 4096, 2), (9, 5), None, 2048),
        (129, (7, 6, 3), (9, 5), 64, 64),
    ],
)
def test_automatic_fbp_tiles_bound_both_input_and_output_working_sets(
    monkeypatch, n_views, shape, detector_shape, override, expected
):
    import jax

    # check-public-imports: allow-private
    from tomojax.recon import _fbp_pallas as implementation

    grid = Grid(*shape, 1, 1, 1)
    detector = Detector(*detector_shape, 1, 1)
    chosen = []

    def call(*args, **kwargs):
        chosen.append(args[3])
        return lambda *_: jnp.zeros((grid.nx * grid.ny * grid.nz,), jnp.float32)

    monkeypatch.setattr(implementation, "_backprojection_call", call)
    # Trace only: large data/output shapes must not need large allocations just
    # to verify the static launch policy and an explicit caller override.
    jax.eval_shape(
        lambda poses, rows: implementation.backproject_filtered_pallas(
            poses,
            rows,
            grid=grid,
            detector=detector,
            z_mode="integer",
            block_size=override,
        ),
        jax.ShapeDtypeStruct((n_views, 4, 4), jnp.float32),
        jax.ShapeDtypeStruct((n_views, detector.nv, detector.nu), jnp.float32),
    )
    assert chosen == [expected]


@pytest.mark.gpu
@pytest.mark.parametrize("vz", [1.0, 0.9])
def test_medium_working_set_fbp_recovers_independent_off_axis_gaussian(vz):
    import jax

    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid = Grid(512, 512, 11, 1.0, 1.0, vz)
    detector = Detector(767, 11, 1.0, 1.0, (0.23, 0.0))
    angles = np.linspace(7, 187, 181, endpoint=False)
    geometry = ParallelGeometry(grid, detector, angles)
    x = np.arange(grid.nx) - (grid.nx - 1) / 2
    y = np.arange(grid.ny) - (grid.ny - 1) / 2
    z = (np.arange(grid.nz) - (grid.nz - 1) / 2) * vz
    u = detector.center[0] + np.arange(detector.nu) - (detector.nu - 1) / 2
    v = np.arange(detector.nv) - (detector.nv - 1) / 2
    theta = np.deg2rad(angles)
    sx, sy, sz = 8.0, 10.0, 4.0
    mean = 100 * np.cos(theta) + 70 * np.sin(theta)
    variance = (sx * np.cos(theta)) ** 2 + (sy * np.sin(theta)) ** 2
    profile = np.sqrt(2 * np.pi) * sx * sy / np.sqrt(variance[:, None])
    profile = profile * np.exp(-0.5 * (u[None, :] - mean[:, None]) ** 2 / variance[:, None])
    data = (profile[:, None, :] * np.exp(-0.5 * (v / sz) ** 2)[None, :, None]).astype(np.float32)
    truth = np.exp(-0.5 * (((x[:, None] - 100) / sx) ** 2 + ((y[None, :] + 70) / sy) ** 2))
    truth = truth[:, :, None] * np.exp(-0.5 * (z / sz) ** 2)[None, None, :]
    actual = np.asarray(fbp(geometry, grid, detector, data))
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.008


@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("z_mode", ["integer", "fixed", "general"])
@pytest.mark.parametrize("shape", [(10, 6), (11, 7)])
@pytest.mark.parametrize("padding", [(1024, 0), (0, 1024), (512, 512)])
def test_fbp_interpolation_preserves_physical_fractions_under_zero_padding(
    backend, z_mode, shape, padding
):
    """Integer detector origins must not round away physical subpixel positions."""
    import jax
    from scipy.ndimage import map_coordinates

    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    nu, nv = shape
    # Dyadic grid positions isolate interpolation from coordinate-grid rounding.
    centre_z = (0.5 if nv % 2 == 0 else 0.0) if z_mode == "integer" else 0.125
    grid = Grid(
        13,
        11,
        3,
        0.5,
        1.25,
        1.0 if z_mode == "integer" else 0.75,
        vol_center=(0.25, -0.125, centre_z),
    )
    detector = Detector(nu, nv, 0.875, 1.0, (0.375, 0.0 if z_mode == "integer" else 0.125))
    angles = [7.0, 31.0, 78.0, 117.0, 161.0]
    geometry = (
        ParallelGeometry(grid, detector, angles)
        if z_mode != "general"
        else LaminographyGeometry(grid, detector, angles, tilt_deg=17.0)
    )
    poses = np.asarray([geometry.pose_for_view(i) for i in range(5)], np.float32)
    if z_mode == "fixed":
        poses[:, 2, 3] = 0.375
    filtered = np.random.default_rng(739).normal(size=(5, nv, nu)).astype(np.float32)
    coordinates = [
        (np.arange(n) - (n - 1) / 2) * spacing + centre
        for n, spacing, centre in zip(
            (grid.nx, grid.ny, grid.nz), (grid.vx, grid.vy, grid.vz), grid.vol_center, strict=True
        )
    ]
    xyz = np.stack(np.meshgrid(*coordinates, indexing="ij"))
    expected = np.zeros((grid.nx, grid.ny, grid.nz))
    for pose, image in zip(poses, filtered, strict=True):
        world = np.einsum("ij,jxyz->ixyz", pose[:3, :3].astype(np.float64), xyz)
        world += pose[:3, 3, None, None, None]
        u = (world[0] - detector.center[0]) / detector.du + (nu - 1) / 2
        v = (world[2] - detector.center[1]) / detector.dv + (nv - 1) / 2
        expected += map_coordinates(
            image.astype(np.float64), np.stack([v, u]), order=1, mode="grid-constant"
        )

    def run(pad_u, pad_v):
        expanded = Detector(
            nu + 2 * pad_u, nv + 2 * pad_v, detector.du, detector.dv, detector.center
        )
        rows = jnp.asarray(np.pad(filtered, ((0, 0), (pad_v, pad_v), (pad_u, pad_u))))

        def call(p, y):
            if backend == "jax":
                return _backproject_voxels_jax(p, y, grid=grid, detector=expanded)
            return backproject_filtered_pallas(p, y, grid=grid, detector=expanded, z_mode=z_mode)

        return np.asarray(jax.jit(call)(jnp.asarray(poses), rows))

    original, padded = run(0, 0), run(*padding)
    np.testing.assert_allclose(padded, expected, rtol=4e-5, atol=2e-6)
    np.testing.assert_array_equal(padded, original)


@pytest.mark.parametrize("bad", ["angle", "origin", "detector", "scale"])
@pytest.mark.parametrize("host", [False, True])
def test_fbp_rejects_nonfinite_geometry_before_any_work(bad, host) -> None:
    nan = float("nan")
    grid = Grid(3, 2, 1, 1, 1, 1, vol_origin=(nan, 0, 0) if bad == "origin" else None)
    detector = Detector(5, 1, 1, 1, (nan, 0) if bad == "detector" else (0, 0))
    geometry = ParallelGeometry(grid, detector, [0, nan if bad == "angle" else 60, 120])
    scale = nan if bad == "scale" else None
    config = FBPHostConfig(scale=scale) if host else FBPConfig(scale=scale)
    run = fbp_host if host else fbp
    projections = np.ones((3, 1, 5), np.float32) if host else jnp.ones((3, 1, 5))
    with pytest.raises(ValueError, match="must be finite"):
        run(geometry, grid, detector, projections, config=config)


def _ramp(offset: np.ndarray, du: float) -> np.ndarray:
    kernel = np.zeros(offset.shape, dtype=np.float64)
    kernel[offset == 0] = 0.25 / du
    odd = offset % 2 != 0
    kernel[odd] = -1 / (np.pi**2 * offset[odd] ** 2 * du)
    return kernel


def _tail(index: np.ndarray, du: float) -> np.ndarray:
    """Independent infinite odd-lag sum, using the trigamma identity."""
    distance = np.where(index < 0, -index - 1, index)
    first = distance + 1
    first += first % 2 == 0
    positive = -polygamma(1, first / 2) / (4 * np.pi**2 * du)
    return np.where(index < 0, -positive, positive)


@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("pitch", [(0.8, 0.7, 0.9, 1.8), (0.6, 0.5, 0.7, 1.2)])
def test_fbp_preserves_shifted_fine_annulus(backend, pitch) -> None:
    """A true annulus is not a detector ring: use an independent Gaussian-circle oracle."""
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    du, vx, vy, sigma = pitch
    grid = Grid(64, 62, 1, vx, vy, 1, vol_center=(0.3, -0.7, 0))
    detector = Detector(102, 1, du, 1, (0.43, 0))
    angles = np.linspace(0, 180, 361, endpoint=False)
    geometry = ParallelGeometry(grid, detector, angles)
    radius, amplitude, x0, y0 = 12.6, 0.03, 3.2, -1.9
    origin = grid_volume_origin(grid)
    x = origin[0] + np.arange(grid.nx) * vx
    y = origin[1] + np.arange(grid.ny) * vy
    rho = np.hypot(x[:, None] - x0, y[None, :] - y0)
    truth = amplitude * np.exp(-0.5 * ((rho - radius) / sigma) ** 2)
    truth *= i0e(rho * radius / sigma**2)
    u = (np.arange(detector.nu) - (detector.nu - 1) / 2) * du + detector.center[0]
    theta = np.deg2rad(angles)
    position = u[None] - (x0 * np.cos(theta) - y0 * np.sin(theta))[:, None]
    # Averaging the exact Radon profiles of Gaussians placed on a circle gives
    # the projections; the independent continuous-circle image uses I0 above.
    means = radius * np.cos(np.arange(512) * 2 * np.pi / 512)
    data = np.zeros(position.shape, np.float64)
    for start in range(0, len(means), 64):
        distance = (position[..., None] - means[None, None, start : start + 64]) / sigma
        data += np.exp(-0.5 * distance**2).sum(axis=-1)
    data *= amplitude * np.sqrt(2 * np.pi) * sigma / len(means)
    actual = np.asarray(
        fbp(
            geometry,
            grid,
            detector,
            data[:, None].astype(np.float32),
            config=FBPConfig(backprojector=backend),
        )
    )[:, :, 0]
    assert np.linalg.norm(actual - truth) / np.linalg.norm(truth) < 0.035


@pytest.mark.parametrize("filter_name", ["ramp", "hann"])
@pytest.mark.parametrize("nu", [1, 7, 12])
@pytest.mark.parametrize("pad", [0, 5, 47])
@pytest.mark.parametrize("du", [0.4, 1.7])
def test_endpoint_continuation_matches_independent_infinite_convolution(
    filter_name: str, nu: int, pad: int, du: float
) -> None:
    data = np.random.default_rng(901).uniform(0.1, 3, size=(3, nu)).astype(np.float32)
    index = np.arange(-pad, nu + pad)
    offsets = index[:, None] - np.arange(nu)
    kernel = _ramp(offsets, du)
    left, right = _tail(index, du), _tail(nu - 1 - index, du)
    if filter_name == "hann":
        kernel = 0.5 * kernel + 0.25 * (_ramp(offsets - 1, du) + _ramp(offsets + 1, du))
        left = 0.5 * left + 0.25 * (_tail(index - 1, du) + _tail(index + 1, du))
        right = 0.5 * right + 0.25 * (_tail(nu - 2 - index, du) + _tail(nu - index, du))
    expected = data @ kernel.T + data[:, :1] * left + data[:, -1:] * right
    rows = jnp.asarray(np.pad(data, ((0, 0), (pad, pad))))
    spectrum = rfft_filter_array(filter_name, nu + 2 * pad, du, rows.dtype)
    actual = jax.jit(lambda x: fft_filter_rows(x, spectrum, edge_width=nu))(rows)
    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=5e-7)


@pytest.mark.parametrize("filter_name", ["ramp", "hann"])
@pytest.mark.parametrize("nu", [1, 8, 9])
@pytest.mark.parametrize("pad", [0, 5])
def test_endpoint_continuation_pullback_matches_independent_dense_transpose(
    filter_name: str, nu: int, pad: int
) -> None:
    """The endpoint prior is linear, but its transpose includes both boundary tails."""
    du = 0.7
    index = np.arange(-pad, nu + pad)
    offsets = index[:, None] - np.arange(nu)
    kernel = _ramp(offsets, du)
    left, right = _tail(index, du), _tail(nu - 1 - index, du)
    if filter_name == "hann":
        kernel = 0.5 * kernel + 0.25 * (_ramp(offsets - 1, du) + _ramp(offsets + 1, du))
        left = 0.5 * left + 0.25 * (_tail(index - 1, du) + _tail(index + 1, du))
        right = 0.5 * right + 0.25 * (_tail(nu - 2 - index, du) + _tail(nu - index, du))
    kernel[:, 0] += left
    kernel[:, -1] += right
    rng = np.random.default_rng(612)
    data = rng.uniform(0.1, 2, (2, 3, nu)).astype(np.float32)
    cotangent = rng.normal(size=(2, 3, nu + 2 * pad)).astype(np.float32)
    spectrum = rfft_filter_array(filter_name, nu + 2 * pad, du, jnp.float32)

    def filtered(rows):
        padded = jnp.pad(rows, ((0, 0), (0, 0), (pad, pad)))
        return fft_filter_rows(padded, spectrum, edge_width=nu)

    def pullback(rows, weights):
        _, transpose = jax.vjp(filtered, rows)
        return transpose(weights)[0]

    actual = np.asarray(jax.jit(pullback)(jnp.asarray(data), jnp.asarray(cotangent)))
    expected = cotangent.astype(np.float64) @ kernel
    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=5e-7)
    # Every per-view/row constant is annihilated, so each transpose row sums to zero.
    np.testing.assert_allclose(actual.sum(axis=-1), 0, atol=2e-6)


@pytest.mark.parametrize("filter_name", ["ramp", "hann", "shepp-logan"])
@pytest.mark.parametrize("nu", [1, 13, 640])
def test_constant_exterior_and_measurements_have_zero_ramp_response(
    filter_name: str, nu: int
) -> None:
    rows = jnp.pad(jnp.full((3, nu), 2.3), ((0, 0), (9, 9)))
    spectrum = rfft_filter_array(filter_name, nu + 18, 0.7, rows.dtype)
    actual = fft_filter_rows(rows, spectrum, edge_width=nu)
    np.testing.assert_array_equal(actual, 0)


@pytest.mark.parametrize("filter_name", ["ramp", "hann", "shepp-logan"])
@pytest.mark.parametrize("backend", ["jax", pytest.param("pallas", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("depth", [1, 4])
def test_host_and_device_paths_keep_the_same_endpoint_assumption(
    filter_name: str, backend: str, depth: int
) -> None:
    if backend == "pallas" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid = Grid(9, 7, 5, 0.8, 1.2, 0.7, vol_center=(1.2, -0.4, 0.31))
    detector = Detector(9, 7, 0.9, 1, (0.23, -0.19))
    geometry = ParallelGeometry(grid, detector, [0, 29, 73, 118, 163])
    data = np.random.default_rng(133).uniform(0.1, 2, size=(5, 7, 9)).astype(np.float32)
    cfg = FBPConfig(filter=filter_name, padding="edge", backprojector=backend)
    expected = np.asarray(fbp(geometry, grid, detector, jnp.asarray(data), config=cfg))
    streamed = np.asarray(fbp(geometry, grid, detector, data, config=cfg))
    actual = fbp_host(
        geometry,
        grid,
        detector,
        data,
        config=FBPHostConfig(
            filter=filter_name,
            padding="edge",
            backprojector=backend,
            views_per_batch=3,
            slices_per_batch=depth,
        ),
    )
    np.testing.assert_allclose(streamed, expected, atol=3e-6, rtol=4e-5)
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=4e-5)


def test_edge_continuation_reduces_severe_truncation_error_not_just_residual() -> None:
    grid = Grid(64, 64, 1, 1, 1, 1)
    detector = Detector(32, 1, 1, 1, (-3, 0))
    angles = np.linspace(0, 180, 180, endpoint=False)
    geometry = ParallelGeometry(grid, detector, angles)
    sigma = 20
    u = np.arange(32) - 15.5 - 3
    profile = np.sqrt(2 * np.pi) * sigma * np.exp(-0.5 * (u / sigma) ** 2)
    data = np.broadcast_to(profile.astype(np.float32), (180, 1, 32))
    x = np.arange(64) - 31.5
    truth = np.exp(-0.5 * (x[:, None] ** 2 + x[None, :] ** 2) / sigma**2)
    region = np.hypot(x[:, None], x[None, :]) < 12.8
    zero = np.asarray(fbp(geometry, grid, detector, data))[:, :, 0]
    edge = np.asarray(fbp(geometry, grid, detector, data, config=FBPConfig(padding="edge")))[
        :, :, 0
    ]
    zero_error = np.linalg.norm((zero - truth)[region]) / np.linalg.norm(truth[region])
    edge_error = np.linalg.norm((edge - truth)[region]) / np.linalg.norm(truth[region])
    assert edge_error < 0.65
    assert edge_error < 0.6 * zero_error
    # This prior suppresses the discontinuity but cannot recover missing curvature.
    assert edge_error > 0.3


def test_zero_endpoint_data_do_not_change_under_edge_continuation() -> None:
    grid = Grid(13, 11, 1, 1, 1, 1)
    detector = Detector(9, 1, 1, 1, (0.3, 0))
    geometry = ParallelGeometry(grid, detector, [0, 37, 84, 123, 162])
    data = np.random.default_rng(77).uniform(0.1, 2, size=(5, 1, 9)).astype(np.float32)
    data[..., [0, -1]] = 0
    zero = fbp(geometry, grid, detector, data)
    edge = fbp(geometry, grid, detector, data, config=FBPConfig(padding="edge"))
    np.testing.assert_array_equal(edge, zero)


@pytest.mark.parametrize("host", [False, True])
def test_edge_padding_refuses_oblique_filters_instead_of_silently_using_zero(host: bool) -> None:
    grid, detector = Grid(3, 2, 1, 1, 1, 1), Detector(5, 3, 1, 1)
    geometry = LaminographyGeometry(grid, detector, [0, 37, 84, 123, 162], tilt_deg=30)
    data = np.ones((5, 3, 5), dtype=np.float32)
    cfg = FBPHostConfig(padding="edge") if host else FBPConfig(padding="edge")
    with pytest.raises(ValueError, match="separable"):
        (fbp_host if host else fbp)(geometry, grid, detector, data, config=cfg)


def test_padding_is_a_recorded_expert_setting_and_validated() -> None:
    assert FBPConfig().padding == "zero"
    assert replace(FBPConfig(), padding="edge").padding == "edge"
    with pytest.raises(ValueError, match="padding"):
        FBPConfig(padding="unknown")


@pytest.mark.gpu
@pytest.mark.parametrize("z_mode", ["integer", "fixed"])
@pytest.mark.parametrize("nz", [1, 33, 129, 257])
def test_parallel_fbp_keeps_partial_slabs_and_accumulator(z_mode, nz) -> None:
    """Compare thin/partial slabs and masked zero extension with the JAX sum."""
    import jax

    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    grid = Grid(11, 7, nz, 0.7, 1.2, 1.0 if z_mode == "integer" else 0.73)
    detector = Detector(13, 21, 0.9, 1.0, (-0.19, 0.0 if z_mode == "integer" else 0.31))
    angles = [7.0, 33.0, 81.0, 119.0, 164.0]
    geometry = ParallelGeometry(grid, detector, angles)
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(len(angles))], jnp.float32)
    if z_mode == "fixed":
        poses = poses.at[:, 2, 3].set(0.375)
    rng = np.random.default_rng(843)
    rows = jnp.asarray(rng.normal(size=(len(angles), detector.nv, detector.nu)), jnp.float32)
    initial = rng.normal(size=(grid.nx, grid.ny, grid.nz)).astype(np.float32)
    expected = (
        np.asarray(
            jax.jit(lambda y: _backproject_voxels_jax(poses, y, grid=grid, detector=detector))(rows)
        )
        + initial
    )
    for block in (128, 512):
        actual = jax.jit(
            lambda p, y, a, block=block: backproject_filtered_pallas(
                p,
                y,
                grid=grid,
                detector=detector,
                z_mode=z_mode,
                block_size=block,
                accumulate=a,
            )
        )(poses, rows, jnp.asarray(initial.copy()))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("interpret", [False, True])
def test_axial_fbp_indices_keep_affine_cuda_addresses(interpret):
    """Only interpret stores need a sentinel for padded rows aliasing a column."""
    import jax
    from jax.experimental import pallas as pl

    implementation = importlib.import_module("tomojax.recon._fbp_pallas")

    def indices(out):
        flat, *_ = implementation._voxel_indices(
            (5, 3, 3), block_size=128, z_mode="integer", interpret=interpret
        )
        out[:] = flat.reshape(-1)

    actual = np.asarray(
        pl.pallas_call(
            indices,
            out_shape=jax.ShapeDtypeStruct((128,), jnp.int32),
            grid=(1, 1),
            interpret=True,
        )()
    )
    column, row = np.arange(32)[:, None], np.arange(4)[None, :]
    expected = column * 3 + row
    if interpret:
        expected = np.where((column < 15) & (row < 3), expected, 45)
    np.testing.assert_array_equal(actual, expected.reshape(-1))


@pytest.mark.parametrize("interpret", [True, pytest.param(False, marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    ("step", "nv", "nz", "first", "pitch"),
    [(1, 6, 5, 0, 0.7), (2, 21, 7, -3, 1.3), (3, 64, 17, 7, 0.125), (4, 63, 33, -1, 0.2)],
)
@pytest.mark.parametrize(("phase", "drift"), [(0.0, 0.0), (4e-6, -1e-8), (-4e-6, 1e-8)])
def test_integer_fbp_rows_preserve_physical_alignment(
    interpret, step, nv, nz, first, pitch, phase, drift
):
    """Strided, shifted and barely aligned rows use the same nearest physical pixels."""
    from dataclasses import replace

    import jax

    if not interpret and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    center = -0.31
    grid = Grid(
        7,
        5,
        nz,
        0.7,
        0.9,
        (step + drift) * pitch,
        vol_origin=(-1.8, -2.4, (first + phase - (nv - 1) / 2) * pitch + center),
    )
    detector = Detector(65, nv, 0.9, pitch, (0.17, center))
    implementation = importlib.import_module("tomojax.recon.fbp")
    assert implementation.supports_parallel_fbp_z_integer(grid, detector)
    angles = [7.0, 33.0, 81.0, 164.0]
    geometry = ParallelGeometry(grid, detector, angles)
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(len(angles))], jnp.float32)
    rng = np.random.default_rng(16103)
    rows = jnp.asarray(rng.normal(size=(len(angles), nv, detector.nu)), jnp.float32)
    initial = rng.normal(size=(grid.nx, grid.ny, grid.nz)).astype(np.float32)
    # Independently sample the same nearest physical rows on an exact unit-pitch
    # lattice. XY coordinates and poses are unchanged; no row interpolation or
    # reconstruction fit enters this reference.
    exact_grid = replace(grid, vz=step, vol_origin=(-1.8, -2.4, first - (nv - 1) / 2))
    exact_detector = replace(detector, dv=1.0, center=(detector.center[0], 0.0))
    expected = (
        np.asarray(
            jax.jit(
                lambda y: _backproject_voxels_jax(
                    poses, y, grid=exact_grid, detector=exact_detector
                )
            )(rows)
        )
        + initial
    )
    for block in (128, 512):
        actual = jax.jit(
            lambda p, y, a, block=block: backproject_filtered_pallas(
                p,
                y,
                grid=grid,
                detector=detector,
                z_mode="integer",
                interpret=interpret,
                block_size=block,
                accumulate=a,
            )
        )(poses, rows, jnp.asarray(initial.copy()))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("z_mode", ["integer", "fixed"])
@pytest.mark.parametrize("nz", [33, 65, 129])
def test_interpret_axial_bands_preserve_long_slab_masks(z_mode, nz):
    """Exact binary pitches isolate long-row tile masks from CPU FMA differences."""
    import jax

    grid = Grid(7, 5, nz, 0.5, 0.75, 1.0 if z_mode == "integer" else 0.5)
    detector = Detector(13, 21, 0.5, 1.0, (0.25, 0.0 if z_mode == "integer" else 0.25))
    geometry = ParallelGeometry(grid, detector, [0.0, 90.0, 180.0, 270.0])
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(4)], jnp.float32)
    if z_mode == "fixed":
        poses = poses.at[:, 2, 3].set(0.375)
    rng = np.random.default_rng(8327)
    rows = jnp.asarray(rng.normal(size=(4, detector.nv, detector.nu)), jnp.float32)
    initial = rng.normal(size=(grid.nx, grid.ny, grid.nz)).astype(np.float32)
    expected = (
        np.asarray(
            jax.jit(lambda y: _backproject_voxels_jax(poses, y, grid=grid, detector=detector))(rows)
        )
        + initial
    )
    for block in (128, 512):
        actual = backproject_filtered_pallas(
            poses,
            rows,
            grid=grid,
            detector=detector,
            z_mode=z_mode,
            block_size=block,
            interpret=True,
            accumulate=jnp.asarray(initial.copy()),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("z_mode", ["integer", "fixed", "general"])
@pytest.mark.parametrize(
    ("shape", "nv", "nu", "expected"),
    [((32, 32, 3), 3, 5, 128), ((32, 32, 3), 2048, 2048, 512), ((512, 512, 65), 65, 13, 2048)],
)
def test_axial_launch_keeps_small_outputs_and_general_layout(
    monkeypatch, z_mode, shape, nv, nu, expected
):
    """Planning large inputs does not allocate them or underfill small output grids."""
    import jax

    implementation = importlib.import_module("tomojax.recon._fbp_pallas")
    grid = Grid(*shape, 1.0, 1.0, 1.0)
    detector = Detector(nu, nv, 1.0, 1.0)
    sizes = []

    def call(grid, detector, n, block_size, **options):
        sizes.append(block_size)
        return lambda *args: jnp.zeros(grid.nx * grid.ny * grid.nz, jnp.float32)

    monkeypatch.setattr(implementation, "_backprojection_call", call)
    jax.eval_shape(
        lambda p, y: backproject_filtered_pallas(p, y, grid=grid, detector=detector, z_mode=z_mode),
        jax.ShapeDtypeStruct((5, 4, 4), jnp.float32),
        jax.ShapeDtypeStruct((5, nv, nu), jnp.float32),
    )
    size = 512 if z_mode == "general" and expected == 2048 else expected
    assert sizes == [size] * (2 if shape[2] == 65 and z_mode != "general" else 1)


@pytest.mark.parametrize("z_mode", ["integer", "fixed", "general"])
@pytest.mark.parametrize("block_size", [64, 128, 512, 1024, 2048])
def test_axial_launch_warps_follow_only_large_axial_programs(monkeypatch, z_mode, block_size):
    from jax.experimental import pallas as pl

    implementation = importlib.import_module("tomojax.recon._fbp_pallas")
    warps = []

    def call(*args, **options):
        warps.append(options["compiler_params"].num_warps)

    monkeypatch.setattr(pl, "pallas_call", call)
    implementation._backprojection_call.cache_clear()
    try:
        implementation._backprojection_call(
            Grid(7, 5, 33, 1.0, 1.0, 1.0),
            Detector(13, 21, 1.0, 1.0),
            5,
            block_size,
            z_mode=z_mode,
            interpret=False,
        )
        assert warps == [8 if z_mode != "general" and block_size >= 2048 else 4]
    finally:
        implementation._backprojection_call.cache_clear()


@pytest.mark.parametrize("z_mode", ["integer", "fixed", "general"])
@pytest.mark.parametrize("nz", [16, 32, 33, 65, 80, 81, 96, 129])
@pytest.mark.parametrize("accumulate", [False, True])
def test_axial_fbp_tail_calls_keep_full_accumulator_and_global_rows(
    monkeypatch, z_mode, nz, accumulate
):
    import jax

    implementation = importlib.import_module("tomojax.recon._fbp_pallas")
    grid = Grid(7, 5, nz, 1.0, 1.0, 1.0)
    detector = Detector(13, nz, 1.0, 1.0)
    calls = []

    def call(grid, detector, n, block_size, **options):
        calls.append((options.get("z_start", 0), options.get("row_count"), options["accumulate"]))
        return lambda *operands: (
            operands[-1]
            if options["accumulate"]
            else jnp.zeros(grid.nx * grid.ny * grid.nz, jnp.float32)
        )

    monkeypatch.setattr(implementation, "_backprojection_call", call)
    poses = jax.ShapeDtypeStruct((5, 4, 4), jnp.float32)
    rows = jax.ShapeDtypeStruct((5, nz, 13), jnp.float32)
    initial = jax.ShapeDtypeStruct((grid.nx, grid.ny, nz), jnp.float32)
    result = jax.eval_shape(
        lambda p, y, a: backproject_filtered_pallas(
            p,
            y,
            grid=grid,
            detector=detector,
            z_mode=z_mode,
            accumulate=a if accumulate else None,
        ),
        poses,
        rows,
        initial,
    )
    assert result.shape == initial.shape
    if z_mode != "general" and nz in (33, 65, 80, 129):
        prefix = nz // 32 * 32
        assert calls == [(0, prefix, True), (prefix, nz - prefix, True)]
    else:
        assert calls == [(0, None, accumulate)]


def test_fbp_short_tail_preserves_independent_last_row_weak_feature():
    """The final physical row is not the short tile's local row zero."""
    grid = Grid(32, 32, 33, 0.25, 0.25, 1.0)
    detector = Detector(257, 33, 0.125, 1.0, (0.35, 0.0))
    angles = np.linspace(7, 187, 37, endpoint=False)
    geometry = ParallelGeometry(grid, detector, angles)
    u = (np.arange(257) - 128) * 0.125 + 0.35
    z = np.arange(33) - 16
    phi = np.deg2rad(angles)
    x = (np.arange(32) - 15.5) * 0.25
    xx, yy = x[:, None], x[None, :]
    background_z = 1 + 0.1 * np.cos(0.13 * z)
    feature_z = np.exp(-((z - 16) ** 2) / 4.5)

    def profile(amplitude, sigma, cx, cy):
        mean = cx * np.cos(phi) - cy * np.sin(phi)
        return (
            amplitude
            * np.sqrt(2 * np.pi)
            * sigma
            * np.exp(-((u[None, :] - mean[:, None]) ** 2) / (2 * sigma**2))
        )

    background = profile(0.02, 3.0, -0.5, 0.5)[:, None, :] * background_z[None, :, None]
    feature = profile(2e-5, 0.6, 1.125, -0.625)[:, None, :] * feature_z[None, :, None]
    truth = 2e-5 * np.exp(-((xx - 1.125) ** 2 + (yy + 0.625) ** 2) / (2 * 0.6**2))
    truth = truth[:, :, None] * feature_z
    full = np.asarray(fbp(geometry, grid, detector, (background + feature).astype(np.float32)))
    broad = np.asarray(fbp(geometry, grid, detector, background.astype(np.float32)))
    signal = full - broad
    mask = (xx - 1.125) ** 2 + (yy + 0.625) ** 2 < 1.5**2
    assert np.linalg.norm((signal - truth)[mask]) / np.linalg.norm(truth[mask]) < 0.03
    assert np.unravel_index(np.argmax(signal), signal.shape) == (20, 13, 32)
