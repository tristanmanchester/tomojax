"""Independent analytic checks for FBP filtering and physical normalization."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon import FBPConfig, fbp
from tomojax.recon.filters import fft_filter_rows, rfft_filter_array


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
            z_integer=False,
            separable=separable,
        )

    np.testing.assert_allclose(run(batch, backend), run(5, "jax"), rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("use_explicit_detector_grid", [False, True])
def test_fbp_is_invariant_to_change_of_length_units(use_explicit_detector_grid: bool) -> None:
    """Scaling all lengths and line integrals leaves attenuation values unchanged."""
    angles = np.asarray([0.0, 37.0, 81.0, 127.0], dtype=np.float32)
    projections = np.random.default_rng(17).normal(size=(4, 3, 9)).astype(np.float32)
    reconstructions = []
    for scale in [1.0, 3.0]:
        grid = Grid(7, 8, 3, 0.8 * scale, 1.1 * scale, 1.3 * scale)
        detector = Detector(9, 3, 0.9 * scale, 1.3 * scale)
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
    np.testing.assert_allclose(reconstructions[0], reconstructions[1], rtol=5e-5, atol=5e-6)


@pytest.mark.parametrize("interpret", [True, pytest.param(False, marks=pytest.mark.gpu)])
@pytest.mark.parametrize("z_integer", [False, True])
def test_voxel_pallas_fbp_matches_jax_on_partial_blocks(interpret: bool, z_integer: bool) -> None:
    import jax

    # check-public-imports: allow-private
    from tomojax.recon._fbp_pallas import backproject_filtered_pallas

    # check-public-imports: allow-private
    from tomojax.recon.fbp import _backproject_voxels_jax

    if not interpret and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    grid = Grid(7, 6, 3, 0.8, 1.2, 1.0 if z_integer else 0.7)
    detector = Detector(9, 5, 0.9, 1.0, (0.17, 0.0 if z_integer else 0.21))
    geometry = ParallelGeometry(grid, detector, [0.0, 31.0, 78.0, 117.0, 161.0])
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(5)], dtype=jnp.float32)
    projections = jnp.asarray(np.random.default_rng(18).normal(size=(5, 5, 9)), dtype=jnp.float32)
    ramp = rfft_filter_array("ramp", 9, detector.du, jnp.float32)
    expected = jax.jit(
        lambda y: _backproject_voxels_jax(
            poses, fft_filter_rows(y, ramp), grid=grid, detector=detector
        )
    )(projections)
    actual = backproject_filtered_pallas(
        poses,
        fft_filter_rows(projections, ramp),
        grid=grid,
        detector=detector,
        z_integer=z_integer,
        interpret=interpret,
        block_size=64,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_explicit_pallas_fbp_requires_cuda_arrays() -> None:
    import jax

    from tomojax.geometry import LaminographyGeometry

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
