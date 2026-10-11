"""Independent analytic checks for FBP filtering and physical normalization."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry
from tomojax.recon import FBPConfig, FBPHostConfig, fbp, fbp_host

# check-public-imports: allow-private
from tomojax.recon._fbp_pallas import backproject_filtered_pallas
from tomojax.recon.fbp import _backproject_voxels_jax
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
            z_mode="general",
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
        (32, (4096, 4096, 2), (9, 5), None, 512),
        (129, (7, 6, 3), (9, 5), None, 128),
        (129, (7, 6, 3), (512, 512), None, 512),
        (129, (512, 512, 11), (767, 11), None, 128),
        (129, (768, 768, 17), (1089, 17), None, 128),
        (129, (4096, 4096, 2), (9, 5), None, 512),
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
