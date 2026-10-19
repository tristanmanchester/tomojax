"""Real and interpreted GPU kernels must handle detector tails and layouts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

# check-public-imports: allow-private
from tomojax.core._sampled_cuda import run as sampled_run

# check-public-imports: allow-private
from tomojax.core.cuda_projector import api as cuda

# check-public-imports: allow-private
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry


@pytest.mark.parametrize("interpret", [True, pytest.param(False, marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    ("state_mode", "layout"),
    [("inline", "detector_vu"), ("inline", "detector_uv"), ("cached", "detector_vu")],
)
@pytest.mark.parametrize("tilted", [False, True])
@pytest.mark.parametrize("unroll", [None, 2])
def test_partial_detector_tiles_match_reference(
    interpret: bool, state_mode: str, layout: str, tilted: bool, unroll: int | None
) -> None:
    if not interpret and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    grid = Grid(6, 5, 3, 0.8, 1.1, 1.0)
    detector = Detector(9, 7, 0.7, 1.0)
    geometry = (
        LaminographyGeometry(grid, detector, [0.0, 37.0], tilt_deg=30)
        if tilted
        else ParallelGeometry(grid, detector, [0.0, 37.0])
    )
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(2)], dtype=jnp.float32)
    rng = np.random.default_rng(731)
    volume = jnp.asarray(rng.normal(size=(6, 5, 3)), dtype=jnp.float32)
    target = jnp.asarray(rng.normal(size=(2, 7, 9)), dtype=jnp.float32)
    options = cuda.CudaProjectorOptions(
        interpret=interpret,
        state_mode=state_mode,
        layout_variant=layout,
        tile_shape=(4, 8),
        num_warps=1,
        unroll=unroll,
    )
    metadata = cuda.cuda_projector_actual_sinogram_variant_metadata(
        poses,
        grid,
        detector,
        tile_shape=(4, 8),
        state_mode=state_mode,
        layout_variant=layout,
    )
    assert metadata["tile_shape"] == [4, 8]
    expected = jax.jit(jax.vmap(lambda t: forward_project_view_T(t, grid, detector, volume)))(poses)
    actual = cuda.forward_project_views_T_cuda(poses, grid, detector, volume, options=options)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=2e-5)
    single = cuda.forward_project_view_T_cuda(poses[1], grid, detector, volume, options=options)
    np.testing.assert_allclose(single, expected[1], atol=1e-5, rtol=2e-5)
    actual_sse = cuda.forward_project_residual_sse_T_cuda(
        poses, grid, detector, volume, target, options=options
    )
    expected_sse = jnp.sum((expected - target) ** 2)
    np.testing.assert_allclose(actual_sse, expected_sse, rtol=2e-5)


@pytest.mark.parametrize("interpret", [True, pytest.param(False, marks=pytest.mark.gpu)])
@pytest.mark.parametrize("near_integer", ["tiny_tilt", "spacing_drift"])
def test_auto_variant_preserves_small_geometry_effects(interpret: bool, near_integer: str) -> None:
    if not interpret and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    pose = np.eye(4, dtype=np.float32)
    if near_integer == "tiny_tilt":
        grid = Grid(4, 128, 3, 1.0, 1.0, 0.01)
        detector = Detector(4, 3, 1.0, 0.01)
        angle = 5e-6
        pose[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    else:
        grid = Grid(4, 4, 1001, 1.0, 1.0, 1.0)
        detector = Detector(4, 1001, 1.0, 1.000009, (0.0, 0.0045))
    pose = jnp.asarray(pose)
    volume = jnp.asarray(
        np.random.default_rng(17).normal(size=(grid.nx, grid.ny, grid.nz)), dtype=jnp.float32
    )
    options = cuda.CudaProjectorOptions(
        interpret=interpret, step_size=0.8, tile_shape=(4, 4), num_warps=1
    )
    expected = jax.jit(lambda x: forward_project_view_T(pose, grid, detector, x, step_size=0.8))(
        volume
    )
    actual = cuda.forward_project_views_T_cuda(pose[None], grid, detector, volume, options=options)[
        0
    ]
    single = cuda.forward_project_view_T_cuda(pose, grid, detector, volume, options=options)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(single, expected, rtol=2e-5, atol=1e-5)


@pytest.mark.parametrize("interpret", [True, pytest.param(False, marks=pytest.mark.gpu)])
@pytest.mark.parametrize("shape", [(5, 7, 3), (1, 4, 1)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("integer_z", [False, True])
def test_sampler_zero_extension_matches_scipy(interpret, shape, dtype, integer_z) -> None:
    if not interpret and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    axes = [[-1.25, -1, -0.75, 0, 0.5, n - 1, n - 0.25, n + 0.25] for n in shape]
    if integer_z:
        axes[2] = [-2, -1, 0, 1, 2, shape[2] - 1, shape[2], shape[2] + 1]
    coords = np.asarray(np.meshgrid(*axes, indexing="ij"), dtype=np.float32).reshape(3, -1)
    active = np.arange(coords.shape[1]) % 11 != 0
    volume = jnp.asarray(np.random.default_rng(271).normal(size=shape), dtype=dtype)

    count = coords.shape[1]
    params = dict(zip(("nx", "ny", "nz"), shape, strict=True))
    params.update(
        nv=1,
        nu=count,
        tile_v=1,
        tile_u=32,
        num_warps=1,
        step_size=1.0,
        n_steps=1,
        kernel_variant_id=int(integer_z),
        interpret=interpret,
    )
    state = (
        *tuple(jnp.asarray(x) for x in coords),
        jnp.asarray(active, jnp.int32),
        *tuple(jnp.zeros(1, jnp.float32) for _ in range(3)),
    )
    actual = sampled_run(jnp.zeros((1, 1)), volume, mode=0, params=params, state=state).ravel()
    expected = map_coordinates(
        np.asarray(volume, dtype=np.float32), coords, order=1, mode="grid-constant", cval=0
    )
    np.testing.assert_allclose(actual, np.where(active, expected, 0), rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("interpret", [True, pytest.param(False, marks=pytest.mark.gpu)])
@pytest.mark.parametrize("layout", ["detector_vu", "detector_uv"])
@pytest.mark.parametrize("unroll", [None, 2])
def test_weighted_fused_loss_gradient_matches_autodiff(interpret, layout, unroll) -> None:
    if not interpret and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    grid = Grid(5, 4, 3, 0.8, 1.1, 1.3)
    detector = Detector(7, 5, 0.9, 1.2, (0.25, -0.4))
    geometry = LaminographyGeometry(grid, detector, [17.0, 73.0, 121.0], tilt_deg=37)
    poses = jnp.asarray([geometry.pose_for_view(i) for i in range(3)], dtype=jnp.float32)
    poses = poses.at[1, :3, 3].set(jnp.array([0.2, -0.3, 0.4]))
    poses = poses.at[2, 0, 3].set(100.0)  # Entire view misses the volume.
    rng = np.random.default_rng(641)
    volume = jnp.asarray(rng.normal(size=(5, 4, 3)), dtype=jnp.float32)
    target = jnp.asarray(rng.normal(size=(3, 5, 7)), dtype=jnp.float32)
    weights = jnp.array([0.5, 1.7, 0.8], dtype=jnp.float32)[:, None, None]

    def loss(x):
        prediction = jax.vmap(
            lambda t: forward_project_view_T(t, grid, detector, x, step_size=0.7)
        )(poses)
        return 0.5 * jnp.sum(((prediction - target) * weights) ** 2)

    expected_loss, expected_gradient = jax.jit(jax.value_and_grad(loss))(volume)
    actual_loss, actual_gradient = cuda.forward_project_loss_and_grad_T_cuda(
        poses,
        grid,
        detector,
        volume,
        target,
        weights=weights,
        options=cuda.CudaProjectorOptions(
            interpret=interpret,
            step_size=0.7,
            tile_shape=(4, 4),
            num_warps=1,
            layout_variant=layout,
            unroll=unroll,
        ),
    )
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=2e-6)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-5, atol=1e-5)


@pytest.mark.parametrize("interpret", [True, pytest.param(False, marks=pytest.mark.gpu)])
@pytest.mark.parametrize("state_mode", ["inline", "cached"])
@pytest.mark.parametrize("unroll", [None, 2])
def test_empty_ray_tiles_produce_zero_projection(interpret, state_mode, unroll) -> None:
    if not interpret and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    grid = Grid(3, 4, 2, 0.8, 1.1, 1.3)
    detector = Detector(5, 3, 0.7, 1.2)
    pose = jnp.eye(4, dtype=jnp.float32).at[0, 3].set(100.0)
    volume = jnp.ones((3, 4, 2), dtype=jnp.float32)
    options = cuda.CudaProjectorOptions(
        interpret=interpret,
        state_mode=state_mode,
        unroll=unroll,
        tile_shape=(4, 4),
        num_warps=1,
        kernel_variant="generic",
    )
    single = cuda.forward_project_view_T_cuda(pose, grid, detector, volume, options=options)
    stack = cuda.forward_project_views_T_cuda(pose[None], grid, detector, volume, options=options)
    sse = cuda.forward_project_residual_sse_T_cuda(
        pose[None], grid, detector, volume, jnp.ones((1, 3, 5)), options=options
    )
    np.testing.assert_array_equal(single, np.zeros((3, 5)))
    np.testing.assert_array_equal(stack, np.zeros((1, 3, 5)))
    assert float(sse) == 15.0


@pytest.mark.parametrize("interpret", [True, pytest.param(False, marks=pytest.mark.gpu)])
@pytest.mark.parametrize("layout", ["detector_vu", "detector_uv"])
@pytest.mark.parametrize("unroll", [None, 2])
def test_backprojection_replays_shifted_detector_rays(interpret, layout, unroll) -> None:
    if not interpret and jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    grid = Grid(5, 4, 3, 0.8, 1.1, 1.3)
    detector = Detector(7, 5, 0.9, 1.2, (0.25, -0.4))
    geometry = LaminographyGeometry(grid, detector, [37.0], tilt_deg=29)
    pose = jnp.asarray(geometry.pose_for_view(0), dtype=jnp.float32)
    pose = pose.at[:3, 3].set(jnp.array([0.2, -0.3, 0.4]))
    image = jnp.asarray(np.random.default_rng(641).normal(size=(5, 7)), dtype=jnp.float32)
    volume = jnp.zeros((5, 4, 3), dtype=jnp.float32)
    expected = jax.jit(
        jax.grad(
            lambda x: jnp.vdot(
                forward_project_view_T(pose, grid, detector, x, step_size=0.7), image
            )
        )
    )(volume)
    actual = cuda.backproject_view_T_cuda(
        pose,
        grid,
        detector,
        image,
        step_size=0.7,
        interpret=interpret,
        tile_shape=(4, 4),
        num_warps=1,
        layout_variant=layout,
        unroll=unroll,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)


def test_prepared_single_view_reuses_launch_without_capturing_ray_arrays(caplog) -> None:
    grid, detector = Grid(4, 4, 3, 1, 1, 1), Detector(5, 3, 1, 1)
    pose = jnp.eye(4, dtype=jnp.float32)
    volume = jnp.arange(48, dtype=jnp.float32).reshape(4, 4, 3)
    options = cuda.CudaProjectorOptions(interpret=True, tile_shape=(2, 4), num_warps=1)
    bound = cuda.bind_forward_project_view_T_cuda(pose, grid, detector, options=options)
    # The first CUDA trace lazily registers a JAX tracing context.
    for _ in range(2):
        jax.block_until_ready(bound(volume))
    for shift in [0.0, 0.6]:
        shifted = pose.at[0, 3].set(shift)
        state = cuda.prepare_forward_project_view_T_cuda_state(
            shifted, grid, detector, tile_shape=(2, 4), num_warps=1
        )
        expected = forward_project_view_T(shifted, grid, detector, volume)
        caplog.clear()
        with jax.log_compiles(True), caplog.at_level("WARNING"):
            actual = cuda.forward_project_view_T_cuda_with_state(state, volume, interpret=True)
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-5)
        assert not any("Compiling jit(" in record.message for record in caplog.records)


@pytest.mark.parametrize("kind", ["single", "stack", "sse"])
def test_inline_launch_reuses_compilation_with_new_poses(kind, caplog) -> None:
    grid, detector = Grid(4, 4, 3, 1, 1, 1), Detector(5, 3, 1, 1)
    poses = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (2, 4, 4))
    volume = jnp.arange(48, dtype=jnp.float32).reshape(4, 4, 3)
    target = jnp.ones((2, 3, 5), dtype=jnp.float32)
    options = cuda.CudaProjectorOptions(interpret=True, tile_shape=(2, 4), num_warps=1)

    def project(t):
        if kind == "single":
            return cuda.forward_project_view_T_cuda(t[0], grid, detector, volume, options=options)
        if kind == "stack":
            return cuda.forward_project_views_T_cuda(t, grid, detector, volume, options=options)
        return cuda.forward_project_residual_sse_T_cuda(
            t, grid, detector, volume, target, options=options
        )

    for _ in range(2):
        jax.block_until_ready(project(poses))
    shifted = poses.at[0, 0, 3].set(0.6)
    reference = jax.vmap(lambda t: forward_project_view_T(t, grid, detector, volume))(shifted)
    expected = reference[0] if kind == "single" else reference
    if kind == "sse":
        expected = jnp.sum((reference - target) ** 2)
    caplog.clear()
    with jax.log_compiles(True), caplog.at_level("WARNING"):
        actual = project(shifted)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-5)
    assert not any("Compiling jit(" in record.message for record in caplog.records)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", [1, 2])
def test_sampled_cuda_accumulates_without_mutating_initial(mode) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("requires a CUDA GPU")
    grid, detector = Grid(4, 3, 2, 1, 1, 1), Detector(5, 3, 1, 1)
    poses = jnp.eye(4, dtype=jnp.float32)[None]
    volume = jnp.arange(24, dtype=jnp.float32).reshape(4, 3, 2) / 24
    target = jnp.ones((1, 3, 5), jnp.float32)
    initial = jnp.full((24,), 2.3, jnp.float32)
    params = dict(
        nx=4,
        ny=3,
        nz=2,
        nv=3,
        nu=5,
        vx=1,
        vy=1,
        vz=1,
        du=1,
        dv=1,
        det_center_x=0,
        det_center_z=0,
        vol_origin_x=-1.5,
        vol_origin_y=-1,
        vol_origin_z=-0.5,
        tile_v=2,
        tile_u=4,
        num_warps=1,
        step_size=0.7,
        n_steps=12,
    )

    def project(x):
        return forward_project_view_T(poses[0], grid, detector, x, step_size=0.7, n_steps=12)[None]

    def objective(x):
        prediction = project(x)
        return (
            jnp.vdot(prediction, target) if mode == 1 else 0.5 * jnp.sum((prediction - target) ** 2)
        )

    expected = jax.grad(objective)(volume).ravel() + initial
    call = jax.jit(
        lambda x, acc: sampled_run(poses, x, target, jnp.ones(1), acc, mode=mode, params=params)
    )
    actual = call(volume, initial)
    actual = actual[1] if mode == 2 else actual
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)
    np.testing.assert_array_equal(initial, jnp.full((24,), 2.3, jnp.float32))
