"""Real and interpreted GPU kernels must handle detector tails and layouts."""

from __future__ import annotations

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

# check-public-imports: allow-private
from tomojax.core.pallas import api as pallas

# check-public-imports: allow-private
from tomojax.core.pallas._pallas_sampling import _trilinear_load_active

# check-public-imports: allow-private
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry


@pytest.mark.parametrize("interpret", [True, pytest.param(False, marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    ("state_mode", "layout"),
    [("inline", "detector_vu"), ("inline", "detector_uv"), ("cached", "detector_vu")],
)
@pytest.mark.parametrize("tilted", [False, True])
def test_partial_detector_tiles_match_reference(
    interpret: bool, state_mode: str, layout: str, tilted: bool
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
    options = pallas.PallasProjectorOptions(
        interpret=interpret,
        state_mode=state_mode,
        layout_variant=layout,
        tile_shape=(4, 8),
        num_warps=1,
        unroll=2,
    )
    metadata = pallas.pallas_projector_actual_sinogram_variant_metadata(
        poses,
        grid,
        detector,
        tile_shape=(4, 8),
        state_mode=state_mode,
        layout_variant=layout,
    )
    assert metadata["tile_shape"] == [4, 8]
    expected = jax.jit(jax.vmap(lambda t: forward_project_view_T(t, grid, detector, volume)))(poses)
    actual = pallas.forward_project_views_T_pallas(poses, grid, detector, volume, options=options)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=2e-5)
    single = pallas.forward_project_view_T_pallas(poses[1], grid, detector, volume, options=options)
    np.testing.assert_allclose(single, expected[1], atol=1e-5, rtol=2e-5)
    actual_sse = pallas.forward_project_residual_sse_T_pallas(
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
    options = pallas.PallasProjectorOptions(
        interpret=interpret, step_size=0.8, tile_shape=(4, 4), num_warps=1
    )
    expected = jax.jit(lambda x: forward_project_view_T(pose, grid, detector, x, step_size=0.8))(
        volume
    )
    actual = pallas.forward_project_views_T_pallas(
        pose[None], grid, detector, volume, options=options
    )[0]
    single = pallas.forward_project_view_T_pallas(pose, grid, detector, volume, options=options)
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

    def kernel(volume_ref, coords_ref, active_ref, out_ref):
        lane = pl.program_id(0) * 32 + jnp.arange(32)
        result = _trilinear_load_active(
            volume_ref,
            plt.load(coords_ref.at[0, lane]),
            plt.load(coords_ref.at[1, lane]),
            plt.load(coords_ref.at[2, lane]),
            nx=shape[0],
            ny=shape[1],
            nz=shape[2],
            active=plt.load(active_ref.at[lane]),
            kernel_variant_id=int(integer_z),
        )
        out_ref[:] = result

    actual = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((coords.shape[1],), jnp.float32),
        grid=(coords.shape[1] // 32,),
        in_specs=[pl.no_block_spec] * 3,
        out_specs=pl.BlockSpec((32,), lambda i: (i,)),
        interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=1),
    )(volume.ravel(), jnp.asarray(coords), jnp.asarray(active))
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
    actual_loss, actual_gradient = pallas.forward_project_loss_and_grad_T_pallas(
        poses,
        grid,
        detector,
        volume,
        target,
        weights=weights,
        options=pallas.PallasProjectorOptions(
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
