from __future__ import annotations

from collections.abc import Callable
import functools
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from ._pallas_config import (
    _KERNEL_VARIANT_IDS,
    _LAYOUT_VARIANT_IDS,
    PallasProjectorUnsupported,
    _ensure_canonical_detector_grid,
    _ensure_float32_volume,
    _normalize_num_warps,
    _normalize_state_mode,
    _prepare_volume_for_pallas_gather,
    _resolve_effective_pallas_n_steps_for_stack,
    _supports_parallel_z_rotation_stack,
    _unsupported,
    _validate_public_sinogram_call,
    pallas_projector_actual_sinogram_variant_metadata,
)
from ._pallas_kernels import (
    _projector_loss_grad_kernel,
    _projector_parallel_z_views_kernel,
    _projector_residual_sse_kernel,
)
from ._pallas_stack_call import forward_project_residual_sse_T_pallas_with_state
from ._pallas_stack_state import prepare_forward_project_views_T_pallas_state
from .geometry.base import Detector, Grid, _grid_volume_origin
from .projector import _resolve_n_steps
from .validation import validate_pose_stack, validate_projection_stack, validate_volume


@functools.lru_cache(maxsize=32)
def _cached_loss_grad_pallas_call(
    *,
    nx: int,
    ny: int,
    nz: int,
    nv: int,
    nu: int,
    n_views: int,
    du: float,
    dv: float,
    det_center_x: float,
    det_center_z: float,
    vol_origin_x: float,
    vol_origin_y: float,
    vol_origin_z: float,
    vx: float,
    vy: float,
    vz: float,
    step_size: float,
    n_steps: int,
    tile_v: int,
    tile_u: int,
    num_warps: int,
    kernel_variant_id: int,
    layout_variant_id: int,
    unroll: int | None,
    compute_loss: bool,
    interpret: bool,
) -> Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
]:
    kernel = functools.partial(
        _projector_loss_grad_kernel,
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        nu=int(nu),
        nv=int(nv),
        du=float(du),
        dv=float(dv),
        det_center_x=float(det_center_x),
        det_center_z=float(det_center_z),
        vol_origin_x=float(vol_origin_x),
        vol_origin_y=float(vol_origin_y),
        vol_origin_z=float(vol_origin_z),
        vx=float(vx),
        vy=float(vy),
        vz=float(vz),
        step_size=float(step_size),
        n_steps=int(n_steps),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        kernel_variant_id=int(kernel_variant_id),
        layout_variant_id=int(layout_variant_id),
        unroll=unroll,
        compute_loss=bool(compute_loss),
    )
    grid_shape = (int(n_views), math.ceil(int(nv) / int(tile_v)), math.ceil(int(nu) / int(tile_u)))
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(grid_shape, jnp.float32),
            jax.ShapeDtypeStruct((int(nx) * int(ny) * int(nz),), jnp.float32),
        ),
        grid=grid_shape,
        in_specs=[
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=(
            pl.BlockSpec((1, 1, 1), lambda view, pv, pu: (view, pv, pu)),
            pl.no_block_spec,
        ),
        input_output_aliases={4: 1},
        interpret=bool(interpret),
        compiler_params=plt.CompilerParams(num_warps=int(num_warps)),
        name="tomojax_forward_loss_grad_T_pallas",
    )


def forward_project_loss_and_grad_T_pallas(
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    target: jnp.ndarray,
    *,
    weights: jnp.ndarray | None = None,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    interpret: bool = False,
    tile_shape: tuple[int, int] = (16, 4),
    num_warps: int = 1,
    kernel_variant: str = "auto",
    layout_variant: str = "detector_vu",
    compute_loss: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return projection loss and explicit gradient using one Pallas kernel."""
    vol = jnp.asarray(volume)
    nx, ny, nz = validate_volume(vol, grid, context="forward_project_loss_and_grad_T_pallas")
    _ensure_float32_volume(vol)
    n_views, _, _ = validate_projection_stack(
        target,
        detector,
        context="forward_project_loss_and_grad_T_pallas target",
    )
    validate_pose_stack(T_all, n_views, context="forward_project_loss_and_grad_T_pallas")
    _ensure_canonical_detector_grid(detector, det_grid)
    variant = pallas_projector_actual_sinogram_variant_metadata(
        T_all,
        grid,
        detector,
        det_grid=det_grid,
        tile_shape=tile_shape,
        num_warps=num_warps,
        kernel_variant=kernel_variant,
        layout_variant=layout_variant,
        state_mode="cached",
        gather_dtype=gather_dtype,
    )
    tile_v, tile_u = (int(v) for v in variant["tile_shape"])
    num_warps_value = _normalize_num_warps(num_warps)
    kernel_variant_id = _KERNEL_VARIANT_IDS[str(variant["kernel_variant"])]
    layout_variant_id = _LAYOUT_VARIANT_IDS[str(variant["layout_variant"])]
    if not interpret and jax.default_backend() == "cpu":
        raise PallasProjectorUnsupported(
            _unsupported("real Pallas lowering is unavailable on CPU; pass interpret=True")
        )
    step_size_value = float(grid.vy) if step_size is None else float(step_size)
    n_steps_value = _resolve_n_steps(grid, step_size_value, n_steps)
    vol_origin = _grid_volume_origin(grid)
    if weights is None:
        weights_arr = jnp.ones((int(n_views), 1, 1), dtype=jnp.float32)
    else:
        weights_arr = jnp.asarray(weights, dtype=jnp.float32).reshape((int(n_views), 1, 1))
    grad_init = jnp.zeros((int(nx) * int(ny) * int(nz),), dtype=jnp.float32)
    effective_n_steps_value = _resolve_effective_pallas_n_steps_for_stack(
        T_all,
        grid,
        step_size_value,
        n_steps_value,
    )
    call = _cached_loss_grad_pallas_call(
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        nv=int(detector.nv),
        nu=int(detector.nu),
        n_views=int(n_views),
        du=float(detector.du),
        dv=float(detector.dv),
        det_center_x=float(detector.det_center[0]),
        det_center_z=float(detector.det_center[1]),
        vol_origin_x=float(vol_origin[0]),
        vol_origin_y=float(vol_origin[1]),
        vol_origin_z=float(vol_origin[2]),
        vx=float(grid.vx),
        vy=float(grid.vy),
        vz=float(grid.vz),
        step_size=float(step_size_value),
        n_steps=int(effective_n_steps_value),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        num_warps=int(num_warps_value),
        kernel_variant_id=int(kernel_variant_id),
        layout_variant_id=int(layout_variant_id),
        unroll=unroll,
        compute_loss=bool(compute_loss),
        interpret=bool(interpret),
    )
    partial_loss, grad_flat = call(
        jnp.asarray(T_all, dtype=jnp.float32),
        _prepare_volume_for_pallas_gather(vol, str(variant["gather_dtype"])),
        jnp.asarray(target, dtype=jnp.float32),
        weights_arr,
        grad_init,
    )
    return jnp.sum(partial_loss, dtype=jnp.float32), grad_flat.reshape((int(nx), int(ny), int(nz)))


@functools.lru_cache(maxsize=32)
def _cached_parallel_z_views_pallas_call(
    *,
    nx: int,
    ny: int,
    nz: int,
    nv: int,
    nu: int,
    n_views: int,
    du: float,
    dv: float,
    det_center_x: float,
    det_center_z: float,
    vol_origin_x: float,
    vol_origin_y: float,
    vol_origin_z: float,
    vx: float,
    vy: float,
    vz: float,
    step_size: float,
    n_steps: int,
    tile_v: int,
    tile_u: int,
    num_warps: int,
    unroll: int | None,
    interpret: bool,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    kernel = functools.partial(
        _projector_parallel_z_views_kernel,
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        nu=int(nu),
        nv=int(nv),
        du=float(du),
        dv=float(dv),
        det_center_x=float(det_center_x),
        det_center_z=float(det_center_z),
        vol_origin_x=float(vol_origin_x),
        vol_origin_y=float(vol_origin_y),
        vol_origin_z=float(vol_origin_z),
        vx=float(vx),
        vy=float(vy),
        vz=float(vz),
        step_size=float(step_size),
        n_steps=int(n_steps),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        unroll=unroll,
    )
    grid_shape = (int(n_views), math.ceil(int(nv) / int(tile_v)), math.ceil(int(nu) / int(tile_u)))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((int(n_views), int(nv), int(nu)), jnp.float32),
        grid=grid_shape,
        in_specs=[
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec(
            (1, int(tile_v), int(tile_u)),
            lambda view, pv, pu: (view, pv, pu),
        ),
        interpret=bool(interpret),
        compiler_params=plt.CompilerParams(num_warps=int(num_warps)),
        name="tomojax_forward_project_parallel_z_views_pallas",
    )


def forward_project_parallel_z_views_pallas(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    interpret: bool = False,
    tile_shape: tuple[int, int] = (16, 4),
    num_warps: int = 1,
) -> jnp.ndarray:
    """Specialized stack projector for ParallelGeometry z-axis rotations only."""
    (
        nx,
        ny,
        nz,
        nv,
        nu,
        n_views,
        _volume_size,
        step_size_value,
        _n_steps_value,
        effective_n_steps_value,
        (tile_v, tile_u),
        num_warps_value,
        _kernel_variant_id,
        _layout_variant_id,
    ) = _validate_public_sinogram_call(
        T_stack,
        grid,
        detector,
        volume,
        step_size=step_size,
        n_steps=n_steps,
        gather_dtype=gather_dtype,
        det_grid=det_grid,
        interpret=interpret,
        tile_shape=tile_shape,
        num_warps=num_warps,
        kernel_variant="z_integer4",
        layout_variant="detector_vu",
        state_mode="inline",
    )
    if not _supports_parallel_z_rotation_stack(T_stack, grid, detector, det_grid):
        raise PallasProjectorUnsupported(
            _unsupported(
                "parallel z-axis specialization requires zero-translation ParallelGeometry poses"
            )
        )

    T = jnp.asarray(T_stack, dtype=jnp.float32)
    cos = T[:, 0, 0]
    sin = T[:, 1, 0]
    vol_origin = _grid_volume_origin(grid)
    call = _cached_parallel_z_views_pallas_call(
        nx=nx,
        ny=ny,
        nz=nz,
        nv=nv,
        nu=nu,
        n_views=n_views,
        du=float(detector.du),
        dv=float(detector.dv),
        det_center_x=float(detector.det_center[0]),
        det_center_z=float(detector.det_center[1]),
        vol_origin_x=float(vol_origin[0]),
        vol_origin_y=float(vol_origin[1]),
        vol_origin_z=float(vol_origin[2]),
        vx=float(grid.vx),
        vy=float(grid.vy),
        vz=float(grid.vz),
        step_size=float(step_size_value),
        n_steps=int(effective_n_steps_value),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        num_warps=int(num_warps_value),
        unroll=unroll,
        interpret=bool(interpret),
    )
    return call(cos, sin, _prepare_volume_for_pallas_gather(volume, gather_dtype))


def forward_project_residual_sse_T_pallas(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    target: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    interpret: bool = False,
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "auto",
    layout_variant: str = "detector_vu",
    state_mode: str = "inline",
) -> jnp.ndarray:
    """Return SSE between projected views and target without materializing the projection."""
    (
        nx,
        ny,
        nz,
        nv,
        nu,
        n_views,
        _volume_size,
        step_size_value,
        _n_steps_value,
        effective_n_steps_value,
        (tile_v, tile_u),
        num_warps_value,
        kernel_variant_id,
        layout_variant_id,
    ) = _validate_public_sinogram_call(
        T_stack,
        grid,
        detector,
        volume,
        step_size=step_size,
        n_steps=n_steps,
        gather_dtype=gather_dtype,
        det_grid=det_grid,
        interpret=interpret,
        tile_shape=tile_shape,
        num_warps=num_warps,
        kernel_variant=kernel_variant,
        layout_variant=layout_variant,
        state_mode=state_mode,
    )
    validate_projection_stack(
        target,
        detector,
        context="forward_project_residual_sse_T_pallas",
    )
    if int(target.shape[0]) != int(n_views):
        raise PallasProjectorUnsupported(
            _unsupported(
                "target view count does not match pose stack: "
                f"got {int(target.shape[0])}, expected {int(n_views)}"
            )
        )
    if _normalize_state_mode(state_mode) != "inline":
        state = prepare_forward_project_views_T_pallas_state(
            T_stack,
            grid,
            detector,
            step_size=step_size,
            n_steps=n_steps,
            gather_dtype=gather_dtype,
            det_grid=det_grid,
            tile_shape=tile_shape,
            num_warps=num_warps,
            kernel_variant=kernel_variant,
            layout_variant=layout_variant,
        )
        return forward_project_residual_sse_T_pallas_with_state(
            state,
            volume,
            target,
            interpret=interpret,
            unroll=unroll,
        )

    vol_origin = _grid_volume_origin(grid)
    kernel = functools.partial(
        _projector_residual_sse_kernel,
        nx=nx,
        ny=ny,
        nz=nz,
        nu=nu,
        nv=nv,
        du=float(detector.du),
        dv=float(detector.dv),
        det_center_x=float(detector.det_center[0]),
        det_center_z=float(detector.det_center[1]),
        vol_origin_x=float(vol_origin[0]),
        vol_origin_y=float(vol_origin[1]),
        vol_origin_z=float(vol_origin[2]),
        vx=float(grid.vx),
        vy=float(grid.vy),
        vz=float(grid.vz),
        step_size=float(step_size_value),
        n_steps=int(effective_n_steps_value),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        kernel_variant_id=int(kernel_variant_id),
        layout_variant_id=int(layout_variant_id),
        unroll=unroll,
    )
    tile_grid_v = math.ceil(nv / tile_v)
    tile_grid_u = math.ceil(nu / tile_u)
    partials = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n_views, tile_grid_v, tile_grid_u), jnp.float32),
        grid=(n_views, tile_grid_v, tile_grid_u),
        in_specs=[
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec((1, 1, 1), lambda view, pv, pu: (view, pv, pu)),
        interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=num_warps_value),
        name="tomojax_forward_project_residual_sse_T_pallas",
    )(
        jnp.asarray(T_stack, dtype=jnp.float32),
        _prepare_volume_for_pallas_gather(volume, gather_dtype),
        jnp.asarray(target, dtype=jnp.float32),
    )
    return jnp.sum(partials, dtype=jnp.float32)
