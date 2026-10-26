from __future__ import annotations

from collections.abc import Callable
import functools

import jax
import jax.numpy as jnp

from tomojax.core._sampled_cuda import run
from tomojax.core.geometry.base import Detector, Grid, grid_volume_origin
from tomojax.core.projector import _resolve_n_steps
from tomojax.core.validation import validate_pose_stack, validate_projection_stack, validate_volume

from ._config import (
    _KERNEL_VARIANT_IDS,
    _LAYOUT_VARIANT_IDS,
    CudaProjectorUnsupported,
    _effective_cuda_n_steps,
    _ensure_canonical_detector_grid,
    _ensure_float32_volume,
    _normalize_num_warps,
    _normalize_state_mode,
    _prepare_volume_for_cuda_gather,
    _supports_parallel_z_rotation_stack,
    _unsupported,
    _validate_public_sinogram_call,
    cuda_projector_actual_sinogram_variant_metadata,
)
from ._inline_call import inline_projector_call
from ._stack_call import forward_project_residual_sse_T_cuda_with_state
from ._stack_state import prepare_forward_project_views_T_cuda_state


@functools.lru_cache(maxsize=32)
def _cached_loss_grad_cuda_call(
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
    params = locals()

    @jax.jit
    def call(poses, volume, target, weights, initial):
        return run(poses, volume, target, weights, initial, mode=2, params=params)

    return call


def forward_project_loss_and_grad_T_cuda(
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
    """Return projection loss and explicit gradient using one CUDA kernel."""
    vol = jnp.asarray(volume)
    nx, ny, nz = validate_volume(vol, grid, context="forward_project_loss_and_grad_T_cuda")
    _ensure_float32_volume(vol)
    n_views, _, _ = validate_projection_stack(
        target,
        detector,
        context="forward_project_loss_and_grad_T_cuda target",
    )
    validate_pose_stack(T_all, n_views, context="forward_project_loss_and_grad_T_cuda")
    _ensure_canonical_detector_grid(detector, det_grid)
    variant = cuda_projector_actual_sinogram_variant_metadata(
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
        raise CudaProjectorUnsupported(
            _unsupported("real CUDA lowering is unavailable on CPU; pass interpret=True")
        )
    step_size_value = float(grid.vy) if step_size is None else float(step_size)
    n_steps_value = _resolve_n_steps(grid, step_size_value, n_steps)
    vol_origin = grid_volume_origin(grid)
    if weights is None:
        weights_arr = jnp.ones((int(n_views), 1, 1), dtype=jnp.float32)
    else:
        weights_arr = jnp.asarray(weights, dtype=jnp.float32).reshape((int(n_views), 1, 1))
    grad_init = jnp.zeros((int(nx) * int(ny) * int(nz),), dtype=jnp.float32)
    effective_n_steps_value = _effective_cuda_n_steps(
        T_all,
        grid,
        step_size_value,
        n_steps_value,
    )
    call = _cached_loss_grad_cuda_call(
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        nv=int(detector.nv),
        nu=int(detector.nu),
        n_views=int(n_views),
        du=float(detector.du),
        dv=float(detector.dv),
        det_center_x=float(detector.center[0]),
        det_center_z=float(detector.center[1]),
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
        _prepare_volume_for_cuda_gather(vol, str(variant["gather_dtype"])),
        jnp.asarray(target, dtype=jnp.float32),
        weights_arr,
        grad_init,
    )
    return jnp.sum(partial_loss, dtype=jnp.float32), grad_flat.reshape((int(nz), int(ny), int(nx)))


@functools.lru_cache(maxsize=32)
def _cached_parallel_z_views_cuda_call(
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
    params = locals() | {"kernel_variant_id": 1}

    @jax.jit
    def call(cos, sin, volume):
        poses = jnp.broadcast_to(jnp.eye(4, dtype=jnp.float32), (n_views, 4, 4))
        poses = poses.at[:, 0, 0].set(cos).at[:, 1, 1].set(cos)
        poses = poses.at[:, 1, 0].set(sin).at[:, 0, 1].set(-sin)
        return run(poses, volume, mode=0, params=params)

    return call


def forward_project_parallel_z_views_cuda(
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
        raise CudaProjectorUnsupported(
            _unsupported(
                "parallel z-axis specialization requires zero-translation ParallelGeometry poses"
            )
        )

    T = jnp.asarray(T_stack, dtype=jnp.float32)
    cos = T[:, 0, 0]
    sin = T[:, 1, 0]
    vol_origin = grid_volume_origin(grid)
    call = _cached_parallel_z_views_cuda_call(
        nx=nx,
        ny=ny,
        nz=nz,
        nv=nv,
        nu=nu,
        n_views=n_views,
        du=float(detector.du),
        dv=float(detector.dv),
        det_center_x=float(detector.center[0]),
        det_center_z=float(detector.center[1]),
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
    return call(cos, sin, _prepare_volume_for_cuda_gather(volume, gather_dtype))


def forward_project_residual_sse_T_cuda(
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
        context="forward_project_residual_sse_T_cuda",
    )
    if int(target.shape[0]) != int(n_views):
        raise CudaProjectorUnsupported(
            _unsupported(
                "target view count does not match pose stack: "
                f"got {int(target.shape[0])}, expected {int(n_views)}"
            )
        )
    if _normalize_state_mode(state_mode) != "inline":
        state = prepare_forward_project_views_T_cuda_state(
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
        return forward_project_residual_sse_T_cuda_with_state(
            state,
            volume,
            target,
            interpret=interpret,
            unroll=unroll,
        )

    vol_origin = grid_volume_origin(grid)
    call = inline_projector_call(
        n_views=n_views,
        nx=nx,
        ny=ny,
        nz=nz,
        nu=nu,
        nv=nv,
        du=float(detector.du),
        dv=float(detector.dv),
        det_center_x=float(detector.center[0]),
        det_center_z=float(detector.center[1]),
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
        num_warps=int(num_warps_value),
        interpret=bool(interpret),
        residual=True,
    )
    partials = call(
        jnp.asarray(T_stack, dtype=jnp.float32),
        _prepare_volume_for_cuda_gather(volume, gather_dtype),
        jnp.asarray(target, dtype=jnp.float32),
    )
    return jnp.sum(partials, dtype=jnp.float32)
