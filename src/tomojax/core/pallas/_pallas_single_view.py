from __future__ import annotations

from collections.abc import Callable
import functools
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.base import Detector, Grid, grid_volume_origin
from tomojax.core.projector import _projector_traversal_state, _resolve_n_steps
from tomojax.core.validation import validate_detector, validate_pose_matrix, validate_volume

from ._pallas_config import (
    _KERNEL_VARIANT_IDS,
    PallasForwardProjectorTraversalState,
    PallasProjectorTraversalMetadata,
    PallasProjectorUnsupported,
    _ensure_canonical_detector_grid,
    _ensure_float32_volume,
    _normalize_gather_dtype,
    _normalize_state_mode,
    _prepare_volume_for_pallas_gather,
    _resolve_effective_pallas_n_steps,
    _unsupported,
    _validate_public_call,
    pallas_projector_actual_variant_metadata,
)
from ._pallas_kernels import _projector_kernel, _trilinear_load_when_tile_active


def prepare_forward_project_view_T_pallas_state(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "auto",
    layout_variant: str = "detector_vu",
) -> PallasForwardProjectorTraversalState:
    """Prepare fixed-geometry traversal state for the experimental cached Pallas path."""
    nv, nu = validate_detector(detector, "prepare_forward_project_view_T_pallas_state")
    validate_pose_matrix(T, context="prepare_forward_project_view_T_pallas_state")
    normalized_gather_dtype = _normalize_gather_dtype(gather_dtype)
    _ensure_canonical_detector_grid(detector, det_grid)
    variant = pallas_projector_actual_variant_metadata(
        T,
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
    if variant["layout_variant"] != "detector_vu":
        raise PallasProjectorUnsupported(
            _unsupported(
                "cached traversal state currently supports layout_variant='detector_vu' only"
            )
        )

    step_size_value = float(grid.vy) if step_size is None else float(step_size)
    resolved_n_steps = _resolve_n_steps(grid, step_size_value, n_steps)
    effective_n_steps = _resolve_effective_pallas_n_steps(
        T,
        grid,
        step_size_value,
        resolved_n_steps,
    )
    (
        ix0,
        iy0,
        iz0,
        dix,
        diy,
        diz,
        n_steps_ray,
        step_size32,
        _resolved_from_state,
        n_rays,
    ) = _projector_traversal_state(
        jnp.asarray(T, dtype=jnp.float32),
        grid,
        detector,
        step_size=step_size_value,
        n_steps=n_steps,
        det_grid=det_grid,
    )
    expected_rays = int(nv) * int(nu)
    if int(n_rays) != expected_rays:
        raise PallasProjectorUnsupported(
            _unsupported(f"cached traversal state expected {expected_rays} rays; got {n_rays}")
        )
    return PallasForwardProjectorTraversalState(
        traversal=PallasProjectorTraversalMetadata(
            ix0=jnp.ravel(jnp.asarray(ix0, dtype=jnp.float32), order="C"),
            iy0=jnp.ravel(jnp.asarray(iy0, dtype=jnp.float32), order="C"),
            iz0=jnp.ravel(jnp.asarray(iz0, dtype=jnp.float32), order="C"),
            n_steps_ray=jnp.ravel(jnp.asarray(n_steps_ray, dtype=jnp.int32), order="C"),
            step_size=float(np.asarray(step_size32, dtype=np.float32)),
            n_steps=int(effective_n_steps),
            resolved_n_steps=int(resolved_n_steps),
            nx=int(grid.nx),
            ny=int(grid.ny),
            nz=int(grid.nz),
            nv=int(nv),
            nu=int(nu),
            tile_shape=(int(variant["tile_shape"][0]), int(variant["tile_shape"][1])),
            num_warps=int(variant["num_warps"]),
            kernel_variant=str(variant["kernel_variant"]),
            kernel_variant_id=_KERNEL_VARIANT_IDS[str(variant["kernel_variant"])],
            gather_dtype=normalized_gather_dtype,
        ),
        dix=float(np.ravel(np.asarray(dix, dtype=np.float32))[0]),
        diy=float(np.ravel(np.asarray(diy, dtype=np.float32))[0]),
        diz=float(np.ravel(np.asarray(diz, dtype=np.float32))[0]),
    )


def block_forward_project_view_T_pallas_state(
    state: PallasForwardProjectorTraversalState,
) -> PallasForwardProjectorTraversalState:
    """Block until prepared traversal-state arrays are materialized."""
    jax.block_until_ready((state.ix0, state.iy0, state.iz0, state.n_steps_ray))
    return state


class BoundForwardProjectViewTPallas:
    """Fixed-geometry Pallas projector callable for repeated-volume workflows."""

    def __init__(
        self,
        state: PallasForwardProjectorTraversalState,
        *,
        interpret: bool = False,
        unroll: int | None = None,
    ) -> None:
        if not interpret and jax.default_backend() == "cpu":
            raise PallasProjectorUnsupported(
                _unsupported("real Pallas lowering is unavailable on CPU; pass interpret=True")
            )
        self.state = state
        self.interpret = bool(interpret)
        self.unroll = unroll

        tile_v, tile_u = state.tile_shape
        kernel = functools.partial(
            _projector_kernel_cached,
            nx=int(state.nx),
            ny=int(state.ny),
            nz=int(state.nz),
            nu=int(state.nu),
            nv=int(state.nv),
            dix=float(state.dix),
            diy=float(state.diy),
            diz=float(state.diz),
            step_size=float(state.step_size),
            n_steps=int(state.n_steps),
            tile_v=int(tile_v),
            tile_u=int(tile_u),
            kernel_variant_id=int(state.kernel_variant_id),
            unroll=unroll,
        )
        grid_shape = (math.ceil(state.nv / tile_v), math.ceil(state.nu / tile_u))
        self._call: Callable[
            [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
            jnp.ndarray,
        ] = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((state.nv, state.nu), jnp.float32),
            grid=grid_shape,
            in_specs=[
                pl.no_block_spec,
                pl.no_block_spec,
                pl.no_block_spec,
                pl.no_block_spec,
                pl.no_block_spec,
            ],
            out_specs=pl.BlockSpec((tile_v, tile_u), lambda pv, pu: (pv, pu)),
            interpret=interpret,
            compiler_params=plt.CompilerParams(num_warps=state.num_warps),
            name="tomojax_forward_project_view_T_pallas_bound_cached",
        )

    def __call__(self, volume: jnp.ndarray) -> jnp.ndarray:
        """Project ``volume`` with the cached single-view traversal state."""
        nx, ny, nz = validate_volume(
            volume,
            Grid(nx=self.state.nx, ny=self.state.ny, nz=self.state.nz, vx=1.0, vy=1.0, vz=1.0),
            context="BoundForwardProjectViewTPallas.__call__",
            name="volume",
        )
        _ensure_float32_volume(volume)
        if (nx, ny, nz) != (self.state.nx, self.state.ny, self.state.nz):
            raise PallasProjectorUnsupported(
                _unsupported(
                    "volume shape does not match cached traversal state: "
                    f"got {(nx, ny, nz)}, expected "
                    f"{(self.state.nx, self.state.ny, self.state.nz)}"
                )
            )
        return self._call(
            self.state.ix0,
            self.state.iy0,
            self.state.iz0,
            self.state.n_steps_ray,
            _prepare_volume_for_pallas_gather(volume, self.state.gather_dtype),
        )


def bind_forward_project_view_T_pallas(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
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
    block_state: bool = True,
) -> BoundForwardProjectViewTPallas:
    """Bind fixed geometry once and return a callable that projects volumes."""
    state = prepare_forward_project_view_T_pallas_state(
        T,
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
    if block_state:
        block_forward_project_view_T_pallas_state(state)
    return BoundForwardProjectViewTPallas(state, interpret=interpret, unroll=unroll)


def _projector_kernel_cached(
    ix0_ref: Any,
    iy0_ref: Any,
    iz0_ref: Any,
    n_steps_ray_ref: Any,
    volume_ref: Any,
    out_ref: Any,
    *,
    nx: int,
    ny: int,
    nz: int,
    nu: int,
    nv: int,
    dix: float,
    diy: float,
    diz: float,
    step_size: float,
    n_steps: int,
    tile_v: int,
    tile_u: int,
    kernel_variant_id: int,
    unroll: int | None,
) -> None:
    tile_v_start = pl.program_id(0) * tile_v
    tile_u_start = pl.program_id(1) * tile_u
    det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[jnp.newaxis, :]
    det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[:, jnp.newaxis]
    in_detector = (det_u < nu) & (det_v < nv)
    state_idx = jnp.clip(det_v * nu + det_u, 0, (nu * nv) - 1)

    ix0 = plt.load(ix0_ref.at[state_idx], mask=in_detector, other=0.0)
    iy0 = plt.load(iy0_ref.at[state_idx], mask=in_detector, other=0.0)
    iz0 = plt.load(iz0_ref.at[state_idx], mask=in_detector, other=0.0)
    n_steps_ray = plt.load(n_steps_ray_ref.at[state_idx], mask=in_detector, other=0)

    step_size32 = jnp.float32(step_size)
    dix32 = jnp.float32(dix)
    diy32 = jnp.float32(diy)
    diz32 = jnp.float32(diz)

    def body(step_idx, carry):
        acc, ix, iy, iz = carry
        active = step_idx < n_steps_ray
        sample = _trilinear_load_when_tile_active(
            volume_ref,
            ix,
            iy,
            iz,
            nx=nx,
            ny=ny,
            nz=nz,
            active=active,
            kernel_variant_id=kernel_variant_id,
        )
        return (
            acc + sample.astype(jnp.float32) * active.astype(jnp.float32) * step_size32,
            ix + dix32,
            iy + diy32,
            iz + diz32,
        )

    init = (
        jnp.zeros_like(ix0, dtype=jnp.float32),
        ix0,
        iy0,
        iz0,
    )
    if unroll is None:
        tile_steps = jnp.minimum(
            jnp.max(jnp.where(in_detector, n_steps_ray, 0)),
            jnp.asarray(n_steps, dtype=jnp.int32),
        )
        acc, _, _, _ = jax.lax.fori_loop(0, tile_steps, body, init)
    else:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init, unroll=unroll)
    out_ref[...] = acc.astype(jnp.float32)


def forward_project_view_T_pallas_with_state(
    state: PallasForwardProjectorTraversalState,
    volume: jnp.ndarray,
    *,
    interpret: bool = False,
    unroll: int | None = None,
) -> jnp.ndarray:
    """Forward project with a prepared traversal state using the experimental Pallas path."""
    nx, ny, nz = validate_volume(
        volume,
        Grid(nx=state.nx, ny=state.ny, nz=state.nz, vx=1.0, vy=1.0, vz=1.0),
        context="forward_project_view_T_pallas_with_state",
        name="volume",
    )
    _ensure_float32_volume(volume)
    if (nx, ny, nz) != (state.nx, state.ny, state.nz):
        raise PallasProjectorUnsupported(
            _unsupported(
                "volume shape does not match cached traversal state: "
                f"got {(nx, ny, nz)}, expected {(state.nx, state.ny, state.nz)}"
            )
        )
    if not interpret and jax.default_backend() == "cpu":
        raise PallasProjectorUnsupported(
            _unsupported("real Pallas lowering is unavailable on CPU; pass interpret=True")
        )

    tile_v, tile_u = state.tile_shape
    kernel = functools.partial(
        _projector_kernel_cached,
        nx=int(state.nx),
        ny=int(state.ny),
        nz=int(state.nz),
        nu=int(state.nu),
        nv=int(state.nv),
        dix=float(state.dix),
        diy=float(state.diy),
        diz=float(state.diz),
        step_size=float(state.step_size),
        n_steps=int(state.n_steps),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        kernel_variant_id=int(state.kernel_variant_id),
        unroll=unroll,
    )
    grid_shape = (math.ceil(state.nv / tile_v), math.ceil(state.nu / tile_u))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((state.nv, state.nu), jnp.float32),
        grid=grid_shape,
        in_specs=[
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec((tile_v, tile_u), lambda pv, pu: (pv, pu)),
        interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=state.num_warps),
        name="tomojax_forward_project_view_T_pallas_cached",
    )(
        state.ix0,
        state.iy0,
        state.iz0,
        state.n_steps_ray,
        _prepare_volume_for_pallas_gather(volume, state.gather_dtype),
    )


def forward_project_view_T_pallas(
    T: jnp.ndarray,
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
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "auto",
    layout_variant: str = "detector_vu",
    state_mode: str = "inline",
) -> jnp.ndarray:
    """Forward project one view using the experimental detector-tiled Pallas path."""
    (
        nx,
        ny,
        nz,
        nv,
        nu,
        _volume_size,
        step_size_value,
        _n_steps_value,
        effective_n_steps_value,
        (tile_v, tile_u),
        num_warps_value,
        kernel_variant_id,
        layout_variant_id,
    ) = _validate_public_call(
        T,
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
    if _normalize_state_mode(state_mode) != "inline":
        state = prepare_forward_project_view_T_pallas_state(
            T,
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
        return forward_project_view_T_pallas_with_state(
            state,
            volume,
            interpret=interpret,
            unroll=unroll,
        )
    vol_origin = grid_volume_origin(grid)
    kernel = functools.partial(
        _projector_kernel,
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
    grid_shape = (math.ceil(nv / tile_v), math.ceil(nu / tile_u))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((nv, nu), jnp.float32),
        grid=grid_shape,
        in_specs=[
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec((tile_v, tile_u), lambda pv, pu: (pv, pu)),
        interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=num_warps_value),
        name="tomojax_forward_project_view_T_pallas",
    )(
        jnp.asarray(T, dtype=jnp.float32),
        _prepare_volume_for_pallas_gather(volume, gather_dtype),
    )
