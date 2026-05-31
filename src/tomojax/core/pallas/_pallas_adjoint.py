from __future__ import annotations

from collections.abc import Callable
import functools
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
import jax.numpy as jnp

from tomojax.core.geometry.base import Detector, Grid, grid_volume_origin
from tomojax.core.projector import _resolve_n_steps
from tomojax.core.validation import (
    validate_detector,
    validate_detector_image,
    validate_grid,
    validate_pose_matrix,
    validate_pose_stack,
    validate_projection_stack,
)

from ._pallas_config import (
    _LAYOUT_VARIANT_IDS,
    PallasProjectorUnsupported,
    _ensure_canonical_detector_grid,
    _normalize_layout_variant,
    _normalize_num_warps,
    _normalize_tile_shape,
    _safe_detector_tile_shape,
    _unsupported,
)
from ._pallas_kernels import _backproject_kernel


@functools.lru_cache(maxsize=32)
def _cached_backproject_view_pallas_call(
    *,
    nx: int,
    ny: int,
    nz: int,
    nv: int,
    nu: int,
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
    layout_variant_id: int,
    unroll: int | None,
    interpret: bool,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    kernel = functools.partial(
        _backproject_kernel,
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
        layout_variant_id=int(layout_variant_id),
        unroll=unroll,
    )
    grid_shape = (math.ceil(int(nv) / int(tile_v)), math.ceil(int(nu) / int(tile_u)))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((int(nx) * int(ny) * int(nz),), jnp.float32),
        grid=grid_shape,
        in_specs=[
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.no_block_spec,
        input_output_aliases={2: 0},
        interpret=bool(interpret),
        compiler_params=plt.CompilerParams(num_warps=int(num_warps)),
        name="tomojax_backproject_view_T_pallas",
    )


def backproject_view_T_pallas(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    image: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    interpret: bool = False,
    tile_shape: tuple[int, int] = (8, 4),
    num_warps: int = 1,
    layout_variant: str = "detector_vu",
) -> jnp.ndarray:
    """Backproject one view using an experimental atomic Pallas adjoint kernel."""
    img = jnp.asarray(image, dtype=jnp.float32)
    nx, ny, nz = validate_grid(grid, "backproject_view_T_pallas")
    nv, nu = validate_detector(detector, "backproject_view_T_pallas")
    validate_detector_image(img, detector, context="backproject_view_T_pallas", name="image")
    validate_pose_matrix(T, context="backproject_view_T_pallas")
    _ensure_canonical_detector_grid(detector, det_grid)
    tile_v, tile_u = _safe_detector_tile_shape(
        list(_normalize_tile_shape(tile_shape)),
        detector,
        max_generic_tile_u=8,
    )
    num_warps_value = _normalize_num_warps(num_warps)
    layout_variant_id = _LAYOUT_VARIANT_IDS[_normalize_layout_variant(layout_variant)]
    if not interpret and jax.default_backend() == "cpu":
        raise PallasProjectorUnsupported(
            _unsupported("real Pallas lowering is unavailable on CPU; pass interpret=True")
        )
    step_size_value = float(grid.vy) if step_size is None else float(step_size)
    n_steps_value = _resolve_n_steps(grid, step_size_value, n_steps)
    vol_origin = grid_volume_origin(grid)
    call = _cached_backproject_view_pallas_call(
        nx=nx,
        ny=ny,
        nz=nz,
        nv=nv,
        nu=nu,
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
        n_steps=int(n_steps_value),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        num_warps=int(num_warps_value),
        layout_variant_id=int(layout_variant_id),
        unroll=unroll,
        interpret=bool(interpret),
    )
    init = jnp.zeros((int(nx) * int(ny) * int(nz),), dtype=jnp.float32)
    out = call(jnp.asarray(T, dtype=jnp.float32), img, init)
    return out.reshape((int(nx), int(ny), int(nz)))


def sum_backproject_views_T_pallas(
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    images: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    tile_shape: tuple[int, int] = (8, 4),
    num_warps: int = 1,
    layout_variant: str = "detector_vu",
) -> jnp.ndarray:
    """Sum one-view Pallas adjoints over a view stack.

    This optional backend helper intentionally does not replace the default JAX
    adjoint used by differentiable public paths.
    """
    del gather_dtype
    n_views, _, _ = validate_projection_stack(
        images,
        detector,
        context="sum_backproject_views_T_pallas",
    )
    validate_pose_stack(T_all, n_views, context="sum_backproject_views_T_pallas")
    _ensure_canonical_detector_grid(detector, det_grid)
    validate_grid(grid, "sum_backproject_views_T_pallas")
    img = jnp.asarray(images, dtype=jnp.float32)

    def backproject_one(T_i: jnp.ndarray, img_i: jnp.ndarray) -> jnp.ndarray:
        return backproject_view_T_pallas(
            T_i,
            grid,
            detector,
            img_i,
            step_size=step_size,
            n_steps=n_steps,
            unroll=unroll,
            det_grid=det_grid,
            tile_shape=tile_shape,
            num_warps=num_warps,
            layout_variant=layout_variant,
        )

    if int(n_views) == 1:
        return backproject_one(T_all[0], img[0])
    return jnp.sum(jax.vmap(backproject_one)(T_all, img), axis=0, dtype=jnp.float32)
