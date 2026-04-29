from __future__ import annotations

import functools
import math
import operator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from .geometry.base import Detector, Grid, _grid_volume_origin
from .projector import _build_detector_grid, _resolve_n_steps
from .validation import (
    validate_detector,
    validate_detector_grid,
    validate_pose_matrix,
    validate_volume,
)


class PallasProjectorUnsupported(ValueError):
    """Raised when the experimental Pallas projector cannot handle a call."""


def _unsupported(message: str) -> str:
    return f"pallas_projector_unsupported: {message}"


def _normalize_gather_dtype(gather_dtype: str) -> str:
    if not isinstance(gather_dtype, str):
        raise PallasProjectorUnsupported(
            _unsupported(f"gather_dtype must be a string; got {type(gather_dtype).__name__}")
        )
    gd = gather_dtype.lower()
    if gd not in {"fp32", "float32", "single"}:
        raise PallasProjectorUnsupported(
            _unsupported(f"gather_dtype={gather_dtype!r}; v1 supports fp32 only")
        )
    return "fp32"


def _normalize_tile_shape(tile_shape: tuple[int, int]) -> tuple[int, int]:
    try:
        tile_v = operator.index(tile_shape[0])
        tile_u = operator.index(tile_shape[1])
    except Exception as exc:
        raise PallasProjectorUnsupported(
            _unsupported(f"tile_shape must be two positive integers; got {tile_shape!r}")
        ) from exc
    if tile_v <= 0 or tile_u <= 0:
        raise PallasProjectorUnsupported(
            _unsupported(f"tile_shape must be two positive integers; got {tile_shape!r}")
        )
    return int(tile_v), int(tile_u)


def _ensure_float32_volume(volume: jnp.ndarray) -> None:
    dtype = getattr(volume, "dtype", None)
    if dtype != jnp.dtype(jnp.float32):
        raise PallasProjectorUnsupported(_unsupported(f"volume dtype must be float32; got {dtype}"))


def _ensure_canonical_detector_grid(
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> None:
    if det_grid is None:
        return
    try:
        validate_detector_grid(
            det_grid,
            detector,
            context="forward_project_view_T_pallas",
        )
    except ValueError as exc:
        raise PallasProjectorUnsupported(_unsupported(str(exc))) from exc

    Xr_expected, Zr_expected = _build_detector_grid(detector)
    Xr, Zr = det_grid
    try:
        Xr_host = np.asarray(Xr, dtype=np.float32)
        Zr_host = np.asarray(Zr, dtype=np.float32)
    except Exception as exc:
        raise PallasProjectorUnsupported(
            _unsupported("det_grid must be None or the canonical eager detector grid")
        ) from exc
    if not (
        np.array_equal(Xr_host, Xr_expected)
        and np.array_equal(Zr_host, Zr_expected)
    ):
        raise PallasProjectorUnsupported(
            _unsupported("det_grid must be None or get_detector_grid_device(detector)")
        )


def _validate_public_call(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    step_size: float | None,
    n_steps: int | None,
    gather_dtype: str,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    interpret: bool,
    tile_shape: tuple[int, int],
) -> tuple[int, int, int, int, int, int, float, int, tuple[int, int]]:
    nx, ny, nz = validate_volume(
        volume,
        grid,
        context="forward_project_view_T_pallas",
        name="volume",
    )
    nv, nu = validate_detector(detector, "forward_project_view_T_pallas")
    validate_pose_matrix(T, context="forward_project_view_T_pallas")
    _normalize_gather_dtype(gather_dtype)
    _ensure_float32_volume(volume)
    _ensure_canonical_detector_grid(detector, det_grid)
    tile_v, tile_u = _normalize_tile_shape(tile_shape)
    if not interpret and jax.default_backend() == "cpu":
        raise PallasProjectorUnsupported(
            _unsupported("real Pallas lowering is unavailable on CPU; pass interpret=True")
        )

    if step_size is None:
        step_size_value = float(grid.vy)
    else:
        step_size_value = float(step_size)
    n_steps_value = _resolve_n_steps(grid, step_size_value, n_steps)
    return nx, ny, nz, nv, nu, nx * ny * nz, step_size_value, n_steps_value, (
        tile_v,
        tile_u,
    )


def pallas_projector_unsupported_reason(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    tile_shape: tuple[int, int] = (8, 8),
) -> str | None:
    """Return a benchmark-friendly unsupported reason, or ``None`` if eligible."""
    try:
        _validate_public_call(
            T,
            grid,
            detector,
            volume,
            step_size=step_size,
            n_steps=n_steps,
            gather_dtype=gather_dtype,
            det_grid=det_grid,
            interpret=False,
            tile_shape=tile_shape,
        )
    except PallasProjectorUnsupported as exc:
        return str(exc)
    except ValueError as exc:
        return _unsupported(str(exc))
    return None


def _trilinear_load(
    volume_ref: Any,
    ix_f: jnp.ndarray,
    iy_f: jnp.ndarray,
    iz_f: jnp.ndarray,
    *,
    nx: int,
    ny: int,
    nz: int,
) -> jnp.ndarray:
    fx = jnp.floor(ix_f).astype(jnp.int32)
    fy = jnp.floor(iy_f).astype(jnp.int32)
    fz = jnp.floor(iz_f).astype(jnp.int32)
    cx, cy, cz = fx + 1, fy + 1, fz + 1

    wx1 = ix_f - fx.astype(jnp.float32)
    wy1 = iy_f - fy.astype(jnp.float32)
    wz1 = iz_f - fz.astype(jnp.float32)
    wx0 = jnp.float32(1.0) - wx1
    wy0 = jnp.float32(1.0) - wy1
    wz0 = jnp.float32(1.0) - wz1

    def gather(ix: jnp.ndarray, iy: jnp.ndarray, iz: jnp.ndarray) -> jnp.ndarray:
        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        idx = ix * (ny * nz) + iy * nz + iz
        idx = jnp.clip(idx, 0, (nx * ny * nz) - 1)
        return plt.load(volume_ref.at[idx], mask=inb, other=0.0)

    c000 = gather(fx, fy, fz) * (wx0 * wy0 * wz0)
    c001 = gather(fx, fy, cz) * (wx0 * wy0 * wz1)
    c010 = gather(fx, cy, fz) * (wx0 * wy1 * wz0)
    c011 = gather(fx, cy, cz) * (wx0 * wy1 * wz1)
    c100 = gather(cx, fy, fz) * (wx1 * wy0 * wz0)
    c101 = gather(cx, fy, cz) * (wx1 * wy0 * wz1)
    c110 = gather(cx, cy, fz) * (wx1 * wy1 * wz0)
    c111 = gather(cx, cy, cz) * (wx1 * wy1 * wz1)
    return c000 + c001 + c010 + c011 + c100 + c101 + c110 + c111


def _projector_kernel(
    T_ref: Any,
    volume_ref: Any,
    out_ref: Any,
    *,
    nx: int,
    ny: int,
    nz: int,
    nu: int,
    nv: int,
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
    unroll: int | None,
) -> None:
    tile_v_start = pl.program_id(0) * tile_v
    tile_u_start = pl.program_id(1) * tile_u
    det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)
    det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)

    xr = (
        (det_u.astype(jnp.float32) - jnp.float32(nu / 2.0 - 0.5)) * jnp.float32(du)
        + jnp.float32(det_center_x)
    )
    zr = (
        (det_v.astype(jnp.float32) - jnp.float32(nv / 2.0 - 0.5)) * jnp.float32(dv)
        + jnp.float32(det_center_z)
    )
    xr = xr[jnp.newaxis, :]
    zr = zr[:, jnp.newaxis]

    def tload(row: int, col: int):
        return plt.load(T_ref.at[row, col])

    # Rinv = T[:3, :3].T. The existing projector evaluates world y as the ray
    # parameter, so ``base`` is object_from_world([x, 0, z]).
    t00, t01, t02, t03 = tload(0, 0), tload(0, 1), tload(0, 2), tload(0, 3)
    t10, t11, t12, t13 = tload(1, 0), tload(1, 1), tload(1, 2), tload(1, 3)
    t20, t21, t22, t23 = tload(2, 0), tload(2, 1), tload(2, 2), tload(2, 3)
    ey_x = t10
    ey_y = t11
    ey_z = t12
    tinv_x = -(t00 * t03 + t10 * t13 + t20 * t23)
    tinv_y = -(t01 * t03 + t11 * t13 + t21 * t23)
    tinv_z = -(t02 * t03 + t12 * t13 + t22 * t23)
    base_x = t00 * xr + t20 * zr + tinv_x
    base_y = t01 * xr + t21 * zr + tinv_y
    base_z = t02 * xr + t22 * zr + tinv_z

    lower_x = jnp.float32(vol_origin_x - vx)
    lower_y = jnp.float32(vol_origin_y - vy)
    lower_z = jnp.float32(vol_origin_z - vz)
    upper_x = jnp.float32(vol_origin_x + nx * vx)
    upper_y = jnp.float32(vol_origin_y + ny * vy)
    upper_z = jnp.float32(vol_origin_z + nz * vz)

    def slab(base: jnp.ndarray, denom: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray):
        eps = jnp.float32(1e-8)
        parallel = jnp.abs(denom) < eps
        safe_denom = jnp.where(parallel, jnp.float32(1.0), denom)
        t1 = (lower - base) / safe_denom
        t2 = (upper - base) / safe_denom
        lo = jnp.minimum(t1, t2)
        hi = jnp.maximum(t1, t2)
        inside = (base >= lower) & (base <= upper)
        inf = jnp.asarray(jnp.inf, dtype=jnp.float32)
        neg_inf = jnp.asarray(float("-inf"), dtype=jnp.float32)
        lo = jnp.where(parallel, jnp.where(inside, neg_inf, inf), lo)
        hi = jnp.where(parallel, jnp.where(inside, inf, neg_inf), hi)
        return lo, hi

    lo_x, hi_x = slab(base_x, ey_x, lower_x, upper_x)
    lo_y, hi_y = slab(base_y, ey_y, lower_y, upper_y)
    lo_z, hi_z = slab(base_z, ey_z, lower_z, upper_z)
    y_entry = jnp.maximum(jnp.maximum(lo_x, lo_y), lo_z)
    y_exit = jnp.minimum(jnp.minimum(hi_x, hi_y), hi_z)
    path_length = jnp.maximum(jnp.float32(0.0), y_exit - y_entry)
    valid_rays = path_length > jnp.float32(0.0)
    y_start = jnp.where(valid_rays, y_entry, jnp.float32(0.0))
    step_size32 = jnp.float32(step_size)
    n_steps_ray = jnp.where(
        valid_rays,
        jnp.ceil(path_length / step_size32).astype(jnp.int32),
        jnp.int32(0),
    )

    q0_x = base_x + y_start * ey_x
    q0_y = base_y + y_start * ey_y
    q0_z = base_z + y_start * ey_z
    ix0 = (q0_x - jnp.float32(vol_origin_x)) / jnp.float32(vx)
    iy0 = (q0_y - jnp.float32(vol_origin_y)) / jnp.float32(vy)
    iz0 = (q0_z - jnp.float32(vol_origin_z)) / jnp.float32(vz)
    dix = step_size32 * ey_x / jnp.float32(vx)
    diy = step_size32 * ey_y / jnp.float32(vy)
    diz = step_size32 * ey_z / jnp.float32(vz)

    def body(step_idx, carry):
        acc, ix, iy, iz = carry
        sample = _trilinear_load(
            volume_ref,
            ix,
            iy,
            iz,
            nx=nx,
            ny=ny,
            nz=nz,
        )
        active = (step_idx < n_steps_ray).astype(jnp.float32)
        return (
            acc + sample.astype(jnp.float32) * active * step_size32,
            ix + dix,
            iy + diy,
            iz + diz,
        )

    init = (
        jnp.zeros((tile_v, tile_u), dtype=jnp.float32),
        ix0,
        iy0,
        iz0,
    )
    if unroll is None:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init)
    else:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init, unroll=unroll)
    out_ref[...] = acc.astype(jnp.float32)


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
    tile_shape: tuple[int, int] = (8, 8),
) -> jnp.ndarray:
    """Forward project one view using the experimental detector-tiled Pallas path."""
    nx, ny, nz, nv, nu, volume_size, step_size_value, n_steps_value, (tile_v, tile_u) = (
        _validate_public_call(
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
        )
    )
    vol_origin = _grid_volume_origin(grid)
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
        n_steps=int(n_steps_value),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
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
        compiler_params=plt.CompilerParams(num_warps=4),
        name="tomojax_forward_project_view_T_pallas",
    )(
        jnp.asarray(T, dtype=jnp.float32),
        jnp.ravel(volume, order="C"),
    )
