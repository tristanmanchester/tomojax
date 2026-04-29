from __future__ import annotations

import functools
import math
import operator
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from .geometry.base import Detector, Grid, _grid_volume_origin
from .projector import _build_detector_grid, _projector_traversal_state, _resolve_n_steps
from .validation import (
    validate_detector,
    validate_detector_grid,
    validate_pose_stack,
    validate_pose_matrix,
    validate_projection_stack,
    validate_volume,
)


class PallasProjectorUnsupported(ValueError):
    """Raised when the experimental Pallas projector cannot handle a call."""


_SUPPORTED_NUM_WARPS = frozenset({1, 2, 4, 8})
_SUPPORTED_KERNEL_VARIANTS = frozenset({"auto", "generic", "z_integer4"})
_SUPPORTED_LAYOUT_VARIANTS = frozenset({"detector_uv", "detector_vu"})
_SUPPORTED_STATE_MODES = frozenset({"cached", "inline", "precompute_inclusive"})
_KERNEL_VARIANT_IDS = {"generic": 0, "z_integer4": 1}
_LAYOUT_VARIANT_IDS = {"detector_vu": 0, "detector_uv": 1}


@dataclass(frozen=True)
class PallasForwardProjectorTraversalState:
    """Prepared per-ray traversal state for fixed-geometry Pallas benchmarking."""

    ix0: jnp.ndarray
    iy0: jnp.ndarray
    iz0: jnp.ndarray
    n_steps_ray: jnp.ndarray
    dix: float
    diy: float
    diz: float
    step_size: float
    n_steps: int
    resolved_n_steps: int
    nx: int
    ny: int
    nz: int
    nv: int
    nu: int
    tile_shape: tuple[int, int]
    num_warps: int
    kernel_variant: str
    kernel_variant_id: int


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


def _normalize_num_warps(num_warps: int) -> int:
    try:
        value = operator.index(num_warps)
    except Exception as exc:
        raise PallasProjectorUnsupported(
            _unsupported(f"num_warps must be one of {sorted(_SUPPORTED_NUM_WARPS)}; got {num_warps!r}")
        ) from exc
    if value not in _SUPPORTED_NUM_WARPS:
        raise PallasProjectorUnsupported(
            _unsupported(f"num_warps must be one of {sorted(_SUPPORTED_NUM_WARPS)}; got {num_warps!r}")
        )
    return int(value)


def _normalize_kernel_variant(kernel_variant: str) -> str:
    if not isinstance(kernel_variant, str):
        raise PallasProjectorUnsupported(
            _unsupported(
                f"kernel_variant must be a string; got {type(kernel_variant).__name__}"
            )
        )
    value = kernel_variant.lower()
    if value not in _SUPPORTED_KERNEL_VARIANTS:
        raise PallasProjectorUnsupported(
            _unsupported(
                "kernel_variant must be one of "
                f"{sorted(_SUPPORTED_KERNEL_VARIANTS)}; got {kernel_variant!r}"
            )
        )
    return value


def _normalize_layout_variant(layout_variant: str) -> str:
    if not isinstance(layout_variant, str):
        raise PallasProjectorUnsupported(
            _unsupported(
                f"layout_variant must be a string; got {type(layout_variant).__name__}"
            )
        )
    value = layout_variant.lower()
    if value not in _SUPPORTED_LAYOUT_VARIANTS:
        raise PallasProjectorUnsupported(
            _unsupported(
                "layout_variant must be one of "
                f"{sorted(_SUPPORTED_LAYOUT_VARIANTS)}; got {layout_variant!r}"
            )
        )
    return value


def _normalize_state_mode(state_mode: str) -> str:
    if not isinstance(state_mode, str):
        raise PallasProjectorUnsupported(
            _unsupported(f"state_mode must be a string; got {type(state_mode).__name__}")
        )
    value = state_mode.lower()
    if value not in _SUPPORTED_STATE_MODES:
        raise PallasProjectorUnsupported(
            _unsupported(
                f"state_mode must be one of {sorted(_SUPPORTED_STATE_MODES)}; got {state_mode!r}"
            )
        )
    return value


def pallas_projector_variant_metadata(
    *,
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "generic",
    layout_variant: str = "detector_vu",
    state_mode: str = "inline",
    gather_dtype: str = "fp32",
) -> dict[str, Any]:
    """Return normalized metadata for the Pallas variant this module can run."""
    tile_v, tile_u = _normalize_tile_shape(tile_shape)
    actual_kernel_variant = _normalize_kernel_variant(kernel_variant)
    return {
        "tile_shape": [tile_v, tile_u],
        "num_warps": _normalize_num_warps(num_warps),
        "kernel_variant": "generic" if actual_kernel_variant == "auto" else actual_kernel_variant,
        "layout_variant": _normalize_layout_variant(layout_variant),
        "state_mode": _normalize_state_mode(state_mode),
        "gather_dtype": _normalize_gather_dtype(gather_dtype),
    }


def pallas_projector_actual_variant_metadata(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    *,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "generic",
    layout_variant: str = "detector_vu",
    state_mode: str = "inline",
    gather_dtype: str = "fp32",
) -> dict[str, Any]:
    """Return normalized metadata for the selected Pallas variant."""
    requested_kernel_variant = _normalize_kernel_variant(kernel_variant)
    metadata = pallas_projector_variant_metadata(
        tile_shape=tile_shape,
        num_warps=num_warps,
        kernel_variant=kernel_variant,
        layout_variant=layout_variant,
        state_mode=state_mode,
        gather_dtype=gather_dtype,
    )
    metadata["kernel_variant"] = _select_kernel_variant(
        T,
        grid,
        detector,
        det_grid,
        requested_kernel_variant,
    )
    return metadata


def pallas_projector_actual_sinogram_variant_metadata(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    *,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "generic",
    layout_variant: str = "detector_vu",
    state_mode: str = "inline",
    gather_dtype: str = "fp32",
) -> dict[str, Any]:
    """Return normalized metadata for the selected batched-sinogram Pallas variant."""
    requested_kernel_variant = _normalize_kernel_variant(kernel_variant)
    metadata = pallas_projector_variant_metadata(
        tile_shape=tile_shape,
        num_warps=num_warps,
        kernel_variant=kernel_variant,
        layout_variant=layout_variant,
        state_mode=state_mode,
        gather_dtype=gather_dtype,
    )
    metadata["kernel_variant"] = _select_kernel_variant_for_stack(
        T_stack,
        grid,
        detector,
        det_grid,
        requested_kernel_variant,
    )
    return metadata


def pallas_projector_traversal_metadata(
    T: jnp.ndarray,
    grid: Grid,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
) -> dict[str, int]:
    """Return resolved JAX and effective Pallas traversal counts."""
    step_size_value = float(grid.vy) if step_size is None else float(step_size)
    resolved_n_steps = _resolve_n_steps(grid, step_size_value, n_steps)
    effective_n_steps = _resolve_effective_pallas_n_steps(
        T,
        grid,
        step_size_value,
        resolved_n_steps,
    )
    return {
        "resolved_n_steps": int(resolved_n_steps),
        "effective_pallas_n_steps": int(effective_n_steps),
    }


def pallas_projector_sinogram_traversal_metadata(
    T_stack: jnp.ndarray,
    grid: Grid,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
) -> dict[str, int]:
    """Return resolved and max effective traversal counts for a Pallas pose stack."""
    step_size_value = float(grid.vy) if step_size is None else float(step_size)
    resolved_n_steps = _resolve_n_steps(grid, step_size_value, n_steps)
    try:
        T_host = np.asarray(T_stack, dtype=np.float32)
    except Exception:
        T_host = np.empty((0, 4, 4), dtype=np.float32)
    if T_host.ndim != 3 or T_host.shape[1:] != (4, 4) or T_host.shape[0] == 0:
        effective_n_steps = int(resolved_n_steps)
    else:
        effective_n_steps = max(
            _resolve_effective_pallas_n_steps(
                jnp.asarray(T_host[index]),
                grid,
                step_size_value,
                resolved_n_steps,
            )
            for index in range(T_host.shape[0])
        )
    return {
        "resolved_n_steps": int(resolved_n_steps),
        "effective_pallas_n_steps": int(effective_n_steps),
    }


def _resolve_effective_pallas_n_steps(
    T: jnp.ndarray,
    grid: Grid,
    step_size: float,
    resolved_n_steps: int,
) -> int:
    try:
        T_host = np.asarray(T, dtype=np.float32)
    except Exception:
        return int(resolved_n_steps)
    if T_host.shape != (4, 4):
        return int(resolved_n_steps)

    ray_dir = T_host[:3, :3].T[:, 1]
    support_lengths = np.asarray(
        [
            (int(grid.nx) + 1) * float(grid.vx),
            (int(grid.ny) + 1) * float(grid.vy),
            (int(grid.nz) + 1) * float(grid.vz),
        ],
        dtype=np.float64,
    )
    abs_dir = np.abs(ray_dir.astype(np.float64))
    active_axes = abs_dir > 1e-8
    if not np.any(active_axes):
        return int(resolved_n_steps)

    max_path_length = float(np.min(support_lengths[active_axes] / abs_dir[active_axes]))
    # Preserve a small fp32/slab-boundary guard; per-ray n_steps_ray still masks
    # the exact active samples.
    effective_n_steps = int(math.ceil(max_path_length / float(step_size))) + 2
    return max(1, min(int(resolved_n_steps), effective_n_steps))


def _select_kernel_variant(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    requested_kernel_variant: str,
) -> str:
    if requested_kernel_variant == "generic":
        return "generic"
    supports_z_integer = _supports_z_integer4(T, grid, detector, det_grid)
    if requested_kernel_variant == "auto":
        return "z_integer4" if supports_z_integer else "generic"
    if requested_kernel_variant == "z_integer4" and supports_z_integer:
        return "z_integer4"
    raise PallasProjectorUnsupported(
        _unsupported("kernel_variant='z_integer4' requires canonical integer-z detector geometry")
    )


def _supports_z_integer4(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> bool:
    if det_grid is not None:
        try:
            _ensure_canonical_detector_grid(detector, det_grid)
        except PallasProjectorUnsupported:
            return False
    try:
        T_host = np.asarray(T, dtype=np.float64)
    except Exception:
        return False
    if T_host.shape != (4, 4) or not np.isfinite(T_host).all():
        return False

    t02 = float(T_host[0, 2])
    t12 = float(T_host[1, 2])
    t22 = float(T_host[2, 2])
    t03 = float(T_host[0, 3])
    t13 = float(T_host[1, 3])
    t23 = float(T_host[2, 3])
    tinv_z = -(t02 * t03 + t12 * t13 + t22 * t23)
    tol = 1e-5
    if abs(t02) > tol or abs(t12) > tol:
        return False

    first_z = (
        (-(float(detector.nv) / 2.0 - 0.5)) * float(detector.dv)
        + float(detector.det_center[1])
    )
    iz0 = (t22 * first_z + tinv_z - float(_grid_volume_origin(grid)[2])) / float(grid.vz)
    diz_dv = t22 * float(detector.dv) / float(grid.vz)
    return abs(iz0 - round(iz0)) <= tol and abs(diz_dv - round(diz_dv)) <= tol


def _select_kernel_variant_for_stack(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    requested_kernel_variant: str,
) -> str:
    if requested_kernel_variant == "generic":
        return "generic"
    try:
        T_host = np.asarray(T_stack, dtype=np.float32)
    except Exception:
        if requested_kernel_variant == "auto":
            return "generic"
        raise PallasProjectorUnsupported(
            _unsupported("kernel_variant='z_integer4' requires host-visible pose stack")
        ) from None
    if T_host.ndim != 3 or T_host.shape[1:] != (4, 4):
        if requested_kernel_variant == "auto":
            return "generic"
        raise PallasProjectorUnsupported(
            _unsupported("kernel_variant='z_integer4' requires pose stack shape (n_views, 4, 4)")
        )
    supports_all = all(
        _supports_z_integer4(jnp.asarray(T_host[index]), grid, detector, det_grid)
        for index in range(T_host.shape[0])
    )
    if requested_kernel_variant == "auto":
        return "z_integer4" if supports_all else "generic"
    if requested_kernel_variant == "z_integer4" and supports_all:
        return "z_integer4"
    raise PallasProjectorUnsupported(
        _unsupported("kernel_variant='z_integer4' requires all views to use integer-z detector geometry")
    )


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
    num_warps: int,
    kernel_variant: str,
    layout_variant: str,
    state_mode: str,
) -> tuple[int, int, int, int, int, int, float, int, int, tuple[int, int], int, int, int]:
    nx, ny, nz = validate_volume(
        volume,
        grid,
        context="forward_project_view_T_pallas",
        name="volume",
    )
    nv, nu = validate_detector(detector, "forward_project_view_T_pallas")
    validate_pose_matrix(T, context="forward_project_view_T_pallas")
    _ensure_float32_volume(volume)
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
        state_mode=state_mode,
        gather_dtype=gather_dtype,
    )
    if variant["state_mode"] != "inline" and variant["layout_variant"] != "detector_vu":
        raise PallasProjectorUnsupported(
            _unsupported("cached traversal state currently supports layout_variant='detector_vu' only")
        )
    tile_v, tile_u = variant["tile_shape"]
    if not interpret and jax.default_backend() == "cpu":
        raise PallasProjectorUnsupported(
            _unsupported("real Pallas lowering is unavailable on CPU; pass interpret=True")
        )

    if step_size is None:
        step_size_value = float(grid.vy)
    else:
        step_size_value = float(step_size)
    n_steps_value = _resolve_n_steps(grid, step_size_value, n_steps)
    effective_n_steps_value = _resolve_effective_pallas_n_steps(
        T,
        grid,
        step_size_value,
        n_steps_value,
    )
    kernel_variant_id = _KERNEL_VARIANT_IDS[str(variant["kernel_variant"])]
    layout_variant_id = _LAYOUT_VARIANT_IDS[str(variant["layout_variant"])]
    return nx, ny, nz, nv, nu, nx * ny * nz, step_size_value, n_steps_value, effective_n_steps_value, (
        tile_v,
        tile_u,
    ), int(variant["num_warps"]), kernel_variant_id, layout_variant_id


def _validate_public_sinogram_call(
    T_stack: jnp.ndarray,
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
    num_warps: int,
    kernel_variant: str,
    layout_variant: str,
    state_mode: str,
) -> tuple[int, int, int, int, int, int, int, float, int, int, tuple[int, int], int, int, int]:
    nx, ny, nz = validate_volume(
        volume,
        grid,
        context="forward_project_views_T_pallas",
        name="volume",
    )
    nv, nu = validate_detector(detector, "forward_project_views_T_pallas")
    shape = getattr(T_stack, "shape", None)
    if shape is None or len(shape) != 3:
        validate_pose_stack(T_stack, 0, context="forward_project_views_T_pallas")
    n_views = int(shape[0])
    if n_views <= 0:
        raise PallasProjectorUnsupported(_unsupported("pose stack must contain at least one view"))
    validate_pose_stack(T_stack, n_views, context="forward_project_views_T_pallas")
    _ensure_float32_volume(volume)
    _ensure_canonical_detector_grid(detector, det_grid)
    variant = pallas_projector_actual_sinogram_variant_metadata(
        T_stack,
        grid,
        detector,
        det_grid=det_grid,
        tile_shape=tile_shape,
        num_warps=num_warps,
        kernel_variant=kernel_variant,
        layout_variant=layout_variant,
        state_mode=state_mode,
        gather_dtype=gather_dtype,
    )
    if variant["state_mode"] != "inline":
        raise PallasProjectorUnsupported(
            _unsupported("batched sinogram Pallas supports state_mode='inline' only")
        )
    tile_v, tile_u = variant["tile_shape"]
    if not interpret and jax.default_backend() == "cpu":
        raise PallasProjectorUnsupported(
            _unsupported("real Pallas lowering is unavailable on CPU; pass interpret=True")
        )

    if step_size is None:
        step_size_value = float(grid.vy)
    else:
        step_size_value = float(step_size)
    n_steps_value = _resolve_n_steps(grid, step_size_value, n_steps)
    effective_n_steps_value = max(
        1,
        max(
            _resolve_effective_pallas_n_steps(
                T_stack[index],
                grid,
                step_size_value,
                n_steps_value,
            )
            for index in range(n_views)
        ),
    )
    kernel_variant_id = _KERNEL_VARIANT_IDS[str(variant["kernel_variant"])]
    layout_variant_id = _LAYOUT_VARIANT_IDS[str(variant["layout_variant"])]
    return (
        nx,
        ny,
        nz,
        nv,
        nu,
        n_views,
        nx * ny * nz,
        step_size_value,
        n_steps_value,
        effective_n_steps_value,
        (tile_v, tile_u),
        int(variant["num_warps"]),
        kernel_variant_id,
        layout_variant_id,
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
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "generic",
    layout_variant: str = "detector_vu",
    state_mode: str = "inline",
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
            num_warps=num_warps,
            kernel_variant=kernel_variant,
            layout_variant=layout_variant,
            state_mode=state_mode,
        )
    except PallasProjectorUnsupported as exc:
        return str(exc)
    except ValueError as exc:
        return _unsupported(str(exc))
    return None


def pallas_projector_sinogram_unsupported_reason(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "generic",
    layout_variant: str = "detector_vu",
    state_mode: str = "inline",
) -> str | None:
    """Return a benchmark-friendly unsupported reason for batched sinogram Pallas."""
    try:
        _validate_public_sinogram_call(
            T_stack,
            grid,
            detector,
            volume,
            step_size=step_size,
            n_steps=n_steps,
            gather_dtype=gather_dtype,
            det_grid=det_grid,
            interpret=False,
            tile_shape=tile_shape,
            num_warps=num_warps,
            kernel_variant=kernel_variant,
            layout_variant=layout_variant,
            state_mode=state_mode,
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
    active_mask: jnp.ndarray | None = None,
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
        if active_mask is not None:
            inb = inb & active_mask
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


def _trilinear_load_z_integer(
    volume_ref: Any,
    ix_f: jnp.ndarray,
    iy_f: jnp.ndarray,
    iz_f: jnp.ndarray,
    *,
    nx: int,
    ny: int,
    nz: int,
    active_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    fx = jnp.floor(ix_f).astype(jnp.int32)
    fy = jnp.floor(iy_f).astype(jnp.int32)
    iz = jnp.floor(iz_f + jnp.float32(0.5)).astype(jnp.int32)
    cx, cy = fx + 1, fy + 1

    wx1 = ix_f - fx.astype(jnp.float32)
    wy1 = iy_f - fy.astype(jnp.float32)
    wx0 = jnp.float32(1.0) - wx1
    wy0 = jnp.float32(1.0) - wy1

    def gather(ix: jnp.ndarray, iy: jnp.ndarray) -> jnp.ndarray:
        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        if active_mask is not None:
            inb = inb & active_mask
        idx = ix * (ny * nz) + iy * nz + iz
        idx = jnp.clip(idx, 0, (nx * ny * nz) - 1)
        return plt.load(volume_ref.at[idx], mask=inb, other=0.0)

    c00 = gather(fx, fy) * (wx0 * wy0)
    c01 = gather(fx, cy) * (wx0 * wy1)
    c10 = gather(cx, fy) * (wx1 * wy0)
    c11 = gather(cx, cy) * (wx1 * wy1)
    return c00 + c01 + c10 + c11


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
    kernel_variant_id: int,
    layout_variant_id: int,
    unroll: int | None,
) -> None:
    tile_v_start = pl.program_id(0) * tile_v
    tile_u_start = pl.program_id(1) * tile_u
    if layout_variant_id == _LAYOUT_VARIANT_IDS["detector_uv"]:
        det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[:, jnp.newaxis]
        det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[jnp.newaxis, :]
    else:
        det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[jnp.newaxis, :]
        det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[:, jnp.newaxis]
    in_detector = (det_u < nu) & (det_v < nv)

    xr = (
        (det_u.astype(jnp.float32) - jnp.float32(nu / 2.0 - 0.5)) * jnp.float32(du)
        + jnp.float32(det_center_x)
    )
    zr = (
        (det_v.astype(jnp.float32) - jnp.float32(nv / 2.0 - 0.5)) * jnp.float32(dv)
        + jnp.float32(det_center_z)
    )

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
        active = (step_idx < n_steps_ray) & in_detector
        if kernel_variant_id == _KERNEL_VARIANT_IDS["z_integer4"]:
            sample = _trilinear_load_z_integer(
                volume_ref,
                ix,
                iy,
                iz,
                nx=nx,
                ny=ny,
                nz=nz,
                active_mask=active,
            )
        else:
            sample = _trilinear_load(
                volume_ref,
                ix,
                iy,
                iz,
                nx=nx,
                ny=ny,
                nz=nz,
                active_mask=active,
            )
        return (
            acc + sample.astype(jnp.float32) * active.astype(jnp.float32) * step_size32,
            ix + dix,
            iy + diy,
            iz + diz,
        )

    init = (
        jnp.zeros_like(ix0, dtype=jnp.float32),
        ix0,
        iy0,
        iz0,
    )
    if unroll is None:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init)
    else:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init, unroll=unroll)
    if layout_variant_id == _LAYOUT_VARIANT_IDS["detector_uv"]:
        out_ref[...] = acc.T.astype(jnp.float32)
    else:
        out_ref[...] = acc.astype(jnp.float32)


def _projector_views_kernel(
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
    kernel_variant_id: int,
    layout_variant_id: int,
    unroll: int | None,
) -> None:
    view_idx = pl.program_id(0)
    tile_v_start = pl.program_id(1) * tile_v
    tile_u_start = pl.program_id(2) * tile_u
    if layout_variant_id == _LAYOUT_VARIANT_IDS["detector_uv"]:
        det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[:, jnp.newaxis]
        det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[jnp.newaxis, :]
    else:
        det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[jnp.newaxis, :]
        det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[:, jnp.newaxis]
    in_detector = (det_u < nu) & (det_v < nv)

    xr = (
        (det_u.astype(jnp.float32) - jnp.float32(nu / 2.0 - 0.5)) * jnp.float32(du)
        + jnp.float32(det_center_x)
    )
    zr = (
        (det_v.astype(jnp.float32) - jnp.float32(nv / 2.0 - 0.5)) * jnp.float32(dv)
        + jnp.float32(det_center_z)
    )

    def tload(row: int, col: int):
        return plt.load(T_ref.at[view_idx, row, col])

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
        active = (step_idx < n_steps_ray) & in_detector
        if kernel_variant_id == _KERNEL_VARIANT_IDS["z_integer4"]:
            sample = _trilinear_load_z_integer(
                volume_ref,
                ix,
                iy,
                iz,
                nx=nx,
                ny=ny,
                nz=nz,
                active_mask=active,
            )
        else:
            sample = _trilinear_load(
                volume_ref,
                ix,
                iy,
                iz,
                nx=nx,
                ny=ny,
                nz=nz,
                active_mask=active,
            )
        return (
            acc + sample.astype(jnp.float32) * active.astype(jnp.float32) * step_size32,
            ix + dix,
            iy + diy,
            iz + diz,
        )

    init = (
        jnp.zeros_like(ix0, dtype=jnp.float32),
        ix0,
        iy0,
        iz0,
    )
    if unroll is None:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init)
    else:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init, unroll=unroll)
    if layout_variant_id == _LAYOUT_VARIANT_IDS["detector_uv"]:
        out_ref[0, :, :] = acc.T.astype(jnp.float32)
    else:
        out_ref[0, :, :] = acc.astype(jnp.float32)


def _projector_residual_sse_kernel(
    T_ref: Any,
    volume_ref: Any,
    target_ref: Any,
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
    kernel_variant_id: int,
    layout_variant_id: int,
    unroll: int | None,
) -> None:
    view_idx = pl.program_id(0)
    tile_v_idx = pl.program_id(1)
    tile_u_idx = pl.program_id(2)
    tile_v_start = tile_v_idx * tile_v
    tile_u_start = tile_u_idx * tile_u
    if layout_variant_id == _LAYOUT_VARIANT_IDS["detector_uv"]:
        det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[:, jnp.newaxis]
        det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[jnp.newaxis, :]
    else:
        det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[jnp.newaxis, :]
        det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[:, jnp.newaxis]
    in_detector = (det_u < nu) & (det_v < nv)

    xr = (
        (det_u.astype(jnp.float32) - jnp.float32(nu / 2.0 - 0.5)) * jnp.float32(du)
        + jnp.float32(det_center_x)
    )
    zr = (
        (det_v.astype(jnp.float32) - jnp.float32(nv / 2.0 - 0.5)) * jnp.float32(dv)
        + jnp.float32(det_center_z)
    )

    def tload(row: int, col: int):
        return plt.load(T_ref.at[view_idx, row, col])

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
        active = (step_idx < n_steps_ray) & in_detector
        if kernel_variant_id == _KERNEL_VARIANT_IDS["z_integer4"]:
            sample = _trilinear_load_z_integer(
                volume_ref,
                ix,
                iy,
                iz,
                nx=nx,
                ny=ny,
                nz=nz,
                active_mask=active,
            )
        else:
            sample = _trilinear_load(
                volume_ref,
                ix,
                iy,
                iz,
                nx=nx,
                ny=ny,
                nz=nz,
                active_mask=active,
            )
        return (
            acc + sample.astype(jnp.float32) * active.astype(jnp.float32) * step_size32,
            ix + dix,
            iy + diy,
            iz + diz,
        )

    init = (
        jnp.zeros_like(ix0, dtype=jnp.float32),
        ix0,
        iy0,
        iz0,
    )
    if unroll is None:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init)
    else:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init, unroll=unroll)
    target = plt.load(target_ref.at[view_idx, det_v, det_u], mask=in_detector, other=0.0)
    residual = jnp.where(in_detector, acc.astype(jnp.float32) - target.astype(jnp.float32), 0.0)
    out_ref[0, 0, 0] = jnp.sum(residual * residual).astype(jnp.float32)


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
    kernel_variant: str = "generic",
    layout_variant: str = "detector_vu",
) -> PallasForwardProjectorTraversalState:
    """Prepare fixed-geometry traversal state for the experimental cached Pallas path."""
    nv, nu = validate_detector(detector, "prepare_forward_project_view_T_pallas_state")
    validate_pose_matrix(T, context="prepare_forward_project_view_T_pallas_state")
    _normalize_gather_dtype(gather_dtype)
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
            _unsupported("cached traversal state currently supports layout_variant='detector_vu' only")
        )

    if step_size is None:
        step_size_value = float(grid.vy)
    else:
        step_size_value = float(step_size)
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
        ix0=jnp.ravel(jnp.asarray(ix0, dtype=jnp.float32), order="C"),
        iy0=jnp.ravel(jnp.asarray(iy0, dtype=jnp.float32), order="C"),
        iz0=jnp.ravel(jnp.asarray(iz0, dtype=jnp.float32), order="C"),
        n_steps_ray=jnp.ravel(jnp.asarray(n_steps_ray, dtype=jnp.int32), order="C"),
        dix=float(np.ravel(np.asarray(dix, dtype=np.float32))[0]),
        diy=float(np.ravel(np.asarray(diy, dtype=np.float32))[0]),
        diz=float(np.ravel(np.asarray(diz, dtype=np.float32))[0]),
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
            jnp.ravel(volume, order="C"),
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
    kernel_variant: str = "generic",
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
        active = (step_idx < n_steps_ray) & in_detector
        if kernel_variant_id == _KERNEL_VARIANT_IDS["z_integer4"]:
            sample = _trilinear_load_z_integer(
                volume_ref,
                ix,
                iy,
                iz,
                nx=nx,
                ny=ny,
                nz=nz,
                active_mask=active,
            )
        else:
            sample = _trilinear_load(
                volume_ref,
                ix,
                iy,
                iz,
                nx=nx,
                ny=ny,
                nz=nz,
                active_mask=active,
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
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init)
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
        jnp.ravel(volume, order="C"),
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
    kernel_variant: str = "generic",
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
        volume_size,
        step_size_value,
        n_steps_value,
        effective_n_steps_value,
        (tile_v, tile_u),
        num_warps_value,
        kernel_variant_id,
        layout_variant_id,
    ) = (
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
            num_warps=num_warps,
            kernel_variant=kernel_variant,
            layout_variant=layout_variant,
            state_mode=state_mode,
        )
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
        jnp.ravel(volume, order="C"),
    )


def forward_project_views_T_pallas(
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
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "generic",
    layout_variant: str = "detector_vu",
    state_mode: str = "inline",
) -> jnp.ndarray:
    """Forward project a stack of views using one experimental batched Pallas launch."""
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
    vol_origin = _grid_volume_origin(grid)
    kernel = functools.partial(
        _projector_views_kernel,
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
    grid_shape = (n_views, math.ceil(nv / tile_v), math.ceil(nu / tile_u))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n_views, nv, nu), jnp.float32),
        grid=grid_shape,
        in_specs=[
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec((1, tile_v, tile_u), lambda view, pv, pu: (view, pv, pu)),
        interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=num_warps_value),
        name="tomojax_forward_project_views_T_pallas",
    )(
        jnp.asarray(T_stack, dtype=jnp.float32),
        jnp.ravel(volume, order="C"),
    )


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
    kernel_variant: str = "generic",
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
        jnp.ravel(volume, order="C"),
        jnp.asarray(target, dtype=jnp.float32),
    )
    return jnp.sum(partials, dtype=jnp.float32)
