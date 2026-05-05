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
    validate_detector_image,
    validate_grid,
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
_GATHER_DTYPE_ALIASES = {
    "fp32": "fp32",
    "float32": "fp32",
    "single": "fp32",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
}
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
    gather_dtype: str


@dataclass(frozen=True)
class PallasForwardProjectorStackTraversalState:
    """Prepared per-ray traversal state for fixed pose-stack Pallas benchmarking."""

    ix0: jnp.ndarray
    iy0: jnp.ndarray
    iz0: jnp.ndarray
    n_steps_ray: jnp.ndarray
    dix: jnp.ndarray
    diy: jnp.ndarray
    diz: jnp.ndarray
    step_size: float
    n_steps: int
    resolved_n_steps: int
    nx: int
    ny: int
    nz: int
    nv: int
    nu: int
    n_views: int
    tile_shape: tuple[int, int]
    num_warps: int
    kernel_variant: str
    kernel_variant_id: int
    gather_dtype: str


def _unsupported(message: str) -> str:
    return f"pallas_projector_unsupported: {message}"


def _normalize_gather_dtype(gather_dtype: str) -> str:
    if not isinstance(gather_dtype, str):
        raise PallasProjectorUnsupported(
            _unsupported(f"gather_dtype must be a string; got {type(gather_dtype).__name__}")
        )
    gd = gather_dtype.lower()
    if gd == "auto":
        try:
            platform = jax.devices()[0].platform if jax.devices() else "cpu"
        except Exception:
            platform = "cpu"
        if platform == "tpu":
            return "bf16"
        if platform == "gpu":
            return "fp16"
        return "fp32"
    if gd not in _GATHER_DTYPE_ALIASES:
        raise PallasProjectorUnsupported(
            _unsupported(
                "gather_dtype must be one of 'auto', 'fp32', 'float32', 'single', "
                "'bf16', 'bfloat16', 'fp16', 'float16', or 'half'; "
                f"got {gather_dtype!r}"
            )
        )
    return _GATHER_DTYPE_ALIASES[gd]


def _pallas_gather_jnp_dtype(gather_dtype: str) -> jnp.dtype:
    normalized = _normalize_gather_dtype(gather_dtype)
    if normalized == "bf16":
        return jnp.bfloat16
    if normalized == "fp16":
        return jnp.float16
    return jnp.float32


def _prepare_volume_for_pallas_gather(volume: jnp.ndarray, gather_dtype: str) -> jnp.ndarray:
    target = _pallas_gather_jnp_dtype(gather_dtype)
    vol_cast = volume if volume.dtype == target else volume.astype(target)
    return jnp.ravel(vol_cast, order="C")


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


def _divisors_at_most(value: int, limit: int) -> tuple[int, ...]:
    limit = max(1, min(int(value), int(limit)))
    return tuple(candidate for candidate in range(limit, 0, -1) if int(value) % candidate == 0)


def _is_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 0 and (value & (value - 1)) == 0


def _largest_power2_tile_divisors(
    *,
    nv: int,
    nu: int,
    tile_v: int,
    tile_u: int,
) -> list[int]:
    best = (1, 1)
    best_area = 1
    for candidate_v in _divisors_at_most(nv, tile_v):
        for candidate_u in _divisors_at_most(nu, tile_u):
            area = int(candidate_v) * int(candidate_u)
            if _is_power_of_two(area) and area > best_area:
                best = (int(candidate_v), int(candidate_u))
                best_area = int(area)
                break
    return [best[0], best[1]]


def _safe_detector_tile_shape(
    tile_shape: list[int],
    detector: Detector,
    *,
    max_generic_tile_u: int | None = None,
    allow_remainder_tiles: bool = False,
) -> list[int]:
    """Resolve detector tiles to exact detector divisors for real Pallas lowering."""
    tile_v = int(tile_shape[0])
    tile_u = int(tile_shape[1])
    if max_generic_tile_u is not None:
        tile_u = min(tile_u, int(max_generic_tile_u))
    exact = _largest_power2_tile_divisors(
        nv=int(detector.nv),
        nu=int(detector.nu),
        tile_v=tile_v,
        tile_u=tile_u,
    )
    if int(exact[0]) * int(exact[1]) > 1:
        return exact
    if allow_remainder_tiles:
        remainder_tile_v = max(1, min(int(detector.nv), int(tile_v)))
        remainder_tile_u = max(1, min(int(detector.nu), int(tile_u)))
        if _is_power_of_two(remainder_tile_v * remainder_tile_u):
            return [remainder_tile_v, remainder_tile_u]
    # JAX Pallas Triton lowering checks load/store tensor sizes are powers of two
    # (`jax/_src/pallas/triton/lowering.py::_check_tensor_size` in JAX 0.10.0).
    return exact


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


def _trilinear_load_when_tile_active(
    volume_ref: Any,
    ix: jnp.ndarray,
    iy: jnp.ndarray,
    iz: jnp.ndarray,
    *,
    nx: int,
    ny: int,
    nz: int,
    active: jnp.ndarray,
    kernel_variant_id: int,
) -> jnp.ndarray:
    def do_load(_):
        if kernel_variant_id == _KERNEL_VARIANT_IDS["z_integer4"]:
            return _trilinear_load_z_integer(
                volume_ref,
                ix,
                iy,
                iz,
                nx=nx,
                ny=ny,
                nz=nz,
            )
        return _trilinear_load(
            volume_ref,
            ix,
            iy,
            iz,
            nx=nx,
            ny=ny,
            nz=nz,
        )

    # Pallas/Triton lowering in JAX 0.10.0 supports reduce_max but not reduce_or.
    tile_active = jnp.max(active.astype(jnp.int32)) != jnp.int32(0)
    return jax.lax.cond(
        tile_active,
        do_load,
        lambda _: jnp.zeros_like(ix, dtype=jnp.float32),
        operand=None,
    )


def pallas_projector_variant_metadata(
    *,
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "auto",
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
    kernel_variant: str = "auto",
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
    metadata["tile_shape"] = _safe_detector_tile_shape(
        metadata["tile_shape"],
        detector,
        max_generic_tile_u=8 if metadata["kernel_variant"] == "generic" else None,
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
    kernel_variant: str = "auto",
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
    metadata["tile_shape"] = _safe_detector_tile_shape(
        metadata["tile_shape"],
        detector,
        max_generic_tile_u=8 if metadata["kernel_variant"] == "generic" else None,
        allow_remainder_tiles=metadata["state_mode"] == "cached"
        and metadata["layout_variant"] == "detector_vu",
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


def _resolve_effective_pallas_n_steps_for_stack(
    T_stack: jnp.ndarray,
    grid: Grid,
    step_size: float,
    resolved_n_steps: int,
) -> int:
    try:
        T_host = np.asarray(T_stack, dtype=np.float32)
    except Exception:
        return int(resolved_n_steps)
    if T_host.ndim != 3 or T_host.shape[1:] != (4, 4) or T_host.shape[0] == 0:
        return int(resolved_n_steps)

    ray_dirs = T_host[:, 1, :3].astype(np.float64)
    support_lengths = np.asarray(
        [
            (int(grid.nx) + 1) * float(grid.vx),
            (int(grid.ny) + 1) * float(grid.vy),
            (int(grid.nz) + 1) * float(grid.vz),
        ],
        dtype=np.float64,
    )
    abs_dir = np.abs(ray_dirs)
    active_axes = abs_dir > 1e-8
    if not np.any(active_axes, axis=1).all():
        return int(resolved_n_steps)

    path_lengths = np.min(
        np.where(active_axes, support_lengths[np.newaxis, :] / np.maximum(abs_dir, 1e-30), np.inf),
        axis=1,
    )
    max_path_length = float(np.max(path_lengths))
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


def _supports_z_integer4_for_stack(
    T_stack: jnp.ndarray,
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
        T_host = np.asarray(T_stack, dtype=np.float64)
    except Exception:
        return False
    if T_host.ndim != 3 or T_host.shape[1:] != (4, 4):
        return False
    if T_host.shape[0] == 0 or not np.isfinite(T_host).all():
        return False

    tol = 1e-5
    if np.any(np.abs(T_host[:, 0, 2]) > tol) or np.any(np.abs(T_host[:, 1, 2]) > tol):
        return False

    tinv_z = -(
        T_host[:, 0, 2] * T_host[:, 0, 3]
        + T_host[:, 1, 2] * T_host[:, 1, 3]
        + T_host[:, 2, 2] * T_host[:, 2, 3]
    )
    first_z = (
        (-(float(detector.nv) / 2.0 - 0.5)) * float(detector.dv)
        + float(detector.det_center[1])
    )
    iz0 = (T_host[:, 2, 2] * first_z + tinv_z - float(_grid_volume_origin(grid)[2])) / float(
        grid.vz
    )
    diz_dv = T_host[:, 2, 2] * float(detector.dv) / float(grid.vz)
    return bool(
        np.all(np.abs(iz0 - np.round(iz0)) <= tol)
        and np.all(np.abs(diz_dv - np.round(diz_dv)) <= tol)
    )


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
    supports_all = _supports_z_integer4_for_stack(T_host, grid, detector, det_grid)
    if requested_kernel_variant == "auto":
        return "z_integer4" if supports_all else "generic"
    if requested_kernel_variant == "z_integer4" and supports_all:
        return "z_integer4"
    raise PallasProjectorUnsupported(
        _unsupported("kernel_variant='z_integer4' requires all views to use integer-z detector geometry")
    )


def _supports_parallel_z_rotation_stack(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> bool:
    if not _supports_z_integer4_for_stack(T_stack, grid, detector, det_grid):
        return False
    try:
        T_host = np.asarray(T_stack, dtype=np.float64)
    except Exception:
        return False
    if T_host.ndim != 3 or T_host.shape[1:] != (4, 4):
        return False
    tol = 1e-5
    c = T_host[:, 0, 0]
    s = T_host[:, 1, 0]
    return bool(
        np.all(np.abs(T_host[:, 0, 1] + s) <= tol)
        and np.all(np.abs(T_host[:, 1, 1] - c) <= tol)
        and np.all(np.abs(T_host[:, 0, 2]) <= tol)
        and np.all(np.abs(T_host[:, 1, 2]) <= tol)
        and np.all(np.abs(T_host[:, 2, 0]) <= tol)
        and np.all(np.abs(T_host[:, 2, 1]) <= tol)
        and np.all(np.abs(T_host[:, 2, 2] - 1.0) <= tol)
        and np.all(np.abs(T_host[:, :3, 3]) <= tol)
        and np.all(np.abs(T_host[:, 3, :3]) <= tol)
        and np.all(np.abs(T_host[:, 3, 3] - 1.0) <= tol)
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
    effective_n_steps_value = _resolve_effective_pallas_n_steps_for_stack(
        T_stack,
        grid,
        step_size_value,
        n_steps_value,
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
    kernel_variant: str = "auto",
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
    kernel_variant: str = "auto",
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


def _trilinear_load_z_integer(
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
    iz = jnp.floor(iz_f + jnp.float32(0.5)).astype(jnp.int32)
    cx, cy = fx + 1, fy + 1

    wx1 = ix_f - fx.astype(jnp.float32)
    wy1 = iy_f - fy.astype(jnp.float32)
    wx0 = jnp.float32(1.0) - wx1
    wy0 = jnp.float32(1.0) - wy1

    def gather(ix: jnp.ndarray, iy: jnp.ndarray) -> jnp.ndarray:
        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        idx = ix * (ny * nz) + iy * nz + iz
        idx = jnp.clip(idx, 0, (nx * ny * nz) - 1)
        return plt.load(volume_ref.at[idx], mask=inb, other=0.0)

    c00 = gather(fx, fy) * (wx0 * wy0)
    c01 = gather(fx, cy) * (wx0 * wy1)
    c10 = gather(cx, fy) * (wx1 * wy0)
    c11 = gather(cx, cy) * (wx1 * wy1)
    return c00 + c01 + c10 + c11


def _trilinear_atomic_add(
    out_ref: Any,
    ray_vals: jnp.ndarray,
    ix_f: jnp.ndarray,
    iy_f: jnp.ndarray,
    iz_f: jnp.ndarray,
    *,
    nx: int,
    ny: int,
    nz: int,
    active: jnp.ndarray,
) -> None:
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

    def add(ix: jnp.ndarray, iy: jnp.ndarray, iz: jnp.ndarray, weight: jnp.ndarray) -> None:
        inb = active & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        idx = ix * (ny * nz) + iy * nz + iz
        idx = jnp.clip(idx, 0, (nx * ny * nz) - 1)
        plt.atomic_add(out_ref, (idx,), ray_vals * weight, mask=inb)

    add(fx, fy, fz, wx0 * wy0 * wz0)
    add(fx, fy, cz, wx0 * wy0 * wz1)
    add(fx, cy, fz, wx0 * wy1 * wz0)
    add(fx, cy, cz, wx0 * wy1 * wz1)
    add(cx, fy, fz, wx1 * wy0 * wz0)
    add(cx, fy, cz, wx1 * wy0 * wz1)
    add(cx, cy, fz, wx1 * wy1 * wz0)
    add(cx, cy, cz, wx1 * wy1 * wz1)


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
        tile_steps = jnp.minimum(
            jnp.max(jnp.where(valid_rays, n_steps_ray, 0)),
            jnp.asarray(n_steps, dtype=jnp.int32),
        )
        acc, _, _, _ = jax.lax.fori_loop(0, tile_steps, body, init)
    else:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init, unroll=unroll)
    if layout_variant_id == _LAYOUT_VARIANT_IDS["detector_uv"]:
        out_ref[...] = acc.T.astype(jnp.float32)
    else:
        out_ref[...] = acc.astype(jnp.float32)


def _backproject_kernel(
    T_ref: Any,
    image_ref: Any,
    _init_ref: Any,
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
    last_step = jnp.float32(max(n_steps - 1, 0))
    ray_vals = plt.load(image_ref.at[det_v, det_u], mask=in_detector, other=0.0) * step_size32

    def body(s, carry):
        ix, iy, iz = carry
        original_step = jnp.int32(n_steps - 1) - s
        active = in_detector & (original_step < n_steps_ray)
        _trilinear_atomic_add(
            out_ref,
            ray_vals,
            ix,
            iy,
            iz,
            nx=nx,
            ny=ny,
            nz=nz,
            active=active,
        )
        return ix - dix, iy - diy, iz - diz

    init = (
        ix0 + dix * last_step,
        iy0 + diy * last_step,
        iz0 + diz * last_step,
    )
    if unroll is None:
        jax.lax.fori_loop(0, n_steps, body, init)
    else:
        jax.lax.fori_loop(0, n_steps, body, init, unroll=unroll)


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
        tile_steps = jnp.minimum(
            jnp.max(jnp.where(valid_rays, n_steps_ray, 0)),
            jnp.asarray(n_steps, dtype=jnp.int32),
        )
        acc, _, _, _ = jax.lax.fori_loop(0, tile_steps, body, init)
    else:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init, unroll=unroll)
    if layout_variant_id == _LAYOUT_VARIANT_IDS["detector_uv"]:
        out_ref[...] = acc.T[jnp.newaxis, :, :].astype(jnp.float32)
    else:
        out_ref[...] = acc[jnp.newaxis, :, :].astype(jnp.float32)


def _projector_parallel_z_views_kernel(
    cos_ref: Any,
    sin_ref: Any,
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
    view_idx = pl.program_id(0)
    tile_v_start = pl.program_id(1) * tile_v
    tile_u_start = pl.program_id(2) * tile_u
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
    c = plt.load(cos_ref.at[view_idx])
    s = plt.load(sin_ref.at[view_idx])

    base_x = c * xr
    base_y = -s * xr
    base_z = zr
    ey_x = s
    ey_y = c

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
    z_inside = (base_z >= lower_z) & (base_z <= upper_z)
    y_entry = jnp.maximum(lo_x, lo_y)
    y_exit = jnp.minimum(hi_x, hi_y)
    path_length = jnp.maximum(jnp.float32(0.0), y_exit - y_entry)
    valid_rays = z_inside & (path_length > jnp.float32(0.0))
    y_start = jnp.where(valid_rays, y_entry, jnp.float32(0.0))
    step_size32 = jnp.float32(step_size)
    n_steps_ray = jnp.where(
        valid_rays,
        jnp.ceil(path_length / step_size32).astype(jnp.int32),
        jnp.int32(0),
    )

    q0_x = base_x + y_start * ey_x
    q0_y = base_y + y_start * ey_y
    ix0 = (q0_x - jnp.float32(vol_origin_x)) / jnp.float32(vx)
    iy0 = (q0_y - jnp.float32(vol_origin_y)) / jnp.float32(vy)
    iz0 = (base_z - jnp.float32(vol_origin_z)) / jnp.float32(vz)
    dix = step_size32 * ey_x / jnp.float32(vx)
    diy = step_size32 * ey_y / jnp.float32(vy)

    def body(step_idx, carry):
        acc, ix, iy = carry
        active = step_idx < n_steps_ray
        sample = _trilinear_load_z_integer(
            volume_ref,
            ix,
            iy,
            iz0,
            nx=nx,
            ny=ny,
            nz=nz,
        )
        return (
            acc + sample.astype(jnp.float32) * active.astype(jnp.float32) * step_size32,
            ix + dix,
            iy + diy,
        )

    init = (
        jnp.zeros_like(ix0, dtype=jnp.float32),
        ix0,
        iy0,
    )
    if unroll is None:
        acc, _, _ = jax.lax.fori_loop(0, n_steps, body, init)
    else:
        acc, _, _ = jax.lax.fori_loop(0, n_steps, body, init, unroll=unroll)
    out_ref[...] = jnp.where(in_detector, acc.astype(jnp.float32), 0.0)[jnp.newaxis, :, :]


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


def _projector_loss_grad_kernel(
    T_ref: Any,
    volume_ref: Any,
    target_ref: Any,
    weights_ref: Any,
    _grad_init_ref: Any,
    loss_ref: Any,
    grad_ref: Any,
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
    compute_loss: bool,
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

    def fwd_body(step_idx, carry):
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
        tile_steps = jnp.minimum(
            jnp.max(jnp.where(in_detector, n_steps_ray, 0)),
            jnp.asarray(n_steps, dtype=jnp.int32),
        )
        acc, _, _, _ = jax.lax.fori_loop(0, tile_steps, fwd_body, init)
    else:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, fwd_body, init, unroll=unroll)

    target = plt.load(target_ref.at[view_idx, det_v, det_u], mask=in_detector, other=0.0)
    weight = plt.load(weights_ref.at[view_idx, 0, 0])
    raw_residual = jnp.where(in_detector, acc.astype(jnp.float32) - target.astype(jnp.float32), 0.0)
    weighted_residual = raw_residual * weight
    if compute_loss:
        loss_ref[0, 0, 0] = (
            jnp.float32(0.5) * jnp.sum(weighted_residual * weighted_residual).astype(jnp.float32)
        )
    else:
        loss_ref[0, 0, 0] = jnp.float32(0.0)
    grad_residual = raw_residual * weight * weight * step_size32

    def bwd_body(s, carry):
        ix, iy, iz = carry
        original_step = jnp.int32(max(n_steps - 1, 0)) - s
        active = in_detector & (original_step < n_steps_ray)
        _trilinear_atomic_add(
            grad_ref,
            grad_residual,
            ix,
            iy,
            iz,
            nx=nx,
            ny=ny,
            nz=nz,
            active=active,
        )
        return ix - dix, iy - diy, iz - diz

    if unroll is None:
        tile_steps = jnp.minimum(
            jnp.max(jnp.where(in_detector, n_steps_ray, 0)),
            jnp.asarray(n_steps, dtype=jnp.int32),
        )
        tile_last_step = jnp.maximum(tile_steps - jnp.int32(1), jnp.int32(0)).astype(
            jnp.float32
        )

        def bwd_tile_body(s, carry):
            ix, iy, iz = carry
            original_step = tile_steps - jnp.int32(1) - s
            active = in_detector & (original_step < n_steps_ray)
            _trilinear_atomic_add(
                grad_ref,
                grad_residual,
                ix,
                iy,
                iz,
                nx=nx,
                ny=ny,
                nz=nz,
                active=active,
            )
            return ix - dix, iy - diy, iz - diz

        bwd_tile_init = (
            ix0 + dix * tile_last_step,
            iy0 + diy * tile_last_step,
            iz0 + diz * tile_last_step,
        )
        jax.lax.fori_loop(0, tile_steps, bwd_tile_body, bwd_tile_init)
    else:
        last_step = jnp.float32(max(n_steps - 1, 0))
        bwd_init = (
            ix0 + dix * last_step,
            iy0 + diy * last_step,
            iz0 + diz * last_step,
        )
        jax.lax.fori_loop(0, n_steps, bwd_body, bwd_init, unroll=unroll)


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
        gather_dtype=normalized_gather_dtype,
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
        _prepare_volume_for_pallas_gather(volume, gather_dtype),
    )


def prepare_forward_project_views_T_pallas_state(
    T_stack: jnp.ndarray,
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
) -> PallasForwardProjectorStackTraversalState:
    """Prepare fixed pose-stack traversal state for repeated Pallas sinogram calls."""
    (
        nx,
        ny,
        nz,
        nv,
        nu,
        n_views,
        _volume_size,
        step_size_value,
        resolved_n_steps,
        effective_n_steps,
        (tile_v, tile_u),
        num_warps_value,
        kernel_variant_id,
        layout_variant_id,
    ) = _validate_public_sinogram_call(
        T_stack,
        grid,
        detector,
        jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32),
        step_size=step_size,
        n_steps=n_steps,
        gather_dtype=gather_dtype,
        det_grid=det_grid,
        interpret=True,
        tile_shape=tile_shape,
        num_warps=num_warps,
        kernel_variant=kernel_variant,
        layout_variant=layout_variant,
        state_mode="cached",
    )
    if layout_variant_id != _LAYOUT_VARIANT_IDS["detector_vu"]:
        raise PallasProjectorUnsupported(
            _unsupported("cached pose-stack traversal state supports detector_vu only")
        )

    T = jnp.asarray(T_stack, dtype=jnp.float32)
    ix_all: list[jnp.ndarray] = []
    iy_all: list[jnp.ndarray] = []
    iz_all: list[jnp.ndarray] = []
    n_steps_ray_all: list[jnp.ndarray] = []
    dix_all: list[jnp.ndarray] = []
    diy_all: list[jnp.ndarray] = []
    diz_all: list[jnp.ndarray] = []
    for view in range(int(n_views)):
        ix0, iy0, iz0, dix, diy, diz, n_steps_ray, _step32, _resolved, n_rays = (
            _projector_traversal_state(
                T[view],
                grid,
                detector,
                step_size=step_size_value,
                n_steps=n_steps,
                det_grid=det_grid,
            )
        )
        expected_rays = int(nv) * int(nu)
        if int(n_rays) != expected_rays:
            raise PallasProjectorUnsupported(
                _unsupported(f"cached traversal state expected {expected_rays} rays; got {n_rays}")
            )
        ix_all.append(jnp.ravel(jnp.asarray(ix0, dtype=jnp.float32), order="C"))
        iy_all.append(jnp.ravel(jnp.asarray(iy0, dtype=jnp.float32), order="C"))
        iz_all.append(jnp.ravel(jnp.asarray(iz0, dtype=jnp.float32), order="C"))
        n_steps_ray_all.append(jnp.ravel(jnp.asarray(n_steps_ray, dtype=jnp.int32), order="C"))
        dix_all.append(jnp.ravel(jnp.asarray(dix, dtype=jnp.float32), order="C")[0])
        diy_all.append(jnp.ravel(jnp.asarray(diy, dtype=jnp.float32), order="C")[0])
        diz_all.append(jnp.ravel(jnp.asarray(diz, dtype=jnp.float32), order="C")[0])

    return PallasForwardProjectorStackTraversalState(
        ix0=jnp.concatenate(ix_all, axis=0),
        iy0=jnp.concatenate(iy_all, axis=0),
        iz0=jnp.concatenate(iz_all, axis=0),
        n_steps_ray=jnp.concatenate(n_steps_ray_all, axis=0),
        dix=jnp.asarray(dix_all, dtype=jnp.float32),
        diy=jnp.asarray(diy_all, dtype=jnp.float32),
        diz=jnp.asarray(diz_all, dtype=jnp.float32),
        step_size=float(step_size_value),
        n_steps=int(effective_n_steps),
        resolved_n_steps=int(resolved_n_steps),
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        nv=int(nv),
        nu=int(nu),
        n_views=int(n_views),
        tile_shape=(int(tile_v), int(tile_u)),
        num_warps=int(num_warps_value),
        kernel_variant="generic"
        if kernel_variant_id == _KERNEL_VARIANT_IDS["generic"]
        else "z_integer4",
        kernel_variant_id=int(kernel_variant_id),
        gather_dtype=_normalize_gather_dtype(gather_dtype),
    )


def block_forward_project_views_T_pallas_state(
    state: PallasForwardProjectorStackTraversalState,
) -> PallasForwardProjectorStackTraversalState:
    """Block until prepared stack traversal-state arrays are materialized."""
    jax.block_until_ready((state.ix0, state.iy0, state.iz0, state.n_steps_ray, state.dix))
    return state


def _projector_views_kernel_cached(
    ix0_ref: Any,
    iy0_ref: Any,
    iz0_ref: Any,
    n_steps_ray_ref: Any,
    dix_ref: Any,
    diy_ref: Any,
    diz_ref: Any,
    volume_ref: Any,
    out_ref: Any,
    *,
    nx: int,
    ny: int,
    nz: int,
    nu: int,
    nv: int,
    n_views: int,
    step_size: float,
    n_steps: int,
    tile_v: int,
    tile_u: int,
    kernel_variant_id: int,
    unroll: int | None,
) -> None:
    view_idx = pl.program_id(0)
    tile_v_start = pl.program_id(1) * tile_v
    tile_u_start = pl.program_id(2) * tile_u
    det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[jnp.newaxis, :]
    det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[:, jnp.newaxis]
    in_detector = (det_u < nu) & (det_v < nv) & (view_idx < n_views)
    state_idx = view_idx * jnp.int32(nv * nu) + jnp.clip(det_v * nu + det_u, 0, (nu * nv) - 1)

    ix0 = plt.load(ix0_ref.at[state_idx], mask=in_detector, other=0.0)
    iy0 = plt.load(iy0_ref.at[state_idx], mask=in_detector, other=0.0)
    iz0 = plt.load(iz0_ref.at[state_idx], mask=in_detector, other=0.0)
    n_steps_ray = plt.load(n_steps_ray_ref.at[state_idx], mask=in_detector, other=0)
    dix = plt.load(dix_ref.at[view_idx])
    diy = plt.load(diy_ref.at[view_idx])
    diz = plt.load(diz_ref.at[view_idx])
    step_size32 = jnp.float32(step_size)

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
        tile_steps = jnp.minimum(
            jnp.max(jnp.where(in_detector, n_steps_ray, 0)),
            jnp.asarray(n_steps, dtype=jnp.int32),
        )
        acc, _, _, _ = jax.lax.fori_loop(0, tile_steps, body, init)
    else:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init, unroll=unroll)
    out_ref[...] = jnp.where(in_detector, acc.astype(jnp.float32), 0.0)[jnp.newaxis, :, :]


def _projector_residual_sse_kernel_cached(
    ix0_ref: Any,
    iy0_ref: Any,
    iz0_ref: Any,
    n_steps_ray_ref: Any,
    dix_ref: Any,
    diy_ref: Any,
    diz_ref: Any,
    volume_ref: Any,
    target_ref: Any,
    out_ref: Any,
    *,
    nx: int,
    ny: int,
    nz: int,
    nu: int,
    nv: int,
    n_views: int,
    step_size: float,
    n_steps: int,
    tile_v: int,
    tile_u: int,
    kernel_variant_id: int,
    unroll: int | None,
) -> None:
    view_idx = pl.program_id(0)
    tile_v_start = pl.program_id(1) * tile_v
    tile_u_start = pl.program_id(2) * tile_u
    det_u = tile_u_start + jnp.arange(tile_u, dtype=jnp.int32)[jnp.newaxis, :]
    det_v = tile_v_start + jnp.arange(tile_v, dtype=jnp.int32)[:, jnp.newaxis]
    in_detector = (det_u < nu) & (det_v < nv) & (view_idx < n_views)
    state_idx = view_idx * jnp.int32(nv * nu) + jnp.clip(det_v * nu + det_u, 0, (nu * nv) - 1)

    ix0 = plt.load(ix0_ref.at[state_idx], mask=in_detector, other=0.0)
    iy0 = plt.load(iy0_ref.at[state_idx], mask=in_detector, other=0.0)
    iz0 = plt.load(iz0_ref.at[state_idx], mask=in_detector, other=0.0)
    n_steps_ray = plt.load(n_steps_ray_ref.at[state_idx], mask=in_detector, other=0)
    dix = plt.load(dix_ref.at[view_idx])
    diy = plt.load(diy_ref.at[view_idx])
    diz = plt.load(diz_ref.at[view_idx])
    step_size32 = jnp.float32(step_size)

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
        tile_steps = jnp.minimum(
            jnp.max(jnp.where(in_detector, n_steps_ray, 0)),
            jnp.asarray(n_steps, dtype=jnp.int32),
        )
        acc, _, _, _ = jax.lax.fori_loop(0, tile_steps, body, init)
    else:
        acc, _, _, _ = jax.lax.fori_loop(0, n_steps, body, init, unroll=unroll)
    target = plt.load(target_ref.at[view_idx, det_v, det_u], mask=in_detector, other=0.0)
    residual = jnp.where(in_detector, acc.astype(jnp.float32) - target.astype(jnp.float32), 0.0)
    out_ref[0, 0, 0] = jnp.sum(residual * residual).astype(jnp.float32)


@functools.lru_cache(maxsize=32)
def _cached_projector_views_state_pallas_call(
    *,
    nx: int,
    ny: int,
    nz: int,
    nv: int,
    nu: int,
    n_views: int,
    step_size: float,
    n_steps: int,
    tile_v: int,
    tile_u: int,
    num_warps: int,
    kernel_variant_id: int,
    unroll: int | None,
    interpret: bool,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    kernel = functools.partial(
        _projector_views_kernel_cached,
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        nu=int(nu),
        nv=int(nv),
        n_views=int(n_views),
        step_size=float(step_size),
        n_steps=int(n_steps),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        kernel_variant_id=int(kernel_variant_id),
        unroll=unroll,
    )
    grid_shape = (int(n_views), math.ceil(int(nv) / int(tile_v)), math.ceil(int(nu) / int(tile_u)))
    out_nv = int(grid_shape[1]) * int(tile_v)
    out_nu = int(grid_shape[2]) * int(tile_u)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((int(n_views), out_nv, out_nu), jnp.float32),
        grid=grid_shape,
        in_specs=[
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
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
        name="tomojax_forward_project_views_T_pallas_cached_state",
    )


@functools.lru_cache(maxsize=32)
def _cached_projector_residual_sse_state_pallas_call(
    *,
    nx: int,
    ny: int,
    nz: int,
    nv: int,
    nu: int,
    n_views: int,
    step_size: float,
    n_steps: int,
    tile_v: int,
    tile_u: int,
    num_warps: int,
    kernel_variant_id: int,
    unroll: int | None,
    interpret: bool,
) -> Callable[
    [
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
    jnp.ndarray,
]:
    kernel = functools.partial(
        _projector_residual_sse_kernel_cached,
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        nu=int(nu),
        nv=int(nv),
        n_views=int(n_views),
        step_size=float(step_size),
        n_steps=int(n_steps),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        kernel_variant_id=int(kernel_variant_id),
        unroll=unroll,
    )
    grid_shape = (int(n_views), math.ceil(int(nv) / int(tile_v)), math.ceil(int(nu) / int(tile_u)))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (int(n_views), math.ceil(int(nv) / int(tile_v)), math.ceil(int(nu) / int(tile_u))),
            jnp.float32,
        ),
        grid=grid_shape,
        in_specs=[
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec((1, 1, 1), lambda view, pv, pu: (view, pv, pu)),
        interpret=bool(interpret),
        compiler_params=plt.CompilerParams(num_warps=int(num_warps)),
        name="tomojax_forward_project_residual_sse_T_pallas_cached_state",
    )


def forward_project_views_T_pallas_with_state(
    state: PallasForwardProjectorStackTraversalState,
    volume: jnp.ndarray,
    *,
    interpret: bool = False,
    unroll: int | None = None,
) -> jnp.ndarray:
    """Forward project a stack with prepared traversal state."""
    nx, ny, nz = validate_volume(
        volume,
        Grid(nx=state.nx, ny=state.ny, nz=state.nz, vx=1.0, vy=1.0, vz=1.0),
        context="forward_project_views_T_pallas_with_state",
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
    call = _cached_projector_views_state_pallas_call(
        nx=int(state.nx),
        ny=int(state.ny),
        nz=int(state.nz),
        nv=int(state.nv),
        nu=int(state.nu),
        n_views=int(state.n_views),
        step_size=float(state.step_size),
        n_steps=int(state.n_steps),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        num_warps=int(state.num_warps),
        kernel_variant_id=int(state.kernel_variant_id),
        unroll=unroll,
        interpret=bool(interpret),
    )
    return call(
        state.ix0,
        state.iy0,
        state.iz0,
        state.n_steps_ray,
        state.dix,
        state.diy,
        state.diz,
        _prepare_volume_for_pallas_gather(volume, state.gather_dtype),
    )[:, : int(state.nv), : int(state.nu)]


def forward_project_residual_sse_T_pallas_with_state(
    state: PallasForwardProjectorStackTraversalState,
    volume: jnp.ndarray,
    target: jnp.ndarray,
    *,
    interpret: bool = False,
    unroll: int | None = None,
) -> jnp.ndarray:
    """Return residual SSE for a stack using prepared traversal state."""
    nx, ny, nz = validate_volume(
        volume,
        Grid(nx=state.nx, ny=state.ny, nz=state.nz, vx=1.0, vy=1.0, vz=1.0),
        context="forward_project_residual_sse_T_pallas_with_state",
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
    if tuple(int(dim) for dim in target.shape) != (state.n_views, state.nv, state.nu):
        raise PallasProjectorUnsupported(
            _unsupported(
                "target shape does not match cached traversal state: "
                f"got {tuple(int(dim) for dim in target.shape)}, "
                f"expected {(state.n_views, state.nv, state.nu)}"
            )
        )
    if not interpret and jax.default_backend() == "cpu":
        raise PallasProjectorUnsupported(
            _unsupported("real Pallas lowering is unavailable on CPU; pass interpret=True")
        )
    tile_v, tile_u = state.tile_shape
    call = _cached_projector_residual_sse_state_pallas_call(
        nx=int(state.nx),
        ny=int(state.ny),
        nz=int(state.nz),
        nv=int(state.nv),
        nu=int(state.nu),
        n_views=int(state.n_views),
        step_size=float(state.step_size),
        n_steps=int(state.n_steps),
        tile_v=int(tile_v),
        tile_u=int(tile_u),
        num_warps=int(state.num_warps),
        kernel_variant_id=int(state.kernel_variant_id),
        unroll=unroll,
        interpret=bool(interpret),
    )
    partials = call(
        state.ix0,
        state.iy0,
        state.iz0,
        state.n_steps_ray,
        state.dix,
        state.diy,
        state.diz,
        _prepare_volume_for_pallas_gather(volume, state.gather_dtype),
        jnp.asarray(target, dtype=jnp.float32),
    )
    return jnp.sum(partials, dtype=jnp.float32)


class BoundForwardProjectViewsTPallas:
    """Fixed pose-stack Pallas projector callable for repeated-volume workflows."""

    def __init__(
        self,
        state: PallasForwardProjectorStackTraversalState,
        *,
        interpret: bool = False,
        unroll: int | None = None,
    ) -> None:
        self.state = state
        self.interpret = bool(interpret)
        self.unroll = unroll

    def __call__(self, volume: jnp.ndarray) -> jnp.ndarray:
        return forward_project_views_T_pallas_with_state(
            self.state,
            volume,
            interpret=self.interpret,
            unroll=self.unroll,
        )


class BoundForwardProjectResidualSseTPallas:
    """Fixed pose-stack Pallas residual callable for repeated-volume workflows."""

    def __init__(
        self,
        state: PallasForwardProjectorStackTraversalState,
        target: jnp.ndarray,
        *,
        interpret: bool = False,
        unroll: int | None = None,
    ) -> None:
        self.state = state
        self.target = jnp.asarray(target, dtype=jnp.float32)
        self.interpret = bool(interpret)
        self.unroll = unroll

    def __call__(self, volume: jnp.ndarray) -> jnp.ndarray:
        return forward_project_residual_sse_T_pallas_with_state(
            self.state,
            volume,
            self.target,
            interpret=self.interpret,
            unroll=self.unroll,
        )


def bind_forward_project_views_T_pallas(
    T_stack: jnp.ndarray,
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
) -> BoundForwardProjectViewsTPallas:
    """Bind fixed pose-stack traversal state once and return a sinogram callable."""
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
    if block_state:
        block_forward_project_views_T_pallas_state(state)
    return BoundForwardProjectViewsTPallas(state, interpret=interpret, unroll=unroll)


def bind_forward_project_residual_sse_T_pallas(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
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
    block_state: bool = True,
) -> BoundForwardProjectResidualSseTPallas:
    """Bind fixed pose-stack traversal state once and return a residual SSE callable."""
    validate_projection_stack(
        target,
        detector,
        context="bind_forward_project_residual_sse_T_pallas",
    )
    if int(target.shape[0]) != int(T_stack.shape[0]):
        raise PallasProjectorUnsupported(
            _unsupported(
                "target view count does not match pose stack: "
                f"got {int(target.shape[0])}, expected {int(T_stack.shape[0])}"
            )
        )
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
    if block_state:
        block_forward_project_views_T_pallas_state(state)
    return BoundForwardProjectResidualSseTPallas(
        state,
        target,
        interpret=interpret,
        unroll=unroll,
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
    kernel_variant: str = "auto",
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
        return forward_project_views_T_pallas_with_state(
            state,
            volume,
            interpret=interpret,
            unroll=unroll,
        )
    vol_origin = _grid_volume_origin(grid)
    call = _cached_projector_views_pallas_call(
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
        kernel_variant_id=int(kernel_variant_id),
        layout_variant_id=int(layout_variant_id),
        unroll=unroll,
        interpret=bool(interpret),
    )
    return call(
        jnp.asarray(T_stack, dtype=jnp.float32),
        _prepare_volume_for_pallas_gather(volume, gather_dtype),
    )


@functools.lru_cache(maxsize=32)
def _cached_projector_views_pallas_call(
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
    interpret: bool,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    kernel = functools.partial(
        _projector_views_kernel,
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
    )
    grid_shape = (int(n_views), math.ceil(int(nv) / int(tile_v)), math.ceil(int(nu) / int(tile_u)))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((int(n_views), int(nv), int(nu)), jnp.float32),
        grid=grid_shape,
        in_specs=[
            pl.no_block_spec,
            pl.no_block_spec,
        ],
        out_specs=pl.BlockSpec(
            (1, int(tile_v), int(tile_u)),
            lambda view, pv, pu: (view, pv, pu),
        ),
        interpret=bool(interpret),
        compiler_params=plt.CompilerParams(num_warps=int(num_warps)),
        name="tomojax_forward_project_views_T_pallas",
    )


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
    vol_origin = _grid_volume_origin(grid)
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

    This is an internal reconstruction benchmark helper. It intentionally does
    not replace the default JAX adjoint used by differentiable public paths.
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
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
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
