from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .geometry.base import Detector, Grid, _grid_volume_origin
from .projector import _build_detector_grid, _resolve_n_steps
from .validation import (
    validate_detector,
    validate_detector_grid,
    validate_pose_matrix,
    validate_pose_stack,
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
            _unsupported(
                f"num_warps must be one of {sorted(_SUPPORTED_NUM_WARPS)}; got {num_warps!r}"
            )
        ) from exc
    if value not in _SUPPORTED_NUM_WARPS:
        raise PallasProjectorUnsupported(
            _unsupported(
                f"num_warps must be one of {sorted(_SUPPORTED_NUM_WARPS)}; got {num_warps!r}"
            )
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
            _unsupported(f"kernel_variant must be a string; got {type(kernel_variant).__name__}")
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
            _unsupported(f"layout_variant must be a string; got {type(layout_variant).__name__}")
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
    effective_n_steps = math.ceil(max_path_length / float(step_size)) + 2
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
    effective_n_steps = math.ceil(max_path_length / float(step_size)) + 2
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

    first_z = (-(float(detector.nv) / 2.0 - 0.5)) * float(detector.dv) + float(
        detector.det_center[1]
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
    first_z = (-(float(detector.nv) / 2.0 - 0.5)) * float(detector.dv) + float(
        detector.det_center[1]
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
        _unsupported(
            "kernel_variant='z_integer4' requires all views to use integer-z detector geometry"
        )
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
    if not (np.array_equal(Xr_host, Xr_expected) and np.array_equal(Zr_host, Zr_expected)):
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
            _unsupported(
                "cached traversal state currently supports layout_variant='detector_vu' only"
            )
        )
    tile_v, tile_u = variant["tile_shape"]
    if not interpret and jax.default_backend() == "cpu":
        raise PallasProjectorUnsupported(
            _unsupported("real Pallas lowering is unavailable on CPU; pass interpret=True")
        )

    step_size_value = float(grid.vy) if step_size is None else float(step_size)
    n_steps_value = _resolve_n_steps(grid, step_size_value, n_steps)
    effective_n_steps_value = _resolve_effective_pallas_n_steps(
        T,
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
        nx * ny * nz,
        step_size_value,
        n_steps_value,
        effective_n_steps_value,
        (
            tile_v,
            tile_u,
        ),
        int(variant["num_warps"]),
        kernel_variant_id,
        layout_variant_id,
    )


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
            _unsupported(
                "cached traversal state currently supports layout_variant='detector_vu' only"
            )
        )
    tile_v, tile_u = variant["tile_shape"]
    if not interpret and jax.default_backend() == "cpu":
        raise PallasProjectorUnsupported(
            _unsupported("real Pallas lowering is unavailable on CPU; pass interpret=True")
        )

    step_size_value = float(grid.vy) if step_size is None else float(step_size)
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
    """Return an unsupported reason, or ``None`` if eligible."""
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
    """Return an unsupported reason for batched sinogram Pallas."""
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
