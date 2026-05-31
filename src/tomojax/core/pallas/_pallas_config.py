from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.base import Detector, Grid
from tomojax.core.projector import _resolve_n_steps
from tomojax.core.validation import (
    validate_detector,
    validate_pose_matrix,
    validate_pose_stack,
    validate_volume,
)

from ._pallas_geometry_support import (
    _ensure_canonical_detector_grid,
    _supports_parallel_z_rotation_stack,
)
from ._pallas_state import (
    PallasForwardProjectorStackTraversalState,
    PallasForwardProjectorTraversalState,
    PallasProjectorTraversalMetadata,
    PallasProjectorUnsupported,
    _unsupported,
)
from ._pallas_variants import (
    _KERNEL_VARIANT_IDS,
    _LAYOUT_VARIANT_IDS,
    _normalize_gather_dtype,
    _normalize_kernel_variant,
    _normalize_layout_variant,
    _normalize_num_warps,
    _normalize_state_mode,
    _normalize_tile_shape,
    _pallas_gather_jnp_dtype,
    _prepare_volume_for_pallas_gather,
    _safe_detector_tile_shape,
    pallas_projector_actual_sinogram_variant_metadata,
    pallas_projector_actual_variant_metadata,
    pallas_projector_variant_metadata,
)

__all__ = [
    "_KERNEL_VARIANT_IDS",
    "_LAYOUT_VARIANT_IDS",
    "PallasForwardProjectorStackTraversalState",
    "PallasForwardProjectorTraversalState",
    "PallasProjectorTraversalMetadata",
    "PallasProjectorUnsupported",
    "_normalize_gather_dtype",
    "_normalize_kernel_variant",
    "_normalize_layout_variant",
    "_normalize_num_warps",
    "_normalize_state_mode",
    "_normalize_tile_shape",
    "_pallas_gather_jnp_dtype",
    "_prepare_volume_for_pallas_gather",
    "_resolve_effective_pallas_n_steps",
    "_resolve_effective_pallas_n_steps_for_stack",
    "_safe_detector_tile_shape",
    "_supports_parallel_z_rotation_stack",
    "_unsupported",
    "_validate_public_call",
    "_validate_public_sinogram_call",
    "pallas_projector_actual_sinogram_variant_metadata",
    "pallas_projector_actual_variant_metadata",
    "pallas_projector_sinogram_traversal_metadata",
    "pallas_projector_sinogram_unsupported_reason",
    "pallas_projector_traversal_metadata",
    "pallas_projector_unsupported_reason",
    "pallas_projector_variant_metadata",
]


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


def _ensure_float32_volume(volume: jnp.ndarray) -> None:
    dtype = getattr(volume, "dtype", None)
    if dtype != jnp.dtype(jnp.float32):
        raise PallasProjectorUnsupported(_unsupported(f"volume dtype must be float32; got {dtype}"))


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
