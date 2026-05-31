"""Pallas option normalization and kernel variant selection."""

from __future__ import annotations

import operator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.base import Detector, Grid

from ._pallas_geometry_support import _supports_z_integer4, _supports_z_integer4_for_stack
from ._pallas_state import PallasProjectorUnsupported, _unsupported

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


__all__ = [
    "_KERNEL_VARIANT_IDS",
    "_LAYOUT_VARIANT_IDS",
    "_normalize_gather_dtype",
    "_normalize_kernel_variant",
    "_pallas_gather_jnp_dtype",
    "_prepare_volume_for_pallas_gather",
    "pallas_projector_actual_sinogram_variant_metadata",
    "pallas_projector_actual_variant_metadata",
    "pallas_projector_variant_metadata",
]
