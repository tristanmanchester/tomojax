from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from tomojax.core.geometry import Detector
from tomojax.core.projector import get_detector_grid_device


def zero_center_detector_grid(detector: Detector) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return detector coordinates for the same detector shape with zero physical centre."""
    zero_detector = dataclasses.replace(detector, det_center=(0.0, 0.0))
    return get_detector_grid_device(zero_detector)


def offset_detector_grid(
    base_grid: tuple[jnp.ndarray, jnp.ndarray],
    *,
    det_u_px: object = 0.0,
    det_v_px: object = 0.0,
    native_du: float,
    native_dv: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply detector/ray-grid centre offsets to a zero-centre detector grid.

    Offsets are supplied in native detector pixels and converted to physical detector
    coordinates before being added to the flattened ``(u, v)`` grid vectors.
    """
    Xr, Zr = base_grid
    u_offset = jnp.asarray(det_u_px, dtype=jnp.float32) * jnp.float32(native_du)
    v_offset = jnp.asarray(det_v_px, dtype=jnp.float32) * jnp.float32(native_dv)
    return Xr + u_offset, Zr + v_offset


def detector_grid_from_center_offset(
    detector: Detector,
    *,
    det_u_px: object = 0.0,
    det_v_px: object = 0.0,
    native_du: float | None = None,
    native_dv: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a dynamic detector grid from canonical native-pixel centre offsets."""
    return offset_detector_grid(
        zero_center_detector_grid(detector),
        det_u_px=det_u_px,
        det_v_px=det_v_px,
        native_du=float(detector.du) if native_du is None else float(native_du),
        native_dv=float(detector.dv) if native_dv is None else float(native_dv),
    )
