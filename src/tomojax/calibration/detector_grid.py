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
    return transform_detector_grid(
        base_grid,
        det_u_px=det_u_px,
        det_v_px=det_v_px,
        detector_roll_deg=0.0,
        native_du=native_du,
        native_dv=native_dv,
    )


def transform_detector_grid(
    base_grid: tuple[jnp.ndarray, jnp.ndarray],
    *,
    det_u_px: object = 0.0,
    det_v_px: object = 0.0,
    detector_roll_deg: object = 0.0,
    native_du: float,
    native_dv: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rotate a zero-centre detector grid, then apply native-pixel centre offsets.

    Positive roll is a right-handed detector-plane rotation using the convention::

        u_rot = cos(r) * u - sin(r) * v
        v_rot = sin(r) * u + cos(r) * v

    The roll is applied around the detector centre before detector/ray-grid centre
    offsets are added, so centre and roll remain independent calibration variables.
    """
    Xr, Zr = base_grid
    roll_rad = jnp.deg2rad(jnp.asarray(detector_roll_deg, dtype=jnp.float32))
    cos_r = jnp.cos(roll_rad)
    sin_r = jnp.sin(roll_rad)
    Xrot = cos_r * Xr - sin_r * Zr
    Zrot = sin_r * Xr + cos_r * Zr
    u_offset = jnp.asarray(det_u_px, dtype=jnp.float32) * jnp.float32(native_du)
    v_offset = jnp.asarray(det_v_px, dtype=jnp.float32) * jnp.float32(native_dv)
    return Xrot + u_offset, Zrot + v_offset


def detector_grid_from_center_offset(
    detector: Detector,
    *,
    det_u_px: object = 0.0,
    det_v_px: object = 0.0,
    native_du: float | None = None,
    native_dv: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a dynamic detector grid from canonical native-pixel centre offsets."""
    return detector_grid_from_calibration(
        detector,
        det_u_px=det_u_px,
        det_v_px=det_v_px,
        detector_roll_deg=0.0,
        native_du=native_du,
        native_dv=native_dv,
    )


def detector_grid_from_calibration(
    detector: Detector,
    *,
    det_u_px: object = 0.0,
    det_v_px: object = 0.0,
    detector_roll_deg: object = 0.0,
    native_du: float | None = None,
    native_dv: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a detector grid from centre offsets and detector-plane roll."""
    return transform_detector_grid(
        zero_center_detector_grid(detector),
        det_u_px=det_u_px,
        det_v_px=det_v_px,
        detector_roll_deg=detector_roll_deg,
        native_du=float(detector.du) if native_du is None else float(native_du),
        native_dv=float(detector.dv) if native_dv is None else float(native_dv),
    )


def detector_grid_from_detector_roll(
    detector: Detector,
    *,
    detector_roll_deg: object,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a rolled detector grid preserving the detector's physical centre."""
    return detector_grid_from_calibration(
        detector,
        det_u_px=float(detector.det_center[0]) / float(detector.du),
        det_v_px=float(detector.det_center[1]) / float(detector.dv),
        detector_roll_deg=detector_roll_deg,
    )


def detector_grid_from_geometry_inputs(
    detector: Detector,
    geometry_inputs: object,
) -> tuple[jnp.ndarray, jnp.ndarray] | None:
    """Return a replay detector grid when saved metadata contains detector roll."""
    roll = None
    if isinstance(geometry_inputs, dict):
        roll = geometry_inputs.get("detector_roll_deg")
    else:
        roll = getattr(geometry_inputs, "detector_roll_deg", None)
    if roll is None:
        return None
    return detector_grid_from_detector_roll(detector, detector_roll_deg=roll)
