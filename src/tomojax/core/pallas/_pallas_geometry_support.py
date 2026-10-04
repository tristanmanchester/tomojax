"""Geometry eligibility checks for specialized Pallas projector variants."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.base import Detector, Grid, grid_volume_origin
from tomojax.core.projector import _build_detector_grid
from tomojax.core.validation import validate_detector_grid

from ._pallas_state import PallasProjectorUnsupported, _unsupported


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


def _supports_z_integer4(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> bool:
    try:
        T_host = np.asarray(T, dtype=np.float64)
    except Exception:
        return False
    if T_host.shape != (4, 4):
        return False
    return _supports_z_integer4_for_stack(T_host[None], grid, detector, det_grid)


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
    # A small tilt is not necessarily a small error in voxel coordinates: long
    # rays through thin slices amplify it. Built-in z-axis rotations have exact
    # zeros here. Use the generic kernel for any mixing of x/y into object z.
    if np.any(T_host[:, :2, 2] != 0):
        return False

    tinv_z = -(
        T_host[:, 0, 2] * T_host[:, 0, 3]
        + T_host[:, 1, 2] * T_host[:, 1, 3]
        + T_host[:, 2, 2] * T_host[:, 2, 3]
    )
    first_z = (-(float(detector.nv) / 2.0 - 0.5)) * float(detector.dv) + float(
        detector.det_center[1]
    )
    iz0 = (T_host[:, 2, 2] * first_z + tinv_z - float(grid_volume_origin(grid)[2])) / float(grid.vz)
    diz_dv = T_host[:, 2, 2] * float(detector.dv) / float(grid.vz)
    # Bound the error over every row, not just the first row and the increment.
    # A near-integer increment can drift by many tolerances on a tall detector.
    row_error = np.abs(iz0 - np.round(iz0))
    row_error += (detector.nv - 1) * np.abs(diz_dv - np.round(diz_dv))
    return bool(np.all(row_error <= tol))


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
        and np.all(T_host[:, 2, :3] == [0.0, 0.0, 1.0])
        and np.all(T_host[:, :3, 3] == 0)
        and np.all(np.abs(T_host[:, 3, :3]) <= tol)
        and np.all(np.abs(T_host[:, 3, 3] - 1.0) <= tol)
    )


__all__ = [
    "_ensure_canonical_detector_grid",
    "_supports_parallel_z_rotation_stack",
    "_supports_z_integer4",
    "_supports_z_integer4_for_stack",
]
