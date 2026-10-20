"""Geometry artifact serialization."""
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tomojax.core.geometry.lamino import laminography_axis_unit

if TYPE_CHECKING:
    from collections.abc import Mapping


def nominal_axis_unit_from_inputs(geometry_inputs: Mapping[str, object]) -> np.ndarray:
    """Resolve the nominal lab-frame rotation axis from persisted geometry metadata.

    Parameters
    ----------
    geometry_inputs : mapping
        Persisted axis vector or geometry type and laminography tilt.

    Returns
    -------
    numpy.ndarray
        Normalized three-component lab-frame axis.

    Raises
    ------
    ValueError
        If the axis is not finite and nonzero, or the tilt convention is invalid.
    """
    if geometry_inputs.get("axis_unit_lab") is not None:
        axis = np.asarray(geometry_inputs["axis_unit_lab"], dtype=np.float64)
    elif str(geometry_inputs.get("geometry_type", "parallel")).lower() in {
        "lamino",
        "laminography",
    }:
        if "tilt_deg" not in geometry_inputs:
            raise ValueError("Missing laminography tilt_deg or axis_unit_lab")
        axis = laminography_axis_unit(
            float(geometry_inputs["tilt_deg"]),
            str(geometry_inputs.get("tilt_about", "x")),
        )
    else:
        axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("nominal axis_unit_lab must be a finite 3-vector")
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("nominal axis_unit_lab must be non-zero")
    return axis / norm
