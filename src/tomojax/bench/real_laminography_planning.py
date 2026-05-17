"""Planning helpers for real-laminography developer workflows."""

from __future__ import annotations

import argparse
import math
from typing import Protocol

import numpy as np


class BinningArgs(Protocol):
    """Args-like object carrying real-laminography binning fields."""

    bin_factor: int
    effective_bin_factor: int


class PoseBoundsArgs(BinningArgs, Protocol):
    """Args-like object carrying pose bound profile fields."""

    pose_bounds_profile: str


def validate_bin_factor(value: object) -> int:
    """Validate and normalize a binning factor."""
    try:
        factor = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bin factor must be an integer >= 1, got {value!r}") from exc
    if factor < 1:
        raise ValueError(f"bin factor must be an integer >= 1, got {value!r}")
    return factor


def resolve_fixture_bin_factor(
    *,
    projection_shape: tuple[int, int, int],
    slab_nz: int,
    requested_bin_factor: int,
    smoke_shape: tuple[int, int, int] | None,
) -> int:
    """Resolve the effective bin factor needed to satisfy a smoke shape."""
    factor = validate_bin_factor(requested_bin_factor)
    if smoke_shape is None:
        return factor
    _target_views, target_nv, target_nu = smoke_shape
    if target_nv < 1 or target_nu < 1:
        raise ValueError(f"diagnostic shape must be positive, got {smoke_shape!r}")
    _n_views, nv, nu = projection_shape
    factor = max(factor, math.ceil(float(nv) / float(target_nv)))
    factor = max(factor, math.ceil(float(nu) / float(target_nu)))
    factor = max(factor, math.ceil(float(slab_nz) / float(max(1, target_nv))))
    return validate_bin_factor(factor)


def view_indices_for_smoke_shape(
    n_views: int,
    smoke_shape: tuple[int, int, int] | None,
) -> np.ndarray:
    """Return deterministic view indices for an optional smoke workload shape."""
    if smoke_shape is None or int(smoke_shape[0]) >= int(n_views):
        return np.arange(int(n_views), dtype=np.int64)
    target = max(1, int(smoke_shape[0]))
    return np.unique(np.rint(np.linspace(0, int(n_views) - 1, target)).astype(np.int64))


def binned_pixel_scale(args: BinningArgs) -> float:
    """Return the scale from native pixels into current binned pixels."""
    fallback = getattr(args, "bin_factor", 1)
    return 1.0 / float(max(1, int(getattr(args, "effective_bin_factor", fallback))))


def scaled_symmetric_bound(name: str, value: float, args: BinningArgs) -> str:
    """Format a symmetric parameter bound after binned-pixel scaling."""
    scaled = float(value) * binned_pixel_scale(args)
    return f"{name}={-scaled:.8g}:{scaled:.8g}"


def setup_det_u_bounds(args: BinningArgs) -> str:
    """Return the staged real-laminography detector-u search bound."""
    return scaled_symmetric_bound("det_u_px", 24.0, args)


def pose_phi_bounds(args: PoseBoundsArgs) -> str:
    """Return the staged real-laminography phi pose bound."""
    if str(args.pose_bounds_profile) == "wide":
        return "phi=-0.0872665:0.0872665"
    return "phi=-0.00872665:0.00872665"


def pose_dx_dz_bounds(args: PoseBoundsArgs) -> str:
    """Return the staged real-laminography dx/dz pose bounds."""
    value = 16.0 if str(args.pose_bounds_profile) == "wide" else 10.0
    dx = scaled_symmetric_bound("dx", value, args)
    dz = scaled_symmetric_bound("dz", value, args)
    return f"{dx},{dz}"


def pose_polish_bounds(args: PoseBoundsArgs) -> str:
    """Return the staged real-laminography 5-DOF polish bounds."""
    dx_dz = pose_dx_dz_bounds(args)
    if str(args.pose_bounds_profile) == "wide":
        return (
            "alpha=-0.0349066:0.0349066,beta=-0.0349066:0.0349066,"
            f"phi=-0.0872665:0.0872665,{dx_dz}"
        )
    return (
        "alpha=-0.00872665:0.00872665,beta=-0.00872665:0.00872665,"
        f"phi=-0.00872665:0.00872665,{dx_dz}"
    )


def parse_shape3(text: str) -> tuple[int, int, int]:
    """Parse a three-axis shape accepted by real-laminography CLI options."""
    parts = str(text).lower().replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected projection shape as N,NV,NU or NxNVxNU, got {text!r}"
        )
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer projection shape {text!r}") from exc
    if any(value <= 0 for value in shape):
        raise argparse.ArgumentTypeError(
            f"projection shape dimensions must be positive, got {text!r}"
        )
    return (shape[0], shape[1], shape[2])


__all__ = [
    "binned_pixel_scale",
    "parse_shape3",
    "pose_dx_dz_bounds",
    "pose_phi_bounds",
    "pose_polish_bounds",
    "resolve_fixture_bin_factor",
    "scaled_symmetric_bound",
    "setup_det_u_bounds",
    "validate_bin_factor",
    "view_indices_for_smoke_shape",
]
