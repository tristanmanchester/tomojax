"""Planning helpers for real-laminography developer workflows."""

from __future__ import annotations

import argparse
import math
from typing import Any, Protocol

import jax.numpy as jnp
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


def real_lamino_grid_origin_z(grid: Any) -> float:
    """Return the physical z coordinate of local slice zero."""
    if grid.vol_origin is not None:
        return float(grid.vol_origin[2])
    center_z = 0.0 if grid.vol_center is None else float(grid.vol_center[2])
    return center_z - ((float(grid.nz) / 2.0) - 0.5) * float(grid.vz)


def real_lamino_global_z_to_phys(global_z: int, *, full_nz: int) -> float:
    """Map a global z index to physical z in the real-lamino frame."""
    return float(global_z) - ((float(full_nz) / 2.0) - 0.5)


def real_lamino_phys_z_to_local_index(phys_z: float, grid: Any) -> int:
    """Map physical z to the nearest local slab index."""
    return int(round((float(phys_z) - real_lamino_grid_origin_z(grid)) / float(grid.vz)))


def real_lamino_global_z_to_local_index(global_z: int, *, full_nz: int, grid: Any) -> int:
    """Map a global z index to a local slab index."""
    return real_lamino_phys_z_to_local_index(
        real_lamino_global_z_to_phys(global_z, full_nz=full_nz),
        grid,
    )


def real_lamino_local_z_to_global_index(local_z: int, *, full_nz: int, grid: Any) -> int:
    """Map a local slab index back to a global z index."""
    phys_z = real_lamino_grid_origin_z(grid) + float(local_z) * float(grid.vz)
    return int(round(phys_z + ((float(full_nz) / 2.0) - 0.5)))


def real_lamino_xy_at_global_z(
    volume: np.ndarray,
    *,
    grid: Any,
    full_nz: int,
    global_z: int,
) -> np.ndarray:
    """Return an XY image at a global z coordinate using display orientation."""
    local_z = real_lamino_global_z_to_local_index(global_z, full_nz=full_nz, grid=grid)
    local_z = int(np.clip(local_z, 0, np.asarray(volume).shape[2] - 1))
    return np.asarray(volume, dtype=np.float32)[:, :, local_z].T


def map_real_lamino_global_z_to_binned(
    global_z: int,
    *,
    original_full_nz: int,
    binned_full_nz: int,
    binned_grid: Any,
) -> int:
    """Map a native global z coordinate into a binned real-lamino coordinate frame."""
    phys_z = real_lamino_global_z_to_phys(int(global_z), full_nz=int(original_full_nz))
    local_z = real_lamino_phys_z_to_local_index(phys_z, binned_grid)
    local_z = int(np.clip(local_z, 0, int(binned_grid.nz) - 1))
    mapped = real_lamino_local_z_to_global_index(
        local_z,
        full_nz=int(binned_full_nz),
        grid=binned_grid,
    )
    return int(np.clip(mapped, 0, int(binned_full_nz) - 1))


def prepare_real_lamino_binned_fixture(
    args: argparse.Namespace,
    *,
    native: Any,
    raw_projections: np.ndarray,
    thetas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Derive the optional binned real-data fixture and mutate working args."""
    original_shape = tuple(int(v) for v in raw_projections.shape)
    original_full_nz = int(original_shape[1])
    view_indices = view_indices_for_smoke_shape(original_shape[0], args.smoke_shape)
    raw_selected = np.asarray(raw_projections[view_indices], dtype=np.float32)
    thetas_selected = np.asarray(thetas[view_indices], dtype=np.float32)
    bin_factor = resolve_fixture_bin_factor(
        projection_shape=tuple(int(v) for v in raw_selected.shape),
        slab_nz=int(args.slab_nz),
        requested_bin_factor=int(args.bin_factor),
        smoke_shape=args.smoke_shape,
    )

    center_phys_z = real_lamino_global_z_to_phys(
        int(args.slab_center_z),
        full_nz=original_full_nz,
    )
    base_grid = native.Grid(
        nx=int(original_shape[2]),
        ny=int(original_shape[2]),
        nz=int(args.slab_nz),
        vx=1.0,
        vy=1.0,
        vz=1.0,
        vol_center=(0.0, 0.0, center_phys_z),
    )
    base_detector = native.Detector(
        nu=int(original_shape[2]),
        nv=int(original_shape[1]),
        du=1.0,
        dv=1.0,
        det_center=(0.0, 0.0),
    )
    if bin_factor > 1:
        working_raw = np.asarray(
            native.bin_projections(jnp.asarray(raw_selected, dtype=jnp.float32), bin_factor),
            dtype=np.float32,
        )
        grid = native.scale_grid(base_grid, bin_factor)
        detector = native.scale_detector(base_detector, bin_factor)
    else:
        working_raw = raw_selected
        grid = base_grid
        detector = base_detector

    binned_detector_nz = int(detector.nv)
    original_preview_z = int(args.preview_z)
    original_slab_center_z = int(args.slab_center_z)
    original_stack_z_range = tuple(native._parse_range(str(args.stack_z_range)))
    args.slab_nz = int(grid.nz)
    args.effective_bin_factor = int(bin_factor)
    args.effective_view_indices = [int(v) for v in view_indices.tolist()]

    provenance = {
        "enabled": bool(bin_factor > 1 or len(view_indices) != original_shape[0]),
        "requested_bin_factor": int(args.bin_factor),
        "effective_bin_factor": int(bin_factor),
        "requested_smoke_shape": None if args.smoke_shape is None else list(args.smoke_shape),
        "original_projection_shape": list(original_shape),
        "selected_projection_shape_before_binning": list(raw_selected.shape),
        "working_projection_shape": list(working_raw.shape),
        "view_indices": [int(v) for v in view_indices.tolist()],
        "original_slab_nz": int(base_grid.nz),
        "working_slab_nz": int(grid.nz),
        "coordinate_full_nz": int(original_full_nz),
        "working_detector_nz": int(binned_detector_nz),
        "original_preview_global_z": int(original_preview_z),
        "working_preview_global_z": int(original_preview_z),
        "original_slab_center_global_z": int(original_slab_center_z),
        "working_slab_center_global_z": int(original_slab_center_z),
        "original_stack_z_range": list(original_stack_z_range),
        "working_stack_z_range": list(original_stack_z_range),
        "grid": grid.to_dict(),
        "detector": detector.to_dict(),
        "detector_shift_bound_scale": float(binned_pixel_scale(args)),
        "pose_dx_dz_bound_scale": float(binned_pixel_scale(args)),
    }
    geometry_inputs = {"grid": grid, "detector": detector, "full_nz": int(original_full_nz)}
    return working_raw, thetas_selected, geometry_inputs, provenance


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


def select_real_lamino_final_candidates(
    candidates: list[dict[str, Any]],
    *,
    policy: str,
) -> list[dict[str, Any]]:
    """Select staged geometry candidates to score for final reconstruction."""
    normalized = str(policy).strip().lower().replace("-", "_")
    if normalized == "all":
        return candidates
    if normalized == "last_valid":
        return [candidates[-1]]
    if normalized == "setup_only":
        setup_candidates = [
            candidate
            for candidate in candidates
            if str(candidate.get("source_stage", "")).startswith("01_setup_geometry/")
            or str(candidate.get("source_stage", "")) == "01_setup_geometry/01_cor"
        ]
        return setup_candidates or [candidates[-1]]
    raise ValueError(
        "final candidate policy must be one of 'all', 'last_valid', or 'setup_only'; "
        f"got {policy!r}"
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
    "map_real_lamino_global_z_to_binned",
    "parse_shape3",
    "pose_dx_dz_bounds",
    "pose_phi_bounds",
    "pose_polish_bounds",
    "prepare_real_lamino_binned_fixture",
    "real_lamino_global_z_to_local_index",
    "real_lamino_global_z_to_phys",
    "real_lamino_grid_origin_z",
    "real_lamino_local_z_to_global_index",
    "real_lamino_phys_z_to_local_index",
    "real_lamino_xy_at_global_z",
    "resolve_fixture_bin_factor",
    "scaled_symmetric_bound",
    "select_real_lamino_final_candidates",
    "setup_det_u_bounds",
    "validate_bin_factor",
    "view_indices_for_smoke_shape",
]
