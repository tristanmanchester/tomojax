from __future__ import annotations

"""Helpers for materializing geometry objects from persisted NXtomo metadata."""

from dataclasses import dataclass
from typing import Sequence, TypedDict

import numpy as np

from ..core.geometry import Detector, Geometry, Grid, LaminographyGeometry, ParallelGeometry
from ..core.geometry.base import DetectorDict, GridDict, PoseMatrix, RayPair


type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


class LoadedGeometryMetaRequired(TypedDict):
    detector: DetectorDict
    thetas_deg: Sequence[float] | np.ndarray


class LoadedGeometryMeta(LoadedGeometryMetaRequired, total=False):
    grid: GridDict
    geometry_type: str
    tilt_deg: float
    tilt_about: str
    angle_offset_deg: np.ndarray
    misalign_spec: dict[str, JsonValue]
    align_params: np.ndarray


GridOverride = Grid | tuple[int, int, int] | list[int] | None


def _volume_shape_nxyz(volume_shape: Sequence[int] | None) -> tuple[int, int, int] | None:
    if volume_shape is None:
        return None
    dims = tuple(int(v) for v in volume_shape)
    if len(dims) != 3:
        raise ValueError(f"volume_shape must provide exactly 3 dims, got {dims!r}")
    return dims


def _normalize_geometry_type(geometry_type: str | None) -> str:
    gtype = "parallel" if geometry_type is None else str(geometry_type).strip().lower()
    if gtype == "parallel":
        return gtype
    if gtype in {"lamino", "laminography"}:
        return "lamino"
    raise ValueError(
        f"Unsupported geometry_type {geometry_type!r}; expected 'parallel' or 'lamino'"
    )


@dataclass
class AugmentedGeometry:
    """Geometry wrapper that applies saved per-view 5-DOF alignment params."""

    base: Geometry
    align_params: np.ndarray

    def pose_for_view(self, i: int) -> PoseMatrix:
        T_nom = np.asarray(self.base.pose_for_view(i), dtype=np.float32)
        T_delta = _se3_from_5d_np(self.align_params[i])
        T = T_nom @ T_delta
        return tuple(map(tuple, T))

    def rays_for_view(self, i: int) -> RayPair:
        return self.base.rays_for_view(i)


def _rot_x_np(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float32,
    )


def _rot_y_np(b: float) -> np.ndarray:
    c, s = np.cos(b), np.sin(b)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float32,
    )


def _rot_z_np(p: float) -> np.ndarray:
    c, s = np.cos(p), np.sin(p)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _se3_from_5d_np(params5: np.ndarray) -> np.ndarray:
    alpha, beta, phi, dx, dz = np.asarray(params5, dtype=np.float32)
    R = _rot_y_np(float(beta)) @ _rot_x_np(float(alpha)) @ _rot_z_np(float(phi))
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.array([dx, 0.0, dz], dtype=np.float32)
    return T


def _detector_from_meta(meta: LoadedGeometryMeta) -> Detector:
    det_d = meta["detector"]
    det_center = det_d.get("det_center", [0.0, 0.0])
    return Detector(
        nu=int(det_d["nu"]),
        nv=int(det_d["nv"]),
        du=float(det_d["du"]),
        dv=float(det_d["dv"]),
        det_center=(float(det_center[0]), float(det_center[1])),
    )


def _grid_from_meta(
    meta: LoadedGeometryMeta,
    detector: Detector,
    grid_override: GridOverride,
    volume_shape: Sequence[int] | None = None,
) -> Grid:
    if isinstance(grid_override, Grid):
        return grid_override

    grid_d = meta.get("grid")
    if grid_d is None:
        if grid_override is not None:
            nx, ny, nz = map(int, grid_override)
        elif (shape := _volume_shape_nxyz(volume_shape)) is not None:
            nx, ny, nz = shape
        else:
            nx = int(detector.nu)
            ny = int(detector.nu)
            nz = int(detector.nv)
        return Grid(
            nx=nx,
            ny=ny,
            nz=nz,
            vx=float(detector.du),
            vy=float(detector.du),
            vz=float(detector.dv),
        )

    vol_origin = (
        tuple(float(v) for v in grid_d["vol_origin"])
        if grid_d.get("vol_origin") is not None
        else None
    )
    vol_center = (
        tuple(float(v) for v in grid_d["vol_center"])
        if grid_d.get("vol_center") is not None
        else None
    )

    if grid_override is not None:
        nx, ny, nz = map(int, grid_override)
        return Grid(
            nx=nx,
            ny=ny,
            nz=nz,
            vx=float(grid_d["vx"]),
            vy=float(grid_d["vy"]),
            vz=float(grid_d["vz"]),
            vol_origin=vol_origin,
            vol_center=vol_center,
        )

    return Grid(
        nx=int(grid_d["nx"]),
        ny=int(grid_d["ny"]),
        nz=int(grid_d["nz"]),
        vx=float(grid_d["vx"]),
        vy=float(grid_d["vy"]),
        vz=float(grid_d["vz"]),
        vol_origin=vol_origin,
        vol_center=vol_center,
    )


def _resolve_thetas_deg(
    meta: LoadedGeometryMeta,
    *,
    apply_saved_angle_offset: bool,
) -> np.ndarray:
    thetas = np.asarray(meta["thetas_deg"], dtype=np.float32)
    if not apply_saved_angle_offset:
        return thetas

    angle_offset = meta.get("angle_offset_deg")
    if angle_offset is None:
        return thetas

    offset = np.asarray(angle_offset, dtype=np.float32)
    if offset.shape != thetas.shape:
        return thetas
    if not np.isfinite(offset).any() or np.allclose(offset, 0.0):
        return thetas

    # TomoJAX's misalign CLI already bakes scheduled angle offsets into
    # `thetas_deg` and stores the raw schedule separately for provenance.
    if meta.get("misalign_spec") is not None:
        return thetas

    return thetas + offset


def _base_geometry(
    *,
    meta: LoadedGeometryMeta,
    grid: Grid,
    detector: Detector,
    thetas_deg: Sequence[float],
) -> Geometry:
    gtype = _normalize_geometry_type(meta.get("geometry_type"))
    if gtype == "parallel":
        return ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas_deg)

    tilt_deg = float(meta.get("tilt_deg", 30.0))
    tilt_about = str(meta.get("tilt_about", "x"))
    return LaminographyGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=thetas_deg,
        tilt_deg=tilt_deg,
        tilt_about=tilt_about,
    )


def build_geometry_from_meta(
    meta: LoadedGeometryMeta,
    *,
    grid_override: GridOverride = None,
    apply_saved_alignment: bool = False,
    volume_shape: Sequence[int] | None = None,
) -> tuple[Grid, Detector, Geometry]:
    """Build geometry from NXtomo metadata with sensible fallbacks.

    When `grid` metadata is missing, the grid is inferred from detector dimensions
    unless an explicit `grid_override` or `volume_shape` is supplied; both reuse
    detector pixel spacings as voxel spacings. When `apply_saved_alignment` is
    True, any saved `align_params` are composed onto the nominal poses. Saved
    alignments must provide one row per view and at least five columns ordered
    as `[alpha, beta, phi, dx, dz]`; extra columns are ignored. Saved
    `angle_offset_deg` is applied unless it is known to have already been baked
    into `thetas_deg`.
    """

    detector = _detector_from_meta(meta)
    grid = _grid_from_meta(meta, detector, grid_override, volume_shape)
    thetas_deg = _resolve_thetas_deg(
        meta,
        apply_saved_angle_offset=apply_saved_alignment,
    )
    geom = _base_geometry(meta=meta, grid=grid, detector=detector, thetas_deg=thetas_deg)

    if apply_saved_alignment and meta.get("align_params") is not None:
        align_params = np.asarray(meta["align_params"], dtype=np.float32)
        if align_params.ndim != 2:
            raise ValueError("align_params must be a 2-D array with shape (n_views, >=5)")
        if align_params.shape[0] != len(thetas_deg):
            raise ValueError(
                f"align_params row count ({align_params.shape[0]}) must match number of views ({len(thetas_deg)})"
            )
        if align_params.shape[1] < 5:
            raise ValueError(
                f"align_params must provide at least 5 columns [alpha, beta, phi, dx, dz], got {align_params.shape[1]}"
            )
        geom = AugmentedGeometry(base=geom, align_params=align_params[:, :5])

    return grid, detector, geom


def build_nominal_geometry_from_meta(
    meta: LoadedGeometryMeta,
    grid_override: GridOverride = None,
    *,
    volume_shape: Sequence[int] | None = None,
) -> tuple[Grid, Detector, Geometry]:
    """Build geometry without composing any saved alignment metadata."""
    return build_geometry_from_meta(
        meta,
        grid_override=grid_override,
        apply_saved_alignment=False,
        volume_shape=volume_shape,
    )
