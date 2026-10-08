"""Helpers for materializing geometry objects from persisted NXtomo metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np

from tomojax.core.geometry import (
    ConeBeam,
    ConeGeometry,
    Detector,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
    RotationAxisGeometry,
    normalize_axis_unit,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.core.geometry.base import (
        DetectorDict,
        GridDict,
        PoseMatrix,
        RayPair,
        ScanGeometry,
    )

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


class LoadedGeometryMetaRequired(TypedDict):
    """Required metadata for constructing a geometry object."""

    detector: DetectorDict
    angles: Sequence[float] | np.ndarray


class LoadedGeometryMeta(LoadedGeometryMetaRequired, total=False):
    """Optional persisted geometry metadata fields."""

    grid: GridDict
    geometry_type: str
    tilt_deg: float
    tilt_about: str
    axis_unit_lab: Sequence[float]
    detector_roll_deg: float
    angle_offset_deg: np.ndarray
    misalign_spec: dict[str, JsonValue]
    align_params: np.ndarray
    align_gauge: dict[str, JsonValue]
    cone_beam: dict[str, float]
    cone_segments: list[dict[str, object]]


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
    if gtype in {"cone", "cone_beam"}:
        return "cone"
    raise ValueError(
        f"Unsupported geometry_type {geometry_type!r}; expected 'parallel', 'lamino' or 'cone'"
    )


@dataclass
class AugmentedGeometry:
    """Geometry wrapper that applies saved per-view pose corrections.

    ``translation_frame`` follows `tomojax.alignment.api.apply_pose_update`: object
    translations compose after the nominal pose; detector translations add to
    its lab translation.
    """

    base: ScanGeometry
    align_params: np.ndarray
    translation_frame: str = "object"

    @property
    def grid(self) -> Grid:
        """The base geometry's reconstruction grid."""
        return self.base.grid

    @property
    def detector(self) -> Detector:
        """The base geometry's detector."""
        return self.base.detector

    @property
    def angles(self) -> Sequence[float]:
        """The base geometry's view angles."""
        return self.base.angles

    def pose_for_view(self, i: int) -> PoseMatrix:
        """Return nominal pose with saved pose correction applied."""
        T_nom = np.asarray(self.base.pose_for_view(i), dtype=np.float32)
        T_delta = _se3_from_pose_params_np(self.align_params[i])
        T = T_nom @ T_delta
        if self.translation_frame == "detector":
            T[:3, 3] = T_nom[:3, 3] + T_delta[:3, 3]
        return tuple(map(tuple, T))

    def rays_for_view(self, i: int) -> RayPair:
        """Return ray callbacks from the wrapped base geometry."""
        return self.base.rays_for_view(i)

    def __getattr__(self, name: str) -> object:
        return getattr(self.base, name)


@dataclass
class DetectorRollGeometry:
    """Geometry wrapper that preserves calibrated detector roll metadata."""

    base: ScanGeometry
    detector_roll_deg: float

    @property
    def grid(self) -> Grid:
        """The base geometry's reconstruction grid."""
        return self.base.grid

    @property
    def detector(self) -> Detector:
        """The base geometry's detector."""
        return self.base.detector

    @property
    def angles(self) -> Sequence[float]:
        """The base geometry's view angles."""
        return self.base.angles

    def pose_for_view(self, i: int) -> PoseMatrix:
        """Return the wrapped geometry pose."""
        return self.base.pose_for_view(i)

    def rays_for_view(self, i: int) -> RayPair:
        """Return ray callbacks from the wrapped base geometry."""
        return self.base.rays_for_view(i)

    def __getattr__(self, name: str) -> object:
        return getattr(self.base, name)


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


def _se3_from_pose_params_np(pose_params: np.ndarray) -> np.ndarray:
    row = np.asarray(pose_params, dtype=np.float32)
    alpha, beta, phi, dx, dz = row[:5]
    dy = row[5] if row.size > 5 else 0.0
    R = _rot_y_np(float(beta)) @ _rot_x_np(float(alpha)) @ _rot_z_np(float(phi))
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.array([dx, dy, dz], dtype=np.float32)
    return T


def _detector_from_meta(meta: LoadedGeometryMeta) -> Detector:
    return Detector.from_dict(meta["detector"])


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
        # A cone beam magnifies the axis plane: one voxel per detector pixel there.
        beam = meta.get("cone_beam")
        scale = 1.0
        if _normalize_geometry_type(meta.get("geometry_type")) == "cone" and isinstance(beam, dict):
            scale = float(beam["source_to_axis"]) / float(beam["source_to_detector"])
        return Grid(
            nx=nx,
            ny=ny,
            nz=nz,
            vx=float(detector.du) * scale,
            vy=float(detector.du) * scale,
            vz=float(detector.dv) * scale,
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


def _resolve_angles(
    meta: LoadedGeometryMeta,
    *,
    apply_saved_angle_offset: bool,
) -> np.ndarray:
    thetas = np.asarray(meta["angles"], dtype=np.float32)
    if not apply_saved_angle_offset:
        return thetas

    angle_offset = meta.get("angle_offset_deg")
    if angle_offset is None:
        return thetas

    offset = np.asarray(angle_offset, dtype=np.float32)
    if offset.shape != thetas.shape:
        return thetas
    if not np.isfinite(offset).all() or np.allclose(offset, 0.0):
        return thetas

    # TomoJAX's misalign CLI already bakes scheduled angle offsets into
    # `angles` and stores the raw schedule separately for provenance.
    if meta.get("misalign_spec") is not None:
        return thetas

    return thetas + offset


def _base_geometry(
    *,
    meta: LoadedGeometryMeta,
    grid: Grid,
    detector: Detector,
    angles: Sequence[float],
) -> ScanGeometry:
    gtype = _normalize_geometry_type(meta.get("geometry_type"))
    if gtype == "cone":
        beam_meta = meta.get("cone_beam")
        if not isinstance(beam_meta, dict):
            raise ValueError("cone geometry metadata needs a 'cone_beam' mapping")
        axis = meta.get("axis_unit_lab")
        return ConeGeometry(
            grid=grid,
            detector=detector,
            angles=angles,
            beam=ConeBeam(**{str(k): float(v) for k, v in beam_meta.items()}),
            tilt_deg=float(meta.get("tilt_deg", 0.0)),
            tilt_about=str(meta.get("tilt_about", "x")),
            axis_unit=None if axis is None else tuple(float(x) for x in axis),  # type: ignore[arg-type]
        )
    if meta.get("axis_unit_lab") is not None:
        return RotationAxisGeometry(
            grid=grid,
            detector=detector,
            angles=angles,
            axis_unit_lab=normalize_axis_unit(meta["axis_unit_lab"]),  # type: ignore[arg-type]
        )

    if gtype == "parallel":
        return ParallelGeometry(grid=grid, detector=detector, angles=angles)

    tilt_deg = float(meta.get("tilt_deg", 30.0))
    tilt_about = str(meta.get("tilt_about", "x"))
    return LaminographyGeometry(
        grid=grid,
        detector=detector,
        angles=angles,
        tilt_deg=tilt_deg,
        tilt_about=tilt_about,
    )


def _with_detector_roll_metadata(
    geom: ScanGeometry,
    meta: LoadedGeometryMeta,
) -> ScanGeometry:
    detector_roll = meta.get("detector_roll_deg")
    if detector_roll is None:
        return geom
    roll = float(detector_roll)
    if not np.isfinite(roll):
        return geom
    return DetectorRollGeometry(base=geom, detector_roll_deg=roll)


def _segments_from_meta(
    meta: LoadedGeometryMeta,
    grid: Grid,
    angles: Sequence[float],
    *,
    poses: bool,
) -> ScanGeometry:
    """A ``ConeSegments`` geometry from saved ``cone_segments`` metadata."""
    from tomojax.core.geometry.cone import ConeSegments

    entries = cast("list[dict[str, Any]]", meta.get("cone_segments", []))
    params = meta.get("align_params") if poses else None
    frame = str(meta.get("align_gauge", {}).get("pose_translation_frame", "detector"))
    segments: list[ScanGeometry] = []
    start = 0
    for entry in entries:
        views = int(entry["views"])
        piece = cast("LoadedGeometryMeta", {**entry, "geometry_type": "cone"})
        segment = _base_geometry(
            meta=piece,
            grid=grid,
            detector=Detector.from_dict(entry["detector"]),
            angles=list(angles[start : start + views]),
        )
        if params is not None:
            table = np.asarray(params, dtype=np.float32)[start : start + views, :6]
            if np.any(table):
                segment = AugmentedGeometry(segment, table, translation_frame=frame)
        segments.append(segment)
        start += views
    if start != len(angles):
        raise ValueError(f"cone_segments describe {start} views; the dataset has {len(angles)}")
    return ConeSegments(tuple(segments))


def build_geometry_from_meta(
    meta: LoadedGeometryMeta,
    *,
    grid_override: GridOverride = None,
    poses: bool = False,
    volume_shape: Sequence[int] | None = None,
) -> tuple[Grid, Detector, ScanGeometry]:
    """Build geometry from NXtomo metadata with sensible fallbacks.

    When `grid` metadata is missing, the grid is inferred from detector dimensions
    unless an explicit `grid_override` or `volume_shape` is supplied; both reuse
    detector pixel spacings as voxel spacings. When `poses` is
    True, any saved `align_params` are composed onto the nominal poses. Saved
    alignments must provide one row per view and five columns ordered as
    `[alpha, beta, phi, dx, dz]`, optionally followed by `dy`; further columns
    are ignored. Saved
    `angle_offset_deg` is applied unless it is known to have already been baked
    into `angles`.
    """
    detector = _detector_from_meta(meta)
    grid = _grid_from_meta(meta, detector, grid_override, volume_shape)
    angles = _resolve_angles(
        meta,
        apply_saved_angle_offset=poses,
    )
    if meta.get("cone_segments"):
        return (
            grid,
            detector,
            _segments_from_meta(
                meta,
                grid,
                [float(t) for t in angles],
                poses=poses,
            ),
        )
    geom = _with_detector_roll_metadata(
        _base_geometry(meta=meta, grid=grid, detector=detector, angles=angles),
        meta,
    )

    if poses and meta.get("align_params") is not None:
        align_params = np.asarray(meta["align_params"], dtype=np.float32)
        if align_params.ndim != 2:
            raise ValueError("align_params must be a 2-D array with shape (n_views, >=5)")
        if align_params.shape[0] != len(angles):
            raise ValueError(
                f"align_params row count ({align_params.shape[0]}) must match "
                f"number of views ({len(angles)})"
            )
        if align_params.shape[1] < 5:
            raise ValueError(
                "align_params must provide at least 5 columns "
                f"[alpha, beta, phi, dx, dz], got {align_params.shape[1]}"
            )
        frame = str(meta.get("align_gauge", {}).get("pose_translation_frame", "object"))
        if frame not in {"object", "detector"}:
            raise ValueError(f"unknown saved pose translation frame {frame!r}")
        geom = AugmentedGeometry(
            base=geom, align_params=align_params[:, :6], translation_frame=frame
        )

    return grid, detector, geom


def composed_poses(geometry: ScanGeometry, corrections: np.ndarray, frame: str) -> np.ndarray:
    """``geometry``'s poses followed by ``corrections`` (in ``frame``), as one detector-frame table.

    One row of ``corrections`` per view. The table moves the nominal geometry
    as its own poses and then the corrections do.
    """
    from scipy.spatial.transform import Rotation

    from tomojax.core.geometry.views import stack_view_poses

    n = len(corrections)
    start = np.asarray(stack_view_poses(_nominal(geometry), n), np.float64)
    corrected = AugmentedGeometry(geometry, np.asarray(corrections, np.float32), frame)
    moved = np.asarray(stack_view_poses(corrected, n), np.float64)
    rotation = np.einsum("nji,njk->nik", start[:, :3, :3], moved[:, :3, :3])
    beta, alpha, phi = Rotation.from_matrix(rotation).as_euler("YXZ").T
    shift = moved[:, :3, 3] - start[:, :3, 3]
    # (alpha, beta, phi, dx, dz, dy), as Scan.poses.
    table = np.stack([alpha, beta, phi, shift[:, 0], shift[:, 2], shift[:, 1]], axis=1)
    return table.astype(np.float32)


def _nominal(geometry: ScanGeometry) -> ScanGeometry:
    """``geometry`` without its per-view poses (each segment's, for segments)."""
    from tomojax.core.geometry import ConeSegments

    if isinstance(geometry, ConeSegments):
        return ConeSegments(tuple(_nominal(s) for s in geometry.segments))
    return geometry.base if isinstance(geometry, AugmentedGeometry) else geometry


def detector_poses(geometry: ScanGeometry) -> np.ndarray | None:
    """``geometry``'s per-view poses as one detector-frame table; None for none."""
    from tomojax.core.geometry import ConeSegments

    if isinstance(geometry, ConeSegments):
        tables = [detector_poses(s) for s in geometry.segments]
        if all(t is None for t in tables):
            return None
        return np.concatenate([
            np.zeros((len(s.angles), 6), np.float32) if t is None else t
            for s, t in zip(geometry.segments, tables, strict=True)
        ])  # fmt: skip
    if not isinstance(geometry, AugmentedGeometry):
        return None
    params = np.asarray(geometry.align_params, np.float32)
    if geometry.translation_frame == "detector":
        return params
    return composed_poses(geometry, np.zeros_like(params), "detector")
