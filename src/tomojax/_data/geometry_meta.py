"""Helpers for materializing geometry objects from persisted NXtomo metadata."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
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
from tomojax.core.geometry.transforms import rot_x, rot_y, rot_z

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.core.geometry.base import (
        Angles,
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
    tilt: float
    tilt_about: str
    axis_unit_lab: Sequence[float]
    detector_roll: float
    angle_offset: np.ndarray
    misalign_spec: dict[str, JsonValue]
    align_params: np.ndarray
    align_gauge: dict[str, JsonValue]
    cone_beam: dict[str, float]
    cone_segments: list[dict[str, object]]
    trajectory_lengths: Sequence[int]


GridOverride = Grid | tuple[int, int, int] | list[int] | None


def _volume_shape_nxyz(volume_shape: Sequence[int] | None) -> tuple[int, int, int] | None:
    if volume_shape is None:
        return None
    dims = tuple(int(v) for v in volume_shape)
    if len(dims) != 3:
        raise ValueError(f"volume_shape must provide exactly 3 dims, got {dims!r}")
    return dims[2], dims[1], dims[0]


def normalize_geometry_type(geometry_type: str | None) -> str:
    """Use the same geometry labels when reading, writing, and materializing metadata."""
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
class _WrappedGeometry:
    """A geometry that answers for ``base`` except where a subclass changes the poses."""

    base: ScanGeometry

    @property
    def grid(self) -> Grid:
        """The base geometry's reconstruction grid."""
        return self.base.grid

    @property
    def detector(self) -> Detector:
        """The base geometry's detector."""
        return self.base.detector

    @property
    def angles(self) -> Angles:
        """The base geometry's view angles."""
        return self.base.angles

    def pose_matrix(self, i: int) -> PoseMatrix:
        """Return the wrapped geometry pose."""
        return self.base.pose_matrix(i)

    def rays_for_view(self, i: int) -> RayPair:
        """Return ray callbacks from the wrapped base geometry."""
        return self.base.rays_for_view(i)

    def __getattr__(self, name: str) -> object:
        # Only once built: unpickling looks attributes up before ``base`` is set.
        if name == "base" or "base" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.base, name)


@dataclass
class AugmentedGeometry(_WrappedGeometry):
    """Geometry wrapper that applies saved per-view pose corrections.

    Object-frame translations compose after the nominal pose; detector-frame
    translations add to its lab translation. Rotations compose on the right.
    """

    align_params: np.ndarray
    translation_frame: str = "object"

    def pose_matrix(self, i: int) -> PoseMatrix:
        """Return nominal pose with saved pose correction applied."""
        T_nom = np.asarray(self.base.pose_matrix(i), dtype=np.float32)
        T_delta = _se3_from_pose_params_np(self.align_params[i])
        T = T_nom @ T_delta
        if self.translation_frame == "detector":
            T[:3, 3] = T_nom[:3, 3] + T_delta[:3, 3]
        return tuple(map(tuple, T))


def _se3_from_pose_params_np(pose_params: np.ndarray) -> np.ndarray:
    row = np.asarray(pose_params, dtype=np.float32)
    alpha, beta, phi, dx, dz, dy = row
    R = rot_y(float(beta)) @ rot_x(float(alpha)) @ rot_z(float(phi))
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
        stated = meta.get("voxel_size")
        if stated is not None:  # the file states the voxel; the shape follows the detector
            vx, vy, vz = (float(v) for v in cast("Sequence[float]", stated))
            return Grid(nx=nx, ny=ny, nz=nz, vx=vx, vy=vy, vz=vz)
        # A cone beam magnifies the axis plane: one voxel per detector pixel there.
        # Segmented scans sample at their most magnified arrangement.
        segments = cast("list[dict[str, Any]]", meta.get("cone_segments") or [])
        beams = [e.get("cone_beam") for e in segments] or [meta.get("cone_beam")]
        beams = [b for b in beams if isinstance(b, dict)]
        scale = 1.0
        if normalize_geometry_type(meta.get("geometry_type")) == "cone" and beams:
            scale = min(float(b["source_to_axis"]) / float(b["source_to_detector"]) for b in beams)
        return Grid(
            nx=nx,
            ny=ny,
            nz=nz,
            vx=float(detector.du) * scale,
            vy=float(detector.du) * scale,
            vz=float(detector.dv) * scale,
        )

    if grid_override is None:
        return Grid.from_dict(grid_d)
    nx, ny, nz = map(int, grid_override)
    return Grid.from_dict({**grid_d, "nx": nx, "ny": ny, "nz": nz})


def _resolve_angles(
    meta: LoadedGeometryMeta,
    *,
    apply_saved_angle_offset: bool,
) -> np.ndarray:
    thetas = np.asarray(meta["angles"])
    if not apply_saved_angle_offset:
        return thetas

    angle_offset = meta.get("angle_offset")
    # Simulated misalignment already bakes these offsets into `angles`; the
    # separate schedule is provenance, not a requested second correction.
    if angle_offset is None or meta.get("misalign_spec") is not None:
        return thetas

    offset = np.asarray(angle_offset)
    if offset.shape != thetas.shape:
        raise ValueError(f"angle_offset must have shape {thetas.shape}, got {offset.shape}")
    if not np.isfinite(offset).all():
        raise ValueError("angle_offset must contain only finite values")

    return thetas + offset


def _base_geometry(
    *,
    meta: LoadedGeometryMeta,
    grid: Grid,
    detector: Detector,
    angles: Sequence[float] | np.ndarray,
) -> ScanGeometry:
    gtype = normalize_geometry_type(meta.get("geometry_type"))
    roll = float(meta.get("detector_roll", 0.0))
    if gtype == "cone":
        if roll:
            # Only a private wrapper could write this; a cone roll lives in its beam.
            raise ValueError(
                "A cone scan's detector roll belongs in cone_beam; this file has a "
                "top-level detector_roll: pass geometry=... with "
                "ConeBeam(detector_roll=...)"
            )
        beam_meta = meta.get("cone_beam")
        if not isinstance(beam_meta, dict):
            raise ValueError("Missing cone-beam distances: pass geometry=...")
        axis = meta.get("axis_unit_lab")
        return ConeGeometry(
            grid=grid,
            detector=detector,
            angles=angles,
            beam=ConeBeam.from_dict(beam_meta),
            tilt=float(meta.get("tilt", 0.0)),
            tilt_about=str(meta.get("tilt_about", "x")),
            axis_unit=None if axis is None else tuple(float(x) for x in axis),  # type: ignore[arg-type]
        )
    if meta.get("axis_unit_lab") is not None:
        axis = normalize_axis_unit(meta["axis_unit_lab"])  # type: ignore[arg-type]
        # Setup calibration records even an unchanged axial direction. Keep its
        # canonical model: this has no laminographic missing cone and supports
        # the same axial inverses as a scan without redundant axis metadata.
        if axis == (0.0, 0.0, 1.0):
            return ParallelGeometry(grid, detector, angles, detector_roll=roll)
        return RotationAxisGeometry(grid, detector, angles, axis, detector_roll=roll)

    if gtype == "parallel":
        return ParallelGeometry(grid, detector, angles, detector_roll=roll)

    if "tilt" not in meta:
        raise ValueError(
            "Missing laminography tilt: pass geometry=... with an explicit tilt or axis_unit_lab"
        )
    tilt = float(meta["tilt"])
    # An explicit zero tilt also has the canonical axial poses.
    if tilt == 0.0:
        return ParallelGeometry(grid, detector, angles, detector_roll=roll)
    tilt_about = str(meta.get("tilt_about", "x"))
    return LaminographyGeometry(grid, detector, angles, tilt, tilt_about, detector_roll=roll)


def _segments_from_meta(
    meta: LoadedGeometryMeta,
    grid: Grid,
    angles: Sequence[float],
    *,
    apply_poses: bool,
) -> ScanGeometry:
    """A ``ConeSegments`` geometry from saved ``cone_segments`` metadata."""
    from tomojax.core.geometry.cone import ConeSegments

    entries = cast("list[dict[str, Any]]", meta.get("cone_segments", []))
    saved = _saved_poses(meta, len(angles), default_frame="detector") if apply_poses else None
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
        if saved is not None:
            params, frame = saved
            table = params[start : start + views]
            if np.any(table):
                segment = AugmentedGeometry(segment, table, translation_frame=frame)
        segments.append(segment)
        start += views
    if start != len(angles):
        raise ValueError(f"cone_segments describe {start} views; the dataset has {len(angles)}")
    return ConeSegments(
        tuple(segments), trajectory_lengths=tuple(meta.get("trajectory_lengths", ()))
    )


def _saved_poses(
    meta: LoadedGeometryMeta, views: int, *, default_frame: str
) -> tuple[np.ndarray, str] | None:
    """Validated saved six-column poses and their translation frame."""
    saved = meta.get("align_params")
    if saved is None:
        return None
    params = np.asarray(saved, dtype=np.float32)
    if params.ndim != 2:
        raise ValueError("align_params must be a 2-D array with shape (n_views, 6)")
    if params.shape[0] != views:
        raise ValueError(
            f"align_params row count ({params.shape[0]}) must match number of views ({views})"
        )
    if params.shape[1] != 6:
        raise ValueError(
            "align_params must provide exactly 6 columns "
            f"[alpha, beta, phi, x, z, y], got {params.shape[1]}"
        )
    frame = str(meta.get("align_gauge", {}).get("pose_translation_frame", default_frame))
    if frame not in {"object", "detector"}:
        raise ValueError(f"unknown saved pose translation frame {frame!r}")
    return params, frame


def build_geometry_from_meta(
    meta: LoadedGeometryMeta,
    *,
    grid_override: GridOverride = None,
    apply_poses: bool = False,
    volume_shape: Sequence[int] | None = None,
) -> tuple[Grid, Detector, ScanGeometry]:
    """Build geometry from NXtomo metadata with sensible fallbacks.

    When `grid` metadata is missing, the grid is inferred from detector dimensions
    unless an explicit `grid_override` or `volume_shape` is supplied; both reuse
    detector pixel spacings as voxel spacings. When `apply_poses` is
    True, any saved `align_params` are composed onto the nominal poses. Saved
    alignments must provide one row per view and six columns ordered as
    `[alpha, beta, phi, x, z, y]`. Saved
    `angle_offset` is applied unless it is known to have already been baked
    into `angles`.
    """
    detector = _detector_from_meta(meta)
    grid = _grid_from_meta(meta, detector, grid_override, volume_shape)
    angles = _resolve_angles(
        meta,
        apply_saved_angle_offset=apply_poses,
    )
    if meta.get("cone_segments"):
        return (
            grid,
            detector,
            _segments_from_meta(
                meta,
                grid,
                [float(t) for t in angles],
                apply_poses=apply_poses,
            ),
        )
    geom = _base_geometry(meta=meta, grid=grid, detector=detector, angles=angles)

    saved = _saved_poses(meta, len(angles), default_frame="object") if apply_poses else None
    if saved is not None:
        align_params, frame = saved
        geom = AugmentedGeometry(base=geom, align_params=align_params, translation_frame=frame)

    return grid, detector, geom


def composed_poses(geometry: ScanGeometry, corrections: np.ndarray, frame: str) -> np.ndarray:
    """``geometry``'s poses followed by ``corrections`` (in ``frame``), as one detector-frame table.

    One row of ``corrections`` per view. The table moves the nominal geometry
    as its own poses and then the corrections do.
    """
    from scipy.spatial.transform import Rotation

    from tomojax.core.geometry.views import stack_view_poses

    n = len(corrections)
    start = np.asarray(stack_view_poses(nominal_geometry(geometry), n), np.float64)
    corrected = AugmentedGeometry(geometry, np.asarray(corrections, np.float32), frame)
    moved = np.asarray(stack_view_poses(corrected, n), np.float64)
    rotation = np.einsum("nji,njk->nik", start[:, :3, :3], moved[:, :3, :3])
    beta, alpha, phi = Rotation.from_matrix(rotation).as_euler("YXZ").T
    shift = moved[:, :3, 3] - start[:, :3, 3]
    # (alpha, beta, phi, x, z, y), as Scan.poses.
    table = np.stack([alpha, beta, phi, shift[:, 0], shift[:, 2], shift[:, 1]], axis=1)
    return table.astype(np.float32)


def nominal_geometry(geometry: ScanGeometry) -> ScanGeometry:
    """``geometry`` without its per-view poses (each segment's, for segments)."""
    from tomojax.core.geometry import ConeSegments

    if isinstance(geometry, ConeSegments):
        segments = tuple(nominal_geometry(s) for s in geometry.segments)
        if all(a is b for a, b in zip(segments, geometry.segments, strict=True)):
            return geometry
        return replace(geometry, segments=segments)
    if isinstance(geometry, AugmentedGeometry):
        return nominal_geometry(geometry.base)
    return geometry


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
    if (
        geometry.translation_frame == "detector"
        and nominal_geometry(geometry.base) is geometry.base
    ):
        return params
    return composed_poses(geometry, np.zeros_like(params), "detector")


def views_of(geometry: ScanGeometry, kept: np.ndarray) -> ScanGeometry:
    """``geometry`` of the views ``kept`` (increasing indices), wrappers and segments kept."""
    from tomojax.geometry import ConeSegments

    if isinstance(geometry, ConeSegments):
        parts, start = [], 0
        for segment in geometry.segments:
            n = len(segment.angles)
            mine = kept[(kept >= start) & (kept < start + n)] - start
            if len(mine):
                parts.append(views_of(segment, mine))
            start += n
        stops = np.searchsorted(kept, np.cumsum(geometry.trajectory_lengths))
        counts = np.diff(np.r_[0, stops])
        trajectories = tuple(int(n) for n in counts if n)
        if len(parts) == 1 and len(trajectories) == 1:
            return parts[0]
        return ConeSegments(tuple(parts), trajectory_lengths=trajectories)
    if isinstance(geometry, AugmentedGeometry):
        return replace(
            geometry,
            base=views_of(geometry.base, kept),
            align_params=np.asarray(geometry.align_params)[kept],
        )
    if not is_dataclass(geometry) or isinstance(geometry, type):
        raise TypeError(f"cannot select views of a {type(geometry).__name__} geometry")
    fields: dict[str, Any] = vars(geometry)  # a wrapper's base, or a geometry's own angles
    if "base" in fields:
        return replace(geometry, base=views_of(fields["base"], kept))
    return replace(geometry, angles=np.asarray(fields["angles"])[kept])
