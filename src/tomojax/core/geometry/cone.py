"""Cone-beam geometry: a point source and a flat detector, the sample on a turntable."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import numpy as np

from .lamino import laminography_tilt_matrix
from .transforms import align_u_to_v

if TYPE_CHECKING:
    from collections.abc import Sequence

    import jax
    import jax.numpy as jnp

    from .base import Detector, Grid, PoseMatrix, RayPair, ScanGeometry


@dataclass(frozen=True)
class ConeBeam:
    """Source and detector placement in the lab frame, shared by every view.

    The beam travels along +y. The source sits at ``(0, -source_to_axis, 0)``
    and the detector plane at ``y = source_to_detector - source_to_axis``; the
    rotation axis passes through ``(axis_offset, 0, 0)``, so the magnification
    at the axis is ``source_to_detector / source_to_axis``. ``axis_offset`` is
    the lab-CT centre-of-rotation offset: the axis's lateral distance from the
    line through the source and the unshifted detector centre, in physical
    units. ``Detector.det_center`` offsets the detector within its plane. Roll,
    pitch and yaw rotate the detector about its centre: roll about the beam
    (y), pitch about detector u (x) and yaw about detector v (z), applied in
    that order.
    """

    source_to_axis: float
    source_to_detector: float
    detector_roll_deg: float = 0.0
    detector_pitch_deg: float = 0.0
    detector_yaw_deg: float = 0.0
    axis_offset: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < float(self.source_to_axis) < float(self.source_to_detector)):
            raise ValueError("ConeBeam needs 0 < source_to_axis < source_to_detector")

    @property
    def magnification(self) -> float:
        """Magnification of the rotation axis on the detector."""
        return float(self.source_to_detector) / float(self.source_to_axis)

    def source(self) -> np.ndarray:
        """Source position in the lab frame."""
        return np.array([0.0, -float(self.source_to_axis), 0.0])

    def detector_frame(self, detector: Detector) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lab position of the detector centre and its unit u and v directions."""
        roll, pitch, yaw = np.deg2rad(
            [self.detector_roll_deg, self.detector_pitch_deg, self.detector_yaw_deg]
        )
        rotation = _rotation("z", yaw) @ _rotation("x", pitch) @ _rotation("y", roll)
        centre = np.array(
            [
                float(detector.det_center[0]),
                float(self.source_to_detector) - float(self.source_to_axis),
                float(detector.det_center[1]),
            ]
        )
        return centre, rotation @ np.array([1.0, 0.0, 0.0]), rotation @ np.array([0.0, 0.0, 1.0])

    def to_dict(self) -> dict[str, float]:
        """Return JSON-compatible metadata."""
        return {
            "source_to_axis": float(self.source_to_axis),
            "source_to_detector": float(self.source_to_detector),
            "detector_roll_deg": float(self.detector_roll_deg),
            "detector_pitch_deg": float(self.detector_pitch_deg),
            "detector_yaw_deg": float(self.detector_yaw_deg),
            "axis_offset": float(self.axis_offset),
        }


def _rotation(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class ConeGeometry:
    """Cone-beam scan of a sample rotating about +z, or a tilted axis.

    Poses follow ``ParallelGeometry`` (``tilt_deg == 0``) or
    ``LaminographyGeometry``: ``pose_for_view`` returns world_from_object, and
    the object's +z axis is the rotation axis. ``axis_unit``, a lab-frame unit
    vector, replaces the tilt for an arbitrary calibrated axis. The beam is
    described by ``beam``; projectors read it to trace rays from the source to
    each pixel.
    """

    grid: Grid
    detector: Detector
    thetas_deg: Sequence[float]
    beam: ConeBeam
    tilt_deg: float = 0.0
    tilt_about: str = "x"
    axis_unit: tuple[float, float, float] | None = None

    def pose_for_view(self, i: int) -> PoseMatrix:
        """Return the world-from-object pose for one view."""
        return tuple(map(tuple, self.poses(np.asarray([self.thetas_deg[i]]))[0]))

    def poses(self, thetas_deg: np.ndarray | None = None) -> np.ndarray:
        """Return every view's world-from-object pose as an ``(n, 4, 4)`` FP64 array."""
        angles = np.deg2rad(
            np.asarray(self.thetas_deg if thetas_deg is None else thetas_deg, dtype=np.float64)
        )
        rotation = np.zeros((angles.size, 3, 3))
        rotation[:, 0, 0] = rotation[:, 1, 1] = np.cos(angles)
        rotation[:, 1, 0] = np.sin(angles)
        rotation[:, 0, 1] = -np.sin(angles)
        rotation[:, 2, 2] = 1.0
        axis = self.rotation_axis()
        if not np.allclose(axis, [0.0, 0.0, 1.0], atol=1e-12):
            rotation = align_u_to_v(np.array([0.0, 0.0, 1.0]), axis) @ rotation
        poses = np.zeros((angles.size, 4, 4))
        poses[:, :3, :3] = rotation
        poses[:, 0, 3] = float(self.beam.axis_offset)
        poses[:, 3, 3] = 1.0
        return poses

    def rotation_axis(self) -> np.ndarray:
        """Return the lab-frame unit rotation axis."""
        if self.axis_unit is not None:
            axis = np.asarray(self.axis_unit, dtype=np.float64)
            return axis / np.linalg.norm(axis)
        return laminography_tilt_matrix(self.tilt_deg, self.tilt_about) @ np.array([0.0, 0.0, 1.0])

    def geometry_metadata(self) -> dict[str, object]:
        """Return the metadata that reconstructs this geometry from a saved dataset."""
        meta: dict[str, object] = {"cone_beam": self.beam.to_dict()}
        if self.axis_unit is not None:
            meta["axis_unit_lab"] = [float(x) for x in self.rotation_axis()]
        elif abs(float(self.tilt_deg)) > 1e-12:
            meta["tilt_deg"] = float(self.tilt_deg)
            meta["tilt_about"] = str(self.tilt_about)
        return meta

    def rays_for_view(self, i: int) -> RayPair:
        """Return world-frame ray callbacks (source origin, unit direction) for inspection."""
        del i
        source = self.beam.source()
        centre, u_dir, v_dir = self.beam.detector_frame(self.detector)
        nu, nv = int(self.detector.nu), int(self.detector.nv)
        du, dv = float(self.detector.du), float(self.detector.dv)

        def origin_fn(_u: int, _v: int) -> tuple[float, float, float]:
            return float(source[0]), float(source[1]), float(source[2])

        def dir_fn(u: int, v: int) -> tuple[float, float, float]:
            pixel = centre + (u - (nu - 1) / 2) * du * u_dir + (v - (nv - 1) / 2) * dv * v_dir
            ray = pixel - source
            ray = ray / np.linalg.norm(ray)
            return float(ray[0]), float(ray[1]), float(ray[2])

        return origin_fn, dir_fn


@dataclass(frozen=True)
class ConeSegments:
    """A scan made of cone-beam segments, each with its own source and detector arrangement.

    Segments share the reconstruction grid and the detector's pixel count; their
    views follow one another. Multi-orbit scans (a source at several heights),
    stacked scans of a tall sample and tiled fields of view are segmented. The
    iterative solvers and projectors take them; FDK, which assumes one circular
    orbit, does not.
    """

    segments: tuple[ScanGeometry, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("ConeSegments needs at least one segment")
        first = self.segments[0]
        for segment in self.segments:
            if beam_of(segment) is None:
                raise ValueError("ConeSegments holds cone-beam geometries")
            if segment.grid != first.grid:
                raise ValueError("ConeSegments' segments share one reconstruction grid")
            detector, shared = segment.detector, first.detector
            if (detector.nu, detector.nv) != (shared.nu, shared.nv):
                raise ValueError("ConeSegments' detectors have the same pixel count")

    @property
    def grid(self) -> Grid:
        """The shared reconstruction grid."""
        return self.segments[0].grid

    @property
    def detector(self) -> Detector:
        """The first segment's detector (every segment has its pixel count)."""
        return self.segments[0].detector

    @property
    def thetas_deg(self) -> list[float]:
        """Every view's rotation angle, segment after segment."""
        return [float(t) for s in self.segments for t in s.thetas_deg]

    @property
    def beam(self) -> ConeBeam:
        """Not defined: each segment has its own beam (see :func:`cone_parts`)."""
        raise ValueError(
            "this scan has one source-detector arrangement per segment; reconstruct it "
            "with cgls, fista or spdhg (FDK and alignment take one arrangement)"
        )

    def _locate(self, i: int) -> tuple[ScanGeometry, int]:
        for segment in self.segments:
            count = len(segment.thetas_deg)
            if i < count:
                return segment, i
            i -= count
        raise IndexError("view index out of range")

    def pose_for_view(self, i: int) -> PoseMatrix:
        """Return the world-from-object pose of view ``i`` in its segment."""
        segment, j = self._locate(i)
        return segment.pose_for_view(j)

    def rays_for_view(self, i: int) -> RayPair:
        """Return the ray callbacks of view ``i`` in its segment."""
        segment, j = self._locate(i)
        return segment.rays_for_view(j)

    def stack_poses(self, n_views: int, dtype: jnp.dtype) -> jax.Array:
        """Every segment's poses, stacked (see ``stack_view_poses``)."""
        import jax.numpy as jnp

        from .views import stack_view_poses

        stacks = [stack_view_poses(s, len(s.thetas_deg), dtype=dtype) for s in self.segments]
        return jnp.concatenate(stacks)[: int(n_views)]


def beam_of(geometry: object) -> ConeBeam | None:
    """Return a geometry's cone beam, or None for parallel-beam geometries.

    Raises for :class:`ConeSegments`, which has one beam per segment.
    """
    beam = getattr(geometry, "beam", None)
    return beam if isinstance(beam, ConeBeam) else None


def segments_of(geometry: object) -> ConeSegments | None:
    """The :class:`ConeSegments` that ``geometry`` is or wraps (with poses, say), if any."""
    for _ in range(16):
        if isinstance(geometry, ConeSegments):
            return geometry
        inner = getattr(geometry, "base", None) or getattr(geometry, "geometry", None)
        if inner is None:
            return None
        geometry = inner
    return None


def is_cone_beam(geometry: object) -> bool:
    """Whether ``geometry`` traces diverging rays (a cone beam, or cone segments)."""
    return segments_of(geometry) is not None or beam_of(geometry) is not None


def cone_parts(
    geometry: object, detector: Detector | None = None
) -> tuple[tuple[int | None, ConeBeam, Detector], ...] | None:
    """``(views, beam, detector)`` for each run of views sharing an arrangement.

    None for parallel beams; ``views`` is None for a single arrangement, which
    serves every view. ``detector`` is the scan's detector, binned perhaps
    (default the geometry's): a segment's detector is it moved by the
    segment's centre offset from the scan's.
    """
    segments = segments_of(geometry)
    if segments is None:
        beam = beam_of(geometry)
        if beam is None:
            return None
        if detector is None:
            detector = cast("ScanGeometry", geometry).detector
        return ((None, beam, detector),)
    scan = segments.detector if detector is None else detector
    reference = segments.detector
    parts = []
    for segment in segments.segments:
        own = segment.detector
        if (own.du, own.dv) != (reference.du, reference.dv):
            raise ValueError("ConeSegments' detectors have the same pixel pitch")
        centre = (
            scan.det_center[0] + own.det_center[0] - reference.det_center[0],
            scan.det_center[1] + own.det_center[1] - reference.det_center[1],
        )
        beam = cast("ConeBeam", beam_of(segment))
        parts.append((len(segment.thetas_deg), beam, replace(scan, det_center=centre)))
    return tuple(parts)


def require_parallel_beam(geometry: object, context: str) -> None:
    """Raise for cone-beam geometries in code that models parallel rays only."""
    if beam_of(geometry) is not None:
        raise ValueError(
            f"{context} models parallel rays; cone-beam geometries need the iterative "
            "solvers (cgls, fista_tv, spdhg_tv) or fdk"
        )


__all__ = [
    "ConeBeam",
    "ConeGeometry",
    "ConeSegments",
    "beam_of",
    "cone_parts",
    "is_cone_beam",
    "require_parallel_beam",
    "segments_of",
]
