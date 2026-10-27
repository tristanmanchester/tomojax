"""Cone-beam geometry: a point source and a flat detector, the sample on a turntable."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

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

    The beam travels along +y. The source sits at
    ``(0, -source_to_axis, source_height)``
    and the detector plane at ``y = source_to_detector - source_to_axis``; the
    rotation axis passes through ``(axis_offset, 0, 0)``, so the magnification
    at the axis is ``source_to_detector / source_to_axis``. ``axis_offset`` is
    the lab-CT centre-of-rotation offset: the axis's lateral distance from the
    line through the source and the unshifted detector centre, in physical
    units. ``source_height`` is the calibrated lab-frame z position of the source
    and unshifted detector centre, not an editable per-view object correction.
    ``Detector.center`` offsets the detector relative to that centre. Roll,
    pitch and yaw rotate the detector about its centre: roll about the beam
    (y), pitch about detector u (x) and yaw about detector v (z), applied in
    that order.

    Parameters
    ----------
    source_to_axis : float
        Source-to-axis distance in physical units; strictly positive.
    source_to_detector : float
        Source-to-detector distance, greater than source_to_axis.
    detector_roll_deg : float, optional
        Detector roll about the lab y axis, in degrees.
    detector_pitch_deg : float, optional
        Detector pitch about the x axis, in degrees.
    detector_yaw_deg : float, optional
        Detector yaw about the z axis, in degrees.
    axis_offset : float, optional
        Lateral x displacement of the rotation axis, in physical units.
    source_height : float, optional
        Lab-frame z position of the source and unshifted detector centre, in physical units.

    Raises
    ------
    ValueError
        If 0 < source_to_axis < source_to_detector does not hold.
    """

    source_to_axis: float
    source_to_detector: float
    detector_roll_deg: float = 0.0
    detector_pitch_deg: float = 0.0
    detector_yaw_deg: float = 0.0
    axis_offset: float = 0.0
    source_height: float = field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        if not (0.0 < float(self.source_to_axis) < float(self.source_to_detector)):
            raise ValueError("ConeBeam needs 0 < source_to_axis < source_to_detector")

    @property
    def magnification(self) -> float:
        """Magnification of the rotation axis on the detector.

        Returns
        -------
        float
            Source-to-detector divided by source-to-axis distance.
        """
        return float(self.source_to_detector) / float(self.source_to_axis)

    def source(self) -> np.ndarray:
        """Source position in the lab frame.

        Returns
        -------
        numpy.ndarray
            Lab-frame source (x, y, z) position, in physical units.
        """
        return np.array([0.0, -float(self.source_to_axis), float(self.source_height)])

    def detector_frame(self, detector: Detector) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lab position of the detector centre and its unit u and v directions.

        Parameters
        ----------
        detector : Detector
            Detector with its physical in-plane centre offset.

        Returns
        -------
        centre : numpy.ndarray
            Lab-frame detector-centre position in physical units.
        u_dir : numpy.ndarray
            Lab-frame unit direction of increasing detector columns.
        v_dir : numpy.ndarray
            Lab-frame unit direction of increasing detector rows.
        """
        roll, pitch, yaw = np.deg2rad(
            [self.detector_roll_deg, self.detector_pitch_deg, self.detector_yaw_deg]
        )
        rotation = _rotation("z", yaw) @ _rotation("x", pitch) @ _rotation("y", roll)
        centre = np.array(
            [
                float(detector.center[0]),
                float(self.source_to_detector) - float(self.source_to_axis),
                float(self.source_height) + float(detector.center[1]),
            ]
        )
        return centre, rotation @ np.array([1.0, 0.0, 0.0]), rotation @ np.array([0.0, 0.0, 1.0])

    def to_dict(self) -> dict[str, float]:
        """Return JSON-compatible metadata.

        Returns
        -------
        dict
            Source distances, axis offset and detector rotations, ready for JSON.
        """
        return {
            "source_to_axis": float(self.source_to_axis),
            "source_to_detector": float(self.source_to_detector),
            "detector_roll_deg": float(self.detector_roll_deg),
            "detector_pitch_deg": float(self.detector_pitch_deg),
            "detector_yaw_deg": float(self.detector_yaw_deg),
            "axis_offset": float(self.axis_offset),
            "source_height": float(self.source_height),
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
    the object's +z axis is the rotation axis, turned by ``angles[i]`` degrees
    at view ``i``. ``axis_unit``, a lab-frame unit
    vector, replaces the tilt for an arbitrary calibrated axis. The beam is
    described by ``beam``; projectors read it to trace rays from the source to
    each pixel.

    Parameters
    ----------
    grid : Grid
        Reconstruction voxel grid.
    detector : Detector
        Detector dimensions, pitches and physical centre.
    angles : sequence of float
        View rotation angles in degrees, in acquisition order.
    beam : ConeBeam
        Source and detector placement in the lab frame.
    tilt_deg : float, optional
        Axis tilt away from +z, in degrees; defaults to zero.
    tilt_about : {'x', 'z'}
        Tilt convention used when axis_unit is absent.
    axis_unit : tuple of float or None, optional
        Explicit lab-frame rotation axis, replacing the tilt settings.
    """

    grid: Grid
    detector: Detector
    angles: Sequence[float]
    beam: ConeBeam
    tilt_deg: float = 0.0
    tilt_about: str = "x"
    axis_unit: tuple[float, float, float] | None = None

    def pose_for_view(self, i: int) -> PoseMatrix:
        """Return the world-from-object pose for one view.

        Parameters
        ----------
        i : int
            Zero-based view index.

        Returns
        -------
        tuple of tuple of float
            Row-major 4 x 4 homogeneous world-from-object transform.

        Raises
        ------
        IndexError
            If the view index is outside the angle sequence.
        """
        return tuple(map(tuple, self.poses(angles=np.asarray([self.angles[i]]))[0]))

    def poses(self, *, angles: np.ndarray | None = None) -> np.ndarray:
        """Return every view's world-from-object pose as an ``(n, 4, 4)`` FP64 array.

        ``angles``, in degrees, replaces the geometry's own view angles.

        Parameters
        ----------
        angles : numpy.ndarray or None, optional
            Replacement view angles in degrees; None uses this geometry's angles.

        Returns
        -------
        numpy.ndarray
            World-from-object poses with shape (views, 4, 4), in float64.
        """
        radians = np.deg2rad(
            np.asarray(self.angles if angles is None else angles, dtype=np.float64)
        )
        rotation = np.zeros((radians.size, 3, 3))
        rotation[:, 0, 0] = rotation[:, 1, 1] = np.cos(radians)
        rotation[:, 1, 0] = np.sin(radians)
        rotation[:, 0, 1] = -np.sin(radians)
        rotation[:, 2, 2] = 1.0
        axis = self.rotation_axis()
        if not np.allclose(axis, [0.0, 0.0, 1.0], atol=1e-12):
            rotation = align_u_to_v(np.array([0.0, 0.0, 1.0]), axis) @ rotation
        poses = np.zeros((radians.size, 4, 4))
        poses[:, :3, :3] = rotation
        poses[:, 0, 3] = float(self.beam.axis_offset)
        poses[:, 3, 3] = 1.0
        return poses

    def rotation_axis(self) -> np.ndarray:
        """Return the lab-frame unit rotation axis.

        Returns
        -------
        numpy.ndarray
            Three-component lab-frame unit rotation axis.

        Raises
        ------
        ValueError
            If the tilt convention is unsupported for a nonzero tilt.
        """
        if self.axis_unit is not None:
            axis = np.asarray(self.axis_unit, dtype=np.float64)
            return axis / np.linalg.norm(axis)
        return laminography_tilt_matrix(self.tilt_deg, self.tilt_about) @ np.array([0.0, 0.0, 1.0])

    def geometry_metadata(self) -> dict[str, object]:
        """Return the metadata that reconstructs this geometry from a saved dataset.

        Returns
        -------
        dict
            Serializable beam and rotation-axis metadata.
        """
        meta: dict[str, object] = {"cone_beam": self.beam.to_dict()}
        if self.axis_unit is not None:
            meta["axis_unit_lab"] = [float(x) for x in self.rotation_axis()]
        elif abs(float(self.tilt_deg)) > 1e-12:
            meta["tilt_deg"] = float(self.tilt_deg)
            meta["tilt_about"] = str(self.tilt_about)
        return meta

    def rays_for_view(self, i: int) -> RayPair:
        """Return world-frame ray callbacks (source origin, unit direction) for inspection.

        Parameters
        ----------
        i : int
            Zero-based view index.

        Returns
        -------
        tuple of callable
            Origin and unit-direction callbacks taking detector pixel indices (u, v).
        """
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
    """A scan of cone-beam segments with separate source and detector arrangements.

    Segments share the reconstruction grid and the detector's pixel count; their
    views follow one another. Multi-orbit scans (a source at several heights),
    stacked scans of a tall sample and tiled fields of view are segmented. The
    iterative solvers and projectors take them. FDK retains every arrangement
    but is only an approximate inverse for noncircular trajectories.

    Parameters
    ----------
    segments : tuple of ScanGeometry
        Nonempty cone-beam geometries in acquisition order, sharing a grid and pixel sizes.
    trajectory_lengths : tuple of int, optional
        View counts of independent acquisition trajectories, in order. Empty
        defaults to one trajectory per segment. Calibration arrangements may
        change within a trajectory: a single acquisition with per-view detector
        offsets has one trajectory but many geometry segments.

    Raises
    ------
    ValueError
        If segments disagree in sampling, or trajectory counts do not partition the views.
    """

    segments: tuple[ScanGeometry, ...]
    trajectory_lengths: tuple[int, ...] = field(default=(), kw_only=True)

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
            if (detector.du, detector.dv) != (shared.du, shared.dv):
                raise ValueError("ConeSegments' detectors have the same pixel pitch")
        counts = tuple(len(segment.angles) for segment in self.segments)
        lengths = self.trajectory_lengths or counts
        if any(n <= 0 for n in lengths) or sum(lengths) != sum(counts):
            raise ValueError("trajectory_lengths must positively partition all views")
        object.__setattr__(self, "trajectory_lengths", tuple(lengths))

    @property
    def grid(self) -> Grid:
        """The shared reconstruction grid.

        Returns
        -------
        Grid
            Reconstruction grid shared by all segments.
        """
        return self.segments[0].grid

    @property
    def detector(self) -> Detector:
        """The first segment's detector (every segment has its pixel count).

        Returns
        -------
        Detector
            First segment's detector, with the shared pixel counts and pitches.
        """
        return self.segments[0].detector

    @property
    def angles(self) -> list[float]:
        """Every view's rotation angle, segment after segment.

        Returns
        -------
        list of float
            Rotation angles in degrees, concatenated in segment order.
        """
        return [float(t) for s in self.segments for t in s.angles]

    @property
    def beam(self) -> ConeBeam:
        """Not defined: each segment has its own beam (see :func:`cone_parts`).

        Raises
        ------
        ValueError
            Always: each segment has its own beam; a single-beam method cannot use this scan.
        """
        raise ValueError(
            "this scan has one source-detector arrangement per segment; reconstruct it "
            "with cgls, fista or spdhg (FDK takes one arrangement)"
        )

    def _locate(self, i: int) -> tuple[ScanGeometry, int]:
        for segment in self.segments:
            count = len(segment.angles)
            if i < count:
                return segment, i
            i -= count
        raise IndexError("view index out of range")

    def pose_for_view(self, i: int) -> PoseMatrix:
        """Return the world-from-object pose of view ``i`` in its segment.

        Parameters
        ----------
        i : int
            Zero-based view index.

        Returns
        -------
        tuple of tuple of float
            Row-major 4 x 4 homogeneous world-from-object transform.

        Raises
        ------
        IndexError
            If the view index lies beyond the last segment.
        """
        segment, j = self._locate(i)
        return segment.pose_for_view(j)

    def rays_for_view(self, i: int) -> RayPair:
        """Return the ray callbacks of view ``i`` in its segment.

        Parameters
        ----------
        i : int
            Zero-based view index.

        Returns
        -------
        tuple of callable
            Origin and unit-direction callbacks taking detector pixel indices (u, v).

        Raises
        ------
        IndexError
            If the view index lies beyond the last segment.
        """
        segment, j = self._locate(i)
        return segment.rays_for_view(j)

    def stack_poses(self, n_views: int, dtype: jnp.dtype) -> jax.Array:
        """Every segment's poses, stacked (see ``stack_view_poses``).

        Parameters
        ----------
        n_views : int
            Number of leading views to return.
        dtype : dtype
            Output pose dtype.

        Returns
        -------
        jax.Array
            Concatenated world-from-object poses, shape (views, 4, 4).
        """
        import jax.numpy as jnp

        from .views import stack_view_poses

        stacks = [stack_view_poses(s, len(s.angles), dtype=dtype) for s in self.segments]
        return jnp.concatenate(stacks)[: int(n_views)]


def beam_of(geometry: object) -> ConeBeam | None:
    """Return a geometry's cone beam, or None for parallel-beam geometries.

    Raises for :class:`ConeSegments`, which has one beam per segment.

    Parameters
    ----------
    geometry : object
        Geometry with an optional cone beam.

    Returns
    -------
    ConeBeam or None
        Beam metadata, or None for parallel-beam geometry.

    Raises
    ------
    ValueError
        If geometry has a separate beam per segment rather than one shared beam.
    """
    beam = getattr(geometry, "beam", None)
    return beam if isinstance(beam, ConeBeam) else None


def segments_of(geometry: object) -> ConeSegments | None:
    """The :class:`ConeSegments` that ``geometry`` is or wraps (with poses, say), if any."""
    while not isinstance(geometry, ConeSegments):
        inner = getattr(geometry, "base", None)
        if inner is None:
            inner = getattr(geometry, "geometry", None)
        if inner is None:
            return None
        geometry = inner
    return geometry


def is_cone_beam(geometry: object) -> bool:
    """Whether a geometry traces diverging rays.

    Parameters
    ----------
    geometry : object
        Any geometry, including posed or segmented cone-beam geometries.

    Returns
    -------
    bool
        True for a cone beam or cone segments, False for parallel rays.
    """
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
        centre = (
            scan.center[0] + own.center[0] - reference.center[0],
            scan.center[1] + own.center[1] - reference.center[1],
        )
        beam = cast("ConeBeam", beam_of(segment))
        parts.append((len(segment.angles), beam, replace(scan, center=centre)))
    return tuple(parts)


def require_parallel_beam(geometry: object, context: str) -> None:
    """Raise for cone-beam geometries in code that models parallel rays only."""
    if is_cone_beam(geometry):
        raise ValueError(
            f"{context} models parallel rays; cone-beam geometries need the iterative "
            "solvers (cgls, fista_tv, spdhg_tv) or fdk"
        )


def view_frames(geometry: ScanGeometry) -> np.ndarray:
    """Return source, detector centre and pixel steps for each cone-beam view.

    Rows are ``[source, centre, u * du, v * dv]`` (ASTRA's ``cone_vec`` order):
    pixel ``(row r, column c)`` of view i lies at ``centre + (c - nu / 2 + 1/2) u
    + (r - nv / 2 + 1/2) v``. Every segment of a :class:`ConeSegments`, in order,
    with the corrections of any pose wrapper around it.

    Parameters
    ----------
    geometry : ScanGeometry
        Cone-beam geometry, including segments and pose wrappers.

    Returns
    -------
    numpy.ndarray
        Object-frame source, detector centre and pixel vectors, shape (views, 12).

    Raises
    ------
    ValueError
        If geometry is not cone-beam.
    """
    from tomojax.core.geometry.views import host_view_poses

    parts = cone_parts(geometry)
    if parts is None:
        raise ValueError("view_frames needs cone-beam views")
    n_views = len(cast("Any", geometry).angles)
    poses = host_view_poses(geometry, n_views)
    if poses is None:
        poses = np.asarray([geometry.pose_for_view(i) for i in range(n_views)], np.float32)
    # Geometry export stays on the host, with the same rounded physical poses
    # used by the projector's coefficient preparation.
    poses = poses.astype(np.float64)
    rows, start = [], 0
    for count, beam, detector in parts:
        stop = start + (n_views if count is None else count)
        rotation, translation = poses[start:stop, :3, :3], poses[start:stop, :3, 3]
        centre, e_u, e_v = beam.detector_frame(detector)

        rows.append(
            np.concatenate(
                [
                    np.einsum("nji,nj->ni", rotation, beam.source()[None, :] - translation),
                    np.einsum("nji,nj->ni", rotation, centre[None, :] - translation),
                    np.einsum("nji,j->ni", rotation, e_u * detector.du),
                    np.einsum("nji,j->ni", rotation, e_v * detector.dv),
                ],
                axis=1,
            )
        )
        start = stop
    return np.concatenate(rows)


__all__ = [
    "ConeBeam",
    "ConeGeometry",
    "ConeSegments",
    "beam_of",
    "cone_parts",
    "is_cone_beam",
    "require_parallel_beam",
    "segments_of",
    "view_frames",
]
