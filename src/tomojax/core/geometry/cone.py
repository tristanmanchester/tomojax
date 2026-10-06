"""Cone-beam geometry: a point source and a flat detector, the sample on a turntable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .lamino import laminography_tilt_matrix
from .transforms import align_u_to_v

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .base import Detector, Grid, PoseMatrix, RayPair


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


def beam_of(geometry: object) -> ConeBeam | None:
    """Return a geometry's cone beam, or None for parallel-beam geometries."""
    beam = getattr(geometry, "beam", None)
    return beam if isinstance(beam, ConeBeam) else None


def require_parallel_beam(geometry: object, context: str) -> None:
    """Raise for cone-beam geometries in code that models parallel rays only."""
    if beam_of(geometry) is not None:
        raise ValueError(
            f"{context} models parallel rays; cone-beam geometries need the iterative "
            "solvers (cgls, fista_tv, spdhg_tv) or fdk"
        )


__all__ = ["ConeBeam", "ConeGeometry", "beam_of", "require_parallel_beam"]
