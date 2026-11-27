"""Record-to-geometry resolution, before any detector pixels are read.

Explicit geometry and grids remain authoritative. When a file records a pitch
but no rotation center, the detector middle ``(0.0, 0.0)`` is used and recorded
with ``detector_center_status="not_recorded"``; fit a center explicitly with
:func:`tomojax.align`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast, get_args

import numpy as np

from tomojax._parameters import GeometryName, TiltAbout
from tomojax._scan.model import frame_output_grid, geometry_of, require_nominal_geometry
from tomojax._typed_arrays import object_mapping
from tomojax.errors import InvalidParameter
from tomojax.geometry import (
    ConeBeam,
    ConeGeometry,
    Detector,
    Grid,
    LaminographyGeometry,
    ParallelGeometry,
)
from tomojax.io.api import replace_dataset
from tomojax.progress import warn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.geometry import ScanGeometry
    from tomojax.io import ProjectionDataset
    from tomojax.io.api import ViewOffsets

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeclaredGeometry:
    """Geometry supplied when opening frames with missing or overridden acquisition metadata.

    Parameters
    ----------
    kind : str or None
        Acquisition geometry. None uses the recorded geometry.
    tilt : float or None
        Angle of the rotation axis from the axial geometry.
    tilt_about : str
        Coordinate axis about which the rotation axis is tilted.
    detector_roll, detector_pitch, detector_yaw : float
        Detector rotations about its normal, horizontal and vertical axes respectively.
    source_to_axis, source_to_detector : float or None
        Cone source distances to the rotation axis and detector.
    axis_offset : float
        Horizontal displacement of the rotation axis in the cone geometry.
    detector_center : tuple of float or None
        Detector center offset in horizontal and vertical coordinates.
    """

    kind: GeometryName | None = None
    tilt: float | None = field(default=None, metadata={"unit": "degree"})
    tilt_about: TiltAbout = "x"
    detector_roll: float = field(default=0.0, metadata={"unit": "degree"})
    detector_pitch: float = field(default=0.0, metadata={"unit": "degree"})
    detector_yaw: float = field(default=0.0, metadata={"unit": "degree"})
    source_to_axis: float | None = field(default=None, metadata={"unit": "geometry_length"})
    source_to_detector: float | None = field(default=None, metadata={"unit": "geometry_length"})
    axis_offset: float = field(default=0.0, metadata={"unit": "geometry_length"})
    detector_center: tuple[float, float] | None = field(
        default=None, metadata={"unit": "geometry_length"}
    )

    def __post_init__(self) -> None:
        if self.kind is not None and self.kind not in get_args(GeometryName.__value__):
            raise InvalidParameter("kind", f"unknown geometry {self.kind!r}")
        if self.tilt_about not in get_args(TiltAbout.__value__):
            raise InvalidParameter(
                "tilt_about", f"tilt_about must be x or z, got {self.tilt_about!r}"
            )
        if self.kind == "lamino" and self.tilt is None:
            raise InvalidParameter("tilt", "lamino geometry needs tilt")
        if self.kind == "cone":
            if self.source_to_axis is None or self.source_to_detector is None:
                field = "source_to_axis" if self.source_to_axis is None else "source_to_detector"
                raise InvalidParameter(
                    field, "cone geometry needs source_to_axis and source_to_detector"
                )
            ConeBeam(self.source_to_axis, self.source_to_detector)


def _declared_grid(
    declared: DeclaredGeometry,
    detector: Detector,
    grid: Grid | None,
) -> Grid:
    """Resolve the default grid for a CLI-declared geometry."""
    if grid is not None:
        return grid
    if declared.kind == "cone":
        assert declared.source_to_axis is not None
        assert declared.source_to_detector is not None
        mag = float(declared.source_to_detector) / float(declared.source_to_axis)
        vx, vz = detector.du / mag, detector.dv / mag
        return Grid(detector.nu, detector.nu, detector.nv, vx, vx, vz)
    return Grid(
        detector.nu,
        detector.nu,
        detector.nv,
        detector.du,
        detector.du,
        detector.dv,
        vol_center=(0.0, 0.0, detector.center[1]),
    )


def _declared_geometry(
    record: ProjectionDataset,
    declared: DeclaredGeometry,
    *,
    grid: Grid | None,
) -> tuple[ProjectionDataset, ScanGeometry]:
    """Build a declared geometry directly from frame shape, pitches and angles."""
    detector = record.detector
    if detector is None:
        if declared.kind == "cone":
            raise ValueError(
                "Cone geometry requires a detector pixel size: pass --pixel-size or detector=..."
            )
        record = record.with_parallel_detector()
        detector = record.detector
    assert detector is not None
    if declared.detector_center is not None:
        detector = replace(detector, center=declared.detector_center)
    angles = cast("Sequence[float]", np.asarray(record.angles, np.float32))
    resolved_grid = _declared_grid(declared, detector, grid or record.grid)
    if declared.kind == "parallel":
        geom: ScanGeometry = ParallelGeometry(
            resolved_grid, detector, angles, detector_roll=declared.detector_roll
        )
    elif declared.kind == "lamino":
        assert declared.tilt is not None
        geom = LaminographyGeometry(
            resolved_grid,
            detector,
            angles,
            declared.tilt,
            tilt_about=declared.tilt_about,
            detector_roll=declared.detector_roll,
        )
    elif declared.kind == "cone":
        assert declared.source_to_axis is not None
        assert declared.source_to_detector is not None
        beam = ConeBeam(
            declared.source_to_axis,
            declared.source_to_detector,
            detector_roll=declared.detector_roll,
            detector_pitch=declared.detector_pitch,
            detector_yaw=declared.detector_yaw,
            axis_offset=declared.axis_offset,
        )
        geom = ConeGeometry(
            resolved_grid,
            detector,
            angles,
            beam,
            tilt=declared.tilt or 0.0,
            tilt_about=declared.tilt_about,
        )
    else:
        raise ValueError(f"unknown geometry {declared.kind!r}")
    return replace_dataset(record, detector=detector), geom


def _updated_geometry_metadata(
    record: ProjectionDataset,
    *,
    geometry: ScanGeometry | None,
    grid: Grid | None,
    declares_geom: bool,
    detector: Detector | None,
    pixel_size: float | tuple[float, float] | None,
) -> tuple[Detector | None, dict[str, Any]]:
    """Update detector and provenance flags from explicit caller overrides."""
    meta = dict(record.geometry_metadata)
    if grid is not None:
        meta.pop("grid_follows_detector", None)
    if pixel_size is not None:  # the shape from the frames, the center from the file or the middle
        du, dv = np.broadcast_to(np.asarray(pixel_size, np.float64), (2,)).tolist()
        _, nv, nu = cast("np.ndarray", record.projections).shape  # frames: always present
        center = (0.0, 0.0) if record.detector is None else record.detector.center
        detector = Detector(nu, nv, du, dv, center)
        meta["detector_pixel_size_status"] = "provided"
        meta.pop("length_unit", None)
    explicit_center = (
        geometry is not None or declares_geom or (detector is not None and pixel_size is None)
    )
    if explicit_center:
        meta["detector_center_status"] = "provided"
    if geometry is not None or declares_geom:
        meta["axis_status"] = "provided"
        if "acquisition" in meta:
            meta["acquisition"] = {
                **meta["acquisition"],
                "geometry_status": "provided",
                "geometry_note": "Geometry supplied explicitly by caller.",
            }
    elif object_mapping(meta.get("acquisition")).get("geometry_status") == "not_recorded":
        LOG.warning(
            "%s records no rotation axis or pixel scale: declare the geometry when loading "
            "to reconstruct or align it",
            record.source_path,
        )
    return detector, meta


def frame_geometry(
    record: ProjectionDataset,
    *,
    offsets: ViewOffsets | None = None,
    angles: Sequence[float] | np.ndarray | None = None,
    detector: Detector | None = None,
    grid: Grid | None = None,
    geometry: ScanGeometry | None = None,
    apply_poses: bool = True,
    pixel_size: float | tuple[float, float] | None = None,
    declared: DeclaredGeometry | None = None,
) -> tuple[ProjectionDataset, ScanGeometry]:
    """Resolve caller metadata before constructing any geometry from a file.

    Readers may return incomplete records. An explicit geometry or CLI
    declaration must not first require constructing (and guessing) the missing
    source geometry. Recorded per-view detector offsets move from the file's
    detector to a caller's ``detector``; an explicit ``geometry`` or declared
    geometry replaces them, with a warning.
    """
    if pixel_size is not None and (detector is not None or geometry is not None):
        raise InvalidParameter(
            "pixel_size",
            "pixel_size= and detector= (or geometry=) both describe the detector: pass one",
        )
    if offsets is None and not apply_poses:
        require_nominal_geometry(record)
    declares_geom = declared is not None and declared.kind is not None
    use_offsets = apply_poses and geometry is None and not declares_geom
    if (
        offsets is not None
        and (geometry is not None or declares_geom)
        and (message := offsets.dropped_message())
    ):
        warn(message, stacklevel=4)
    grid = geometry.grid if geometry is not None else grid
    detector, meta = _updated_geometry_metadata(
        record,
        geometry=geometry,
        grid=grid,
        declares_geom=declares_geom,
        detector=detector,
        pixel_size=pixel_size,
    )
    view_angles = record.angles if angles is None else np.asarray(angles, np.float32)
    sampling_changed = (detector is not None and detector != record.detector) or (
        angles is not None and not np.array_equal(view_angles, record.angles)
    )
    record = replace_dataset(
        record,
        angles=view_angles,
        grid=grid or record.grid,
        geometry_metadata=meta,
    )
    if detector is not None:
        record = record.with_detector(detector)
    if geometry is None and not declares_geom:
        record = record.with_parallel_detector()
    if offsets is not None:
        # A pitch override retains the scanner's center in its original pixel units.
        explicit = detector if pixel_size is None else None
        record = offsets.resolve(record, detector=explicit, applied=use_offsets)
    if declares_geom and geometry is None:
        assert declared is not None
        record, resolved = _declared_geometry(record, declared, grid=grid)
        return record, frame_output_grid(resolved, record)
    if geometry is not None:
        return record, geometry
    resolved = geometry_of(record, apply_poses=apply_poses)
    if record.grid is not None and sampling_changed:
        resolved = frame_output_grid(resolved, record)
    return record, resolved


class Located(Protocol):
    """A scan or frames: a geometry and the record it came from."""

    __dataclass_fields__: ClassVar[dict[str, Any]]

    @property
    def geometry(self) -> ScanGeometry: ...

    @property
    def source(self) -> ProjectionDataset | None: ...


def center_of(
    declared: DeclaredGeometry | None,
) -> tuple[tuple[float, float] | None, DeclaredGeometry | None]:
    """A declaration's detector center, and the declaration without it.

    The center is applied after everything else has resolved, as an edit of the
    loaded geometry (:func:`centered`); a declaration of nothing else is dropped.
    """
    if declared is None or declared.detector_center is None:
        return None, declared
    rest = replace(declared, detector_center=None)
    return declared.detector_center, (None if rest.kind is None else rest)


def centered[T: Located](item: T, center: tuple[float, float] | None) -> T:
    """``item`` with its first detector centered at ``center`` and every segment moved alike."""
    if center is None:
        return item
    from tomojax._scan.model import shifted_center

    u, v = item.geometry.detector.center
    geometry = shifted_center(item.geometry, center[0] - u, center[1] - v)
    source = item.source
    if source is not None:
        meta = {**source.geometry_metadata, "detector_center_status": "provided"}
        source = replace_dataset(source, geometry_metadata=meta)
    return replace(item, geometry=frame_output_grid(geometry, source), source=source)
