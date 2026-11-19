"""What a file lacks, as flags: ``tj.load``'s arguments for ``tomojax load`` and ``correct``."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from tomojax.geometry import ScanGeometry

# The flags ``tomojax load`` and ``correct`` list in --help.
PUBLIC = (
    "--flats",
    "--darks",
    "--angles",
    "--pixel-size",
    "--geometry",
    "--tilt",
    "--apply-poses",
)


def add_inputs(parser: argparse.ArgumentParser) -> None:
    """Add the flags that supply what INPUT does not record, under ``tj.load``'s names."""
    group = parser.add_argument_group(
        "what INPUT does not record (tj.load's arguments; lengths share one unit)"
    )
    group.add_argument(
        "--flats", metavar="PATH|LEVEL", help="Flat frames (HDF5, TIFF or XRM), or one level"
    )
    group.add_argument(
        "--darks", metavar="PATH|LEVEL", help="Dark frames (HDF5, TIFF or XRM), or one level"
    )
    group.add_argument("--angles", metavar="FILE", help="Angles in degrees (.npy, or one per line)")
    group.add_argument(
        "--pixel-size",
        type=float,
        nargs="+",
        metavar=("SIZE", "SIZE_V"),
        help="Detector pixel size (one value, or u then v)",
    )
    group.add_argument(
        "--detector-center",
        type=float,
        nargs=2,
        metavar=("U", "V"),
        help="Detector centre, in the pixel size's unit (default: the file's, or the middle)",
    )
    group.add_argument(
        "--geometry",
        choices=("parallel", "lamino", "cone"),
        help="Declare the acquisition geometry the file does not record",
    )
    group.add_argument("--tilt", type=float, help="Rotation axis tilt in degrees (lamino, cone)")
    group.add_argument(
        "--tilt-about", choices=("x", "z"), default="x", help="The axis the tilt turns about"
    )
    group.add_argument("--detector-roll", type=float, default=0.0, help="Detector roll, degrees")
    group.add_argument("--detector-pitch", type=float, default=0.0, help="Cone: pitch, degrees")
    group.add_argument("--detector-yaw", type=float, default=0.0, help="Cone: yaw, degrees")
    group.add_argument("--source-to-axis", type=float, help="Cone beam: source to axis")
    group.add_argument("--source-to-detector", type=float, help="Cone beam: source to detector")
    group.add_argument(
        "--axis-offset", type=float, default=0.0, help="Cone beam: axis offset from the central ray"
    )
    group.add_argument("--data-path", help="HDF5 path of the stack of frames")
    group.add_argument("--image-key-path", help="HDF5 path of the image_key")
    group.add_argument("--angles-path", help="HDF5 path of the rotation angles")
    group.add_argument(
        "--apply-poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the poses and scanner shifts the file records (--no-apply-poses: nominal)",
    )


def load_keywords(args: argparse.Namespace) -> dict[str, Any]:
    """``tj.load``'s keywords from the flags (the geometry is :func:`declared`)."""
    from tomojax.io.api import load_angles

    angles = cast("str | None", args.angles)
    sizes = cast("list[float] | None", args.pixel_size)
    if sizes is not None and len(sizes) > 2:
        raise ValueError("--pixel-size takes one value, or two (u then v)")
    return {
        "flats": _level_or_path(cast("str | None", args.flats)),
        "darks": _level_or_path(cast("str | None", args.darks)),
        "angles": None if angles is None else load_angles(Path(angles)),
        "pixel_size": None if sizes is None else (sizes[0], sizes[-1]),
        "data_path": cast("str | None", args.data_path),
        "image_key_path": cast("str | None", args.image_key_path),
        "angles_path": cast("str | None", args.angles_path),
        "apply_poses": cast("bool", args.apply_poses),
    }


def input_paths(args: argparse.Namespace) -> list[str]:
    """The files the flags name, which must exist."""
    named = (args.flats, args.darks, args.angles)
    return [p for p in named if isinstance(p, str) and _level_or_path(p) == p]


def declared(geometry: ScanGeometry, args: argparse.Namespace) -> ScanGeometry:
    """``geometry``, or the one ``--geometry`` declares on its grid, detector and angles."""
    from tomojax.geometry import (
        ConeBeam,
        ConeGeometry,
        LaminographyGeometry,
        ParallelGeometry,
    )

    kind = cast("str | None", args.geometry)
    center = cast("list[float] | None", args.detector_center)
    grid, detector = geometry.grid, geometry.detector
    if center is not None:
        detector = replace(detector, center=(center[0], center[1]))
    if kind is None:
        return geometry if center is None else replace(cast("Any", geometry), detector=detector)
    angles = cast("Any", geometry).angles
    roll, tilt = cast("float", args.detector_roll), cast("float | None", args.tilt)
    about = cast("str", args.tilt_about)
    if kind == "parallel":
        return ParallelGeometry(grid, detector, angles, detector_roll=roll)
    if kind == "lamino":
        if tilt is None:
            raise ValueError("--geometry lamino needs --tilt")
        return LaminographyGeometry(
            grid, detector, angles, tilt, tilt_about=about, detector_roll=roll
        )
    sod, sdd = (
        cast("float | None", args.source_to_axis),
        cast("float | None", args.source_to_detector),
    )
    if sod is None or sdd is None:
        raise ValueError("--geometry cone needs --source-to-axis and --source-to-detector")
    beam = ConeBeam(
        sod,
        sdd,
        detector_roll=roll,
        detector_pitch=cast("float", args.detector_pitch),
        detector_yaw=cast("float", args.detector_yaw),
        axis_offset=cast("float", args.axis_offset),
    )
    return ConeGeometry(grid, detector, angles, beam, tilt=tilt or 0.0, tilt_about=about)


def _level_or_path(value: str | None) -> float | str | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


__all__ = ["PUBLIC", "add_inputs", "declared", "input_paths", "load_keywords"]
