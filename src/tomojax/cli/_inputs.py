"""What a file lacks, as flags: ``tj.load``'s arguments for ``tomojax correct``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, get_args

from tomojax._loading_geometry import DeclaredGeometry
from tomojax._parameters import GeometryName, TiltAbout

if TYPE_CHECKING:
    import numpy as np


class InputArgs(argparse.Namespace):
    """Namespace populated by the acquisition flags."""

    flats: str | None
    darks: str | None
    angles: str | None
    pixel_size: list[float] | None
    detector_center: list[float] | None
    geometry: GeometryName | None
    tilt: float | None
    tilt_about: TiltAbout | None
    detector_roll: float | None
    detector_pitch: float | None
    detector_yaw: float | None
    source_to_axis: float | None
    source_to_detector: float | None
    axis_offset: float | None
    data_path: str | None
    image_key_path: str | None
    angles_path: str | None
    apply_poses: bool


class LoadKeywords(TypedDict):
    flats: float | str | None
    darks: float | str | None
    angles: np.ndarray | None
    pixel_size: tuple[float, float] | None
    data_path: str | None
    image_key_path: str | None
    angles_path: str | None
    apply_poses: bool


# The flags ``tomojax correct`` lists in --help for acquisition inputs.
PUBLIC = (
    "--flats",
    "--darks",
    "--angles",
    "--pixel-size",
    "--geometry",
    "--tilt",
    "--apply-poses",
)

_CONE_ONLY = (
    "source_to_axis",
    "source_to_detector",
    "detector_pitch",
    "detector_yaw",
    "axis_offset",
)
_GEOM_DEPENDENT = (*_CONE_ONLY, "tilt", "tilt_about", "detector_roll")


def add_inputs(parser: argparse.ArgumentParser) -> None:
    """Add the flags that supply what INPUT does not record, under ``tj.load``'s names."""
    group = parser.add_argument_group(
        "what INPUT does not record (tj.load's arguments; lengths share one unit)"
    )
    _ = group.add_argument(
        "--flats", metavar="PATH|LEVEL", help="Flat frames (HDF5, TIFF or XRM), or one level"
    )
    _ = group.add_argument(
        "--darks", metavar="PATH|LEVEL", help="Dark frames (HDF5, TIFF or XRM), or one level"
    )
    _ = group.add_argument(
        "--angles",
        metavar="FILE",
        help="Angles in degrees (.npy, CSV first column, or one per line)",
    )
    _ = group.add_argument(
        "--pixel-size",
        type=float,
        nargs="+",
        metavar="SIZE",
        help="Detector pixel size: one value, or two (u then v)",
    )
    _ = group.add_argument(
        "--detector-center",
        type=float,
        nargs=2,
        metavar=("U", "V"),
        help="Detector center, in the pixel size's unit (default: the file's, or the middle)",
    )
    _ = group.add_argument(
        "--geometry",
        choices=get_args(GeometryName.__value__),
        help="Declare the acquisition geometry the file does not record",
    )
    _ = group.add_argument(
        "--tilt", type=float, help="Rotation axis tilt in degrees (lamino, cone)"
    )
    _ = group.add_argument(
        "--tilt-about",
        choices=get_args(TiltAbout.__value__),
        default=None,
        help="The axis the tilt turns about (x)",
    )
    _ = group.add_argument(
        "--detector-roll", type=float, default=None, help="Detector roll, degrees"
    )
    _ = group.add_argument(
        "--detector-pitch", type=float, default=None, help="Cone: pitch, degrees"
    )
    _ = group.add_argument("--detector-yaw", type=float, default=None, help="Cone: yaw, degrees")
    _ = group.add_argument("--source-to-axis", type=float, help="Cone beam: source to axis")
    _ = group.add_argument("--source-to-detector", type=float, help="Cone beam: source to detector")
    _ = group.add_argument(
        "--axis-offset",
        type=float,
        default=None,
        help="Cone beam: axis offset from the central ray",
    )
    _ = group.add_argument("--data-path", help="HDF5 path of the stack of frames")
    _ = group.add_argument("--image-key-path", help="HDF5 path of the image_key")
    _ = group.add_argument("--angles-path", help="HDF5 path of the rotation angles")
    _ = group.add_argument(
        "--apply-poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the poses and scanner shifts the file records (--no-apply-poses: nominal)",
    )


def validate_inputs(
    parser: argparse.ArgumentParser,
    args: InputArgs,
) -> DeclaredGeometry | None:
    """Validate input and geometry flag combinations at parse time (usage status 2)."""
    sizes = args.pixel_size
    if sizes is not None and len(sizes) not in (1, 2):
        parser.error("--pixel-size takes one value, or two (u then v)")
    kind = args.geometry
    center = args.detector_center
    if kind is None:
        dangling = [
            f"--{name.replace('_', '-')}"
            for name in _GEOM_DEPENDENT
            if getattr(args, name, None) is not None
        ]
        if dangling:
            verb = "requires" if len(dangling) == 1 else "require"
            parser.error(f"{', '.join(dangling)} {verb} --geometry")
        if center is None:
            return None
        return DeclaredGeometry(detector_center=(center[0], center[1]))

    if kind != "cone":
        cone_given = [
            f"--{name.replace('_', '-')}"
            for name in _CONE_ONLY
            if getattr(args, name, None) is not None
        ]
        if cone_given:
            verb = "requires" if len(cone_given) == 1 else "require"
            parser.error(f"{', '.join(cone_given)} {verb} --geometry cone")
    if kind == "parallel" and (args.tilt is not None or args.tilt_about is not None):
        parser.error("--tilt and --tilt-about require --geometry lamino or cone")
    if kind == "lamino" and args.tilt is None:
        parser.error("--geometry lamino needs --tilt")
    if kind == "cone" and (args.source_to_axis is None or args.source_to_detector is None):
        parser.error("--geometry cone needs --source-to-axis and --source-to-detector")
    return DeclaredGeometry(
        kind=kind,
        tilt=args.tilt,
        tilt_about="x" if args.tilt_about is None else args.tilt_about,
        detector_roll=0.0 if args.detector_roll is None else args.detector_roll,
        detector_pitch=0.0 if args.detector_pitch is None else args.detector_pitch,
        detector_yaw=0.0 if args.detector_yaw is None else args.detector_yaw,
        source_to_axis=args.source_to_axis,
        source_to_detector=args.source_to_detector,
        axis_offset=0.0 if args.axis_offset is None else args.axis_offset,
        detector_center=None if center is None else (center[0], center[1]),
    )


def load_keywords(args: InputArgs) -> LoadKeywords:
    """``tj.load``'s keywords from the validated flags."""
    from tomojax.io.api import load_angles

    angles = args.angles
    sizes = args.pixel_size
    return {
        "flats": _level_or_path(args.flats),
        "darks": _level_or_path(args.darks),
        "angles": None if angles is None else load_angles(Path(angles)),
        "pixel_size": None if sizes is None else (sizes[0], sizes[-1]),
        "data_path": args.data_path,
        "image_key_path": args.image_key_path,
        "angles_path": args.angles_path,
        "apply_poses": args.apply_poses,
    }


def input_paths(args: InputArgs) -> list[str]:
    """The files the flags name, which must exist."""
    named = (args.flats, args.darks, args.angles)
    return [p for p in named if isinstance(p, str) and _level_or_path(p) == p]


def _level_or_path(value: str | None) -> float | str | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


__all__ = ["PUBLIC", "add_inputs", "input_paths", "load_keywords", "validate_inputs"]
