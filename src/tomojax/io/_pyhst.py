"""Explicit PyHST parameter files describing axial EDF radiographs.

The parameter file is data, never executed as Python. Detector columns use
zero-based pixel centres, as in PyHST's texture coordinate ``h + 0.5``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np

from tomojax.geometry import Detector, Grid

from ._acquisition import RawAcquisition
from ._datasets import ProjectionDataset
from ._edf import EdfFrames


def _parameters(path: Path) -> dict[str, str] | None:
    with path.open("rb") as file:
        text = file.read(65536).decode("utf-8", errors="replace")
    fields = {
        key.strip().upper(): value.split("#", 1)[0].strip()
        for line in text.splitlines()
        if "=" in line
        for key, value in (line.split("=", 1),)
    }
    required = {"FILE_PREFIX", "NUM_FIRST_IMAGE", "NUM_LAST_IMAGE", "NUM_IMAGE_1", "NUM_IMAGE_2"}
    return fields if required <= fields.keys() else None


def _yes(fields: dict[str, str], key: str, *, default: bool = False) -> bool:
    return fields.get(key, "YES" if default else "NO").upper() in {"YES", "TRUE", "1"}


def _path(owner: Path, value: str) -> Path:
    """Relocated acquisitions retain their basenames, not their beamline mount."""
    path = Path(value.strip("'\""))
    target = path if path.is_absolute() else owner.parent / path
    return target if target.parent.is_dir() else owner.parent / path.name


def _series(owner: Path, fields: dict[str, str], prefix: str = "") -> tuple[EdfFrames, np.ndarray]:
    first = int(fields[prefix + "NUM_FIRST_IMAGE"])
    last = int(fields[prefix + "NUM_LAST_IMAGE"])
    indices = np.arange(first, last + 1, int(fields.get(prefix + "FILE_INTERVAL", "1")))
    stem = _path(owner, fields[prefix + ("PREFIX" if prefix else "FILE_PREFIX")])
    width = (
        0
        if _yes(fields, prefix + "NUMBER_LENGTH_VARIES")
        else int(fields[prefix + "LENGTH_OF_NUMERICAL_PART"])
    )
    postfix = fields.get(prefix + ("POSTFIX" if prefix else "FILE_POSTFIX"), ".edf")
    files = [stem.with_name(f"{stem.name}{i:0{width}d}{postfix}") for i in indices]
    return EdfFrames.open(files), indices.astype(np.float64)


def _geometry(fields: dict[str, str], count: int) -> tuple[Detector, Grid, np.ndarray]:
    nu, nv = (int(fields[f"NUM_IMAGE_{i}"]) for i in (1, 2))
    du, dv = (float(fields[f"IMAGE_PIXEL_SIZE_{i}"]) * 1e-3 for i in (1, 2))
    axis = float(fields["ROTATION_AXIS_POSITION"])
    detector = Detector(nu, nv, du, dv, center=(((nu - 1) / 2 - axis) * du, 0))
    # START/END_VOXEL are one-based and inclusive. AXIS_TO_THE_CENTER moves
    # the requested xy box to the rotation axis, independent of START_VOXEL.
    starts = np.asarray([int(fields.get(f"START_VOXEL_{i}", "1")) - 1 for i in (1, 2, 3)])
    ends = np.asarray(
        [
            int(fields.get(f"END_VOXEL_{i}", str(n)))
            for i, n in zip((1, 2, 3), (nu, nu, nv), strict=True)
        ]
    )
    shape = ends - starts
    centre = (starts + (shape - 1) / 2 - [axis, axis, (nv - 1) / 2]) * [du, du, dv]
    options: dict[str, Any] = ast.literal_eval(fields.get("OPTIONS", "{}"))
    centred = options.get(
        "axis_to_the_center", "Y" if _yes(fields, "AXIS_TO_THE_CENTER", default=True) else "N"
    )
    if str(centred).upper() == "Y":
        centre[:2] = 0
    nx, ny, nz = (int(n) for n in shape)
    cx, cy, cz = (float(v) for v in centre)
    grid = Grid(nx, ny, nz, du, du, dv, vol_center=(cx, cy, cz))
    angles = float(fields.get("ANGLE_OFFSET", "0")) + float(
        fields["ANGLE_BETWEEN_PROJECTIONS"]
    ) * np.arange(count)
    return detector, grid, angles.astype(np.float32)


def _calibration(owner: Path, fields: dict[str, str], indices: np.ndarray):
    darks = (
        EdfFrames.open([_path(owner, fields["BACKGROUND_FILE"])])
        if _yes(fields, "SUBTRACT_BACKGROUND")
        else 0
    )
    flats, positions, view_positions = None, None, None
    white = None
    if not _yes(fields, "CORRECT_FLATFIELD"):
        # TAKE_LOGARITHM with no flat division declares transmission input.
        white = 1.0
    elif _yes(fields, "FLATFIELD_CHANGING"):
        flats, positions = _series(owner, fields, "FF_")
        view_positions = indices - indices[0] + positions[0]
    else:
        flats = EdfFrames.open([_path(owner, fields["FLATFIELD_FILE"])])
    return flats, darks, positions, view_positions, white


def read_pyhst(path: Path) -> RawAcquisition | None:
    """Read explicitly selected, fixed-axis EDF input geometry, not solver settings."""
    if not path.is_file() or (fields := _parameters(path)) is None:
        return None
    if not _yes(fields, "TAKE_LOGARITHM", default=True):
        raise ValueError(f"{path}: PyHST input is already logarithmic, not raw detector frames")
    unsupported = (
        not _yes(fields, "ROTATION_VERTICAL", default=True)
        or _yes(fields, "DO_AXIS_CORRECTION")
        # MULTIFRAME is a storage-mode enum (including 2), not just YES/NO.
        or fields.get("MULTIFRAME", "0").upper() not in {"0", "NO", "FALSE"}
        or fields.get("DISTORTION_FILE", "None") != "None"
        or int(fields.get("BINNING", "1")) != 1
        or float(fields.get("SOURCE_DISTANCE", "0")) != 0
        or float(fields.get("DZPERPROJ", "0")) != 0
    )
    if unsupported:
        raise ValueError(
            f"{path}: PyHST reader requires single-image EDFs with a fixed axial parallel geometry"
        )
    counts, indices = _series(path, fields)
    detector, grid, angles = _geometry(fields, len(counts))
    if counts.shape[1:] != (detector.nv, detector.nu):
        raise ValueError(f"{path}: EDF image dimensions disagree with PyHST geometry")
    angle_file = fields.get("ANGLES_FILE", "None").strip("'\"")
    if angle_file != "None":
        angles = np.loadtxt(_path(path, angle_file), dtype=np.float32, ndmin=1)
    flats, darks, positions, view_positions, white = _calibration(path, fields, indices)
    record = ProjectionDataset(
        projections=np.broadcast_to(np.float32(0), counts.shape),
        angles=angles,
        detector=detector,
        grid=grid,
        source_path=str(path),
        source_format="esrf_pyhst",
        geometry_metadata={
            "detector_center_status": "recorded",
            "acquisition": {
                "layout": "esrf_pyhst",
                "length_unit": "mm",
                "metadata_path": str(path),
            },
            "pyhst": fields,
        },
    )
    return RawAcquisition(
        record,
        counts,
        flats=flats,
        darks=darks,
        flat_positions=positions,
        view_positions=view_positions,
        white_level=white,
    )
